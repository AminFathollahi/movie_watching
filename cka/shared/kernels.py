"""Searchlight centered kernel alignment (CKA): window indexing, kernels and likelihoods."""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np
import torch

from cifti_io import get_bm_axis, get_cortex_vertex_indices

log = logging.getLogger(__name__)
NOISE_FLOOR = 1e-3


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def window_index(timing, run_trs, bin_sec, skip_sec, delay_sec, tr=1.0):
    """Clip id and run index of every retained window, in the order of `preprocess_fmri`."""
    bin_trs, skip_trs = max(1, int(np.round(bin_sec / tr))), max(1, int(np.round(skip_sec / tr)))
    videos, runs = [], []
    for run, (_, rows) in enumerate(timing.groupby("run_id", sort=False)):
        start = float(np.sum(run_trs[:run])) * tr
        for _, row in rows.iterrows():
            dur = row["duration_sec"]
            n_windows = max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1) if dur >= bin_sec else 0
            first = int(np.round((row["onset_sec"] - start + delay_sec) / tr))
            if n_windows == 0 or first >= run_trs[run] or first < 0:
                continue
            for i in range(n_windows):
                if first + i * skip_trs + bin_trs > run_trs[run]:
                    break
                videos.append(row["video_id"])
                runs.append(run)
    return np.array(videos), np.array(runs)


def neighbour_columns(template, left_surface, right_surface, workbench, cache_dir, subject, k):
    """(n_grayordinates, k) neighbour columns in grayordinate space; -1 marks medial-wall neighbours."""
    from rsa.searchlight import get_neighbors

    left, right = get_cortex_vertex_indices(get_bm_axis(template))
    columns, offset = [], 0
    for hem, surface, index in (("left", left_surface, left), ("right", right_surface, right)):
        neighbours = get_neighbors(surface, workbench, subject, hem, k, Path(cache_dir))
        index = index.astype(np.int64)
        to_column = np.full(neighbours.shape[0], -1, dtype=np.int64)
        to_column[index] = np.arange(len(index))
        found = to_column[neighbours[index]]
        columns.append(np.where(found >= 0, found + offset, -1))
        offset += len(index)
    return np.concatenate(columns).astype(np.int32)


def model_gram(x):
    """Frobenius-normalized, window-centered Gram matrix of an embedding, flattened."""
    centered = x - x.mean(0, keepdims=True)
    gram = centered @ centered.T
    return (gram / np.linalg.norm(gram)).astype(np.float32).ravel()


def searchlight_batches(ncols, verts, batch):
    """Yield (vertex batch, neighbour columns) grouped by neighbourhood size."""
    n_valid = (ncols >= 0).sum(1)
    order = np.argsort(ncols < 0, axis=1, kind="stable")
    sorted_cols = np.take_along_axis(ncols, order, 1)
    selected = np.zeros(ncols.shape[0], bool)
    selected[verts] = True
    for size in np.unique(n_valid):
        if size < 2:
            continue
        group = np.where((n_valid == size) & selected)[0]
        for s in range(0, len(group), batch):
            bv = group[s:s + batch]
            yield bv, sorted_cols[bv, :size].astype(np.int64)


def noncv_pass(fmri, grams, ncols, verts, device, batch=64):
    """Non-cross-validated CKA between each searchlight's window Gram matrix and each model Gram matrix."""
    model = torch.from_numpy(np.stack(list(grams.values()), 1)).to(device)
    data = torch.from_numpy(fmri.T.astype(np.float32)).to(device)
    out = {name: np.zeros(ncols.shape[0], np.float32) for name in grams}
    for bv, cols in searchlight_batches(ncols, verts, batch):
        hood = data[torch.from_numpy(cols).to(device)].permute(0, 2, 1).float()
        hood = hood - hood.mean(1, keepdim=True)
        gram = torch.bmm(hood, hood.transpose(1, 2)).reshape(len(bv), -1)
        score = (gram @ model) / torch.linalg.norm(gram, dim=1, keepdim=True).clamp(min=1e-10)
        for j, name in enumerate(grams):
            out[name][bv] = score[:, j].cpu().numpy()
    return out


def to_vmk(path):
    """Load an (M, K, V) partition array as a contiguous (V, M, K) array."""
    partitions = np.load(path, mmap_mode="r")
    n_groups, n_windows, n_vertices = partitions.shape
    out = np.empty((n_vertices, n_groups, n_windows), np.float32)
    for c0 in range(0, n_vertices, 4096):
        out[c0:c0 + 4096] = np.ascontiguousarray(partitions[:, :, c0:c0 + 4096].transpose(2, 0, 1))
    return out


def whitener(sigma):
    """Symmetric inverse square root of the window-centered noise covariance and its diagnostics."""
    n = sigma.shape[0]
    centering = np.eye(n) - 1.0 / n
    values, vectors = np.linalg.eigh(centering @ sigma @ centering)
    null = np.argmax(np.abs(vectors.sum(0)))
    null_value = float(values[null])
    keep = np.arange(n) != null
    values, vectors = values[keep], vectors[:, keep]
    mean = values.mean()
    floored = np.maximum(values, NOISE_FLOOR * mean)
    stats = {
        "eigenvalue_min": float(values.min()), "eigenvalue_max": float(values.max()),
        "eigenvalue_mean": float(mean), "n_floored": int((values < NOISE_FLOOR * mean).sum()),
        "dropped_null_eigenvalue": null_value, "mean_variance": float(np.diag(sigma).mean()),
    }
    return (vectors / np.sqrt(floored)) @ vectors.T, stats


def noise_covariance(ut, device, chunk=2048):
    """Covariance between windows of the deviations of each partition from the partition mean."""
    n_vertices, n_groups, n_windows = ut.shape
    total = np.zeros((n_windows, n_windows))
    for c0 in range(0, n_vertices, chunk):
        y = torch.from_numpy(ut[c0:c0 + chunk]).to(device)
        d = (y - y.mean(1, keepdim=True)).reshape(-1, n_windows)
        total += (d.T @ d).double().cpu().numpy()
    return total / (n_groups - 1) / n_vertices


def noise_model(ut, run_windows, work_dir, stem, device):
    sigma = noise_covariance(ut, device)
    whiten, stats = whitener(sigma)
    corr = sigma / np.outer(np.sqrt(np.diag(sigma)), np.sqrt(np.diag(sigma)))
    run = np.repeat(np.arange(len(run_windows)), run_windows)
    n = len(run)

    def lag(lag_):
        return float(np.mean([corr[i, i + lag_] for i in range(n - lag_) if run[i] == run[i + lag_]]))

    stats = {"lag1_mean_corr_within_runs": lag(1), "lag2_mean_corr_within_runs": lag(2),
             "cross_run_mean_abs_corr": float(np.abs(corr[run[:, None] != run[None, :]]).mean()), **stats}
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / f"{stem}_noise_covariance.npy", sigma)
    np.save(work_dir / f"{stem}_whitener.npy", whiten)
    (work_dir / f"{stem}_noise_stats.json").write_text(json.dumps(stats, indent=1))
    log.info("noise covariance: %s", json.dumps(stats))
    return whiten


def cv_gram(y, n_groups, n_vertices):
    """Cross-validated Gram matrix over windows: (S'S - Z'Z) / (M (M - 1) P), summing over group pairs m != n."""
    n_batch, _, n_windows = y.shape
    group_sum = y.reshape(n_batch, n_vertices, n_groups, n_windows).sum(2)
    return (torch.bmm(group_sum.transpose(1, 2), group_sum) - torch.bmm(y.transpose(1, 2), y)) / (
        n_groups * (n_groups - 1) * n_vertices)


def double_center(g):
    return g - g.mean(1, keepdim=True) - g.mean(2, keepdim=True) + g.mean((1, 2), keepdim=True)


def prefetch(ut, jobs, depth=3, workers=2):
    q = queue.Queue(depth)

    def work(chunk):
        for bv, cols in chunk:
            q.put((bv, ut[cols]))

    for i in range(workers):
        threading.Thread(target=work, args=(jobs[i::workers],), daemon=True).start()
    for _ in range(len(jobs)):
        yield q.get()


def cv_pass(ut, ncols, embeddings, whiten, verts, device, batch=32):
    """Cross-validated CKA with and without autoregressive whitening; {(model, measure): map}."""
    torch.backends.cuda.matmul.allow_tf32 = False
    _, n_groups, n_windows = ut.shape
    names = list(embeddings)
    w = torch.from_numpy(whiten.astype(np.float32)).to(device)
    plain, white = [], []
    for name in names:
        centered = embeddings[name] - embeddings[name].mean(0, keepdims=True)
        whitened = whiten @ centered
        plain.append(model_gram(embeddings[name]))
        gram = whitened @ whitened.T
        white.append((gram / np.linalg.norm(gram)).astype(np.float32).ravel())
    plain = torch.from_numpy(np.stack(plain, 1)).to(device)
    white = torch.from_numpy(np.stack(white, 1)).to(device)
    out = {(name, m): np.zeros(ncols.shape[0], np.float32) for name in names for m in ("cv", "cv-ar")}
    jobs = list(searchlight_batches(ncols, verts, batch))
    log.info("cv pass: %d batches of up to %d searchlights, %d vertices, M %d, K %d",
             len(jobs), batch, len(verts), n_groups, n_windows)
    for j, (bv, hood) in enumerate(prefetch(ut, jobs)):
        n_batch, size = hood.shape[:2]
        y = torch.from_numpy(hood).to(device).reshape(n_batch, size * n_groups, n_windows)
        gram = cv_gram(y, n_groups, size)
        centered = double_center(gram).reshape(n_batch, -1)
        plain_score = (centered @ plain) / torch.linalg.norm(centered, dim=1).clamp(min=1e-20)[:, None]
        whitened = (w @ gram @ w).reshape(n_batch, -1)
        white_score = (whitened @ white) / torch.linalg.norm(whitened, dim=1).clamp(min=1e-20)[:, None]
        for jn, name in enumerate(names):
            out[(name, "cv")][bv] = plain_score[:, jn].cpu().numpy()
            out[(name, "cv-ar")][bv] = white_score[:, jn].cpu().numpy()
        if j % 200 == 0:
            log.info("  batch %d/%d", j, len(jobs))
    return out


def loglik(r, n_windows):
    n_pairs = n_windows * (n_windows - 1) / 2
    return (-(n_pairs / 2) * np.log1p(-np.minimum(r.astype(np.float64) ** 2, 1 - 1e-12))).astype(np.float32)


def loglik_diff(r_a, r_b, n_windows):
    return (loglik(r_a, n_windows).astype(np.float64) - loglik(r_b, n_windows)).astype(np.float32)
