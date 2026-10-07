"""Searchlight centered kernel alignment (CKA): window indexing, Gram matrices, semi-partial and commonality maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cifti_io import get_bm_axis, get_cortex_vertex_indices
from encoding.shared.fold_evaluator import ALL_SUBSETS, partition_variance

COMPONENTS = "avj"
RESIDUAL_FLOOR = 1e-6


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


def noise_ceiling_pass(subjects, ncols, verts, device, batch=1024):
    """Lower and upper bounds on the mean over subjects of the subject CKA (Nili et al. 2014).

    subjects: one (V, K) response array per subject (memory maps are fine). With b_s the unit-norm, window-centered Gram
    matrix of subject s in a searchlight and B the mean of the b_s, upper = mean_s cos(b_s, B) and
    lower = mean_s cos(b_s, B_-s), B_-s = (N B - b_s) / (N - 1).
    """
    n, n_windows = len(subjects), subjects[0].shape[1]
    lower, upper = np.zeros(ncols.shape[0], np.float32), np.zeros(ncols.shape[0], np.float32)
    for bv, cols in searchlight_batches(ncols, verts, batch):
        needed, local = np.unique(cols, return_inverse=True)
        local = torch.from_numpy(local.reshape(cols.shape)).to(device)
        data = torch.empty((n, len(needed), n_windows), dtype=torch.float16, device=device)
        for s, y in enumerate(subjects):
            data[s] = torch.from_numpy(np.asarray(y[needed], np.float16))

        def scaled(s):
            hood = data[s][local].float()
            hood = hood - hood.mean(2, keepdim=True)
            norm = torch.linalg.norm(torch.bmm(hood, hood.transpose(1, 2)), dim=(1, 2)).clamp(min=1e-10)
            return hood / norm.sqrt()[:, None, None]

        mean = torch.zeros((len(bv), n_windows, n_windows), device=device)
        for s in range(n):
            h = scaled(s)
            mean.baddbmm_(h.transpose(1, 2), h, alpha=1.0 / n)
        square = mean.square().sum((1, 2))
        low, up = torch.zeros_like(square), torch.zeros_like(square)
        for s in range(n):
            h = scaled(s)
            dot = (torch.bmm(h, mean) * h).sum((1, 2))
            up += dot / square.sqrt()
            low += (n * dot - 1) / (n * n * square - 2 * n * dot + 1).clamp(min=1e-20).sqrt()
        lower[bv], upper[bv] = (low / n).cpu().numpy(), (up / n).cpu().numpy()
    return lower, upper


def semipartial_coefficients(gram):
    """Rows x: coefficients on (c_A, c_V, c_J) giving the cosine of G_cv with the residual of G_x on the other two, and the residual norms ||r_x||."""
    coefficients, norms = np.zeros((3, 3)), np.zeros(3)
    for x in range(3):
        others = [i for i in range(3) if i != x]
        beta = np.linalg.pinv(gram[np.ix_(others, others)]) @ gram[others, x]
        coefficients[x, x], coefficients[x, others] = 1.0, -beta
        norms[x] = np.sqrt(max(1.0 - gram[x, others] @ beta, 0.0))
    return coefficients, norms


def semipartial(cka, cosines):
    """Correlations (3, n) of the brain matrix with the residual of each component matrix on the other two.

    cka (3, n): cosines of the brain matrix with the a, v, j matrices; cosines (3, 3) among them. Nothing is fitted to the brain.
    """
    coefficients, norms = semipartial_coefficients(cosines)
    used = norms >= RESIDUAL_FLOOR
    scale = np.where(used[:, None], coefficients / np.where(used, norms, 1.0)[:, None], 0.0)
    return scale @ np.asarray(cka, np.float64)


def commonality(cka, cosines):
    """Commonality partition of the brain matrix over the a, v, j matrices (Seibold & McPhee 1979).

    R2(S) = c_S' R_SS^-1 c_S is the squared multiple correlation of the brain matrix with the component matrices in S
    (c: cosines with the brain, R: cosines among the components); the seven regions follow by inclusion-exclusion,
    as in the encoding partition. Returns {name: (n,)} with r2_{subset}, the seven regions and shared_av.
    """
    c = np.asarray(cka, np.float64)
    r2 = {}
    for subset in ALL_SUBSETS:
        rows = [COMPONENTS.index(band) for band in subset]
        r2[subset] = np.einsum("in,ij,jn->n", c[rows], np.linalg.pinv(cosines[np.ix_(rows, rows)]), c[rows])
    return {**{f"r2_{k}": v for k, v in r2.items()}, **partition_variance(r2), "shared_av": r2["a"] + r2["v"] - r2["av"]}
