"""Searchlight RSA of the group-average responses with Euclidean, Mahalanobis or cosine RDMs.

Brain RDMs: squared Euclidean distance between window patterns (euclidean); the same after noise normalization
(mahalanobis), with the noise covariance estimated from the four presentations of the repeated clip (residuals of each
window around its mean over presentations, shrunk towards the diagonal: Schäfer & Strimmer 2005, as in rsatoolbox;
Walther et al. 2016); 1 - cosine similarity (cosine); or crossnobis (Walther et al. 2016): over the windows of the
repeated clip only, the noise-normalized distance cross-validated across its four presentations,
d(i, j) = mean over presentations m != m' of (x_i^m - x_j^m)' P (x_i^m' - x_j^m'), with P as for mahalanobis. Model RDMs: Euclidean distance between per-run-centered
embedding windows (1 - cosine for cosine; for crossnobis, of each clip window's embedding averaged over the
presentations).

Spearman: rho_x, the Spearman correlation of the brain RDM with each component RDM; sp_x, the semi-partial Spearman correlation: the
Pearson correlation of the brain RDM ranks with the residual of the x RDM ranks after least-squares regression on the
other two RDM ranks (Tong et al. 2022); partial_x, the partial Spearman correlation given the other two (Pearson partial correlation of the
ranks; Clarke & Tyler 2014). Kendall: tau-b maps only.

With --subjects-list, the Euclidean Spearman maps are computed on the GPU from each subject's own binned responses
(z-scored per run), one file per kind as for the group average, with the within-block maps x_block01.. (--n-blocks) for
a two-factor bootstrap; rsa/rdm_groupstats.py tests them over subjects. --shift circularly shifts each run's brain
windows by half the run as a misaligned baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import rankdata

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from shared.cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap
from cka.shared.kernels import partial_correlations, semipartial, window_index
from rsa.searchlight import REPEATED_CLIPS, get_neighbors
from rsa.shared.rsa_utils import assert_segment_timing, preprocess_fmri, process_model_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from rsa.shared.rdm_searchlight import (  # noqa: F401
    BRAIN_RDMS,
    DEFINITIONS,
    DISTANCES,
    OWN_TAG,
    _brain_rdm,
    _searchlight_kendall,
    _searchlight_spearman,
    _sort_count_inversions,
    _tau_b,
    _tied_pairs,
    _unit_ranks,
    hood_columns,
    model_rdm,
    searchlight,
    shrunk_covariance,
    tie_structure,
    unit_ranks,
)
# Same control sets as encoding/analysis.sh and cka/analysis.sh: "name audio_model video_model".
CONTROL_SETS = [
    "mae dasheng-0.6b-d75 videomaev2-large-d75",
    "mae-large dasheng-1.2b videomaev2-giant-d50",
    "latent openbeats-large-i2 vjepa2-vitl-d75",
    "large-mixed dasheng-1.2b vjepa2-vitg-d75",
    "speech-wavlm wavlm-large-d75 vjepa2-vitl-d75",
    "speech-w2vbert w2v-bert-2.0-d75 vjepa2-vitl-d75",
    "text-contrastive clap-larger pe-core-l14",
    "text-asr whisper-large-v3 pe-core-l14",
]
COMPARATORS = {"spearman": "spearman", "kendall": "kendalltau"}
KINDS = {"models": ["rho_a", "rho_v", "rho_j"], "semipartial": ["sp_a", "sp_v", "sp_j"],
         "partial": ["partial_a", "partial_v", "partial_j"]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preprocessed-dir", required=True)
    p.add_argument("--fmri-suffix", default="raw")
    p.add_argument("--subject", default="group_average")
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--left-surface", required=True)
    p.add_argument("--right-surface", required=True)
    p.add_argument("--workbench", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--geodesic-cache-dir", default=None)
    p.add_argument("--models", nargs="+", default=["pe-av-small-16-frame"])
    p.add_argument("--controls", nargs="+", default=["own", "mae", "text-asr"],
                   help="own (audio and video from the model itself) and/or names from CONTROL_SETS.")
    p.add_argument("--distance", choices=list(DISTANCES), default="euclidean")
    p.add_argument("--comparator", choices=list(COMPARATORS), default="spearman")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=None)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--model-norm", choices=["zscore", "center"], default="center",
                   help="Per-run embedding normalization (fMRI is always z-scored).")
    p.add_argument("--subjects-list", default=None, help="Per-subject mode (Euclidean, Spearman).")
    p.add_argument("--subject-cache", default=None,
                   help="Binned subject responses, {subject}_{suffix}_delay{D}s_bin{B}s_skip{S}s.npy, (V, K) z-scored.")
    p.add_argument("--n-blocks", type=int, default=16, help="Temporal blocks for the two-factor bootstrap.")
    p.add_argument("--shift", action="store_true", help="Circularly shift each run's brain windows by half the run.")
    return p.parse_args()


def repeated_windows(timing, run_trs, args, skip_sec):
    """Window rows of the repeated clip and their position within the clip (the condition label)."""
    videos, _ = window_index(timing, run_trs, args.bin_sec, skip_sec, args.delay_sec, args.tr)
    rows, cond = [], []
    for clip in REPEATED_CLIPS:
        idx = np.flatnonzero(videos == clip)
        rows.append(idx)
        cond.append(np.arange(len(idx)))
    assert len({len(r) for r in rows}) == 1, [len(r) for r in rows]
    return np.concatenate(rows), np.concatenate(cond), len(videos)


def subject_spearman_gpu(data, cols, ncols, model_ranks, batch=96):
    """(n_targets, n_searchlights) Spearman correlation of each searchlight's squared-Euclidean RDM over the rows of
    `data` (windows x grayordinates) with each centered, unit-norm rank vector in `model_ranks`."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, colix = (torch.as_tensor(a, device=dev) for a in np.tril_indices(data.shape[0], k=-1))
    n_pairs = rows.numel()
    X = torch.as_tensor(np.ascontiguousarray(data, np.float32), device=dev)
    M = torch.as_tensor(model_ranks.T, dtype=torch.float32, device=dev)
    cols_t = torch.as_tensor(cols, device=dev)
    mask = torch.as_tensor(np.arange(cols.shape[1])[None] < ncols[:, None], dtype=torch.float32, device=dev)
    ladder = torch.arange(n_pairs, device=dev, dtype=torch.float32) - (n_pairs - 1) / 2
    out = torch.zeros((model_ranks.shape[0], len(cols)), device=dev)
    for start in range(0, len(cols), batch):
        hood = X[:, cols_t[start:start + batch]].permute(1, 0, 2) * mask[start:start + batch, None, :]
        gram = torch.bmm(hood, hood.transpose(1, 2))
        q = torch.diagonal(gram, dim1=1, dim2=2)
        d = q[:, rows] + q[:, colix] - 2 * gram[:, rows, colix]
        # ponytail: ties in the brain RDM get ordinal rather than average ranks; exact float ties are negligible here
        r = torch.empty_like(d).scatter_(1, d.argsort(dim=1), ladder.expand_as(d))
        out[:, start:start + batch] = (r / r.norm(dim=1, keepdim=True) @ M).T
    out[:, torch.as_tensor(ncols < 2, device=dev)] = 0
    return out.cpu().numpy()


def subject_config(k, delay_sec, bin_sec, skip_sec, model_norm):
    return f"k{k}_delay{int(delay_sec)}s_bin{int(bin_sec)}s_skip{int(skip_sec)}s_euclidean_spearman_{model_norm}"


def subject_map_path(output_dir, subject, model, tag, kind, cfg, fmri_suffix, shifted=False):
    """Per-subject Euclidean Spearman maps of one kind (KINDS), followed by their within-block maps."""
    return (Path(output_dir) / "subject_data" / subject / model
            / f"rsa_59k_{fmri_suffix}_{cfg}_{tag}{'_shifted' if shifted else ''}_{kind}.dscalar.nii")


def block_names(names, n_blocks):
    return [f"{x}_block{b + 1:02d}" for x in names for b in range(n_blocks)]


def run_subjects(args, runs, embs, timing, run_trs, skip_sec):
    _, window_runs = window_index(timing, run_trs, args.bin_sec, skip_sec, args.delay_sec, args.tr)
    n = len(window_runs)
    order = np.arange(n)
    if args.shift:
        order = np.concatenate([np.roll(np.flatnonzero(window_runs == r), (window_runs == r).sum() // 2)
                                for r in np.unique(window_runs)])
    left_idx, right_idx = get_cortex_vertex_indices(get_bm_axis(args.template_cifti))
    cache_dir = Path(args.geodesic_cache_dir) if args.geodesic_cache_dir else Path(args.output_dir) / "_geodesic_cache"
    parts, offset = [], 0
    for hem, surf, idx in (("left", args.left_surface, left_idx), ("right", args.right_surface, right_idx)):
        cols, ncols = hood_columns(get_neighbors(surf, args.workbench, "group_average", hem, args.k, cache_dir),
                                   np.asarray(idx))
        parts.append((cols + offset, ncols))
        offset += len(idx)
    cols, ncols = np.concatenate([c for c, _ in parts]), np.concatenate([c for _, c in parts])

    edges = np.round(np.linspace(0, n, args.n_blocks + 1)).astype(int)
    segments = [(0, n)] + list(zip(edges[:-1], edges[1:]))
    ranks, corr = [], []
    for start, end in segments:
        tril = np.tril_indices(end - start, k=-1)
        r = np.stack([unit_ranks(model_rdm(embs[k][start:end], tril, "euclidean")) for _, _, keys in runs for k in keys])
        ranks.append(r)
        corr.append([r[3 * i:3 * i + 3] @ r[3 * i:3 * i + 3].T for i in range(len(runs))])

    cfg = subject_config(args.k, args.delay_sec, args.bin_sec, skip_sec, args.model_norm)
    from preprocessing.preprocess_individual import load_subjects
    subjects = load_subjects(args.subjects_list)

    def path(subject, model, tag, kind):
        return subject_map_path(args.output_dir, subject, model, tag, kind, cfg, args.fmri_suffix, args.shift)

    for i, subject in enumerate(subjects):
        if all(path(subject, model, tag, kind).exists() for model, tag, _ in runs for kind in KINDS):
            continue
        data = np.load(Path(args.subject_cache) / f"{subject}_{args.fmri_suffix}_delay{int(args.delay_sec)}s_"
                       f"bin{int(args.bin_sec)}s_skip{int(skip_sec)}s.npy").T.astype(np.float32)[order]
        assert data.shape[0] == n, (subject, data.shape)
        values = []
        for (start, end), r, rc in zip(segments, ranks, corr):
            v = subject_spearman_gpu(data[start:end], cols, ncols, r)
            per_pair = []
            for p in range(len(runs)):
                block = v[3 * p:3 * p + 3]
                per_pair.append(np.concatenate([block, semipartial(block, rc[p]), partial_correlations(block, rc[p])]))
            values.append(per_pair)
        for p, (model, tag, _) in enumerate(runs):
            for k, (kind, names) in enumerate(KINDS.items()):
                rows = slice(3 * k, 3 * k + 3)
                maps = [values[0][p][rows]]
                if args.n_blocks > 1:
                    maps.append(np.stack([values[b][p][rows] for b in range(1, len(segments))], axis=1)
                                .reshape(-1, maps[0].shape[1]))
                    names = names + block_names(names, args.n_blocks)
                out = path(subject, model, tag, kind)
                out.parent.mkdir(parents=True, exist_ok=True)
                save_cifti_multimap(np.concatenate(maps), names, args.template_cifti, str(out))
        log.info(f"[{i + 1}/{len(subjects)}] {subject}")


def main():
    args = parse_args()
    skip_sec = args.skip_sec if args.skip_sec is not None else args.bin_sec
    sets = {"own": None, **{s.split()[0]: s.split()[1:] for s in CONTROL_SETS}}
    unknown = set(args.controls) - set(sets)
    if unknown:
        raise SystemExit(f"Unknown control sets: {sorted(unknown)}")

    timing = pd.read_csv(args.timing_csv)
    run_trs = np.load(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    assert_segment_timing(timing, args.bin_sec, args.tr, args.delay_sec, skip_sec, run_trs)
    rep_rows, rep_cond, n_windows = repeated_windows(timing, run_trs, args, skip_sec)
    if not args.subjects_list:
        fmri = nib.load(str(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")
                        ).get_fdata(dtype=np.float32).T
        fmri = preprocess_fmri(fmri, timing, run_trs, args.bin_sec, args.tr, args.delay_sec, skip_sec=skip_sec,
                               normalize=True)
        log.info(f"fMRI binned z-scored: {fmri.shape}")
        assert n_windows == fmri.shape[0], (n_windows, fmri.shape)

    folder = f"bin{int(args.bin_sec)}s_skip{int(skip_sec)}s"
    n_cond = int(rep_cond.max()) + 1
    n_part = len(rep_cond) // n_cond
    assert np.array_equal(rep_cond, np.tile(np.arange(n_cond), n_part)), "repeats must be ordered by presentation"
    n_items = n_cond if args.distance == "crossnobis" else n_windows
    tril_idx = np.tril_indices(n_items, k=-1)
    rdms, embs = {}, {}

    def rdm(name, kind):
        key = f"{name}_{kind}"
        if key not in rdms:
            emb = process_model_embeddings(str(Path(args.embeddings_dir) / name / folder / f"{key}.npy"), timing,
                                           bin_sec=args.bin_sec, tr=args.tr, run_trs=run_trs,
                                           delay_sec=args.delay_sec, hrf=False, skip_sec=skip_sec,
                                           model_norm=args.model_norm)
            assert emb.shape[0] == n_windows, (key, emb.shape, n_windows)
            embs[key] = emb
            if args.distance == "crossnobis":
                emb = emb[rep_rows].reshape(n_part, n_cond, -1).mean(axis=0)
            rdms[key] = model_rdm(emb, tril_idx, args.distance)
            log.info(f"Model RDM {key}: {emb.shape}")
        return key

    runs = []
    for model in args.models:
        for tag in args.controls:
            audio, video = sets[tag] or (model, model)
            runs.append((model, OWN_TAG if tag == "own" else tag, [rdm(audio, "a"), rdm(video, "v"), rdm(model, "av")]))

    if args.subjects_list:
        return run_subjects(args, runs, embs, timing, run_trs, skip_sec)

    names, targets = list(rdms), list(rdms.values())
    n_bins = n_items
    log.info(f"n_bins={n_bins} n_pairs={len(tril_idx[0]):,} targets={len(targets)} distance={args.distance}")

    left_idx, right_idx = get_cortex_vertex_indices(get_bm_axis(args.template_cifti))
    cache_dir = Path(args.geodesic_cache_dir) if args.geodesic_cache_dir else Path(args.output_dir) / "_geodesic_cache"
    values = np.zeros((len(targets), fmri.shape[1]))
    offset = 0
    for hem, surf, idx in (("left", args.left_surface, left_idx), ("right", args.right_surface, right_idx)):
        neighbors = get_neighbors(surf, args.workbench, args.subject, hem, args.k, cache_dir)
        values[:, offset:offset + len(idx)] = searchlight(fmri[:, offset:offset + len(idx)], targets, neighbors,
                                                          np.asarray(idx), n_bins, args.distance, args.comparator,
                                                          (rep_rows, rep_cond))
        log.info(f"{hem}: done")
        offset += len(idx)
    value = dict(zip(names, values))

    cfg = (f"k{args.k}_delay{int(args.delay_sec)}s_bin{int(args.bin_sec)}s_skip{int(skip_sec)}s_"
           f"{args.distance}_{COMPARATORS[args.comparator]}_{args.model_norm}")
    for model, tag, keys in runs:
        directory = Path(args.output_dir) / args.subject / model
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"rsa_59k_{args.fmri_suffix}_{cfg}_{tag}"
        c = np.stack([value[k] for k in keys])
        provenance = {
            "audio": keys[0], "video": keys[1], "joint": keys[2], "tag": tag, "k": args.k, "bin_sec": args.bin_sec,
            "skip_sec": skip_sec, "delay_sec": args.delay_sec, "model_norm": args.model_norm, "n_windows": n_bins,
            "brain_rdm": BRAIN_RDMS[args.distance],
            "model_rdm": "1 - cosine similarity" if args.distance == "cosine" else "Euclidean distance",
        }
        if args.comparator == "kendall":
            outputs = [(["tau_a", "tau_v", "tau_j"], c, "models")]
            provenance["tau"] = DEFINITIONS["tau"]
        else:
            ranks = np.stack([unit_ranks(rdms[k]) for k in keys])
            R = ranks @ ranks.T
            outputs = [(["rho_a", "rho_v", "rho_j"], c, "models"),
                       (["sp_a", "sp_v", "sp_j"], semipartial(c, R), "semipartial"),
                       (["partial_a", "partial_v", "partial_j"], partial_correlations(c, R), "partial")]
            provenance.update({k: DEFINITIONS[k] for k in ("rho", "semipartial", "partial")},
                              component_rank_correlation={"order": "a, v, j", "matrix": R.tolist()})
        for maps, data, kind in outputs:
            save_cifti_multimap(data.astype(np.float32), maps, args.template_cifti,
                                str(directory / f"{stem}_{kind}.dscalar.nii"))
        (directory / f"{stem}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        log.info(f"Saved {directory / stem}")


if __name__ == "__main__":
    main()
