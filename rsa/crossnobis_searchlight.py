"""
rsa/crossnobis_searchlight.py
==============================
Inter-subject crossvalidated Mahalanobis (crossnobis) searchlight RSA.

Cross-validates across N=175 subjects watching the same full movie, using
K≈626 five-second bins (vs the old design: N=4 runs, K=15 bins per clip).

Formula (Euclidean metric, i.e. identity precision matrix):

    d̂(t1,t2) = 1/(N(N−1)) × [||Σᵢ xᵢ(t1) − Σᵢ xᵢ(t2)||²
                               − Σᵢ ||xᵢ(t1) − xᵢ(t2)||²]

The expected value equals the true Euclidean neural distance (noise cancels
because subjects are independent). Power gain over 4-run repeated-clip design:
~75× reduction in τ_a variance (K=626 vs K=15, n_pairs=195k vs 105).

The gram-matrix decomposition avoids materialising subject-pair differences:
    ||Σᵢ d_i||² − Σᵢ ||d_i||² is computed from sum_X and gram_sum alone.

Ref: Walther et al. 2016; Schütt et al. 2023 §5.1.1.

Usage:
    python rsa/crossnobis_searchlight.py \\
        --subjects-list /path/to/subjects.txt \\
        --raw-dir       /path/to/individual-59k/ \\
        --sg-filter --psc \\
        --timing-csv    /path/to/movie_timing.csv \\
        --embeddings-dir /path/to/model_embeddings \\
        --template-cifti /path/to/template.dscalar.nii \\
        --left-surface  /path/to/L.surf.gii \\
        --right-surface /path/to/R.surf.gii \\
        --workbench     /path/to/wb_command \\
        --output-dir    /path/to/outputs/rsa/raw \\
        --model pe-av-small-16-frame --modality av \\
        --k 100 --bin-sec 5.0 --delay-sec 5.0 --tr 1.0

Output:
    {output_dir}/groupstats/{model}_{modality}/
        k{k}_delay{d}s_bin{b}s_skip{s}s_rho_a_isub/
            crossnobis_isub_rho_a_k{k}_delay{d}s_bin{b}s_skip{s}s.npy
            crossnobis_isub_rho_a_k{k}_delay{d}s_bin{b}s_skip{s}s.dscalar.nii
"""

import argparse
import gc
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.rsa_utils import preprocess_fmri, process_model_embeddings
from rsa.searchlight import get_neighbors, _rho_sigmap
from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_JOBS = int(os.environ.get("_RSA_N_JOBS", -1))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Inter-subject crossnobis searchlight RSA (full movie, N subjects). "
            "Walther et al. 2016; Schütt et al. 2023."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subjects-list", required=True, dest="subjects_list",
                   help="Text file with one subject ID per line.")
    p.add_argument("--raw-dir",       required=True, dest="raw_dir",
                   help="Directory of raw 7T CIFTI files (individual-59k/).")
    p.add_argument("--sg-filter",  action="store_true", dest="sg_filter",
                   help="Apply Savitzky-Golay high-pass filter per run.")
    p.add_argument("--psc",        action="store_true",
                   help="Convert to percent signal change per run.")
    p.add_argument("--gsr",        action="store_true",
                   help="Apply global signal regression per run.")
    p.add_argument("--timing-csv",     required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--left-surface",   required=True)
    p.add_argument("--right-surface",  required=True)
    p.add_argument("--workbench",      required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--model",          required=True)
    p.add_argument("--modality",       required=True)
    p.add_argument("--k",   type=int,   required=True,
                   help="Searchlight neighbourhood size (vertices).")
    p.add_argument("--bin-sec",  type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec",
                   help="Window stride (default: bin-sec, no overlap).")
    p.add_argument("--delay-sec",type=float, default=5.0)
    p.add_argument("--hrf",  action="store_true",
                   help="Convolve model embeddings with SPM canonical HRF.")
    p.add_argument("--tr",   type=float, required=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir")
    p.add_argument("--n-jobs", type=int, default=N_JOBS, dest="n_jobs")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def load_subjects(subjects_list: str) -> list[str]:
    subs = []
    for ln in Path(subjects_list).read_text().strip().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            subs.append(ln.split()[0])
    return subs


def build_model_rdm(
    emb_path: str,
    timing_df: pd.DataFrame,
    bin_sec: float,
    tr: float,
    run_trs: np.ndarray,
    delay_sec: float,
    skip_sec: float,
    hrf: bool,
) -> np.ndarray:
    """Full-movie model correlation-distance RDM, shape (K, K) float64."""
    emb = process_model_embeddings(
        emb_path, timing_df, bin_sec, tr, run_trs,
        delay_sec=delay_sec, hrf=hrf, skip_sec=skip_sec, normalize=True,
    ).astype(np.float64)
    mu   = emb.mean(axis=1, keepdims=True)
    ec   = emb - mu
    nrms = np.linalg.norm(ec, axis=1, keepdims=True)
    nrms[nrms < 1e-10] = 1.0
    en   = ec / nrms
    rdm  = np.clip(1.0 - en @ en.T, 0.0, 2.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm


# =============================================================================
# Per-vertex computation
# =============================================================================

def _crossnobis_isub_vertex(
    surf_v: int,
    all_data_hem: np.ndarray,    # (N_subs, K_bins, n_hem_verts) float32  [shared]
    model_rdm_tril: np.ndarray,  # (n_pairs,) float64
    neighbors: np.ndarray,       # (n_surf_verts, k) int32
    vertex_to_col: np.ndarray,   # (n_surf_verts,) int32  (-1 = medial wall)
    tril_idx: tuple,             # (rows, cols) from np.tril_indices(K, k=-1)
) -> float:
    """Inter-subject crossnobis RSA at one searchlight vertex.

    Steps
    -----
    1. Gather neighbourhood: hood (N, K, n_nb)
    2. Cross term via sum_X (K, n_nb): ||sum_X(t1) - sum_X(t2)||²
    3. Self term via gram_sum (K, K): Σᵢ ||xᵢ(t1) - xᵢ(t2)||²
    4. xnobis = (cross - self) / (N*(N-1))
    5. τ_a (Kendall) vs model RDM lower triangle
    """
    neighbor_surf = neighbors[surf_v]
    neighbor_cols = vertex_to_col[neighbor_surf]
    neighbor_cols = neighbor_cols[neighbor_cols >= 0]
    n_nb = len(neighbor_cols)
    if n_nb < 2:
        return 0.0

    # (N, K, n_nb) — copy from shared array to avoid cache thrashing
    hood = all_data_hem[:, :, neighbor_cols].astype(np.float64)
    N, K, _ = hood.shape
    rows, cols = tril_idx

    # Cross term: ||Σᵢ(xᵢ(t1)-xᵢ(t2))||² = ||sum_X(t1) - sum_X(t2)||²
    sum_X    = hood.sum(axis=0)                                     # (K, n_nb)
    sum_X_sq = np.einsum("kv,kv->k", sum_X, sum_X)                 # (K,)
    sum_XXT  = sum_X @ sum_X.T                                      # (K, K)
    cross_sq = sum_X_sq[rows] + sum_X_sq[cols] - 2.0 * sum_XXT[rows, cols]  # (n_pairs,)

    # Self term: Σᵢ||xᵢ(t1)-xᵢ(t2)||² via gram matrix
    # gram_sum[t1,t2] = Σᵢ xᵢ(t1)·xᵢ(t2)  — accumulated per subject
    gram_sum = np.zeros((K, K), dtype=np.float64)
    for i in range(N):
        h = hood[i]        # (K, n_nb)
        gram_sum += h @ h.T
    sum_sq  = np.diag(gram_sum)                                     # (K,)
    self_sq = sum_sq[rows] + sum_sq[cols] - 2.0 * gram_sum[rows, cols]  # (n_pairs,)

    xnobis = (cross_sq - self_sq) / (N * (N - 1))

    tau, _ = kendalltau(xnobis, model_rdm_tril, method="auto")
    return float(tau) if np.isfinite(tau) else 0.0


# =============================================================================
# Hemisphere searchlight
# =============================================================================

def run_isub_hemisphere(
    all_data_hem: np.ndarray,    # (N, K, n_hem_verts) float32
    model_rdm_tril: np.ndarray,  # (n_pairs,) float64
    neighbors: np.ndarray,       # (n_surf_verts, k) int32
    surface_indices: np.ndarray, # (n_hem_verts,) int32
    vertex_to_col: np.ndarray,   # (n_surf_verts,) int32
    n_jobs: int = -1,
) -> np.ndarray:
    K        = all_data_hem.shape[1]
    tril_idx = np.tril_indices(K, k=-1)
    n_verts  = all_data_hem.shape[2]
    n_pairs  = len(tril_idx[0])
    log.info(
        f"  Inter-subject crossnobis searchlight: {n_verts:,} vertices, "
        f"K={K} bins, N={all_data_hem.shape[0]} subjects, "
        f"n_pairs={n_pairs:,}  (n_jobs={n_jobs})"
    )
    return np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_crossnobis_isub_vertex)(
                int(surface_indices[v]),
                all_data_hem, model_rdm_tril,
                neighbors, vertex_to_col, tril_idx,
            )
            for v in range(n_verts)
        ),
        dtype=np.float32,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    timing_df = pd.read_csv(args.timing_csv)
    subjects  = load_subjects(args.subjects_list)
    N_req     = len(subjects)

    bin_int   = int(args.bin_sec)
    skip_int  = int(args.skip_sec)
    delay_tag = f"delay{int(args.delay_sec)}s"
    stem      = (f"crossnobis_isub_rho_a_k{args.k}_{delay_tag}"
                 f"_bin{bin_int}s_skip{skip_int}s")
    out_root  = (Path(args.output_dir) / "groupstats" /
                 f"{args.model}_{args.modality}" /
                 f"k{args.k}_{delay_tag}_bin{bin_int}s_skip{skip_int}s_rho_a_isub")
    npy_out   = out_root / f"{stem}.npy"
    cifti_out = out_root / f"{stem}.dscalar.nii"

    if npy_out.exists() and cifti_out.exists():
        log.info(f"Output already exists — skipping: {npy_out.name}")
        return

    log.info(
        f"Inter-subject crossnobis: {N_req} subjects requested, "
        f"model={args.model}, modality={args.modality}"
    )

    # CIFTI vertex structure (left/right split)
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left  = len(left_indices)
    n_right = len(right_indices)
    n_total = n_left + n_right

    # Pre-processing args for preprocess_subject()
    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr
    )

    # ── Load and bin all subjects ─────────────────────────────────────────────
    from preprocess_individual import preprocess_subject

    all_data  = None       # (N_valid, K, n_total) float32 — allocated on first subject
    K_bins    = None
    ref_run_trs = None
    n_valid   = 0

    for idx, sub in enumerate(subjects):
        log.info(f"  [{idx+1}/{N_req}] Loading {sub} ...")
        try:
            data, _, run_trs = preprocess_subject(
                sub, Path(args.raw_dir), args.tr, prep_args
            )
        except Exception as e:
            log.warning(f"  {sub}: load error — {e}; skipping")
            continue

        try:
            binned = preprocess_fmri(
                data, timing_df, run_trs, args.bin_sec, args.tr,
                delay_sec=args.delay_sec, skip_sec=args.skip_sec, normalize=True,
            )  # (K, n_total)
        except Exception as e:
            log.warning(f"  {sub}: bin error — {e}; skipping")
            del data; gc.collect()
            continue

        K = binned.shape[0]

        if K_bins is None:
            K_bins = K
            ref_run_trs = run_trs.copy()
            all_data = np.zeros((N_req, K, n_total), dtype=np.float32)
            log.info(f"  Allocated all_data: {all_data.shape}  "
                     f"~{all_data.nbytes/1e9:.1f} GB")

        if K != K_bins:
            log.warning(f"  {sub}: expected {K_bins} bins, got {K} — skipping")
            del data, binned; gc.collect()
            continue

        if binned.shape[1] != n_total:
            log.warning(f"  {sub}: expected {n_total} verts, got {binned.shape[1]} — skipping")
            del data, binned; gc.collect()
            continue

        all_data[n_valid] = binned
        n_valid += 1
        del data, binned; gc.collect()

    if K_bins is None or n_valid < 2:
        raise RuntimeError(f"Loaded only {n_valid} subjects — need ≥2.")

    all_data = all_data[:n_valid]   # trim unused rows
    log.info(
        f"Loaded {n_valid}/{N_req} subjects: K={K_bins} bins, "
        f"n_pairs={K_bins*(K_bins-1)//2:,}  "
        f"all_data~{all_data.nbytes/1e9:.1f} GB"
    )

    # ── Model RDM (full movie, K×K) ───────────────────────────────────────────
    emb_file = (Path(args.embeddings_dir) / args.model /
                f"bin{bin_int}s_skip{skip_int}s" /
                f"{args.model}_{args.modality}.npy")
    log.info(f"Building full-movie model RDM: {emb_file.name}")
    model_rdm = build_model_rdm(
        str(emb_file), timing_df, args.bin_sec, args.tr, ref_run_trs,
        args.delay_sec, args.skip_sec, args.hrf,
    )
    if model_rdm.shape[0] != K_bins:
        raise RuntimeError(
            f"Model RDM K={model_rdm.shape[0]} ≠ fMRI K={K_bins}. "
            "Mismatch in bin-sec/skip-sec/delay-sec."
        )
    tril_idx       = np.tril_indices(K_bins, k=-1)
    model_rdm_tril = model_rdm[tril_idx].astype(np.float64)
    log.info(
        f"  Model RDM: {model_rdm.shape}  "
        f"range=[{model_rdm_tril.min():.3f}, {model_rdm_tril.max():.3f}]"
    )

    # ── Per-hemisphere searchlight ────────────────────────────────────────────
    cache_dir = (Path(args.geodesic_cache_dir) if args.geodesic_cache_dir
                 else Path(args.output_dir) / "_geodesic_cache")

    corr_full = np.zeros(n_total, dtype=np.float32)
    offset    = 0

    for hem, surf_path, surf_indices, col_slice in [
        ("left",  args.left_surface,  left_indices,  slice(0,       n_left)),
        ("right", args.right_surface, right_indices, slice(n_left,  n_total)),
    ]:
        n_hem = len(surf_indices)
        log.info(f"  Hemisphere: {hem}  ({n_hem:,} vertices)")

        neighbors = get_neighbors(
            surf_path, args.workbench, "group_average", hem, args.k, cache_dir,
        )
        surf_indices_i32 = surf_indices.astype(np.int32)
        n_surf_verts     = neighbors.shape[0]
        vertex_to_col    = np.full(n_surf_verts, -1, dtype=np.int32)
        vertex_to_col[surf_indices_i32] = np.arange(n_hem, dtype=np.int32)

        # Slice the hemisphere columns (view, no copy)
        all_data_hem = all_data[:, :, col_slice]  # (N, K, n_hem) — view

        corr_hem = run_isub_hemisphere(
            all_data_hem, model_rdm_tril, neighbors,
            surf_indices_i32, vertex_to_col, args.n_jobs,
        )
        corr_full[offset : offset + n_hem] = corr_hem

        del neighbors, vertex_to_col, corr_hem
        gc.collect()
        offset += n_hem

    del all_data; gc.collect()

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_out), corr_full)
    log.info(
        f"Saved .npy: {npy_out.name}  "
        f"mean={corr_full.mean():.4f}  max={corr_full.max():.4f}"
    )

    save_cifti_map(corr_full, args.template_cifti, str(cifti_out), "crossnobis_isub_rho_a")
    log.info(f"Saved CIFTI: {cifti_out.name}")

    # Analytical significance map (n_pairs ≈ 195k → highly sensitive)
    n_pairs = len(model_rdm_tril)
    _, sigmap = _rho_sigmap(corr_full, n_pairs)
    sigmap_path = out_root / f"{stem}_sigmap.dscalar.nii"
    save_cifti_map(sigmap, args.template_cifti, str(sigmap_path), "crossnobis_isub_sigmap")
    log.info(f"Saved sigmap: {sigmap_path.name}")

    log.info("Done.")


if __name__ == "__main__":
    main()
