"""
rsa/run_searchlight.py
===================================
Vertex-wise searchlight RSA on HCP 7T movie-watching fMRI data.

For each vertex, a geodesic neighbourhood of k nearest vertices is assembled.
The fMRI RDM within that neighbourhood is correlated (Spearman or Pearson)
with the model RDM. The resulting correlation map is saved as a CIFTI dscalar.

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Pre-filtered, pre-z-scored CIFTI produced by preprocess_individual.py
  with --timing-csv (filtered mode). The file is expected at:
    {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii

STREAMING MODE (--raw-dir):
  Raw 7T CIFTI dtseries files. The subject is preprocessed on-the-fly
  (movie TPs with hemodynamic delay, GSR, z-score); no CIFTI is saved.
  The geodesic distance cache is shared across subjects (written once).

Both modes are parallel-safe: each call processes one (subject, model,
modality) tuple. GNU parallel in run_analysis.sh spawns N such processes
simultaneously.

Geodesic distance matrices are computed once with wb_command and cached as
.dconn.nii files. Re-runs skip this step automatically.

Usage (disk mode):
  python run_searchlight.py \\
      --preprocessed-dir <path> --fmri-suffix gsr_zscore_delay5s \\
      --subject group_average --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path> \\
      --left-surface <path> --right-surface <path> \\
      --workbench <path> --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --k 100 --bin-sec 2.0 --delay-sec 5.0 \\
      [--normalize] [--blockdiag] [--hrf] \\
      --method spearman

Usage (streaming mode):
  python run_searchlight.py \\
      --raw-dir <path> --subject 100610 --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path> \\
      --left-surface <path> --right-surface <path> \\
      --workbench <path> --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --k 100 --bin-sec 2.0 --delay-sec 5.0 --tr 1.0 \\
      [--sg-filter] [--psc] [--no-gsr] [--no-z-score] \\
      --method spearman
"""

import argparse
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.shared.rsa_utils import (
    load_fmri_cifti, preprocess_fmri,
    process_model_embeddings, compute_rdm, correlate_rdms,
)
from rsa.shared.cifti_io import (
    get_bm_axis, save_cifti_map, get_cortex_vertex_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

N_JOBS = -1


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Vertex-wise searchlight RSA on movie fMRI (59k grayordinate space).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — mutually exclusive modes
    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk mode] Directory of pre-filtered, pre-z-scored CIFTIs "
                          "(output of preprocess_individual.py --timing-csv). "
                          "File: {dir}/{subject}_{fmri-suffix}_cortex_59k.dtseries.nii")
    inp.add_argument("--fmri-suffix", default="gsr_zscore_delay5s", dest="fmri_suffix",
                     help="[disk mode] Filename suffix that encodes preprocessing "
                          "(e.g. 'gsr_zscore_delay5s').")
    inp.add_argument("--raw-dir", default=None,
                     help="[streaming mode] Root directory of raw 7T CIFTI dtseries files. "
                          "The subject is preprocessed on-the-fly; no CIFTI is saved.")

    p.add_argument("--timing-csv", required=True,
                   help="Path to movie_timing.csv.")
    p.add_argument("--embeddings-dir", required=True,
                   help="Base directory for model embeddings.")
    p.add_argument("--template-cifti", required=True,
                   help="Template 59k dscalar.nii for CIFTI output header.")

    # Surfaces (for geodesic distances) — must be 59k_fs_LR
    p.add_argument("--left-surface", required=True,
                   help="Left 59k midthickness .surf.gii.")
    p.add_argument("--right-surface", required=True,
                   help="Right 59k midthickness .surf.gii.")
    p.add_argument("--workbench", required=True,
                   help="Path to wb_command binary.")

    p.add_argument("--output-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID (used for output subdirectory; also selects "
                        "which subject to preprocess in streaming mode).")

    p.add_argument("--model", required=True,
                   help="Model name (must match subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True, choices=["v", "a", "av"],
                   help="Embedding modality: v=video, a=audio, av=joint.")
    p.add_argument("--k", type=int, required=True,
                   help="Searchlight neighbourhood size (k nearest vertices).")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Hemodynamic delay in seconds. In disk mode: config label only. "
                        "In streaming mode: applied when filtering movie TPs.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF. Use only when fMRI was "
                        "preprocessed with --delay-sec 0.")
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score normalization of embeddings.")
    p.add_argument("--blockdiag", action="store_true",
                   help="Block-diagonal normalization (per-segment instead of global).")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"],
                   help="RDM correlation method.")
    p.add_argument("--tr", type=float, required=True,
                   help="TR in seconds.")

    # Streaming preprocessing flags (ignored in disk mode)
    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true",
                      help="Savitzky-Golay high-pass filter.")
    prep.add_argument("--psc", default=False, action="store_true",
                      help="Percent signal change normalization.")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction,
                      help="Global signal regression.")
    prep.add_argument("--z-score", default=True, action=argparse.BooleanOptionalAction,
                      dest="z_score", help="Z-score per vertex.")

    return p.parse_args()


# =============================================================================
# Geodesic distance
# =============================================================================

def compute_geodesic_distance(surface_path: str, workbench: str,
                               cache_path: str) -> str:
    """Compute all-to-all geodesic distances for a surface and cache the result.

    Args:
        surface_path: str — path to .surf.gii (59k_fs_LR midthickness)
        workbench: str — path to wb_command binary
        cache_path: str — path for the output .dconn.nii cache file

    Returns:
        str — path to the .dconn.nii distance file
    """
    if os.path.exists(cache_path):
        log.info(f"  Geodesic distance cached: {os.path.basename(cache_path)}")
        return cache_path

    log.info(f"  Computing geodesic distances for {os.path.basename(surface_path)} ...")
    cmd = [
        workbench,
        "-surface-geodesic-distance-all-to-all",
        surface_path,
        cache_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"wb_command failed (exit {result.returncode}):\n{result.stderr}"
        )
    log.info(f"  Geodesic distances saved: {os.path.basename(cache_path)}")
    return cache_path


def get_k_nearest_neighbors(dconn_path: str, k: int) -> np.ndarray:
    """Extract the k nearest neighbours for each vertex from a .dconn.nii file.

    Args:
        dconn_path: str — path to dense connectivity CIFTI (.dconn.nii)
        k: int — neighbourhood size

    Returns:
        (n_vertices, k) int32 — vertex indices of k nearest neighbours per vertex
    """
    log.info(f"  Loading geodesic distances from {os.path.basename(dconn_path)} ...")
    img = nib.load(dconn_path)
    dist_matrix = img.get_fdata(dtype=np.float32)  # (n_verts, n_verts)
    n_verts = dist_matrix.shape[0]
    log.info(f"  Distance matrix: {n_verts} × {n_verts}")

    np.fill_diagonal(dist_matrix, np.inf)
    neighbors = np.argsort(dist_matrix, axis=1)[:, :k].astype(np.int32)
    return neighbors


# =============================================================================
# Searchlight RSA
# =============================================================================

def _searchlight_vertex(v_idx: int, fmri: np.ndarray,
                         model_rdm: np.ndarray, neighbors: np.ndarray,
                         method: str) -> tuple[float, float]:
    """Compute RSA for a single vertex searchlight neighbourhood."""
    hood = fmri[:, neighbors[v_idx]]  # (n_bins, k)
    fmri_rdm = compute_rdm(hood, method="correlation")
    return correlate_rdms(fmri_rdm, model_rdm, method=method)


def run_searchlight(fmri: np.ndarray, model_rdm: np.ndarray,
                    neighbors: np.ndarray, method: str = "spearman",
                    n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Run searchlight RSA across all vertices in parallel.

    Args:
        fmri: (n_bins, n_vertices) float32
        model_rdm: (n_bins, n_bins) float64
        neighbors: (n_vertices, k) int32
        method: str
        n_jobs: int

    Returns:
        corr_map: (n_vertices,) float32
        pval_map: (n_vertices,) float32
    """
    n_verts = fmri.shape[1]
    log.info(f"  Running searchlight RSA on {n_verts} vertices ...")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_searchlight_vertex)(v, fmri, model_rdm, neighbors, method)
        for v in range(n_verts)
    )
    corr_map = np.array([r for r, _ in results], dtype=np.float32)
    pval_map = np.array([p for _, p in results], dtype=np.float32)
    return corr_map, pval_map


# =============================================================================
# Output naming
# =============================================================================

def _cifti_path(args) -> str:
    """Construct the per-subject preprocessed CIFTI path from directory + suffix."""
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")


def _config_label(args) -> str:
    parts = [
        f"k{args.k}",
        "norm" if args.normalize else "nonorm",
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        f"bin{args.bin_sec:.0f}s",
        args.method,
    ]
    if args.blockdiag:
        parts.append("blockdiag")
    return "_".join(parts)


# =============================================================================
# Core analysis (shared between disk and streaming modes)
# =============================================================================

def _run_analysis(args, fmri_data: np.ndarray, timing_df: pd.DataFrame,
                  config: str, out_root: Path):
    """Run searchlight RSA on pre-loaded (n_vertices, T_included) fMRI data."""
    corr_out = out_root / f"rsa_{args.method}.dscalar.nii"
    pval_out = out_root / "rsa_pval_fdr.dscalar.nii"
    if corr_out.exists() and pval_out.exists():
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    fmri_binned = preprocess_fmri(fmri_data, timing_df, args.bin_sec, args.tr)
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"{int(args.bin_sec)}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, hrf=args.hrf,
        normalize=args.normalize, blockdiag=args.blockdiag,
        tr=args.tr,
    )

    n_bins      = min(fmri_binned.shape[0], emb.shape[0])
    fmri_binned = fmri_binned[:n_bins]
    emb         = emb[:n_bins]

    model_rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    log.info(f"  Model RDM: {model_rdm.shape}")

    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    cache_dir = Path(args.output_dir) / "_geodesic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    surfaces = {
        "left":  {"surf": args.left_surface,  "fmri": fmri_binned[:, :n_left]},
        "right": {"surf": args.right_surface, "fmri": fmri_binned[:, n_left:]},
    }

    n_total   = fmri_binned.shape[1]
    corr_full = np.zeros(n_total, dtype=np.float32)
    pval_raw  = np.ones(n_total,  dtype=np.float32)

    for hem, info in surfaces.items():
        cache_path = str(cache_dir / f"{hem}_geodesic_k{args.k}.dconn.nii")
        dconn      = compute_geodesic_distance(info["surf"], args.workbench, cache_path)
        neighbors  = get_k_nearest_neighbors(dconn, args.k)

        corr_hem, pval_hem = run_searchlight(
            info["fmri"], model_rdm, neighbors, method=args.method, n_jobs=N_JOBS
        )

        n_hem = corr_hem.shape[0]
        if hem == "left":
            corr_full[:n_hem]                 = corr_hem
            pval_raw[:n_hem]                  = pval_hem.astype(np.float32)
        else:
            corr_full[n_left: n_left + n_hem] = corr_hem
            pval_raw[n_left: n_left + n_hem]  = pval_hem.astype(np.float32)

    _, pval_fdr = fdrcorrection(pval_raw)
    pval_fdr    = pval_fdr.astype(np.float32)

    out_root.mkdir(parents=True, exist_ok=True)
    save_cifti_map(corr_full, args.template_cifti, str(corr_out),
                   map_name=f"rsa_{args.method}")
    save_cifti_map(pval_fdr, args.template_cifti, str(pval_out),
                   map_name="rsa_pval_fdr")
    log.info(f"  Saved: {corr_out.name}  max_r={corr_full.max():.4f}")
    log.info(f"  Saved: {pval_out.name}")

    n_sig = (pval_fdr < 0.05).sum()
    log.info(f"  Significant vertices (whole-brain FDR q<0.05): {n_sig} / {n_total}")
    stats_path = out_root / "stats_summary.txt"
    stats_path.write_text(
        f"model={args.model}\nmodality={args.modality}\nconfig={config}\n"
        f"n_bins={n_bins}\nn_vertices={n_total}\n"
        f"n_sig_fdr05={n_sig}\n"
        f"max_r={corr_full.max():.4f}\nmean_r={corr_full.mean():.4f}\n"
    )
    log.info(f"  Stats saved: {stats_path}")


# =============================================================================
# Disk mode
# =============================================================================

def _run_disk(args):
    timing_df = pd.read_csv(args.timing_csv)
    config    = _config_label(args)
    out_root  = Path(args.output_dir) / args.subject / args.model / config

    cifti = _cifti_path(args)
    log.info(f"Searchlight RSA: {args.model} / {args.modality} / {config}")
    log.info(f"  fMRI: {cifti}")

    log.info("  Loading fMRI ...")
    fmri_data = load_fmri_cifti(cifti)   # (n_vertices, T_included)
    log.info(f"  fMRI loaded: {fmri_data.shape}")

    _run_analysis(args, fmri_data, timing_df, config, out_root)
    log.info("Done.")


# =============================================================================
# Streaming mode
# =============================================================================

def _run_streaming(args):
    from preprocess_individual import preprocess_subject_filtered

    timing_df = pd.read_csv(args.timing_csv)
    config    = _config_label(args)
    sub       = args.subject
    raw_dir   = Path(args.raw_dir)
    out_root  = Path(args.output_dir) / sub / args.model / config

    corr_out = out_root / f"rsa_{args.method}.dscalar.nii"
    pval_out = out_root / "rsa_pval_fdr.dscalar.nii"
    if corr_out.exists() and pval_out.exists():
        log.info(f"[{sub}] Outputs already exist — skipping")
        return

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc,
        gsr=args.gsr, z_score=args.z_score,
    )

    log.info(f"[{sub}] Preprocessing raw CIFTI (delay={args.delay_sec}s) ...")
    data, _bm_axis, _run_trs = preprocess_subject_filtered(
        sub, raw_dir, timing_df, args.delay_sec, args.tr, prep_args
    )
    log.info(f"[{sub}] Preprocessed: {data.shape}")

    _run_analysis(args, data, timing_df, config, out_root)
    del data
    log.info(f"[{sub}] Done.")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.preprocessed_dir and args.raw_dir:
        log.error("--preprocessed-dir and --raw-dir are mutually exclusive.")
        sys.exit(1)
    if not args.preprocessed_dir and not args.raw_dir:
        log.error("Either --preprocessed-dir (disk mode) or --raw-dir (streaming mode) is required.")
        sys.exit(1)

    if args.raw_dir:
        _run_streaming(args)
    else:
        _run_disk(args)


if __name__ == "__main__":
    main()
