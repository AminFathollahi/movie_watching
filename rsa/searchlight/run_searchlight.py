"""
rsa/searchlight/run_searchlight.py
===================================
Vertex-wise searchlight RSA on HCP 7T movie-watching fMRI data.

For each vertex, a geodesic neighbourhood of k nearest vertices is assembled.
The fMRI RDM within that neighbourhood is correlated (Spearman or Pearson)
with the model RDM. The resulting correlation map is saved as a CIFTI dscalar.

Geodesic distance matrices are computed once with wb_command and cached as
.dconn.nii files. Re-runs skip this step automatically.

Usage:
  python run_searchlight.py \\
      --fmri-cifti <path> --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path> \\
      --left-surface <path> --right-surface <path> \\
      --workbench <path> --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --k 100 --bin-sec 2.0 --delay-sec 5.0 \\
      [--normalize] [--blockdiag] [--hrf] \\
      --method spearman [--subject avg]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from statsmodels.stats.multitest import fdrcorrection

# Shared RSA utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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

N_JOBS = -1  # use all available cores for parallelism


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Vertex-wise searchlight RSA on movie fMRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--fmri-cifti", required=True,
                   help="Preprocessed 59k CIFTI dtseries (group-avg or per-subject).")
    p.add_argument("--timing-csv", required=True,
                   help="Path to movie_timing.csv.")
    p.add_argument("--embeddings-dir", required=True,
                   help="Base directory for model embeddings.")
    p.add_argument("--template-cifti", required=True,
                   help="Template dscalar.nii for CIFTI output header.")

    # Surfaces (for geodesic distances)
    p.add_argument("--left-surface", required=True,
                   help="Left 59k midthickness .surf.gii.")
    p.add_argument("--right-surface", required=True,
                   help="Right 59k midthickness .surf.gii.")
    p.add_argument("--workbench", required=True,
                   help="Path to wb_command binary.")

    # Output
    p.add_argument("--output-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID or 'group_average' (used for output subdirectory).")

    # Analysis config
    p.add_argument("--model", required=True,
                   help="Model name (must match subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True, choices=["v", "a", "av"],
                   help="Embedding modality: v=video, a=audio, av=joint.")
    p.add_argument("--k", type=int, required=True,
                   help="Searchlight neighbourhood size (k nearest vertices).")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Hemodynamic delay in seconds (applied to fMRI).")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF instead of boxcar delay.")
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score normalization of embeddings.")
    p.add_argument("--blockdiag", action="store_true",
                   help="Block-diagonal normalization (per-segment instead of global).")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"],
                   help="RDM correlation method.")
    p.add_argument("--tr", type=float, required=True,
                   help="TR in seconds.")

    return p.parse_args()


# =============================================================================
# Geodesic distance
# =============================================================================

def compute_geodesic_distance(surface_path: str, workbench: str,
                               cache_path: str) -> str:
    """Compute all-to-all geodesic distances for a surface and cache the result.

    Uses wb_command -surface-geodesic-distance-all-to-all. Skips computation
    if the cache file already exists.

    Args:
        surface_path: str — path to .surf.gii
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

    # Set diagonal to inf so each vertex does not select itself
    np.fill_diagonal(dist_matrix, np.inf)
    neighbors = np.argsort(dist_matrix, axis=1)[:, :k].astype(np.int32)
    return neighbors


# =============================================================================
# Searchlight RSA
# =============================================================================

def _searchlight_vertex(v_idx: int, fmri: np.ndarray,
                         model_rdm: np.ndarray, neighbors: np.ndarray,
                         method: str) -> tuple[float, float]:
    """Compute RSA for a single vertex searchlight neighbourhood.

    Args:
        v_idx: int — vertex index
        fmri: (n_bins, n_vertices) float32
        model_rdm: (n_bins, n_bins) float64
        neighbors: (k,) int32 — neighbour indices
        method: str — "spearman" or "pearson"

    Returns:
        (r, p): float, float
    """
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
        method: str — "spearman" or "pearson"
        n_jobs: int — joblib parallelism (-1 = all cores)

    Returns:
        corr_map: (n_vertices,) float32 — RSA correlation per vertex
        pval_map: (n_vertices,) float32 — p-value per vertex
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

def _config_label(args) -> str:
    """Build a descriptive label for the current analysis configuration."""
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
# Main
# =============================================================================

def main():
    args = parse_args()

    timing_df = pd.read_csv(args.timing_csv)
    config = _config_label(args)

    out_root = Path(args.output_dir) / args.subject / args.model / config
    out_root.mkdir(parents=True, exist_ok=True)

    # Cache directory for geodesic distance matrices
    cache_dir = Path(args.output_dir) / "_geodesic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Skip if outputs already exist
    corr_out = out_root / f"rsa_{args.method}.dscalar.nii"
    pval_out = out_root / "rsa_pval_fdr.dscalar.nii"
    if corr_out.exists() and pval_out.exists():
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    log.info(f"Searchlight RSA: {args.model} / {args.modality} / {config}")
    log.info(f"  fMRI: {args.fmri_cifti}")

    # Load and preprocess fMRI
    # When hrf=True, embeddings are HRF-convolved; fMRI window is not shifted.
    # When hrf=False, fMRI window is shifted by delay_sec; embeddings are unshifted.
    fmri_delay = 0.0 if args.hrf else args.delay_sec
    log.info("  Loading fMRI ...")
    fmri_raw = load_fmri_cifti(args.fmri_cifti)  # (n_vertices, T_total)
    fmri_binned = preprocess_fmri(fmri_raw, timing_df, args.bin_sec, fmri_delay,
                                   args.tr)  # (n_bins, n_vertices)
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    # Load and process embeddings
    emb_file = (Path(args.embeddings_dir) / args.model /
                f"{int(args.bin_sec)}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, hrf=args.hrf,
        normalize=args.normalize, blockdiag=args.blockdiag,
        tr=args.tr,
    )  # (n_bins, n_features)

    # Safety trim in case of off-by-one from rounding
    n_bins = min(fmri_binned.shape[0], emb.shape[0])
    fmri_binned = fmri_binned[:n_bins]
    emb = emb[:n_bins]

    model_rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    log.info(f"  Model RDM: {model_rdm.shape}")

    # Get hemisphere vertex slices from CIFTI brain model axis
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    surfaces = {
        "left":  {"surf": args.left_surface,  "fmri": fmri_binned[:, :n_left]},
        "right": {"surf": args.right_surface, "fmri": fmri_binned[:, n_left:]},
    }

    n_total = fmri_binned.shape[1]
    corr_full = np.zeros(n_total, dtype=np.float32)
    pval_raw  = np.ones(n_total,  dtype=np.float32)   # uncorrected, filled per hemisphere

    for hem, info in surfaces.items():
        cache_path = str(cache_dir / f"{hem}_geodesic_k{args.k}.dconn.nii")
        dconn = compute_geodesic_distance(info["surf"], args.workbench, cache_path)
        neighbors = get_k_nearest_neighbors(dconn, args.k)

        corr_hem, pval_hem = run_searchlight(
            info["fmri"], model_rdm, neighbors, method=args.method, n_jobs=N_JOBS
        )

        n_hem = corr_hem.shape[0]
        if hem == "left":
            corr_full[:n_hem]      = corr_hem
            pval_raw[:n_hem]       = pval_hem.astype(np.float32)
        else:
            corr_full[n_left: n_left + n_hem] = corr_hem
            pval_raw[n_left: n_left + n_hem]  = pval_hem.astype(np.float32)

    # Whole-brain FDR correction (more conservative than per-hemisphere)
    _, pval_fdr = fdrcorrection(pval_raw)
    pval_fdr = pval_fdr.astype(np.float32)

    # Save full-brain maps (correct shape = n_total = 59412)
    save_cifti_map(corr_full, args.template_cifti, str(corr_out),
                   map_name=f"rsa_{args.method}")
    save_cifti_map(pval_fdr, args.template_cifti, str(pval_out),
                   map_name="rsa_pval_fdr")
    log.info(f"  Saved: {corr_out.name}  max_r={corr_full.max():.4f}")
    log.info(f"  Saved: {pval_out.name}")

    # Summary stats
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
    log.info("Done.")


if __name__ == "__main__":
    main()
