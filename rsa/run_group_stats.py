"""
rsa/run_group_stats.py
======================
Aggregate per-subject searchlight RSA maps into group-level statistics.

For each vertex across subjects:
  1. Stack per-subject ρ maps → (n_subjects, n_grayords)
  2. Fisher-z transform:  Z = arctanh(ρ)
  3. One-sample t-test (H₀: mean Z = 0)  → one-tailed p for ρ > 0
  4. Mean ρ = tanh(mean Z)
  5. Cohen's d = mean(Z) / std(Z, ddof=1)
  6. Whole-brain FDR correction (Benjamini-Hochberg)
  7. Sigmap = sign(mean_ρ) × −log₁₀(p)  [both uncorrected and FDR]
  8. Cluster masks: significant vertices (p < α) in connected components ≥ min_cluster_size
  9. Cluster border masks: vertices inside mask adjacent to ≥1 non-significant neighbour

Output: multi-map CIFTI dscalar with 8 maps:
  mean_rho | cohens_d | sigmap_uncorr | sigmap_fdr |
  cluster_mask_uncorr | cluster_borders_uncorr | cluster_mask_fdr | cluster_borders_fdr

Also writes summary.json to the same directory.

Usage:
  python run_group_stats.py \\
    --output-dir  /path/to/rsa/sg_psc_gsr \\
    --model       pe-av-small-16-frame \\
    --modality    av \\
    --k           150 \\
    --bin-sec     2.0 \\
    --delay-sec   5.0 \\
    --method      spearman \\
    --fmri-tag    sg_psc_gsr \\
    --template-cifti /path/to/group_average_sg_psc_cortex_59k.dtseries.nii \\
    --left-surface   /path/to/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii \\
    --right-surface  /path/to/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from statsmodels.stats.multitest import fdrcorrection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.shared.cifti_io import (
    get_bm_axis,
    get_cortex_vertex_indices,
    save_cifti_multimap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate per-subject RSA maps to group-level statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",     required=True,
                   help="Root RSA output directory (contains per-subject subdirs).")
    p.add_argument("--model",          required=True)
    p.add_argument("--modality",       required=True, choices=["v", "a", "av"])
    p.add_argument("--k",              type=int, required=True)
    p.add_argument("--bin-sec",        type=float, required=True)
    p.add_argument("--delay-sec",      type=float, default=5.0)
    p.add_argument("--method",         required=True, choices=["spearman", "pearson"])
    p.add_argument("--fmri-tag",       required=True,
                   help="Preprocessing tag in per-subject CIFTI filenames "
                        "(e.g. sg_psc_gsr).")
    p.add_argument("--template-cifti", required=True,
                   help="59k CIFTI whose BrainModelAxis defines grayordinate space.")
    p.add_argument("--left-surface",   required=True,
                   help="Left 59k midthickness .surf.gii (for cluster adjacency).")
    p.add_argument("--right-surface",  required=True,
                   help="Right 59k midthickness .surf.gii (for cluster adjacency).")
    p.add_argument("--alpha",          type=float, default=0.05,
                   help="Significance threshold for cluster detection.")
    p.add_argument("--min-cluster-size", type=int, default=10,
                   help="Minimum cluster size (vertices) to include in masks.")
    return p.parse_args()


# =============================================================================
# Surface adjacency
# =============================================================================

def _load_faces(surf_path: str) -> np.ndarray:
    """Return (n_faces, 3) int32 face array from a GIFTI surface file."""
    surf = nib.load(surf_path)
    return surf.darrays[1].data.astype(np.int32)


def _surface_adjacency(faces: np.ndarray,
                        vertex_idx: np.ndarray,
                        n_cifti_verts: int) -> csr_matrix:
    """Sparse adjacency matrix in CIFTI grayordinate space.

    Re-indexes the full-surface face array to CIFTI vertex space, keeping
    only faces whose three vertices are all grayordinates.

    Args:
        faces:         (n_faces, 3) full-surface face array
        vertex_idx:    (n_cifti_verts,) vertex indices of CIFTI grayordinates
        n_cifti_verts: number of CIFTI grayordinates for this hemisphere

    Returns:
        (n_cifti_verts, n_cifti_verts) uint8 sparse adjacency matrix
    """
    idx_map = np.full(int(faces.max()) + 1, -1, dtype=np.int32)
    idx_map[vertex_idx] = np.arange(n_cifti_verts, dtype=np.int32)

    f0 = idx_map[faces[:, 0]]
    f1 = idx_map[faces[:, 1]]
    f2 = idx_map[faces[:, 2]]
    valid = (f0 >= 0) & (f1 >= 0) & (f2 >= 0)
    fc = np.column_stack([f0[valid], f1[valid], f2[valid]])

    i = np.concatenate([fc[:, 0], fc[:, 1], fc[:, 2],
                        fc[:, 1], fc[:, 2], fc[:, 0]])
    j = np.concatenate([fc[:, 1], fc[:, 2], fc[:, 0],
                        fc[:, 0], fc[:, 1], fc[:, 2]])
    return csr_matrix(
        (np.ones(len(i), dtype=np.uint8), (i, j)),
        shape=(n_cifti_verts, n_cifti_verts),
    )


# =============================================================================
# Cluster analysis
# =============================================================================

def _cluster_mask_and_borders(sig_mask: np.ndarray,
                               adj: csr_matrix,
                               min_cluster_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute cluster membership mask and border mask.

    Args:
        sig_mask:         (n_cifti_verts,) bool — significant vertices
        adj:              sparse adjacency in CIFTI vertex space
        min_cluster_size: minimum vertices per cluster to retain

    Returns:
        cluster_mask:  (n_cifti_verts,) float32 — 1 inside valid clusters, 0 outside
        border_mask:   (n_cifti_verts,) float32 — 1 at cluster borders, 0 elsewhere
    """
    cluster_mask = np.zeros(len(sig_mask), dtype=np.float32)
    border_mask  = np.zeros(len(sig_mask), dtype=np.float32)

    if not sig_mask.any():
        return cluster_mask, border_mask

    sig_idx = np.where(sig_mask)[0]
    sig_adj = adj[sig_idx][:, sig_idx]
    n_comp, labels = connected_components(sig_adj, directed=False)

    for comp in range(n_comp):
        comp_local  = np.where(labels == comp)[0]
        comp_global = sig_idx[comp_local]
        if len(comp_global) < min_cluster_size:
            continue

        cluster_mask[comp_global] = 1.0

        # Border: within-cluster vertices that have a non-cluster neighbour
        for v in comp_global:
            nbrs = adj[v].indices
            if np.any(~sig_mask[nbrs]):
                border_mask[v] = 1.0

    return cluster_mask, border_mask


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    bin_sec_int  = int(args.bin_sec)
    delay_tag    = f"delay{int(args.delay_sec)}s"
    config       = f"k{args.k}_{delay_tag}_bin{bin_sec_int}s_{args.method}"

    out_dir = (Path(args.output_dir) / "group_stats" /
               args.model / config)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect per-subject ρ maps ────────────────────────────────────────────
    # Expected path: {output_dir}/{subject}/{model}/{config}/rsa_59k_{fmri_tag}_k{k}_{delay_tag}_bin{bin}_{method}_maps.dscalar.nii
    fname_pattern = (f"rsa_59k_{args.fmri_tag}_k{args.k}_{delay_tag}"
                     f"_bin{bin_sec_int}_{args.method}_maps.dscalar.nii")

    subject_rho_files = sorted([
        f for f in Path(args.output_dir).glob(
            f"*/{args.model}/{config}/{fname_pattern}"
        )
        if f.parent.parent.parent.name != "group_stats"
    ])

    if not subject_rho_files:
        log.error(
            f"No per-subject ρ maps found matching:\n"
            f"  {Path(args.output_dir)}/*/{args.model}/{config}/{fname_pattern}\n"
            f"Run per-subject RSA first (run_analysis.sh persubject)."
        )
        sys.exit(1)

    log.info(f"Found {len(subject_rho_files)} per-subject ρ maps")

    rho_maps = []
    for f in subject_rho_files:
        data = nib.load(str(f)).get_fdata(dtype=np.float32).squeeze()
        rho_maps.append(data)
        log.info(f"  Loaded: {f.parent.parent.parent.name}  "
                 f"shape={data.shape}  max={data.max():.4f}")

    rho_stack = np.stack(rho_maps, axis=0)   # (n_subjects, n_grayords)
    n_subs, n_grays = rho_stack.shape
    log.info(f"Stacked: {rho_stack.shape}")

    # ── Fisher-z transform ────────────────────────────────────────────────────
    Z = np.arctanh(np.clip(rho_stack, -1 + 1e-7, 1 - 1e-7)).astype(np.float64)

    # ── One-sample t-test (H₀: mean Z = 0) ───────────────────────────────────
    t_vals, p_two = stats.ttest_1samp(Z, popmean=0.0, axis=0)
    t_vals = t_vals.astype(np.float32)

    # One-tailed p for ρ > 0: when t > 0 use p/2, else 1 - p/2
    p_uncorr = np.where(t_vals > 0,
                        (p_two / 2.0),
                        (1.0 - p_two / 2.0)).astype(np.float32)

    # ── Summary statistics ────────────────────────────────────────────────────
    mean_z   = Z.mean(axis=0).astype(np.float32)
    std_z    = Z.std(axis=0, ddof=1).astype(np.float32)
    std_z[std_z == 0] = 1e-10

    mean_rho = np.tanh(mean_z).astype(np.float32)
    cohens_d = (mean_z / std_z).astype(np.float32)

    log.info(f"mean_rho range: [{mean_rho.min():.4f}, {mean_rho.max():.4f}]")
    log.info(f"Cohen's d range: [{cohens_d.min():.4f}, {cohens_d.max():.4f}]")

    # ── FDR correction ────────────────────────────────────────────────────────
    _, p_fdr = fdrcorrection(p_uncorr)
    p_fdr = p_fdr.astype(np.float32)
    log.info(f"Uncorrected p<{args.alpha}: {(p_uncorr < args.alpha).sum():,} / {n_grays:,}")
    log.info(f"FDR q<{args.alpha}:        {(p_fdr    < args.alpha).sum():,} / {n_grays:,}")

    # ── Sigmaps: sign(mean_ρ) × −log₁₀(p) ───────────────────────────────────
    # Convention: positive values = significant positive correlation
    #             negative values = significant negative correlation
    # Clip p from below to avoid log(0) = -inf
    eps = np.finfo(np.float32).tiny
    sigmap_uncorr = (np.sign(mean_rho) *
                     (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)
    sigmap_fdr    = (np.sign(mean_rho) *
                     (-np.log10(np.maximum(p_fdr,    eps)))).astype(np.float32)

    # ── Surface adjacency for cluster analysis ────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    log.info(f"Building surface adjacency (LH: {n_left} verts, RH: {n_grays - n_left} verts) ...")
    faces_lh = _load_faces(args.left_surface)
    faces_rh = _load_faces(args.right_surface)
    adj_lh   = _surface_adjacency(faces_lh, left_indices,  n_left)
    adj_rh   = _surface_adjacency(faces_rh, right_indices, n_grays - n_left)
    log.info("  Adjacency built.")

    # ── Cluster masks (uncorrected) ───────────────────────────────────────────
    sig_lh_uncorr = p_uncorr[:n_left]          < args.alpha
    sig_rh_uncorr = p_uncorr[n_left:]          < args.alpha

    cm_lh_unc, cb_lh_unc = _cluster_mask_and_borders(sig_lh_uncorr, adj_lh, args.min_cluster_size)
    cm_rh_unc, cb_rh_unc = _cluster_mask_and_borders(sig_rh_uncorr, adj_rh, args.min_cluster_size)

    cluster_mask_uncorr   = np.concatenate([cm_lh_unc, cm_rh_unc])
    cluster_borders_uncorr = np.concatenate([cb_lh_unc, cb_rh_unc])

    # ── Cluster masks (FDR) ───────────────────────────────────────────────────
    sig_lh_fdr = p_fdr[:n_left]  < args.alpha
    sig_rh_fdr = p_fdr[n_left:]  < args.alpha

    cm_lh_fdr, cb_lh_fdr = _cluster_mask_and_borders(sig_lh_fdr, adj_lh, args.min_cluster_size)
    cm_rh_fdr, cb_rh_fdr = _cluster_mask_and_borders(sig_rh_fdr, adj_rh, args.min_cluster_size)

    cluster_mask_fdr   = np.concatenate([cm_lh_fdr, cm_rh_fdr])
    cluster_borders_fdr = np.concatenate([cb_lh_fdr, cb_rh_fdr])

    log.info(f"Uncorr clusters (≥{args.min_cluster_size} verts): "
             f"{int(cluster_mask_uncorr.sum()):,} verts  "
             f"borders: {int(cluster_borders_uncorr.sum()):,}")
    log.info(f"FDR clusters   (≥{args.min_cluster_size} verts): "
             f"{int(cluster_mask_fdr.sum()):,} verts  "
             f"borders: {int(cluster_borders_fdr.sum()):,}")

    # ── Save multi-map CIFTI ──────────────────────────────────────────────────
    out_path = out_dir / f"group_stats_{n_subs}subs.dscalar.nii"
    maps = np.stack([
        mean_rho,
        cohens_d,
        sigmap_uncorr,
        sigmap_fdr,
        cluster_mask_uncorr,
        cluster_borders_uncorr,
        cluster_mask_fdr,
        cluster_borders_fdr,
    ], axis=0)   # (8, n_grayords)

    map_names = [
        "mean_rho",
        "cohens_d",
        "sigmap_uncorr",
        "sigmap_fdr",
        "cluster_mask_uncorr",
        "cluster_borders_uncorr",
        "cluster_mask_fdr",
        "cluster_borders_fdr",
    ]

    save_cifti_multimap(maps, map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "model":              args.model,
        "modality":           args.modality,
        "config":             config,
        "fmri_tag":           args.fmri_tag,
        "n_subjects":         n_subs,
        "n_grayordinates":    n_grays,
        "alpha":              args.alpha,
        "min_cluster_size":   args.min_cluster_size,
        "n_sig_uncorr":       int((p_uncorr < args.alpha).sum()),
        "n_sig_fdr":          int((p_fdr    < args.alpha).sum()),
        "n_cluster_verts_uncorr": int(cluster_mask_uncorr.sum()),
        "n_cluster_verts_fdr":    int(cluster_mask_fdr.sum()),
        "max_cohens_d":       float(cohens_d.max()),
        "max_sigmap_uncorr":  float(sigmap_uncorr.max()),
        "max_sigmap_fdr":     float(sigmap_fdr.max()),
        "mean_rho_range":     [float(mean_rho.min()), float(mean_rho.max())],
        "subjects":           [f.parent.parent.parent.name for f in subject_rho_files],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: {summary_path}")
    log.info(
        f"Done.  n={n_subs} subjects  "
        f"uncorr={summary['n_cluster_verts_uncorr']:,}  "
        f"fdr={summary['n_cluster_verts_fdr']:,} cluster verts"
    )


if __name__ == "__main__":
    main()
