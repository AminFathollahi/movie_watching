"""
rsa/run_group_stats.py
======================
Aggregate per-subject searchlight RSA maps into group-level statistics.

For each vertex across subjects:
  1. Stack per-subject ρ maps → (n_subjects, n_grayords)
  2. Fisher-z transform:  Z = arctanh(ρ)
  3. One-sample t-test (H₀: mean Z = 0) → one-tailed t/p for ρ > 0
  4. Mean ρ = tanh(mean Z)
  5. Cohen's d = mean(Z) / std(Z, ddof=1)
  6. TFCE + sign-flipping permutation test (n_permutations, default 5000)
     → FWE-corrected significance mask at p < α (95th percentile of null)
  7. Cluster border mask: significant vertices adjacent to non-significant neighbours

TFCE is computed via mne.stats.permutation_cluster_1samp_test with
threshold=dict(start=0, step=0.2), using full surface adjacency from the
59k midthickness GIFTI meshes. The null distribution is built from the
maximum TFCE statistic per sign-flip permutation; the FWE threshold is the
(1-α)-th percentile of that distribution.

Output: multi-map CIFTI dscalar with 7 maps:
  mean_rho | cohens_d | t_stat | sigmap_uncorr |
  tfce_stat | tfce_fwe_mask | tfce_fwe_borders

Also writes summary.json to the same directory.

Runtime note: TFCE on ~59k vertices with 5000 permutations typically takes
1–4 hours. Use --n-jobs -1 to parallelise across all available CPU cores.

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
    --right-surface  /path/to/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii \\
    --n-permutations 5000 \\
    --n-jobs -1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from scipy.sparse import block_diag as sp_block_diag
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import (
    get_bm_axis,
    get_cortex_vertex_indices,
    save_cifti_multimap,
    get_combined_map_names,
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
    p.add_argument("--output-dir",       required=True,
                   help="Root RSA output directory (contains per-subject subdirs).")
    p.add_argument("--model",            required=True)
    p.add_argument("--modality",         required=True, choices=["v", "a", "av"])
    p.add_argument("--k",                type=int, required=True)
    p.add_argument("--bin-sec",          type=float, required=True)
    p.add_argument("--delay-sec",        type=float, default=5.0)
    p.add_argument("--method",           required=True, choices=["spearman", "pearson"])
    p.add_argument("--fmri-tag",         required=True,
                   help="Preprocessing tag in per-subject CIFTI filenames "
                        "(e.g. sg_psc_gsr).")
    p.add_argument("--template-cifti",   required=True,
                   help="59k CIFTI whose BrainModelAxis defines grayordinate space.")
    p.add_argument("--left-surface",     required=True,
                   help="Left 59k midthickness .surf.gii (for TFCE adjacency).")
    p.add_argument("--right-surface",    required=True,
                   help="Right 59k midthickness .surf.gii (for TFCE adjacency).")
    p.add_argument("--alpha",            type=float, default=0.05,
                   help="FWE significance threshold (p < alpha).")
    p.add_argument("--min-cluster-size", type=int, default=10,
                   help="Minimum cluster size (vertices) to include in border mask.")
    p.add_argument("--n-permutations",   type=int, default=5000,
                   help="Number of sign-flip permutations for TFCE null distribution.")
    p.add_argument("--n-jobs",           type=int, default=-1,
                   help="Parallel jobs for permutation test (-1 = all cores).")
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
    adj = csr_matrix(
        (np.ones(len(i), dtype=np.uint8), (i, j)),
        shape=(n_cifti_verts, n_cifti_verts),
    )
    # Binarize: triangles sharing an edge produce duplicate (i,j) pairs whose
    # weights sum to 2 after CSR construction. Set all stored values to 1.
    adj.data[:] = 1
    return adj


# =============================================================================
# Cluster borders
# =============================================================================

def _cluster_borders(sig_mask: np.ndarray,
                     adj: csr_matrix,
                     min_cluster_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute cluster membership mask and border mask from a binary significance mask.

    Two-pass algorithm:
      Pass 1 — connected components on sig_mask; retain only those with
               ≥ min_cluster_size vertices → cluster_mask.
      Pass 2 — border_mask via sparse matmul: a cluster vertex is a border
               iff at least one of its surface neighbours is outside cluster_mask.
               Using the fully-built cluster_mask (not sig_mask) ensures that
               the shared edge between two large adjacent clusters is correctly
               treated as interior (not a border).

    Args:
        sig_mask:         (n_cifti_verts,) bool — significant vertices
        adj:              (n_cifti_verts × n_cifti_verts) symmetric CSR adjacency
        min_cluster_size: minimum vertices per cluster to retain

    Returns:
        cluster_mask:  (n_cifti_verts,) float32 — 1 inside valid clusters, 0 outside
        border_mask:   (n_cifti_verts,) float32 — 1 at cluster borders, 0 elsewhere
    """
    cluster_mask = np.zeros(len(sig_mask), dtype=np.float32)

    if not sig_mask.any():
        return cluster_mask, cluster_mask.copy()

    # Pass 1: label connected components among significant vertices only
    sig_idx = np.where(sig_mask)[0]
    sig_adj = adj[sig_idx][:, sig_idx]
    n_comp, labels = connected_components(sig_adj, directed=False)

    for comp in range(n_comp):
        comp_verts = sig_idx[labels == comp]
        if len(comp_verts) >= min_cluster_size:
            cluster_mask[comp_verts] = 1.0

    # Pass 2: border via sparse matmul.
    # For each vertex, count how many neighbours are outside cluster_mask.
    # A cluster vertex with any non-cluster neighbour → border.
    not_cluster = (cluster_mask == 0).astype(np.float32)
    nbr_noncluster = np.asarray(adj.dot(not_cluster)).ravel()
    border_mask = ((cluster_mask > 0) & (nbr_noncluster > 0)).astype(np.float32)

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
    fname_pattern = (f"rsa_59k_{args.fmri_tag}_k{args.k}_{delay_tag}"
                     f"_bin{bin_sec_int}_{args.method}_maps.dscalar.nii")

    subject_rho_files = sorted([
        f for f in Path(args.output_dir).glob(
            f"*/{args.model}/{fname_pattern}"
        )
        if f.parent.parent.name not in ("group_stats", "group_average")
    ])

    if not subject_rho_files:
        log.error(
            f"No per-subject ρ maps found matching:\n"
            f"  {Path(args.output_dir)}/*/{args.model}/{fname_pattern}\n"
            f"Run per-subject RSA first (run_analysis.sh persubject)."
        )
        sys.exit(1)

    log.info(f"Found {len(subject_rho_files)} per-subject ρ maps")

    sl_map_name = f"searchlight_{args.method}_rho"
    rho_maps = []
    for f in subject_rho_files:
        img = nib.load(str(f))
        map_names = get_combined_map_names(str(f))
        if sl_map_name in map_names:
            idx = map_names.index(sl_map_name)
            data = img.get_fdata(dtype=np.float32)[idx]
        else:
            data = img.get_fdata(dtype=np.float32).squeeze()
        rho_maps.append(data)
        log.info(f"  Loaded [{sl_map_name}]: {f.parent.parent.name}  "
                 f"shape={data.shape}  max={data.max():.4f}")

    rho_stack = np.stack(rho_maps, axis=0)   # (n_subjects, n_grayords)
    n_subs, n_grays = rho_stack.shape
    log.info(f"Stacked: {rho_stack.shape}")

    # ── Fisher-z transform ────────────────────────────────────────────────────
    Z = np.arctanh(np.clip(rho_stack, -1 + 1e-7, 1 - 1e-7)).astype(np.float64)

    # ── One-sample t-test (H₀: mean Z = 0) ───────────────────────────────────
    t_vals, p_two = stats.ttest_1samp(Z, popmean=0.0, axis=0)
    t_vals = t_vals.astype(np.float32)

    # One-tailed p for ρ > 0
    p_uncorr = np.where(t_vals > 0,
                        p_two / 2.0,
                        1.0 - p_two / 2.0).astype(np.float32)

    # ── Summary statistics ────────────────────────────────────────────────────
    mean_z   = Z.mean(axis=0).astype(np.float32)
    std_z    = Z.std(axis=0, ddof=1).astype(np.float32)
    std_z[std_z == 0] = 1e-10

    mean_rho = np.tanh(mean_z).astype(np.float32)
    cohens_d = (mean_z / std_z).astype(np.float32)

    # sign(mean_ρ) × −log₁₀(p_uncorr); positive = significant positive ρ
    eps = np.finfo(np.float32).tiny
    sigmap_uncorr = (np.sign(mean_rho) *
                     (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)

    log.info(f"mean_rho range: [{mean_rho.min():.4f}, {mean_rho.max():.4f}]")
    log.info(f"Cohen's d range: [{cohens_d.min():.4f}, {cohens_d.max():.4f}]")
    log.info(f"Uncorrected p<{args.alpha}: {(p_uncorr < args.alpha).sum():,} / {n_grays:,}")

    # ── Surface adjacency for TFCE ────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    log.info(f"Building surface adjacency (LH: {n_left} verts, "
             f"RH: {n_grays - n_left} verts) ...")
    faces_lh = _load_faces(args.left_surface)
    faces_rh = _load_faces(args.right_surface)
    adj_lh   = _surface_adjacency(faces_lh, left_indices,  n_left)
    adj_rh   = _surface_adjacency(faces_rh, right_indices, n_grays - n_left)
    log.info("  Adjacency built.")

    # Block-diagonal adjacency spanning both hemispheres
    adj_combined = sp_block_diag([adj_lh, adj_rh], format="coo")

    # ── TFCE permutation test ─────────────────────────────────────────────────
    # Sign-flipping permutations; TFCE via threshold=dict(start,step).
    # tail=1: one-tailed test for ρ > 0; H0 = max TFCE per permutation.
    log.info(
        f"Running TFCE permutation test: n_permutations={args.n_permutations}, "
        f"n_jobs={args.n_jobs}  (runtime: 1–4 h for 59k verts)"
    )
    tfce_stat, _, _, h0 = permutation_cluster_1samp_test(
        Z,
        threshold=dict(start=0, step=0.2),
        n_permutations=args.n_permutations,
        adjacency=adj_combined,
        tail=1,
        n_jobs=args.n_jobs,
        seed=42,
        out_type="indices",
        verbose=False,
    )
    # Keep float64 for the threshold comparison — float32 quantization can
    # flip borderline vertices across the threshold (demonstrated: a vertex
    # 1e-10 above the threshold in float64 rounds below it in float32).
    tfce_stat_f64 = np.asarray(tfce_stat, dtype=np.float64)
    h0            = np.asarray(h0,        dtype=np.float64)

    # FWE threshold: 95th percentile of the null max-TFCE distribution.
    # h0 is empty when no vertex had t > 0 (i.e. no positive ρ anywhere).
    if len(h0) == 0:
        log.warning("TFCE null distribution is empty — no positive t-values found. "
                    "Setting threshold to inf; no vertices will be significant.")
        tfce_thresh = np.inf
    else:
        tfce_thresh = float(np.percentile(h0, 100.0 * (1.0 - args.alpha)))
    log.info(f"TFCE FWE threshold (p<{args.alpha}): {tfce_thresh:.4f}")

    # Significance mask computed in float64, then convert stat to float32 for saving
    n_sig_tfce = int((tfce_stat_f64 >= tfce_thresh).sum())
    log.info(f"TFCE-FWE significant vertices: {n_sig_tfce:,} / {n_grays:,}")

    # ── Cluster mask & borders from TFCE FWE significance ────────────────────
    sig_lh_tfce = tfce_stat_f64[:n_left] >= tfce_thresh
    sig_rh_tfce = tfce_stat_f64[n_left:] >= tfce_thresh

    cm_lh, cb_lh = _cluster_borders(sig_lh_tfce, adj_lh, args.min_cluster_size)
    cm_rh, cb_rh = _cluster_borders(sig_rh_tfce, adj_rh, args.min_cluster_size)

    tfce_fwe_mask    = np.concatenate([cm_lh, cm_rh])
    tfce_fwe_borders = np.concatenate([cb_lh, cb_rh])

    # Float32 for storage (CIFTI maps)
    tfce_stat = tfce_stat_f64.astype(np.float32)

    log.info(f"TFCE-FWE clusters (≥{args.min_cluster_size} verts): "
             f"{int(tfce_fwe_mask.sum()):,} verts  "
             f"borders: {int(tfce_fwe_borders.sum()):,}")

    # ── Save multi-map CIFTI ──────────────────────────────────────────────────
    # Continuous maps (mean_rho, cohens_d, t_stat, sigmap_uncorr, tfce_stat)
    # provide the surface overlay; tfce_fwe_mask and tfce_fwe_borders draw
    # the FWE-corrected cluster contours on top.
    out_path = out_dir / f"group_stats_{n_subs}subs.dscalar.nii"
    maps = np.stack([
        mean_rho,
        cohens_d,
        t_vals,
        sigmap_uncorr,
        tfce_stat,
        tfce_fwe_mask,
        tfce_fwe_borders,
    ], axis=0)   # (7, n_grayords)

    map_names = [
        "mean_rho",
        "cohens_d",
        "t_stat",
        "sigmap_uncorr",
        "tfce_stat",
        "tfce_fwe_mask",
        "tfce_fwe_borders",
    ]

    save_cifti_multimap(maps, map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "model":                  args.model,
        "modality":               args.modality,
        "config":                 config,
        "fmri_tag":               args.fmri_tag,
        "n_subjects":             n_subs,
        "n_grayordinates":        n_grays,
        "alpha":                  args.alpha,
        "min_cluster_size":       args.min_cluster_size,
        "n_permutations":         args.n_permutations,
        "tfce_step":              0.2,
        "tfce_fwe_threshold":     tfce_thresh,
        "n_sig_uncorr":           int((p_uncorr < args.alpha).sum()),
        "n_sig_tfce_fwe":         n_sig_tfce,
        "n_cluster_verts_tfce":   int(tfce_fwe_mask.sum()),
        "n_border_verts_tfce":    int(tfce_fwe_borders.sum()),
        "max_cohens_d":           float(cohens_d.max()),
        "max_t_stat":             float(t_vals.max()),
        "max_tfce_stat":          float(tfce_stat.max()),
        "max_sigmap_uncorr":      float(sigmap_uncorr.max()),
        "mean_rho_range":         [float(mean_rho.min()), float(mean_rho.max())],
        "subjects":               [f.parent.parent.name for f in subject_rho_files],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: {summary_path}")
    log.info(
        f"Done.  n={n_subs} subjects  "
        f"tfce_fwe={summary['n_cluster_verts_tfce']:,} cluster verts"
    )


if __name__ == "__main__":
    main()
