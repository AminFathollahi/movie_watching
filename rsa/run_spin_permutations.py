"""
rsa/run_spin_permutations.py
=============================
Surface-based spin permutation significance testing for group-average
searchlight RSA maps.

Background
----------
Standard vertex-level p-values ignore spatial autocorrelation — nearby cortical
vertices are not independent, so false-positive clusters arise.  Spin tests
(Alexander-Bloch et al. 2018, NeuroImage) address this by rotating the entire
cortical map on the sphere; the resulting null maps preserve the spatial
autocorrelation structure of the empirical data.

For each vertex the one-tailed p-value is estimated as the fraction of
n_spin random rotations that produce a value ≥ the empirical value at that
location (plus a pseudo-count of 1 for a proper Monte Carlo estimate).
Benjamini-Hochberg FDR is then applied across all grayordinate vertices.

Algorithm
---------
1.  Load the group-average combined CIFTI dscalar (output of run_analysis.sh)
    and extract the specified RSA map (default: searchlight_spearman_rho).
2.  Load the sphere registration surfaces (*.sphere.59k_fs_LR.surf.gii) and
    extract vertex coordinates.  Coordinates are mapped to the unit sphere.
3.  Generate n_spin random SO(3) rotation matrices (QR decomposition of a
    random Gaussian matrix, determinant corrected).
4.  For each hemisphere independently: rotate the sphere coordinates, find the
    nearest original vertex for each rotated position using a cKDTree → spin
    index permutation.
5.  Build the null distribution:
      null[s, v] = empirical[spin_indices[s, v]]
6.  Compute one-tailed p_uncorr[v] = (count(null[:, v] >= empirical[v]) + 1) / (n_spin + 1)
7.  Apply BH-FDR (Benjamini-Hochberg) across all vertices.
8.  Produce output maps:
      sigmap_uncorr      : sign(rho) × −log10(p_uncorr)
      sigmap_fdr         : sign(rho) × −log10(p_fdr)
      cluster_mask_fdr   : significant FDR vertices in clusters ≥ min_cluster_size
      cluster_borders_fdr: borders around those clusters
9.  Append four maps to the existing combined CIFTI via merge_into_combined().

Usage
-----
python rsa/run_spin_permutations.py \
    --combined-cifti /home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average/pe-av-small-16-frame/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_center_maps.dscalar.nii \
    --map-name searchlight_spearman_rho \
    --left-sphere /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/L.sphere.59k_fs_LR.surf.gii \
    --right-sphere /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/R.sphere.59k_fs_LR.surf.gii \
    --left-surface /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii \
    --right-surface /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii \
    --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \
    --n-spin 1000 \
    --alpha 0.05 \
    --output-dir /home/amin/Research/Representation/Movie/outputs/rsa/spin_tests

Reference
---------
Alexander-Bloch A et al. (2018). On testing for spatial correspondence
between maps of human brain structure and function. NeuroImage 178, 540-551.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import (
    get_bm_axis,
    get_cortex_vertex_indices,
    merge_into_combined,
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
        description="Spin permutation significance testing for group-average RSA maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--combined-cifti",  required=True, dest="combined_cifti",
                   help="Path to combined *_maps.dscalar.nii to update.")
    p.add_argument("--map-name",        default="searchlight_spearman_rho",
                   dest="map_name",
                   help="Scalar map name within the combined CIFTI to test.")
    p.add_argument("--left-sphere",     required=True, dest="left_sphere",
                   help="Left hemisphere sphere registration .surf.gii "
                        "(*.sphere.59k_fs_LR.surf.gii).")
    p.add_argument("--right-sphere",    required=True, dest="right_sphere",
                   help="Right hemisphere sphere registration .surf.gii.")
    p.add_argument("--left-surface",    required=True, dest="left_surface",
                   help="Left hemisphere midthickness .surf.gii (for cluster adjacency).")
    p.add_argument("--right-surface",   required=True, dest="right_surface",
                   help="Right hemisphere midthickness .surf.gii (for cluster adjacency).")
    p.add_argument("--template-cifti",  required=True, dest="template_cifti",
                   help="59k CIFTI whose BrainModelAxis defines grayordinate space.")
    p.add_argument("--n-spin",          type=int,   default=1000, dest="n_spin",
                   help="Number of spin permutations (≥1000 recommended).")
    p.add_argument("--alpha",           type=float, default=0.05,
                   help="Significance threshold for cluster detection.")
    p.add_argument("--min-cluster-size", type=int,  default=10, dest="min_cluster_size",
                   help="Minimum cluster size (vertices) to include in masks.")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--output-dir",      default=None, dest="output_dir",
                   help="Optional: also save a standalone significance dscalar here.")
    return p.parse_args()


# =============================================================================
# Sphere coordinate helpers
# =============================================================================

def _load_sphere_coords(surf_path: str) -> np.ndarray:
    """Return (n_verts, 3) float64 unit-sphere vertex coordinates."""
    img    = nib.load(surf_path)
    coords = img.darrays[0].data.astype(np.float64)
    norms  = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return coords / norms


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly random SO(3) rotation matrix via QR decomposition."""
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# =============================================================================
# Spin index generation
# =============================================================================

def generate_spin_indices(
    coords: np.ndarray,
    n_spin: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate spin permutation index arrays for one hemisphere.

    Parameters
    ----------
    coords : (n_verts, 3) float64 — unit-sphere vertex coordinates
    n_spin : int — number of random rotations
    seed   : int — random seed

    Returns
    -------
    spin_indices : (n_spin, n_verts) int32
    """
    rng   = np.random.default_rng(seed)
    n     = coords.shape[0]
    tree  = cKDTree(coords)
    spins = np.empty((n_spin, n), dtype=np.int32)

    for s in range(n_spin):
        R         = _random_rotation(rng)
        rotated   = (R @ coords.T).T
        _, idx    = tree.query(rotated, k=1)
        spins[s]  = idx.astype(np.int32)

        if (s + 1) % 200 == 0:
            log.info(f"  Generated {s + 1}/{n_spin} spin permutations ...")

    return spins


# =============================================================================
# P-value computation
# =============================================================================

def compute_spin_pvalues(
    empirical: np.ndarray,
    spin_indices_lh: np.ndarray,
    spin_indices_rh: np.ndarray,
    n_left: int,
) -> np.ndarray:
    """Compute one-tailed spin permutation p-values with pseudo-count.

    p[v] = (count(null[:, v] >= empirical[v]) + 1) / (n_spin + 1)

    The +1 pseudo-count gives an unbiased Monte Carlo estimate and avoids p=0.

    Parameters
    ----------
    empirical        : (n_grayords,) float32 — observed RSA map
    spin_indices_lh  : (n_spin, n_left) int32 — left-hemisphere spin indices
    spin_indices_rh  : (n_spin, n_right) int32 — right-hemisphere spin indices
    n_left           : int — number of left-hemisphere grayordinates

    Returns
    -------
    p_uncorr : (n_grayords,) float32 — one-tailed p-values
    """
    n_spin   = spin_indices_lh.shape[0]
    n_grays  = len(empirical)
    n_right  = n_grays - n_left

    emp_lh = empirical[:n_left]
    emp_rh = empirical[n_left:]

    # Initialise with pseudo-count of 1
    exceed_lh = np.ones(n_left,  dtype=np.float64)
    exceed_rh = np.ones(n_right, dtype=np.float64)

    chunk = 100
    for start in range(0, n_spin, chunk):
        sl         = spin_indices_lh[start : start + chunk]
        sr         = spin_indices_rh[start : start + chunk]
        null_lh_c  = emp_lh[sl]
        null_rh_c  = emp_rh[sr]
        exceed_lh += (null_lh_c >= emp_lh[None, :]).sum(axis=0)
        exceed_rh += (null_rh_c >= emp_rh[None, :]).sum(axis=0)

    denom = float(n_spin + 1)
    p_lh = (exceed_lh / denom).astype(np.float32)
    p_rh = (exceed_rh / denom).astype(np.float32)
    return np.concatenate([p_lh, p_rh])


# =============================================================================
# Surface adjacency and cluster analysis
# =============================================================================

def _load_faces(surf_path: str) -> np.ndarray:
    """Return (n_faces, 3) int32 face array from a GIFTI surface file."""
    surf = nib.load(surf_path)
    return surf.darrays[1].data.astype(np.int32)


def _surface_adjacency(faces: np.ndarray,
                        vertex_idx: np.ndarray,
                        n_cifti_verts: int) -> csr_matrix:
    """Sparse adjacency matrix in CIFTI grayordinate space."""
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


def _cluster_mask_and_borders(
    sig_mask: np.ndarray,
    adj: csr_matrix,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cluster membership and border masks.

    Returns
    -------
    cluster_mask : (n_cifti_verts,) float32 — 1 inside valid clusters, 0 outside
    border_mask  : (n_cifti_verts,) float32 — 1 at cluster borders, 0 elsewhere
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
        for v in comp_global:
            nbrs = adj[v].indices
            if np.any(~sig_mask[nbrs]):
                border_mask[v] = 1.0

    return cluster_mask, border_mask


# =============================================================================
# Map extraction from combined CIFTI
# =============================================================================

def _extract_map(cifti_path: str, map_name: str) -> np.ndarray:
    """Return the 1-D array for a named scalar map in a dscalar CIFTI."""
    img  = nib.load(cifti_path)
    ax0  = img.header.get_axis(0)
    names = [ax0.name[i] for i in range(img.shape[0])]
    if map_name not in names:
        raise KeyError(
            f"Map '{map_name}' not found in {cifti_path}.\n"
            f"Available maps: {names}"
        )
    idx  = names.index(map_name)
    return img.get_fdata(dtype=np.float32)[idx]


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("Spin permutation significance test")
    log.info(f"  Input CIFTI   : {args.combined_cifti}")
    log.info(f"  Map name      : {args.map_name}")
    log.info(f"  n_spin        : {args.n_spin}")
    log.info(f"  alpha         : {args.alpha}")
    log.info(f"  min_cluster   : {args.min_cluster_size}")
    log.info("=" * 70)

    # ── Load empirical RSA map ───────────────────────────────────────────────
    empirical = _extract_map(args.combined_cifti, args.map_name)
    n_grays   = len(empirical)
    log.info(f"  Empirical map: {empirical.shape}  "
             f"mean={empirical.mean():.4f}  max={empirical.max():.4f}")

    # ── Grayordinate vertex split (use SOURCE CIFTI's BM axis) ──────────────
    # The combined CIFTI and template may have different grayordinate counts
    # (e.g., if the template includes subcortical structures). Always derive
    # n_left from the map file itself to guarantee consistent hemisphere slicing.
    bm_axis = get_bm_axis(args.combined_cifti)
    lh_verts, rh_verts = get_cortex_vertex_indices(bm_axis)
    n_left  = len(lh_verts)
    n_right = n_grays - n_left
    log.info(f"  Grayordinates: L={n_left}  R={n_right}  total={n_grays}")

    # ── Load sphere surfaces ─────────────────────────────────────────────────
    log.info("  Loading sphere surfaces ...")
    sphere_coords_lh = _load_sphere_coords(args.left_sphere)
    sphere_coords_rh = _load_sphere_coords(args.right_sphere)
    log.info(f"  Sphere L: {sphere_coords_lh.shape}  R: {sphere_coords_rh.shape}")

    # ── Generate spin indices on the FULL sphere ─────────────────────────────
    log.info(f"  Generating {args.n_spin} spin permutations per hemisphere ...")
    spin_full_lh = generate_spin_indices(sphere_coords_lh, args.n_spin, seed=args.seed)
    spin_full_rh = generate_spin_indices(sphere_coords_rh, args.n_spin, seed=args.seed + 1)

    # Restrict full-sphere spin indices to grayordinate vertex space.
    # lut_*[full_sphere_vertex] → grayordinate column (-1 = medial wall)
    n_full_lh = sphere_coords_lh.shape[0]
    n_full_rh = sphere_coords_rh.shape[0]
    lut_lh = np.full(n_full_lh, -1, dtype=np.int32)
    lut_rh = np.full(n_full_rh, -1, dtype=np.int32)
    lut_lh[lh_verts] = np.arange(n_left,  dtype=np.int32)
    lut_rh[rh_verts] = np.arange(n_right, dtype=np.int32)

    spin_gray_lh = lut_lh[spin_full_lh[:, lh_verts]]  # (n_spin, n_left)
    spin_gray_rh = lut_rh[spin_full_rh[:, rh_verts]]  # (n_spin, n_right)

    # Replace medial-wall hits with identity (conservative: preserves empirical)
    bad_lh = spin_gray_lh < 0
    bad_rh = spin_gray_rh < 0
    if bad_lh.any():
        row_l, col_l = np.where(bad_lh)
        spin_gray_lh[row_l, col_l] = col_l.astype(np.int32)
    if bad_rh.any():
        row_r, col_r = np.where(bad_rh)
        spin_gray_rh[row_r, col_r] = col_r.astype(np.int32)

    del spin_full_lh, spin_full_rh
    log.info("  Spin indices restricted to grayordinate space.")

    # ── Compute p-values ─────────────────────────────────────────────────────
    log.info("  Computing spin p-values ...")
    p_uncorr = compute_spin_pvalues(
        empirical, spin_gray_lh, spin_gray_rh, n_left
    )

    # ── BH-FDR correction ────────────────────────────────────────────────────
    p_fdr = false_discovery_control(p_uncorr, method='bh').astype(np.float32)

    n_sig_uncorr = int((p_uncorr < args.alpha).sum())
    n_sig_fdr    = int((p_fdr    < args.alpha).sum())
    log.info(f"  Uncorrected p<{args.alpha}: {n_sig_uncorr:,} / {n_grays:,} vertices "
             f"({100*n_sig_uncorr/n_grays:.1f}%)")
    log.info(f"  FDR q<{args.alpha}:        {n_sig_fdr:,} / {n_grays:,} vertices "
             f"({100*n_sig_fdr/n_grays:.1f}%)")

    # ── Sigmaps: sign(rho) × −log₁₀(p) ──────────────────────────────────────
    eps = np.finfo(np.float32).tiny
    sigmap_uncorr = (np.sign(empirical) *
                     (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)
    sigmap_fdr    = (np.sign(empirical) *
                     (-np.log10(np.maximum(p_fdr,    eps)))).astype(np.float32)

    # ── Surface adjacency for cluster analysis ────────────────────────────────
    # Template CIFTI BM axis gives correct vertex indices for each hemisphere
    bm_tmpl = get_bm_axis(args.template_cifti)
    lh_verts_tmpl, rh_verts_tmpl = get_cortex_vertex_indices(bm_tmpl)
    n_left_tmpl  = len(lh_verts_tmpl)
    n_right_tmpl = n_grays - n_left_tmpl

    log.info(f"Building surface adjacency ...")
    faces_lh = _load_faces(args.left_surface)
    faces_rh = _load_faces(args.right_surface)
    adj_lh   = _surface_adjacency(faces_lh, lh_verts_tmpl,  n_left_tmpl)
    adj_rh   = _surface_adjacency(faces_rh, rh_verts_tmpl, n_right_tmpl)
    log.info("  Adjacency built.")

    # ── Cluster masks (FDR) ───────────────────────────────────────────────────
    sig_lh_fdr = p_fdr[:n_left_tmpl]  < args.alpha
    sig_rh_fdr = p_fdr[n_left_tmpl:]  < args.alpha

    cm_lh, cb_lh = _cluster_mask_and_borders(sig_lh_fdr, adj_lh, args.min_cluster_size)
    cm_rh, cb_rh = _cluster_mask_and_borders(sig_rh_fdr, adj_rh, args.min_cluster_size)

    cluster_mask_fdr    = np.concatenate([cm_lh, cm_rh])
    cluster_borders_fdr = np.concatenate([cb_lh, cb_rh])

    n_cluster = int(cluster_mask_fdr.sum())
    n_border  = int(cluster_borders_fdr.sum())
    log.info(f"  FDR clusters (≥{args.min_cluster_size} verts): "
             f"{n_cluster:,} verts  borders: {n_border:,}")

    # ── Build output maps ─────────────────────────────────────────────────────
    new_maps = {
        f"{args.map_name}_sigmap_uncorr":      sigmap_uncorr,
        f"{args.map_name}_sigmap_fdr":         sigmap_fdr,
        f"{args.map_name}_cluster_mask_fdr":   cluster_mask_fdr,
        f"{args.map_name}_cluster_borders_fdr": cluster_borders_fdr,
    }

    # ── Append to combined CIFTI ─────────────────────────────────────────────
    for name, arr in new_maps.items():
        merge_into_combined(arr, name, args.combined_cifti, args.template_cifti)

    log.info(f"  Updated combined CIFTI: {args.combined_cifti}")

    # ── Optional standalone output ────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem     = Path(args.combined_cifti).stem.replace("_maps", "")
        out_path = out_dir / f"{stem}_spin{args.n_spin}_significance.dscalar.nii"
        data_2d  = np.stack(list(new_maps.values()), axis=0)
        save_cifti_multimap(
            data_2d, list(new_maps.keys()), args.template_cifti, str(out_path)
        )
        log.info(f"  Standalone significance CIFTI: {out_path.name}")

        summary = {
            "map_name":          args.map_name,
            "n_spin":            args.n_spin,
            "alpha":             args.alpha,
            "min_cluster_size":  args.min_cluster_size,
            "seed":              args.seed,
            "n_grayords":        n_grays,
            "n_sig_uncorr":      n_sig_uncorr,
            "n_sig_fdr":         n_sig_fdr,
            "n_cluster_verts_fdr": n_cluster,
            "n_border_verts_fdr":  n_border,
            "pct_sig_uncorr":    float(100 * n_sig_uncorr / n_grays),
            "pct_sig_fdr":       float(100 * n_sig_fdr    / n_grays),
            "sigmap_uncorr_max": float(sigmap_uncorr.max()),
            "sigmap_fdr_max":    float(sigmap_fdr.max()),
        }
        (out_dir / f"{stem}_spin{args.n_spin}_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        log.info("  Summary JSON saved.")

    log.info("Spin permutation test complete.")


if __name__ == "__main__":
    main()
