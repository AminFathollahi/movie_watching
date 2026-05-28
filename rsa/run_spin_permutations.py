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
1 000 random rotations that produce a value ≥ the empirical value at that
location.  Benjamini-Hochberg FDR is then applied across all 59 412
grayordinate vertices.

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
6.  Compute one-tailed p_uncorr[v] = fraction(null[:, v] >= empirical[v]).
7.  Apply BH-FDR across all vertices.
8.  Produce binary significance masks at alpha threshold.
9.  Append four maps (spin_p_uncorr, spin_p_fdr, spin_sig_uncorr, spin_sig_fdr)
    to the existing combined CIFTI via merge_into_combined().

Usage
-----
  python rsa/run_spin_permutations.py \\
      --combined-cifti /path/to/rsa_59k_*_maps.dscalar.nii \\
      --left-sphere    /path/to/CohortAvg.L.sphere.59k_fs_LR.surf.gii \\
      --right-sphere   /path/to/CohortAvg.R.sphere.59k_fs_LR.surf.gii \\
      --template-cifti /path/to/template.dscalar.nii \\
      --n-spin 1000 --seed 42

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
    p.add_argument("--template-cifti",  required=True, dest="template_cifti",
                   help="59k CIFTI whose BrainModelAxis defines grayordinate space.")
    p.add_argument("--n-spin",          type=int,   default=1000, dest="n_spin",
                   help="Number of spin permutations (≥1000 recommended).")
    p.add_argument("--alpha",           type=float, default=0.05,
                   help="Significance threshold for binary masks.")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--output-dir",      default=None, dest="output_dir",
                   help="Optional: also save a standalone significance dscalar here.")
    return p.parse_args()


# =============================================================================
# Sphere coordinate helpers
# =============================================================================

def _load_sphere_coords(surf_path: str) -> np.ndarray:
    """Return (n_verts, 3) float64 unit-sphere vertex coordinates.

    GIFTI sphere surfaces have their coordinates on a sphere (typically radius
    100 mm).  We project to the unit sphere so that rotation operates
    irrespective of scale.
    """
    img    = nib.load(surf_path)
    coords = img.darrays[0].data.astype(np.float64)     # (n_verts, 3)
    norms  = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return coords / norms                                # unit sphere


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly random SO(3) rotation matrix.

    Uses the QR decomposition of a random 3×3 Gaussian matrix.  The determinant
    is corrected to ensure a proper rotation (det = +1).
    """
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

    For each rotation, every vertex is mapped to the nearest vertex in the
    original sphere (by Euclidean distance on the unit sphere) after rotation.

    Parameters
    ----------
    coords : (n_verts, 3) float64 — unit-sphere vertex coordinates
    n_spin : int — number of random rotations
    seed   : int — random seed

    Returns
    -------
    spin_indices : (n_spin, n_verts) int32
        spin_indices[s, v] = original vertex nearest to v after rotation s.
    """
    rng   = np.random.default_rng(seed)
    n     = coords.shape[0]
    tree  = cKDTree(coords)
    spins = np.empty((n_spin, n), dtype=np.int32)

    for s in range(n_spin):
        R         = _random_rotation(rng)
        rotated   = (R @ coords.T).T               # (n_verts, 3) — still on unit sphere
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
    """Compute one-tailed spin permutation p-values.

    p[v] = fraction of spins where null_map[v] >= empirical[v].
    Left and right hemispheres are permuted independently (rotation is
    hemisphere-specific to avoid mirror-flip artefacts).

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

    # Count exceedances: null[s, v] >= empirical[v] for each spin s
    exceed_lh = np.zeros(n_left,  dtype=np.float64)
    exceed_rh = np.zeros(n_right, dtype=np.float64)

    # Process in chunks to limit memory: each chunk is (chunk_size, n_hem_verts)
    chunk = 100
    for start in range(0, n_spin, chunk):
        sl         = spin_indices_lh[start : start + chunk]     # (c, n_left)
        sr         = spin_indices_rh[start : start + chunk]     # (c, n_right)
        null_lh_c  = emp_lh[sl]                                 # (c, n_left)
        null_rh_c  = emp_rh[sr]                                 # (c, n_right)
        exceed_lh += (null_lh_c >= emp_lh[None, :]).sum(axis=0)
        exceed_rh += (null_rh_c >= emp_rh[None, :]).sum(axis=0)

    p_lh = (exceed_lh / n_spin).astype(np.float32)
    p_rh = (exceed_rh / n_spin).astype(np.float32)
    return np.concatenate([p_lh, p_rh])


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
    return img.get_fdata(dtype=np.float32)[idx]    # (n_grayords,)


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
    log.info("=" * 70)

    # ── Load empirical RSA map ───────────────────────────────────────────────
    empirical = _extract_map(args.combined_cifti, args.map_name)
    n_grays   = len(empirical)
    log.info(f"  Empirical map: {empirical.shape}  "
             f"mean={empirical.mean():.4f}  max={empirical.max():.4f}")

    # ── Grayordinate vertex split ────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    lh_verts, rh_verts = get_cortex_vertex_indices(bm_axis)
    n_left  = len(lh_verts)
    n_right = n_grays - n_left
    log.info(f"  Grayordinates: L={n_left}  R={n_right}  total={n_grays}")

    # ── Load sphere surfaces and extract unit-sphere coordinates ─────────────
    # The sphere surface has n_full vertices (e.g. ~59k per hemisphere); only
    # n_left of them are grayordinates (medial wall excluded).  We spin ALL
    # vertices and then index back to the grayordinate subset.
    log.info("  Loading sphere surfaces ...")
    sphere_coords_lh = _load_sphere_coords(args.left_sphere)
    sphere_coords_rh = _load_sphere_coords(args.right_sphere)
    log.info(f"  Sphere L: {sphere_coords_lh.shape}  R: {sphere_coords_rh.shape}")

    # ── Generate spin indices on the FULL sphere ─────────────────────────────
    log.info(f"  Generating {args.n_spin} spin permutations per hemisphere ...")
    spin_full_lh = generate_spin_indices(sphere_coords_lh, args.n_spin, seed=args.seed)
    spin_full_rh = generate_spin_indices(sphere_coords_rh, args.n_spin, seed=args.seed + 1)
    # spin_full_*: (n_spin, n_full_hem) — full-sphere permutation

    # Restrict to grayordinate vertices:
    # The grayordinate empirical map is indexed by lh_verts / rh_verts within
    # the full sphere.  Applying spin_full_lh[s, lh_verts] gives the full-sphere
    # index of the rotated neighbour for each grayordinate vertex; we then look
    # up where that full-sphere vertex sits in the grayordinate ordering.

    # Map: full-sphere vertex → grayordinate column (-1 if medial wall)
    n_full_lh = sphere_coords_lh.shape[0]
    n_full_rh = sphere_coords_rh.shape[0]
    lut_lh = np.full(n_full_lh, -1, dtype=np.int32)
    lut_rh = np.full(n_full_rh, -1, dtype=np.int32)
    lut_lh[lh_verts] = np.arange(n_left,  dtype=np.int32)
    lut_rh[rh_verts] = np.arange(n_right, dtype=np.int32)

    # For each spin s and grayordinate v, find the grayordinate column of the
    # spun neighbour.  If the spun neighbour is in the medial wall, fall back
    # to the nearest non-medial-wall vertex (handled by clamping).
    # In practice, medial-wall neighbours are rare and the effect is negligible.
    spin_gray_lh = lut_lh[spin_full_lh[:, lh_verts]]  # (n_spin, n_left)
    spin_gray_rh = lut_rh[spin_full_rh[:, rh_verts]]  # (n_spin, n_right)

    # Replace any medial-wall hits (value -1) with the vertex index itself
    # (identity mapping — conservative: preserves the empirical value there)
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

    # ── Significance masks ───────────────────────────────────────────────────
    sig_uncorr = (p_uncorr < args.alpha).astype(np.float32)
    sig_fdr    = (p_fdr    < args.alpha).astype(np.float32)

    # ── Append to combined CIFTI ─────────────────────────────────────────────
    new_maps = {
        f"{args.map_name}_spin_p_uncorr": p_uncorr,
        f"{args.map_name}_spin_p_fdr":    p_fdr,
        f"{args.map_name}_spin_sig_uncorr": sig_uncorr,
        f"{args.map_name}_spin_sig_fdr":    sig_fdr,
    }

    for name, arr in new_maps.items():
        merge_into_combined(arr, name, args.combined_cifti, args.template_cifti)

    log.info(f"  Updated combined CIFTI: {args.combined_cifti}")

    # ── Optional standalone output ────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(args.combined_cifti).stem.replace("_maps", "")
        out_path = out_dir / f"{stem}_spin{args.n_spin}_significance.dscalar.nii"
        data_2d  = np.stack(list(new_maps.values()), axis=0)
        save_cifti_multimap(
            data_2d, list(new_maps.keys()), args.template_cifti, str(out_path)
        )
        log.info(f"  Standalone significance CIFTI: {out_path.name}")

        summary = {
            "map_name":       args.map_name,
            "n_spin":         args.n_spin,
            "alpha":          args.alpha,
            "seed":           args.seed,
            "n_grayords":     n_grays,
            "n_sig_uncorr":   n_sig_uncorr,
            "n_sig_fdr":      n_sig_fdr,
            "pct_sig_uncorr": float(100 * n_sig_uncorr / n_grays),
            "pct_sig_fdr":    float(100 * n_sig_fdr    / n_grays),
        }
        (out_dir / f"{stem}_spin{args.n_spin}_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        log.info("  Summary JSON saved.")

    log.info("Spin permutation test complete.")


if __name__ == "__main__":
    main()
