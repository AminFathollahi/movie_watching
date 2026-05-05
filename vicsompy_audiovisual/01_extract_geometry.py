"""
01_extract_geometry.py
======================
Phase 1: Load Glasser parcellation, define bilateral A1/V1 ROIs, build
pycortex Subsurfaces, and compute 200 Laplace-Beltrami Operator
Eigenfunctions (LBOEs) per ROI hemisphere.

Outputs (to CACHE_DIR):
    sub_a1.pkl  — Subsurface object for bilateral A1 (LBOEs + vertex indices)
    sub_v1.pkl  — Subsurface object for bilateral V1 (LBOEs + vertex indices)

Divergences from vicsompy
-------------------------
1. Surface loading: vicsompy calls Subsurface.get_surfaces(), which reads
   surfaces via cortex.db.get_surf(). We bypass the pycortex DB entirely and
   load the HCP sphere surfaces directly from GIFTI files using nibabel, then
   construct cortex.polyutils.Surface objects and assign them to sub.surfaces.
   This gives identical geometry to vicsompy's surftype='sphere' default while
   avoiding the need for sphere files inside the pycortex subject directory.

2. Geodesic computation SKIPPED: vicsompy's Subsurface.create() also
   computes full pairwise geodesic distances (O(n²) calls). Those are only
   needed for the lookup-splicing step, which we do not use. We assign
   sub.surfaces directly → generate() → make_laplacians(), saving hours.

Run
---
    conda activate vicsompy_av
    python 01_extract_geometry.py
"""

# =============================================================================
# CONFIG
# =============================================================================

DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"

GIFTI_LEFT  = f"{DATA_BASE}/HCP Data/Q1-Q6_RelatedValidation210_LEFT.label.gii"
GIFTI_RIGHT = f"{DATA_BASE}/HCP Data/Q1-Q6_RelatedValidation210_RIGHT.label.gii"

# HCP sphere surfaces — used to build Laplacian on the sphere (identical to
# vicsompy's surftype='sphere').  Loaded via nibabel, bypassing pycortex DB.
SPHERE_GIFTI_LEFT  = f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/S1200.L.sphere.32k_fs_LR.surf.gii"
SPHERE_GIFTI_RIGHT = f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/S1200.R.sphere.32k_fs_LR.surf.gii"

# Parent directory of the pycortex subject folder (still needed for Subsurface
# init, but we never call get_surfaces() / cortex.db.get_surf()).
PYCORTEX_FILESTORE = f"{DATA_BASE}/HCP_S1200_GroupAvg_v1"
CX_SUB   = "hcp_999999_draw_NH"

# Glasser (Q1-Q6) label key values
LABEL_V1_LEFT  = 181   # L_V1_ROI
LABEL_V1_RIGHT = 1     # R_V1_ROI
LABEL_A1_LEFT  = 204   # L_A1_ROI
LABEL_A1_RIGHT = 24    # R_A1_ROI

N_LBOE = 200           # eigenfunctions per hemisphere (matches vicsompy config)

CACHE_DIR = "/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/subsurfaces"

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pickle
import logging

import numpy as np
import nibabel as nib

# Set pycortex filestore BEFORE import
os.environ["PYCORTEX_FILESTORE"] = PYCORTEX_FILESTORE

import cortex
import cortex.database
import cortex.polyutils

# Override in case the env-var wasn't picked up at import time
cortex.database.default_filestore = PYCORTEX_FILESTORE

from vicsompy.surface import Subsurface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — Load Glasser GIFTI and extract boolean ROI masks (32492 verts each)
# =============================================================================

def load_gifti_masks():
    """Return four boolean vertex masks (shape 32492,) for L/R A1 and V1.

    The masks index into the 32k_fs_LR surface space, which matches
    both the pycortex surface (32492 verts/hem) and the MAT fMRI data.
    """
    log.info("Loading Glasser GIFTI parcellations …")
    gi_left  = nib.load(GIFTI_LEFT)
    gi_right = nib.load(GIFTI_RIGHT)

    labels_L = gi_left.darrays[0].data.astype(int)    # (32492,)
    labels_R = gi_right.darrays[0].data.astype(int)   # (32492,)

    log.info(f"  Left : {labels_L.shape[0]} vertices, "
             f"{len(np.unique(labels_L))} unique labels")
    log.info(f"  Right: {labels_R.shape[0]} vertices, "
             f"{len(np.unique(labels_R))} unique labels")

    mask_a1_L = (labels_L == LABEL_A1_LEFT)
    mask_a1_R = (labels_R == LABEL_A1_RIGHT)
    mask_v1_L = (labels_L == LABEL_V1_LEFT)
    mask_v1_R = (labels_R == LABEL_V1_RIGHT)

    for tag, m in [("A1 Left", mask_a1_L), ("A1 Right", mask_a1_R),
                   ("V1 Left", mask_v1_L), ("V1 Right", mask_v1_R)]:
        log.info(f"  {tag}: {m.sum()} vertices")

    return mask_a1_L, mask_a1_R, mask_v1_L, mask_v1_R


# =============================================================================
# STEP 2 — Load sphere surfaces directly from HCP GIFTI files
# =============================================================================

def load_sphere_surfaces():
    """Load the HCP 32k sphere surfaces via nibabel and return a list of two
    cortex.polyutils.Surface objects [left, right].

    This bypasses cortex.db.get_surf() entirely (which would look for sphere
    files inside the pycortex subject folder, where they don't exist).
    The result is geometrically identical to vicsompy's surftype='sphere'.
    """
    surfaces = []
    for path in [SPHERE_GIFTI_LEFT, SPHERE_GIFTI_RIGHT]:
        gi   = nib.load(path)
        pts  = gi.darrays[0].data.astype(np.float64)   # (32492, 3) unit-sphere coords
        poly = gi.darrays[1].data.astype(np.int32)     # (n_faces, 3) triangle indices
        surfaces.append(cortex.polyutils.Surface(pts, poly))
    log.info(f"  Sphere surfaces loaded — "
             f"L: {surfaces[0].pts.shape}, R: {surfaces[1].pts.shape}")
    return surfaces


# =============================================================================
# STEP 3 — Build Subsurface + compute LBOEs
# =============================================================================

def build_subsurface(name, mask_L, mask_R):
    """Create a Subsurface with LBOEs or load from cache.

    Skips Subsurface.get_geometry() (pairwise geodesic distances), which
    is O(n_verts²) and only needed for the look-up splicing step.
    We only need:
      • subsurface_verts_L / subsurface_verts_R  — ROI vertex indices
      • L_eigenvectors / R_eigenvectors          — LBOE basis (n_verts, 200)
    """
    cache_path = os.path.join(CACHE_DIR, f"sub_{name.lower()}.pkl")

    if os.path.exists(cache_path):
        log.info(f"  [{name}] Loading from cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            sub = pickle.load(fh)
        # Backward-compat: old pickles may lack n_lboe attribute
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
            log.info(f"  [{name}] Inferred n_lboe={sub.n_lboe} from cached eigenvectors")
        log.info(f"  [{name}] L eigenvectors: {sub.L_eigenvectors.shape}  "
                 f"R eigenvectors: {sub.R_eigenvectors.shape}  "
                 f"n_lboe={sub.n_lboe}")
        return sub

    log.info(f"  [{name}] Building subsurface (sphere surface via nibabel) …")
    # Subsurface.__init__ only uses CX_SUB to initialise attribute names;
    # we immediately overwrite sub.surfaces before calling generate().
    sub = Subsurface(CX_SUB, [mask_L, mask_R])

    # Assign sphere surfaces directly — bypasses cortex.db.get_surf()
    sub.surfaces = load_sphere_surfaces()

    # generate: creates subsurface_L/R and derives subsurface_verts_L/R
    sub.generate()
    n_L = len(sub.subsurface_verts_L)
    n_R = len(sub.subsurface_verts_R)
    log.info(f"  [{name}] L verts: {n_L}  R verts: {n_R}")

    # Cap LBOEs to what is mathematically feasible: scipy.sparse.linalg.eigs
    # requires k < n (ARPACK constraint).  Small ROIs (e.g. A1) may have fewer
    # vertices than the requested N_LBOE, so we cap per hemisphere and take the
    # minimum so both hemispheres end up with the same number of eigenvectors.
    n_lboe = min(N_LBOE, n_L - 2, n_R - 2)
    if n_lboe < N_LBOE:
        log.warning(f"  [{name}] ROI too small for {N_LBOE} LBOEs "
                    f"(L={n_L}, R={n_R} verts); capping to {n_lboe}")

    # make_laplacians: spectral decomposition of the Laplace-Beltrami operator
    log.info(f"  [{name}] Computing {n_lboe} LBOEs per hemisphere …")
    sub.make_laplacians(n_lboe)
    sub.n_lboe = n_lboe   # store for downstream use (scripts 2 & 3)
    log.info(f"  [{name}] L eigenvectors: {sub.L_eigenvectors.shape}  "
             f"R eigenvectors: {sub.R_eigenvectors.shape}")

    # Cache: strip non-picklable pycortex objects before saving
    _save_subsurface(sub, cache_path)
    return sub


def _save_subsurface(sub, cache_path):
    """Pickle subsurface after nulling out pycortex Surface objects."""
    bak_surfaces    = sub.surfaces
    bak_sub_L       = sub.subsurface_L
    bak_sub_R       = sub.subsurface_R
    sub.surfaces    = None
    sub.subsurface_L = None
    sub.subsurface_R = None

    with open(cache_path, "wb") as fh:
        pickle.dump(sub, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Saved: {cache_path}")

    sub.surfaces    = bak_surfaces
    sub.subsurface_L = bak_sub_L
    sub.subsurface_R = bak_sub_R


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Script 1 — Extract geometry & LBOEs")
    log.info(f"  Pycortex subject : {CX_SUB}")
    log.info(f"  Surface type     : sphere (HCP GIFTI, vicsompy default)")
    log.info(f"  LBOEs per hem.   : {N_LBOE}")
    log.info("=" * 60)

    mask_a1_L, mask_a1_R, mask_v1_L, mask_v1_R = load_gifti_masks()

    log.info("\nBuilding A1 subsurface …")
    sub_a1 = build_subsurface("a1", mask_a1_L, mask_a1_R)

    log.info("\nBuilding V1 subsurface …")
    sub_v1 = build_subsurface("v1", mask_v1_L, mask_v1_R)

    log.info("\nDone. Cached subsurfaces:")
    log.info(f"  {CACHE_DIR}/sub_a1.pkl")
    log.info(f"  {CACHE_DIR}/sub_v1.pkl")


if __name__ == "__main__":
    main()
