"""
01_extract_geometry.py
======================
Phase 1: Load Glasser parcellation, define bilateral audio/video source ROIs,
build pycortex Subsurfaces, and compute Laplace-Beltrami Operator
Eigenfunctions (LBOEs) per ROI hemisphere.

Outputs (to CACHE_DIR):
    sub_{ROI_A}.pkl  — Subsurface object for the first source ROI
    sub_{ROI_B}.pkl  — Subsurface object for the second source ROI

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
    python 01_extract_geometry.py \\
        --roi_a A1 --roi_b V1 \\
        --hcp_dir /path/to/HCP_S1200_GroupAvg_v1 \\
        --glasser_left /path/to/Q1-Q6..._LEFT.label.gii \\
        --glasser_right /path/to/Q1-Q6..._RIGHT.label.gii \\
        --output_base /path/to/outputs
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os
import sys

_DEFAULT_DATA_BASE   = "/home/amin/Research/Representation/Movie/data/Setareh"
_DEFAULT_HCP_DIR     = f"{_DEFAULT_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_DEFAULT_GLASSER_L   = f"{_DEFAULT_DATA_BASE}/HCP Data/Q1-Q6_RelatedValidation210_LEFT.label.gii"
_DEFAULT_GLASSER_R   = f"{_DEFAULT_DATA_BASE}/HCP Data/Q1-Q6_RelatedValidation210_RIGHT.label.gii"
_DEFAULT_ROI_A       = "A1"
_DEFAULT_ROI_B       = "V1"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/group_average"

ROI_A       = _DEFAULT_ROI_A
ROI_B       = _DEFAULT_ROI_B
HCP_DIR     = _DEFAULT_HCP_DIR
GLASSER_L   = _DEFAULT_GLASSER_L
GLASSER_R   = _DEFAULT_GLASSER_R
OUTPUT_BASE = _DEFAULT_OUTPUT_BASE

# Derived paths (rebound in __main__ after arg parsing)
OUTPUT_DIR = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"

# HCP sphere surfaces — used to build Laplacian on the sphere
SPHERE_L = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
SPHERE_R = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"

# Parent directory of the pycortex subject folder
PYCORTEX_FILESTORE = HCP_DIR
CX_SUB             = "hcp_999999_draw_NH"

N_LBOE = 200   # eigenfunctions per hemisphere (capped by ROI size)

# Glasser HCP-MMP1 label codes for common ROI pairs.
# Left and right codes differ because labels are hemisphere-specific.
ROI_DEFS = {
    "A1":  {"L": 204, "R": 24},
    "V1":  {"L": 181, "R": 1},
    "3b":  {"L": 189, "R": 9},
    "TA2": {"L": 287, "R": 107},
    "MST": {"L": 182, "R": 2},
    "A5":  {"L": 305, "R": 125},
    "FFC": {"L": 198, "R": 18},
}

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import pickle
import logging

import numpy as np
import nibabel as nib

os.environ["PYCORTEX_FILESTORE"] = PYCORTEX_FILESTORE
import cortex
import cortex.database
import cortex.polyutils

cortex.database.default_filestore = PYCORTEX_FILESTORE

# cf_modeling root on path so vendor/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vendor.hedger_cf.subsurface import Subsurface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — Load Glasser GIFTI and extract boolean ROI masks
# =============================================================================

def load_gifti_masks():
    """Return four boolean vertex masks (shape 32492,) for L/R ROI_A and ROI_B."""
    for roi in [ROI_A, ROI_B]:
        if roi not in ROI_DEFS:
            raise ValueError(
                f"ROI '{roi}' not in ROI_DEFS. Add its Glasser label codes or "
                f"choose from: {sorted(ROI_DEFS)}"
            )

    log.info("Loading Glasser GIFTI parcellations …")
    gi_left  = nib.load(GLASSER_L)
    gi_right = nib.load(GLASSER_R)

    labels_L = gi_left.darrays[0].data.astype(int)    # (32492,)
    labels_R = gi_right.darrays[0].data.astype(int)   # (32492,)

    log.info(f"  Left : {labels_L.shape[0]} vertices, "
             f"{len(np.unique(labels_L))} unique labels")
    log.info(f"  Right: {labels_R.shape[0]} vertices, "
             f"{len(np.unique(labels_R))} unique labels")

    masks = {}
    for roi in [ROI_A, ROI_B]:
        mL = (labels_L == ROI_DEFS[roi]["L"])
        mR = (labels_R == ROI_DEFS[roi]["R"])
        masks[roi] = (mL, mR)
        log.info(f"  {roi}: L={mL.sum()} verts  R={mR.sum()} verts")

    return masks


# =============================================================================
# STEP 2 — Load sphere surfaces from HCP GIFTIs
# =============================================================================

def load_sphere_surfaces():
    """Load HCP 32k sphere surfaces via nibabel → list of cortex.polyutils.Surface.

    Bypasses cortex.db.get_surf() entirely. Result is geometrically identical
    to vicsompy's surftype='sphere'.
    """
    surfaces = []
    for path in [SPHERE_L, SPHERE_R]:
        gi   = nib.load(path)
        pts  = gi.darrays[0].data.astype(np.float64)
        poly = gi.darrays[1].data.astype(np.int32)
        surfaces.append(cortex.polyutils.Surface(pts, poly))
    log.info(f"  Sphere surfaces loaded — "
             f"L: {surfaces[0].pts.shape}, R: {surfaces[1].pts.shape}")
    return surfaces


# =============================================================================
# STEP 3 — Build Subsurface + compute LBOEs
# =============================================================================

def _save_subsurface(sub, cache_path):
    """Pickle subsurface after nulling out non-picklable pycortex Surface objects."""
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


def build_subsurface(name, mask_L, mask_R):
    """Create a Subsurface with LBOEs, or load from cache.

    Only computes what downstream scripts need:
      • subsurface_verts_L / subsurface_verts_R  — ROI vertex indices
      • L_eigenvectors / R_eigenvectors          — LBOE basis (n_verts, n_lboe)
    Skips get_geometry() (pairwise geodesic distances), which is O(n_verts²) and
    only needed for the lookup-splicing step we do not use.
    """
    cache_path = os.path.join(CACHE_DIR, f"sub_{name.lower()}.pkl")

    if os.path.exists(cache_path):
        log.info(f"  [{name}] Loading from cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            sub = pickle.load(fh)
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
        log.info(f"  [{name}] L eigenvectors: {sub.L_eigenvectors.shape}  "
                 f"R eigenvectors: {sub.R_eigenvectors.shape}  n_lboe={sub.n_lboe}")
        return sub

    log.info(f"  [{name}] Building subsurface …")
    sub = Subsurface(CX_SUB, [mask_L, mask_R])
    sub.surfaces = load_sphere_surfaces()
    sub.generate()

    n_L = len(sub.subsurface_verts_L)
    n_R = len(sub.subsurface_verts_R)
    log.info(f"  [{name}] L verts: {n_L}  R verts: {n_R}")

    n_lboe = min(N_LBOE, n_L - 2, n_R - 2)
    if n_lboe < N_LBOE:
        log.warning(f"  [{name}] ROI too small for {N_LBOE} LBOEs "
                    f"(L={n_L}, R={n_R} verts); capping to {n_lboe}")

    log.info(f"  [{name}] Computing {n_lboe} LBOEs per hemisphere …")
    sub.make_laplacians(n_lboe)
    sub.n_lboe = n_lboe
    log.info(f"  [{name}] L eigenvectors: {sub.L_eigenvectors.shape}  "
             f"R eigenvectors: {sub.R_eigenvectors.shape}")

    _save_subsurface(sub, cache_path)
    return sub


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    log.info("=" * 60)
    log.info("Script 01 — Extract geometry & LBOEs (group average)")
    log.info(f"  ROI A     : {ROI_A}")
    log.info(f"  ROI B     : {ROI_B}")
    log.info(f"  Surface   : 32k_fs_LR sphere (HCP GIFTI)")
    log.info(f"  LBOEs/hem : {N_LBOE} (capped by ROI size)")
    log.info(f"  Cache dir : {CACHE_DIR}")
    log.info("=" * 60)

    masks = load_gifti_masks()

    log.info(f"\nBuilding {ROI_A} subsurface …")
    build_subsurface(ROI_A, masks[ROI_A][0], masks[ROI_A][1])

    log.info(f"\nBuilding {ROI_B} subsurface …")
    build_subsurface(ROI_B, masks[ROI_B][0], masks[ROI_B][1])

    log.info("\nDone. Cached subsurfaces:")
    log.info(f"  {CACHE_DIR}/sub_{ROI_A.lower()}.pkl")
    log.info(f"  {CACHE_DIR}/sub_{ROI_B.lower()}.pkl")
    log.info(f"\nNext: python 02_prep_hcp_timeseries.py --roi_a {ROI_A} --roi_b {ROI_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Subsurfaces and LBOEs for two Glasser ROIs (group-average data)."
    )
    parser.add_argument("--roi_a",          default=_DEFAULT_ROI_A,
                        help="First ROI name, must be in ROI_DEFS (default: %(default)s)")
    parser.add_argument("--roi_b",          default=_DEFAULT_ROI_B,
                        help="Second ROI name, must be in ROI_DEFS (default: %(default)s)")
    parser.add_argument("--hcp_dir",        default=_DEFAULT_HCP_DIR,
                        help="HCP S1200 group-average atlas directory (default: %(default)s)")
    parser.add_argument("--glasser_left",   default=_DEFAULT_GLASSER_L,
                        help="Glasser left hemisphere label GIFTI (default: %(default)s)")
    parser.add_argument("--glasser_right",  default=_DEFAULT_GLASSER_R,
                        help="Glasser right hemisphere label GIFTI (default: %(default)s)")
    parser.add_argument("--output_base",    default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    args = parser.parse_args()

    ROI_A       = args.roi_a
    ROI_B       = args.roi_b
    HCP_DIR     = args.hcp_dir
    GLASSER_L   = args.glasser_left
    GLASSER_R   = args.glasser_right
    OUTPUT_BASE = args.output_base
    OUTPUT_DIR  = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
    CACHE_DIR   = f"{OUTPUT_DIR}/subsurfaces"
    SPHERE_L    = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
    SPHERE_R    = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"
    PYCORTEX_FILESTORE = HCP_DIR

    cortex.database.default_filestore = PYCORTEX_FILESTORE

    main()
