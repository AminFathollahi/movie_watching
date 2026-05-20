"""
01_extract_geometry.py
======================
Build vicsompy Subsurfaces and compute Laplace-Beltrami Operator Eigenfunctions
(LBOEs) for two HCP-MMP1 Glasser ROIs in 59k_fs_LR space.

Design:
  Subsurface59k subclasses vendor/hedger_cf/subsurface.py::Subsurface with two
  deliberate 59k-specific overrides:

  1. get_surfaces() — loads HCP group-average 59k midthickness GIFTIs directly
     instead of querying the pycortex database.

  2. laplacian_decomposition() — uses sigma=1e-6 instead of sigma=0.  The
     Laplacian always has a zero eigenvalue, so (-L - 0·I) is exactly singular
     and SpLU fails.  A tiny sigma shift makes the factorisation non-singular
     while converging to the same eigenmodes.

Everything else — generate(), make_laplacians(), save() — is inherited unchanged.

Run
---
    conda activate vicsompy_av
    python 01_extract_geometry.py \\
        --roi_a A5 --roi_b FFC \\
        --hcp_dir /path/to/HCP_S1200_GroupAvg_v1 \\
        --glasser_dlabel /path/to/Q1-Q6_...59k_fs_LR.dlabel.nii \\
        --output_base /path/to/outputs
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os
import sys

_DEFAULT_HCP_DIR = "/home/amin/Research/Representation/Movie/data/Setareh/HCP_S1200_GroupAvg_v1"
_DEFAULT_GLASSER_DLABEL = (
    f"{_DEFAULT_HCP_DIR}/"
    "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".59k_fs_LR.dlabel.nii"
)
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/per_subject"
_DEFAULT_ROI_A = "A5"
_DEFAULT_ROI_B = "FFC"

# Module-level names — rebound from args in __main__ before main() runs
HCP_DIR        = _DEFAULT_HCP_DIR
GLASSER_DLABEL = _DEFAULT_GLASSER_DLABEL
ROI_A          = _DEFAULT_ROI_A
ROI_B          = _DEFAULT_ROI_B
OUTPUT_BASE    = _DEFAULT_OUTPUT_BASE

SPHERE_LEFT        = f"{HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
SPHERE_RIGHT       = f"{HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"
SPHERE_32K_LEFT    = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
SPHERE_32K_RIGHT   = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"
MIDTHICK_32K_LEFT  = f"{HCP_DIR}/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
MIDTHICK_32K_RIGHT = f"{HCP_DIR}/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
MIDTHICK_59K_LEFT  = f"{HCP_DIR}/S1200.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
MIDTHICK_59K_RIGHT = f"{HCP_DIR}/S1200.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
PYCORTEX_FILESTORE = HCP_DIR
CX_SUB             = "hcp_999999_draw_NH"

OUTPUT_DIR = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"

N_LBOE = 200   # eigenfunctions per hemisphere; capped by ROI size

# =============================================================================
# IMPORTS
# =============================================================================
import pickle
import logging
import subprocess

import numpy as np
import scipy as sp
import nibabel as nib

os.environ["PYCORTEX_FILESTORE"] = PYCORTEX_FILESTORE
import cortex
import cortex.database
import cortex.polyutils

cortex.database.default_filestore = PYCORTEX_FILESTORE

# Add cf_modeling root to path for vendor/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vendor.hedger_cf.subsurface import Subsurface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Subsurface59k — two 59k-specific overrides of Hedger's Subsurface
# =============================================================================

class Subsurface59k(Subsurface):
    """Subsurface in 59k_fs_LR space.

    Inherits all behaviour from vendor/hedger_cf/subsurface.py::Subsurface.
    Overrides exactly two methods for 59k compatibility:
      - get_surfaces(): loads HCP group-average 59k midthickness GIFTIs
      - laplacian_decomposition(): sigma=1e-6 to avoid SpLU singularity
    """

    def __init__(self, boolmasks, surf_left_path, surf_right_path):
        super().__init__(cx_sub=CX_SUB, boolmasks=boolmasks)
        self._surf_left_path  = surf_left_path
        self._surf_right_path = surf_right_path

    def get_surfaces(self):
        """Load group-average 59k midthickness GIFTIs into self.surfaces."""
        self.surfaces = []
        for path in [self._surf_left_path, self._surf_right_path]:
            gi   = nib.load(path)
            pts  = gi.darrays[0].data.astype(np.float64)
            poly = gi.darrays[1].data.astype(np.int32)
            self.surfaces.append(cortex.polyutils.Surface(pts, poly))
        log.info(
            f"  Surfaces loaded — "
            f"L: {self.surfaces[0].pts.shape}, R: {self.surfaces[1].pts.shape}"
        )

    def laplacian_decomposition(self, subsurface, n_components):
        """Identical to vicsompy, except sigma=1e-6 (avoids SpLU singularity)."""
        B, D, W, V = subsurface.laplace_operator
        npt = W.shape[0]
        Dinv = sp.sparse.dia_matrix((D**-1, [0]), (npt, npt)).tocsr()
        L = Dinv.dot((V - W))
        eigenvalues, eigenvectors = sp.sparse.linalg.eigs(
            -L, k=n_components, which="LM", sigma=1e-6)
        return eigenvalues, eigenvectors


# =============================================================================
# STEP 1 — Load ROI masks from HCP-MMP1 59k_fs_LR dlabel.nii
# =============================================================================

def load_dlabel_masks():
    """Extract boolean ROI vertex masks from the HCP-MMP1 59k_fs_LR dlabel.nii."""
    if not os.path.exists(GLASSER_DLABEL):
        raise FileNotFoundError(
            f"Glasser dlabel not found: {GLASSER_DLABEL}\n"
            "Provide the HCP-MMP1 59k_fs_LR dlabel.nii via --glasser_dlabel."
        )
    log.info(f"Loading Glasser parcellation: {os.path.basename(GLASSER_DLABEL)}")
    img  = nib.load(GLASSER_DLABEL)
    data = img.get_fdata(dtype=np.float32)[0]    # (118584,)
    ax0  = img.header.get_axis(0)
    name2key = {name: key for key, (name, _) in ax0.label[0].items()}
    log.info(f"  {len(name2key)} labels, {(data == 0).sum()} background verts (label 0)")

    masks = {}
    for roi in [ROI_A, ROI_B]:
        lk = name2key.get(f"L_{roi}_ROI")
        rk = name2key.get(f"R_{roi}_ROI")
        if lk is None or rk is None:
            available = sorted(
                n.replace("L_", "").replace("_ROI", "")
                for n in name2key if n.startswith("L_") and n.endswith("_ROI")
            )
            raise ValueError(
                f"ROI '{roi}' not found in Glasser label table.\n"
                f"Expected 'L_{roi}_ROI' / 'R_{roi}_ROI'.\n"
                f"Available ROIs: {available}"
            )
        mL = (data[:59292] == lk)
        mR = (data[59292:] == rk)
        masks[roi] = (mL, mR)
        log.info(f"  {roi}: L={mL.sum()} verts  R={mR.sum()} verts  "
                 f"(labels L={lk} R={rk})")
    return masks


# =============================================================================
# STEP 2 — Ensure 59k midthickness surfaces exist (one-time wb_command resample)
# =============================================================================

def ensure_59k_midthickness():
    """Resample group-average 32k midthickness → 59k via wb_command (one-time)."""
    pairs = [
        (MIDTHICK_32K_LEFT,  SPHERE_32K_LEFT,  SPHERE_LEFT,  MIDTHICK_59K_LEFT),
        (MIDTHICK_32K_RIGHT, SPHERE_32K_RIGHT, SPHERE_RIGHT, MIDTHICK_59K_RIGHT),
    ]
    for src, src_sphere, tgt_sphere, dst in pairs:
        if os.path.exists(dst):
            log.info(f"  59k midthickness already exists: {os.path.basename(dst)}")
            continue
        for f in [src, src_sphere, tgt_sphere]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing file for midthickness resample: {f}")
        log.info(f"  Resampling {os.path.basename(src)} → 59k …")
        result = subprocess.run(
            ["wb_command", "-surface-resample",
             src, src_sphere, tgt_sphere, "BARYCENTRIC", dst],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"wb_command -surface-resample failed:\n{result.stderr}"
            )
        log.info(f"  Saved: {dst}")


# =============================================================================
# STEP 3 — Build Subsurface59k + LBOEs (or load from cache)
# =============================================================================

def build_subsurface(name, mask_L, mask_R):
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

    log.info(f"  [{name}] Building 59k subsurface …")
    sub = Subsurface59k([mask_L, mask_R], MIDTHICK_59K_LEFT, MIDTHICK_59K_RIGHT)
    sub.get_surfaces()
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

    sub.subsurface_verts = np.concatenate([sub.subsurface_verts_L, sub.subsurface_verts_R])

    # Downcast to base Subsurface before pickling so downstream scripts
    # can unpickle without importing Subsurface59k.
    sub.__class__ = Subsurface
    sub.save(cache_path)
    sub.__class__ = Subsurface59k
    log.info(f"  Saved: {cache_path}")
    return sub


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    log.info("=" * 60)
    log.info("Script 01 — Extract 59k geometry & LBOEs")
    log.info(f"  ROI A : {ROI_A}")
    log.info(f"  ROI B : {ROI_B}")
    log.info(f"  Surface space: 59k_fs_LR (group-average midthickness, 59292 verts/hem)")
    log.info(f"  LBOEs per hem: {N_LBOE} (capped by ROI size)")
    log.info(f"  Cache dir    : {CACHE_DIR}")
    log.info("=" * 60)

    ensure_59k_midthickness()
    masks = load_dlabel_masks()

    log.info(f"\nBuilding {ROI_A} subsurface …")
    build_subsurface(ROI_A, masks[ROI_A][0], masks[ROI_A][1])

    log.info(f"\nBuilding {ROI_B} subsurface …")
    build_subsurface(ROI_B, masks[ROI_B][0], masks[ROI_B][1])

    log.info("\nDone. Cached subsurfaces:")
    log.info(f"  {CACHE_DIR}/sub_{ROI_A.lower()}.pkl")
    log.info(f"  {CACHE_DIR}/sub_{ROI_B.lower()}.pkl")
    log.info(f"\nNext: python 02_prep_subject_cifti.py --subject <SUBJECT_ID>")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build Subsurfaces and LBOEs for two HCP-MMP1 Glasser ROIs."
    )
    parser.add_argument("--roi_a",          required=True,
                        help="First ROI name (e.g. A5, TA2, 3b)")
    parser.add_argument("--roi_b",          required=True,
                        help="Second ROI name (e.g. FFC, V1, MST)")
    parser.add_argument("--hcp_dir",        required=True,
                        help="HCP S1200 group-average atlas directory")
    parser.add_argument("--glasser_dlabel", required=True,
                        help="HCP-MMP1 59k_fs_LR dlabel.nii file")
    parser.add_argument("--output_base",    required=True,
                        help="Root output directory")
    args = parser.parse_args()

    ROI_A          = args.roi_a
    ROI_B          = args.roi_b
    HCP_DIR        = args.hcp_dir
    GLASSER_DLABEL = args.glasser_dlabel
    OUTPUT_BASE    = args.output_base

    SPHERE_LEFT        = f"{HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
    SPHERE_RIGHT       = f"{HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"
    SPHERE_32K_LEFT    = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
    SPHERE_32K_RIGHT   = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"
    MIDTHICK_32K_LEFT  = f"{HCP_DIR}/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    MIDTHICK_32K_RIGHT = f"{HCP_DIR}/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
    MIDTHICK_59K_LEFT  = f"{HCP_DIR}/S1200.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
    MIDTHICK_59K_RIGHT = f"{HCP_DIR}/S1200.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
    PYCORTEX_FILESTORE = HCP_DIR

    cortex.database.default_filestore = PYCORTEX_FILESTORE

    OUTPUT_DIR = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
    CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"

    main()
