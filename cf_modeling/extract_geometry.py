"""
cf_modeling/extract_geometry.py
================================
Build vicsompy Subsurfaces and compute Laplace-Beltrami Operator
Eigenfunctions (LBOEs) for two HCP-MMP1 Glasser ROIs.

Following Hedger et al. (2025), sphere surfaces are used for the Laplace-
Beltrami decomposition.  Both modes use the 59k_fs_LR sphere surfaces
(S1200.{L,R}.sphere.59k_fs_LR.surf.gii) and the 59k_fs_LR Glasser dlabel
for ROI masks, consistent with the 59k_fs_LR data space of the preprocessed
CIFTIs produced by preprocess_individual.py.

The --mode flag determines the output directory only:
  group_average:  {output_base}/group_average/{ROI_A}_{ROI_B}/subsurfaces/
  per_subject:    {output_base}/per_subject/{ROI_A}_{ROI_B}/subsurfaces/

Subsurfaces are built using the Subsurface class from Hedger et al. (2025)
vicsompy (MIT License). See vendor/hedger_cf/ for source and attribution.

Run
---
    conda activate cfmod
    python extract_geometry.py --mode group_average --roi_a A1 --roi_b V1
    python extract_geometry.py --mode per_subject   --roi_a A5 --roi_b FFC
"""

import argparse
import logging
import os
import pickle

import nibabel as nib
import numpy as np
import scipy as sp

os.environ.setdefault("PYCORTEX_FILESTORE", "/tmp")
import cortex
import cortex.database
import cortex.polyutils

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.hedger_cf.subsurface import Subsurface

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

N_LBOE = 200
CX_SUB  = "hcp_999999_draw_NH"

_DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"
_HCP_DIR   = f"{_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_OUT_BASE  = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"


# =============================================================================
# Subsurface59k — 59k_fs_LR override of the Hedger et al. Subsurface
# =============================================================================

class Subsurface59k(Subsurface):
    """Subsurface for 59k_fs_LR sphere surfaces.

    Extends Subsurface (Hedger et al. 2025, vicsompy, MIT License) with:
      - get_surfaces(): loads HCP group-average 59k_fs_LR sphere GIFTIs
        directly, bypassing the pycortex surface-type lookup.
      - laplacian_decomposition(): uses sigma=1e-6 shift to avoid the SpLU
        singularity at the zero eigenvalue of the graph Laplacian.

    The surface type (sphere) follows the convention in Hedger et al. (2025).
    The mathematical logic — Laplace-Beltrami decomposition and eigenvector
    projection — is unchanged from the original Subsurface class.
    """

    def __init__(self, boolmasks, surf_left_path, surf_right_path):
        super().__init__(cx_sub=CX_SUB, boolmasks=boolmasks)
        self._surf_L = surf_left_path
        self._surf_R = surf_right_path

    def get_surfaces(self):
        self.surfaces = []
        for path in [self._surf_L, self._surf_R]:
            gi = nib.load(path)
            self.surfaces.append(cortex.polyutils.Surface(
                gi.darrays[0].data.astype(np.float64),
                gi.darrays[1].data.astype(np.int32),
            ))
        log.info(f"  59k sphere: L={self.surfaces[0].pts.shape}  "
                 f"R={self.surfaces[1].pts.shape}")

    def laplacian_decomposition(self, subsurface, n_components):
        B, D, W, V = subsurface.laplace_operator
        npt = W.shape[0]
        Dinv = sp.sparse.dia_matrix((D ** -1, [0]), (npt, npt)).tocsr()
        L = Dinv.dot(V - W)
        eigenvalues, eigenvectors = sp.sparse.linalg.eigs(
            -L, k=n_components, which="LM", sigma=1e-6)
        return eigenvalues, eigenvectors


# =============================================================================
# ROI mask loading
# =============================================================================

def load_dlabel_masks(glasser_dlabel, roi_a, roi_b):
    """Load ROI boolean masks from 59k_fs_LR Glasser dlabel.nii."""
    log.info(f"Loading Glasser dlabel: {os.path.basename(glasser_dlabel)}")
    img  = nib.load(glasser_dlabel)
    data = img.get_fdata(dtype=np.float32)[0]   # (118584,)
    ax0  = img.header.get_axis(0)
    name2key = {name: key for key, (name, _) in ax0.label[0].items()}
    masks = {}
    for roi in [roi_a, roi_b]:
        lk = name2key.get(f"L_{roi}_ROI")
        rk = name2key.get(f"R_{roi}_ROI")
        if lk is None or rk is None:
            available = sorted(
                n.replace("L_", "").replace("_ROI", "")
                for n in name2key if n.startswith("L_") and n.endswith("_ROI")
            )
            raise ValueError(
                f"ROI '{roi}' not found in dlabel. Available: {available}")
        mL = (data[:59292] == lk)
        mR = (data[59292:] == rk)
        masks[roi] = (mL, mR)
        log.info(f"  {roi}: L={mL.sum()} verts  R={mR.sum()} verts  "
                 f"(labels L={lk} R={rk})")
    return masks


# =============================================================================
# Subsurface builder
# =============================================================================

def _save_subsurface(sub, cache_path):
    """Save subsurface by downcasting to base Subsurface for portability."""
    sub.__class__ = Subsurface
    sub.save(cache_path)
    sub.__class__ = Subsurface59k
    log.info(f"  Saved: {cache_path}")


def build_subsurface(name, mask_L, mask_R, cache_dir, surf_L, surf_R):
    """Build or load a Subsurface59k with LBOEs.

    Cached at cache_dir/sub_{name.lower()}.pkl.
    Sets sub.subsurface_verts = concat([verts_L, verts_R]) (absolute bilateral
    indices, verts_R offset by 59292) for null-model mean computation in
    script 02.
    """
    cache_path = os.path.join(cache_dir, f"sub_{name.lower()}.pkl")

    if os.path.exists(cache_path):
        log.info(f"  [{name}] Loading cached: {cache_path}")
        with open(cache_path, "rb") as fh:
            sub = pickle.load(fh)
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
        if not hasattr(sub, "subsurface_verts"):
            sub.subsurface_verts = np.concatenate([
                sub.subsurface_verts_L, sub.subsurface_verts_R])
        log.info(f"  [{name}] L={sub.L_eigenvectors.shape}  "
                 f"R={sub.R_eigenvectors.shape}  n_lboe={sub.n_lboe}")
        return sub

    log.info(f"  [{name}] Building 59k sphere subsurface …")
    sub = Subsurface59k([mask_L, mask_R], surf_L, surf_R)
    sub.get_surfaces()
    sub.generate()

    n_L = len(sub.subsurface_verts_L)
    n_R = len(sub.subsurface_verts_R)    # absolute indices (includes 59292 offset)
    log.info(f"  [{name}] L verts: {n_L}  R verts (abs): {n_R}")

    n_lboe = min(N_LBOE, n_L - 2, n_R - 2)
    if n_lboe < N_LBOE:
        log.warning(
            f"  [{name}] Capping LBOEs to {n_lboe} "
            f"(ROI has only {n_L}/{n_R} verts)")

    log.info(f"  [{name}] Computing {n_lboe} LBOEs …")
    sub.make_laplacians(n_lboe)
    sub.n_lboe = n_lboe

    # Absolute bilateral indices for null-model mean timecourse (script 02)
    sub.subsurface_verts = np.concatenate([
        sub.subsurface_verts_L, sub.subsurface_verts_R
    ])

    _save_subsurface(sub, cache_path)
    return sub


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Subsurfaces and LBOEs for two Glasser ROIs "
                    "(59k_fs_LR sphere, following Hedger et al. 2025).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",          required=True,
                   choices=["group_average", "per_subject"])
    p.add_argument("--roi_a",         default="A1")
    p.add_argument("--roi_b",         default="V1")
    p.add_argument("--hcp_dir",       default=_HCP_DIR)
    p.add_argument("--output_base",   default=_OUT_BASE)
    p.add_argument("--glasser_dlabel",
                   default=(f"{_HCP_DIR}/Q1-Q6_RelatedParcellation210"
                            ".CorticalAreas_dil_Final_Final_Areas_Group_Colors"
                            ".59k_fs_LR.dlabel.nii"),
                   help="Glasser HCP-MMP1 59k_fs_LR dlabel.nii.")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["PYCORTEX_FILESTORE"] = args.hcp_dir
    cortex.database.default_filestore = args.hcp_dir

    roi_dir   = f"{args.output_base}/{args.mode}/{args.roi_a}_{args.roi_b}"
    cache_dir = f"{roi_dir}/subsurfaces"
    os.makedirs(cache_dir, exist_ok=True)

    surf_L = f"{args.hcp_dir}/S1200.L.sphere.59k_fs_LR.surf.gii"
    surf_R = f"{args.hcp_dir}/S1200.R.sphere.59k_fs_LR.surf.gii"
    for f in [surf_L, surf_R]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"59k sphere surface not found: {f}\n"
                f"Expected in HCP_DIR: {args.hcp_dir}")

    log.info("=" * 60)
    log.info(f"Script 01 — Extract geometry & LBOEs ({args.mode})")
    log.info(f"  ROIs    : {args.roi_a} × {args.roi_b}  |  max LBOEs: {N_LBOE}")
    log.info(f"  Surface : 59k_fs_LR sphere (Hedger et al. 2025 convention)")
    log.info(f"  Cache   : {cache_dir}")
    log.info("=" * 60)

    masks = load_dlabel_masks(args.glasser_dlabel, args.roi_a, args.roi_b)

    log.info(f"\nBuilding {args.roi_a} subsurface …")
    build_subsurface(args.roi_a, masks[args.roi_a][0], masks[args.roi_a][1],
                     cache_dir, surf_L, surf_R)

    log.info(f"\nBuilding {args.roi_b} subsurface …")
    build_subsurface(args.roi_b, masks[args.roi_b][0], masks[args.roi_b][1],
                     cache_dir, surf_L, surf_R)

    log.info(f"\nDone. Subsurfaces cached to: {cache_dir}")
    log.info(f"Next: python run_cfmodeling.py --mode {args.mode} "
             f"--roi-a {args.roi_a} --roi-b {args.roi_b}")


if __name__ == "__main__":
    main()
