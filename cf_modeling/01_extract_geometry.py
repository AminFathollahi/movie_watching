"""
cf_modeling/01_extract_geometry.py
=====================================
Build vicsompy Subsurface objects and compute Laplace-Beltrami Operator
Eigenfunctions (LBOEs) for two Glasser HCP-MMP1 ROIs.

This script replaces the vendored approach: the Subsurface class is now
imported DIRECTLY from the vicsompy source repository (no installation needed).
Set VICSOMPY_REPO below (or via the --vicsompy-repo CLI flag) to the local
clone of https://github.com/nicholashedger/vicsompy.

Scientific alignment with Hedger et al. (2025)
-----------------------------------------------
* The Laplace-Beltrami decomposition and LBOE projection are taken verbatim
  from vicsompy.surface.Subsurface (MIT License).
* The Laplacian is computed on the FIDUCIAL (midthickness) surface from the
  pycortex database, which is the closest available surface in our pycortex
  subject (hcp_999999_draw_NH).  Hedger et al. used SPHERE surfaces from
  hcp_999999; sphere surfaces are not present in our subject.  Both choices
  are scientifically valid; the LBOEs capture the intrinsic geometry of the
  cortex on the chosen surface.
* A sigma=1e-6 shift is applied in laplacian_decomposition() to regularise the
  Laplacian near the zero eigenvalue (StableSubsurface subclass).  Without this
  shift, scipy's SpLU may fail for small ROIs.
* ROI vertex masks are loaded from the 59k_fs_LR Glasser dlabel.nii.
* Geodesic distance matrices (needed for splicing) are NOT computed; we call
  get_surfaces() → generate() → make_laplacians() only, skipping get_geometry()
  and pad_distance_matrices().  This saves hours of computation.

Output
------
Per ROI pair, in {output_base}/{mode}/{ROI_A}_{ROI_B}/subsurfaces/:
  sub_{roi_lower}.pkl              — Subsurface pickle (our naming convention)
  {roi_lower}_subsurface.pickle   — identical copy in vicsompy naming format
                                     (required by MssCf.make_subsurfaces(force_new=False))

Both files are pickled with class = vicsompy's Subsurface, so they can be
loaded by vicsompy directly without modification.

Usage
-----
  conda activate movie
  python cf_modeling/01_extract_geometry.py \\
      --mode group_average --roi-a 3b --roi-b V1

  python cf_modeling/01_extract_geometry.py \\
      --mode per_subject --roi-a A5 --roi-b FFC
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy as sp

# ---------------------------------------------------------------------------
# Vicsompy import (direct from source repo, no installation)
# ---------------------------------------------------------------------------
VICSOMPY_REPO = os.environ.get(
    "VICSOMPY_REPO",
    "/home/amin/Research/Representation/Movie/Vicarious_somatotopy",
)
if VICSOMPY_REPO not in sys.path:
    sys.path.insert(0, VICSOMPY_REPO)

from vicsompy.surface import Subsurface  # noqa: E402 (after sys.path setup)

# ---------------------------------------------------------------------------
# Pycortex import (must be after sys.path to avoid stale module references)
# ---------------------------------------------------------------------------
import cortex           # noqa: E402
import cortex.database  # noqa: E402
import cortex.polyutils  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
N_LBOE        = 200
CX_SUB        = "hcp_999999_draw_NH"
SURF_TYPE     = "fiducial"    # 'fiducial' = midthickness in pycortex; sphere absent
N_VERTS_PER_HEM = 59292

_DATA_BASE      = "/home/amin/Research/Representation/Movie/data/Setareh"
_HCP_DIR        = f"{_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_OUT_BASE       = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"
_PYCORTEX_STORE = "/home/amin/Research/Representation/Movie/data/hedger2026"
_GLASSER_DLABEL = (
    f"/home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/"
    "Q1-Q6_RelatedParcellation210"
    ".CorticalAreas_dil_Final_Final_Areas_Group_Colors"
    ".59k_fs_LR.dlabel.nii"
)


# =============================================================================
# StableSubsurface — vicsompy Subsurface with Laplacian regularisation
# =============================================================================

class StableSubsurface(Subsurface):
    """Extends vicsompy's Subsurface with a sigma=1e-6 shift in the Laplacian.

    Without regularisation, scipy's eigs() with sigma=0 can fail (singular
    SpLU) for small ROIs or on certain surface representations.  A tiny shift
    sigma=1e-6 avoids this without meaningfully changing the eigenvectors.

    Everything else is inherited verbatim from vicsompy.surface.Subsurface
    (Hedger et al. 2025, MIT License).
    """

    def laplacian_decomposition(self, subsurface, n_components: int):
        """Compute n_components eigenvectors of the graph Laplacian.

        Identical to Subsurface.laplacian_decomposition() except sigma=1e-6
        instead of sigma=0 for numerical stability.

        Parameters
        ----------
        subsurface   : pycortex polyutils Subsurface object (cortex.polyutils)
        n_components : int — number of eigenvectors to compute.

        Returns
        -------
        eigenvalues  : (n_components,) complex
        eigenvectors : (n_verts_roi, n_components) complex
        """
        B, D, W, V = subsurface.laplace_operator
        npt  = W.shape[0]
        Dinv = sp.sparse.dia_matrix((D ** -1, [0]), (npt, npt)).tocsr()
        L    = Dinv.dot(V - W)
        eigenvalues, eigenvectors = sp.sparse.linalg.eigs(
            -L, k=n_components, which="LM", sigma=1e-6
        )
        return eigenvalues, eigenvectors


# =============================================================================
# ROI mask loading from Glasser dlabel
# =============================================================================

def _load_csv_masks(roi: str, masks_dir: str) -> tuple:
    """Load {roi}_{L/R}_mask.csv from masks_dir (vicsompy format).

    Returns
    -------
    mask_L : (59292,) bool
    mask_R : (59292,) bool
    """
    for hem in ("L", "R"):
        path = os.path.join(masks_dir, f"{roi}_{hem}_mask.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CSV mask not found: {path}\n"
                f"Run 03_functional_masks.py (or 00_make_roi_masks.py) first, "
                f"or check --masks-dir."
            )
    mask_L = pd.read_csv(os.path.join(masks_dir, f"{roi}_L_mask.csv"))["mask"].to_numpy(dtype=bool)
    mask_R = pd.read_csv(os.path.join(masks_dir, f"{roi}_R_mask.csv"))["mask"].to_numpy(dtype=bool)
    log.info("  %-20s  L=%4d verts (CSV)  R=%4d verts (CSV)",
             roi, int(mask_L.sum()), int(mask_R.sum()))
    return mask_L, mask_R


def load_dlabel_masks(glasser_dlabel: str, roi_a: str, roi_b: str,
                      masks_dir: str = None) -> dict:
    """Load Boolean vertex masks from the 59k_fs_LR Glasser dlabel.

    If an ROI name is not found in the dlabel (e.g. a functional ROI like
    'auditory_cx'), falls back to loading {roi}_{L/R}_mask.csv from masks_dir.
    Raises FileNotFoundError if the CSV is also missing.

    Parameters
    ----------
    glasser_dlabel : str — path to the dlabel.nii file.
    roi_a, roi_b   : str — ROI names (Glasser short names, or functional names
                           with CSV fallback).
    masks_dir      : str | None — directory of CSV masks for fallback.

    Returns
    -------
    masks : dict[roi_name, (mask_L, mask_R)]
        mask_L : (59292,) bool
        mask_R : (59292,) bool
    """
    log.info("Loading Glasser dlabel: %s", os.path.basename(glasser_dlabel))
    img    = nib.load(glasser_dlabel)
    data   = img.get_fdata(dtype=np.float32)[0]     # (118584,)
    ax0    = img.header.get_axis(0)
    n2k    = {name: key for key, (name, _) in ax0.label[0].items()}

    masks = {}
    for roi in [roi_a, roi_b]:
        lk = n2k.get(f"L_{roi}_ROI")
        rk = n2k.get(f"R_{roi}_ROI")

        if lk is None or rk is None:
            if masks_dir is not None:
                log.info(
                    "  '%s' not found in dlabel — loading CSV from %s",
                    roi, masks_dir,
                )
                masks[roi] = _load_csv_masks(roi, masks_dir)
            else:
                available = sorted(
                    n.replace("L_", "").replace("_ROI", "")
                    for n in n2k
                    if n.startswith("L_") and n.endswith("_ROI")
                )
                raise ValueError(
                    f"ROI '{roi}' not found in dlabel and --masks-dir not set. "
                    f"Available Glasser names:\n  {available}"
                )
            continue

        mask_L = (data[:N_VERTS_PER_HEM] == lk)
        mask_R = (data[N_VERTS_PER_HEM:] == rk)
        masks[roi] = (mask_L, mask_R)
        log.info(
            "  %-6s  L=%4d (label=%3d)  R=%4d (label=%3d)",
            roi, int(mask_L.sum()), lk, int(mask_R.sum()), rk,
        )
    return masks


# =============================================================================
# Subsurface builder
# =============================================================================

def build_subsurface(
    name: str,
    mask_L: np.ndarray,
    mask_R: np.ndarray,
    cache_dir: str,
    cx_sub: str = CX_SUB,
    surf_type: str = SURF_TYPE,
    n_lboe_max: int = N_LBOE,
) -> Subsurface:
    """Build (or load from cache) a StableSubsurface with LBOEs.

    The subsurface is built WITHOUT computing geodesic distances
    (get_geometry() / pad_distance_matrices() are skipped).  This reduces
    build time from hours to minutes.

    Two cache files are written:
      sub_{name.lower()}.pkl            — our internal naming convention
      {name.lower()}_subsurface.pickle  — vicsompy naming convention
                                          (for MssCf.make_subsurfaces(force_new=False))

    Parameters
    ----------
    name       : str — ROI short name (e.g. '3b', 'V1').
    mask_L     : (59292,) bool — left-hemisphere vertex mask.
    mask_R     : (59292,) bool — right-hemisphere vertex mask.
    cache_dir  : str — directory to cache pkl files.
    cx_sub     : str — pycortex subject name.
    surf_type  : str — pycortex surface type ('fiducial', 'sphere', …).
    n_lboe_max : int — maximum LBOEs; capped to min(n_lboe_max, n_L-2, n_R-2).

    Returns
    -------
    sub : StableSubsurface (class = Subsurface from vicsompy, for pickle compat)
    """
    our_path     = os.path.join(cache_dir, f"sub_{name.lower()}.pkl")
    vicsompy_path = os.path.join(cache_dir, f"{name.lower()}_subsurface.pickle")

    # ── Load from cache if both files exist ───────────────────────────────
    if os.path.exists(our_path) and os.path.exists(vicsompy_path):
        log.info("  [%s] Loading cached subsurface from: %s", name, our_path)
        with open(our_path, "rb") as fh:
            sub = pickle.load(fh)
        # Patch missing attributes added after original save
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
        if not hasattr(sub, "subsurface_verts"):
            sub.subsurface_verts = np.concatenate([
                sub.subsurface_verts_L, sub.subsurface_verts_R])
        log.info(
            "  [%s]  L_eig=%s  R_eig=%s  n_lboe=%d",
            name, sub.L_eigenvectors.shape, sub.R_eigenvectors.shape, sub.n_lboe,
        )
        return sub

    # ── Build from scratch ────────────────────────────────────────────────
    log.info("  [%s] Building subsurface from pycortex DB (%s, %s) …",
             name, cx_sub, surf_type)
    sub = StableSubsurface(cx_sub=cx_sub, boolmasks=[mask_L, mask_R],
                           surftype=surf_type)
    sub.get_surfaces()   # loads fiducial surfaces from pycortex filestore
    sub.generate()       # creates subsurface_L/R, sets subsurface_verts_L/R
    # NOTE: get_geometry() and pad_distance_matrices() are intentionally skipped.
    # They compute geodesic distance matrices (n_verts × n_verts) which take
    # hours and are only needed for CF profile splicing, which we do not do.

    n_L = len(sub.subsurface_verts_L)
    # subsurface_verts_R contains indices in full bilateral sphere space, so the
    # right hemisphere indices are offset by +N_VERTS_PER_HEM.  The actual vertex
    # count for the right ROI is therefore len(...) not len(...) - N_VERTS_PER_HEM.
    n_R = len(sub.subsurface_verts_R)
    log.info("  [%s]  L verts: %d   R verts: %d", name, n_L, n_R)

    # Cap n_lboe to be safely below the number of vertices in each hemisphere.
    n_lboe = min(n_lboe_max, n_L - 2, n_R - 2)
    if n_lboe < n_lboe_max:
        log.warning(
            "  [%s] Capping LBOEs to %d (ROI has only %d/%d verts; need n_lboe < n_verts).",
            name, n_lboe, n_L, n_R,
        )
    if n_lboe < 2:
        raise ValueError(
            f"ROI '{name}' has too few vertices (L={n_L}, R={len(sub.subsurface_verts_R)}) "
            f"to compute at least 2 LBOEs.  Enlarge the ROI or reduce --n-lboe."
        )

    log.info("  [%s] Computing %d LBOEs …", name, n_lboe)
    sub.make_laplacians(n_lboe)
    sub.n_lboe = n_lboe

    # Store bilateral sphere-space vertex indices for null-model mean computation.
    # subsurface_verts_R already includes the +59292 L-hemisphere offset.
    sub.subsurface_verts = np.concatenate([
        sub.subsurface_verts_L, sub.subsurface_verts_R
    ])
    log.info(
        "  [%s]  L_eig=%s  R_eig=%s  n_lboe=%d",
        name, sub.L_eigenvectors.shape, sub.R_eigenvectors.shape, sub.n_lboe,
    )

    # ── Save as base Subsurface class (no StableSubsurface in pickle) ────
    # Downcast to Subsurface so any script that imports vicsompy.surface.Subsurface
    # can load the pickle without needing StableSubsurface in scope.
    sub.__class__ = Subsurface
    sub.save(our_path)             # uses Subsurface.save() = remove_surfaces + pickle
    sub.__class__ = StableSubsurface  # restore in-memory

    # Create the vicsompy-convention symlink/copy for MssCf.make_subsurfaces(force_new=False)
    import shutil
    shutil.copy2(our_path, vicsompy_path)

    log.info("  [%s] Saved: %s", name, our_path)
    log.info("  [%s] Saved: %s", name, vicsompy_path)
    return sub


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build Subsurfaces and LBOEs for two Glasser ROIs.\n"
            "Imports Subsurface directly from the vicsompy source repo "
            "(no installation required)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi-a", default="3b",   dest="roi_a")
    p.add_argument("--roi-b", default="V1",   dest="roi_b")
    p.add_argument("--n-lboe", type=int, default=N_LBOE, dest="n_lboe",
                   help="Maximum LBOEs per ROI (capped to n_verts-2 if smaller).")
    p.add_argument("--pycortex-store", default=_PYCORTEX_STORE, dest="pycortex_store",
                   help="Pycortex filestore root directory.")
    p.add_argument("--cx-sub", default=CX_SUB, dest="cx_sub",
                   help="Pycortex subject name.")
    p.add_argument("--surf-type", default=SURF_TYPE, dest="surf_type",
                   help="Pycortex surface type ('fiducial', 'sphere', …).")
    p.add_argument("--glasser-dlabel", default=_GLASSER_DLABEL, dest="glasser_dlabel",
                   help="59k_fs_LR Glasser HCP-MMP1 dlabel.nii.")
    p.add_argument("--masks-dir", default=None, dest="masks_dir",
                   help="Directory of CSV mask files (fallback when ROI name is not "
                        "found in the Glasser dlabel, e.g. for functional ROIs).")
    p.add_argument("--output-base", default=_OUT_BASE, dest="output_base",
                   help="CF modeling output root directory.")
    p.add_argument("--vicsompy-repo", default=VICSOMPY_REPO, dest="vicsompy_repo",
                   help="Path to local vicsompy repository (source import).")
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure vicsompy repo is on sys.path (may differ from module-level default)
    if args.vicsompy_repo not in sys.path:
        sys.path.insert(0, args.vicsompy_repo)

    # Initialise pycortex filestore
    os.environ["PYCORTEX_FILESTORE"] = args.pycortex_store
    cortex.database.default_filestore = args.pycortex_store
    cortex.db = cortex.database.Database(args.pycortex_store)

    roi_dir   = os.path.join(args.output_base, args.mode,
                             f"{args.roi_a}_{args.roi_b}")
    cache_dir = os.path.join(roi_dir, "subsurfaces")
    os.makedirs(cache_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("01 — Extract geometry & LBOEs")
    log.info("  Mode       : %s", args.mode)
    log.info("  ROIs       : %s × %s", args.roi_a, args.roi_b)
    log.info("  Max LBOEs  : %d", args.n_lboe)
    log.info("  Surface    : %s (%s)", args.surf_type, args.cx_sub)
    log.info("  Vicsompy   : %s", args.vicsompy_repo)
    log.info("  Cache dir  : %s", cache_dir)
    if args.masks_dir:
        log.info("  Masks dir  : %s (CSV fallback enabled)", args.masks_dir)
    log.info("=" * 60)

    masks = load_dlabel_masks(args.glasser_dlabel, args.roi_a, args.roi_b,
                               masks_dir=args.masks_dir)

    log.info("\nBuilding '%s' subsurface …", args.roi_a)
    build_subsurface(
        args.roi_a,
        masks[args.roi_a][0], masks[args.roi_a][1],
        cache_dir,
        cx_sub=args.cx_sub, surf_type=args.surf_type, n_lboe_max=args.n_lboe,
    )

    log.info("\nBuilding '%s' subsurface …", args.roi_b)
    build_subsurface(
        args.roi_b,
        masks[args.roi_b][0], masks[args.roi_b][1],
        cache_dir,
        cx_sub=args.cx_sub, surf_type=args.surf_type, n_lboe_max=args.n_lboe,
    )

    log.info("\nDone. Subsurfaces cached to: %s", cache_dir)
    log.info("Next: python cf_modeling/02_fit_cf_model.py --mode %s "
             "--roi-a %s --roi-b %s", args.mode, args.roi_a, args.roi_b)


if __name__ == "__main__":
    main()
