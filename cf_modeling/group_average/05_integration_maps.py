"""
05_integration_maps.py
======================
Derives the Figure 3a display maps from the null-corrected split R² produced
by Script 04. No model fitting — pure map arithmetic.

Figure 3a in Hedger et al. (2025) shows, per vertex:
    X-axis = null-corrected ROI_B R²   (blue)
    Y-axis = null-corrected ROI_A R²   (red)
    Both high = purple                  → bimodal / audiovisual integration zone
    Neither   = transparent

TWO WAYS to view this in HCP Workbench (wb_view):

A) Continuous 2D dual-overlay (matches Figure 3a exactly):
   Load both dscalar files as separate overlays:
     Overlay 1: R2_{ROI_B}_nc.dscalar.nii  → colormap "Blues", min transparent
     Overlay 2: R2_{ROI_A}_nc.dscalar.nii  → colormap "Reds",  min transparent

B) Discrete 4-category dlabel (bimodal_map.dlabel.nii, this script):
   Opens as a label file in wb_view with fixed RGBA colors per category.

C) Python / pycortex: see the visualization notebook in this directory.

Outputs (all in OUTPUT_DIR or CIFTI_DIR):
    integration_score.{npy,dscalar.nii}   — √(R2_A_nc × R2_B_nc), high = bimodal
    modality_balance.{npy,dscalar.nii}    — R2_A_nc − R2_B_nc (diverging)
    top_{N}_percentile_integration.{npy,dscalar.nii}  — binary top-N% mask
    bimodal_map.dlabel.nii                — 4-category label map for wb_view

Run
---
    conda activate vicsompy_av
    python 05_integration_maps.py \\
        --roi_a A1 --roi_b V1 \\
        --output_base /path/to/outputs \\
        --template_cifti /path/to/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os

_DEFAULT_DATA_BASE   = "/home/amin/Research/Representation/Movie/data/Setareh"
_DEFAULT_ROI_A       = "A1"
_DEFAULT_ROI_B       = "V1"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/group_average"
_DEFAULT_TEMPLATE    = (
    f"{_DEFAULT_DATA_BASE}/HCP_S1200_GroupAvg_v1/"
    "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
)

ROI_A          = _DEFAULT_ROI_A
ROI_B          = _DEFAULT_ROI_B
OUTPUT_BASE    = _DEFAULT_OUTPUT_BASE
TEMPLATE_CIFTI = _DEFAULT_TEMPLATE

OUTPUT_DIR = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
CIFTI_DIR  = f"{OUTPUT_DIR}/cifti_maps"

MASK_PERCENTILE  = 90    # top 10% of vertices as integration mask
DLABEL_THRESHOLD = 0.0   # threshold for bimodal dlabel (0.0 = simply positive)

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging

import numpy as np
import nibabel as nib
from nibabel.cifti2.cifti2_axes import LabelAxis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def save_map(arr, name, template):
    """Save (59412,) array as .npy and .dscalar.nii."""
    arr_f32 = arr.astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr_f32)
    cifti = nib.Cifti2Image(arr_f32.reshape(1, -1),
                            header=template.header,
                            nifti_header=template.nifti_header)
    nib.save(cifti, os.path.join(CIFTI_DIR, f"{name}.dscalar.nii"))
    log.info(f"  Saved {name}: min={arr.min():.4f}  max={arr.max():.4f}  "
             f"mean={arr.mean():.4f}  frac>0={np.mean(arr>0):.1%}")


def square_(a, b):
    """√(clip(a,0) × clip(b,0)) — high only where both are positive."""
    return np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)).astype(np.float32)


def save_dlabel(label_arr, name, template):
    """Save integer label array (59412,) as a CIFTI dlabel file for wb_view."""
    bm_axis = template.header.get_axis(1)
    lt = {
        0: ('??',             (0.5, 0.5, 0.5, 0.0)),
        1: (f'{ROI_A}_only',  (0.8, 0.1, 0.1, 1.0)),
        2: (f'{ROI_B}_only',  (0.1, 0.1, 0.8, 1.0)),
        3: ('bimodal',        (0.6, 0.0, 0.8, 1.0)),
    }
    label_col    = np.empty(1, dtype=object)
    label_col[0] = lt
    la     = LabelAxis(name=np.array([name]), label=label_col)
    header = nib.Cifti2Header.from_axes((la, bm_axis))
    img    = nib.Cifti2Image(label_arr.reshape(1, -1).astype(np.float32), header=header)
    img.nifti_header["intent_code"] = 3007
    path = os.path.join(CIFTI_DIR, f"{name}.dlabel.nii")
    nib.save(img, path)
    counts = {k: int(np.sum(label_arr == k)) for k in range(4)}
    log.info(f"  Saved {name}.dlabel.nii — "
             f"neither={counts[0]}  {ROI_A}_only={counts[1]}  "
             f"{ROI_B}_only={counts[2]}  bimodal={counts[3]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(CIFTI_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Script 05 — Integration maps")
    log.info(f"  ROI A : {ROI_A}  ROI B : {ROI_B}")
    log.info(f"  dlabel threshold : {DLABEL_THRESHOLD}")
    log.info("=" * 60)

    log.info("\nLoading null-corrected R² maps …")
    R2_a_nc = np.load(os.path.join(OUTPUT_DIR, f"R2_{ROI_A}_nc.npy"))
    R2_b_nc = np.load(os.path.join(OUTPUT_DIR, f"R2_{ROI_B}_nc.npy"))
    log.info(f"  R2_{ROI_A}_nc: mean={R2_a_nc.mean():.4f}  frac>0={np.mean(R2_a_nc>0):.1%}")
    log.info(f"  R2_{ROI_B}_nc: mean={R2_b_nc.mean():.4f}  frac>0={np.mean(R2_b_nc>0):.1%}")

    template = nib.load(TEMPLATE_CIFTI)

    # ── Integration score: √(R2_A_nc × R2_B_nc) ─────────────────────────────
    integration_score = square_(R2_a_nc, R2_b_nc)
    save_map(integration_score, "integration_score", template)

    # ── Modality balance ──────────────────────────────────────────────────────
    modality_balance = (R2_a_nc - R2_b_nc).astype(np.float32)
    save_map(modality_balance, "modality_balance", template)

    # ── Integration mask: top-N% vertices ────────────────────────────────────
    threshold = np.percentile(integration_score, MASK_PERCENTILE)
    integration_mask = (integration_score >= threshold).astype(np.float32)
    log.info(f"\n  Integration mask: threshold={threshold:.5f}  "
             f"n_verts={int(integration_mask.sum())}  "
             f"({100*(1-MASK_PERCENTILE/100):.0f}% of all vertices)")
    save_map(integration_mask, f"top_{MASK_PERCENTILE}_percentile_integration", template)

    # ── Bimodal dlabel ────────────────────────────────────────────────────────
    log.info("\nBuilding bimodal dlabel map …")
    a_hot = R2_a_nc > DLABEL_THRESHOLD
    b_hot = R2_b_nc > DLABEL_THRESHOLD
    label_arr = np.zeros(59412, dtype=np.int32)
    label_arr[a_hot & ~b_hot] = 1
    label_arr[~a_hot & b_hot] = 2
    label_arr[a_hot  &  b_hot] = 3
    save_dlabel(label_arr, "bimodal_map", template)

    p95_a = np.percentile(R2_a_nc[R2_a_nc > 0], 95) if (R2_a_nc > 0).any() else 0
    p95_b = np.percentile(R2_b_nc[R2_b_nc > 0], 95) if (R2_b_nc > 0).any() else 0
    log.info("\n--- wb_view: continuous 2D dual-overlay (Figure 3a equivalent) ---")
    log.info(f"  Overlay 1 (blue): R2_{ROI_B}_nc.dscalar.nii  colormap=Blues  "
             f"[0, {p95_b:.4f}]  min=transparent")
    log.info(f"  Overlay 2 (red):  R2_{ROI_A}_nc.dscalar.nii  colormap=Reds   "
             f"[0, {p95_a:.4f}]  min=transparent")
    log.info("--- wb_view: discrete 4-color (dlabel) ---")
    log.info(f"  File: {CIFTI_DIR}/bimodal_map.dlabel.nii")

    log.info("\nScript 05 complete.")
    log.info(f"Next: python 06_rsa_overlap.py --roi_a {ROI_A} --roi_b {ROI_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute integration maps from null-corrected R² (Figure 3a)."
    )
    parser.add_argument("--roi_a",          default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",          default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base",    default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--template_cifti", default=_DEFAULT_TEMPLATE,
                        help="32k template dscalar.nii (default: %(default)s)")
    args = parser.parse_args()

    ROI_A          = args.roi_a
    ROI_B          = args.roi_b
    OUTPUT_BASE    = args.output_base
    TEMPLATE_CIFTI = args.template_cifti
    OUTPUT_DIR     = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
    CIFTI_DIR      = f"{OUTPUT_DIR}/cifti_maps"

    main()
