"""
04_null_corrected_maps.py
=========================
Null-model correction mirroring Hedger et al. (2025) Figure 3a methodology.

Subtracts the R² of a single-regressor OLS null model (mean ROI test timecourse)
from the banded-ridge split R² to isolate topographic specificity:

    R2_{ROI_A}_nc = R2_{ROI_A} - R2_null_{ROI_A}
    R2_{ROI_B}_nc = R2_{ROI_B} - R2_null_{ROI_B}

Inputs (from Scripts 02-03):
    PREP_DIR/Y_test.npy
    PREP_DIR/mean_{ROI_A}_test.npy, mean_{ROI_B}_test.npy  (Script 02)
    OUTPUT_DIR/R2_{ROI_A}.npy, R2_{ROI_B}.npy              (Script 03)

Outputs:
    OUTPUT_DIR/R2_null_{ROI_A}.{npy,dscalar.nii}
    OUTPUT_DIR/R2_null_{ROI_B}.{npy,dscalar.nii}
    OUTPUT_DIR/R2_{ROI_A}_nc.{npy,dscalar.nii}   ← primary output
    OUTPUT_DIR/R2_{ROI_B}_nc.{npy,dscalar.nii}   ← primary output

Run
---
    conda activate vicsompy_av
    python 04_null_corrected_maps.py \\
        --roi_a A1 --roi_b V1 \\
        --output_base /path/to/outputs \\
        --template_cifti /path/to/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os
import sys

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
PREP_DIR   = f"{OUTPUT_DIR}/prep"
CIFTI_DIR  = f"{OUTPUT_DIR}/cifti_maps"

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.ridge_utils import fit_null_r2

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
             f"frac>0={np.mean(arr > 0):.1%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(CIFTI_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Script 04 — Null-corrected R² maps (Hedger et al. Fig 3a method)")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info("=" * 60)

    log.info("\nLoading test arrays …")
    Y_test   = np.load(os.path.join(PREP_DIR, "Y_test.npy"))
    R2_a     = np.load(os.path.join(OUTPUT_DIR, f"R2_{ROI_A}.npy"))
    R2_b     = np.load(os.path.join(OUTPUT_DIR, f"R2_{ROI_B}.npy"))
    mean_a   = np.load(os.path.join(PREP_DIR, f"mean_{ROI_A}_test.npy"))
    mean_b   = np.load(os.path.join(PREP_DIR, f"mean_{ROI_B}_test.npy"))

    log.info(f"  Y_test: {Y_test.shape}")
    log.info(f"  R2_{ROI_A}: mean={R2_a.mean():.4f}  frac>0={np.mean(R2_a>0):.1%}")
    log.info(f"  R2_{ROI_B}: mean={R2_b.mean():.4f}  frac>0={np.mean(R2_b>0):.1%}")

    # ── OLS null models ───────────────────────────────────────────────────────
    # fit_null_r2(regressor, Y): Y is (T, n_targets), regressor is (T,)
    log.info(f"\nFitting null model for {ROI_A} …")
    R2_null_a = fit_null_r2(mean_a, Y_test)

    log.info(f"Fitting null model for {ROI_B} …")
    R2_null_b = fit_null_r2(mean_b, Y_test)

    log.info(f"  R2_null_{ROI_A}: mean={R2_null_a.mean():.4f}  frac>0={np.mean(R2_null_a>0):.1%}")
    log.info(f"  R2_null_{ROI_B}: mean={R2_null_b.mean():.4f}  frac>0={np.mean(R2_null_b>0):.1%}")

    # ── Null-corrected maps ───────────────────────────────────────────────────
    R2_a_nc = (R2_a - R2_null_a).astype(np.float32)
    R2_b_nc = (R2_b - R2_null_b).astype(np.float32)
    log.info(f"\n  R2_{ROI_A}_nc: mean={R2_a_nc.mean():.4f}  frac>0={np.mean(R2_a_nc>0):.1%}")
    log.info(f"  R2_{ROI_B}_nc: mean={R2_b_nc.mean():.4f}  frac>0={np.mean(R2_b_nc>0):.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"\nSaving maps to {OUTPUT_DIR} …")
    template = nib.load(TEMPLATE_CIFTI)
    save_map(R2_null_a, f"R2_null_{ROI_A}", template)
    save_map(R2_null_b, f"R2_null_{ROI_B}", template)
    save_map(R2_a_nc,   f"R2_{ROI_A}_nc",   template)
    save_map(R2_b_nc,   f"R2_{ROI_B}_nc",   template)

    log.info("\nScript 04 complete.")
    log.info(f"Next: python 05_integration_maps.py --roi_a {ROI_A} --roi_b {ROI_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Null-model correction of split R² maps (Hedger et al. Fig 3a)."
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
    PREP_DIR       = f"{OUTPUT_DIR}/prep"
    CIFTI_DIR      = f"{OUTPUT_DIR}/cifti_maps"

    main()
