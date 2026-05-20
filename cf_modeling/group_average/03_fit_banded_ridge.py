"""
03_fit_banded_ridge.py
======================
Phase 3: Load the pre-processed arrays from Script 2, fit a himalaya
MultipleKernelRidgeCV (banded ridge) model, evaluate on the held-out test
set, compute variance partitioning maps, and export CIFTI dscalar maps.

Variance decomposition
----------------------
For two bands (ROI_A, ROI_B):
    Y_hat_full  = pipeline.predict(X_test)              → full model
    Y_hat_split = pipeline.predict(X_test, split=True)  → per-band predictions
    R2_full  = r2_score(Y_test, Y_hat_full)             → (59412,)
    [R2_A, R2_B] = r2_score_split(Y_test, Y_hat_split)  → per-band contributions
    Shared_R2 = R2_A + R2_B - R2_full

Inputs (from Script 2):
    PREP_DIR/X_train.npy, Y_train.npy, X_test.npy, Y_test.npy,
    run_onsets.npy, band_sizes.npy

Outputs:
    OUTPUT_DIR/R2_full.npy, R2_{ROI_A}.npy, R2_{ROI_B}.npy, Shared_R2.npy
    CIFTI_DIR/R2_full.dscalar.nii, R2_{ROI_A}.dscalar.nii,
              R2_{ROI_B}.dscalar.nii, Shared_R2.dscalar.nii

Run
---
    conda activate vicsompy_av
    python 03_fit_banded_ridge.py \\
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

ROI_A         = _DEFAULT_ROI_A
ROI_B         = _DEFAULT_ROI_B
OUTPUT_BASE   = _DEFAULT_OUTPUT_BASE
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
from himalaya.scoring import r2_score, r2_score_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.ridge_utils import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def be_to_npy(var, backend):
    """Convert a backend tensor (or list of tensors) to numpy."""
    if isinstance(var, list):
        return [backend.to_numpy(v) for v in var]
    return backend.to_numpy(var)


def save_map(arr, name, template):
    """Save (59412,) array as .npy and .dscalar.nii."""
    arr_f32 = arr.astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr_f32)
    cifti = nib.Cifti2Image(arr_f32.reshape(1, -1),
                            header=template.header,
                            nifti_header=template.nifti_header)
    nib.save(cifti, os.path.join(CIFTI_DIR, f"{name}.dscalar.nii"))
    log.info(f"  {name}: mean={arr.mean():.4f}  frac>0.01={np.mean(arr>0.01):.1%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(CIFTI_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Script 03 — Fit banded ridge (group average)")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info("=" * 60)

    for fname in ["X_train", "Y_train", "X_test", "Y_test", "run_onsets", "band_sizes"]:
        path = os.path.join(PREP_DIR, f"{fname}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run 02_prep_hcp_timeseries.py first.")

    log.info(f"\nLoading pre-processed arrays from {PREP_DIR} …")
    X_train    = np.load(os.path.join(PREP_DIR, "X_train.npy"))
    Y_train    = np.load(os.path.join(PREP_DIR, "Y_train.npy"))
    X_test     = np.load(os.path.join(PREP_DIR, "X_test.npy"))
    Y_test     = np.load(os.path.join(PREP_DIR, "Y_test.npy"))
    run_onsets = np.load(os.path.join(PREP_DIR, "run_onsets.npy"))
    band_sizes = np.load(os.path.join(PREP_DIR, "band_sizes.npy"))

    log.info(f"  X_train={X_train.shape}  Y_train={Y_train.shape}")
    log.info(f"  X_test ={X_test.shape}   Y_test ={Y_test.shape}")
    log.info(f"  run_onsets={run_onsets.tolist()}")
    log.info(f"  band_sizes: {ROI_A}={band_sizes[0]}  {ROI_B}={band_sizes[1]}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline, backend = build_pipeline(
        n_samples_train=X_train.shape[0],
        run_onsets=run_onsets,
        band_sizes=band_sizes.tolist(),
        roi_names=[ROI_A.lower(), ROI_B.lower()],
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    log.info(f"\nFitting …")
    pipeline.fit(X_train, Y_train)
    log.info("  Fit complete.")

    # ── Full-model R² ─────────────────────────────────────────────────────────
    log.info("Computing full-model R² on test set …")
    Y_hat_full = pipeline.predict(X_test)
    R2_full    = be_to_npy(r2_score(Y_test, Y_hat_full), backend)

    # ── Per-band R² ───────────────────────────────────────────────────────────
    log.info("Computing split (per-band) R² on test set …")
    Y_hat_split  = pipeline.predict(X_test, split=True)
    split_scores = be_to_npy(r2_score_split(Y_test, Y_hat_split), backend)
    R2_a   = split_scores[0]
    R2_b   = split_scores[1]
    Shared = R2_a + R2_b - R2_full

    log.info("\nVariance partitioning — test set means:")
    log.info(f"  R2_full  : {np.nanmean(R2_full):.4f}  "
             f"(>0.01: {np.mean(R2_full > 0.01):.1%})")
    log.info(f"  R2_{ROI_A}: {np.nanmean(R2_a):.4f}")
    log.info(f"  R2_{ROI_B}: {np.nanmean(R2_b):.4f}")
    log.info(f"  Shared   : {np.nanmean(Shared):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"\nSaving outputs to {OUTPUT_DIR} …")
    template = nib.load(TEMPLATE_CIFTI)
    save_map(R2_full,  "R2_full",         template)
    save_map(R2_a,     f"R2_{ROI_A}",     template)
    save_map(R2_b,     f"R2_{ROI_B}",     template)
    save_map(Shared,   "Shared_R2",       template)

    log.info("\nScript 03 complete.")
    log.info(f"Next: python 04_null_corrected_maps.py --roi_a {ROI_A} --roi_b {ROI_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit banded ridge regression on group-average HCP data."
    )
    parser.add_argument("--roi_a",          default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",          default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base",    default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--template_cifti", default=_DEFAULT_TEMPLATE,
                        help="32k template dscalar.nii for CIFTI header (default: %(default)s)")
    args = parser.parse_args()

    ROI_A          = args.roi_a
    ROI_B          = args.roi_b
    OUTPUT_BASE    = args.output_base
    TEMPLATE_CIFTI = args.template_cifti
    OUTPUT_DIR     = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
    PREP_DIR       = f"{OUTPUT_DIR}/prep"
    CIFTI_DIR      = f"{OUTPUT_DIR}/cifti_maps"

    main()
