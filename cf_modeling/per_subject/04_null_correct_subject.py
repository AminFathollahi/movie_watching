"""
04_null_correct_subject.py
==========================
Per-subject Phase 4: null-model correction (Hedger et al. 2025 Fig 3a method).

Subtracts the R² of a single-regressor OLS null model (mean ROI test timecourse)
from the banded-ridge split R² to isolate topographic specificity:

    R2_{roi}_nc = R2_{roi} - R2_null_{roi}

The null model uses precomputed mean ROI test timecourses saved by script 02.
No subsurface loading or pipeline reconstruction required.

Inputs (from scripts 02 and 03):
    subjects/<sub>/Y_test.npy
    subjects/<sub>/R2_<ROI_A>.npy, R2_<ROI_B>.npy
    subjects/<sub>/mean_<ROI_A>_test.npy, mean_<ROI_B>_test.npy

Outputs:
    subjects/<sub>/R2_null_<ROI_A>.npy, R2_null_<ROI_B>.npy
    subjects/<sub>/R2_<ROI_A>_nc.npy, R2_<ROI_B>_nc.npy

Run
---
    conda activate vicsompy_av
    python 04_null_correct_subject.py --subject 100610 \\
        --roi_a A5 --roi_b FFC \\
        --output_base /path/to/outputs
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os
import sys

_DEFAULT_ROI_A       = "A5"
_DEFAULT_ROI_B       = "FFC"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/per_subject"

ROI_A       = _DEFAULT_ROI_A
ROI_B       = _DEFAULT_ROI_B
OUTPUT_ROOT = f"{_DEFAULT_OUTPUT_BASE}/{ROI_A}_{ROI_B}"

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.ridge_utils import fit_null_r2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# MAIN
# =============================================================================

def main(sub_id: str):
    subj_dir = os.path.join(OUTPUT_ROOT, "subjects", sub_id)

    log.info("=" * 60)
    log.info(f"Script 04 — Null-corrected R² for subject {sub_id}")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info("=" * 60)

    Y_test = np.load(os.path.join(subj_dir, "Y_test.npy"))             # (T_test, 108441)
    R2_a   = np.load(os.path.join(subj_dir, f"R2_{ROI_A}.npy"))        # (108441,)
    R2_b   = np.load(os.path.join(subj_dir, f"R2_{ROI_B}.npy"))
    mean_a = np.load(os.path.join(subj_dir, f"mean_{ROI_A}_test.npy")) # (T_test,)
    mean_b = np.load(os.path.join(subj_dir, f"mean_{ROI_B}_test.npy"))

    log.info(f"  Y_test  : {Y_test.shape}")
    log.info(f"  R2_{ROI_A}: mean={R2_a.mean():.4f}  frac>0={np.mean(R2_a > 0):.1%}")
    log.info(f"  R2_{ROI_B}: mean={R2_b.mean():.4f}  frac>0={np.mean(R2_b > 0):.1%}")

    # ── OLS null model R² ─────────────────────────────────────────────────────
    # fit_null_r2(regressor, Y): Y is (T, n_targets), regressor is (T,)
    # Matches MssCf.test_null_model_precomputed() OLS math (same formula,
    # himalaya convention Y.T in place of vicsompy's (n_targets, T) data).
    log.info("Fitting OLS null models …")
    R2_null_a = fit_null_r2(mean_a, Y_test)    # (108441,) float32
    R2_null_b = fit_null_r2(mean_b, Y_test)    # (108441,) float32

    R2_a_nc = (R2_a - R2_null_a).astype(np.float32)
    R2_b_nc = (R2_b - R2_null_b).astype(np.float32)

    log.info(f"  R2_null_{ROI_A}: mean={R2_null_a.mean():.4f}  "
             f"frac>0={np.mean(R2_null_a > 0):.1%}")
    log.info(f"  R2_null_{ROI_B}: mean={R2_null_b.mean():.4f}  "
             f"frac>0={np.mean(R2_null_b > 0):.1%}")
    log.info(f"  R2_{ROI_A}_nc:   mean={R2_a_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_a_nc > 0):.1%}")
    log.info(f"  R2_{ROI_B}_nc:   mean={R2_b_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_b_nc > 0):.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    for name, arr in [
        (f"R2_null_{ROI_A}", R2_null_a),
        (f"R2_null_{ROI_B}", R2_null_b),
        (f"R2_{ROI_A}_nc",   R2_a_nc),
        (f"R2_{ROI_B}_nc",   R2_b_nc),
    ]:
        np.save(os.path.join(subj_dir, f"{name}.npy"), arr)
        log.info(f"  Saved: {name}.npy")

    log.info(f"\nScript 04 complete for subject {sub_id}.")
    log.info(f"After all subjects: python 05_group_average.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-subject null-model correction of split R² maps."
    )
    parser.add_argument("--subject",     required=True,
                        help="6-digit HCP subject ID")
    parser.add_argument("--roi_a",       default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",       default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base", default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    args = parser.parse_args()

    ROI_A       = args.roi_a
    ROI_B       = args.roi_b
    OUTPUT_ROOT = f"{args.output_base}/{ROI_A}_{ROI_B}"

    main(args.subject)
