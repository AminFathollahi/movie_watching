"""
03_fit_subject.py
=================
Per-subject Phase 3: fit the himalaya banded ridge model and compute variance
partitioning maps on the held-out test set.

Variance decomposition
----------------------
For two bands (ROI_A, ROI_B):
    Y_hat_full  = pipeline.predict(X_test)              → full model
    Y_hat_split = pipeline.predict(X_test, split=True)  → per-band predictions
    R2_full     = r2_score(Y_test, Y_hat_full)          → (108441,)
    [R2_A, R2_B] = r2_score_split(Y_test, Y_hat_split)  → per-band contributions
    Shared_R2   = R2_A + R2_B - R2_full

Inputs (from script 02):
    subjects/<sub>/X_train.npy, Y_train.npy, X_test.npy, Y_test.npy,
    run_onsets.npy, band_sizes.npy

Outputs:
    subjects/<sub>/R2_full.npy, R2_<ROI_A>.npy, R2_<ROI_B>.npy, Shared_R2.npy

Run
---
    conda activate vicsompy_av
    python 03_fit_subject.py --subject 100610 \\
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


# =============================================================================
# MAIN
# =============================================================================

def main(sub_id: str):
    subj_dir = os.path.join(OUTPUT_ROOT, "subjects", sub_id)

    for fname in ["X_train", "Y_train", "X_test", "Y_test", "run_onsets", "band_sizes"]:
        if not os.path.exists(os.path.join(subj_dir, f"{fname}.npy")):
            raise FileNotFoundError(
                f"Missing {subj_dir}/{fname}.npy. Run 02_prep_subject_cifti.py first."
            )

    log.info("=" * 60)
    log.info(f"Script 03 — Fit banded ridge for subject {sub_id}")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info("=" * 60)

    X_train    = np.load(os.path.join(subj_dir, "X_train.npy"))
    Y_train    = np.load(os.path.join(subj_dir, "Y_train.npy"))
    X_test     = np.load(os.path.join(subj_dir, "X_test.npy"))
    Y_test     = np.load(os.path.join(subj_dir, "Y_test.npy"))
    run_onsets = np.load(os.path.join(subj_dir, "run_onsets.npy"))
    band_sizes = np.load(os.path.join(subj_dir, "band_sizes.npy"))

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
    Y_hat_full = pipeline.predict(X_test)                               # (T_test, 108441)
    R2_full    = be_to_npy(r2_score(Y_test, Y_hat_full), backend)      # (108441,)

    # ── Per-band R² (variance partitioning) ───────────────────────────────────
    log.info("Computing split (per-band) R² on test set …")
    Y_hat_split  = pipeline.predict(X_test, split=True)                # list[2 × (T_test, 108441)]
    split_scores = be_to_npy(
        r2_score_split(Y_test, Y_hat_split), backend
    )                                                                   # (2, 108441)
    R2_a   = split_scores[0]                                            # (108441,)
    R2_b   = split_scores[1]                                            # (108441,)

    # ── Shared variance ───────────────────────────────────────────────────────
    Shared = R2_a + R2_b - R2_full

    log.info("\nVariance partitioning — test set means:")
    log.info(f"  R2_full  : {np.nanmean(R2_full):.4f}  "
             f"(>0.01: {np.mean(R2_full > 0.01):.1%})")
    log.info(f"  R2_{ROI_A}: {np.nanmean(R2_a):.4f}  "
             f"(>0: {np.mean(R2_a > 0):.1%})")
    log.info(f"  R2_{ROI_B}: {np.nanmean(R2_b):.4f}  "
             f"(>0: {np.mean(R2_b > 0):.1%})")
    log.info(f"  Shared   : {np.nanmean(Shared):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"\nSaving R² maps to {subj_dir} …")
    np.save(os.path.join(subj_dir, "R2_full.npy"),      R2_full.astype(np.float32))
    np.save(os.path.join(subj_dir, f"R2_{ROI_A}.npy"), R2_a.astype(np.float32))
    np.save(os.path.join(subj_dir, f"R2_{ROI_B}.npy"), R2_b.astype(np.float32))
    np.save(os.path.join(subj_dir, "Shared_R2.npy"),    Shared.astype(np.float32))
    log.info(f"\nScript 03 complete for subject {sub_id}.")
    log.info(f"Next: python 04_null_correct_subject.py --subject {sub_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-subject banded ridge fit and variance partitioning."
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
