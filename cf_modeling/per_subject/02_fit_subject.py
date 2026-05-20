"""
02_fit_subject.py
=================
Per-subject CF pipeline (merged): SG+PSC preprocessing → banded ridge fit →
null-model correction. Saves only the final R² maps; no intermediate arrays.

Replaces the separate 02_prep / 03_fit / 04_null_correct scripts.

Data flow (per run)
-------------------
  load CIFTI → (118584, T) 59k bilateral surface
  Savitzky-Golay high-pass filter (window=201, order=3) per vertex   ← Hedger et al.
  Percent signal change (PSC)                                         ← Hedger et al.
  Split: last 103 TRs → test clip; remaining → training
  Z-score per vertex (train and test segments independently)          ← vicsompy

Then across runs:
  Concatenate training segments → (118584, T_train)
  Concatenate test clips       → (118584, 412)
  Project onto LBOEs           → X_train (T_train, 2*n_lboe), X_test (412, 2*n_lboe)
  Extract 108441 grayordinates → Y_train (T_train, 108441), Y_test (412, 108441)

Fit himalaya MultipleKernelRidgeCV, compute split R² and null-corrected R².

Outputs saved to OUTPUT_ROOT/{ROI_A}_{ROI_B}/subjects/{subject}/
  R2_full.npy, R2_{ROI_A}.npy, R2_{ROI_B}.npy, Shared_R2.npy
  R2_null_{ROI_A}.npy, R2_null_{ROI_B}.npy
  R2_{ROI_A}_nc.npy, R2_{ROI_B}_nc.npy      ← primary outputs for group average

Run
---
    conda activate vicsompy_av
    python 02_fit_subject.py --subject 100610 \\
        --roi_a A5 --roi_b FFC \\
        --output_base /path/to/outputs \\
        --cifti_dir /path/to/HCP/fMRI_CIFTI
"""

# =============================================================================
# CONFIG
# =============================================================================
import os
import sys

_DEFAULT_ROI_A       = "A5"
_DEFAULT_ROI_B       = "FFC"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/per_subject"
_DEFAULT_CIFTI_DIR   = "/media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI"

ROI_A       = _DEFAULT_ROI_A
ROI_B       = _DEFAULT_ROI_B
CIFTI_DIR   = _DEFAULT_CIFTI_DIR
OUTPUT_ROOT = f"{_DEFAULT_OUTPUT_BASE}/{ROI_A}_{ROI_B}"
CACHE_DIR   = f"{OUTPUT_ROOT}/subsurfaces"

PHASE_MAP     = {1: "AP", 2: "PA", 3: "PA", 4: "AP"}
RUNS          = [1, 2, 3, 4]
TEST_DURATION = 103   # TRs held out per run (Hedger et al. Methods)

# Savitzky-Golay parameters — exactly Hedger et al. (2025) config.yml
SG_WINDOW = 201
SG_ORDER  = 3

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging
import pickle

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from himalaya.scoring import r2_score, r2_score_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.cifti_io import CiftiHandler, _extract_grayords
from shared.ridge_utils import build_pipeline, project_onto_lboes, fit_null_r2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# PREPROCESSING
# =============================================================================

def sg_psc(data: np.ndarray) -> np.ndarray:
    """Savitzky-Golay high-pass + percent signal change (Hedger et al. 2025).

    Parameters
    ----------
    data : (n_vertices, T) — raw BOLD for one run

    Returns
    -------
    psc : (n_vertices, T) float32
    """
    data64   = data.astype(np.float64)
    baseline = savgol_filter(data64, window_length=SG_WINDOW,
                             polyorder=SG_ORDER, axis=1)
    mean_baseline = baseline.mean(axis=1, keepdims=True)
    mean_baseline[np.abs(mean_baseline) < 1.0] = 1.0
    data64 -= baseline
    del baseline
    data64 /= mean_baseline
    data64 *= 100.0
    return data64.astype(np.float32)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_data_path(subject: str, run_number: int) -> str:
    phase = PHASE_MAP[run_number]
    fname = (f"{subject}_tfMRI_MOVIE{run_number}"
             f"_7T_{phase}_Atlas_1.6mm_hp2000_clean.dtseries.nii")
    return os.path.join(CIFTI_DIR, fname)


def load_and_preprocess(subject: str):
    """Load all 4 runs, apply SG+PSC, split into train/test.

    Returns
    -------
    all_data    : (118584, T_train) float32 — z-scored training surface
    concat_test : (118584, 412) float32     — z-scored test clips
    run_onsets  : (4,) int                  — LORO-CV onsets in training concat
    brainmodel  : CiftiHandler              — for grayordinate extraction
    """
    dpaths = [get_data_path(subject, r) for r in RUNS]
    for p in dpaths:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"CIFTI not found: {p}\n"
                f"Check --cifti_dir ({CIFTI_DIR}) and --subject ({subject})."
            )

    train_runs, test_runs = [], []
    for path in dpaths:
        log.info(f"  Loading: {os.path.basename(path)}")
        ch = CiftiHandler(path)
        ch.get_data()
        ch.decompose_cifti(ch.data)          # → ch.surface (118584, T)
        raw_surf = ch.surface
        del ch                               # free ch.data + all CIFTI memory before sg_psc
        surf = sg_psc(raw_surf)             # SG high-pass + PSC
        del raw_surf
        test_runs.append(
            stats.zscore(surf[:, -TEST_DURATION:], axis=1).astype(np.float32))
        train_runs.append(
            stats.zscore(surf[:, :-TEST_DURATION], axis=1).astype(np.float32))

    rundurs    = [r.shape[1] for r in train_runs]
    run_onsets = np.concatenate([[0], np.cumsum(rundurs)])[:-1]
    all_data    = np.hstack(train_runs);  del train_runs
    concat_test = np.concatenate(test_runs, axis=1);  del test_runs

    brainmodel = CiftiHandler(dpaths[0])
    brainmodel.get_data()

    return all_data, concat_test, run_onsets, brainmodel


def load_subsurfaces():
    subs = {}
    for roi in [ROI_A, ROI_B]:
        path = os.path.join(CACHE_DIR, f"sub_{roi.lower()}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run 01_extract_geometry.py first."
            )
        with open(path, "rb") as fh:
            sub = pickle.load(fh)
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
        log.info(f"  {roi}: L={sub.L_eigenvectors.shape}  "
                 f"R={sub.R_eigenvectors.shape}  n_lboe={sub.n_lboe}")
        subs[roi] = sub
    return subs


# =============================================================================
# HELPERS
# =============================================================================

def be_to_npy(var, backend):
    if isinstance(var, list):
        return [backend.to_numpy(v) for v in var]
    return backend.to_numpy(var)


def save_map(arr, name, subj_dir):
    np.save(os.path.join(subj_dir, f"{name}.npy"), arr.astype(np.float32))
    log.info(f"  {name}: mean={arr.mean():.4f}  frac>0={np.mean(arr>0):.1%}")


# =============================================================================
# MAIN
# =============================================================================

def main(sub_id: str):
    subj_dir = os.path.join(OUTPUT_ROOT, "subjects", sub_id)
    os.makedirs(subj_dir, exist_ok=True)

    log.info("=" * 60)
    log.info(f"02_fit_subject — subject {sub_id}")
    log.info(f"  ROIs      : {ROI_A} × {ROI_B}")
    log.info(f"  CIFTI dir : {CIFTI_DIR}")
    log.info(f"  Output    : {subj_dir}")
    log.info("=" * 60)

    # ── Subsurfaces ───────────────────────────────────────────────────────────
    log.info("\nLoading subsurfaces …")
    subs = load_subsurfaces()
    sub_a, sub_b = subs[ROI_A], subs[ROI_B]

    # ── Load + preprocess CIFTI data ──────────────────────────────────────────
    log.info(f"\nLoading + preprocessing CIFTI data for {sub_id} …")
    all_data, concat_test, run_onsets, brainmodel = load_and_preprocess(sub_id)
    bm_axis = brainmodel.brain_models
    log.info(f"  train: {all_data.shape}  test: {concat_test.shape}")
    log.info(f"  run_onsets: {run_onsets.tolist()}")

    # ── Design matrices (X) ───────────────────────────────────────────────────
    log.info("\nBuilding LBOE design matrices …")
    X_train, band_sizes = project_onto_lboes(all_data,    [sub_a, sub_b])
    X_test,  _          = project_onto_lboes(concat_test, [sub_a, sub_b])
    log.info(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    log.info(f"  band_sizes: {band_sizes}")

    # ── CIFTI grayordinate targets (Y) ────────────────────────────────────────
    log.info("\nExtracting CIFTI grayordinates …")
    Y_train = _extract_grayords(all_data,    bm_axis)   # (T_train, 108441)
    Y_test  = _extract_grayords(concat_test, bm_axis)   # (412, 108441)
    log.info(f"  Y_train: {Y_train.shape}  Y_test: {Y_test.shape}")

    # Compute mean ROI test timecourses from surface space before freeing arrays
    verts_a = np.concatenate([sub_a.subsurface_verts_L, sub_a.subsurface_verts_R])
    verts_b = np.concatenate([sub_b.subsurface_verts_L, sub_b.subsurface_verts_R])
    mean_a  = np.nanmean(concat_test[verts_a, :], axis=0).astype(np.float32)  # (T_test,)
    mean_b  = np.nanmean(concat_test[verts_b, :], axis=0).astype(np.float32)

    # Free large surface arrays — Y arrays are now the only copy of the data
    del all_data, concat_test

    # ── Fit banded ridge ──────────────────────────────────────────────────────
    log.info("\nFitting banded ridge …")
    pipeline, backend = build_pipeline(
        n_samples_train=X_train.shape[0],
        run_onsets=run_onsets,
        band_sizes=band_sizes,
        roi_names=[ROI_A.lower(), ROI_B.lower()],
    )
    pipeline.fit(X_train, Y_train)
    log.info("  Fit complete.")

    # ── Full-model R² ─────────────────────────────────────────────────────────
    Y_hat_full = pipeline.predict(X_test)
    R2_full    = be_to_npy(r2_score(Y_test, Y_hat_full), backend)

    # ── Per-band R² (variance partitioning) ──────────────────────────────────
    Y_hat_split  = pipeline.predict(X_test, split=True)
    split_scores = be_to_npy(r2_score_split(Y_test, Y_hat_split), backend)
    R2_a   = split_scores[0]
    R2_b   = split_scores[1]
    Shared = R2_a + R2_b - R2_full

    log.info(f"\n  R2_full:    {np.nanmean(R2_full):.4f}")
    log.info(f"  R2_{ROI_A}: {np.nanmean(R2_a):.4f}")
    log.info(f"  R2_{ROI_B}: {np.nanmean(R2_b):.4f}")
    log.info(f"  Shared:     {np.nanmean(Shared):.4f}")

    # ── Null model (mean ROI timecourse as single regressor) ─────────────────
    log.info("\nFitting null models …")
    R2_null_a = fit_null_r2(mean_a, Y_test)
    R2_null_b = fit_null_r2(mean_b, Y_test)
    R2_a_nc   = (R2_a - R2_null_a).astype(np.float32)
    R2_b_nc   = (R2_b - R2_null_b).astype(np.float32)

    log.info(f"  R2_{ROI_A}_nc: mean={R2_a_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_a_nc>0):.1%}")
    log.info(f"  R2_{ROI_B}_nc: mean={R2_b_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_b_nc>0):.1%}")

    # ── Save R² results ───────────────────────────────────────────────────────
    log.info(f"\nSaving results to {subj_dir} …")
    save_map(R2_full,    "R2_full",            subj_dir)
    save_map(R2_a,       f"R2_{ROI_A}",        subj_dir)
    save_map(R2_b,       f"R2_{ROI_B}",        subj_dir)
    save_map(Shared,     "Shared_R2",          subj_dir)
    save_map(R2_null_a,  f"R2_null_{ROI_A}",   subj_dir)
    save_map(R2_null_b,  f"R2_null_{ROI_B}",   subj_dir)
    save_map(R2_a_nc,    f"R2_{ROI_A}_nc",     subj_dir)
    save_map(R2_b_nc,    f"R2_{ROI_B}_nc",     subj_dir)

    log.info(f"\n02_fit_subject complete for subject {sub_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-subject CF pipeline: preprocess → fit → null correct."
    )
    parser.add_argument("--subject",     required=True)
    parser.add_argument("--roi_a",       default=_DEFAULT_ROI_A)
    parser.add_argument("--roi_b",       default=_DEFAULT_ROI_B)
    parser.add_argument("--output_base", default=_DEFAULT_OUTPUT_BASE)
    parser.add_argument("--cifti_dir",   default=_DEFAULT_CIFTI_DIR)
    args = parser.parse_args()

    ROI_A       = args.roi_a
    ROI_B       = args.roi_b
    CIFTI_DIR   = args.cifti_dir
    OUTPUT_ROOT = f"{args.output_base}/{ROI_A}_{ROI_B}"
    CACHE_DIR   = f"{OUTPUT_ROOT}/subsurfaces"

    main(args.subject)
