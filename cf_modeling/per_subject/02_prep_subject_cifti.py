"""
02_prep_subject_cifti.py
========================
Per-subject Phase 2: load 4 HCP 7T CIFTI runs, apply Hedger et al. (2025)
preprocessing, then the vicsompy train/test split.

Data flow (per run)
-------------------
  load CIFTI → (118584, T) bilateral surface
  Savitzky-Golay high-pass filter (3rd order, 201-sample window)
  Convert to percent signal change (PSC)  ← matches Hedger et al. _sg_psc files
  Split: hold out last 103 TRs as test clip
  z-score per vertex (train and test separately) ← vicsompy split_test_sequence
  Concatenate training TRs across runs: (118584, T_train)
  Concatenate test clips across runs:   (118584, T_test=412)
  Project training and test data onto LBOEs → X_train, X_test
  Extract CIFTI grayordinates as Y_train, Y_test (108441 targets per timepoint)
  Compute mean ROI test timecourses for the null model (script 04)

Run
---
    conda activate vicsompy_av
    python 02_prep_subject_cifti.py --subject 100610 \\
        --roi_a A5 --roi_b FFC \\
        --output_base /path/to/outputs \\
        --cifti_dir /path/to/HCP/fMRI_CIFTI
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
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

# HCP 7T phase-encoding direction per run number (1-indexed, matches filenames)
PHASE_MAP = {1: "AP", 2: "PA", 3: "PA", 4: "AP"}
RUNS      = [1, 2, 3, 4]
TEST_DURATION = 103   # TRs held out from the end of each run as test clip

# Savitzky-Golay + PSC parameters — exactly Hedger et al. (2025) config.yml
SG_WINDOW = 201   # window_length in TRs (must be odd; 201 s at TR=1s)
SG_ORDER  = 3     # polynomial order

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging
import pickle

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

# cf_modeling root on path so vendor/ and shared/ are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.cifti_io import CiftiHandler, _extract_grayords
from shared.ridge_utils import project_onto_lboes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def sg_psc(data: np.ndarray) -> np.ndarray:
    """Savitzky-Golay high-pass filter + percent signal change.

    Replicates Hedger et al. (2025) preprocessing applied before vicsompy loads
    data (their files are named *_sg_psc.nii):
        1. SG low-pass (window=201, order=3) → estimates slow drift/baseline
        2. Subtract baseline → high-pass filtered signal
        3. Divide by mean baseline → percent signal change

    Parameters
    ----------
    data : (n_vertices, T) float — raw BOLD timeseries for one run

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


def get_data_path(subject: str, run_number: int) -> str:
    """Construct path to the flat CIFTI dtseries.nii for one HCP 7T run.

    HCP 7T filenames encode subject ID, run number, and phase-encoding
    direction (AP/PA). Files live in a single flat directory (CIFTI_DIR).
    """
    phase = PHASE_MAP[run_number]
    fname = (f"{subject}_tfMRI_MOVIE{run_number}"
             f"_7T_{phase}_Atlas_1.6mm_hp2000_clean.dtseries.nii")
    return os.path.join(CIFTI_DIR, fname)


def load_run_as_surface(path: str) -> np.ndarray:
    """Load one CIFTI run → (118584, T) bilateral 59k surface."""
    ch = CiftiHandler(path)
    ch.get_data()
    ch.decompose_cifti(ch.data)   # sets ch.surface (118584, T)
    surface = ch.surface.copy()   # copy before freeing ch
    del ch
    return surface


def load_and_split(subject: str):
    """Load all 4 HCP 7T MOVIE runs and apply the vicsompy train/test split.

    For each run:
      - hold out the last TEST_DURATION TRs → test clip
      - z-score remaining TRs per vertex → training data

    Returns
    -------
    all_data   : (118584, T_train) — training surface, z-scored per vertex
    concat_test: (118584, T_test) — test clips concatenated (4 × 103 = 412 TRs)
    run_onsets : (4,) — training-concat onset indices for LORO-CV
    brainmodel : CiftiHandler — for BrainModelAxis extraction
    dpaths     : list of str — raw CIFTI file paths
    """
    dpaths = [get_data_path(subject, r) for r in RUNS]
    for p in dpaths:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"CIFTI file not found: {p}\n"
                f"Check --cifti_dir (currently: {CIFTI_DIR}) "
                f"and --subject (currently: {subject})."
            )

    train_runs, test_runs = [], []
    for path in dpaths:
        log.info(f"  Loading: {os.path.basename(path)}")
        raw_surf = load_run_as_surface(path)                     # (118584, T)
        surf = sg_psc(raw_surf)                                  # SG high-pass + PSC
        del raw_surf
        test_runs.append(
            stats.zscore(surf[:, -TEST_DURATION:], axis=1))     # (118584, 103)
        train_runs.append(
            stats.zscore(surf[:, :-TEST_DURATION], axis=1))     # (118584, T-103)

    rundurs    = [r.shape[1] for r in train_runs]
    run_onsets = np.concatenate([[0], np.cumsum(rundurs)])[:-1]
    all_data   = np.hstack(train_runs)                          # (118584, T_train)
    concat_test = np.concatenate(test_runs, axis=1)             # (118584, 412)

    brainmodel = CiftiHandler(dpaths[0])
    brainmodel.get_data()

    return all_data, concat_test, run_onsets, brainmodel, dpaths


def load_subsurfaces():
    """Load pre-built Subsurface pkls from script 01."""
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
        subs[roi] = sub
        log.info(f"  {roi} subsurface: L={sub.L_eigenvectors.shape}  "
                 f"R={sub.R_eigenvectors.shape}  n_lboe={sub.n_lboe}")
    return subs


# =============================================================================
# MAIN
# =============================================================================

def main(sub_id: str):
    subj_dir = os.path.join(OUTPUT_ROOT, "subjects", sub_id)
    os.makedirs(subj_dir, exist_ok=True)

    log.info("=" * 60)
    log.info(f"Script 02 — Prepare subject {sub_id}")
    log.info(f"  ROIs      : {ROI_A} × {ROI_B}")
    log.info(f"  CIFTI dir : {CIFTI_DIR}")
    log.info(f"  Output dir: {subj_dir}")
    log.info("=" * 60)

    # ── Load subsurfaces ─────────────────────────────────────────────────────
    log.info("\nLoading subsurfaces …")
    subs = load_subsurfaces()
    sub_a, sub_b = subs[ROI_A], subs[ROI_B]

    # ── Load + split CIFTI data ───────────────────────────────────────────────
    log.info(f"\nLoading CIFTI data for subject {sub_id} …")
    all_data, concat_test, run_onsets, brainmodel, dpaths = load_and_split(sub_id)
    bm_axis = brainmodel.brain_models
    log.info(f"  all_data (train):   {all_data.shape}")
    log.info(f"  concat_test:        {concat_test.shape}")
    log.info(f"  run_onsets:         {run_onsets.tolist()}")

    # ── LBOE design matrices ──────────────────────────────────────────────────
    log.info("\nBuilding LBOE design matrices …")
    X_train, band_sizes = project_onto_lboes(all_data,    [sub_a, sub_b])
    X_test,  _          = project_onto_lboes(concat_test, [sub_a, sub_b])
    log.info(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    log.info(f"  band_sizes: {band_sizes}")

    # ── CIFTI grayordinate targets ────────────────────────────────────────────
    log.info("\nExtracting CIFTI grayordinates for Y targets …")
    Y_train = _extract_grayords(all_data,    bm_axis)     # (T_train, 108441)
    Y_test  = _extract_grayords(concat_test, bm_axis)     # (T_test,  108441)
    log.info(f"  Y_train: {Y_train.shape}  Y_test: {Y_test.shape}")

    # ── Null model mean ROI test timecourses ──────────────────────────────────
    # subsurface_verts = np.concatenate([verts_L, verts_R]) — set in script 01
    log.info("\nComputing mean ROI test timecourses for null model …")
    for roi_name, s in [(ROI_A, sub_a), (ROI_B, sub_b)]:
        mean_tc = np.nanmean(concat_test[s.subsurface_verts, :], axis=0)
        np.save(os.path.join(subj_dir, f"mean_{roi_name}_test.npy"),
                mean_tc.astype(np.float32))
        log.info(f"  mean_{roi_name}_test: {mean_tc.shape}  std={mean_tc.std():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"\nSaving to {subj_dir} …")
    np.save(os.path.join(subj_dir, "X_train.npy"),
            X_train.astype(np.float32))
    np.save(os.path.join(subj_dir, "X_test.npy"),
            X_test.astype(np.float32))
    np.save(os.path.join(subj_dir, "Y_train.npy"),    Y_train)
    np.save(os.path.join(subj_dir, "Y_test.npy"),     Y_test)
    np.save(os.path.join(subj_dir, "run_onsets.npy"), run_onsets)
    np.save(os.path.join(subj_dir, "band_sizes.npy"),
            np.array(band_sizes, dtype=np.int32))
    log.info(f"  Saved: X_train X_test Y_train Y_test run_onsets band_sizes "
             f"mean_{ROI_A}_test mean_{ROI_B}_test")
    log.info(f"\nScript 02 complete for subject {sub_id}.")
    log.info(f"Next: python 03_fit_subject.py --subject {sub_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-subject CIFTI prep: LBOE design matrices and Y targets."
    )
    parser.add_argument("--subject",     required=True,
                        help="6-digit HCP subject ID")
    parser.add_argument("--roi_a",       default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",       default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base", default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--cifti_dir",   default=_DEFAULT_CIFTI_DIR,
                        help="Flat directory containing subject CIFTI dtseries files "
                             "(default: %(default)s)")
    args = parser.parse_args()

    ROI_A       = args.roi_a
    ROI_B       = args.roi_b
    CIFTI_DIR   = args.cifti_dir
    OUTPUT_ROOT = f"{args.output_base}/{ROI_A}_{ROI_B}"
    CACHE_DIR   = f"{OUTPUT_ROOT}/subsurfaces"

    main(args.subject)
