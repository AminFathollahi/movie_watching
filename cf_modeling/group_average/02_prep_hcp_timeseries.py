"""
02_prep_hcp_timeseries.py
=========================
Phase 2: Load fMRI MAT files, apply the HCP movie-watching train/test split
using movie_timing.csv, z-score per run (faithfully reproducing
HcpSubject.split_test_sequence()), project BOLD onto the LBOEs from Script 1
to build X_train / X_test, and extract the 59k grayordinate target arrays
Y_train / Y_test from the template CIFTI's BrainModelAxis.

Inputs (from Script 1):
    CACHE_DIR/sub_{ROI_A}.pkl, sub_{ROI_B}.pkl

Outputs (to PREP_DIR):
    X_train.npy        (T_train, 2*n_lboe_a + 2*n_lboe_b) — design matrix
    Y_train.npy        (T_train, 59412) — grayordinate BOLD, training set
    X_test.npy         (328, 2*n_lboe_a + 2*n_lboe_b)    — design matrix
    Y_test.npy         (328, 59412)     — grayordinate BOLD, test clips
    run_onsets.npy     (4,)             — TR indices where each training run starts
    band_sizes.npy     (2,)             — [n_roi_a_cols, n_roi_b_cols]
    mean_{ROI_A}_test.npy, mean_{ROI_B}_test.npy — for script 04 null model

Movie timing / train-test logic
--------------------------------
• All 18 video segments are concatenated in the MAT files (no inter-run gaps).
• The last video of each run (video5, video9, video14, video18, each 82 TRs)
  is the repeated validation clip.
• Training data: videos 1-4, 6-8, 10-13, 15-17 — z-scored PER RUN independently.
• Test data: videos 5, 9, 14, 18 — z-scored per clip, then concatenated across
  the 4 repetitions → 4 × 82 = 328-TR array (concatenated_test_sequence).

Run
---
    conda activate vicsompy_av
    python 02_prep_hcp_timeseries.py \\
        --roi_a A1 --roi_b V1 \\
        --output_base /path/to/outputs \\
        --data_base /path/to/data
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os

_DEFAULT_DATA_BASE   = "/home/amin/Research/Representation/Movie/data/Setareh"
_DEFAULT_ROI_A       = "A1"
_DEFAULT_ROI_B       = "V1"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/group_average"

ROI_A       = _DEFAULT_ROI_A
ROI_B       = _DEFAULT_ROI_B
DATA_BASE   = _DEFAULT_DATA_BASE
OUTPUT_BASE = _DEFAULT_OUTPUT_BASE

# Derived paths (rebound in __main__ after arg parsing)
OUTPUT_DIR = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
PREP_DIR   = f"{OUTPUT_DIR}/prep"
CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"

MAT_LEFT         = f"{DATA_BASE}/HCP Data/notmean_left_Meanfile.mat"
MAT_RIGHT        = f"{DATA_BASE}/HCP Data/notmean_right_Meanfile.mat"
MOVIE_TIMING_CSV = f"{DATA_BASE}/Data/movie_timing.csv"
TEMPLATE_CIFTI   = (
    f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
    "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
)

# Validation clips (last video of each run) — the repeated segments
TEST_VIDEO_IDS = ["video5", "video9", "video14", "video18"]

TR = 1.0   # seconds; matches vicsompy config (analysis: TR: 1.0)

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io
import scipy.stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# HELPER: load MAT file → (32492, T)
# =============================================================================

def load_mat_file(path):
    """Load a MATLAB .mat file and return BOLD as (32492, T).

    Handles both v5.0 (scipy.io) and v7.3 HDF5 (h5py) formats.
    Auto-detects the variable name by looking for an array whose first or
    second dimension equals 32492.
    """
    log.info(f"  Loading {os.path.basename(path)} …")
    data = None

    # --- Try scipy.io (v5.0) ---
    try:
        mat = scipy.io.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        for k in keys:
            arr = np.asarray(mat[k])
            if arr.ndim == 2 and 32492 in arr.shape:
                data = arr
                log.info(f"    Found key='{k}', shape={arr.shape}")
                break
        if data is None and keys:
            data = np.asarray(mat[max(keys, key=lambda k: np.asarray(mat[k]).size)])
    except Exception:
        pass

    # --- Try h5py (v7.3 HDF5) ---
    if data is None:
        try:
            import h5py
            log.info("    (Trying h5py for HDF5/v7.3 format)")
            with h5py.File(path, "r") as f:
                keys = [k for k in f.keys() if not k.startswith("#")]
                for k in keys:
                    shape = f[k].shape
                    if 32492 in shape:
                        data = f[k][:]
                        log.info(f"    Found key='{k}', shape={shape}")
                        break
                if data is None and keys:
                    data = f[keys[0]][:]
        except Exception as e:
            raise RuntimeError(f"Could not load {path}: {e}")

    if data is None:
        raise RuntimeError(f"No suitable array found in {path}")

    data = np.array(data, dtype=np.float32)

    if data.shape[0] != 32492:
        if data.shape[1] == 32492:
            data = data.T
        else:
            raise ValueError(
                f"Neither dimension equals 32492 in {path}: shape={data.shape}"
            )

    log.info(f"    Final shape: {data.shape}  (verts × time)")
    return data   # (32492, T)


# =============================================================================
# STEP 1 — Parse movie_timing.csv and compute per-video MAT indices
# =============================================================================

def parse_timing(timing_csv):
    """Return timing DataFrame augmented with cumulative indices into the
    concatenated MAT data (videos ordered 1..18, no inter-run gaps).
    """
    df = pd.read_csv(timing_csv)
    df = df.sort_values("onset_sec").reset_index(drop=True)
    df["mat_start"] = df["duration_sec"].cumsum().shift(1, fill_value=0).astype(int)
    df["mat_end"]   = df["duration_sec"].cumsum().astype(int)

    log.info(f"  Total TRs in MAT: {df['mat_end'].iloc[-1]}")
    for _, row in df.iterrows():
        tag = " ← TEST" if row["video_id"] in TEST_VIDEO_IDS else ""
        log.info(f"    {row['video_id']:8s} run={row['run_id']}  "
                 f"mat=[{row['mat_start']:4d},{row['mat_end']:4d})  "
                 f"dur={row['duration_sec']:3d}s{tag}")
    return df


# =============================================================================
# STEP 2 — Build train/test BOLD arrays with per-run z-scoring
# =============================================================================

def split_and_zscore(left_data, right_data, timing_df):
    """Apply the vicsompy train/test split and per-run z-scoring."""
    test_ids = set(TEST_VIDEO_IDS)
    run_ids  = sorted(timing_df["run_id"].unique())

    left_train_runs  = []
    right_train_runs = []
    left_test_clips  = []
    right_test_clips = []

    for run_id in run_ids:
        run_rows = timing_df[timing_df["run_id"] == run_id].sort_values("video_id")

        test_row = run_rows[run_rows["video_id"].isin(test_ids)]
        if len(test_row) != 1:
            raise ValueError(f"Expected 1 test clip in run {run_id}, got {len(test_row)}")
        ts = test_row.iloc[0]["mat_start"]
        te = test_row.iloc[0]["mat_end"]

        L_test = scipy.stats.zscore(left_data[:, ts:te],  axis=1)
        R_test = scipy.stats.zscore(right_data[:, ts:te], axis=1)
        left_test_clips.append(L_test.astype(np.float32))
        right_test_clips.append(R_test.astype(np.float32))

        train_rows = run_rows[~run_rows["video_id"].isin(test_ids)]
        if len(train_rows) == 0:
            raise ValueError(f"No training videos found for run {run_id}")

        train_indices = []
        for _, row in train_rows.iterrows():
            train_indices.extend(range(row["mat_start"], row["mat_end"]))
        train_indices = np.array(train_indices)

        L_train_run = scipy.stats.zscore(left_data[:, train_indices],  axis=1).astype(np.float32)
        R_train_run = scipy.stats.zscore(right_data[:, train_indices], axis=1).astype(np.float32)
        left_train_runs.append(L_train_run)
        right_train_runs.append(R_train_run)

        log.info(f"  Run {run_id}: train={L_train_run.shape[1]} TRs, "
                 f"test={L_test.shape[1]} TRs (z-scored per block)")

    run_lengths = [r.shape[1] for r in left_train_runs]
    run_onsets  = np.concatenate([[0], np.cumsum(run_lengths)])[:-1].astype(int)
    log.info(f"  Training run_onsets (for LORO-CV): {run_onsets.tolist()}")
    log.info(f"  Total training TRs: {sum(run_lengths)}")

    return left_train_runs, right_train_runs, left_test_clips, right_test_clips, run_onsets


# =============================================================================
# STEP 3 — Extract grayordinate target indices from template CIFTI
# =============================================================================

def get_grayordinate_indices(template_cifti_path):
    """Return vertex index arrays (in 32k surface space) for each cortex structure."""
    template_img = nib.load(template_cifti_path)
    bm_axis = template_img.header.get_axis(1)

    left_verts = right_verts = None
    for name, _, model in bm_axis.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_verts  = model.vertex
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_verts = model.vertex

    if left_verts is None or right_verts is None:
        raise RuntimeError("Could not find cortical structures in template CIFTI")

    log.info(f"  Grayordinate vertices — left: {len(left_verts)}, "
             f"right: {len(right_verts)}, total: {len(left_verts)+len(right_verts)}")
    return left_verts, right_verts, template_img


# =============================================================================
# STEP 4 — Project BOLD onto LBOEs → design matrix columns
# =============================================================================

def project_onto_lboes(left_data, right_data, sub_a, sub_b):
    """Project BOLD timeseries for ROI_A and ROI_B onto the precomputed LBOEs.

    Design matrix column order:
        cols 0 .. 2*n_lboe_a-1          : L ROI_A | R ROI_A
        cols 2*n_lboe_a .. end           : L ROI_B | R ROI_B
    """
    combined = np.vstack([left_data, right_data])   # (64984, T)

    def _proj(verts, eigvecs):
        return combined[verts, :].T @ eigvecs.real  # (T, n_lboe)

    dm_l_a = _proj(sub_a.subsurface_verts_L, sub_a.L_eigenvectors)
    dm_r_a = _proj(sub_a.subsurface_verts_R, sub_a.R_eigenvectors)
    dm_l_b = _proj(sub_b.subsurface_verts_L, sub_b.L_eigenvectors)
    dm_r_b = _proj(sub_b.subsurface_verts_R, sub_b.R_eigenvectors)

    return np.hstack([dm_l_a, dm_r_a, dm_l_b, dm_r_b]).astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(PREP_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Script 02 — Prepare HCP group-average timeseries")
    log.info(f"  ROI A: {ROI_A}  ROI B: {ROI_B}")
    log.info("=" * 60)

    # --- Load LBOEs from Script 1 ---
    log.info("\nLoading subsurfaces from Script 1 …")
    for roi in [ROI_A, ROI_B]:
        path = os.path.join(CACHE_DIR, f"sub_{roi.lower()}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run 01_extract_geometry.py first."
            )

    with open(os.path.join(CACHE_DIR, f"sub_{ROI_A.lower()}.pkl"), "rb") as fh:
        sub_a = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, f"sub_{ROI_B.lower()}.pkl"), "rb") as fh:
        sub_b = pickle.load(fh)

    for sub in [sub_a, sub_b]:
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
    log.info(f"  {ROI_A} — L: {sub_a.L_eigenvectors.shape}  R: {sub_a.R_eigenvectors.shape}  n_lboe={sub_a.n_lboe}")
    log.info(f"  {ROI_B} — L: {sub_b.L_eigenvectors.shape}  R: {sub_b.R_eigenvectors.shape}  n_lboe={sub_b.n_lboe}")

    # --- Load MAT data ---
    log.info("\nLoading fMRI MAT files …")
    left_data  = load_mat_file(MAT_LEFT)
    right_data = load_mat_file(MAT_RIGHT)

    if left_data.shape != right_data.shape:
        raise ValueError(f"Shape mismatch: left={left_data.shape}, right={right_data.shape}")
    log.info(f"  T_total = {left_data.shape[1]} TRs across all 18 videos")

    # --- Parse timing CSV ---
    log.info("\nParsing movie_timing.csv …")
    timing_df = parse_timing(MOVIE_TIMING_CSV)

    expected_T = int(timing_df["duration_sec"].sum())
    if left_data.shape[1] != expected_T:
        log.warning(f"  T_total={left_data.shape[1]} != expected {expected_T}. "
                    "Check MAT files contain all 18 video segments.")

    # --- Train/test split ---
    log.info("\nApplying train/test split and per-run z-scoring …")
    L_train_runs, R_train_runs, L_test_clips, R_test_clips, run_onsets = (
        split_and_zscore(left_data, right_data, timing_df)
    )
    del left_data, right_data

    L_train = np.hstack(L_train_runs)          # (32492, T_train)
    R_train = np.hstack(R_train_runs)
    L_test_concat = np.hstack(L_test_clips)    # (32492, 328)
    R_test_concat = np.hstack(R_test_clips)
    log.info(f"  Training: (32492, {L_train.shape[1]})  Test: (32492, {L_test_concat.shape[1]})")

    # --- Grayordinate targets ---
    log.info("\nExtracting grayordinate vertex indices from template CIFTI …")
    left_gray_verts, right_gray_verts, _ = get_grayordinate_indices(TEMPLATE_CIFTI)

    Y_train = np.hstack([
        L_train[left_gray_verts,  :].T,
        R_train[right_gray_verts, :].T,
    ]).astype(np.float32)

    Y_test = np.hstack([
        L_test_concat[left_gray_verts,  :].T,
        R_test_concat[right_gray_verts, :].T,
    ]).astype(np.float32)

    log.info(f"  Y_train: {Y_train.shape}  Y_test: {Y_test.shape}")

    # --- Design matrices ---
    log.info("\nProjecting BOLD onto LBOEs …")
    X_train = project_onto_lboes(L_train, R_train, sub_a, sub_b)
    X_test  = project_onto_lboes(L_test_concat, R_test_concat, sub_a, sub_b)
    n_a_cols = 2 * sub_a.n_lboe
    n_b_cols = 2 * sub_b.n_lboe
    log.info(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    log.info(f"  band_sizes: {ROI_A}={n_a_cols}  {ROI_B}={n_b_cols}")

    # --- Mean ROI test timecourses for null model ---
    log.info("\nSaving mean ROI test timecourses for null model …")
    for roi_name, sub in [(ROI_A, sub_a), (ROI_B, sub_b)]:
        verts_L = sub.subsurface_verts_L
        verts_R = sub.subsurface_verts_R - 32492
        tc_L = L_test_concat[verts_L, :]
        tc_R = R_test_concat[verts_R, :]
        mean_tc = (tc_L.sum(axis=0) + tc_R.sum(axis=0)) / (tc_L.shape[0] + tc_R.shape[0])
        np.save(os.path.join(PREP_DIR, f"mean_{roi_name}_test.npy"), mean_tc.astype(np.float32))
        log.info(f"  mean_{roi_name}_test.npy: {mean_tc.shape}  std={mean_tc.std():.4f}")

    # --- Save ---
    log.info(f"\nSaving to {PREP_DIR} …")
    np.save(os.path.join(PREP_DIR, "X_train.npy"),    X_train)
    np.save(os.path.join(PREP_DIR, "Y_train.npy"),    Y_train)
    np.save(os.path.join(PREP_DIR, "X_test.npy"),     X_test)
    np.save(os.path.join(PREP_DIR, "Y_test.npy"),     Y_test)
    np.save(os.path.join(PREP_DIR, "run_onsets.npy"), run_onsets)
    np.save(os.path.join(PREP_DIR, "band_sizes.npy"),
            np.array([n_a_cols, n_b_cols], dtype=np.int32))
    log.info(f"  Saved: X_train Y_train X_test Y_test run_onsets band_sizes "
             f"mean_{ROI_A}_test mean_{ROI_B}_test")
    log.info("\nScript 02 complete.")
    log.info(f"Next: python 03_fit_banded_ridge.py --roi_a {ROI_A} --roi_b {ROI_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare HCP group-average timeseries: LBOE projection + train/test split."
    )
    parser.add_argument("--roi_a",       default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",       default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base", default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--data_base",   default=_DEFAULT_DATA_BASE,
                        help="Data root directory (default: %(default)s)")
    args = parser.parse_args()

    ROI_A       = args.roi_a
    ROI_B       = args.roi_b
    DATA_BASE   = args.data_base
    OUTPUT_BASE = args.output_base
    OUTPUT_DIR  = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
    PREP_DIR    = f"{OUTPUT_DIR}/prep"
    CACHE_DIR   = f"{OUTPUT_DIR}/subsurfaces"
    MAT_LEFT    = f"{DATA_BASE}/HCP Data/notmean_left_Meanfile.mat"
    MAT_RIGHT   = f"{DATA_BASE}/HCP Data/notmean_right_Meanfile.mat"
    MOVIE_TIMING_CSV = f"{DATA_BASE}/Data/movie_timing.csv"
    TEMPLATE_CIFTI   = (f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
                        "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii")

    main()
