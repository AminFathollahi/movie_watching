"""
02_prep_hcp_timeseries.py
=========================
Phase 2: Load fMRI MAT files, apply the HCP movie-watching train/test split
using movie_timing.csv, z-score per run (faithfully reproducing
HcpSubject.split_test_sequence()), project BOLD onto the LBOEs from Script 1
to build X_train / X_test, and extract the 59 k grayordinate target arrays
Y_train / Y_test from the template CIFTI's BrainModelAxis.

Inputs (from Script 1):
    CACHE_DIR/sub_a1.pkl, sub_v1.pkl

Outputs (to PREP_DIR):
    X_train.npy        (T_train, 2*n_lboe_a1 + 2*n_lboe_v1) — design matrix, training set
    Y_train.npy        (T_train, 59412) — grayordinate BOLD, training set
    X_test.npy         (328, 2*n_lboe_a1 + 2*n_lboe_v1)   — design matrix, test clips
    Y_test.npy         (328, 59412)     — grayordinate BOLD, concatenated test clips
    run_onsets.npy     (4,)             — TR indices where each training run starts

Movie timing / train-test logic
--------------------------------
• All 18 video segments are concatenated in the MAT files (no inter-run gaps).
• The last video of each run (video5, video9, video14, video18, each 82 TRs)
  is the repeated validation clip.
• Training data: videos 1-4, 6-8, 10-13, 15-17 — z-scored PER RUN independently.
• Test data: videos 5, 9, 14, 18 — z-scored per clip, then concatenated across
  the 4 repetitions → 4 × 82 = 328-TR array (concatenated_test_sequence).

Divergences from vicsompy (HcpSubject)
---------------------------------------
1. Split criterion: vicsompy removes the last 103 TRs of EACH RUN as the test
   sequence (hardcoded test_duration=103 in config.yml). We identify test clips
   from movie_timing.csv (82 TRs), which is the actual repeated-clip duration.

2. Data source: vicsompy loads CIFTI dtseries files with all grayordinates
   pre-extracted. We load per-hemisphere MAT files (32492 verts × T) and
   derive the 59k grayordinate targets from the template CIFTI's BrainModelAxis.

3. Z-scoring axis: vicsompy does scipy.stats.zscore(run[:, :], axis=1) on data
   shaped (n_verts, T) — i.e., per-vertex z-score over time. We replicate this
   exactly: z-score over the time axis for each run block independently.

All other parameters (TR=1, design-matrix column ordering) are identical to vicsompy.
n_LBOEs is capped per ROI: min(200, n_verts_L-2, n_verts_R-2), so A1 uses fewer
eigenfunctions than V1 when the A1 Glasser patch has fewer than 202 vertices/hem.

Run
---
    conda activate vicsompy_av
    python 02_prep_hcp_timeseries.py
"""

# =============================================================================
# CONFIG
# =============================================================================

DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"

MAT_LEFT  = f"{DATA_BASE}/HCP Data/notmean_left_Meanfile.mat"
MAT_RIGHT = f"{DATA_BASE}/HCP Data/notmean_right_Meanfile.mat"

MOVIE_TIMING_CSV = f"{DATA_BASE}/Data/movie_timing.csv"

TEMPLATE_CIFTI = (
    f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
    "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
)

CACHE_DIR = "/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/subsurfaces"
PREP_DIR  = "/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/prep"

# Validation clips (last video of each run) — the repeated segments
TEST_VIDEO_IDS = ["video5", "video9", "video14", "video18"]

TR = 1.0   # seconds; matches vicsompy config (analysis: TR: 1.0)

# N_LBOE is intentionally not hardcoded here — the actual count per ROI is read
# from the n_lboe attribute of the cached Subsurface objects produced by Script 1.
# (Small ROIs such as A1 may have fewer vertices than the 200-eigenfunction target.)

# =============================================================================
# IMPORTS
# =============================================================================

import os
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
os.makedirs(PREP_DIR, exist_ok=True)

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
            # fallback: largest array
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

    # Ensure shape is (32492, T) — orient so verts are rows
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

    New columns added:
        mat_start  — first TR index for this video in the MAT array
        mat_end    — one-past-last TR index
    """
    df = pd.read_csv(timing_csv)
    # Sort by onset_sec (not video_id) — "video10" sorts before "video2" lexicographically,
    # which would give wrong cumulative mat_start/mat_end offsets.
    df = df.sort_values("onset_sec").reset_index(drop=True)

    # The MAT data is the 18 videos concatenated in order with no gaps.
    # cumulative_start = sum of durations of all preceding videos.
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
    """Apply the vicsompy train/test split and per-run z-scoring.

    vicsompy reference (HcpSubject.split_test_sequence):
        for each run v (shape n_verts × T_run):
            test_data[run]  = zscore(v[:, -test_duration:], axis=1)
            train_data[run] = zscore(v[:, :-test_duration:], axis=1)
        concatenated_test = hstack(test_data)  ← we replicate this exactly

    Here we replicate the per-run z-score on the training block AND on each
    82-TR test clip, then concatenate the test clips → 328-TR test array.

    Parameters
    ----------
    left_data  : (32492, T_total)
    right_data : (32492, T_total)
    timing_df  : DataFrame with mat_start, mat_end, run_id, video_id

    Returns
    -------
    left_train_runs   : list of (32492, T_run_i) arrays — one per run, z-scored
    right_train_runs  : same for right hemisphere
    left_test_clips   : list of (32492, 82) — one per test clip, z-scored
    right_test_clips  : same for right hemisphere
    run_onsets_train  : (4,) int array — cumulative onset of each run in train concat
    """
    test_ids = set(TEST_VIDEO_IDS)
    run_ids  = sorted(timing_df["run_id"].unique())

    left_train_runs  = []
    right_train_runs = []
    left_test_clips  = []
    right_test_clips = []

    for run_id in run_ids:
        run_rows = timing_df[timing_df["run_id"] == run_id].sort_values("video_id")

        # --- Test clip (last video of this run) ---
        test_row = run_rows[run_rows["video_id"].isin(test_ids)]
        if len(test_row) != 1:
            raise ValueError(f"Expected 1 test clip in run {run_id}, got {len(test_row)}")
        ts = test_row.iloc[0]["mat_start"]
        te = test_row.iloc[0]["mat_end"]

        # z-score per-vertex over the test clip's time axis (matches vicsompy axis=1)
        L_test = scipy.stats.zscore(left_data[:, ts:te],  axis=1)
        R_test = scipy.stats.zscore(right_data[:, ts:te], axis=1)
        left_test_clips.append(L_test.astype(np.float32))
        right_test_clips.append(R_test.astype(np.float32))

        # --- Training clips (all other videos in this run) ---
        train_rows = run_rows[~run_rows["video_id"].isin(test_ids)]
        if len(train_rows) == 0:
            raise ValueError(f"No training videos found for run {run_id}")

        # Collect all training TRs for this run as one contiguous block
        train_indices = []
        for _, row in train_rows.iterrows():
            train_indices.extend(range(row["mat_start"], row["mat_end"]))
        train_indices = np.array(train_indices)

        L_train_run = left_data[:, train_indices]    # (32492, T_run)
        R_train_run = right_data[:, train_indices]

        # z-score per-vertex over this run's training time axis
        L_train_run = scipy.stats.zscore(L_train_run, axis=1).astype(np.float32)
        R_train_run = scipy.stats.zscore(R_train_run, axis=1).astype(np.float32)

        left_train_runs.append(L_train_run)
        right_train_runs.append(R_train_run)

        log.info(f"  Run {run_id}: train={L_train_run.shape[1]} TRs, "
                 f"test={L_test.shape[1]} TRs (z-scored per block)")

    # run_onsets: cumulative start indices in the concatenated training array
    run_lengths  = [r.shape[1] for r in left_train_runs]
    run_onsets   = np.concatenate([[0], np.cumsum(run_lengths)])[:-1].astype(int)
    log.info(f"  Training run_onsets (for LORO-CV): {run_onsets.tolist()}")
    log.info(f"  Total training TRs: {sum(run_lengths)}")

    return left_train_runs, right_train_runs, left_test_clips, right_test_clips, run_onsets


# =============================================================================
# STEP 3 — Extract grayordinate target indices from template CIFTI
# =============================================================================

def get_grayordinate_indices(template_cifti_path):
    """Return vertex index arrays (in 32k surface space) for each cortex
    structure as listed in the template dscalar.nii BrainModelAxis.

    Returns
    -------
    left_verts  : (n_left_gray,)  indices into 32k-left surface
    right_verts : (n_right_gray,) indices into 32k-right surface
    template_img: loaded Cifti2Image (kept for header reuse in Script 3)
    """
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

    n_total = len(left_verts) + len(right_verts)
    log.info(f"  Grayordinate vertices — left: {len(left_verts)}, "
             f"right: {len(right_verts)}, total: {n_total}")
    return left_verts, right_verts, template_img


# =============================================================================
# STEP 4 — Project BOLD onto LBOEs → design matrix columns
# =============================================================================

def project_onto_lboes(left_data, right_data, sub_a1, sub_v1):
    """Project BOLD timeseries for A1 and V1 onto the precomputed LBOEs.

    Mirrors vicsompy's make_dm() logic:
        dm_mod_hem = data[subsurface_verts_hem, :].T  @  eigenvectors.real
                   = (T, n_roi_verts)  @  (n_roi_verts, n_lboe)
                   = (T, n_lboe)

    Design matrix column order (matching vicsompy's modality ordering):
        cols   0 .. 2*n_lboe_a1-1          : Left A1 | Right A1  (audio band)
        cols   2*n_lboe_a1 .. end           : Left V1 | Right V1  (video band)

    n_lboe_a1 and n_lboe_v1 may be < 200 when the ROI has fewer vertices than
    the requested eigenfunction count (e.g. A1 with ~70 vertices per hemisphere).

    Parameters
    ----------
    left_data  : (32492, T) — for this sample block
    right_data : (32492, T)
    sub_a1, sub_v1 : loaded Subsurface objects (no pycortex Surface objs needed)

    Returns
    -------
    X : (T, 2*n_lboe_a1 + 2*n_lboe_v1) float32
    """
    # Combined surface data (64984, T) — right-hem verts are offset by 32492
    # (Subsurface.generate() stores subsurface_verts_R with +32492 offset already)
    combined = np.vstack([left_data, right_data])   # (64984, T)

    def _proj(verts, eigvecs):
        # verts   : 1D array of indices into combined rows
        # eigvecs : (n_roi_verts, n_lboe) complex — take .real
        return combined[verts, :].T @ eigvecs.real  # (T, n_lboe)

    dm_la1 = _proj(sub_a1.subsurface_verts_L, sub_a1.L_eigenvectors)
    dm_ra1 = _proj(sub_a1.subsurface_verts_R, sub_a1.R_eigenvectors)
    dm_lv1 = _proj(sub_v1.subsurface_verts_L, sub_v1.L_eigenvectors)
    dm_rv1 = _proj(sub_v1.subsurface_verts_R, sub_v1.R_eigenvectors)

    return np.hstack([dm_la1, dm_ra1, dm_lv1, dm_rv1]).astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Script 2 — Prepare HCP timeseries")
    log.info("=" * 60)

    # --- Load LBOEs from Script 1 ---
    log.info("\nLoading subsurfaces from Script 1 …")
    for name in ["sub_a1", "sub_v1"]:
        path = os.path.join(CACHE_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run 01_extract_geometry.py first."
            )

    with open(os.path.join(CACHE_DIR, "sub_a1.pkl"), "rb") as fh:
        sub_a1 = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, "sub_v1.pkl"), "rb") as fh:
        sub_v1 = pickle.load(fh)

    # Backward-compat: old pickles may lack n_lboe attribute
    for sub, tag in [(sub_a1, "a1"), (sub_v1, "v1")]:
        if not hasattr(sub, "n_lboe"):
            sub.n_lboe = sub.L_eigenvectors.shape[1]
    log.info(f"  A1 — L eigenvectors: {sub_a1.L_eigenvectors.shape}, "
             f"R eigenvectors: {sub_a1.R_eigenvectors.shape}  (n_lboe={sub_a1.n_lboe})")
    log.info(f"  V1 — L eigenvectors: {sub_v1.L_eigenvectors.shape}, "
             f"R eigenvectors: {sub_v1.R_eigenvectors.shape}  (n_lboe={sub_v1.n_lboe})")

    # --- Load MAT data ---
    log.info("\nLoading fMRI MAT files …")
    left_data  = load_mat_file(MAT_LEFT)    # (32492, T_total)
    right_data = load_mat_file(MAT_RIGHT)   # (32492, T_total)

    if left_data.shape != right_data.shape:
        raise ValueError(
            f"Shape mismatch: left={left_data.shape}, right={right_data.shape}"
        )
    T_total = left_data.shape[1]
    log.info(f"  T_total = {T_total} TRs across all 18 videos")

    # --- Parse timing CSV ---
    log.info("\nParsing movie_timing.csv …")
    timing_df = parse_timing(MOVIE_TIMING_CSV)

    expected_T = int(timing_df["duration_sec"].sum())
    if T_total != expected_T:
        log.warning(
            f"  T_total={T_total} != expected {expected_T} (sum of video durations). "
            "Check that the MAT files contain exactly the 18 video segments."
        )

    # --- Train/test split with per-run z-scoring ---
    log.info("\nApplying train/test split and per-run z-scoring …")
    (L_train_runs, R_train_runs,
     L_test_clips, R_test_clips,
     run_onsets) = split_and_zscore(left_data, right_data, timing_df)

    # Free raw data from memory
    del left_data, right_data

    # Concatenate training runs
    L_train = np.hstack(L_train_runs)   # (32492, T_train)
    R_train = np.hstack(R_train_runs)   # (32492, T_train)
    T_train = L_train.shape[1]
    log.info(f"  Training data: (32492, {T_train}) per hemisphere")

    # Concatenate z-scored test clips → 4 × 82 = 328-TR test array
    # vicsompy equivalent: concatenated_test_sequence = hstack(test_data)
    L_test_concat = np.hstack(L_test_clips)  # (32492, 328)
    R_test_concat = np.hstack(R_test_clips)  # (32492, 328)
    T_test = L_test_concat.shape[1]
    log.info(f"  Test data (concatenated 4×82): (32492, {T_test}) per hemisphere")

    # --- Grayordinate target indices from template CIFTI ---
    log.info("\nExtracting grayordinate vertex indices from template CIFTI …")
    left_gray_verts, right_gray_verts, _ = get_grayordinate_indices(TEMPLATE_CIFTI)

    # Y_train: (T_train, 59412) — targets for himalaya (n_samples, n_targets)
    Y_train = np.hstack([
        L_train[left_gray_verts,  :].T,   # (T_train, n_left_gray)
        R_train[right_gray_verts, :].T,   # (T_train, n_right_gray)
    ]).astype(np.float32)

    # Y_test: (328, 59412)
    Y_test = np.hstack([
        L_test_concat[left_gray_verts,  :].T,
        R_test_concat[right_gray_verts, :].T,
    ]).astype(np.float32)

    log.info(f"  Y_train: {Y_train.shape}")
    log.info(f"  Y_test : {Y_test.shape}")

    # --- Design matrix: project BOLD onto LBOEs ---
    log.info("\nProjecting BOLD onto LBOEs …")

    X_train = project_onto_lboes(L_train, R_train, sub_a1, sub_v1)
    n_audio_cols = 2 * sub_a1.n_lboe
    n_video_cols = 2 * sub_v1.n_lboe
    log.info(f"  X_train: {X_train.shape}  "
             f"(Audio cols 0-{n_audio_cols-1} [{n_audio_cols} cols], "
             f"Video cols {n_audio_cols}-{n_audio_cols+n_video_cols-1} [{n_video_cols} cols])")

    X_test = project_onto_lboes(L_test_concat, R_test_concat, sub_a1, sub_v1)
    log.info(f"  X_test : {X_test.shape}")

    # --- Save outputs ---
    log.info(f"\nSaving to {PREP_DIR} …")
    np.save(os.path.join(PREP_DIR, "X_train.npy"),    X_train)
    np.save(os.path.join(PREP_DIR, "Y_train.npy"),    Y_train)
    np.save(os.path.join(PREP_DIR, "X_test.npy"),     X_test)
    np.save(os.path.join(PREP_DIR, "Y_test.npy"),     Y_test)
    np.save(os.path.join(PREP_DIR, "run_onsets.npy"), run_onsets)
    # Band sizes: [n_audio_cols, n_video_cols] — read by Script 3 to set
    # ColumnKernelizer slices without re-loading the subsurface pickles.
    np.save(os.path.join(PREP_DIR, "band_sizes.npy"),
            np.array([n_audio_cols, n_video_cols], dtype=np.int32))

    log.info("  Saved: X_train.npy  Y_train.npy  X_test.npy  "
             "Y_test.npy  run_onsets.npy  band_sizes.npy")
    log.info("\nScript 2 complete.")


if __name__ == "__main__":
    main()
