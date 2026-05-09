"""
03_fit_banded_ridge.py
======================
Phase 3: Load the pre-processed arrays from Script 2, fit a himalaya
MultipleKernelRidgeCV (banded ridge) model, evaluate on the held-out test
set, compute variance partitioning maps, and export CIFTI dscalar maps.

Cross-validation strategy
--------------------------
Hyperparameter tuning (alpha selection) uses leave-one-run-out (LORO) CV on
X_train / Y_train, respecting the run_onsets from Script 2.
Final evaluation is performed on the independent X_test / Y_test (4 × 82 = 328
concatenated test TRs, never seen during fitting).

Variance decomposition
----------------------
For two bands (audio = AUDIO_ROI, video = VIDEO_ROI):
    Y_hat_full  = pipeline.predict(X_test)              → full model
    Y_hat_split = pipeline.predict(X_test, split=True)  → per-band predictions
    R2_full  = r2_score(Y_test, Y_hat_full)             → (59412,)
    [R2_audio, R2_video] = r2_score_split(Y_test, Y_hat_split)
    Shared_R2 = R2_audio + R2_video - R2_full

ColumnKernelizer slices:
    'audio' → slice(0, n_audio_cols)              — L + R audio ROI LBOEs
    'video' → slice(n_audio_cols, X.shape[1])     — L + R video ROI LBOEs

Inputs (from Script 2):
    PREP_DIR/X_train.npy, Y_train.npy, X_test.npy, Y_test.npy, run_onsets.npy

Outputs:
    OUTPUT_DIR/R2_full.npy, R2_{AUDIO_ROI}.npy, R2_{VIDEO_ROI}.npy, Shared_R2.npy
    CIFTI_DIR/R2_full.dscalar.nii, R2_{AUDIO_ROI}.dscalar.nii,
              R2_{VIDEO_ROI}.dscalar.nii, Shared_R2.dscalar.nii

Run
---
    conda activate vicsompy_av
    python 03_fit_banded_ridge.py
"""
# =============================================================================
# CONFIG
# =============================================================================
import os
DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"

TEMPLATE_CIFTI = (
    f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
    "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
)

# ── ROI selection ─────────────────────────────────────────────────────────────
# Must match the ROI_DEFS / AUDIO_ROI / VIDEO_ROI set in 01_extract_geometry.py.
# Keys drive output file names: R2_<AUDIO_ROI>.npy, R2_<VIDEO_ROI>.npy, etc.
AUDIO_ROI = os.getenv("AUDIO_ROI")
VIDEO_ROI = os.getenv("VIDEO_ROI")

OUTPUT_DIR = f"/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/{AUDIO_ROI}_{VIDEO_ROI}"
PREP_DIR   = f"{OUTPUT_DIR}/prep"
CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"
CIFTI_DIR  = f"{OUTPUT_DIR}/cifti_maps"

# himalaya model parameters — all identical to vicsompy config.yml
BACKEND_ENGINE        = "torch"        # vicsompy: backend_engine: "torch"
SOLVER                = "random_search"# vicsompy: solver: "random_search"
N_ITER                = 20             # vicsompy: n_iter: 20
ALPHA_MIN             = 1              # vicsompy: alpha_min: 1
ALPHA_MAX             = 20             # vicsompy: alpha_max: 20
ALPHA_VALS            = 20             # vicsompy: alpha_vals: 20
N_TARGETS_BATCH       = 400            # vicsompy: n_targets_batch: 400
N_ALPHAS_BATCH        = 10             # vicsompy: n_alphas_batch: 10
N_TARGETS_BATCH_REFIT = 400            # vicsompy: n_targets_batch_refit: 400

# Pre-processing — identical to vicsompy config.yml
WITH_MEAN = True   # vicsompy: with_mean: True
WITH_STD  = True   # vicsompy: with_std: True

# N_LBOE is not hardcoded here — band sizes are read from band_sizes.npy
# produced by Script 2 (which reads actual n_lboe from the cached Subsurfaces).
# =============================================================================
# IMPORTS
# =============================================================================
import os
import logging

import numpy as np
import nibabel as nib

from himalaya.backend import set_backend
from himalaya.kernel_ridge import (
    ColumnKernelizer,
    Kernelizer,
    MultipleKernelRidgeCV,
)
from himalaya.scoring import r2_score, r2_score_split

from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# HELPER
# =============================================================================
# generate_leave_one_run_out is inlined here to avoid vicsompy.utils which has a
# top-level `import pkg_resources` (setuptools) not available in vicsompy_av.
# The function is copied verbatim from vicsompy/utils.py (originally from the
# gallantlab voxelwise_tutorials package).
def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):
    """Generate a leave-one-run-out split for cross-validation.

    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples   : int — total number of samples in the training set
    run_onsets  : array of int (n_runs,) — indices of the run onsets
    random_state: None | int | RandomState
    n_runs_out  : int — number of runs to leave out per fold (default 1)

    Yields
    ------
    train : array of int
    val   : array of int
    """
    from sklearn.utils.validation import check_random_state
    random_state = check_random_state(random_state)
    n_runs = len(run_onsets)
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])
    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError(
            "Some runs have no samples. Check that run_onsets does not "
            "include any repeated index, nor the last index.")
    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for d in [OUTPUT_DIR, CIFTI_DIR]:
    os.makedirs(d, exist_ok=True)

def be_to_npy(var, backend):
    """Convert a backend tensor (or list of tensors) to numpy."""
    if isinstance(var, list):
        return [backend.to_numpy(v) for v in var]
    return backend.to_numpy(var)


# =============================================================================
# STEP 1 — Load pre-processed arrays
# =============================================================================

def load_prep_data():
    log.info(f"Loading pre-processed arrays from {PREP_DIR} …")
    for fname in ["X_train", "Y_train", "X_test", "Y_test", "run_onsets", "band_sizes"]:
        path = os.path.join(PREP_DIR, f"{fname}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run 02_prep_hcp_timeseries.py first."
            )

    X_train    = np.load(os.path.join(PREP_DIR, "X_train.npy"))
    Y_train    = np.load(os.path.join(PREP_DIR, "Y_train.npy"))
    X_test     = np.load(os.path.join(PREP_DIR, "X_test.npy"))
    Y_test     = np.load(os.path.join(PREP_DIR, "Y_test.npy"))
    run_onsets = np.load(os.path.join(PREP_DIR, "run_onsets.npy"))
    band_sizes = np.load(os.path.join(PREP_DIR, "band_sizes.npy"))  # [n_audio, n_video]

    n_audio_cols = int(band_sizes[0])
    n_video_cols = int(band_sizes[1])
    log.info(f"  X_train   : {X_train.shape}  "
             f"(audio cols 0-{n_audio_cols-1}, video cols {n_audio_cols}-{n_audio_cols+n_video_cols-1})")
    log.info(f"  Y_train   : {Y_train.shape}  (T_train, 59412)")
    log.info(f"  X_test    : {X_test.shape}")
    log.info(f"  Y_test    : {Y_test.shape}   (328, 59412)")
    log.info(f"  run_onsets: {run_onsets.tolist()}")
    log.info(f"  band_sizes: audio={n_audio_cols} cols, video={n_video_cols} cols")

    return X_train, Y_train, X_test, Y_test, run_onsets, n_audio_cols, n_video_cols


# =============================================================================
# STEP 2 — Build himalaya pipeline (identical structure to vicsompy)
# =============================================================================

def build_pipeline(n_samples_train, run_onsets, n_audio_cols, n_video_cols, backend):
    """Construct the ColumnKernelizer → MultipleKernelRidgeCV pipeline.

    Replicates vicsompy's MssCf.prep_pipeline() sequence:
        make_preproc() → kernelize() → setup_model() → complete_pipeline()

    CV scheme: leave-one-run-out (vicsompy: generate_leave_one_run_out +
    check_cv), used only for alpha hyper-parameter selection during training.

    Parameters
    ----------
    n_audio_cols : int   — number of A1 LBOE columns (2 * n_lboe_a1)
    n_video_cols : int   — number of V1 LBOE columns (2 * n_lboe_v1)
    """
    # --- Cross-validation (LORO) ---
    # vicsompy: generate_leave_one_run_out(n_samples_train, run_durations)
    #           check_cv(cv)
    cv_gen = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv     = check_cv(cv_gen)

    n_runs = len(run_onsets)
    log.info(f"  CV: leave-one-run-out, {n_runs} runs, "
             f"run_onsets={run_onsets.tolist()}")

    # --- Preprocessing: StandardScaler → Kernelizer (linear kernel) ---
    # vicsompy config: with_mean=True, with_std=True, kernel="linear"
    # Each band gets its OWN scaler instance (safe clone per band)
    def make_preproc():
        return make_pipeline(
            StandardScaler(with_mean=WITH_MEAN, with_std=WITH_STD),
            Kernelizer(kernel="linear"),
        )

    # --- ColumnKernelizer: two bands ---
    # vicsompy's define_modality_indices() + kernelize():
    #   start_and_end = [0, n_audio_cols, n_audio_cols + n_video_cols]
    #   modality names = ["visual","somato"] → here ["audio","video"]
    # n_audio_cols = 2 * n_lboe_a1  (may be < 400 when A1 ROI is small)
    # n_video_cols = 2 * n_lboe_v1
    audio_slice = slice(0,              n_audio_cols)
    video_slice = slice(n_audio_cols,   n_audio_cols + n_video_cols)
    log.info(f"  ColumnKernelizer: audio={audio_slice}, video={video_slice}")

    column_kernelizer = ColumnKernelizer(
        [
            ("audio", make_preproc(), audio_slice),   # A1 LBOEs
            ("video", make_preproc(), video_slice),   # V1 LBOEs
        ]
    )

    # --- MultipleKernelRidgeCV (identical to vicsompy setup_model()) ---
    alphas = np.logspace(ALPHA_MIN, ALPHA_MAX, ALPHA_VALS)
    solver_params = dict(
        n_iter=N_ITER,
        alphas=alphas,
        n_targets_batch=N_TARGETS_BATCH,
        n_alphas_batch=N_ALPHAS_BATCH,
        n_targets_batch_refit=N_TARGETS_BATCH_REFIT,
    )
    model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=SOLVER,
        solver_params=solver_params,
        cv=cv,
    )

    # --- Full pipeline (vicsompy: complete_pipeline()) ---
    pipeline = make_pipeline(column_kernelizer, model)
    return pipeline


# =============================================================================
# STEP 3 — Fit, predict, and compute variance decomposition
# =============================================================================

def fit_and_decompose(X_train, Y_train, X_test, Y_test, run_onsets,
                      n_audio_cols, n_video_cols, backend):
    """Fit the banded ridge pipeline and compute test-set variance partitioning.

    Returns
    -------
    R2_full   : (59412,)  — full-model R² on test set
    R2_audio  : (59412,)  — audio-band contribution R² on test set
    R2_video  : (59412,)  — video-band contribution R² on test set
    Shared_R2 : (59412,)  — shared / audiovisual-integration component
    """
    n_samples_train = X_train.shape[0]
    pipeline = build_pipeline(n_samples_train, run_onsets,
                               n_audio_cols, n_video_cols, backend)

    # --- Fit (vicsompy: nm.fit(all_data)) ---
    log.info(f"\nFitting pipeline …  X_train={X_train.shape}, Y_train={Y_train.shape}")
    pipeline.fit(X_train, Y_train)
    log.info("  Fit complete.")

    # --- Full-model R² on test set (vicsompy: nm.test_xval → xval_score) ---
    log.info("Computing full-model R² on test set …")
    Y_hat_full = pipeline.predict(X_test)                          # (328, 59412)
    R2_full = be_to_npy(r2_score(Y_test, Y_hat_full), backend)    # (59412,)

    # --- Per-band R² on test set (vicsompy: nm.test_xval → test_split_scores) ---
    # pipeline.predict(X_test, split=True) returns a list of 2 tensors,
    # one per ColumnKernelizer band: [Y_hat_audio, Y_hat_video]
    log.info("Computing split (per-band) R² on test set …")
    Y_hat_split = pipeline.predict(X_test, split=True)            # list[2 × (328,59412)]
    split_scores = be_to_npy(
        r2_score_split(Y_test, Y_hat_split), backend
    )  # (2, 59412)

    R2_audio = split_scores[0]    # (59412,)
    R2_video = split_scores[1]    # (59412,)

    # --- Shared / audiovisual-integration variance ---
    # Standard variance-partitioning formula:
    #   Shared = R2_audio + R2_video − R2_full
    # Positive values indicate regions where audio and video spatial patterns
    # jointly predict activity (audiovisual integration).
    Shared_R2 = R2_audio + R2_video - R2_full                     # (59412,)

    log.info("\nVariance partitioning (test set) — means across vertices:")
    log.info(f"  R2_full        : {np.nanmean(R2_full):.4f}  "
             f"(fraction > 0.01: {np.mean(R2_full > 0.01):.1%})")
    log.info(f"  R2_{AUDIO_ROI:<6s}: {np.nanmean(R2_audio):.4f}")
    log.info(f"  R2_{VIDEO_ROI:<6s}: {np.nanmean(R2_video):.4f}")
    log.info(f"  Shared         : {np.nanmean(Shared_R2):.4f}")

    return R2_full, R2_audio, R2_video, Shared_R2


# =============================================================================
# STEP 4 — Save .npy and .dscalar.nii CIFTI maps
# =============================================================================

def save_results(R2_full, R2_audio, R2_video, Shared_R2):
    """Save the four variance maps as .npy and as CIFTI dscalar.nii.

    File names use the configured AUDIO_ROI / VIDEO_ROI keys so that outputs
    from different ROI pairs don't overwrite each other.
    """
    template_img = nib.load(TEMPLATE_CIFTI)

    maps = {
        "R2_full"              : R2_full,
        f"R2_{AUDIO_ROI}"     : R2_audio,
        f"R2_{VIDEO_ROI}"     : R2_video,
        "Shared_R2"            : Shared_R2,
    }

    for name, arr in maps.items():
        arr_f32 = arr.astype(np.float32)

        # .npy
        npy_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(npy_path, arr_f32)

        # .dscalar.nii
        data_2d   = arr_f32.reshape(1, -1)   # (1, 59412)
        cifti_img = nib.Cifti2Image(
            data_2d,
            header=template_img.header,
            nifti_header=template_img.nifti_header,
        )
        cifti_path = os.path.join(CIFTI_DIR, f"{name}.dscalar.nii")
        nib.save(cifti_img, cifti_path)

        log.info(f"  {name}: {npy_path}")
        log.info(f"  {name}: {cifti_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Script 3 — Fit banded ridge regression")
    log.info(f"  Backend       : {BACKEND_ENGINE}  (vicsompy default)")
    log.info(f"  Solver        : {SOLVER}  (vicsompy default)")
    log.info(f"  Alphas        : logspace({ALPHA_MIN},{ALPHA_MAX},{ALPHA_VALS})")
    log.info(f"  n_iter        : {N_ITER}")
    log.info(f"  Targets batch : {N_TARGETS_BATCH}")
    log.info("=" * 60)

    # --- Load backend ---
    log.info(f"\nInitialising himalaya backend: '{BACKEND_ENGINE}' …")
    backend = set_backend(BACKEND_ENGINE, on_error="warn")
    log.info(f"  Backend ready: {backend}")

    # --- Load data ---
    X_train, Y_train, X_test, Y_test, run_onsets, n_audio_cols, n_video_cols = (
        load_prep_data()
    )

    # --- Fit and decompose ---
    R2_full, R2_audio, R2_video, Shared_R2 = fit_and_decompose(
        X_train, Y_train, X_test, Y_test, run_onsets,
        n_audio_cols, n_video_cols, backend
    )

    # --- Save outputs ---
    log.info(f"\nSaving outputs …")
    save_results(R2_full, R2_audio, R2_video, Shared_R2)

    log.info(f"\nScript 3 complete.")
    log.info(f"  Numpy maps : {OUTPUT_DIR}/R2_*.npy")
    log.info(f"  CIFTI maps : {CIFTI_DIR}/R2_*.dscalar.nii")


if __name__ == "__main__":
    main()
