"""
04_null_corrected_maps.py
=========================
Null-model correction mirroring Hedger et al. (2025) Figure 3a methodology.

The banded ridge split R² (R2_audio, R2_video from Script 03) reflects how well
the full *topographic* pattern of A1/V1 predicts each vertex. But some of that
prediction comes from a non-topographic component: the overall mean activity level
of A1 (or V1) at each timepoint. Any vertex that simply tracks the global A1
signal will show R2_audio > 0 regardless of whether it has a specific topographic
preference within A1.

The null model isolates topographic specificity:
    null_audio: predict Y from mean(A1 activity) alone — a single regressor
    null_video: predict Y from mean(V1 activity) alone

    R2_audio_nc = R2_audio - R2_null_audio   ← topographic A1 specificity only
    R2_video_nc = R2_video - R2_null_video   ← topographic V1 specificity only

Positive null-corrected values = the topographic (spatially structured) CF model
outperforms the non-topographic mean regressor. These are the maps that go into
the 2D colormap in Figure 3a of the paper.

Mean time-course reconstruction from LBOEs (avoids reloading 786 MB MAT files):
    X_test[:, 0:64]  = BOLD_A1_L @ eigvec_L,  so mean BOLD_A1_L ≈ X_test @ mean_evec_L
    where mean_evec_L = eigvec_L.mean(axis=0)  (mean over the 77 A1 left vertices)

Inputs (from Scripts 02-03):
    PREP_DIR/X_test.npy          (328, 528)
    PREP_DIR/Y_test.npy          (328, 59412)
    PREP_DIR/band_sizes.npy      [128, 400]
    CACHE_DIR/sub_a1.pkl         eigenvectors for mean reconstruction
    CACHE_DIR/sub_v1.pkl
    OUTPUT_DIR/R2_audio.npy      (59412,)  split R² from Script 03
    OUTPUT_DIR/R2_video.npy      (59412,)

Outputs:
    OUTPUT_DIR/R2_null_audio.{npy,dscalar.nii}
    OUTPUT_DIR/R2_null_video.{npy,dscalar.nii}
    OUTPUT_DIR/R2_audio_nc.{npy,dscalar.nii}   ← primary output
    OUTPUT_DIR/R2_video_nc.{npy,dscalar.nii}   ← primary output

Run
---
    conda activate vicsompy_av
    python 04_null_corrected_maps.py
"""

# =============================================================================
# CONFIG
# =============================================================================

DATA_BASE  = "/home/amin/Research/Representation/Movie/data/Setareh"
TEMPLATE_CIFTI = (f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
                  "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii")

OUTPUT_DIR = "/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual"
PREP_DIR   = f"{OUTPUT_DIR}/prep"
CACHE_DIR  = f"{OUTPUT_DIR}/subsurfaces"
CIFTI_DIR  = f"{OUTPUT_DIR}/cifti_maps"

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pickle
import logging

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(CIFTI_DIR, exist_ok=True)

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


def reconstruct_mean_timecourse(X_band, eigvec_L, eigvec_R, n_lboe):
    """Reconstruct the mean ROI time course from LBOE projections.

    X_band is the (T, 2*n_lboe) design matrix columns for one source region.
    mean BOLD over L vertices ≈ X_band[:, :n_lboe] @ eigvec_L.mean(axis=0)
    because X(t,k) = sum_v BOLD(v,t)*eigvec_L(v,k), so
    sum_k mean_v(eigvec_L[:,k]) * X(t,k) ≈ mean_v BOLD(v,t).
    Bilateral mean combines both hemispheres weighted by vertex count.
    """
    n_L = eigvec_L.shape[0]
    n_R = eigvec_R.shape[0]

    mean_evec_L = eigvec_L.real.mean(axis=0)   # (n_lboe,)
    mean_evec_R = eigvec_R.real.mean(axis=0)   # (n_lboe,)

    mean_L = X_band[:, :n_lboe]        @ mean_evec_L   # (T,)
    mean_R = X_band[:, n_lboe:2*n_lboe] @ mean_evec_R  # (T,)

    # Weighted bilateral mean
    return (mean_L * n_L + mean_R * n_R) / (n_L + n_R)  # (T,)


def fit_null_r2(regressor, Y):
    """Fit a single-regressor OLS null model and return R² for all targets.

    Model: Y ~ beta_0 + beta_1 * regressor
    Vectorised across all 59412 targets at once.

    Parameters
    ----------
    regressor : (T,)
    Y         : (T, 59412)

    Returns
    -------
    R2_null : (59412,)
    """
    T = Y.shape[0]
    dm = np.column_stack([regressor, np.ones(T)])           # (T, 2)
    betas, _, _, _ = np.linalg.lstsq(dm, Y, rcond=None)    # (2, 59412)
    Y_hat = dm @ betas                                       # (T, 59412)

    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)               # (59412,)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)      # (59412,)
    # guard against zero-variance targets
    R2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return R2.astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Script 04 — Null-corrected R² maps (Hedger et al. Fig 3a method)")
    log.info("=" * 60)

    # --- Load subsurface eigenvectors ---
    log.info("\nLoading subsurface pickles …")
    with open(os.path.join(CACHE_DIR, "sub_a1.pkl"), "rb") as fh:
        sub_a1 = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, "sub_v1.pkl"), "rb") as fh:
        sub_v1 = pickle.load(fh)
    if not hasattr(sub_a1, "n_lboe"):
        sub_a1.n_lboe = sub_a1.L_eigenvectors.shape[1]
    if not hasattr(sub_v1, "n_lboe"):
        sub_v1.n_lboe = sub_v1.L_eigenvectors.shape[1]
    log.info(f"  A1: n_lboe={sub_a1.n_lboe}  L={sub_a1.L_eigenvectors.shape}  R={sub_a1.R_eigenvectors.shape}")
    log.info(f"  V1: n_lboe={sub_v1.n_lboe}  L={sub_v1.L_eigenvectors.shape}  R={sub_v1.R_eigenvectors.shape}")

    # --- Load design matrix and targets ---
    log.info("\nLoading test arrays …")
    X_test     = np.load(os.path.join(PREP_DIR, "X_test.npy"))      # (328, 528)
    Y_test     = np.load(os.path.join(PREP_DIR, "Y_test.npy"))      # (328, 59412)
    band_sizes = np.load(os.path.join(PREP_DIR, "band_sizes.npy"))
    n_audio    = int(band_sizes[0])   # 128
    n_lboe_a1  = sub_a1.n_lboe       # 64
    n_lboe_v1  = sub_v1.n_lboe       # 200
    log.info(f"  X_test: {X_test.shape}   Y_test: {Y_test.shape}")

    # --- Load existing split R² from Script 03 ---
    log.info("\nLoading split R² from Script 03 …")
    R2_audio = np.load(os.path.join(OUTPUT_DIR, "R2_audio.npy"))   # (59412,)
    R2_video = np.load(os.path.join(OUTPUT_DIR, "R2_video.npy"))   # (59412,)
    log.info(f"  R2_audio: mean={R2_audio.mean():.4f}  frac>0={np.mean(R2_audio>0):.1%}")
    log.info(f"  R2_video: mean={R2_video.mean():.4f}  frac>0={np.mean(R2_video>0):.1%}")

    # --- Reconstruct mean time courses from LBOE projections ---
    # This approximates the "mean A1/V1 regressor" used in Hedger et al.'s null model
    # without reloading the 786 MB MAT files.
    log.info("\nReconstructing mean A1/V1 time courses from LBOE projections …")
    X_audio = X_test[:, :n_audio]                         # (328, 128) A1 band
    X_video = X_test[:, n_audio:]                         # (328, 400) V1 band

    mean_a1 = reconstruct_mean_timecourse(
        X_audio, sub_a1.L_eigenvectors, sub_a1.R_eigenvectors, n_lboe_a1)
    mean_v1 = reconstruct_mean_timecourse(
        X_video, sub_v1.L_eigenvectors, sub_v1.R_eigenvectors, n_lboe_v1)

    log.info(f"  mean_a1: shape={mean_a1.shape}  std={mean_a1.std():.4f}")
    log.info(f"  mean_v1: shape={mean_v1.shape}  std={mean_v1.std():.4f}")

    # --- Fit null models ---
    # null_audio: how much does the MEAN A1 level alone predict each vertex?
    # Any R2 here reflects global A1 responsiveness, not topographic specificity.
    log.info("\nFitting null model for audio (mean A1 regressor) …")
    R2_null_audio = fit_null_r2(mean_a1, Y_test)

    log.info("Fitting null model for video (mean V1 regressor) …")
    R2_null_video = fit_null_r2(mean_v1, Y_test)

    log.info(f"  R2_null_audio: mean={R2_null_audio.mean():.4f}  frac>0={np.mean(R2_null_audio>0):.1%}")
    log.info(f"  R2_null_video: mean={R2_null_video.mean():.4f}  frac>0={np.mean(R2_null_video>0):.1%}")

    # --- Null-corrected maps ---
    # Positive values: the topographic CF model outperforms the mean regressor.
    # These are the maps that go into the Figure 3a 2D colormap.
    # Not clipped: negative values are informative (null model beats CF model there).
    R2_audio_nc = R2_audio - R2_null_audio
    R2_video_nc = R2_video - R2_null_video
    log.info(f"\n  R2_audio_nc: mean={R2_audio_nc.mean():.4f}  frac>0={np.mean(R2_audio_nc>0):.1%}")
    log.info(f"  R2_video_nc: mean={R2_video_nc.mean():.4f}  frac>0={np.mean(R2_video_nc>0):.1%}")

    # --- Save all maps ---
    log.info(f"\nSaving maps to {OUTPUT_DIR} …")
    template = nib.load(TEMPLATE_CIFTI)
    save_map(R2_null_audio, "R2_null_audio", template)
    save_map(R2_null_video, "R2_null_video", template)
    save_map(R2_audio_nc,   "R2_audio_nc",   template)
    save_map(R2_video_nc,   "R2_video_nc",   template)

    log.info("\nScript 04 complete.")
    log.info("  Next: python 05_audiovisual_integration_maps.py")


if __name__ == "__main__":
    main()
