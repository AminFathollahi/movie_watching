"""
04_null_corrected_maps.py
=========================
Null-model correction mirroring Hedger et al. (2025) Figure 3a methodology.

The banded ridge split R² (R2_{AUDIO_ROI}, R2_{VIDEO_ROI} from Script 03)
reflects how well the topographic pattern of the audio/video ROI predicts each
vertex. Some of that prediction is non-topographic (global mean activity).

The null model isolates topographic specificity:
    null_audio: predict Y from mean(audio ROI activity) — single regressor
    null_video: predict Y from mean(video ROI activity)

    R2_{AUDIO_ROI}_nc = R2_{AUDIO_ROI} - R2_null_{AUDIO_ROI}
    R2_{VIDEO_ROI}_nc = R2_{VIDEO_ROI} - R2_null_{VIDEO_ROI}

Inputs (from Scripts 02-03):
    PREP_DIR/X_test.npy, Y_test.npy, band_sizes.npy
    CACHE_DIR/sub_{AUDIO_ROI}.pkl, sub_{VIDEO_ROI}.pkl
    OUTPUT_DIR/R2_{AUDIO_ROI}.npy, R2_{VIDEO_ROI}.npy  (from Script 03)

Outputs:
    OUTPUT_DIR/R2_null_{AUDIO_ROI}.{npy,dscalar.nii}
    OUTPUT_DIR/R2_null_{VIDEO_ROI}.{npy,dscalar.nii}
    OUTPUT_DIR/R2_{AUDIO_ROI}_nc.{npy,dscalar.nii}   ← primary output
    OUTPUT_DIR/R2_{VIDEO_ROI}_nc.{npy,dscalar.nii}   ← primary output

Run
---
    conda activate vicsompy_av
    python 04_null_corrected_maps.py
"""


# =============================================================================
# CONFIG
# =============================================================================

import os

DATA_BASE  = "/home/amin/Research/Representation/Movie/data/Setareh"
TEMPLATE_CIFTI = (f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
                  "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii")

# ── ROI selection ─────────────────────────────────────────────────────────────
# Must match the ROI_DEFS / AUDIO_ROI / VIDEO_ROI set in 01_extract_geometry.py.
AUDIO_ROI = os.getenv("AUDIO_ROI")
VIDEO_ROI = os.getenv("VIDEO_ROI")

OUTPUT_DIR = f"/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual/{AUDIO_ROI}_{VIDEO_ROI}"
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
    with open(os.path.join(CACHE_DIR, f"sub_{AUDIO_ROI.lower()}.pkl"), "rb") as fh:
        sub_audio = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, f"sub_{VIDEO_ROI.lower()}.pkl"), "rb") as fh:
        sub_video = pickle.load(fh)
    if not hasattr(sub_audio, "n_lboe"):
        sub_audio.n_lboe = sub_audio.L_eigenvectors.shape[1]
    if not hasattr(sub_video, "n_lboe"):
        sub_video.n_lboe = sub_video.L_eigenvectors.shape[1]
    log.info(f"  {AUDIO_ROI.upper()}: n_lboe={sub_audio.n_lboe}  "
             f"L={sub_audio.L_eigenvectors.shape}  R={sub_audio.R_eigenvectors.shape}")
    log.info(f"  {VIDEO_ROI.upper()}: n_lboe={sub_video.n_lboe}  "
             f"L={sub_video.L_eigenvectors.shape}  R={sub_video.R_eigenvectors.shape}")

    # --- Load design matrix and targets ---
    log.info("\nLoading test arrays …")
    X_test     = np.load(os.path.join(PREP_DIR, "X_test.npy"))
    Y_test     = np.load(os.path.join(PREP_DIR, "Y_test.npy"))
    band_sizes = np.load(os.path.join(PREP_DIR, "band_sizes.npy"))
    n_audio    = int(band_sizes[0])
    n_lboe_audio = sub_audio.n_lboe
    n_lboe_video = sub_video.n_lboe
    log.info(f"  X_test: {X_test.shape}   Y_test: {Y_test.shape}")

    # --- Load existing split R² from Script 03 ---
    log.info("\nLoading split R² from Script 03 …")
    R2_audio = np.load(os.path.join(OUTPUT_DIR, f"R2_{AUDIO_ROI}.npy"))
    R2_video = np.load(os.path.join(OUTPUT_DIR, f"R2_{VIDEO_ROI}.npy"))
    log.info(f"  R2_{AUDIO_ROI}: mean={R2_audio.mean():.4f}  frac>0={np.mean(R2_audio>0):.1%}")
    log.info(f"  R2_{VIDEO_ROI}: mean={R2_video.mean():.4f}  frac>0={np.mean(R2_video>0):.1%}")

    # --- Reconstruct mean time courses from LBOE projections ---
    log.info(f"\nReconstructing mean {AUDIO_ROI.upper()}/{VIDEO_ROI.upper()} "
             "time courses from LBOE projections …")
    X_audio = X_test[:, :n_audio]
    X_video = X_test[:, n_audio:]

    mean_audio = reconstruct_mean_timecourse(
        X_audio, sub_audio.L_eigenvectors, sub_audio.R_eigenvectors, n_lboe_audio)
    mean_video = reconstruct_mean_timecourse(
        X_video, sub_video.L_eigenvectors, sub_video.R_eigenvectors, n_lboe_video)

    log.info(f"  mean_{AUDIO_ROI}: shape={mean_audio.shape}  std={mean_audio.std():.4f}")
    log.info(f"  mean_{VIDEO_ROI}: shape={mean_video.shape}  std={mean_video.std():.4f}")

    # --- Fit null models ---
    log.info(f"\nFitting null model for audio (mean {AUDIO_ROI.upper()} regressor) …")
    R2_null_audio = fit_null_r2(mean_audio, Y_test)

    log.info(f"Fitting null model for video (mean {VIDEO_ROI.upper()} regressor) …")
    R2_null_video = fit_null_r2(mean_video, Y_test)

    log.info(f"  R2_null_{AUDIO_ROI}: mean={R2_null_audio.mean():.4f}  "
             f"frac>0={np.mean(R2_null_audio>0):.1%}")
    log.info(f"  R2_null_{VIDEO_ROI}: mean={R2_null_video.mean():.4f}  "
             f"frac>0={np.mean(R2_null_video>0):.1%}")

    # --- Null-corrected maps ---
    R2_audio_nc = R2_audio - R2_null_audio
    R2_video_nc = R2_video - R2_null_video
    log.info(f"\n  R2_{AUDIO_ROI}_nc: mean={R2_audio_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_audio_nc>0):.1%}")
    log.info(f"  R2_{VIDEO_ROI}_nc: mean={R2_video_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_video_nc>0):.1%}")

    # --- Save all maps ---
    log.info(f"\nSaving maps to {OUTPUT_DIR} …")
    template = nib.load(TEMPLATE_CIFTI)
    save_map(R2_null_audio, f"R2_null_{AUDIO_ROI}", template)
    save_map(R2_null_video, f"R2_null_{VIDEO_ROI}", template)
    save_map(R2_audio_nc,   f"R2_{AUDIO_ROI}_nc",   template)
    save_map(R2_video_nc,   f"R2_{VIDEO_ROI}_nc",   template)

    log.info("\nScript 04 complete.")
    log.info("  Next: python 05_audiovisual_integration_maps.py")


if __name__ == "__main__":
    main()
