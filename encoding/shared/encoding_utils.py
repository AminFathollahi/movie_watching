"""
encoding/shared/encoding_utils.py
===================================
Shared utilities for ridge encoding models.

All functions are stateless and accept explicit arguments — no hardcoded paths
or parameters. Configure everything in analysis.sh and pass via CLI.

fMRI preprocessing convention
------------------------------
The input CIFTI is the output of preprocess_individual.py in continuous mode
(all 4 runs concatenated, 59412 grayordinates). Default preprocessing is raw
(no SG, no PSC, no GSR). build_fmri_arrays() uses the global onset_sec from
timing_df and run_trs to locate each clip, applies the haemodynamic delay,
bins to bin_sec resolution, z-scores each run's training bins and each run's
held-out clips separately, and splits by video_id.

This matches the convention used in rsa/shared/rsa_utils.py (preprocess_fmri).
"""

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import gamma


# =============================================================================
# HRF
# =============================================================================

def spm_hrf(tr: float, oversampling: int = 16) -> np.ndarray:
    """Generate an SPM canonical HRF sampled at the given TR.

    Args:
        tr: float — repetition time in seconds
        oversampling: int — internal oversampling factor for precision

    Returns:
        (n_trs,) float64 — HRF kernel, normalised to sum to 1
    """
    dt = tr / oversampling
    t = np.arange(0, 32 + dt, dt)
    hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    hrf = hrf[::oversampling]
    hrf /= hrf.sum()
    return hrf


# =============================================================================
# fMRI array builders
# =============================================================================

def _bin_and_split_fmri(fmri: np.ndarray, timing_df: pd.DataFrame,
                         test_video_ids: list, bin_sec: float, tr: float,
                         run_trs: np.ndarray, delay_sec: float = 0.0,
                         skip_sec: float = None) -> tuple:
    """Extract movie segments from continuous fMRI, z-score per run, bin, split.

    Uses global onset_sec from timing_df (same convention as rsa_utils.preprocess_fmri):
    onset_sec is cumulative across runs; run_start_sec is subtracted internally to
    get the within-run TR index.  Rest periods between clips are skipped.

    Responses are z-scored per run (Hedger et al. 2025): each run's training bins
    with their own mean/std, and each run's held-out clips with their own mean/std.

    Args:
        fmri          : (n_vertices, T_total) float32 — full continuous preprocessed signal
        timing_df     : DataFrame with columns: video_id, onset_sec (global), duration_sec, run_id
        test_video_ids: list[str] — video IDs held out for test set
        bin_sec       : float — temporal bin size in seconds
        tr            : float — TR in seconds
        run_trs       : (n_runs,) int — TRs per run (from preprocess_individual run_trs.npy)
        delay_sec     : float — haemodynamic shift applied to onset_sec (default 0;
                        apply pre-delay in preprocess_individual or pass here)
        skip_sec      : float — window stride in seconds (default: bin_sec, no overlap)

    Returns:
        Y_train    : (n_train_bins, n_vertices) float32
        Y_test     : (n_test_bins, n_vertices) float32
        run_onsets : list[int] — training run onset bin indices for LORO-CV
    """
    if skip_sec is None:
        skip_sec = bin_sec
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
    run_col  = 'run_id' if 'run_id' in timing_df.columns else 'run'
    n_verts  = fmri.shape[0]

    train_segs   = []
    test_segs    = []
    run_onsets   = []
    n_train_bins = 0
    global_tr_offset = 0

    for run_idx, (run_id, run_df) in enumerate(
            timing_df.groupby(run_col, sort=True)):
        run_tr_count = int(run_trs[run_idx])
        run_data     = fmri[:, global_tr_offset: global_tr_offset + run_tr_count]
        global_tr_offset += run_tr_count

        run_start_sec = float(np.sum(run_trs[:run_idx])) * tr

        run_train_segs: list[np.ndarray] = []
        run_test_segs:  list[np.ndarray] = []

        for _, row in run_df.iterrows():
            vid_id = str(row["video_id"])
            dur    = row["duration_sec"]
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))

            n_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                      if dur >= bin_sec else 0)
            if n_wins == 0 or start_tr < 0 or start_tr >= run_tr_count:
                continue

            windows = []
            for i in range(n_wins):
                w_start = start_tr + i * skip_trs
                w_end   = w_start + bin_trs
                if w_start >= run_tr_count or w_end > run_tr_count:
                    break
                windows.append(run_data[:, w_start:w_end].mean(axis=1))

            if not windows:
                continue
            binned = np.stack(windows, axis=0)  # (n_wins, n_verts)

            if vid_id in test_video_ids:
                run_test_segs.append(binned)
            else:
                run_train_segs.append(binned)

        # ── Z-score per run: training bins and held-out clips separately ─────
        if run_train_segs:
            run_train = np.concatenate(run_train_segs, axis=0).astype(np.float32)
            mu = run_train.mean(axis=0, keepdims=True)
            sd = run_train.std(axis=0,  keepdims=True)
            sd[sd == 0] = 1.0
            run_train = (run_train - mu) / sd
            run_onsets.append(n_train_bins)
            train_segs.append(run_train)
            n_train_bins += run_train.shape[0]

        if run_test_segs:
            run_test = np.concatenate(run_test_segs, axis=0).astype(np.float32)
            test_sd = run_test.std(axis=0, keepdims=True)
            test_sd[test_sd == 0] = 1.0
            test_segs.append((run_test - run_test.mean(axis=0, keepdims=True)) / test_sd)

    Y_train = (np.concatenate(train_segs, axis=0) if train_segs
               else np.empty((0, n_verts), dtype=np.float32))
    Y_test  = (np.concatenate(test_segs,  axis=0) if test_segs
               else np.empty((0, n_verts), dtype=np.float32))
    return Y_train, Y_test, run_onsets


def build_fmri_arrays(cifti_path: str, run_trs_path: str,
                       timing_df: pd.DataFrame,
                       test_video_ids: list, bin_sec: float, tr: float,
                       delay_sec: float = 0.0,
                       skip_sec: float = None) -> tuple:
    """Load continuous preprocessed CIFTI and build binned train/test arrays.

    Loads the full-run CIFTI (all 4 runs concatenated) produced by
    preprocess_individual.py, then calls _bin_and_split_fmri() to extract
    movie segments using global onset_sec from timing_df.

    Args:
        cifti_path    : str — path to {subject}_{suffix}_cortex_59k.dtseries.nii
        run_trs_path  : str — path to {subject}_{suffix}_run_trs.npy
        timing_df     : pd.DataFrame — with video_id, onset_sec (global), duration_sec, run_id
        test_video_ids: list[str] — video IDs held out for testing
        bin_sec       : float — temporal bin size in seconds
        tr            : float — TR in seconds
        delay_sec     : float — haemodynamic delay to apply (default 0)
        skip_sec      : float — window stride in seconds (default: bin_sec, no overlap)

    Returns:
        Y_train    : (n_train_bins, n_vertices) float32
        Y_test     : (n_test_bins, n_vertices) float32
        run_onsets : list[int] — training run onset bin indices for LORO-CV
    """
    img  = nib.load(cifti_path)
    fmri = img.get_fdata(dtype=np.float32).T   # (n_vertices, T_total)
    run_trs = np.load(run_trs_path)
    return _bin_and_split_fmri(fmri, timing_df, test_video_ids,
                                bin_sec, tr, run_trs, delay_sec, skip_sec)


# =============================================================================
# HRF application
# =============================================================================

def apply_hrf_to_segment(segment: np.ndarray, hrf_kernel: np.ndarray) -> np.ndarray:
    """Convolve each feature column with the HRF kernel.

    Args:
        segment: (T, n_features) float64
        hrf_kernel: (n_hrf,) float64

    Returns:
        (T, n_features) float64 — convolved and trimmed to original length
    """
    from scipy.signal import fftconvolve
    out = np.empty_like(segment)
    for j in range(segment.shape[1]):
        conv = fftconvolve(segment[:, j], hrf_kernel, mode="full")
        out[:, j] = conv[: len(segment)]
    return out


def save_cifti_maps(maps: dict, template_path: str, output_path: str) -> None:
    axis = nib.load(template_path).header.get_axis(1)
    header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(list(maps)), axis))
    data = np.stack([np.asarray(values, dtype=np.float32) for values in maps.values()])
    nib.save(nib.Cifti2Image(data, header=header), output_path)
