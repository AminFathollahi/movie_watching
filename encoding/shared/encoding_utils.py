"""
encoding/shared/encoding_utils.py
===================================
Shared utilities for ridge encoding models.

All functions are stateless and accept explicit arguments — no hardcoded paths
or parameters. Configure everything in run_analysis.sh and pass via CLI.

fMRI preprocessing convention
------------------------------
The input CIFTI is the output of preprocess_individual.py in continuous mode
(SG → PSC → GSR per run, all 4 runs concatenated, 59412 grayordinates).
No separate filtering step is needed; build_fmri_arrays() uses the global
onset_sec from timing_df and run_trs to locate each clip, applies the
haemodynamic delay, bins to bin_sec resolution, z-scores per run (training
statistics applied to both train and test), and splits by video_id.

This matches the convention used in rsa/shared/rsa_utils.py (preprocess_fmri).
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.model_selection import PredefinedSplit

log = logging.getLogger(__name__)


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
                         run_trs: np.ndarray, delay_sec: float = 0.0) -> tuple:
    """Extract movie segments from continuous fMRI, z-score per run, bin, split.

    Uses global onset_sec from timing_df (same convention as rsa_utils.preprocess_fmri):
    onset_sec is cumulative across runs; run_start_sec is subtracted internally to
    get the within-run TR index.  Rest periods between clips are skipped.

    Z-scoring is per-run on training bins; the same mean/std are applied to test
    bins from the same run to avoid data leakage.

    Args:
        fmri         : (n_vertices, T_total) float32 — full continuous preprocessed signal
        timing_df    : DataFrame with columns: video_id, onset_sec (global), duration_sec, run_id
        test_video_ids: list[str] — video IDs held out for test set
        bin_sec      : float — temporal bin size in seconds
        tr           : float — TR in seconds
        run_trs      : (n_runs,) int — TRs per run (from preprocess_individual run_trs.npy)
        delay_sec    : float — haemodynamic shift applied to onset_sec (default 0;
                       apply pre-delay in preprocess_individual or pass here)

    Returns:
        Y_train    : (n_train_bins, n_vertices) float32
        Y_test     : (n_test_bins, n_vertices) float32
        run_onsets : list[int] — training run onset bin indices for LORO-CV
    """
    bin_trs   = max(1, int(np.round(bin_sec / tr)))
    run_col   = 'run_id' if 'run_id' in timing_df.columns else 'run'
    n_verts   = fmri.shape[0]

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
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            n_bins   = int(np.floor(row["duration_sec"] / bin_sec))
            if n_bins == 0 or start_tr < 0 or start_tr >= run_tr_count:
                continue
            end_tr = start_tr + n_bins * bin_trs
            if end_tr > run_tr_count:
                n_bins = (run_tr_count - start_tr) // bin_trs
                end_tr = start_tr + n_bins * bin_trs
                if n_bins == 0:
                    continue

            seg    = run_data[:, start_tr:end_tr]          # (n_verts, n_bins*bin_trs)
            binned = (seg.reshape(n_verts, n_bins, bin_trs)
                        .mean(axis=2).T)                   # (n_bins, n_verts)

            if vid_id in test_video_ids:
                run_test_segs.append(binned)
            else:
                run_train_segs.append(binned)

        # ── Z-score per run (training stats, applied to test too) ────────────
        if run_train_segs:
            run_train = np.concatenate(run_train_segs, axis=0).astype(np.float32)
            mu = run_train.mean(axis=0, keepdims=True)
            sd = run_train.std(axis=0,  keepdims=True)
            sd[sd == 0] = 1.0
            run_train = (run_train - mu) / sd
            run_onsets.append(n_train_bins)
            train_segs.append(run_train)
            n_train_bins += run_train.shape[0]
        else:
            mu = sd = None

        if run_test_segs:
            run_test = np.concatenate(run_test_segs, axis=0).astype(np.float32)
            if mu is not None:          # apply same z-score params as training
                run_test = (run_test - mu) / sd
            test_segs.append(run_test)

    Y_train = (np.concatenate(train_segs, axis=0) if train_segs
               else np.empty((0, n_verts), dtype=np.float32))
    Y_test  = (np.concatenate(test_segs,  axis=0) if test_segs
               else np.empty((0, n_verts), dtype=np.float32))
    return Y_train, Y_test, run_onsets


def build_fmri_arrays(cifti_path: str, run_trs_path: str,
                       timing_df: pd.DataFrame,
                       test_video_ids: list, bin_sec: float, tr: float,
                       delay_sec: float = 0.0) -> tuple:
    """Load continuous preprocessed CIFTI and build binned train/test arrays.

    Loads the full-run CIFTI (SG→PSC→GSR, all 4 runs concatenated) produced
    by preprocess_individual.py, then calls _bin_and_split_fmri() to extract
    movie segments using global onset_sec from timing_df.

    Args:
        cifti_path    : str — path to {subject}_{suffix}_cortex_59k.dtseries.nii
        run_trs_path  : str — path to {subject}_{suffix}_run_trs.npy
        timing_df     : pd.DataFrame — with video_id, onset_sec (global), duration_sec, run_id
        test_video_ids: list[str] — video IDs held out for testing
        bin_sec       : float — temporal bin size in seconds
        tr            : float — TR in seconds
        delay_sec     : float — haemodynamic delay to apply (default 0)

    Returns:
        Y_train    : (n_train_bins, n_vertices) float32
        Y_test     : (n_test_bins, n_vertices) float32
        run_onsets : list[int] — training run onset bin indices for LORO-CV
    """
    img  = nib.load(cifti_path)
    fmri = img.get_fdata(dtype=np.float32).T   # (n_vertices, T_total)
    run_trs = np.load(run_trs_path)
    return _bin_and_split_fmri(fmri, timing_df, test_video_ids,
                                bin_sec, tr, run_trs, delay_sec)


# =============================================================================
# Feature processing
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


def build_embedding_arrays(emb_path: str, timing_df: pd.DataFrame,
                            test_video_ids: list, bin_sec: float,
                            hrf: bool, normalize: bool) -> tuple:
    """Load and process model embeddings to produce train/test design matrices.

    Embeddings are pre-computed at bin_sec resolution. The hemodynamic delay is
    handled at fMRI preprocessing time. When hrf=True, the fMRI was preprocessed
    with --delay-sec 0 and embeddings are convolved with the SPM HRF kernel.

    Args:
        emb_path: str — path to .npy embedding file, shape (n_total_bins, n_features)
        timing_df: pd.DataFrame — with video_id, duration_sec, run_id
        test_video_ids: list[str]
        bin_sec: float — temporal bin size in seconds
        hrf: bool — convolve with SPM HRF (use when fMRI preprocessed with delay=0)
        normalize: bool — per-run z-score of training embeddings

    Returns:
        X_train: (n_train_bins, n_features) float32
        X_test: (n_test_bins, n_features) float32
    """
    embeddings = np.load(emb_path).astype(np.float64)
    hrf_kernel = spm_hrf(bin_sec) if hrf else None

    train_segs   = []
    test_segs    = []
    seg_idx      = 0
    current_run  = None
    train_run_segs: list[list] = [[]]

    for _, row in timing_df.iterrows():
        vid_id = str(row["video_id"])
        run_id = row["run_id"]
        n_bins = int(round(row["duration_sec"] / bin_sec))

        seg = embeddings[seg_idx: seg_idx + n_bins].copy()
        seg_idx += n_bins

        if hrf and hrf_kernel is not None:
            seg = apply_hrf_to_segment(seg, hrf_kernel)

        if vid_id in test_video_ids:
            test_segs.append(seg)
        else:
            if run_id != current_run:
                if current_run is not None:
                    train_run_segs.append([])
                current_run = run_id
            train_run_segs[-1].append(seg)

    all_train = []
    for run_segs in train_run_segs:
        if not run_segs:
            continue
        run_data = np.vstack(run_segs)
        if normalize:
            mu = run_data.mean(axis=0, keepdims=True)
            sd = run_data.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            run_data = (run_data - mu) / sd
        all_train.append(run_data)

    X_train = np.vstack(all_train).astype(np.float32)
    X_test  = np.vstack(test_segs).astype(np.float32)
    return X_train, X_test


# =============================================================================
# Cross-validation helpers
# =============================================================================

def generate_leave_one_run_out(n_samples: int, run_onsets: list,
                                random_state=None, n_runs_out: int = 1):
    """Generate leave-one-run-out cross-validation splits.

    Adapted from gallantlab/voxelwise_tutorials (MIT License).

    Args:
        n_samples: int — total number of training samples
        run_onsets: list[int] — index of the first sample in each run
        random_state: ignored (kept for API compatibility)
        n_runs_out: int — number of runs to hold out per fold (default: 1)

    Yields:
        (train_indices, val_indices): np.ndarray, np.ndarray
    """
    run_onsets = list(run_onsets) + [n_samples]
    n_runs = len(run_onsets) - 1

    for i in range(n_runs):
        val_start = run_onsets[i]
        val_end   = run_onsets[i + 1]
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([
            np.arange(run_onsets[j], run_onsets[j + 1])
            for j in range(n_runs) if j != i
        ])
        yield train_idx, val_idx


def make_loro_splitter(n_train: int, run_onsets: list) -> PredefinedSplit:
    """Build a PredefinedSplit object for himalaya LORO cross-validation.

    Args:
        n_train: int — number of training samples
        run_onsets: list[int] — first sample index of each run

    Returns:
        sklearn.model_selection.PredefinedSplit
    """
    fold_ids = np.zeros(n_train, dtype=int)
    run_boundaries = list(run_onsets) + [n_train]
    for fold, (s, e) in enumerate(zip(run_boundaries[:-1], run_boundaries[1:])):
        fold_ids[s:e] = fold
    return PredefinedSplit(fold_ids)


# =============================================================================
# Encoding model
# =============================================================================

def run_encoding_model(X_train: np.ndarray, Y_train: np.ndarray,
                        X_test: np.ndarray, Y_test: np.ndarray,
                        run_onsets: list, alphas: np.ndarray,
                        chunk_size: int = 2000) -> np.ndarray:
    """Fit a ridge encoding model with LORO-CV alpha selection.

    Uses himalaya RidgeCV with SVD solver. Alpha is selected independently per
    vertex via leave-one-run-out cross-validation on the training set.

    Args:
        X_train: (n_train, n_features) float32
        Y_train: (n_train, n_vertices) float32
        X_test: (n_test, n_features) float32
        Y_test: (n_test, n_vertices) float32
        run_onsets: list[int] — run boundary indices for LORO-CV
        alphas: (n_alphas,) array — regularisation strengths to search
        chunk_size: int — number of vertices processed per batch

    Returns:
        (n_vertices,) float32 — Pearson r correlation on test set per vertex
    """
    from himalaya.ridge import RidgeCV
    from himalaya.backend import set_backend
    from scipy.stats import pearsonr

    backend = set_backend("numpy", on_error="warn")
    cv_splitter = make_loro_splitter(len(Y_train), run_onsets)

    model = RidgeCV(
        alphas=alphas,
        cv=cv_splitter,
        solver="svd",
        Y_in_cpu=True,
    )

    log.info(f"  Fitting RidgeCV: X_train={X_train.shape}, Y_train={Y_train.shape}")
    model.fit(X_train, Y_train)

    log.info("  Predicting on test set ...")
    Y_hat = model.predict(X_test)

    n_vertices = Y_test.shape[1]
    r_vals = np.zeros(n_vertices, dtype=np.float32)
    for start in range(0, n_vertices, chunk_size):
        end = min(start + chunk_size, n_vertices)
        for v in range(start, end):
            r, _ = pearsonr(Y_test[:, v], Y_hat[:, v])
            r_vals[v] = r

    return r_vals


# =============================================================================
# CIFTI output
# =============================================================================

def save_cifti(data_1d: np.ndarray, template_path: str, output_path: str,
               map_name: str = "encoding_r") -> None:
    """Save a 1-D cortical map as a CIFTI dscalar.nii (59k grayordinate space).

    Args:
        data_1d: (n_grayordinates,) float32
        template_path: str — reference CIFTI whose BrainModelAxis is reused
        output_path: str — destination path
        map_name: str — label shown in wb_view
    """
    img_ref = nib.load(template_path)
    bm_axis = img_ref.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis([map_name])
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    arr = data_1d.astype(np.float32).reshape(1, -1)
    img = nib.Cifti2Image(arr, header=header)
    nib.save(img, output_path)
