"""
rsa/shared/rsa_utils.py
=======================
Shared utilities for RSA analyses (searchlight and Glasser parcellation).

All functions are stateless and accept explicit arguments — no hardcoded paths
or parameters. Configure everything in run_analysis.sh and pass via CLI.

fMRI preprocessing convention
------------------------------
The input CIFTI (output of preprocess_individual.py --timing-csv) is already:
  - Filtered to movie timepoints with hemodynamic delay applied
  - Z-scored per vertex over all included timepoints

preprocess_fmri() therefore only bins each video segment — no delay shift and
no z-scoring. Video positions in the filtered CIFTI are determined by the
cumulative sum of duration_sec from timing_df.
"""

import logging
import math 
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import gamma, spearmanr, pearsonr, zscore
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

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
    hrf = (gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0)
    hrf = hrf[::oversampling]
    hrf /= hrf.sum()
    return hrf


# =============================================================================
# fMRI loading and binning
# =============================================================================

def load_fmri_cifti(cifti_path: str) -> np.ndarray:
    """Load a preprocessed CIFTI dtseries (59k cortex, filtered+z-scored).

    Args:
        cifti_path: str — path to .dtseries.nii produced by preprocess_individual.py
                          with --timing-csv (filtered mode)

    Returns:
        (n_vertices, T_included) float32
    """

    img = nib.load(cifti_path)
    return img.get_fdata(dtype=np.float32).T  # (n_vertices, T)




def preprocess_fmri(fmri_continuous: np.ndarray, timing_df: pd.DataFrame,
                    run_trs: np.ndarray, bin_sec: float, tr: float,
                    delay_sec: float = 0.0) -> np.ndarray:
    """Bin and z-score fMRI data to match model embeddings.

    Parameters
    ----------
    fmri_continuous : (n_vertices, T_total) float32
    timing_df       : DataFrame with columns: duration_sec, onset_sec, and
                      optionally run / run_id for multi-run data.
                      If no run column is present, the entire DataFrame is treated
                      as a single run (useful for tests and simple use-cases).
                      onset_sec is GLOBAL time (cumulative across runs, matching
                      /data/HCP Data/movie_timing.csv). The cumulative run-start
                      offset (sum of prior run_trs × tr) is subtracted internally
                      to obtain the within-run TR index.
    run_trs         : (n_runs,) int — number of TRs per run (from preprocess_individual)
    bin_sec         : temporal bin width in seconds
    tr              : repetition time in seconds
    delay_sec       : haemodynamic shift applied to onset_sec (default 0)

    Returns
    -------
    (total_bins, n_vertices) float32 — binned + per-run z-scored fMRI
    """
    bin_trs = max(1, int(np.round(bin_sec / tr)))
    run_col = ('run' if 'run' in timing_df.columns
               else ('run_id' if 'run_id' in timing_df.columns else None))

    if run_col is None:
        # No run column — treat all rows as a single run
        groups = [(None, timing_df)]
    else:
        groups = list(timing_df.groupby(run_col, sort=False))

    binned_runs = []
    global_tr_offset = 0

    for run_idx, (run_id, run_df) in enumerate(groups):
        run_tr_count = run_trs[run_idx]
        run_data = fmri_continuous[:, global_tr_offset : global_tr_offset + run_tr_count]
        global_tr_offset += run_tr_count

        # Cumulative run-start in seconds (for converting global → within-run onset)
        run_start_sec = float(np.sum(run_trs[:run_idx])) * tr

        run_segments = []
        for _, row in run_df.iterrows():
            # 1. Match the segmentation script's floor logic
            expected_bins = int(np.floor(row["duration_sec"] / bin_sec))
            if expected_bins == 0:
                continue

            # 2. Convert global onset → within-run TR, then apply haemodynamic delay
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            end_tr = start_tr + (expected_bins * bin_trs)

            # 3. Handle run boundary cutoffs
            if start_tr >= run_tr_count or start_tr < 0:
                continue
            if end_tr > run_tr_count:
                # Recalculate how many full bins actually fit before the run ends
                expected_bins = (run_tr_count - start_tr) // bin_trs
                end_tr = start_tr + (expected_bins * bin_trs)
                if expected_bins == 0:
                    continue

            seg = run_data[:, start_tr:end_tr]

            binned = (seg.reshape(run_data.shape[0], expected_bins, bin_trs)
                         .mean(axis=2).T)
            run_segments.append(binned)

        if run_segments:
            run_binned_concat = np.concatenate(run_segments, axis=0).astype(np.float32)
            run_zscored = zscore(run_binned_concat, axis=0, nan_policy='omit')
            binned_runs.append(run_zscored)

    return np.concatenate(binned_runs, axis=0)


def process_model_embeddings(emb_path: str, timing_df: pd.DataFrame,
                             bin_sec: float, hrf: bool, tr: float,
                             run_trs: np.ndarray, delay_sec: float = 0.0) -> np.ndarray:
    """Load and align model embeddings to the fMRI binning scheme.

    Parameters
    ----------
    emb_path  : path to .npy embedding file (total_bins × n_features)
    timing_df : same DataFrame as passed to preprocess_fmri — must be consistent.
                onset_sec is GLOBAL time (cumulative across runs). Same format
                as /data/HCP Data/movie_timing.csv.
    bin_sec   : temporal bin width in seconds
    hrf       : if True, call process_model_embeddings_with_hrf instead
    tr        : repetition time in seconds
    run_trs   : (n_runs,) int — TRs per run (used only for boundary truncation)
    delay_sec : haemodynamic shift applied to onset_sec (default 0)

    Returns
    -------
    (total_bins, n_features) float32 — binned + per-run z-scored embeddings
    """
    embeddings = np.load(emb_path)
    bin_trs = max(1, int(np.round(bin_sec / tr)))
    run_col = ('run' if 'run' in timing_df.columns
               else ('run_id' if 'run_id' in timing_df.columns else None))

    if run_col is None:
        groups = [(None, timing_df)]
    else:
        groups = list(timing_df.groupby(run_col, sort=False))

    processed_runs = []
    seg_idx = 0

    for run_idx, (run_id, run_df) in enumerate(groups):
        run_tr_count = run_trs[run_idx]
        run_segments = []

        # Cumulative run-start for global → within-run onset conversion
        run_start_sec = float(np.sum(run_trs[:run_idx])) * tr

        for _, row in run_df.iterrows():
            # 1. Base bins from segmentation math
            base_bins = int(np.floor(row["duration_sec"] / bin_sec))

            # 2. Mirror fMRI boundary truncation exactly (global → within-run)
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            end_tr = start_tr + (base_bins * bin_trs)

            final_bins = base_bins
            if start_tr >= run_tr_count or start_tr < 0:
                final_bins = 0
            elif end_tr > run_tr_count:
                final_bins = (run_tr_count - start_tr) // bin_trs

            # 3. Extract from embeddings using the original base_bins for the index counter
            # (The .npy file always has the full clip — advance seg_idx by base_bins
            # even when only final_bins are used, so the index stays aligned.)
            if final_bins > 0:
                seg = embeddings[seg_idx : seg_idx + final_bins].astype(np.float64)
                run_segments.append(seg)

            seg_idx += base_bins

        if run_segments:
            run_emb_concat = np.concatenate(run_segments, axis=0)
            run_zscored = zscore(run_emb_concat, axis=0, nan_policy='omit')
            processed_runs.append(run_zscored)

    return np.concatenate(processed_runs, axis=0).astype(np.float32)

def process_model_embeddings_with_hrf(emb_path_tr: str, timing_df: pd.DataFrame,
                                      bin_sec: float, tr: float) -> np.ndarray:
    """
    Load TR-resolution embeddings, convolve with HRF, then bin and z-score per run.
    Note: emb_path_tr MUST point to the 1s/TR resolution embeddings.
    """
    # 1. Load high-resolution embeddings
    embeddings_tr = np.load(emb_path_tr).astype(np.float64) 
    
    # 2. Convolve at TR resolution FIRST
    hrf_kernel = spm_hrf(tr)
    embeddings_convolved = _apply_hrf_to_segment(embeddings_tr, hrf_kernel)

    bin_trs = max(1, int(round(bin_sec / tr)))
    run_col = 'run' if 'run' in timing_df.columns else ('run_id' if 'run_id' in timing_df.columns else None)
    
    processed_runs = []
    tr_idx = 0  # Tracks the TR row index in the convolved embeddings file

    for run_id, run_df in timing_df.groupby(run_col, sort=False):
        run_segments = []
        for _, row in run_df.iterrows():
            # Slice the convolved TRs based on stimulus duration
            n_seg_trs = int(round(row["duration_sec"] / tr))
            seg_trs = embeddings_convolved[tr_idx : tr_idx + n_seg_trs]
            tr_idx += n_seg_trs

            # 3. Bin the convolved TRs down to bin_sec
            n_bins = seg_trs.shape[0] // bin_trs
            if n_bins > 0:
                binned = (seg_trs[:n_bins * bin_trs]
                          .reshape(n_bins, bin_trs, -1)
                          .mean(axis=1))
                run_segments.append(binned)
            
        if run_segments:
            run_emb_concat = np.concatenate(run_segments, axis=0)
            # 4. Z-score per run
            run_zscored = zscore(run_emb_concat, axis=0, nan_policy='omit')
            processed_runs.append(run_zscored)

    return np.concatenate(processed_runs, axis=0).astype(np.float32)


def align_and_assert_bins(fmri_binned: np.ndarray, model_binned: np.ndarray) -> tuple:
    """
    Ensures exact structural alignment between brain and model before RSA.
    """
    f_bins = fmri_binned.shape[0]
    m_bins = model_binned.shape[0]
    
    if f_bins != m_bins:
        raise AssertionError(
            f"CRITICAL ALIGNMENT ERROR: Bin counts do not match!\n"
            f"  Brain Bins: {f_bins}\n"
            f"  Model Bins: {m_bins}\n"
            f"Check delay_sec overlapping run boundaries or missing TRs."
        )
    return fmri_binned, model_binned


# =============================================================================
# Model embedding processing
# =============================================================================

def _apply_hrf_to_segment(segment: np.ndarray, hrf_kernel: np.ndarray) -> np.ndarray:
    """Convolve each feature dimension with the HRF kernel.

    Args:
        segment: (T, n_features) float64
        hrf_kernel: (n_hrf,) float64

    Returns:
        (T, n_features) float64 — convolved and trimmed to original length
    """
    from scipy.signal import fftconvolve
    out = np.zeros_like(segment)
    for j in range(segment.shape[1]):
        conv = fftconvolve(segment[:, j], hrf_kernel, mode="full")
        out[:, j] = conv[: len(segment)]
    return out


# =============================================================================
# RDM and RSA
# =============================================================================

def compute_rdm(data: np.ndarray, method: str = "correlation") -> np.ndarray:
    """Compute a representational dissimilarity matrix (RDM).

    Args:
        data: (n_conditions, n_features) — one row per stimulus/condition
        method: str — "correlation" (1 - Pearson r) or "cosine" (1 - cosine sim)

    Returns:
        (n_conditions, n_conditions) float64 — symmetric RDM with zero diagonal
    """
    if method == "cosine":
        sim = cosine_similarity(data)
        return 1.0 - sim
    return squareform(pdist(data, metric="correlation"))


def correlate_rdms(rdm1: np.ndarray, rdm2: np.ndarray,
                   method: str = "spearman") -> tuple[float, float]:
    """Correlate the lower triangles of two RDMs.

    Args:
        rdm1: (n, n) float64 — symmetric RDM
        rdm2: (n, n) float64 — symmetric RDM
        method: str — "spearman" or "pearson"

    Returns:
        (r, p): correlation coefficient and p-value
    """
    n = rdm1.shape[0]
    idx = np.tril_indices(n, k=-1)
    v1, v2 = rdm1[idx], rdm2[idx]

    if method == "spearman":
        r, p = spearmanr(v1, v2)
    elif method == "pearson":
        r, p = pearsonr(v1, v2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'.")
    return float(r), float(p)
