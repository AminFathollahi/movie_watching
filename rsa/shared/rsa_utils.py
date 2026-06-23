"""
rsa/shared/rsa_utils.py
=======================
Shared utilities for RSA analyses (searchlight and Glasser parcellation).

All functions are stateless and accept explicit arguments — no hardcoded paths
or parameters. Configure everything in analysis.sh and pass via CLI.

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
                    delay_sec: float = 0.0,
                    skip_sec: float = None,
                    normalize: bool = True) -> np.ndarray:
    """Bin and normalize fMRI data to match model embeddings.

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
    skip_sec        : window stride in seconds (default: bin_sec, i.e., no overlap).
                      Set skip_sec < bin_sec for overlapping windows.
    normalize       : if True (default) z-score per run; if False demean only
                      (subtract per-run mean, do not divide by std).

    Returns
    -------
    (total_windows, n_vertices) float32 — windowed + per-run normalized fMRI
    """
    if skip_sec is None:
        skip_sec = bin_sec
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
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
            dur = row["duration_sec"]
            # Number of complete windows: floor((dur - bin_sec) / skip_sec) + 1
            # Equivalent to floor(dur / bin_sec) when skip_sec == bin_sec.
            n_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                      if dur >= bin_sec else 0)
            if n_wins == 0:
                continue

            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))

            if start_tr >= run_tr_count or start_tr < 0:
                continue

            windows = []
            for i in range(n_wins):
                w_start = start_tr + i * skip_trs
                w_end   = w_start + bin_trs
                if w_start >= run_tr_count or w_end > run_tr_count:
                    break  # drop windows that exceed run boundary
                windows.append(run_data[:, w_start:w_end].mean(axis=1))

            if windows:
                run_segments.append(np.stack(windows, axis=0))  # (n_wins, n_verts)

        if run_segments:
            run_binned_concat = np.concatenate(run_segments, axis=0).astype(np.float32)
            if normalize:
                run_normed = zscore(run_binned_concat, axis=0, nan_policy='omit')
            else:
                run_normed = run_binned_concat - run_binned_concat.mean(axis=0, keepdims=True)
            binned_runs.append(run_normed)

    return np.concatenate(binned_runs, axis=0)


def process_model_embeddings(emb_path: str, timing_df: pd.DataFrame,
                             bin_sec: float, tr: float,
                             run_trs: np.ndarray, delay_sec: float = 0.0,
                             hrf: bool = False,
                             skip_sec: float = None,
                             normalize: bool = True) -> np.ndarray:
    """Load and align model embeddings to the fMRI binning scheme.

    Parameters
    ----------
    emb_path  : path to .npy embedding file (total_windows × n_features)
    timing_df : same DataFrame as passed to preprocess_fmri — must be consistent.
                onset_sec is GLOBAL time (cumulative across runs). Same format
                as /data/HCP Data/movie_timing.csv.
    bin_sec   : temporal bin width in seconds
    tr        : repetition time in seconds
    run_trs   : (n_runs,) int — TRs per run (used only for boundary truncation)
    delay_sec : haemodynamic shift applied to onset_sec (default 0)
    hrf       : if True, convolve each segment with the SPM canonical HRF at
                bin_sec resolution. Use with delay_sec=0 (the convolution replaces
                the boxcar delay). Matches encoding's build_embedding_arrays behaviour.
    skip_sec  : window stride in seconds (default: bin_sec, i.e., no overlap).
                Must match the skip_sec used to segment the stimulus and passed
                to preprocess_fmri so brain and model window counts stay aligned.
    normalize : if True (default) z-score per run; if False demean only.

    Returns
    -------
    (total_windows, n_features) float32 — windowed + per-run normalized embeddings
    """
    if skip_sec is None:
        skip_sec = bin_sec
    embeddings = np.load(emb_path)
    hrf_kernel = spm_hrf(bin_sec) if hrf else None
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
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
            dur = row["duration_sec"]
            # Base window count from clip duration (mirrors segmentation formula)
            base_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                         if dur >= bin_sec else 0)

            # Mirror preprocess_fmri boundary truncation exactly
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))

            final_wins = 0
            if 0 <= start_tr < run_tr_count:
                for i in range(base_wins):
                    w_start = start_tr + i * skip_trs
                    w_end   = w_start + bin_trs
                    if w_start >= run_tr_count or w_end > run_tr_count:
                        break
                    final_wins += 1

            # Extract from embeddings; always advance seg_idx by base_wins so
            # the pointer stays aligned with the .npy file (which has the full
            # clip count, regardless of run boundary truncation).
            if final_wins > 0:
                seg = embeddings[seg_idx : seg_idx + final_wins].astype(np.float64)
                if hrf_kernel is not None:
                    seg = _apply_hrf_to_segment(seg, hrf_kernel)
                run_segments.append(seg)

            seg_idx += base_wins

        if run_segments:
            run_emb_concat = np.concatenate(run_segments, axis=0)
            if normalize:
                run_normed = zscore(run_emb_concat, axis=0, nan_policy='omit')
            else:
                run_normed = run_emb_concat - run_emb_concat.mean(axis=0, keepdims=True)
            processed_runs.append(run_normed)

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


def get_run_bin_counts(timing_df: pd.DataFrame, run_trs: np.ndarray,
                       bin_sec: float, tr: float,
                       delay_sec: float = 0.0,
                       skip_sec: float = None) -> np.ndarray:
    """Return the number of temporal bins produced by preprocess_fmri for each run.

    Mirrors preprocess_fmri logic exactly. Used by searchlight.py to align
    block boundaries with fMRI run boundaries (Schütt et al. 2023 §5.1.3).

    Returns
    -------
    (n_runs,) int — bin count for each run, sums to total bins in fmri_binned
    """
    if skip_sec is None:
        skip_sec = bin_sec
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
    run_col = ('run' if 'run' in timing_df.columns
               else ('run_id' if 'run_id' in timing_df.columns else None))

    if run_col is None:
        groups = [(None, timing_df)]
    else:
        groups = list(timing_df.groupby(run_col, sort=False))

    counts = []
    for run_idx, (run_id, run_df) in enumerate(groups):
        run_tr_count  = run_trs[run_idx]
        run_start_sec = float(np.sum(run_trs[:run_idx])) * tr
        run_count     = 0

        for _, row in run_df.iterrows():
            dur    = row["duration_sec"]
            n_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                      if dur >= bin_sec else 0)
            if n_wins == 0:
                continue
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            if start_tr >= run_tr_count or start_tr < 0:
                continue
            for i in range(n_wins):
                w_start = start_tr + i * skip_trs
                w_end   = w_start + bin_trs
                if w_start >= run_tr_count or w_end > run_tr_count:
                    break
                run_count += 1

        counts.append(run_count)

    return np.array(counts, dtype=int)


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


def assert_segment_timing(timing_df: pd.DataFrame, bin_sec: float, tr: float,
                           delay_sec: float = 0.0, skip_sec: float = None,
                           run_trs: np.ndarray = None) -> None:
    """Verify that model segment windows and fMRI extraction windows are aligned.

    For each clip in timing_df, logs the expected model start/end times
    (based on onset_sec and skip_sec stride) and the corresponding fMRI TRs
    (onset + delay). Raises AssertionError if any window would fall outside
    its run boundary.

    Parameters
    ----------
    timing_df : same DataFrame passed to preprocess_fmri / process_model_embeddings
    bin_sec   : temporal bin width in seconds
    tr        : repetition time in seconds
    delay_sec : haemodynamic shift applied to fMRI onset (default 0)
    skip_sec  : window stride in seconds (default: bin_sec)
    run_trs   : (n_runs,) int — TRs per run; if None, boundary checks are skipped
    """
    if skip_sec is None:
        skip_sec = bin_sec
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
    run_col  = ('run' if 'run' in timing_df.columns
                else ('run_id' if 'run_id' in timing_df.columns else None))

    log.info(
        f"Segment alignment check: bin={bin_sec}s skip={skip_sec}s delay={delay_sec}s"
    )

    if run_col is None:
        groups = [(None, timing_df)]
    else:
        groups = list(timing_df.groupby(run_col, sort=False))

    errors = []
    for run_idx, (run_id, run_df) in enumerate(groups):
        run_tr_count = int(run_trs[run_idx]) if run_trs is not None else None
        run_start_sec = (float(np.sum(run_trs[:run_idx])) * tr
                         if run_trs is not None else 0.0)
        for _, row in run_df.iterrows():
            dur = row["duration_sec"]
            n_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                      if dur >= bin_sec else 0)
            if n_wins == 0:
                continue
            within_run_onset = row["onset_sec"] - run_start_sec
            fmri_start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            for i in range(n_wins):
                w_start_tr = fmri_start_tr + i * skip_trs
                w_end_tr   = w_start_tr + bin_trs
                model_t0   = row["onset_sec"] + i * skip_sec
                model_t1   = model_t0 + bin_sec
                fmri_t0    = within_run_onset + delay_sec + i * skip_sec
                fmri_t1    = fmri_t0 + bin_sec
                if run_tr_count is not None and w_end_tr > run_tr_count:
                    errors.append(
                        f"  run={run_id} clip={row.get('video_id', '?')} win={i}: "
                        f"fMRI TRs [{w_start_tr},{w_end_tr}) exceed run ({run_tr_count} TRs)"
                    )
                log.debug(
                    f"  run={run_id} win={i}: model=[{model_t0:.2f},{model_t1:.2f}]s "
                    f"fMRI=[{fmri_t0:.2f},{fmri_t1:.2f}]s (TRs [{w_start_tr},{w_end_tr}))"
                )
    if errors:
        raise AssertionError(
            "Segment timing alignment errors:\n" + "\n".join(errors)
        )
    log.info("  Segment timing OK — all windows within run boundaries.")


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

    NOTE: This computes a naive (biased) distance estimator. For naturalistic movie
    paradigms without repeated stimulus segments, crossvalidated (Crossnobis) distances
    are not feasible. Bias is uniform across the RDM when noise is homoscedastic, so
    rank-based comparators (Spearman, rho_a) are less affected than Pearson/Euclidean.
    Ref: Schütt et al. 2023 §5.1.1; Walther et al. 2016.
    """
    if method == "cosine":
        sim = cosine_similarity(data)
        return 1.0 - sim
    return squareform(pdist(data, metric="correlation"))


def correlate_rdms_rho_a(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    """Compute rho_a (Kendall's tau_a) between two RDMs.

    rho_a is the recommended RDM comparator from Schütt et al. 2023 §3.5:
    computationally efficient, has an analytically derived noise ceiling,
    and is unaffected by linear scaling of either RDM. Equivalent to
    Kendall's tau_a on the lower-triangle vectors.

    Returns float in [-1, 1].
    """
    from scipy.stats import kendalltau
    n = rdm1.shape[0]
    idx = np.tril_indices(n, k=-1)
    v1 = rdm1[idx].astype(np.float64)
    v2 = rdm2[idx].astype(np.float64)
    tau, _ = kendalltau(v1, v2, method="auto")
    return float(tau)


def correlate_rdms(rdm1: np.ndarray, rdm2: np.ndarray,
                   method: str = "spearman") -> tuple[float, float]:
    """Correlate the lower triangles of two RDMs.

    Args:
        rdm1: (n, n) float64 — symmetric RDM
        rdm2: (n, n) float64 — symmetric RDM
        method: str — "spearman", "pearson", or "rho_a"

    Returns:
        (r, p): correlation coefficient and p-value (nan for rho_a, use bootstrap)
    """
    n = rdm1.shape[0]
    idx = np.tril_indices(n, k=-1)
    v1, v2 = rdm1[idx], rdm2[idx]

    if method == "spearman":
        r, p = spearmanr(v1, v2)
    elif method == "pearson":
        r, p = pearsonr(v1, v2)
    elif method == "rho_a":
        r = correlate_rdms_rho_a(rdm1, rdm2)
        p = float("nan")  # no analytical p-value; use bootstrap (Schütt et al. 2023 §3.5)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman', 'pearson', or 'rho_a'.")
    return float(r), float(p)


def corrected_2factor_bootstrap(
    rho_stack: np.ndarray,
    block_stack: np.ndarray,
    n_boot: int = 2000,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corrected 2-factor bootstrap variance (Schütt et al. 2023, Eq. 5).

    Estimates the variance of the group-mean RSA correlation when generalizing
    to both new subjects AND new movie segments (time-bin blocks).

    The naive 2-factor bootstrap triple-counts the measurement noise variance.
    The corrected estimate cancels this surplus:
        σ̂²_c2f = 2(σ̂²_subj + σ̂²_block) − σ̂²_both   (Eq. 5)
    Bounded: clip(σ̂²_c2f, min=max(σ̂²_subj, σ̂²_block), max=σ̂²_both)

    Parameters
    ----------
    rho_stack  : (n_subs, n_verts) float32 — full-series RSA ρ per subject
    block_stack: (n_subs, n_blocks, n_verts) float32 — per-block RSA ρ
    n_boot     : number of bootstrap iterations (default 2000)
    rng        : numpy Generator (default: default_rng(42))

    Returns
    -------
    var_c2f  : (n_verts,) float64 — corrected 2-factor variance (SE² of the mean)
    var_subj : (n_verts,) float64 — 1-factor subject bootstrap variance
    var_block: (n_verts,) float64 — 1-factor block bootstrap variance
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_subs, n_verts       = rho_stack.shape
    n_subs2, n_blocks, n_verts2 = block_stack.shape
    assert n_subs == n_subs2 and n_verts == n_verts2, (
        "rho_stack and block_stack shape mismatch"
    )

    rho_f64   = rho_stack.astype(np.float64)
    block_f64 = block_stack.astype(np.float64)

    means_subj  = np.empty((n_boot, n_verts), dtype=np.float64)
    means_block = np.empty((n_boot, n_verts), dtype=np.float64)
    means_both  = np.empty((n_boot, n_verts), dtype=np.float64)

    for i in range(n_boot):
        si = rng.integers(0, n_subs,   size=n_subs)
        bi = rng.integers(0, n_blocks, size=n_blocks)

        means_subj[i]  = rho_f64[si].mean(axis=0)
        means_block[i] = block_f64[:, bi, :].mean(axis=(0, 1))
        means_both[i]  = block_f64[si][:, bi, :].mean(axis=(0, 1))

    var_subj  = means_subj.var(axis=0,  ddof=1)
    var_block = means_block.var(axis=0, ddof=1)
    var_both  = means_both.var(axis=0,  ddof=1)

    var_c2f_raw = 2.0 * (var_subj + var_block) - var_both
    lower = np.maximum(var_subj, var_block)
    # Ensure upper ≥ lower; finite-sample fluctuations can make var_both < lower.
    upper = np.maximum(var_both, lower)
    var_c2f = np.clip(var_c2f_raw, a_min=lower, a_max=upper)
    return var_c2f, var_subj, var_block


def aggregate_blocks(block_stack: np.ndarray, n_target: int) -> np.ndarray:
    """Average adjacent blocks to reduce from N_file to n_target.

    Allows a single searchlight run at a large N to serve multiple group_stats
    sweeps at smaller N values without re-running the expensive searchlight.

    Parameters
    ----------
    block_stack : (n_subs, N_file, n_verts) — raw block RSA output from searchlight
    n_target    : desired number of blocks; N_file must be divisible by n_target

    Returns
    -------
    (n_subs, n_target, n_verts) same dtype as input
    """
    n_subs, n_file, n_verts = block_stack.shape
    if n_file == n_target:
        return block_stack
    if n_file % n_target != 0:
        raise ValueError(
            f"Cannot aggregate {n_file} blocks to {n_target}: "
            f"{n_file} must be divisible by {n_target}."
        )
    group = n_file // n_target
    return block_stack.reshape(n_subs, n_target, group, n_verts).mean(axis=2).astype(block_stack.dtype)
