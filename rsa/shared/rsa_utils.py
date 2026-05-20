"""
rsa/shared/rsa_utils.py
=======================
Shared utilities for RSA analyses (searchlight and Glasser parcellation).

All functions are stateless and accept explicit arguments — no hardcoded paths
or parameters. Configure everything in run_analysis.sh and pass via CLI.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, spearmanr, pearsonr
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
    # Double-gamma HRF (SPM canonical)
    hrf = (gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0)
    hrf = hrf[::oversampling]
    hrf /= hrf.sum()
    return hrf


# =============================================================================
# fMRI loading and preprocessing
# =============================================================================

def load_fmri_cifti(cifti_path: str) -> np.ndarray:
    """Load a 59k preprocessed CIFTI dtseries.

    Args:
        cifti_path: str — path to .dtseries.nii (output of preprocess_individual.py)

    Returns:
        (n_vertices, T) float32
    """
    import nibabel as nib
    img = nib.load(cifti_path)
    return img.get_fdata(dtype=np.float32).T  # (n_vertices, T)


def preprocess_fmri(fmri: np.ndarray, timing_df: pd.DataFrame,
                    bin_sec: float, delay_sec: float, tr: float) -> np.ndarray:
    """Extract binned video segments from full-run fMRI with hemodynamic delay.

    For each video segment in timing_df, extracts the fMRI window
    [onset_sec + delay_sec, end_sec + delay_sec) from the full concatenated
    CIFTI. All included TRs across all segments are then z-scored together per
    vertex, matching the normalization scope used for model embeddings in
    process_model_embeddings (global, over included timepoints only). Finally,
    each z-scored segment is binned by averaging within bin_sec windows.

    Args:
        fmri: (n_vertices, T_total) float32 — full concatenated run CIFTI
        timing_df: pd.DataFrame — with onset_sec, end_sec columns
        bin_sec: float — temporal bin size in seconds
        delay_sec: float — hemodynamic delay offset in seconds (0.0 when hrf=True)
        tr: float — TR in seconds

    Returns:
        (n_bins, n_vertices) float32
    """
    bin_trs = max(1, int(round(bin_sec / tr)))
    T_total = fmri.shape[1]

    # Collect trimmed raw segments (before z-scoring or binning)
    raw_segs = []
    for _, row in timing_df.iterrows():
        start_tr = int(round((row["onset_sec"] + delay_sec) / tr))
        end_tr   = int(round((row["end_sec"]   + delay_sec) / tr))
        end_tr   = min(end_tr, T_total)
        seg = fmri[:, start_tr:end_tr]          # (n_vertices, T_seg)
        n_bins = seg.shape[1] // bin_trs
        if n_bins == 0:
            continue
        raw_segs.append(seg[:, :n_bins * bin_trs].copy())

    # Z-score per vertex over all included TRs so normalization is computed
    # only on movie timepoints, comparable to how model features are normalized.
    all_trs = np.concatenate(raw_segs, axis=1)  # (n_vertices, T_included)
    mu = all_trs.mean(axis=1, keepdims=True)
    sd = all_trs.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    all_trs = (all_trs - mu) / sd

    # Bin each z-scored segment and concatenate
    offset = 0
    segments = []
    for seg in raw_segs:
        length = seg.shape[1]
        z_seg = all_trs[:, offset: offset + length]
        n_seg_bins = length // bin_trs
        binned = z_seg.reshape(z_seg.shape[0], n_seg_bins, bin_trs).mean(axis=2).T
        segments.append(binned)
        offset += length

    return np.concatenate(segments, axis=0).astype(np.float32)


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


def process_model_embeddings(emb_path: str, timing_df: pd.DataFrame,
                              bin_sec: float, hrf: bool, normalize: bool,
                              blockdiag: bool, tr: float) -> np.ndarray:
    """Load and process model embeddings to match fMRI binning.

    Embeddings are pre-computed at bin_sec resolution. Hemodynamic delay is
    handled at fMRI extraction time (in preprocess_fmri), not here. When
    hrf=True, embeddings are convolved with the SPM HRF kernel.

    Args:
        emb_path: str — path to .npy file with shape (n_total_bins, n_features)
        timing_df: pd.DataFrame — movie timing table with duration_sec column
        bin_sec: float — temporal bin size in seconds
        hrf: bool — convolve with SPM HRF
        normalize: bool — z-score normalization of embeddings
        blockdiag: bool — if True, normalize within each video segment; else global
        tr: float — TR in seconds (used to build HRF kernel)

    Returns:
        (n_bins, n_features) float32
    """
    embeddings = np.load(emb_path)  # (n_total_bins, n_features)
    hrf_kernel = spm_hrf(bin_sec) if hrf else None

    processed_segments = []
    seg_idx = 0

    for _, row in timing_df.iterrows():
        n_seg_bins = int(round(row["duration_sec"] / bin_sec))
        seg = embeddings[seg_idx: seg_idx + n_seg_bins].astype(np.float64)
        seg_idx += n_seg_bins

        if hrf and hrf_kernel is not None:
            seg = _apply_hrf_to_segment(seg, hrf_kernel)

        if normalize and blockdiag:
            mu = seg.mean(axis=0, keepdims=True)
            sd = seg.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            seg = (seg - mu) / sd

        processed_segments.append(seg)

    all_emb = np.concatenate(processed_segments, axis=0)

    if normalize and not blockdiag:
        mu = all_emb.mean(axis=0, keepdims=True)
        sd = all_emb.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        all_emb = (all_emb - mu) / sd

    return all_emb.astype(np.float32)


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
