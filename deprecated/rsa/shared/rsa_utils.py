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

from shared.binning import (  # noqa: F401
    _apply_hrf_to_segment,
    align_and_assert_bins,
    assert_segment_timing,
    get_run_bin_counts,
    load_fmri_cifti,
    preprocess_fmri,
    process_model_embeddings,
    spm_hrf,
)


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


# =============================================================================
# fMRI loading and binning
# =============================================================================


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


# =============================================================================
# Model embedding processing
# =============================================================================


# =============================================================================
# RDM and RSA
# =============================================================================


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


def load_subject_maps(files, name, n_blocks=1):
    """Stack map `name` of per-subject searchlight.py files and, with n_blocks > 1, their block maps `{name}_block01` ...
    averaged down to n_blocks (None when absent or not a multiple of n_blocks)."""
    rho, blocks = [], []
    for f in files:
        img = nib.load(str(f))
        names = list(img.header.get_axis(0).name)
        data = np.asarray(img.dataobj, dtype=np.float32)
        rho.append(data[names.index(name)])
        blocks.append(data[[i for i, n in enumerate(names) if n.startswith(f"{name}_block")]])
    counts = {len(b) for b in blocks}
    n_file = counts.pop() if len(counts) == 1 else 0
    if n_blocks <= 1 or not n_file or n_file % n_blocks:
        return np.stack(rho), None
    return np.stack(rho), aggregate_blocks(np.stack(blocks), n_blocks)
