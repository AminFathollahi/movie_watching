"""
rsa/run_rdm_diagonal.py
========================
Compute within-movie (block-diagonal) and cross-movie (off-diagonal) RDMs,
and optionally run Glasser + searchlight RSA using those masked model RDMs.

Given a set of temporal embeddings and a timing CSV, this script:
  1. Assigns each temporal bin to its source video (movie).
  2. Computes the full pairwise RDM from the model embeddings.
  3. Produces two masked variants:
       off-diagonal  : retains only cross-movie similarity structure.
       block-diagonal: retains only within-movie similarity structure.

If fMRI data is provided (--preprocessed-dir), a second stage runs RSA for
each masked model RDM using both Glasser parcel-wise and vertex-wise searchlight
analysis.  Both Glasser and searchlight maps are saved in a single multi-map
CIFTI via save_cifti_multimap.

The within-movie (block-diagonal) RDM measures narrative/scene-level temporal
similarity.  The cross-movie (off-diagonal) RDM probes semantic generalisation
across distinct narratives.

Embedding path convention:
  {embeddings_dir}/{model}/bin{bin_sec}s_skip{skip_sec}s/{model}_{modality}.npy

Outputs — stage 1 (always):
  {output_dir}/{stem}_rdm_full.npy
  {output_dir}/{stem}_rdm_offdiag.npy      — NaN within-movie
  {output_dir}/{stem}_rdm_blockdiag.npy    — NaN cross-movie
  {output_dir}/{stem}_segment_labels.npy
  {output_dir}/{stem}_rdm_summary.json

Outputs — stage 2 (requires --preprocessed-dir + other fMRI args):
  {output_dir}/{stem}_rsa_diagonal_maps.dscalar.nii
    Maps (4):
      glasser_{method}_rho_offdiag
      glasser_{method}_rho_blockdiag
      searchlight_{method}_rho_offdiag
      searchlight_{method}_rho_blockdiag

Usage (RDM only):
  python rsa/run_rdm_diagonal.py \
      --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \
      --timing-csv     /home/amin/Research/Representation/Movie/data/movie_timing.csv \
      --output-dir     /home/amin/Research/Representation/Movie/outputs/rsa/raw/rdm_diagonal \
      --model omni3b_layer35 \
      --modality av --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0

Usage (RDM + RSA maps):
  python rsa/run_rdm_diagonal.py \
      --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \
      --timing-csv     /home/amin/Research/Representation/Movie/data/movie_timing.csv \
      --output-dir     /home/amin/Research/Representation/Movie/outputs/rsa/raw/rdm_diagonal \
      --model omni3b_layer35 \
      --modality av --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 \
      --preprocessed-dir /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw \
      --fmri-suffix raw \
      --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \
      --glasser-dlabel /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii \
      --left-surface   /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii \
      --right-surface  /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii \
      --workbench      /opt/workbench/bin_linux64/wb_command \
      --geodesic-cache-dir /home/amin/Research/Representation/Movie/outputs/rsa/_geodesic_cache \
      --k 100 --method spearman
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.rsa_utils import (
    load_fmri_cifti, preprocess_fmri,
    process_model_embeddings, align_and_assert_bins,
)
from rsa.shared.model_registry import (
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    TR_DEFAULT,
    DIAGONAL_MASK_DEFAULT,
)
from cifti_io import (
    get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute within/cross-movie RDMs and optional RSA CIFTI maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Stage 1: RDM computation ──────────────────────────────────────────────
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--timing-csv",     required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--model",
                   default=DIAGONAL_MASK_DEFAULT["model"])
    p.add_argument("--modality",
                   default=DIAGONAL_MASK_DEFAULT["modality"],
                   choices=["a", "v", "av"])
    p.add_argument("--bin-sec",   type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--skip-sec",  type=float, default=BIN_SEC_DEFAULT,
                   help="Sliding-window stride in seconds (default = bin-sec, no overlap).")
    p.add_argument("--delay-sec", type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr",        type=float, default=TR_DEFAULT)
    p.add_argument("--rdm-metric",
                   default="correlation",
                   choices=["cosine", "correlation", "euclidean"])

    # ── Stage 2: RSA CIFTI maps (all optional) ─────────────────────────────
    fmri = p.add_argument_group("RSA CIFTI maps (optional; all must be set together)")
    fmri.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                      help="Directory of continuous, cleaned CIFTIs "
                           "({dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii).")
    fmri.add_argument("--fmri-suffix", default="raw", dest="fmri_suffix",
                      help="Preprocessing suffix in CIFTI filenames.")
    fmri.add_argument("--subject", default="group_average",
                      help="Subject ID (used to locate preprocessed CIFTI).")
    fmri.add_argument("--template-cifti", default=None, dest="template_cifti",
                      help="Template 59k dscalar.nii for CIFTI output header.")
    fmri.add_argument("--glasser-dlabel", default=None, dest="glasser_dlabel",
                      help="Glasser MMP parcellation .dlabel.nii.")
    fmri.add_argument("--left-surface", default=None, dest="left_surface",
                      help="Left 59k midthickness .surf.gii.")
    fmri.add_argument("--right-surface", default=None, dest="right_surface",
                      help="Right 59k midthickness .surf.gii.")
    fmri.add_argument("--workbench", default=None,
                      help="Path to wb_command executable.")
    fmri.add_argument("--k", type=int, default=100,
                      help="Searchlight neighbourhood size (vertices).")
    fmri.add_argument("--method", default="spearman",
                      choices=["spearman", "pearson"],
                      help="RDM correlation method for RSA.")
    fmri.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir",
                      help="Shared geodesic k-NN cache directory.")
    fmri.add_argument("--gpu-batch-size", type=int, default=512, dest="gpu_batch_size",
                      help="Vertices per GPU batch.")
    fmri.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs",
                      help="CPU joblib workers (used only without CUDA).")
    return p.parse_args()


# =============================================================================
# Embedding path resolver (bin{N}s_skip{N}s convention)
# =============================================================================

def _resolve_emb(embeddings_dir: str, model: str, modality: str,
                 bin_sec: float, skip_sec: float) -> Path:
    """Return the embedding .npy path and raise FileNotFoundError if absent."""
    p = (Path(embeddings_dir) / model
         / f"bin{int(bin_sec)}s_skip{int(skip_sec)}s"
         / f"{model}_{modality}.npy")
    if not p.exists():
        raise FileNotFoundError(f"Embedding not found: {p}")
    return p


# =============================================================================
# Segment-label construction
# =============================================================================

def build_segment_labels(
    timing_df: pd.DataFrame,
    bin_sec: float,
    tr: float,
    run_trs,
    delay_sec: float = 0.0,
) -> np.ndarray:
    """Return integer video-index label for every temporal bin.

    Replicates the floor-binning from rsa_utils.preprocess_fmri so that the
    segment labels align exactly with the processed embedding and fMRI arrays.
    """
    run_col = (
        "run" if "run" in timing_df.columns
        else ("run_id" if "run_id" in timing_df.columns else None)
    )

    if run_col is None:
        groups = [(None, timing_df)]
        run_trs_list = [int(timing_df["duration_sec"].sum() / tr) + 1]
    else:
        groups = list(timing_df.groupby(run_col, sort=False))
        if run_trs is None:
            run_trs_list = [
                int(g["duration_sec"].sum() / tr) + 1 for _, g in groups
            ]
        else:
            run_trs_list = list(run_trs)

    all_video_ids = timing_df["video_id"].tolist() if "video_id" in timing_df.columns else list(range(len(timing_df)))
    unique_ids    = list(dict.fromkeys(all_video_ids))
    id_to_idx     = {vid: i for i, vid in enumerate(unique_ids)}

    labels = []
    for run_idx, (_, run_df) in enumerate(groups):
        run_tr_count = run_trs_list[run_idx]
        run_start_sec = sum(run_trs_list[:run_idx]) * tr

        for _, row in run_df.iterrows():
            n_bins = int(np.floor(row["duration_sec"] / bin_sec))
            if n_bins == 0:
                continue

            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            end_tr   = start_tr + n_bins * max(1, int(np.round(bin_sec / tr)))

            if start_tr >= run_tr_count or start_tr < 0:
                continue
            if end_tr > run_tr_count:
                n_bins = (run_tr_count - start_tr) // max(1, int(np.round(bin_sec / tr)))
                if n_bins == 0:
                    continue

            vid_id  = row.get("video_id", row.name)
            vid_idx = id_to_idx.get(vid_id, 0)
            labels.extend([vid_idx] * n_bins)

    return np.array(labels, dtype=np.int32)


# =============================================================================
# RDM masking
# =============================================================================

def mask_rdm_offdiagonal(rdm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Set within-movie pairs to NaN; return cross-movie RDM."""
    same = (labels[:, None] == labels[None, :])
    out  = rdm.copy().astype(np.float64)
    out[same] = np.nan
    return out


def mask_rdm_blockdiagonal(rdm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Set cross-movie pairs to NaN; return within-movie RDM."""
    diff = (labels[:, None] != labels[None, :])
    out  = rdm.copy().astype(np.float64)
    out[diff] = np.nan
    return out


# =============================================================================
# Valid pair indices (lower triangle only, matching the mask)
# =============================================================================

def _valid_pair_indices(labels: np.ndarray, cross_movie: bool):
    """Return (row_idx, col_idx) for the lower-triangle pairs matching the mask.

    Parameters
    ----------
    labels      : (n_bins,) int — video index per time bin
    cross_movie : True → off-diagonal (cross-movie), False → block-diagonal (within)

    Returns
    -------
    rows, cols : 1-D int arrays indexing into an (n_bins, n_bins) square RDM
    """
    tril_r, tril_c = np.tril_indices(len(labels), k=-1)
    if cross_movie:
        keep = labels[tril_r] != labels[tril_c]
    else:
        keep = labels[tril_r] == labels[tril_c]
    return tril_r[keep], tril_c[keep]


# =============================================================================
# Partial model RDM pre-computation (for fast per-vertex RSA)
# =============================================================================

def _precompute_partial_model_rdm(
    emb: np.ndarray,
    valid_row: np.ndarray,
    valid_col: np.ndarray,
    method: str,
) -> np.ndarray:
    """Extract valid pairs from the model RDM and normalise for fast RSA.

    Parameters
    ----------
    emb       : (n_bins, n_features) float32 — z-scored embeddings
    valid_row : (n_pairs,) int — row indices into (n_bins, n_bins) RDM
    valid_col : (n_pairs,) int — column indices
    method    : "spearman" | "pearson"

    Returns
    -------
    model_norm : (n_pairs,) float32 — rank-normalised (spearman) or
                 mean-centred unit-norm (pearson) model distances
    """
    emb64 = emb.astype(np.float64)
    mu    = emb64.mean(axis=1, keepdims=True)
    ec    = emb64 - mu
    nrms  = np.sqrt((ec ** 2).sum(axis=1, keepdims=True))
    nrms[nrms < 1e-10] = 1.0
    en    = ec / nrms
    sim   = en @ en.T                                        # (n_bins, n_bins)
    flat  = (1.0 - sim)[valid_row, valid_col].astype(np.float32)

    if method == "spearman":
        ranks = rankdata(flat).astype(np.float32)
        c     = ranks - ranks.mean()
        nf    = np.linalg.norm(c)
        return (c / nf).astype(np.float32) if nf > 1e-10 else c
    else:
        c  = flat - flat.mean()
        nf = np.linalg.norm(c)
        return (c / nf).astype(np.float32) if nf > 1e-10 else c


# =============================================================================
# Glasser parcellation loading (same as run_glasser.py)
# =============================================================================

def _load_glasser_parcels(dlabel_path: str, fmri_bm_axis) -> dict:
    img        = nib.load(dlabel_path)
    label_data = img.get_fdata(dtype=np.float32).squeeze().astype(np.int32)
    dlabel_bm  = img.header.get_axis(1)
    label_axis = img.header.get_axis(0)

    vertex_label: dict = {}
    for name, sl, struct in dlabel_bm.iter_structures():
        if "CORTEX" not in name:
            continue
        for local_i, vidx in enumerate(struct.vertex):
            vertex_label[(name, int(vidx))] = int(label_data[sl.start + local_i])

    vertex_fmri: dict = {}
    for name, sl, struct in fmri_bm_axis.iter_structures():
        if "CORTEX" not in name:
            continue
        for local_i, vidx in enumerate(struct.vertex):
            vertex_fmri[(name, int(vidx))] = sl.start + local_i

    key_to_indices: dict = {}
    for (hem, vidx), lbl in vertex_label.items():
        if lbl == 0:
            continue
        fmri_pos = vertex_fmri.get((hem, vidx))
        if fmri_pos is not None:
            key_to_indices.setdefault(lbl, []).append(fmri_pos)

    parcels = {}
    for key, (name, _rgba) in label_axis.label[0].items():
        if key == 0:
            continue
        indices = key_to_indices.get(key, [])
        if indices:
            parcels[name] = np.array(sorted(indices), dtype=np.int32)

    log.info(f"  Loaded {len(parcels)} Glasser parcels")
    return parcels


# =============================================================================
# Glasser partial RSA
# =============================================================================

def compute_partial_parcel_rsa(
    fmri: np.ndarray,
    emb: np.ndarray,
    parcels: dict,
    valid_row: np.ndarray,
    valid_col: np.ndarray,
    n_grayords: int,
    method: str,
) -> np.ndarray:
    """Parcel-wise RSA using only the valid (masked) RDM pairs.

    Parameters
    ----------
    fmri      : (n_bins, n_grayords) float32
    emb       : (n_bins, n_features) float32 — z-scored embeddings
    parcels   : dict — {name: grayordinate_indices}
    valid_row : (n_pairs,) int — valid RDM row indices
    valid_col : (n_pairs,) int — valid RDM column indices
    n_grayords: int
    method    : "spearman" | "pearson"

    Returns
    -------
    corr_map : (n_grayords,) float32
    """
    # Model partial RDM (flat, valid pairs only)
    emb64  = emb.astype(np.float64)
    mu     = emb64.mean(axis=1, keepdims=True)
    ec     = emb64 - mu
    nrm    = np.sqrt((ec ** 2).sum(axis=1, keepdims=True))
    nrm[nrm < 1e-10] = 1.0
    en     = ec / nrm
    sim    = en @ en.T
    model_flat = (1.0 - sim)[valid_row, valid_col]

    if len(model_flat) < 2:
        log.warning("  No valid pairs for this mask — returning zero map.")
        return np.zeros(n_grayords, dtype=np.float32)

    corr_map = np.zeros(n_grayords, dtype=np.float32)

    for name, indices in parcels.items():
        if len(indices) < 2:
            continue
        pf    = fmri[:, indices].astype(np.float64)
        mu_p  = pf.mean(axis=1, keepdims=True)
        ecp   = pf - mu_p
        np_   = np.sqrt((ecp ** 2).sum(axis=1, keepdims=True))
        np_[np_ < 1e-10] = 1.0
        enp   = ecp / np_
        sim_p = enp @ enp.T
        parcel_flat = (1.0 - sim_p)[valid_row, valid_col]

        if method == "spearman":
            r, _ = spearmanr(model_flat, parcel_flat)
        else:
            r, _ = pearsonr(model_flat, parcel_flat)

        corr_map[indices] = float(r) if not np.isnan(r) else 0.0

    return corr_map


# =============================================================================
# Searchlight helpers (k-NN caching — reused from run_searchlight.py)
# =============================================================================

def _get_neighbors(surface_path, workbench, subject, hem, k, cache_dir):
    """Thin wrapper: import and reuse run_searchlight.get_neighbors."""
    from rsa.run_searchlight import get_neighbors
    return get_neighbors(surface_path, workbench, subject, hem, k, cache_dir)


# =============================================================================
# CPU per-vertex partial searchlight
# =============================================================================

def _partial_searchlight_vertex(
    surf_v: int,
    fmri: np.ndarray,
    model_norm: np.ndarray,
    neighbors: np.ndarray,
    vertex_to_col: np.ndarray,
    valid_row: np.ndarray,
    valid_col: np.ndarray,
    method: str,
) -> float:
    """Per-vertex RSA using only the valid RDM pairs."""
    neighbor_surf = neighbors[surf_v]
    neighbor_cols = vertex_to_col[neighbor_surf]
    neighbor_cols = neighbor_cols[neighbor_cols >= 0]
    if len(neighbor_cols) < 2:
        return 0.0

    hood = fmri[:, neighbor_cols].astype(np.float64)
    mu   = hood.mean(axis=1, keepdims=True)
    hc   = hood - mu
    nrms = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
    nrms[nrms < 1e-10] = 1.0
    hn   = hc / nrms
    rdm  = 1.0 - hn @ hn.T                            # (n_bins, n_bins)
    fmri_flat = rdm[valid_row, valid_col].astype(np.float32)

    if method == "spearman":
        order = np.argsort(fmri_flat)
        fr    = np.empty(len(fmri_flat), dtype=np.float32)
        fr[order] = np.arange(len(fmri_flat), dtype=np.float32)
        fc  = fr - fr.mean()
        fn  = np.linalg.norm(fc)
        return float(np.dot(fc, model_norm) / fn) if fn > 1e-10 else 0.0
    else:
        fc = fmri_flat - fmri_flat.mean()
        fn = np.linalg.norm(fc)
        return float(np.dot(fc / fn, model_norm)) if fn > 1e-10 else 0.0


# =============================================================================
# GPU batched partial searchlight
# =============================================================================

def _run_partial_searchlight_gpu(
    fmri: np.ndarray,
    model_norm: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    valid_row: np.ndarray,
    valid_col: np.ndarray,
    method: str,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """GPU-batched partial searchlight RSA."""
    import torch

    n_verts = fmri.shape[1]
    n_bins  = fmri.shape[0]
    k       = neighbors.shape[1]

    model_norm_t = torch.from_numpy(model_norm).to(device)
    fmri_t       = torch.from_numpy(fmri.T.astype(np.float32)).to(device)

    valid_row_t = torch.tensor(valid_row, dtype=torch.long, device=device)
    valid_col_t = torch.tensor(valid_col, dtype=torch.long, device=device)

    neighbor_cols_all = vertex_to_col[neighbors]        # (n_surf_verts, k)
    surf_verts_for_v  = surface_indices.astype(np.int32)
    ncols_for_v       = neighbor_cols_all[surf_verts_for_v]  # (n_verts, k)

    full_k_mask     = np.all(ncols_for_v >= 0, axis=1)
    full_k_verts    = np.where(full_k_mask)[0]
    partial_k_verts = np.where(~full_k_mask)[0]

    log.info(
        f"  [GPU] {len(full_k_verts):,} full-k vertices (batch={batch_size}), "
        f"{len(partial_k_verts):,} partial-k vertices (CPU fallback)"
    )

    corr_map = np.zeros(n_verts, dtype=np.float32)

    for start in range(0, len(full_k_verts), batch_size):
        batch_v  = full_k_verts[start : start + batch_size]
        batch_nc = ncols_for_v[batch_v].astype(np.int64)
        B = len(batch_v)

        batch_nc_t = torch.from_numpy(batch_nc).to(device)
        hood = fmri_t[batch_nc_t]                        # (B, k, n_bins)
        hood = hood.permute(0, 2, 1).float()             # (B, n_bins, k)

        mu   = hood.mean(dim=2, keepdim=True)
        hc   = hood - mu
        nrms = torch.linalg.norm(hc, dim=2, keepdim=True).clamp(min=1e-10)
        hn   = hc / nrms                                 # (B, n_bins, k)

        rdm_full  = torch.bmm(hn, hn.permute(0, 2, 1))  # (B, n_bins, n_bins)
        fmri_flat = (1.0 - rdm_full)[:, valid_row_t, valid_col_t]  # (B, n_valid)

        if method == "spearman":
            order = torch.argsort(fmri_flat, dim=1)
            ranks = torch.argsort(order, dim=1).float()
            fc    = ranks - ranks.mean(dim=1, keepdim=True)
            fn    = torch.linalg.norm(fc, dim=1, keepdim=True).clamp(min=1e-10)
            rho   = (fc / fn * model_norm_t).sum(dim=1)
        else:
            fc  = fmri_flat - fmri_flat.mean(dim=1, keepdim=True)
            fn  = torch.linalg.norm(fc, dim=1, keepdim=True).clamp(min=1e-10)
            rho = (fc / fn * model_norm_t).sum(dim=1)

        corr_map[batch_v] = rho.cpu().numpy().astype(np.float32)

    if len(partial_k_verts) > 0:
        fmri_cpu = fmri_t.cpu().numpy().T
        for v in partial_k_verts:
            sv = int(surf_verts_for_v[v])
            corr_map[v] = _partial_searchlight_vertex(
                sv, fmri_cpu, model_norm, neighbors, vertex_to_col,
                valid_row, valid_col, method,
            )

    return corr_map


# =============================================================================
# Public partial searchlight entry point
# =============================================================================

def run_partial_searchlight(
    fmri: np.ndarray,
    emb: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    valid_row: np.ndarray,
    valid_col: np.ndarray,
    method: str = "spearman",
    n_jobs: int = -1,
    batch_size: int = 512,
) -> np.ndarray:
    """Searchlight RSA using only the valid (masked) RDM pairs.

    Parameters
    ----------
    fmri           : (n_bins, n_hem_verts) float32
    emb            : (n_bins, n_features) float32 — z-scored embeddings
    neighbors      : (n_surf_verts, k) int32
    surface_indices: (n_hem_verts,) int32 — surface vertex per grayordinate
    vertex_to_col  : (n_surf_verts,) int32 — surface vertex → fmri column
    valid_row      : (n_pairs,) int — valid row indices into (n_bins, n_bins) RDM
    valid_col      : (n_pairs,) int — valid column indices
    method         : "spearman" | "pearson"
    n_jobs         : CPU joblib workers (used only without CUDA)
    batch_size     : GPU batch size

    Returns
    -------
    corr_map : (n_hem_verts,) float32
    """
    n_verts  = fmri.shape[1]
    model_norm = _precompute_partial_model_rdm(emb, valid_row, valid_col, method)

    # Attempt GPU
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            log.info(f"  Using GPU partial searchlight (device=cuda, batch={batch_size})")
            try:
                return _run_partial_searchlight_gpu(
                    fmri, model_norm, neighbors, surface_indices,
                    vertex_to_col, valid_row, valid_col, method,
                    batch_size=batch_size, device="cuda",
                )
            except _torch.cuda.OutOfMemoryError:
                log.warning("  GPU OOM — falling back to CPU")
                _torch.cuda.empty_cache()
        else:
            log.info("  CUDA not available — using CPU searchlight")
    except ImportError:
        log.info("  torch not installed — using CPU searchlight")

    # CPU joblib path
    log.info(f"  Running CPU partial searchlight ({n_verts:,} vertices, n_jobs={n_jobs}) ...")
    corr_map = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_partial_searchlight_vertex)(
                int(surface_indices[v]), fmri, model_norm,
                neighbors, vertex_to_col,
                valid_row, valid_col, method,
            )
            for v in range(n_verts)
        ),
        dtype=np.float32,
    )
    return corr_map


# =============================================================================
# Stage 2: RSA CIFTI maps
# =============================================================================

def _run_rsa_maps(args, emb, labels, timing_df, out_dir, stem):
    """Load fMRI, run Glasser + searchlight RSA for off-diagonal and block-diagonal."""
    bin_sec_int = int(args.bin_sec)
    method      = args.method

    # ── fMRI loading ─────────────────────────────────────────────────────────
    cifti_path = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")
    trs_path   = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_run_trs.npy")

    log.info(f"  Loading fMRI: {cifti_path}")
    fmri_continuous = load_fmri_cifti(str(cifti_path))
    run_trs         = np.load(str(trs_path))
    fmri_bm_axis    = get_bm_axis(str(cifti_path))
    log.info(f"  fMRI shape: {fmri_continuous.shape}  run_trs: {run_trs.tolist()}")

    fmri_binned = preprocess_fmri(
        fmri_continuous, timing_df, run_trs,
        args.bin_sec, args.tr, args.delay_sec,
    )
    log.info(f"  fMRI binned & z-scored: {fmri_binned.shape}")
    del fmri_continuous

    # Re-load embeddings with hemodynamic delay aligned to the fMRI binning.
    # Must use the CIFTI run_trs (already loaded above), NOT timing-based run_trs
    # computed from clip durations — the latter underestimates run lengths and
    # silently drops clips near each run's end (e.g. video5, video14, video18).
    emb_file = _resolve_emb(args.embeddings_dir, args.model,
                            args.modality, args.bin_sec, args.skip_sec)
    emb_aligned = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec,
    )
    fmri_binned, emb_aligned = align_and_assert_bins(fmri_binned, emb_aligned)
    n_bins     = fmri_binned.shape[0]
    n_grayords = fmri_binned.shape[1]
    log.info(f"  Aligned: n_bins={n_bins}  n_grayords={n_grayords}")

    # Recompute labels aligned to the delay-shifted bins
    labels_aligned = build_segment_labels(
        timing_df, args.bin_sec, args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec,
    )
    if len(labels_aligned) != n_bins:
        raise ValueError(
            f"Aligned label count ({len(labels_aligned)}) ≠ n_bins ({n_bins}). "
            f"Check bin_sec, delay_sec, and timing CSV."
        )

    # ── Valid pair indices ────────────────────────────────────────────────────
    off_row, off_col = _valid_pair_indices(labels_aligned, cross_movie=True)
    blk_row, blk_col = _valid_pair_indices(labels_aligned, cross_movie=False)
    log.info(f"  Off-diagonal valid pairs: {len(off_row):,}  "
             f"Block-diagonal: {len(blk_row):,}")

    # ── Glasser RSA ───────────────────────────────────────────────────────────
    parcels = _load_glasser_parcels(args.glasser_dlabel, fmri_bm_axis)

    log.info(f"  Glasser partial RSA: off-diagonal ...")
    glass_off = compute_partial_parcel_rsa(
        fmri_binned, emb_aligned, parcels, off_row, off_col, n_grayords, method,
    )
    log.info(f"  Glasser partial RSA: block-diagonal ...")
    glass_blk = compute_partial_parcel_rsa(
        fmri_binned, emb_aligned, parcels, blk_row, blk_col, n_grayords, method,
    )

    # ── Searchlight RSA ───────────────────────────────────────────────────────
    lh_verts, rh_verts = get_cortex_vertex_indices(fmri_bm_axis)
    n_left = len(lh_verts)

    if args.geodesic_cache_dir:
        cache_dir = Path(args.geodesic_cache_dir)
    else:
        cache_dir = Path(args.output_dir) / "_geodesic_cache"

    surfaces = {
        "left":  (args.left_surface,  fmri_binned[:, :n_left],  lh_verts),
        "right": (args.right_surface, fmri_binned[:, n_left:],   rh_verts),
    }

    search_off  = np.zeros(n_grayords, dtype=np.float32)
    search_blk  = np.zeros(n_grayords, dtype=np.float32)
    offset = 0

    for hem, (surf_path, fmri_hem, surf_idx) in surfaces.items():
        neighbors = _get_neighbors(
            surf_path, args.workbench, args.subject, hem, args.k, cache_dir,
        )
        n_surf  = neighbors.shape[0]
        surf_idx = surf_idx.astype(np.int32)
        v2c = np.full(n_surf, -1, dtype=np.int32)
        v2c[surf_idx] = np.arange(len(surf_idx), dtype=np.int32)

        log.info(f"  Searchlight partial RSA [{hem}]: off-diagonal ...")
        hem_off = run_partial_searchlight(
            fmri_hem, emb_aligned, neighbors, surf_idx, v2c,
            off_row, off_col, method, args.n_jobs, args.gpu_batch_size,
        )
        log.info(f"  Searchlight partial RSA [{hem}]: block-diagonal ...")
        hem_blk = run_partial_searchlight(
            fmri_hem, emb_aligned, neighbors, surf_idx, v2c,
            blk_row, blk_col, method, args.n_jobs, args.gpu_batch_size,
        )

        n_hem = hem_off.shape[0]
        search_off[offset : offset + n_hem] = hem_off
        search_blk[offset : offset + n_hem] = hem_blk
        offset += n_hem

    # ── Save 4-map CIFTI ──────────────────────────────────────────────────────
    delay_tag   = f"delay{int(args.delay_sec)}s"
    maps_stem   = f"{stem}_rsa_{delay_tag}_{method}_diagonal_maps"
    out_path    = out_dir / f"{maps_stem}.dscalar.nii"

    map_names = [
        f"glasser_{method}_rho_offdiag",
        f"glasser_{method}_rho_blockdiag",
        f"searchlight_{method}_rho_offdiag",
        f"searchlight_{method}_rho_blockdiag",
    ]
    data_2d = np.stack([glass_off, glass_blk, search_off, search_blk], axis=0)

    save_cifti_multimap(data_2d, map_names, args.template_cifti, str(out_path))
    log.info(f"  Saved diagonal RSA maps: {out_path.name}")
    for i, name in enumerate(map_names):
        d = data_2d[i]
        log.info(f"    {name}: mean={d.mean():.4f}  max={d.max():.4f}")

    return str(out_path)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ── Resolve embedding path and load ─────────────────────────────────────
    emb_file = _resolve_emb(args.embeddings_dir, args.model,
                            args.modality, args.bin_sec, args.skip_sec)
    timing_df = pd.read_csv(args.timing_csv)
    log.info(f"Embeddings: {emb_file}")
    log.info(f"Timing CSV: {args.timing_csv}  ({len(timing_df)} clips)")

    # Load CIFTI run_trs from the preprocessed dir so that process_model_embeddings
    # uses the real scan run lengths for boundary calculations.  timing-based
    # run_trs (sum of clip durations) underestimates run lengths and silently
    # drops clips near each run's end (video5, video9, video14, video18, etc.)
    # because their global onset_sec exceeds the estimated run boundary.
    if args.preprocessed_dir:
        trs_path = (Path(args.preprocessed_dir) /
                    f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
        _emb_run_trs = np.load(str(trs_path))
        log.info(f"run_trs (from CIFTI): {_emb_run_trs.tolist()}")
    else:
        raise RuntimeError(
            "--preprocessed-dir is required so that run_trs can be read from "
            f"{{preprocessed_dir}}/{{subject}}_{{fmri_suffix}}_run_trs.npy.  "
            "Timing-based run_trs silently drops clips near each run's end."
        )
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, tr=args.tr,
        run_trs=_emb_run_trs, delay_sec=0.0,
    )
    n_bins = emb.shape[0]
    log.info(f"Embedding shape: {emb.shape}  (n_bins={n_bins})")

    # ── Build segment labels ─────────────────────────────────────────────────
    labels = build_segment_labels(
        timing_df, args.bin_sec, args.tr,
        run_trs=_emb_run_trs, delay_sec=0.0,
    )
    if len(labels) != n_bins:
        raise ValueError(
            f"Label count ({len(labels)}) ≠ embedding bin count ({n_bins}).\n"
            f"Check that bin_sec and timing CSV match the embedding file."
        )
    n_videos = int(labels.max()) + 1
    log.info(f"Segment labels: {len(labels)} bins across {n_videos} videos")

    # ── Compute full RDM ─────────────────────────────────────────────────────
    log.info(f"Computing full RDM (metric={args.rdm_metric}) ...")
    rdm_full = squareform(pdist(emb.astype(np.float64), metric=args.rdm_metric))
    log.info(f"  RDM shape: {rdm_full.shape}")

    # ── Masked variants ──────────────────────────────────────────────────────
    rdm_off   = mask_rdm_offdiagonal(rdm_full, labels)
    rdm_block = mask_rdm_blockdiagonal(rdm_full, labels)

    off_nonnan   = np.sum(~np.isnan(rdm_off))
    block_nonnan = np.sum(~np.isnan(rdm_block))
    total_cells  = n_bins * n_bins
    log.info(f"  Off-diagonal non-NaN: {off_nonnan:,}/{total_cells:,} "
             f"({100*off_nonnan/total_cells:.1f}%)")
    log.info(f"  Block-diagonal non-NaN: {block_nonnan:,}/{total_cells:,} "
             f"({100*block_nonnan/total_cells:.1f}%)")

    # ── Save stage-1 outputs ─────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_sec_int  = int(args.bin_sec)
    skip_sec_int = int(args.skip_sec)
    stem = f"{args.model}_{args.modality}_bin{bin_sec_int}s_skip{skip_sec_int}s"

    np.save(out_dir / f"{stem}_rdm_full.npy",       rdm_full.astype(np.float32))
    np.save(out_dir / f"{stem}_rdm_offdiag.npy",    rdm_off.astype(np.float32))
    np.save(out_dir / f"{stem}_rdm_blockdiag.npy",  rdm_block.astype(np.float32))
    np.save(out_dir / f"{stem}_segment_labels.npy", labels)

    summary = {
        "model":              args.model,
        "modality":           args.modality,
        "bin_sec":            args.bin_sec,
        "rdm_metric":         args.rdm_metric,
        "n_bins":             int(n_bins),
        "n_videos":           int(n_videos),
        "off_nonnan_pct":     float(100 * off_nonnan / total_cells),
        "block_nonnan_pct":   float(100 * block_nonnan / total_cells),
        "rdm_full_mean":      float(np.nanmean(rdm_full)),
        "rdm_off_mean":       float(np.nanmean(rdm_off)),
        "rdm_block_mean":     float(np.nanmean(rdm_block)),
    }
    (out_dir / f"{stem}_rdm_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Stage-1 outputs written to: {out_dir}")

    # ── Stage 2: RSA CIFTI maps ───────────────────────────────────────────────
    need_rsa = all([
        args.preprocessed_dir,
        args.template_cifti,
        args.glasser_dlabel,
        args.left_surface,
        args.right_surface,
        args.workbench,
    ])

    if need_rsa:
        log.info("Stage-2: running Glasser + searchlight RSA with diagonal masks ...")
        cifti_path = _run_rsa_maps(args, emb, labels, timing_df, out_dir, stem)
        summary["rsa_cifti"] = cifti_path
        (out_dir / f"{stem}_rdm_summary.json").write_text(json.dumps(summary, indent=2))
    else:
        missing = [f for f, v in [
            ("--preprocessed-dir", args.preprocessed_dir),
            ("--template-cifti",   args.template_cifti),
            ("--glasser-dlabel",   args.glasser_dlabel),
            ("--left-surface",     args.left_surface),
            ("--right-surface",    args.right_surface),
            ("--workbench",        args.workbench),
        ] if not v]
        if missing:
            log.info(f"Skipping stage-2 RSA maps (missing: {', '.join(missing)}).")


if __name__ == "__main__":
    main()
