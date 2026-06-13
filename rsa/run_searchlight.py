"""
rsa/run_searchlight.py
===================================
Vertex-wise searchlight RSA on HCP 7T movie-watching fMRI data.

For each vertex, a geodesic neighbourhood of k nearest vertices is assembled.
The fMRI RDM within that neighbourhood is correlated (Spearman or Pearson)
with the model RDM. The resulting correlation map is saved as a multi-map
CIFTI dscalar.

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Continuous, cleaned CIFTI produced by preprocess_individual.py.
  Expected at:
    {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii
    {preprocessed_dir}/{subject}_{fmri_suffix}_run_trs.npy

STREAMING MODE (--raw-dir):
  Raw 7T CIFTI dtseries files. The subject is continuously preprocessed on-the-fly;
  no CIFTI is saved.

Geodesic caching
----------------
Two-level cache per (subject, hemisphere, k):
  1. {subject}_{hem}_geodesic.dconn.nii  — full NxN distances (large, ~14 GB)
  2. {subject}_{hem}_neighbors_k{k}.npy  — extracted k-NN indices (small, ~13 MB)

Fast path: if the .npy exists, load it directly — no 14 GB file needed.
Slow path: compute/load dconn → chunk-extract k-NN → save .npy → delete dconn
           (dconn is deleted immediately for per-subject to free 28 GB/subject;
            group_average dconn is kept so other k values can reuse it).

Usage (disk mode):
  python run_searchlight.py \
      --preprocessed-dir <path> --fmri-suffix raw \
      --subject group_average --timing-csv <path> \
      --embeddings-dir <path> --template-cifti <path> \
      --left-surface <path> --right-surface <path> \
      --workbench <path> --output-dir <path> \
      --model pe-av-small-16-frame --modality av \
      --k 100 --bin-sec 5.0 --delay-sec 5.0 \
      --method spearman --tr 1.0
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.shared.rsa_utils import (
    load_fmri_cifti, preprocess_fmri,
    process_model_embeddings, align_and_assert_bins,
    assert_segment_timing,
)
from cifti_io import (
    get_bm_axis, get_cortex_vertex_indices,
    get_combined_map_names, merge_into_combined,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# n_jobs for joblib parallelism inside each subject's searchlight.
# Read from env var _RSA_N_JOBS (set by run_analysis.sh based on BATCH_SIZE).
# Default -1 (all CPUs) is fine when running a single subject, but MUST be
# reduced when many subjects run in parallel to avoid thread oversubscription.
# Rule of thumb: _RSA_N_JOBS = max(1, nproc // BATCH_SIZE)
N_JOBS = int(os.environ.get("_RSA_N_JOBS", -1))

# Rows of the dconn loaded per chunk when extracting k-NN.
# Each chunk uses ~chunk_size × n_surface_verts × 4 bytes of RAM.
# 1000 rows × 59292 verts × 4 bytes ≈ 237 MB — safe on any modern machine.
_DCONN_CHUNK = 1_000


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Vertex-wise searchlight RSA on movie fMRI (59k grayordinate space).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk mode] Directory of continuous cleaned CIFTIs. "
                          "File: {dir}/{subject}_{fmri-suffix}_cortex_59k.dtseries.nii")
    inp.add_argument("--fmri-suffix", default="raw", dest="fmri_suffix",
                     help="[disk mode] Filename suffix encoding preprocessing.")
    inp.add_argument("--raw-dir", default=None,
                     help="[streaming mode] Root directory of raw 7T CIFTI files.")

    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True,
                   help="Template 59k dscalar.nii for CIFTI output header.")
    p.add_argument("--left-surface", required=True,
                   help="Left 59k midthickness .surf.gii.")
    p.add_argument("--right-surface", required=True,
                   help="Right 59k midthickness .surf.gii.")
    p.add_argument("--workbench", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", default="group_average")
    p.add_argument("--model", required=True)
    p.add_argument("--modality", required=True, choices=["v", "a", "av", "at", "vt", "avt", "t"])
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Applied when slicing stimulus blocks from continuous fMRI.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF.")
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec",
                   help="Window stride in seconds (default: bin-sec, i.e. no overlap).")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"])
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--combined-output", default=None, dest="combined_output",
                   help="Path to a combined .dscalar.nii that accumulates maps from "
                        "both searchlight and Glasser runs. This script adds/replaces "
                        "the 'searchlight_{method}_rho' map.")
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir",
                   help="Shared directory for geodesic k-NN .npy caches (preprocessing-"
                        "agnostic). Defaults to {output_dir}/_geodesic_cache if not set. "
                        "All analysis variants should point here so the k_max derivation "
                        "logic works across preprocessing configs.")

    p.add_argument("--gpu-batch-size", type=int, default=512, dest="gpu_batch_size",
                   help="Vertices per GPU batch (default 512; reduce if GPU OOM).")

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true")
    prep.add_argument("--psc", default=False, action="store_true")
    prep.add_argument("--gsr", default=False, action=argparse.BooleanOptionalAction)

    return p.parse_args()


# =============================================================================
# Geodesic distance — two-level cache
# =============================================================================

def _compute_geodesic_dconn(surface_path: str, workbench: str,
                              dconn_path: Path) -> None:
    """Run wb_command to compute all-to-all geodesic distances."""
    if dconn_path.exists():
        log.info(f"  dconn cached: {dconn_path.name}")
        return
    log.info(f"  Computing geodesic distances → {dconn_path.name} ...")
    cmd = [workbench, "-surface-geodesic-distance-all-to-all",
           surface_path, str(dconn_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"wb_command failed (exit {result.returncode}):\n{result.stderr}")
    log.info(f"  dconn saved: {dconn_path.name}")


def _dconn_is_complete(dconn_path: Path, min_size_gb: float = 13.0) -> bool:
    """Return True if a dconn.nii file looks fully written (>= min_size_gb).

    For a 59k surface, the expected dconn size is ~14 GB
    (59292 × 59292 × 4 bytes).  A threshold of 13 GB (≈93 % of expected)
    reliably catches partial files written by a crashed wb_command.
    The previous default of 1 GB was too permissive — a wb_command that ran
    for ~6 min before being killed could already have written >1 GB.
    """
    try:
        return dconn_path.stat().st_size >= int(min_size_gb * 1024 ** 3)
    except OSError:
        return False


def _extract_knn_from_dconn(dconn_path: Path, k: int,
                              chunk_size: int = _DCONN_CHUNK) -> np.ndarray:
    """Extract k-nearest neighbours from a dconn.nii file in row chunks."""
    img = nib.load(str(dconn_path))
    proxy = img.dataobj
    n_verts = img.shape[0]
    log.info(f"  Extracting k={k} neighbours from {dconn_path.name} "
             f"({n_verts} vertices, chunk={chunk_size}) ...")

    neighbors = np.empty((n_verts, k), dtype=np.int32)

    for start in range(0, n_verts, chunk_size):
        end = min(start + chunk_size, n_verts)
        chunk = np.asarray(proxy[start:end], dtype=np.float32)  # (chunk, n_verts)

        # Mask self-distances
        for local_i in range(end - start):
            chunk[local_i, start + local_i] = np.inf

        # argpartition gives the k smallest
        part = np.argpartition(chunk, k, axis=1)[:, :k]
        part_dists = np.take_along_axis(chunk, part, axis=1)
        order = np.argsort(part_dists, axis=1)
        neighbors[start:end] = np.take_along_axis(part, order, axis=1).astype(np.int32)

        del chunk
        gc.collect()

    return neighbors


def _parse_k_from_npy(path: Path) -> int:
    """Extract the k value encoded in a cache filename.

    Expected pattern: ``{anything}_neighbors_k{K}.npy``
    Returns -1 if the pattern is not found (file is not a valid cache).
    """
    m = re.search(r"_neighbors_k(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


def get_neighbors(surface_path: str, workbench: str, subject: str,
                  hem: str, k: int, cache_dir: Path) -> np.ndarray:
    """Return (n_surf_verts, k) int32 k-NN array, using a three-level cache.

    Cache filename convention
    ------------------------
    All cache files follow: ``{subject}_{hem}_neighbors_k{K}.npy``
    The K value in the filename is the number of neighbours stored in that
    file.  Any file with K_file >= k can serve a request for k neighbours.

    Cache lookup order
    ------------------
    Level 1 — exact match
        ``{subject}_{hem}_neighbors_k{k}.npy`` exists → load directly.

    Level 2 — k_max derivation (k_file > k)
        One or more ``_neighbors_k{K_file}.npy`` files exist with K_file > k.
        Use the *smallest* such K_file (minimum memory load), slice to k
        columns, save the result as the exact-match file, return.
        A single precompute run at k=150 therefore covers all k ≤ 150
        for both group_average and per-subject without re-running wb_command.

    Level 3 — geodesic fallback (no usable cache)
        No file with K_file >= k exists.  Run wb_command to compute the full
        all-to-all geodesic distance matrix, extract k-NN, save as
        ``{subject}_{hem}_neighbors_k{k}.npy``, delete the 14 GB dconn
        (per-subject only; group_average dconn is kept for future k values).

    This logic applies identically to group_average and individual subjects.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path   = cache_dir / f"{subject}_{hem}_neighbors_k{k}.npy"
    dconn_path = cache_dir / f"{subject}_{hem}_geodesic.dconn.nii"

    # ── Level 1: exact k cache ─────────────────────────────────────────────────
    if npy_path.exists():
        log.info(f"  [L1] k-NN cache hit (exact k={k}): {npy_path.name}")
        return np.load(str(npy_path))

    # ── Level 2: derive from smallest available k_max > k ─────────────────────
    # Scan for all valid cache files for this subject+hemisphere, keep only
    # those with K_file > k, pick the smallest (minimum data to load/slice).
    best_kmax, best_path = None, None
    for candidate in cache_dir.glob(f"{subject}_{hem}_neighbors_k*.npy"):
        k_file = _parse_k_from_npy(candidate)
        if k_file < 0:
            continue                          # filename doesn't match convention
        if k_file > k:
            if best_kmax is None or k_file < best_kmax:
                best_kmax, best_path = k_file, candidate

    if best_path is not None:
        log.info(
            f"  [L2] Deriving k={k} from k={best_kmax} cache "
            f"(smallest available ≥ k): {best_path.name}"
        )
        neighbors = np.ascontiguousarray(np.load(str(best_path))[:, :k])
        np.save(str(npy_path), neighbors)
        log.info(f"  [L2] Saved derived cache: {npy_path.name}")
        return neighbors

    # ── Level 3: no usable cache — compute from geodesic distances ────────────
    log.info(
        f"  [L3] No cache with K_file >= {k} found for {subject}/{hem}. "
        f"Running wb_command geodesic → {npy_path.name}"
    )

    # Guard against partially-written dconn left by a previous crashed run.
    if dconn_path.exists() and not _dconn_is_complete(dconn_path):
        log.warning(
            f"  dconn appears incomplete ({dconn_path.stat().st_size / 1e9:.2f} GB) "
            f"— deleting and recomputing: {dconn_path.name}"
        )
        dconn_path.unlink()

    _compute_geodesic_dconn(surface_path, workbench, dconn_path)
    neighbors = _extract_knn_from_dconn(dconn_path, k)
    np.save(str(npy_path), neighbors)
    log.info(f"  [L3] Saved new cache: {npy_path.name}")

    # Free the 14 GB dconn for individual subjects; keep for group_average so
    # that future requests for larger k can reuse it without re-running wb_command.
    if subject != "group_average" and dconn_path.exists():
        dconn_path.unlink()
        log.info(f"  [L3] Deleted dconn (per-subject): {dconn_path.name}")

    return neighbors


# =============================================================================
# Searchlight RSA
# =============================================================================

def _precompute_model_rdm(emb: np.ndarray, n_bins: int,
                           tril_idx: tuple, method: str) -> tuple:
    """Precompute model RDM condensed flat form + normalised ranks (once per hemisphere).

    Returns
    -------
    model_rdm_flat : (n_pairs,) float32 — condensed model RDM (same ordering as tril_idx)
    model_norm     : (n_pairs,) float32 — normalised ranks ready for fast Pearson
                     (for Spearman; for Pearson, the normalised flat values)
    """
    emb64 = emb.astype(np.float64)
    mu = emb64.mean(axis=1, keepdims=True)
    ec = emb64 - mu
    norms = np.sqrt((ec ** 2).sum(axis=1, keepdims=True))
    norms[norms < 1e-10] = 1.0
    en = ec / norms
    sim = en @ en.T                                     # (n_bins, n_bins)
    model_flat = (1.0 - sim)[tril_idx].astype(np.float32)

    if method == "spearman":
        ranks = rankdata(model_flat).astype(np.float32)
        centered = ranks - ranks.mean()
        norm_factor = np.linalg.norm(centered)
        model_norm = (centered / norm_factor).astype(np.float32) if norm_factor > 1e-10 else centered
    else:
        # Pearson: normalise the raw distances
        centered = model_flat - model_flat.mean()
        norm_factor = np.linalg.norm(centered)
        model_norm = (centered / norm_factor).astype(np.float32) if norm_factor > 1e-10 else centered

    return model_flat, model_norm


def _searchlight_vertex_fast(surf_v: int, fmri: np.ndarray,
                               model_norm: np.ndarray,
                               neighbors: np.ndarray,
                               vertex_to_col: np.ndarray,
                               tril_idx: tuple,
                               method: str) -> float:
    """Optimised per-vertex RSA: float32 matmul RDM + Spearman via argsort.

    Replaces squareform(pdist) + spearmanr with:
      - float32 row-normalised matmul for the fMRI RDM (matches GPU path precision)
      - Pre-ranked model (scipy.rankdata, tie-safe) + argsort rank of fMRI RDM
      - Pearson dot product on ranks (= Spearman)

    Parameters
    ----------
    surf_v        : int — surface vertex index (0..n_surf_verts-1)
    fmri          : (n_bins, n_hem_verts) float32 — hemisphere fMRI data
    model_norm    : (n_pairs,) float32 — pre-ranked+normalised model RDM vector
    neighbors     : (n_surf_verts, k) int32 — geodesic k-NN in surface space
    vertex_to_col : (n_surf_verts,) int32 — maps surface vertex → fmri column (-1=medial)
    tril_idx      : tuple — np.tril_indices(n_bins, k=-1), precomputed once
    method        : "spearman" or "pearson"
    """
    neighbor_surf = neighbors[surf_v]
    neighbor_cols = vertex_to_col[neighbor_surf]
    neighbor_cols = neighbor_cols[neighbor_cols >= 0]
    if len(neighbor_cols) < 2:
        return 0.0

    hood = fmri[:, neighbor_cols]   # float32, matching GPU path

    # Row-normalise to get unit correlation vectors (fast RDM via matmul)
    mu = hood.mean(axis=1, keepdims=True)
    hc = hood - mu
    norms = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
    norms[norms < 1e-10] = 1.0
    hn = hc / norms
    fmri_flat = (1.0 - hn @ hn.T)[tril_idx]   # float32 condensed RDM

    if method == "spearman":
        # Rank-order the fMRI distances; Pearson on ranks = Spearman
        order = np.argsort(fmri_flat)
        fr = np.empty(len(fmri_flat), dtype=np.float32)
        fr[order] = np.arange(len(fmri_flat), dtype=np.float32)
        fc = fr - fr.mean()
        fn = np.linalg.norm(fc)
        return float(np.dot(fc, model_norm) / fn) if fn > 1e-10 else 0.0
    else:
        # Pearson: normalise fMRI flat and dot with pre-normalised model
        fc = fmri_flat - fmri_flat.mean()
        fn = np.linalg.norm(fc)
        return float(np.dot(fc / fn, model_norm)) if fn > 1e-10 else 0.0


def run_searchlight(fmri: np.ndarray, model_emb: np.ndarray,
                    neighbors: np.ndarray,
                    surface_indices: np.ndarray,
                    vertex_to_col: np.ndarray,
                    method: str = "spearman",
                    n_jobs: int = -1,
                    batch_size: int = 512) -> np.ndarray:
    """Searchlight RSA across all grayordinate vertices.

    GPU is attempted unconditionally when CUDA is available; CUDA OOM triggers
    automatic fallback to CPU joblib.  Pass ``batch_size`` to tune GPU memory use.

    Parameters
    ----------
    fmri           : (n_bins, n_hem_verts) float32 — hemisphere fMRI
    model_emb      : (n_bins, n_features) float32 — model embeddings (for RDM)
    neighbors      : (n_surf_verts, k) int32 — geodesic k-NN
    surface_indices: (n_hem_verts,) int32 — surface vertex index per grayordinate
    vertex_to_col  : (n_surf_verts,) int32 — surface vertex → fmri column (-1=medial)
    method         : "spearman" | "pearson"
    n_jobs         : joblib parallel workers (-1 = all CPUs; for CPU fallback)
    batch_size     : int — vertices per GPU batch (default 512; reduce if OOM).
    """
    n_verts = fmri.shape[1]
    n_bins = fmri.shape[0]
    tril_idx = np.tril_indices(n_bins, k=-1)

    # Always attempt GPU when CUDA is available; fall back to CPU on OOM
    try:
        import torch as _torch
        # Collect all GPU OOM exception types across PyTorch versions.
        # Older PyTorch raises cuda.OutOfMemoryError; newer versions may
        # raise torch.AcceleratorError instead.
        _oom_types = [_torch.cuda.OutOfMemoryError]
        if hasattr(_torch, "AcceleratorError"):
            _oom_types.append(_torch.AcceleratorError)
        _oom_types = tuple(_oom_types)

        if _torch.cuda.is_available():
            log.info(f"  Using GPU searchlight (device=cuda, batch_size={batch_size})")
            try:
                return run_searchlight_gpu(
                    fmri, model_emb, neighbors, surface_indices,
                    vertex_to_col, method, batch_size=batch_size, device="cuda",
                )
            except _oom_types as e:
                log.warning(f"  GPU OOM ({type(e).__name__}) — falling back to CPU searchlight")
                _torch.cuda.empty_cache()
        else:
            log.info("  CUDA not available — using CPU searchlight")
    except ImportError:
        log.info("  torch not installed — using CPU searchlight")

    # CPU joblib path
    log.info(f"  Precomputing model RDM ({method}) ...")
    _, model_norm = _precompute_model_rdm(model_emb, n_bins, tril_idx, method)

    log.info(f"  Running searchlight on {n_verts} vertices "
             f"(n_jobs={n_jobs}) ...")
    corr_map = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_searchlight_vertex_fast)(
                int(surface_indices[v]), fmri, model_norm,
                neighbors, vertex_to_col, tril_idx, method
            )
            for v in range(n_verts)
        ),
        dtype=np.float32,
    )
    return corr_map


def run_searchlight_gpu(
    fmri: np.ndarray,
    model_emb: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    method: str = "spearman",
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """GPU-batched searchlight RSA.

    Processes vertices in batches on a CUDA device.  For vertices whose geodesic
    neighbourhood contains no medial-wall exclusions (the vast majority), computation
    is fully GPU-batched.  Vertices near the medial wall (with at least one invalid
    neighbour) fall back to the optimised CPU implementation.

    Parameters
    ----------
    fmri            : (n_bins, n_hem_verts) float32
    model_emb       : (n_bins, n_features) float32
    neighbors       : (n_surf_verts, k) int32
    surface_indices : (n_hem_verts,) int32
    vertex_to_col   : (n_surf_verts,) int32
    method          : "spearman" | "pearson"
    batch_size      : int — vertices per GPU batch (default 512; reduce if OOM)
    device          : str — torch device (default "cuda")

    Returns
    -------
    corr_map : (n_hem_verts,) float32
    """
    import torch

    n_verts = fmri.shape[1]
    n_bins  = fmri.shape[0]
    k       = neighbors.shape[1]
    tril_idx = np.tril_indices(n_bins, k=-1)
    n_pairs  = len(tril_idx[0])

    log.info(f"  [GPU] Precomputing model RDM ({method}) ...")
    _, model_norm = _precompute_model_rdm(model_emb, n_bins, tril_idx, method)
    model_norm_t  = torch.from_numpy(model_norm).to(device)            # (n_pairs,)
    # fmri on GPU, transposed for fast column gather: (n_hem_verts, n_bins)
    fmri_t = torch.from_numpy(fmri.T.astype(np.float32)).to(device)   # (n_hem_verts, n_bins)

    # Pre-compute neighbour fMRI column indices for every surface vertex
    # neighbor_cols_all[sv, :] = fMRI column indices for sv's k neighbours (-1=medial)
    neighbor_cols_all = vertex_to_col[neighbors]  # (n_surf_verts, k)

    # For every grayordinate vertex, look up its surface vertex → its neighbour columns
    surf_verts_for_v  = surface_indices.astype(np.int32)            # (n_verts,)
    ncols_for_v       = neighbor_cols_all[surf_verts_for_v]         # (n_verts, k)

    # Split into full-k (all neighbours valid) and partial-k (some medial-wall)
    full_k_mask    = np.all(ncols_for_v >= 0, axis=1)
    full_k_verts   = np.where(full_k_mask)[0]   # grayordinate indices
    partial_k_verts = np.where(~full_k_mask)[0]

    log.info(
        f"  [GPU] {len(full_k_verts):,} full-k vertices (GPU batch={batch_size}), "
        f"{len(partial_k_verts):,} partial-k vertices (CPU fallback)"
    )

    corr_map = np.zeros(n_verts, dtype=np.float32)
    tril_row  = torch.tensor(tril_idx[0], dtype=torch.long, device=device)
    tril_col  = torch.tensor(tril_idx[1], dtype=torch.long, device=device)

    # ── GPU batched pass (full-k vertices) ─────────────────────────────────
    for start in range(0, len(full_k_verts), batch_size):
        batch_v  = full_k_verts[start : start + batch_size]            # grayordinate indices
        batch_nc = ncols_for_v[batch_v].astype(np.int64)               # (B, k), all >= 0
        B = len(batch_v)

        batch_nc_t = torch.from_numpy(batch_nc).to(device)             # (B, k)

        # Gather neighbourhood fMRI: (B, k, n_bins) then → (B, n_bins, k)
        # fmri_t: (n_hem_verts, n_bins), batch_nc_t: (B, k)
        hood = fmri_t[batch_nc_t]                                       # (B, k, n_bins)
        hood = hood.permute(0, 2, 1).float()                            # (B, n_bins, k)

        # Row-normalise each (n_bins, k) slice for cosine-based RDM
        mu    = hood.mean(dim=2, keepdim=True)
        hc    = hood - mu
        norms = torch.linalg.norm(hc, dim=2, keepdim=True).clamp(min=1e-10)
        hn    = hc / norms                                              # (B, n_bins, k)

        # RDM via batched matmul: (B, n_bins, n_bins)
        rdm_full  = torch.bmm(hn, hn.permute(0, 2, 1))
        fmri_flat = (1.0 - rdm_full)[:, tril_row, tril_col]            # (B, n_pairs)

        if method == "spearman":
            # Rank the fMRI distances (argsort of argsort = rank)
            order = torch.argsort(fmri_flat, dim=1)
            ranks = torch.argsort(order, dim=1).float()
            fc    = ranks - ranks.mean(dim=1, keepdim=True)
            fn    = torch.linalg.norm(fc, dim=1, keepdim=True).clamp(min=1e-10)
            rho   = (fc / fn * model_norm_t).sum(dim=1)                # (B,)
        else:
            fc    = fmri_flat - fmri_flat.mean(dim=1, keepdim=True)
            fn    = torch.linalg.norm(fc, dim=1, keepdim=True).clamp(min=1e-10)
            rho   = (fc / fn * model_norm_t).sum(dim=1)                # (B,)

        corr_map[batch_v] = rho.cpu().numpy().astype(np.float32)

    # ── CPU fallback (partial-k vertices near medial wall) ──────────────────
    if len(partial_k_verts) > 0:
        log.info(f"  [CPU fallback] {len(partial_k_verts):,} partial-k vertices ...")
        # Bring fmri back to CPU for the per-vertex function
        fmri_cpu = fmri_t.cpu().numpy().T   # (n_bins, n_hem_verts)
        for v in partial_k_verts:
            sv = int(surf_verts_for_v[v])
            corr_map[v] = _searchlight_vertex_fast(
                sv, fmri_cpu, model_norm, neighbors, vertex_to_col, tril_idx, method
            )

    return corr_map


# =============================================================================
# Output naming
# =============================================================================

def _cifti_path(args) -> str:
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")

def _run_trs_path(args) -> str:
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_run_trs.npy")


def _config_label(args) -> str:
    parts = [
        f"k{args.k}",
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        f"bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s",
        args.method,
    ]
    return "_".join(parts)


def _streaming_fmri_tag(args) -> str:
    parts = []
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    if args.gsr:       parts.append("gsr")
    return "_".join(parts) if parts else "raw"


# =============================================================================
# Core analysis
# =============================================================================

def _run_analysis(args, fmri_continuous: np.ndarray, run_trs: np.ndarray,
                  timing_df: pd.DataFrame, config: str, out_root: Path, fmri_tag: str):

    bin_sec_int   = int(args.bin_sec)
    skip_int      = int(args.skip_sec)
    delay_tag     = f"delay{int(args.delay_sec)}s"
    maps_out      = out_root / f"rsa_59k_{fmri_tag}_k{args.k}_{delay_tag}_bin{bin_sec_int}s_skip{skip_int}s_{args.method}_searchlight.npy"
    map_name      = f"searchlight_{args.method}_rho"
    combined_path = Path(args.combined_output) if args.combined_output else None

    # ── Skip / fast-merge logic ───────────────────────────────────────────────
    if maps_out.exists():
        if combined_path is None:
            log.info(f"Output already exists — skipping: {maps_out.name}")
            return
        if map_name in get_combined_map_names(combined_path):
            log.info(f"Output already exists and combined up to date — skipping: {maps_out.name}")
            return
        # .npy exists but combined missing this map → merge without recomputing
        log.info(f"  .npy exists; merging '{map_name}' into combined ...")
        corr_full = np.load(str(maps_out)).astype(np.float32)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        merge_into_combined(corr_full, map_name, combined_path, args.template_cifti)
        return

    assert_segment_timing(timing_df, args.bin_sec, args.tr,
                          args.delay_sec, args.skip_sec, run_trs)

    fmri_binned = preprocess_fmri(
        fmri_continuous, timing_df, run_trs, args.bin_sec, args.tr,
        args.delay_sec, skip_sec=args.skip_sec,
    )
    log.info(f"  fMRI binned & z-scored: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"bin{bin_sec_int}s_skip{skip_int}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")

    emb = process_model_embeddings(
        str(emb_file), timing_df, bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, hrf=args.hrf,
        skip_sec=args.skip_sec,
    )
    log.info(f"  Model binned & z-scored: {emb.shape}")

    # Enforce exact temporal alignment
    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)
    n_bins = fmri_binned.shape[0]
    log.info(f"  n_bins={n_bins}  n_pairs={(n_bins*(n_bins-1)//2):,}")

    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    # Shared geodesic cache: use explicit override if provided, else fall back to
    # the preprocessing-specific subdir.  All callers in run_analysis.sh pass
    # --geodesic-cache-dir so the same .npy files are shared across preprocessing
    # variants and the k_max derivation (Level 2) works correctly.
    if getattr(args, "geodesic_cache_dir", None):
        cache_dir = Path(args.geodesic_cache_dir)
    else:
        cache_dir = Path(args.output_dir) / "_geodesic_cache"

    surfaces = {
        "left":  (args.left_surface,  fmri_binned[:, :n_left],  left_indices),
        "right": (args.right_surface, fmri_binned[:, n_left:],   right_indices),
    }

    n_total   = fmri_binned.shape[1]
    corr_full = np.zeros(n_total, dtype=np.float32)
    offset    = 0

    for hem, (surf_path, fmri_hem, surf_indices) in surfaces.items():
        neighbors = get_neighbors(
            surf_path, args.workbench, args.subject, hem, args.k, cache_dir,
        )

        surf_indices = surf_indices.astype(np.int32)
        n_surf_verts = neighbors.shape[0]
        vertex_to_col = np.full(n_surf_verts, -1, dtype=np.int32)
        vertex_to_col[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)

        corr_hem = run_searchlight(
            fmri_hem, emb, neighbors,
            surface_indices=surf_indices,
            vertex_to_col=vertex_to_col,
            method=args.method, n_jobs=N_JOBS,
            batch_size=args.gpu_batch_size,
        )

        n_hem = corr_hem.shape[0]
        corr_full[offset: offset + n_hem] = corr_hem
        offset += n_hem

        del neighbors, vertex_to_col, corr_hem
        gc.collect()

    del fmri_binned, emb
    gc.collect()

    out_root.mkdir(parents=True, exist_ok=True)
    np.save(str(maps_out), corr_full)
    log.info(f"  Saved: {maps_out.name}  mean_r={corr_full.mean():.4f}  max_r={corr_full.max():.4f}")

    # ── Merge into combined output ────────────────────────────────────────────
    if combined_path is not None:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        merge_into_combined(corr_full, map_name, combined_path, args.template_cifti)

    # ── Per-subject JSON report (flat directory, keyed by subject ID) ─────────
    reports_dir = Path(args.output_dir) / "subject_data" / "subject_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{args.subject}.json"

    # Load existing report if present (subject may have run multiple models)
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    result_key = f"{args.model}_{args.modality}_{config}"
    existing[result_key] = {
        "subject":    args.subject,
        "model":      args.model,
        "modality":   args.modality,
        "config":     config,
        "fmri_tag":   fmri_tag,
        "n_bins":     n_bins,
        "n_vertices": n_total,
        "max_r":      float(corr_full.max()),
        "mean_r":     float(corr_full.mean()),
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    }
    report_path.write_text(json.dumps(existing, indent=2))
    log.info(f"  Report: {report_path.name}")


# =============================================================================
# Disk mode
# =============================================================================

def _run_disk(args):
    timing_df = pd.read_csv(args.timing_csv)
    config    = _config_label(args)
    fmri_tag  = args.fmri_suffix
    sub_dir   = Path(args.output_dir) / args.subject if args.subject == "group_average" else Path(args.output_dir) / "subject_data" / args.subject
    out_root  = sub_dir / f"{args.model}_{args.modality}" / config

    cifti = _cifti_path(args)
    trs_path = _run_trs_path(args)

    log.info(f"Searchlight RSA: {args.model}/{args.modality}/{config}")
    log.info(f"  fMRI: {cifti}")
    
    fmri_data = load_fmri_cifti(cifti)
    run_trs = np.load(trs_path)
    
    log.info(f"  fMRI loaded: {fmri_data.shape}  run_trs: {run_trs.tolist()}")

    _run_analysis(args, fmri_data, run_trs, timing_df, config, out_root, fmri_tag)
    log.info("Done.")


# =============================================================================
# Streaming mode
# =============================================================================

def _run_streaming(args):
    from preprocess_individual import preprocess_subject

    timing_df   = pd.read_csv(args.timing_csv)
    config      = _config_label(args)
    fmri_tag    = _streaming_fmri_tag(args)
    sub         = args.subject

    # Delay tag for output naming consistency
    delay_tag   = f"delay{int(args.delay_sec)}s"
    out_root    = Path(args.output_dir) / "subject_data" / sub / f"{args.model}_{args.modality}" / config
    bin_sec_int = int(args.bin_sec)
    skip_int    = int(args.skip_sec)

    maps_out    = out_root / f"rsa_59k_{fmri_tag}_k{args.k}_{delay_tag}_bin{bin_sec_int}s_skip{skip_int}s_{args.method}_searchlight.npy"

    if maps_out.exists():
        log.info(f"[{sub}] Output already exists — skipping")
        return

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr
    )

    log.info(f"[{sub}] Preprocessing raw CIFTI (continuous mode) ...")
    data, _bm_axis, run_trs = preprocess_subject(
        sub, Path(args.raw_dir), args.tr, prep_args
    )
    log.info(f"[{sub}] Preprocessed continuous: {data.shape}  run_trs: {run_trs.tolist()}")

    _run_analysis(args, data, run_trs, timing_df, config, out_root, fmri_tag)
    del data
    gc.collect()
    log.info(f"[{sub}] Done.")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    if args.preprocessed_dir and args.raw_dir:
        log.error("--preprocessed-dir and --raw-dir are mutually exclusive.")
        sys.exit(1)
    if not args.preprocessed_dir and not args.raw_dir:
        log.error("Either --preprocessed-dir or --raw-dir is required.")
        sys.exit(1)

    if args.raw_dir:
        _run_streaming(args)
    else:
        _run_disk(args)


if __name__ == "__main__":
    main()