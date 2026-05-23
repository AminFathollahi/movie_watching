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
      --preprocessed-dir <path> --fmri-suffix sg_psc_gsr \
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
    process_model_embeddings, align_and_assert_bins
)
from rsa.shared.cifti_io import (
    get_bm_axis, save_cifti_multimap, get_cortex_vertex_indices,
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
    inp.add_argument("--fmri-suffix", default="sg_psc_gsr", dest="fmri_suffix",
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
    p.add_argument("--modality", required=True, choices=["v", "a", "av"])
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Applied when slicing stimulus blocks from continuous fMRI.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF.")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"])
    p.add_argument("--tr", type=float, required=True)

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true")
    prep.add_argument("--psc", default=False, action="store_true")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction)

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


def _dconn_is_complete(dconn_path: Path, min_size_gb: float = 1.0) -> bool:
    """Return True if a dconn.nii file looks fully written (>= min_size_gb)."""
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


def get_neighbors(surface_path: str, workbench: str, subject: str,
                  hem: str, k: int, cache_dir: Path) -> np.ndarray:
    """Return (n_surf_verts, k) int32 k-NN array, using a two-level cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path   = cache_dir / f"{subject}_{hem}_neighbors_k{k}.npy"
    dconn_path = cache_dir / f"{subject}_{hem}_geodesic.dconn.nii"

    if npy_path.exists():
        log.info(f"  k-NN cache hit: {npy_path.name}")
        return np.load(str(npy_path))

    # Need to compute from (or load existing) dconn.
    # Guard against partially-written files left by a previous crashed run:
    # if the dconn exists but is too small to be complete, delete and recompute.
    if dconn_path.exists() and not _dconn_is_complete(dconn_path):
        log.warning(
            f"  dconn appears incomplete ({dconn_path.stat().st_size / 1e9:.2f} GB) "
            f"— deleting and recomputing: {dconn_path.name}"
        )
        dconn_path.unlink()

    _compute_geodesic_dconn(surface_path, workbench, dconn_path)
    neighbors = _extract_knn_from_dconn(dconn_path, k)
    np.save(str(npy_path), neighbors)
    log.info(f"  k-NN saved: {npy_path.name}")

    # Free the 14 GB dconn for individual subjects; keep for group_average
    if subject != "group_average" and dconn_path.exists():
        dconn_path.unlink()
        log.info(f"  Deleted dconn: {dconn_path.name}")

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
    """Optimised per-vertex RSA: float64 matmul RDM + Spearman via argsort.

    Replaces squareform(pdist) + spearmanr with:
      - float64 row-normalised matmul for the fMRI RDM (full precision, no artificial ties)
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

    hood = fmri[:, neighbor_cols].astype(np.float64)   # float64: full precision, no artificial ties

    # Row-normalise to get unit correlation vectors (fast RDM via matmul)
    mu = hood.mean(axis=1, keepdims=True)
    hc = hood - mu
    norms = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
    norms[norms < 1e-10] = 1.0
    hn = hc / norms
    fmri_flat = (1.0 - hn @ hn.T)[tril_idx].astype(np.float32)  # condensed RDM

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
                    n_jobs: int = -1) -> np.ndarray:
    """Searchlight RSA across all grayordinate vertices.

    Parameters
    ----------
    fmri           : (n_bins, n_hem_verts) float32 — hemisphere fMRI
    model_emb      : (n_bins, n_features) float32 — model embeddings (for RDM)
    neighbors      : (n_surf_verts, k) int32 — geodesic k-NN
    surface_indices: (n_hem_verts,) int32 — surface vertex index per grayordinate
    vertex_to_col  : (n_surf_verts,) int32 — surface vertex → fmri column (-1=medial)
    method         : "spearman" | "pearson"
    n_jobs         : joblib parallel workers (-1 = all CPUs)
    """
    n_verts = fmri.shape[1]
    n_bins = fmri.shape[0]
    tril_idx = np.tril_indices(n_bins, k=-1)

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
        f"bin{args.bin_sec:.0f}s",
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
    
    bin_sec_int = int(args.bin_sec)
    
    # The output map appends the delay config for clarity even though it wasn't preprocessed with it
    delay_tag = f"delay{int(args.delay_sec)}s"
    
    # CHANGE THIS LINE:
    maps_out = out_root / f"rsa_59k_{fmri_tag}_k{args.k}_{delay_tag}_bin{bin_sec_int}_{args.method}_maps.dscalar.nii"

    if maps_out.exists():
        log.info(f"Output already exists — skipping: {maps_out.name}")
        return

    fmri_binned = preprocess_fmri(
        fmri_continuous, timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec
    )
    log.info(f"  fMRI binned & z-scored: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"{bin_sec_int}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")

    emb = process_model_embeddings(
        str(emb_file), timing_df, bin_sec=args.bin_sec, hrf=args.hrf, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec
    )
    log.info(f"  Model binned & z-scored: {emb.shape}")

    # Enforce exact temporal alignment
    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)
    n_bins = fmri_binned.shape[0]
    log.info(f"  n_bins={n_bins}  n_pairs={(n_bins*(n_bins-1)//2):,}")

    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

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
        )

        n_hem = corr_hem.shape[0]
        corr_full[offset: offset + n_hem] = corr_hem
        offset += n_hem

        del neighbors, vertex_to_col, corr_hem
        gc.collect()

    del fmri_binned, emb
    gc.collect()

    out_root.mkdir(parents=True, exist_ok=True)
    save_cifti_multimap(
        corr_full.reshape(1, -1),
        [f"{args.method}_rho"],
        args.template_cifti,
        str(maps_out),
    )
    log.info(f"  Saved: {maps_out.name}  max_r={corr_full.max():.4f}")

    # ── Per-subject JSON report (flat directory, keyed by subject ID) ─────────
    reports_dir = Path(args.output_dir) / "subject_reports"
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
    out_root  = Path(args.output_dir) / args.subject / args.model / config

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
    out_root    = Path(args.output_dir) / sub / args.model / config
    bin_sec_int = int(args.bin_sec)
    
    # CHANGE THIS LINE TO MATCH:
    maps_out    = out_root / f"rsa_59k_{fmri_tag}_k{args.k}_{delay_tag}_bin{bin_sec_int}_{args.method}_maps.dscalar.nii"

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