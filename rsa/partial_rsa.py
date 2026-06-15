"""
rsa/partial_rsa.py
=======================
Vertex-wise Partial RSA via Banded Ridge Regression.

For each searchlight neighbourhood, computes the Pearson correlation between:
  (a) the brain RDM vector after projecting out nuisance model RDMs, and
  (b) the target model RDM vector after the same nuisance projection.

This isolates the unique variance explained by the target joint embedding
beyond what is already captured by the nuisance unimodal embeddings.

Methodology
-----------
1.  Bin and z-score fMRI and all model embeddings (per run, matching the
    standard searchlight pipeline).
2.  Compute model RDMs from lower-triangle pairwise correlation distances
    and z-score each RDM vector to zero mean / unit variance.
3.  Find per-band regularization alphas via himalaya RidgeCV (5-fold CV,
    separate alpha per nuisance model — banded ridge).
4.  Build the banded nuisance projection operator C = (X^T X + diag(alphas))^{-1} X^T
    where X = [nuis_1_rdm | nuis_2_rdm], shape (n_pairs, n_nuis).
5.  Pre-compute the target model residual once:
      e_target = y_target - X @ (C @ y_target)
6.  GPU-batched searchlight: for each batch of vertices, compute brain RDM,
    z-score, residualize via the same C, then Pearson-correlate with e_target.
7.  Save as CIFTI .dscalar.nii.

Runs defined in rsa/shared/model_registry.py::PARTIAL_RSA_RUNS.
Default run: run_A (PE-AV joint controlling for AudioMAE + VideoMAE).
             run_B (PE-AV joint controlling for its own unimodal decoders).

Usage
-----
python rsa/partial_rsa.py \
    --run run_A \
    --preprocessed-dir /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw \
    --fmri-suffix raw \
    --timing-csv /home/amin/Research/Representation/Movie/data/movie_timing.csv \
    --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \
    --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \
    --output-dir /home/amin/Research/Representation/Movie/outputs/rsa/partial \
    --subject group_average \
    --bin-sec 5.0 --delay-sec 5.0 --tr 1.0 \
    --k 100 --method spearman \
    --left-surface /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii \
    --right-surface /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii \
    --workbench /opt/workbench/bin_linux64/wb_command \
    --geodesic-cache-dir /home/amin/Research/Representation/Movie/outputs/rsa/_geodesic_cache
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap
from rsa.shared.rsa_utils import (
    load_fmri_cifti,
    preprocess_fmri,
    process_model_embeddings,
    align_and_assert_bins,
    compute_rdm,
)
from rsa.shared.model_registry import (
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    TR_DEFAULT,
    PARTIAL_RSA_RUNS,
    check_embeddings_exist,
    validate_run,
)

# Re-use the neighbour-loading machinery from the main searchlight script.
# Importing it here keeps our code DRY.
sys.path.insert(0, str(Path(__file__).parent))
from searchlight import get_neighbors, _searchlight_vertex_fast  # noqa: E402

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
        description="Vertex-wise partial RSA via banded ridge regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", required=True,
                   help=f"Partial RSA run key. Available: {list(PARTIAL_RSA_RUNS.keys())}")
    p.add_argument("--preprocessed-dir",  required=True, dest="preprocessed_dir")
    p.add_argument("--fmri-suffix",       default="raw",  dest="fmri_suffix")
    p.add_argument("--timing-csv",        required=True)
    p.add_argument("--embeddings-dir",    required=True)
    p.add_argument("--template-cifti",    required=True)
    p.add_argument("--output-dir",        required=True)
    p.add_argument("--subject",           default="group_average")
    p.add_argument("--bin-sec",           type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--delay-sec",         type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr",                type=float, default=TR_DEFAULT)
    p.add_argument("--k",                 type=int,   required=True)
    p.add_argument("--method",            default="spearman",
                   choices=["spearman", "pearson"],
                   help="Distance method for brain RDM construction.")
    p.add_argument("--left-surface",      required=True)
    p.add_argument("--right-surface",     required=True)
    p.add_argument("--workbench",         required=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir")
    p.add_argument("--gpu-batch-size",    type=int, default=512, dest="gpu_batch_size")
    p.add_argument("--n-alphas",          type=int, default=30, dest="n_alphas",
                   help="Number of alpha candidates for per-band RidgeCV.")
    p.add_argument("--cv-folds",          type=int, default=5, dest="cv_folds",
                   help="K-fold CV splits for per-band alpha selection.")
    return p.parse_args()


# =============================================================================
# Nuisance projection (banded ridge, per-band alpha)
# =============================================================================

def _zscore_vec(v: np.ndarray) -> np.ndarray:
    """Z-score a 1-D vector; return zeros if std == 0."""
    mu, sd = v.mean(), v.std()
    return (v - mu) / sd if sd > 1e-10 else np.zeros_like(v)


def _rdm_lower_tri(emb: np.ndarray) -> np.ndarray:
    """Compute correlation-distance RDM and return z-scored lower triangle.

    Parameters
    ----------
    emb : (n_bins, n_features) float32

    Returns
    -------
    (n_pairs,) float32 — z-scored lower-triangle RDM vector
    """
    rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    n = rdm.shape[0]
    idx = np.tril_indices(n, k=-1)
    vec = rdm[idx].astype(np.float32)
    return _zscore_vec(vec)


def fit_banded_projection(
    X_nuis: np.ndarray,
    y_target: np.ndarray,
    n_alphas: int = 30,
    cv_folds: int = 5,
) -> tuple[np.ndarray, list[float]]:
    """Find per-band alphas via himalaya RidgeCV, then build projection matrix C.

    Each column of X_nuis is one nuisance band (1 feature per band).  A separate
    RidgeCV is run for each band to find its optimal regularisation independently
    (banded ridge principle: separate shrinkage per modality).

    Parameters
    ----------
    X_nuis   : (n_pairs, n_nuis) float32 — stacked z-scored nuisance RDM vectors
    y_target : (n_pairs,) float32 — z-scored target model RDM vector
    n_alphas : int — grid size for log-spaced alpha search
    cv_folds : int — number of CV folds

    Returns
    -------
    C      : (n_nuis, n_pairs) float32
        Projection operator: y_hat = X_nuis @ (C @ y), residual = y - y_hat.
    alphas : list[float] — best alpha per nuisance band
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    alpha_grid = np.logspace(0, 20, n_alphas).astype(np.float64)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    n_nuis = X_nuis.shape[1]
    best_alphas = []

    for i in range(n_nuis):
        x_i = X_nuis[:, i : i + 1].astype(np.float64)
        ridge = RidgeCV(alphas=alpha_grid, cv=cv, fit_intercept=False)
        ridge.fit(x_i, y_target.astype(np.float64))
        best_alphas.append(float(ridge.alpha_))
        log.info(f"  Band {i}: best alpha = {best_alphas[-1]:.2e}")

    # Build the combined banded projection matrix.
    # A = X^T X + diag(alphas)  shape (n_nuis, n_nuis)
    # C = A^{-1} X^T            shape (n_nuis, n_pairs)
    # Residual of any y:  e = y - X @ (C @ y)
    A = X_nuis.T.astype(np.float64) @ X_nuis.astype(np.float64)
    A += np.diag(best_alphas)
    C = np.linalg.solve(A, X_nuis.T.astype(np.float64))  # (n_nuis, n_pairs)
    return C.astype(np.float32), best_alphas


def residualize(y: np.ndarray, X_nuis: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Project nuisance out of y.

    Parameters
    ----------
    y      : (n_pairs,) float32
    X_nuis : (n_pairs, n_nuis) float32
    C      : (n_nuis, n_pairs) float32

    Returns
    -------
    (n_pairs,) float32 — y minus its nuisance-predicted component
    """
    return y - X_nuis @ (C @ y)


# =============================================================================
# GPU-batched partial searchlight
# =============================================================================

def run_partial_searchlight_gpu(
    fmri: np.ndarray,
    X_nuis: np.ndarray,
    C: np.ndarray,
    e_target: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    method: str = "spearman",
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """GPU-batched partial RSA searchlight.

    For each batch of grayordinate vertices:
      1. Gather searchlight neighbourhood fMRI → compute correlation-distance RDM
      2. Z-score the lower-triangle brain RDM vector
      3. Residualize: project out nuisance via C (same as for e_target)
      4. Pearson r between residual brain vector and e_target

    Parameters
    ----------
    fmri            : (n_bins, n_hem_verts) float32
    X_nuis          : (n_pairs, n_nuis) float32 — nuisance RDM matrix
    C               : (n_nuis, n_pairs) float32 — banded projection operator
    e_target        : (n_pairs,) float32 — residualized target RDM vector
    neighbors       : (n_surf_verts, k) int32
    surface_indices : (n_hem_verts,) int32
    vertex_to_col   : (n_surf_verts,) int32
    method          : RDM construction ("spearman" or "pearson")
    batch_size      : vertices per GPU batch
    device          : CUDA device string

    Returns
    -------
    corr_map : (n_hem_verts,) float32 — partial RSA correlation per vertex
    """
    import torch

    n_verts = fmri.shape[1]
    n_bins  = fmri.shape[0]
    k       = neighbors.shape[1]
    tril_idx  = np.tril_indices(n_bins, k=-1)
    n_pairs   = len(tril_idx[0])

    # Transfer fixed tensors to GPU
    X_nuis_t  = torch.from_numpy(X_nuis).to(device)    # (n_pairs, n_nuis)
    C_t       = torch.from_numpy(C).to(device)          # (n_nuis, n_pairs)
    e_tgt_t   = torch.from_numpy(e_target).to(device)   # (n_pairs,)

    # Pre-normalize e_target once for fast Pearson (dot product with unit vector)
    e_tgt_c   = e_tgt_t - e_tgt_t.mean()
    e_tgt_n   = e_tgt_c / e_tgt_c.norm().clamp(min=1e-10)

    fmri_t = torch.from_numpy(fmri.T.astype(np.float32)).to(device)  # (n_hem_verts, n_bins)

    neighbor_cols_all = vertex_to_col[neighbors]                       # (n_surf_verts, k)
    surf_verts_for_v  = surface_indices.astype(np.int32)               # (n_verts,)
    ncols_for_v       = neighbor_cols_all[surf_verts_for_v]            # (n_verts, k)

    full_k_mask    = np.all(ncols_for_v >= 0, axis=1)
    full_k_verts   = np.where(full_k_mask)[0]
    partial_k_verts = np.where(~full_k_mask)[0]

    log.info(
        f"  [GPU partial RSA] {len(full_k_verts):,} full-k vertices "
        f"(batch={batch_size}), {len(partial_k_verts):,} partial-k (CPU fallback)"
    )

    corr_map = np.zeros(n_verts, dtype=np.float32)
    tril_row = torch.tensor(tril_idx[0], dtype=torch.long, device=device)
    tril_col = torch.tensor(tril_idx[1], dtype=torch.long, device=device)

    # ── GPU batch loop ────────────────────────────────────────────────────────
    for start in range(0, len(full_k_verts), batch_size):
        batch_v  = full_k_verts[start : start + batch_size]
        batch_nc = ncols_for_v[batch_v].astype(np.int64)               # (B, k)
        B = len(batch_v)

        batch_nc_t = torch.from_numpy(batch_nc).to(device)
        hood       = fmri_t[batch_nc_t]                                 # (B, k, n_bins)
        hood       = hood.permute(0, 2, 1).float()                      # (B, n_bins, k)

        # Pearson correlation distance RDM (same as existing searchlight)
        mu    = hood.mean(dim=2, keepdim=True)
        hc    = hood - mu
        norms = torch.linalg.norm(hc, dim=2, keepdim=True).clamp(min=1e-10)
        hn    = hc / norms                                               # (B, n_bins, k)
        rdm_full   = torch.bmm(hn, hn.permute(0, 2, 1))                # (B, n_bins, n_bins)
        brain_flat = (1.0 - rdm_full)[:, tril_row, tril_col]            # (B, n_pairs)

        if method == "spearman":
            # Rank-order within each vertex's RDM vector
            order  = torch.argsort(brain_flat, dim=1)
            ranks  = torch.argsort(order, dim=1).float()
            brain_flat = ranks

        # Z-score each brain RDM vector  (B, n_pairs)
        mu_b  = brain_flat.mean(dim=1, keepdim=True)
        sd_b  = brain_flat.std(dim=1, keepdim=True).clamp(min=1e-10)
        Y_b   = (brain_flat - mu_b) / sd_b                              # (B, n_pairs)

        # Residualize: E = Y - X_nuis @ (C @ Y^T)  →  (B, n_pairs)
        # C_t @ Y_b.T : (n_nuis, n_pairs) × (n_pairs, B) = (n_nuis, B)
        coef_b = C_t @ Y_b.T                                            # (n_nuis, B)
        E_b    = Y_b - (X_nuis_t @ coef_b).T                           # (B, n_pairs)

        # Pearson r with e_target for each vertex in batch
        # Normalize E_b rows, then dot with pre-normalized e_tgt_n
        E_c    = E_b - E_b.mean(dim=1, keepdim=True)
        E_n    = E_c / E_c.norm(dim=1, keepdim=True).clamp(min=1e-10)  # (B, n_pairs)
        rho    = (E_n * e_tgt_n.unsqueeze(0)).sum(dim=1)               # (B,)

        corr_map[batch_v] = rho.cpu().numpy().astype(np.float32)

    # ── CPU fallback for partial-k vertices ───────────────────────────────────
    if len(partial_k_verts) > 0:
        log.info(f"  [CPU fallback] {len(partial_k_verts):,} partial-k vertices ...")
        fmri_cpu    = fmri.copy()   # (n_bins, n_hem_verts) — already on CPU
        X_nuis_cpu  = X_nuis
        C_cpu       = C
        e_target_cpu = e_target

        tril_idx_cpu = np.tril_indices(n_bins, k=-1)

        for v in partial_k_verts:
            sv = int(surface_indices[v])
            neighbor_surf = neighbors[sv]
            neighbor_cols = vertex_to_col[neighbor_surf]
            neighbor_cols = neighbor_cols[neighbor_cols >= 0]
            if len(neighbor_cols) < 2:
                continue

            hood_cpu = fmri_cpu[:, neighbor_cols].astype(np.float64)
            mu_c  = hood_cpu.mean(axis=1, keepdims=True)
            hcc   = hood_cpu - mu_c
            nc    = np.sqrt((hcc ** 2).sum(axis=1, keepdims=True))
            nc[nc < 1e-10] = 1.0
            hn_c  = hcc / nc
            flat  = (1.0 - hn_c @ hn_c.T)[tril_idx_cpu].astype(np.float32)

            if method == "spearman":
                order_c = np.argsort(flat)
                r_c = np.empty(len(flat), dtype=np.float32)
                r_c[order_c] = np.arange(len(flat), dtype=np.float32)
                flat = r_c

            flat_z = _zscore_vec(flat)
            e_brain_v = residualize(flat_z, X_nuis_cpu, C_cpu)

            # Pearson correlation with e_target
            eb_c = e_brain_v - e_brain_v.mean()
            et_c = e_target_cpu - e_target_cpu.mean()
            denom = (np.linalg.norm(eb_c) * np.linalg.norm(et_c))
            if denom > 1e-10:
                corr_map[v] = float(np.dot(eb_c, et_c) / denom)

    return corr_map


def run_partial_searchlight(
    fmri: np.ndarray,
    X_nuis: np.ndarray,
    C: np.ndarray,
    e_target: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    method: str = "spearman",
    batch_size: int = 512,
) -> np.ndarray:
    """Dispatch GPU (with OOM fallback) or CPU partial searchlight."""
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            try:
                return run_partial_searchlight_gpu(
                    fmri, X_nuis, C, e_target, neighbors,
                    surface_indices, vertex_to_col,
                    method=method, batch_size=batch_size, device="cuda",
                )
            except _torch.cuda.OutOfMemoryError:
                log.warning("  GPU OOM — falling back to CPU partial RSA searchlight")
                _torch.cuda.empty_cache()
        else:
            log.info("  CUDA not available — using CPU partial RSA searchlight")
    except ImportError:
        log.info("  torch not installed — using CPU partial RSA searchlight")

    # CPU path (joblib-parallel over vertices)
    from joblib import Parallel, delayed

    n_verts  = fmri.shape[1]
    n_bins   = fmri.shape[0]
    tril_idx = np.tril_indices(n_bins, k=-1)
    fmri_cpu = fmri.copy()

    def _partial_vertex(v: int) -> float:
        sv = int(surface_indices[v])
        nc = vertex_to_col[neighbors[sv]]
        nc = nc[nc >= 0]
        if len(nc) < 2:
            return 0.0
        hood = fmri_cpu[:, nc].astype(np.float64)
        mu_h = hood.mean(axis=1, keepdims=True)
        hc   = hood - mu_h
        nm   = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
        nm[nm < 1e-10] = 1.0
        flat = (1.0 - (hc / nm) @ (hc / nm).T)[tril_idx].astype(np.float32)
        if method == "spearman":
            o = np.argsort(flat)
            r = np.empty_like(flat)
            r[o] = np.arange(len(flat), dtype=np.float32)
            flat = r
        flat_z    = _zscore_vec(flat)
        e_brain_v = residualize(flat_z, X_nuis, C)
        eb_c = e_brain_v - e_brain_v.mean()
        et_c = e_target - e_target.mean()
        denom = np.linalg.norm(eb_c) * np.linalg.norm(et_c)
        return float(np.dot(eb_c, et_c) / denom) if denom > 1e-10 else 0.0

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_partial_vertex)(v) for v in range(n_verts)
    )
    return np.array(results, dtype=np.float32)


def import_or_default(dotted: str, fallback):
    """Safely import a dotted name; return fallback on ImportError."""
    try:
        import importlib
        parts = dotted.rsplit(".", 1)
        mod   = importlib.import_module(parts[0])
        return getattr(mod, parts[1])
    except Exception:
        return fallback


# =============================================================================
# Main analysis
# =============================================================================

def run_analysis(args):
    cfg = validate_run(args.run)
    log.info("=" * 70)
    log.info(f"Partial RSA — {args.run}: {cfg.description}")
    log.info(f"  Target   : {cfg.target}")
    log.info(f"  Nuisance : {cfg.nuisance}")
    log.info(f"  Subject  : {args.subject}  method={args.method}  k={args.k}")
    log.info(f"  bin_sec={args.bin_sec}  delay_sec={args.delay_sec}  tr={args.tr}")
    log.info("=" * 70)

    # ── fMRI ────────────────────────────────────────────────────────────────
    cifti_path = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")
    trs_path   = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    fmri_raw  = load_fmri_cifti(str(cifti_path))
    run_trs   = np.load(str(trs_path))
    timing_df = pd.read_csv(args.timing_csv)
    log.info(f"  fMRI loaded: {fmri_raw.shape}")

    fmri_binned = preprocess_fmri(
        fmri_raw, timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec
    )
    del fmri_raw
    gc.collect()
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    n_bins  = fmri_binned.shape[0]
    n_pairs = n_bins * (n_bins - 1) // 2
    log.info(f"  n_bins={n_bins}  n_pairs={n_pairs:,}")

    # ── Load and bin all model embeddings ────────────────────────────────────
    def load_emb(model, modality):
        path = check_embeddings_exist(args.embeddings_dir, model, modality, args.bin_sec)
        log.info(f"  Loading: {path.name}")
        emb = process_model_embeddings(
            str(path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec,
        )
        _, emb = align_and_assert_bins(fmri_binned, emb)
        return emb

    target_model, target_mod = cfg.target
    target_emb = load_emb(target_model, target_mod)

    nuis_embs = []
    for nuis_model, nuis_mod in cfg.nuisance:
        nuis_embs.append(load_emb(nuis_model, nuis_mod))

    # ── Compute z-scored lower-triangle RDM vectors ──────────────────────────
    log.info("  Computing model RDMs ...")
    y_target = _rdm_lower_tri(target_emb)
    nuis_vecs = [_rdm_lower_tri(e) for e in nuis_embs]
    X_nuis    = np.column_stack(nuis_vecs).astype(np.float32)  # (n_pairs, n_nuis)

    log.info(f"  RDM shapes: y_target={y_target.shape}  X_nuis={X_nuis.shape}")
    del target_emb, nuis_embs
    gc.collect()

    # ── Banded ridge nuisance projection ─────────────────────────────────────
    log.info("  Fitting banded ridge (per-band alpha via RidgeCV) ...")
    C, best_alphas = fit_banded_projection(
        X_nuis, y_target, n_alphas=args.n_alphas, cv_folds=args.cv_folds
    )
    log.info(f"  Per-band alphas: {[f'{a:.2e}' for a in best_alphas]}")

    e_target = residualize(y_target, X_nuis, C)
    log.info(
        f"  e_target: mean={e_target.mean():.4f}  "
        f"std={e_target.std():.4f}  "
        f"frac_nz={(e_target != 0).mean():.2f}"
    )

    # ── Grayordinate structure ───────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left  = len(left_indices)
    n_total = fmri_binned.shape[1]

    cache_dir = (Path(args.geodesic_cache_dir) if args.geodesic_cache_dir
                 else Path(args.output_dir) / "_geodesic_cache")

    surfaces = {
        "left":  (args.left_surface,  fmri_binned[:, :n_left],  left_indices),
        "right": (args.right_surface, fmri_binned[:, n_left:],   right_indices),
    }

    corr_full = np.zeros(n_total, dtype=np.float32)
    offset    = 0

    for hem, (surf_path, fmri_hem, surf_indices) in surfaces.items():
        log.info(f"  Hemisphere: {hem}  ({fmri_hem.shape[1]} vertices)")
        neighbors = get_neighbors(
            surf_path, args.workbench, args.subject, hem, args.k, cache_dir,
        )
        surf_indices = surf_indices.astype(np.int32)
        n_surf       = neighbors.shape[0]
        v2c          = np.full(n_surf, -1, dtype=np.int32)
        v2c[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)

        corr_hem = run_partial_searchlight(
            fmri_hem, X_nuis, C, e_target,
            neighbors, surf_indices, v2c,
            method=args.method, batch_size=args.gpu_batch_size,
        )

        n_hem = corr_hem.shape[0]
        corr_full[offset : offset + n_hem] = corr_hem
        offset += n_hem

        del neighbors, v2c, corr_hem
        gc.collect()

    del fmri_binned
    gc.collect()

    # ── Save CIFTI ───────────────────────────────────────────────────────────
    bin_sec_int = int(args.bin_sec)
    delay_int   = int(args.delay_sec)
    map_name    = f"partial_rsa_{cfg.label}"
    out_dir     = (Path(args.output_dir) / args.subject / "partial_rsa" /
                   f"k{args.k}_delay{delay_int}s_bin{bin_sec_int}s_{args.method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.label}_k{args.k}_delay{delay_int}s_bin{bin_sec_int}s.dscalar.nii"

    save_cifti_multimap(
        corr_full.reshape(1, -1),
        [map_name],
        args.template_cifti,
        str(out_path),
    )
    log.info(
        f"Saved: {out_path.name}  "
        f"mean_r={corr_full.mean():.4f}  max_r={corr_full.max():.4f}  "
        f"frac>0={float(np.mean(corr_full > 0)):.2f}"
    )


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
