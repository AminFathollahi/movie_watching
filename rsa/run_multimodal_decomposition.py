"""
rsa/run_multimodal_decomposition.py
=====================================
CKA-based Multimodal Interaction Decomposition.

Core Hypothesis
---------------
A truly multimodal embedding J should encode cross-modal interaction structure
— patterns of similarity that cannot be explained by any linear combination of
the unimodal embeddings A and V.  A mere concatenation [A | V] produces no new
representational geometry; genuine joint training can.

Mathematical Framework
-----------------------
1.  Define the **interaction residual** of J w.r.t. unimodal embeddings:
       R = J - X_{AV} (X_{AV}^T X_{AV})^{-1} X_{AV}^T J
    where X_{AV} = [A | V] (concatenated features after z-scoring per feature).
    R contains the component of J that cannot be linearly predicted from A or V.

2.  **Multimodality score** (global):
       MS = ||R||_F^2 / ||J||_F^2
    MS = 0  →  J is fully determined by A and V (simple concatenation).
    MS = 1  →  J encodes only cross-modal patterns absent from both A and V.

3.  **Centered Kernel Alignment (CKA)** (Kornblith et al. 2019) between
    brain responses and model representations, computed either:
      - Per Glasser parcel: K_brain from all vertices in that parcel.
      - Per searchlight vertex: K_brain from the k-NN geodesic neighbourhood
        (same k-NN cache as run_searchlight.py).

    For each parcel or searchlight:
       K_brain        — Gram matrix from binned fMRI at the local vertices
       K_joint        — Gram matrix of J (computed once, shared across vertices)
       K_unimodal     — Gram matrix of [A | V]   (concatenated)
       K_interaction  — Gram matrix of R (interaction residual)

4.  **Specificity Index (SI)** per parcel/vertex:
       SI = CKA(K_brain, K_interaction) / (CKA(K_brain, K_joint) + ε)
    SI ≈ 1  →  brain similarity in this region is driven purely by the
               cross-modal interaction component (true integration region).
    SI ≈ 0  →  brain similarity is explained by unimodal features alone.

Outputs (all saved to a single cka_decomp_{target}.dscalar.nii)
-------
  Maps (4 per method run):
    {method}_cka_joint_{target}
    {method}_cka_unimodal_{target}
    {method}_cka_interaction_{target}
    {method}_specificity_index_{target}

  Additional files:
    {output_dir}/parcel_cka_{target}.csv      — per-parcel table (glasser only)
    {output_dir}/interaction_score_{target}.json — global scores + summary

Model Recommendations for Future Runs
--------------------------------------
  pe-av-large-16-frame  — Tests whether the multimodal interaction residual
                          scales with model capacity.  Larger PE-AV should
                          capture richer cross-modal structure if the integration
                          is capacity-limited.

  cav-mae-sync          — Alternative AV architecture (contrastive vs. generative).
                          If K_interaction correlates with brain regardless of
                          architecture, the effect is in the INFORMATION content
                          of the training data, not the specific objective.

  imagebind             — Aligns audio, video, AND text in a shared space.
                          If the brain's "integration regions" (high SI) correlate
                          strongly with imagebind but NOT with AV-only models,
                          the integration is semantic/linguistic rather than
                          purely perceptual.

Usage
-----
  # Both Glasser parcels and searchlight (default):
  python rsa/run_multimodal_decomposition.py \\
      --embeddings-dir /path/to/model_embeddings \\
      --timing-csv     /path/to/movie_timing.csv \\
      --fmri-cifti     /path/to/group_average_raw_cortex_59k.dtseries.nii \\
      --run-trs        /path/to/group_average_raw_run_trs.npy \\
      --glasser-dlabel /path/to/Q1-Q6_RelatedParcellation210_...dlabel.nii \\
      --template-cifti /path/to/template.dscalar.nii \\
      --left-surface   /path/to/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii \\
      --right-surface  /path/to/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii \\
      --workbench      /opt/workbench/bin_linux64/wb_command \\
      --geodesic-cache-dir /path/to/rsa/_geodesic_cache \\
      --output-dir     /path/to/outputs/multimodal_decomp \\
      --target-model   pe-av-small-16-frame \\
      --method         both

  # Glasser parcels only (no surface/workbench/cache args needed):
  python rsa/run_multimodal_decomposition.py ... --method glasser

  # Searchlight only (no --glasser-dlabel needed):
  python rsa/run_multimodal_decomposition.py ... --method searchlight

References
----------
Kornblith S et al. (2019). Similarity of neural network representations
  revisited. ICML.
Alexander-Bloch A et al. (2018). On testing for spatial correspondence
  between maps of human brain structure and function. NeuroImage.
Huth AG et al. (2016). Natural speech reveals the semantic maps that tile
  human cerebral cortex. Nature.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import (
    get_bm_axis,
    get_cortex_vertex_indices,
    save_cifti_multimap,
)
from rsa.shared.rsa_utils import (
    load_fmri_cifti,
    preprocess_fmri,
    process_model_embeddings,
    align_and_assert_bins,
)
from rsa.shared.model_registry import (
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    TR_DEFAULT,
    check_embeddings_exist,
)
from rsa.run_searchlight import get_neighbors

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
        description="CKA-based multimodal interaction decomposition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embeddings-dir",   required=True)
    p.add_argument("--timing-csv",       required=True)
    p.add_argument("--fmri-cifti",       required=True, dest="fmri_cifti",
                   help="Preprocessed group-average dtseries CIFTI.")
    p.add_argument("--run-trs",          required=True, dest="run_trs",
                   help="run_trs.npy for the group-average CIFTI.")
    p.add_argument("--template-cifti",   required=True)
    p.add_argument("--output-dir",       required=True)

    p.add_argument("--method", default="both",
                   choices=["glasser", "searchlight", "both"],
                   help="Which analysis to run. 'both' runs Glasser parcels and "
                        "vertex-wise searchlight CKA.")

    # Glasser parcel analysis
    p.add_argument("--glasser-dlabel",   default=None, dest="glasser_dlabel",
                   help="Glasser 360-parcel dlabel CIFTI (59k). Required for "
                        "method=glasser or method=both.")

    # Searchlight analysis
    p.add_argument("--k",              type=int, default=100,
                   help="Geodesic k-NN size for searchlight CKA.")
    p.add_argument("--left-surface",   default=None, dest="left_surface",
                   help="Left hemisphere midthickness .surf.gii (for k-NN cache). "
                        "Required for method=searchlight or method=both.")
    p.add_argument("--right-surface",  default=None, dest="right_surface")
    p.add_argument("--workbench",      default="/opt/workbench/bin_linux64/wb_command",
                   help="Path to wb_command (for geodesic distance computation).")
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir",
                   help="Directory for k-NN .npy cache files (shared with "
                        "run_searchlight.py). Required for method=searchlight or both.")
    p.add_argument("--subject",        default="group_average",
                   help="Subject label used for geodesic cache filenames.")
    p.add_argument("--gpu-batch-size", type=int, default=256, dest="gpu_batch_size",
                   help="Vertices per GPU batch for searchlight CKA.")

    # Model specification
    p.add_argument("--target-model",     default="pe-av-small-16-frame",
                   dest="target_model")
    p.add_argument("--target-modality",  default="av", dest="target_modality",
                   choices=["av", "a", "v"])
    p.add_argument("--unimodal-models",  nargs="+",
                   default=["audiomae:a", "videomaev2-large:v"],
                   dest="unimodal_models",
                   help="Space-separated list of model:modality strings.")
    p.add_argument("--bin-sec",   type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--delay-sec", type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr",        type=float, default=TR_DEFAULT)
    return p.parse_args()


# =============================================================================
# Linear algebra helpers
# =============================================================================

def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix: H K H where H = I - 11^T/n."""
    mu = K.mean(axis=0, keepdims=True)
    return K - mu - mu.T + K.mean()


def linear_cka(K1: np.ndarray, K2: np.ndarray) -> float:
    """Centered Kernel Alignment between two (n, n) Gram matrices."""
    K1c = _center_kernel(K1)
    K2c = _center_kernel(K2)
    hsic  = np.sum(K1c * K2c)
    norm1 = np.sqrt(np.sum(K1c * K1c))
    norm2 = np.sqrt(np.sum(K2c * K2c))
    denom = norm1 * norm2
    return float(hsic / denom) if denom > 1e-10 else 0.0


def gram(X: np.ndarray) -> np.ndarray:
    """K = X X^T / n.  X: (n, d)."""
    return (X @ X.T) / X.shape[0]


def compute_interaction_residual(
    J: np.ndarray,
    unimodal_list: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Compute R = J - X_UV (X_UV^T X_UV + eps I)^{-1} X_UV^T J.

    Each matrix is z-scored per feature before concatenation.

    Returns
    -------
    R       : (n, d_J) float64 — interaction residual
    ms_score: float — ||R||_F^2 / ||J||_F^2
    """
    from scipy.stats import zscore as scipy_zscore

    J_z = np.nan_to_num(scipy_zscore(J, axis=0, nan_policy='omit').astype(np.float64))

    U_parts = []
    for U in unimodal_list:
        U_z = np.nan_to_num(scipy_zscore(U, axis=0, nan_policy='omit').astype(np.float64))
        U_parts.append(U_z)
    X_uv = np.concatenate(U_parts, axis=1)

    eps = 1e-6 * np.linalg.norm(X_uv) ** 2 / X_uv.shape[1]
    A   = X_uv.T @ X_uv + eps * np.eye(X_uv.shape[1])
    B   = np.linalg.solve(A, X_uv.T @ J_z)
    R   = J_z - X_uv @ B

    norm_J = np.linalg.norm(J_z, "fro")
    norm_R = np.linalg.norm(R,   "fro")
    ms = float((norm_R / norm_J) ** 2) if norm_J > 1e-10 else 0.0
    return R, ms


# =============================================================================
# Glasser parcel helpers
# =============================================================================

def load_glasser_parcels(dlabel_path: str, n_grayords: int) -> np.ndarray:
    """Return (n_grayords,) int32 parcel labels (0 = medial wall / background)."""
    img    = nib.load(dlabel_path)
    labels = img.get_fdata(dtype=np.float32).squeeze().astype(np.int32)
    if len(labels) != n_grayords:
        raise ValueError(
            f"Glasser dlabel has {len(labels)} grayordinates, expected {n_grayords}."
        )
    return labels


def _run_glasser_cka(
    fmri_binned: np.ndarray,
    K_joint: np.ndarray,
    K_unimodal: np.ndarray,
    K_interaction: np.ndarray,
    glasser_dlabel: str,
    n_grayords: int,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-Glasser-parcel CKA.  Returns (results_list, cka_j, cka_u, cka_i, si) maps."""
    parcel_labels = load_glasser_parcels(glasser_dlabel, n_grayords)
    parcel_ids    = sorted(set(parcel_labels.tolist()) - {0})
    log.info(f"  [Glasser] {len(parcel_ids)} parcels")

    results = []
    for pid in parcel_ids:
        idx = np.where(parcel_labels == pid)[0]
        if len(idx) < 5:
            continue
        K_brain = gram(fmri_binned[:, idx].astype(np.float64))
        cka_j = linear_cka(K_brain, K_joint)
        cka_u = linear_cka(K_brain, K_unimodal)
        cka_i = linear_cka(K_brain, K_interaction)
        si    = cka_i / (cka_j + 1e-12)
        results.append({
            "parcel_id":         int(pid),
            "n_vertices":        len(idx),
            "cka_joint":         cka_j,
            "cka_unimodal":      cka_u,
            "cka_interaction":   cka_i,
            "specificity_index": si,
        })

    results.sort(key=lambda r: r["specificity_index"], reverse=True)
    log.info("  [Glasser] Top-5 parcels by SI:")
    for r in results[:5]:
        log.info(f"    parcel {r['parcel_id']:3d}: SI={r['specificity_index']:.4f}  "
                 f"CKA_int={r['cka_interaction']:.4f}  CKA_joint={r['cka_joint']:.4f}")

    cka_j_map = np.zeros(n_grayords, dtype=np.float32)
    cka_u_map = np.zeros(n_grayords, dtype=np.float32)
    cka_i_map = np.zeros(n_grayords, dtype=np.float32)
    si_map    = np.zeros(n_grayords, dtype=np.float32)
    for r in results:
        idx = np.where(parcel_labels == r["parcel_id"])[0]
        cka_j_map[idx] = r["cka_joint"]
        cka_u_map[idx] = r["cka_unimodal"]
        cka_i_map[idx] = r["cka_interaction"]
        si_map[idx]    = r["specificity_index"]

    return results, cka_j_map, cka_u_map, cka_i_map, si_map


# =============================================================================
# Searchlight CKA helpers
# =============================================================================

def _cka_vertex_cpu(
    sv: int,
    fmri_hem: np.ndarray,
    K_joint_c: np.ndarray,
    K_uni_c: np.ndarray,
    K_int_c: np.ndarray,
    norm_j: float, norm_u: float, norm_i: float,
    neighbors: np.ndarray,
    vertex_to_col: np.ndarray,
) -> tuple[float, float, float]:
    """CKA triplet for a single vertex (CPU, used for partial-k fallback)."""
    ngh_cols = vertex_to_col[neighbors[sv]]
    valid    = ngh_cols[ngh_cols >= 0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0
    X   = fmri_hem[:, valid].astype(np.float64)
    K_b = X @ X.T / len(valid)
    mu  = K_b.mean(axis=0, keepdims=True)
    K_bc = K_b - mu - mu.T + K_b.mean()
    norm_b = float(np.sqrt((K_bc * K_bc).sum()))
    if norm_b < 1e-10:
        return 0.0, 0.0, 0.0
    cka_j = float((K_bc * K_joint_c).sum() / (norm_b * norm_j))
    cka_u = float((K_bc * K_uni_c).sum()   / (norm_b * norm_u))
    cka_i = float((K_bc * K_int_c).sum()   / (norm_b * norm_i))
    return cka_j, cka_u, cka_i


def _run_cka_searchlight_hem_gpu(
    fmri_hem: np.ndarray,
    K_joint: np.ndarray,
    K_unimodal: np.ndarray,
    K_interaction: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    batch_size: int = 256,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU-batched CKA searchlight for one hemisphere.

    Parameters
    ----------
    fmri_hem        : (n_bins, n_hem_verts) float32
    K_joint/unimodal/interaction : (n_bins, n_bins) float64 — model gram matrices
    neighbors       : (n_surf_verts, k) int32
    surface_indices : (n_hem_verts,) int32 — surface vertex per grayordinate
    vertex_to_col   : (n_surf_verts,) int32 — surface vert → fmri column (-1=medial)
    batch_size      : vertices per GPU batch
    device          : torch device string

    Returns
    -------
    cka_j, cka_u, cka_i, si : each (n_hem_verts,) float32
    """
    import torch

    n_verts = fmri_hem.shape[1]
    k       = neighbors.shape[1]

    # Pre-compute centered model Gram matrices on GPU
    def _center_t(K_t: torch.Tensor) -> torch.Tensor:
        """Center (n,n) or (B,n,n) tensor."""
        mu_row = K_t.mean(dim=-1, keepdim=True)
        mu_col = K_t.mean(dim=-2, keepdim=True)
        mu_all = K_t.mean(dim=(-2, -1), keepdim=True)
        return K_t - mu_row - mu_col + mu_all

    Kj_c  = _center_t(torch.tensor(K_joint,      dtype=torch.float32, device=device))
    Ku_c  = _center_t(torch.tensor(K_unimodal,   dtype=torch.float32, device=device))
    Ki_c  = _center_t(torch.tensor(K_interaction, dtype=torch.float32, device=device))

    norm_j = float(torch.sqrt((Kj_c * Kj_c).sum()).clamp(min=1e-10))
    norm_u = float(torch.sqrt((Ku_c * Ku_c).sum()).clamp(min=1e-10))
    norm_i = float(torch.sqrt((Ki_c * Ki_c).sum()).clamp(min=1e-10))

    # fMRI on GPU: (n_hem_verts, n_bins) for fast row gather
    fmri_t = torch.tensor(fmri_hem.T.astype(np.float32), device=device)

    # Pre-compute neighbor fMRI-column indices for every surface vertex
    neighbor_cols_all = vertex_to_col[neighbors]          # (n_surf_verts, k)
    surf_verts_for_v  = surface_indices.astype(np.int32)  # (n_hem_verts,)
    ncols_for_v       = neighbor_cols_all[surf_verts_for_v]  # (n_hem_verts, k)

    full_k_mask    = np.all(ncols_for_v >= 0, axis=1)
    full_k_verts   = np.where(full_k_mask)[0]
    partial_k_verts = np.where(~full_k_mask)[0]
    log.info(
        f"    {len(full_k_verts):,} full-k GPU vertices, "
        f"{len(partial_k_verts):,} partial-k CPU vertices"
    )

    cka_j_map = np.zeros(n_verts, dtype=np.float32)
    cka_u_map = np.zeros(n_verts, dtype=np.float32)
    cka_i_map = np.zeros(n_verts, dtype=np.float32)

    # ── GPU batched pass ─────────────────────────────────────────────────────
    for start in range(0, len(full_k_verts), batch_size):
        batch_v  = full_k_verts[start : start + batch_size]
        batch_nc = ncols_for_v[batch_v].astype(np.int64)   # (B, k)
        B = len(batch_v)

        batch_nc_t = torch.from_numpy(batch_nc).to(device)
        # hood: (B, k, n_bins) → permute to (B, n_bins, k)
        hood = fmri_t[batch_nc_t].permute(0, 2, 1).float()

        # K_brain: (B, n_bins, n_bins)
        K_brain_b = torch.bmm(hood, hood.permute(0, 2, 1)) / k

        # Center batch
        mu_row = K_brain_b.mean(dim=-1, keepdim=True)
        mu_col = K_brain_b.mean(dim=-2, keepdim=True)
        mu_all = K_brain_b.mean(dim=(-2, -1), keepdim=True)
        K_bc   = K_brain_b - mu_row - mu_col + mu_all  # (B, n_bins, n_bins)

        norm_b = torch.sqrt((K_bc * K_bc).sum(dim=(-2, -1))).clamp(min=1e-10)  # (B,)

        cka_j = (K_bc * Kj_c.unsqueeze(0)).sum(dim=(-2, -1)) / (norm_b * norm_j)
        cka_u = (K_bc * Ku_c.unsqueeze(0)).sum(dim=(-2, -1)) / (norm_b * norm_u)
        cka_i = (K_bc * Ki_c.unsqueeze(0)).sum(dim=(-2, -1)) / (norm_b * norm_i)

        cka_j_map[batch_v] = cka_j.cpu().numpy()
        cka_u_map[batch_v] = cka_u.cpu().numpy()
        cka_i_map[batch_v] = cka_i.cpu().numpy()

    # ── CPU fallback (partial-k vertices near medial wall) ──────────────────
    if len(partial_k_verts) > 0:
        fmri_cpu = fmri_hem  # already on CPU
        Kj_c_np = Kj_c.cpu().numpy()
        Ku_c_np = Ku_c.cpu().numpy()
        Ki_c_np = Ki_c.cpu().numpy()
        for v in partial_k_verts:
            sv = int(surf_verts_for_v[v])
            cj, cu, ci = _cka_vertex_cpu(
                sv, fmri_cpu,
                Kj_c_np, Ku_c_np, Ki_c_np,
                norm_j, norm_u, norm_i,
                neighbors, vertex_to_col,
            )
            cka_j_map[v] = cj
            cka_u_map[v] = cu
            cka_i_map[v] = ci

    si_map = (cka_i_map / (cka_j_map + 1e-12)).astype(np.float32)
    return cka_j_map, cka_u_map, cka_i_map, si_map


def _run_cka_searchlight_hem_cpu(
    fmri_hem: np.ndarray,
    K_joint: np.ndarray,
    K_unimodal: np.ndarray,
    K_interaction: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy-batched CKA searchlight for one hemisphere (CPU fallback)."""
    n_verts = fmri_hem.shape[1]
    k       = neighbors.shape[1]

    def _center_np(K):
        mu = K.mean(axis=0, keepdims=True)
        return K - mu - mu.T + K.mean()

    Kj_c  = _center_np(K_joint.astype(np.float32))
    Ku_c  = _center_np(K_unimodal.astype(np.float32))
    Ki_c  = _center_np(K_interaction.astype(np.float32))
    norm_j = float(np.sqrt((Kj_c * Kj_c).sum()).clip(1e-10))
    norm_u = float(np.sqrt((Ku_c * Ku_c).sum()).clip(1e-10))
    norm_i = float(np.sqrt((Ki_c * Ki_c).sum()).clip(1e-10))

    neighbor_cols_all = vertex_to_col[neighbors]         # (n_surf_verts, k)
    surf_verts_for_v  = surface_indices.astype(np.int32)
    ncols_for_v       = neighbor_cols_all[surf_verts_for_v]  # (n_verts, k)

    cka_j_map = np.zeros(n_verts, dtype=np.float32)
    cka_u_map = np.zeros(n_verts, dtype=np.float32)
    cka_i_map = np.zeros(n_verts, dtype=np.float32)

    for start in range(0, n_verts, batch_size):
        batch_v  = np.arange(start, min(start + batch_size, n_verts))
        B = len(batch_v)

        batch_nc = ncols_for_v[batch_v]   # (B, k), may have -1s

        # For simplicity mask each vertex individually (small overhead for CPU)
        for bi, v in enumerate(batch_v):
            cols  = batch_nc[bi]
            valid = cols[cols >= 0]
            if len(valid) == 0:
                continue
            X = fmri_hem[:, valid].astype(np.float32)
            K_b = (X @ X.T) / len(valid)
            mu  = K_b.mean(axis=0, keepdims=True)
            K_bc = K_b - mu - mu.T + K_b.mean()
            norm_b = float(np.sqrt((K_bc * K_bc).sum()))
            if norm_b < 1e-10:
                continue
            cka_j_map[v] = float((K_bc * Kj_c).sum() / (norm_b * norm_j))
            cka_u_map[v] = float((K_bc * Ku_c).sum() / (norm_b * norm_u))
            cka_i_map[v] = float((K_bc * Ki_c).sum() / (norm_b * norm_i))

    si_map = (cka_i_map / (cka_j_map + 1e-12)).astype(np.float32)
    return cka_j_map, cka_u_map, cka_i_map, si_map


def _run_searchlight_cka(
    fmri_binned: np.ndarray,
    K_joint: np.ndarray,
    K_unimodal: np.ndarray,
    K_interaction: np.ndarray,
    args,
    bm_axis,
    lh_verts: np.ndarray,
    rh_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CKA searchlight over both hemispheres; return full-brain maps."""
    cache_dir = Path(args.geodesic_cache_dir)
    n_left    = len(lh_verts)

    surfaces = {
        "left":  (args.left_surface,  fmri_binned[:, :n_left],  lh_verts.astype(np.int32)),
        "right": (args.right_surface, fmri_binned[:, n_left:],  rh_verts.astype(np.int32)),
    }

    cka_j_full = np.zeros(fmri_binned.shape[1], dtype=np.float32)
    cka_u_full = np.zeros(fmri_binned.shape[1], dtype=np.float32)
    cka_i_full = np.zeros(fmri_binned.shape[1], dtype=np.float32)
    si_full    = np.zeros(fmri_binned.shape[1], dtype=np.float32)
    offset = 0

    for hem, (surf_path, fmri_hem, surf_indices) in surfaces.items():
        log.info(f"  [Searchlight] {hem} hemisphere ({fmri_hem.shape[1]:,} verts) ...")
        neighbors = get_neighbors(
            surf_path, args.workbench, args.subject, hem, args.k, cache_dir,
        )
        n_surf_verts = neighbors.shape[0]
        vertex_to_col = np.full(n_surf_verts, -1, dtype=np.int32)
        vertex_to_col[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)

        try:
            import torch as _torch
            if _torch.cuda.is_available():
                log.info(f"    GPU CKA (device=cuda, batch_size={args.gpu_batch_size})")
                try:
                    cka_j, cka_u, cka_i, si = _run_cka_searchlight_hem_gpu(
                        fmri_hem, K_joint, K_unimodal, K_interaction,
                        neighbors, surf_indices, vertex_to_col,
                        batch_size=args.gpu_batch_size, device="cuda",
                    )
                except _torch.cuda.OutOfMemoryError:
                    log.warning("    GPU OOM — falling back to CPU")
                    _torch.cuda.empty_cache()
                    raise RuntimeError("OOM")
            else:
                raise ImportError
        except (ImportError, RuntimeError):
            log.info("    CPU CKA (numpy batch)")
            cka_j, cka_u, cka_i, si = _run_cka_searchlight_hem_cpu(
                fmri_hem, K_joint, K_unimodal, K_interaction,
                neighbors, surf_indices, vertex_to_col,
            )

        n_hem = fmri_hem.shape[1]
        cka_j_full[offset : offset + n_hem] = cka_j
        cka_u_full[offset : offset + n_hem] = cka_u
        cka_i_full[offset : offset + n_hem] = cka_i
        si_full[offset : offset + n_hem]    = si
        offset += n_hem

        del neighbors, vertex_to_col

    return cka_j_full, cka_u_full, cka_i_full, si_full


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ── Validate method-specific arguments ────────────────────────────────────
    if args.method in ("glasser", "both") and args.glasser_dlabel is None:
        raise ValueError("--glasser-dlabel is required for method=glasser or method=both")
    if args.method in ("searchlight", "both"):
        missing = [
            f"--{n.replace('_', '-')}"
            for n in ("left_surface", "right_surface", "geodesic_cache_dir")
            if getattr(args, n) is None
        ]
        if missing:
            raise ValueError(
                f"Required for method=searchlight or method=both: {', '.join(missing)}"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_short = f"{args.target_model}_{args.target_modality}"

    log.info("=" * 70)
    log.info("Multimodal Interaction Decomposition (CKA)")
    log.info(f"  Target      : {args.target_model}/{args.target_modality}")
    log.info(f"  Unimodal    : {args.unimodal_models}")
    log.info(f"  Method      : {args.method}")
    log.info(f"  bin_sec={args.bin_sec}  delay_sec={args.delay_sec}  tr={args.tr}")
    log.info("=" * 70)

    timing_df = pd.read_csv(args.timing_csv)
    run_trs   = np.load(args.run_trs)

    # ── Load and bin fMRI ────────────────────────────────────────────────────
    log.info("  Loading fMRI ...")
    fmri_raw    = load_fmri_cifti(args.fmri_cifti)
    fmri_binned = preprocess_fmri(
        fmri_raw, timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec
    )
    del fmri_raw
    n_bins, n_grayords = fmri_binned.shape
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    # ── Load model embeddings ────────────────────────────────────────────────
    log.info("  Loading model embeddings ...")
    def load_emb(model, modality):
        path = check_embeddings_exist(args.embeddings_dir, model, modality, args.bin_sec)
        emb  = process_model_embeddings(
            str(path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec,
        )
        _, emb = align_and_assert_bins(fmri_binned, emb)
        return emb.astype(np.float64)

    J = load_emb(args.target_model, args.target_modality)
    log.info(f"  Joint embedding J: {J.shape}")

    unimodal_embs  = []
    unimodal_names = []
    for entry in args.unimodal_models:
        model, modality = entry.split(":", 1)
        unimodal_embs.append(load_emb(model, modality))
        unimodal_names.append(f"{model}:{modality}")
        log.info(f"  Unimodal {model}/{modality}: {unimodal_embs[-1].shape}")

    # ── Interaction residual ─────────────────────────────────────────────────
    log.info("  Computing interaction residual ...")
    R, ms_score = compute_interaction_residual(J, unimodal_embs)
    log.info(f"  Global multimodality score (MS): {ms_score:.4f}")

    # ── Model Gram matrices (computed once; shared across all vertices) ───────
    from scipy.stats import zscore as scipy_zscore
    J_z = np.nan_to_num(scipy_zscore(J, axis=0)).astype(np.float64)
    U_z = np.concatenate([
        np.nan_to_num(scipy_zscore(u, axis=0)) for u in unimodal_embs
    ], axis=1).astype(np.float64)
    R_z = R.astype(np.float64)

    K_joint       = gram(J_z)
    K_unimodal    = gram(U_z)
    K_interaction = gram(R_z)
    log.info(
        f"  Gram diag means — joint={np.diag(K_joint).mean():.4f}  "
        f"unimodal={np.diag(K_unimodal).mean():.4f}  "
        f"interaction={np.diag(K_interaction).mean():.4f}"
    )

    # ── CIFTI grayordinate layout (needed by both methods) ───────────────────
    bm_axis  = get_bm_axis(args.template_cifti)
    lh_verts, rh_verts = get_cortex_vertex_indices(bm_axis)

    # ── Run analyses ─────────────────────────────────────────────────────────
    all_maps: dict[str, np.ndarray] = {}
    glasser_results: list[dict] = []

    if args.method in ("glasser", "both"):
        log.info("  Running Glasser parcel CKA ...")
        glasser_results, cka_j, cka_u, cka_i, si = _run_glasser_cka(
            fmri_binned, K_joint, K_unimodal, K_interaction,
            args.glasser_dlabel, n_grayords,
        )
        all_maps[f"glasser_cka_joint_{target_short}"]       = cka_j
        all_maps[f"glasser_cka_unimodal_{target_short}"]    = cka_u
        all_maps[f"glasser_cka_interaction_{target_short}"] = cka_i
        all_maps[f"glasser_specificity_index_{target_short}"] = si

    if args.method in ("searchlight", "both"):
        log.info(f"  Running searchlight CKA (k={args.k}) ...")
        cka_j, cka_u, cka_i, si = _run_searchlight_cka(
            fmri_binned, K_joint, K_unimodal, K_interaction,
            args, bm_axis, lh_verts, rh_verts,
        )
        all_maps[f"searchlight_cka_joint_{target_short}"]       = cka_j
        all_maps[f"searchlight_cka_unimodal_{target_short}"]    = cka_u
        all_maps[f"searchlight_cka_interaction_{target_short}"] = cka_i
        all_maps[f"searchlight_specificity_index_{target_short}"] = si

    # ── Save combined CIFTI (all active maps in one file) ────────────────────
    cka_cifti_path = out_dir / f"cka_decomp_{target_short}.dscalar.nii"
    map_names = list(all_maps.keys())
    data_2d   = np.stack(list(all_maps.values()), axis=0)
    save_cifti_multimap(data_2d, map_names, args.template_cifti, str(cka_cifti_path))
    log.info(f"  Saved CIFTI ({len(map_names)} maps): {cka_cifti_path.name}")
    for name in map_names:
        log.info(f"    {name}")

    # ── Save Glasser CSV ─────────────────────────────────────────────────────
    if glasser_results:
        import csv
        csv_path = out_dir / f"parcel_cka_{target_short}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(glasser_results[0].keys()))
            writer.writeheader()
            writer.writerows(glasser_results)
        log.info(f"  Saved parcel CKA CSV: {csv_path.name}")

    # ── Global summary JSON ──────────────────────────────────────────────────
    summary = {
        "target_model":               args.target_model,
        "target_modality":            args.target_modality,
        "unimodal_models":            args.unimodal_models,
        "method":                     args.method,
        "bin_sec":                    args.bin_sec,
        "delay_sec":                  args.delay_sec,
        "n_bins":                     int(n_bins),
        "n_grayords":                 int(n_grayords),
        "global_multimodality_score": ms_score,
        "maps_saved":                 map_names,
        "model_recommendations": {
            "pe-av-large-16-frame": (
                "Larger PE-AV — tests if interaction residual scales with model capacity."
            ),
            "cav-mae-sync": (
                "Contrastive AV model — architecture-independent interaction test."
            ),
            "imagebind": (
                "Semantic AV+text alignment — distinguishes perceptual from "
                "conceptual integration."
            ),
        },
    }
    if glasser_results:
        summary["n_parcels_analyzed"]   = len(glasser_results)
        summary["mean_specificity_index"] = float(
            np.mean([r["specificity_index"] for r in glasser_results])
        )
        summary["top10_parcels_by_SI"]  = [r["parcel_id"] for r in glasser_results[:10]]

    summary_path = out_dir / f"interaction_score_{target_short}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"  Saved global summary: {summary_path.name}")
    log.info(f"  Global multimodality score (MS = ||R||²/||J||²): {ms_score:.4f}")
    log.info("Multimodal decomposition complete.")


if __name__ == "__main__":
    main()
