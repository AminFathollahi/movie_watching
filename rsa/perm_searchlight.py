"""
rsa/perm_searchlight.py
=======================
Permutation test for the group-average searchlight RSA.

For each vertex the brain RDM is computed exactly as in searchlight.py
(correlation-distance lower triangle over the k geodesic neighbours).
A null distribution is built by permuting the model-RDM condition labels
n_perm times and recomputing the RSA correlation for each permutation.

Mathematical basis
------------------
Permuting condition labels of a symmetric RDM merely reorders the
lower-triangle vector.  For Spearman RSA this means the null distribution
is equivalent to shuffling the unit-normalised rank vector of the brain
(or equivalently of the model), which reduces to a pure index look-up:

    null_rho[v, p] = brain_norm[v] · model_norm[perm_p]

where perm_p is a random permutation of [0, n_pairs).  For a batch of
B vertices and P permutations this is a single GPU matmul:

    null_rho_batch = brain_norm_batch  @  perm_model_batch.T
                     (B, n_pairs)         (P, n_pairs).T

No RDM recomputation is needed across permutations — the brain RDM is
computed once per vertex, ranked once, then re-indexed for every null.

Empirical one-tailed p-value (right tail):
    p_perm[v] = (count(null_rho[v,:] >= actual_rho[v]) + 1) / (n_perm + 1)

Outputs saved to the same config directory as searchlight.py:
  {config}/rsa_59k_{tag}_..._perm{N}_p.npy     — (n_verts,) p-values
  Combined CIFTI maps (appended to --combined-output):
    searchlight_{method}_sigmap_perm            — sign × -log10(p_perm)
    searchlight_{method}_sigmap_perm_fdr        — BH-FDR corrected

Usage
-----
  python rsa/perm_searchlight.py \\
      --preprocessed-dir <path> --fmri-suffix raw \\
      --timing-csv <path> --embeddings-dir <path> \\
      --template-cifti <path> --output-dir <path> \\
      --left-surface <path> --right-surface <path> \\
      --workbench <path> \\
      --model pe-av-small-16-frame --modality av \\
      --k 100 --bin-sec 5.0 --delay-sec 5.0 \\
      --method spearman --tr 1.0 \\
      --n-perm 1000
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.rsa_utils import (
    load_fmri_cifti,
    preprocess_fmri,
    process_model_embeddings,
    align_and_assert_bins,
)
from rsa.searchlight import (
    get_neighbors,
    _precompute_model_rdm,
    _fdr_sigmap,
)
from cifti_io import (
    get_bm_axis,
    get_cortex_vertex_indices,
    get_combined_map_names,
    merge_into_combined,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Permutation test for group-average searchlight RSA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessed-dir", required=True, dest="preprocessed_dir",
                   help="Directory of continuous pre-cleaned CIFTIs "
                        "({dir}/{subject}_{fmri-suffix}_cortex_59k.dtseries.nii).")
    p.add_argument("--fmri-suffix",    default="raw", dest="fmri_suffix")
    p.add_argument("--timing-csv",     required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--left-surface",   required=True)
    p.add_argument("--right-surface",  required=True)
    p.add_argument("--workbench",      required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--subject",        default="group_average")
    p.add_argument("--model",          required=True)
    p.add_argument("--modality",       required=True,
                   choices=["v", "a", "av", "at", "vt", "avt", "t"])
    p.add_argument("--k",              type=int,   required=True)
    p.add_argument("--bin-sec",        type=float, required=True)
    p.add_argument("--skip-sec",       type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec",      type=float, default=5.0)
    p.add_argument("--hrf",            action="store_true")
    p.add_argument("--method",         required=True, choices=["spearman", "pearson"])
    p.add_argument("--tr",             type=float, required=True)
    p.add_argument("--normalize",      action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir")
    p.add_argument("--combined-output",    default=None, dest="combined_output",
                   help="Combined .dscalar.nii to append sigmap_perm maps to "
                        "(same file used by searchlight.py).")
    p.add_argument("--gpu-batch-size",  type=int, default=256, dest="gpu_batch_size",
                   help="Vertices per GPU batch. Lower than searchlight.py default "
                        "because permutation stores (B, n_pairs) brain norm on GPU.")
    p.add_argument("--perm-batch-size", type=int, default=100, dest="perm_batch_size",
                   help="Permutations per matmul batch: (B, n_pairs)×(n_pairs, P) matmul. "
                        "Larger = faster but more GPU memory.")
    p.add_argument("--n-perm",  type=int, default=1000, dest="n_perm",
                   help="Number of permutation iterations.")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# =============================================================================
# Per-vertex CPU fallback
# =============================================================================

def _perm_vertex_cpu(surf_v, fmri_cpu, model_norm, neighbors, vertex_to_col,
                     tril_idx, method, perm_idx_all):
    """Single-vertex permutation test (CPU). Returns (actual_rho, exceed_count)."""
    neighbor_surf = neighbors[surf_v]
    neighbor_cols = vertex_to_col[neighbor_surf]
    neighbor_cols = neighbor_cols[neighbor_cols >= 0]
    if len(neighbor_cols) < 2:
        return 0.0, len(perm_idx_all) // 2

    hood = fmri_cpu[:, neighbor_cols]
    mu   = hood.mean(axis=1, keepdims=True)
    hc   = hood - mu
    nrms = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
    nrms[nrms < 1e-10] = 1.0
    hn   = hc / nrms
    fmri_flat = (1.0 - hn @ hn.T)[tril_idx]

    if method == "spearman":
        order = np.argsort(fmri_flat)
        fr    = np.empty(len(fmri_flat), dtype=np.float32)
        fr[order] = np.arange(len(fmri_flat), dtype=np.float32)
        fc    = fr - fr.mean()
        fn    = np.linalg.norm(fc)
        brain_norm = (fc / fn).astype(np.float32) if fn > 1e-10 else fc.astype(np.float32)
    else:
        fc    = (fmri_flat - fmri_flat.mean()).astype(np.float32)
        fn    = np.linalg.norm(fc)
        brain_norm = (fc / fn) if fn > 1e-10 else fc

    actual_rho   = float(np.dot(brain_norm, model_norm))
    # Permute model (equivalent to permuting brain condition labels)
    null_rhos    = model_norm[perm_idx_all] @ brain_norm   # (n_perm,)
    exceed_count = int((null_rhos >= actual_rho).sum())
    return actual_rho, exceed_count


# =============================================================================
# GPU permutation searchlight
# =============================================================================

def _perm_searchlight_gpu(fmri_hem, model_emb, neighbors, surface_indices,
                           vertex_to_col, method, n_perm,
                           vertex_batch_size, perm_batch_size, device, rng):
    """GPU-batched permutation searchlight for one hemisphere.

    Returns
    -------
    actual_rho : (n_verts,) float32
    p_perm     : (n_verts,) float32  — empirical one-tailed p-values
    """
    import torch

    n_verts  = fmri_hem.shape[1]
    n_bins   = fmri_hem.shape[0]
    tril_idx = np.tril_indices(n_bins, k=-1)
    n_pairs  = len(tril_idx[0])

    _, model_norm = _precompute_model_rdm(model_emb, n_bins, tril_idx, method)
    model_norm_t  = torch.from_numpy(model_norm).to(device)   # (n_pairs,)

    # All permutation indices on CPU; loaded to GPU in perm_batch_size slabs
    perm_idx_all = np.stack(
        [rng.permutation(n_pairs) for _ in range(n_perm)]
    ).astype(np.int64)   # (n_perm, n_pairs)

    # fMRI transposed once to GPU: (n_hem_verts, n_bins)
    fmri_t = torch.from_numpy(fmri_hem.T.astype(np.float32)).to(device)

    neighbor_cols_all = vertex_to_col[neighbors]
    surf_verts_for_v  = surface_indices.astype(np.int32)
    ncols_for_v       = neighbor_cols_all[surf_verts_for_v]
    full_k_mask       = np.all(ncols_for_v >= 0, axis=1)
    full_k_verts      = np.where(full_k_mask)[0]
    partial_k_verts   = np.where(~full_k_mask)[0]

    log.info(
        f"  [GPU] {len(full_k_verts):,} full-k verts (batch={vertex_batch_size}), "
        f"{len(partial_k_verts):,} partial-k (CPU fallback) | "
        f"n_pairs={n_pairs:,}, perm_batch={perm_batch_size}"
    )

    actual_rho   = np.zeros(n_verts, dtype=np.float32)
    exceed_count = np.zeros(n_verts, dtype=np.int64)

    tril_row = torch.tensor(tril_idx[0], dtype=torch.long, device=device)
    tril_col = torch.tensor(tril_idx[1], dtype=torch.long, device=device)

    for b_start in range(0, len(full_k_verts), vertex_batch_size):
        batch_v  = full_k_verts[b_start: b_start + vertex_batch_size]
        batch_nc = ncols_for_v[batch_v].astype(np.int64)
        B = len(batch_v)

        # Gather neighbourhood: (B, n_bins, k)
        hood  = fmri_t[torch.from_numpy(batch_nc).to(device)].permute(0, 2, 1).float()
        mu    = hood.mean(dim=2, keepdim=True)
        hc    = hood - mu
        norms = torch.linalg.norm(hc, dim=2, keepdim=True).clamp(min=1e-10)
        hn    = hc / norms
        del hood, hc, norms

        # Brain RDM lower triangle: (B, n_pairs)
        flat = (1.0 - torch.bmm(hn, hn.permute(0, 2, 1)))[:, tril_row, tril_col]
        del hn

        # Unit-normalised rank (Spearman) or centred value (Pearson)
        if method == "spearman":
            ranks = torch.argsort(torch.argsort(flat, dim=1), dim=1).float()
            fc    = ranks - ranks.mean(dim=1, keepdim=True)
            del ranks
        else:
            fc = flat - flat.mean(dim=1, keepdim=True)
        del flat

        fn           = torch.linalg.norm(fc, dim=1, keepdim=True).clamp(min=1e-10)
        brain_norm_t = fc / fn          # (B, n_pairs) unit vectors
        del fc, fn

        # Actual rho
        rho_batch = (brain_norm_t * model_norm_t).sum(dim=1)   # (B,)
        actual_rho[batch_v] = rho_batch.cpu().float().numpy()

        # Null distribution: batch permutations
        # null_rho[b, p] = brain_norm_t[b] · model_norm_t[perm_p]
        #                = (brain_norm_t @ perm_model_batch.T)[b, p]
        exceed_t = torch.zeros(B, dtype=torch.int64, device=device)
        for p0 in range(0, n_perm, perm_batch_size):
            p1        = min(p0 + perm_batch_size, n_perm)
            idx_t     = torch.from_numpy(perm_idx_all[p0:p1]).to(device)  # (P, n_pairs)
            perm_model = model_norm_t[idx_t]                               # (P, n_pairs)
            null_rho   = brain_norm_t @ perm_model.T                      # (B, P)
            exceed_t  += (null_rho >= rho_batch.unsqueeze(1)).sum(dim=1)
            del idx_t, perm_model, null_rho

        exceed_count[batch_v] = exceed_t.cpu().numpy()
        del brain_norm_t, rho_batch, exceed_t

        if device == "cuda":
            torch.cuda.empty_cache()

        if (b_start // vertex_batch_size) % 20 == 0:
            done = b_start + B
            log.info(
                f"  [{done:,}/{len(full_k_verts):,}] "
                f"mean_rho={actual_rho[batch_v].mean():.4f}"
            )

    # CPU fallback for partial-k vertices (near medial wall)
    if len(partial_k_verts) > 0:
        log.info(f"  [CPU fallback] {len(partial_k_verts):,} partial-k vertices ...")
        fmri_cpu = fmri_t.cpu().numpy().T   # (n_bins, n_hem_verts)
        for v in partial_k_verts:
            sv = int(surf_verts_for_v[v])
            rho_v, exc_v = _perm_vertex_cpu(
                sv, fmri_cpu, model_norm, neighbors, vertex_to_col,
                tril_idx, method, perm_idx_all,
            )
            actual_rho[v]   = rho_v
            exceed_count[v] = exc_v

    p_perm = ((exceed_count + 1.0) / (n_perm + 1.0)).astype(np.float32)
    return actual_rho, p_perm


# =============================================================================
# CPU-only permutation searchlight (no torch)
# =============================================================================

def _perm_searchlight_cpu(fmri_hem, model_emb, neighbors, surface_indices,
                           vertex_to_col, method, n_perm, rng):
    """CPU-only permutation searchlight using joblib threads."""
    from joblib import Parallel, delayed

    n_verts  = fmri_hem.shape[1]
    n_bins   = fmri_hem.shape[0]
    tril_idx = np.tril_indices(n_bins, k=-1)
    n_pairs  = len(tril_idx[0])

    _, model_norm = _precompute_model_rdm(model_emb, n_bins, tril_idx, method)
    perm_idx_all  = np.stack(
        [rng.permutation(n_pairs) for _ in range(n_perm)]
    ).astype(np.int64)

    surf_verts_for_v = surface_indices.astype(np.int32)
    log.info(f"  [CPU perm] {n_verts:,} vertices (joblib) ...")

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_perm_vertex_cpu)(
            int(surf_verts_for_v[v]), fmri_hem, model_norm,
            neighbors, vertex_to_col, tril_idx, method, perm_idx_all,
        )
        for v in range(n_verts)
    )

    rho_arr = np.array([r[0] for r in results], dtype=np.float32)
    exc_arr = np.array([r[1] for r in results], dtype=np.int64)
    p_perm  = ((exc_arr + 1.0) / (n_perm + 1.0)).astype(np.float32)
    return rho_arr, p_perm


# =============================================================================
# Hemisphere dispatcher
# =============================================================================

def _run_hemisphere(fmri_hem, emb, neighbors, surf_indices, vertex_to_col,
                    method, n_perm, gpu_batch_size, perm_batch_size, rng):
    """Try GPU; fall back to CPU on OOM or missing torch."""
    try:
        import torch
        _oom = [torch.cuda.OutOfMemoryError]
        if hasattr(torch, "AcceleratorError"):
            _oom.append(torch.AcceleratorError)
        _oom = tuple(_oom)
        if torch.cuda.is_available():
            log.info("  Attempting GPU permutation searchlight ...")
            try:
                return _perm_searchlight_gpu(
                    fmri_hem, emb, neighbors, surf_indices, vertex_to_col,
                    method, n_perm, gpu_batch_size, perm_batch_size, "cuda", rng,
                )
            except _oom as e:
                log.warning(f"  GPU OOM ({type(e).__name__}) — falling back to CPU")
                torch.cuda.empty_cache()
    except ImportError:
        pass

    return _perm_searchlight_cpu(
        fmri_hem, emb, neighbors, surf_indices, vertex_to_col,
        method, n_perm, rng,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    normalize   = getattr(args, "normalize", True)
    norm_tag    = "" if normalize else "_demean"
    bin_sec_int = int(args.bin_sec)
    skip_int    = int(args.skip_sec)
    delay_tag   = f"delay{int(args.delay_sec)}s"
    full_tag    = f"{args.fmri_suffix}{norm_tag}"
    config      = f"k{args.k}_{delay_tag}_bin{bin_sec_int}s_skip{skip_int}s_{args.method}"
    n_perm      = args.n_perm

    out_root = (Path(args.output_dir) / args.subject
                / f"{args.model}_{args.modality}" / config)
    out_root.mkdir(parents=True, exist_ok=True)

    stem          = (f"rsa_59k_{full_tag}_k{args.k}_{delay_tag}"
                     f"_bin{bin_sec_int}s_skip{skip_int}s_{args.method}")
    perm_p_path   = out_root / f"{stem}_perm{n_perm}_p.npy"
    combined_path = Path(args.combined_output) if args.combined_output else None
    map_perm      = f"searchlight_{args.method}_sigmap_perm"

    # Skip if already complete
    if perm_p_path.exists():
        if combined_path is None or (combined_path.exists() and
                                     map_perm in get_combined_map_names(str(combined_path))):
            log.info("Permutation results already complete — skipping.")
            return

    # ── Load fMRI and model data ──────────────────────────────────────────────
    cifti_path = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")
    trs_path   = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_run_trs.npy")

    log.info(f"Permutation searchlight: {args.model}/{args.modality}/{config}")
    log.info(f"  n_perm={n_perm}  seed={args.seed}")
    log.info(f"  fMRI: {cifti_path}")

    fmri_continuous = load_fmri_cifti(str(cifti_path))
    run_trs         = np.load(str(trs_path))
    timing_df       = pd.read_csv(args.timing_csv)

    fmri_binned = preprocess_fmri(
        fmri_continuous, timing_df, run_trs, args.bin_sec, args.tr,
        args.delay_sec, skip_sec=args.skip_sec, normalize=normalize,
    )
    del fmri_continuous
    gc.collect()
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"bin{bin_sec_int}s_skip{skip_int}s" / f"{args.model}_{args.modality}.npy")
    emb = process_model_embeddings(
        str(emb_file), timing_df, bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, hrf=args.hrf,
        skip_sec=args.skip_sec, normalize=normalize,
    )
    log.info(f"  Embeddings: {emb.shape}")

    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)
    n_bins  = fmri_binned.shape[0]
    n_pairs = n_bins * (n_bins - 1) // 2
    log.info(f"  n_bins={n_bins}  n_pairs={n_pairs:,}")

    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left  = len(left_indices)

    cache_dir = (Path(args.geodesic_cache_dir) if args.geodesic_cache_dir
                 else Path(args.output_dir) / "_geodesic_cache")
    rng = np.random.default_rng(args.seed)

    surfaces = {
        "left":  (args.left_surface,  fmri_binned[:, :n_left],  left_indices),
        "right": (args.right_surface, fmri_binned[:, n_left:],  right_indices),
    }

    n_total   = fmri_binned.shape[1]
    rho_full  = np.zeros(n_total, dtype=np.float32)
    p_full    = np.zeros(n_total, dtype=np.float32)
    offset    = 0

    # ── Per-hemisphere permutation searchlight ────────────────────────────────
    for hem, (surf_path, fmri_hem, surf_indices) in surfaces.items():
        log.info(f"Hemisphere: {hem} ({len(surf_indices):,} vertices)")
        neighbors = get_neighbors(
            surf_path, args.workbench, args.subject, hem, args.k, cache_dir,
        )
        surf_indices  = surf_indices.astype(np.int32)
        n_surf_verts  = neighbors.shape[0]
        vertex_to_col = np.full(n_surf_verts, -1, dtype=np.int32)
        vertex_to_col[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)
        n_hem = len(surf_indices)

        rho_hem, p_hem = _run_hemisphere(
            fmri_hem, emb, neighbors, surf_indices, vertex_to_col,
            args.method, n_perm,
            args.gpu_batch_size, args.perm_batch_size, rng,
        )

        rho_full[offset: offset + n_hem] = rho_hem
        p_full[offset:   offset + n_hem] = p_hem
        offset += n_hem

        log.info(
            f"  {hem}: mean_rho={rho_hem.mean():.4f}  "
            f"n_sig(p<0.05)={(p_hem < 0.05).sum():,}/{n_hem:,}"
        )
        del neighbors, vertex_to_col, rho_hem, p_hem
        gc.collect()

    # ── Save raw p-values ─────────────────────────────────────────────────────
    np.save(str(perm_p_path), p_full)
    log.info(f"Saved p-values: {perm_p_path.name}")

    # ── Significance map: sign(rho) * -log10(p) ──────────────────────────────
    eps         = np.finfo(np.float32).tiny
    sigmap_perm = (np.sign(rho_full) *
                   (-np.log10(np.maximum(p_full, eps)))).astype(np.float32)

    n_sig = int((p_full < 0.05).sum())
    log.info(f"n_sig p<0.05: {n_sig:,} / {n_total:,}")

    # ── Append sigmap_perm to combined CIFTI ─────────────────────────────────
    if combined_path is not None:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        existing = get_combined_map_names(str(combined_path))
        if map_perm not in existing:
            merge_into_combined(sigmap_perm, map_perm,
                                str(combined_path), args.template_cifti)
            log.info(f"Added CIFTI map: {map_perm}")
    else:
        from cifti_io import save_cifti_map
        maps_path = out_root / f"{stem}_perm{n_perm}_sigmap.dscalar.nii"
        save_cifti_map(sigmap_perm, args.template_cifti, str(maps_path), map_perm)
        log.info(f"Saved significance map: {maps_path.name}")

    log.info("Done.")


if __name__ == "__main__":
    main()
