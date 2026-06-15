"""
rsa/noise_ceiling.py
====================
Vertex-wise inter-subject noise ceiling for searchlight RSA.

At vertex v, the brain RDM is the correlation-distance matrix computed
from fMRI patterns in the geodesic k-NN neighbourhood across all time
bins.  For N subjects the noise ceiling is:

  NC_upper(v) = (1/N) Σ_i  ρ_s[ RDM_i(v),  mean_j  [ RDM_j(v) ] ]
  NC_lower(v) = (1/N) Σ_i  ρ_s[ RDM_i(v),  mean_{j≠i}[ RDM_j(v) ] ]

where ρ_s is Spearman correlation on the lower-triangle RDM vectors.

NC_upper includes subject i in the group mean (biased upward).
NC_lower uses the leave-one-out (LOO) group mean (unbiased lower bound).

Any model's group mean RSA score (mean_rho from group_stats.py) is
bounded above by NC_upper.  Overlaying both maps in Workbench lets you
read off — per vertex — what fraction of the explainable variance your
model accounts for.

Algorithm (per hemisphere, GPU-batched)
----------------------------------------
For a batch of B vertices:
  1. For each subject i: gather the (n_bins, k) searchlight neighbourhood,
     compute correlation-distance lower-triangle → raw_rdm_i  (B, n_pairs)
  2. group_mean  = mean_i[ raw_rdm_i ]                               (B, n_pairs)
  3. Rank each raw_rdm_i along n_pairs                   (Spearman step)
  4. NC_upper: Pearson(rank(raw_rdm_i), rank(group_mean)), avg over i
  5. loo_mean_i = (N·group_mean − raw_rdm_i) / (N−1)
     NC_lower:  Pearson(rank(raw_rdm_i), rank(loo_mean_i)), avg over i
  6. Save nc_lower, nc_upper as CIFTI dscalar maps.

Usage
-----
python rsa/noise_ceiling.py \\
    --preprocessed-root /path/to/preprocessed \\
    --subjects 100610 102311 103414 \\
    --fmri-suffix raw \\
    --timing-csv /path/to/movie_timing.csv \\
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 0.0 --tr 1.0 \\
    --method spearman \\
    --left-surface  /path/to/L.midthickness.surf.gii \\
    --right-surface /path/to/R.midthickness.surf.gii \\
    --workbench /opt/workbench/bin_linux64/wb_command \\
    --geodesic-cache-dir /path/to/rsa/_geodesic_cache \\
    --template-cifti /path/to/template.dscalar.nii \\
    --output-dir /path/to/nc_output

Notes on --delay-sec
---------------------
If your preprocessed CIFTIs were produced with --timing-csv (filtered
mode), the haemodynamic delay has already been folded into the TR
selection — pass --delay-sec 0.  If the CIFTIs are raw concatenated
runs, pass --delay-sec 5.0 (or whatever shift matches your searchlight).
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap
from rsa.shared.rsa_utils import load_fmri_cifti, preprocess_fmri

sys.path.insert(0, str(Path(__file__).parent))
from searchlight import get_neighbors  # noqa: E402

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
        description="Vertex-wise inter-subject noise ceiling for searchlight RSA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessed-root", required=True, dest="preprocessed_root",
                   help="Root dir.  Subject S CIFTI resolved as "
                        "{root}/{S}/{S}_{suffix}_cortex_59k.dtseries.nii "
                        "(or flat: {root}/{S}_{suffix}_cortex_59k.dtseries.nii).")
    p.add_argument("--subjects",          nargs="+", required=True,
                   help="Subject IDs (used for file naming).")
    p.add_argument("--fmri-suffix",       default="raw", dest="fmri_suffix")
    p.add_argument("--timing-csv",        required=True, dest="timing_csv")
    p.add_argument("--k",                 type=int, required=True)
    p.add_argument("--bin-sec",           type=float, required=True, dest="bin_sec")
    p.add_argument("--skip-sec",          type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec",         type=float, default=0.0, dest="delay_sec")
    p.add_argument("--tr",                type=float, default=1.0)
    p.add_argument("--method",            default="spearman",
                   choices=["spearman", "pearson"])
    p.add_argument("--left-surface",      required=True, dest="left_surface")
    p.add_argument("--right-surface",     required=True, dest="right_surface")
    p.add_argument("--workbench",         required=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir")
    p.add_argument("--template-cifti",    required=True, dest="template_cifti")
    p.add_argument("--output-dir",        required=True, dest="output_dir")
    p.add_argument("--batch-size",        type=int, default=32, dest="batch_size",
                   help="Vertices per GPU batch.  Reduce if OOM.")
    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def _resolve_paths(root: str, subject: str, suffix: str) -> tuple[Path, Path]:
    """Return (cifti_path, run_trs_path) for a subject, trying nested then flat."""
    r = Path(root)
    nested_cifti = r / subject / f"{subject}_{suffix}_cortex_59k.dtseries.nii"
    flat_cifti   = r / f"{subject}_{suffix}_cortex_59k.dtseries.nii"
    nested_trs   = r / subject / f"{subject}_{suffix}_run_trs.npy"
    flat_trs     = r / f"{subject}_{suffix}_run_trs.npy"

    cifti = nested_cifti if nested_cifti.exists() else flat_cifti
    trs   = nested_trs   if nested_trs.exists()   else flat_trs

    if not cifti.exists():
        raise FileNotFoundError(
            f"CIFTI not found for {subject}.\n"
            f"  Tried: {nested_cifti}\n"
            f"         {flat_cifti}"
        )
    if not trs.exists():
        raise FileNotFoundError(
            f"run_trs.npy not found for {subject}.\n"
            f"  Tried: {nested_trs}\n"
            f"         {flat_trs}"
        )
    return cifti, trs


def _load_and_bin(subject: str, root: str, suffix: str,
                  timing_df: pd.DataFrame, bin_sec: float, skip_sec: float,
                  delay_sec: float, tr: float) -> np.ndarray:
    """Load + bin one subject's fMRI → (n_bins, n_verts) float32."""
    cifti_path, trs_path = _resolve_paths(root, subject, suffix)
    fmri_raw = load_fmri_cifti(str(cifti_path))          # (n_verts, T)
    run_trs  = np.load(str(trs_path))
    binned   = preprocess_fmri(
        fmri_raw, timing_df, run_trs,
        bin_sec=bin_sec, tr=tr, delay_sec=delay_sec, skip_sec=skip_sec,
    )                                                      # (n_bins, n_verts)
    del fmri_raw
    return binned.astype(np.float32)


# =============================================================================
# Tensor helpers
# =============================================================================

def _rank_rows(x):
    """Fractional rank along last dim (argsort of argsort). Input: (..., N)."""
    return x.argsort(dim=-1).argsort(dim=-1).float()


def _pearson_rows(A, B):
    """Pearson r between matching rows.  A, B: (..., N) → (...,)."""
    Ac = A - A.mean(dim=-1, keepdim=True)
    Bc = B - B.mean(dim=-1, keepdim=True)
    denom = (Ac.norm(dim=-1) * Bc.norm(dim=-1)).clamp(min=1e-10)
    return (Ac * Bc).sum(dim=-1) / denom


# =============================================================================
# Core GPU-batched noise ceiling
# =============================================================================

def _compute_nc_gpu(
    fmri_subs: list[np.ndarray],    # N × (n_bins, n_hem_verts) float32
    neighbors: np.ndarray,           # (n_surf_verts, k)
    surface_indices: np.ndarray,     # (n_hem_verts,) int32
    vertex_to_col: np.ndarray,       # (n_surf_verts,) int32
    method: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    N       = len(fmri_subs)
    n_verts = fmri_subs[0].shape[1]
    n_bins  = fmri_subs[0].shape[0]
    tril    = np.tril_indices(n_bins, k=-1)
    n_pairs = len(tril[0])
    tril_r  = torch.tensor(tril[0], dtype=torch.long, device=device)
    tril_c  = torch.tensor(tril[1], dtype=torch.long, device=device)

    # All subjects' fMRI on GPU: list of (n_hem_verts, n_bins)
    fmri_t = [
        torch.from_numpy(f.T.astype(np.float32)).to(device) for f in fmri_subs
    ]

    ncols_for_v = vertex_to_col[neighbors[surface_indices.astype(np.int32)]]  # (n_verts, k)
    full_k_mask = np.all(ncols_for_v >= 0, axis=1)
    full_k_verts    = np.where(full_k_mask)[0]
    partial_k_verts = np.where(~full_k_mask)[0]

    log.info(f"  {len(full_k_verts):,} full-k vertices (GPU batch={batch_size}), "
             f"{len(partial_k_verts):,} partial-k (CPU fallback)")

    nc_upper = np.zeros(n_verts, dtype=np.float32)
    nc_lower = np.zeros(n_verts, dtype=np.float32)

    for start in range(0, len(full_k_verts), batch_size):
        batch_v  = full_k_verts[start : start + batch_size]
        B        = len(batch_v)
        batch_nc = torch.from_numpy(ncols_for_v[batch_v].astype(np.int64)).to(device)  # (B, k)

        # ── Compute raw RDMs for every subject: (N, B, n_pairs) ──────────────
        raw_rdms = []
        for sub_f in fmri_t:
            hood = sub_f[batch_nc]                              # (B, k, n_bins)
            hood = hood.permute(0, 2, 1).float()                # (B, n_bins, k)
            mu   = hood.mean(dim=2, keepdim=True)
            hc   = hood - mu
            nm   = hc.norm(dim=2, keepdim=True).clamp(min=1e-10)
            hn   = hc / nm
            rdm  = torch.bmm(hn, hn.permute(0, 2, 1))          # (B, n_bins, n_bins)
            raw_rdms.append((1.0 - rdm)[:, tril_r, tril_c])    # (B, n_pairs)

        raw_rdms = torch.stack(raw_rdms, dim=0)                 # (N, B, n_pairs)

        # ── Group mean of raw RDMs ────────────────────────────────────────────
        group_mean = raw_rdms.mean(dim=0)                       # (B, n_pairs)

        # ── Rank individual RDMs (for Spearman) ──────────────────────────────
        if method == "spearman":
            ranked_rdms = _rank_rows(raw_rdms)                  # (N, B, n_pairs)
            ranked_gm   = _rank_rows(group_mean)                # (B, n_pairs)
        else:
            ranked_rdms = raw_rdms
            ranked_gm   = group_mean

        # ── NC_upper: Pearson(rank(RDM_i), rank(group_mean)) avg over i ──────
        # Broadcast ranked_gm to (N, B, n_pairs)
        nc_up = _pearson_rows(
            ranked_rdms,
            ranked_gm.unsqueeze(0).expand_as(ranked_rdms),
        )                                                        # (N, B)
        nc_upper[batch_v] = nc_up.mean(dim=0).cpu().numpy()

        # ── NC_lower: LOO mean per subject ────────────────────────────────────
        nc_lo_sum = torch.zeros(B, device=device)
        for i in range(N):
            loo_mean = (N * group_mean - raw_rdms[i]) / (N - 1)   # (B, n_pairs)
            if method == "spearman":
                ranked_loo = _rank_rows(loo_mean)                  # (B, n_pairs)
            else:
                ranked_loo = loo_mean
            nc_lo_sum += _pearson_rows(ranked_rdms[i], ranked_loo) # (B,)

        nc_lower[batch_v] = (nc_lo_sum / N).cpu().numpy()

    # ── CPU fallback for partial-k border vertices ────────────────────────────
    if len(partial_k_verts) > 0:
        log.info(f"  CPU fallback: {len(partial_k_verts):,} partial-k vertices ...")
        fmri_cpu = [f.copy() for f in fmri_subs]   # list of (n_bins, n_hem_verts)
        tril_np  = np.tril_indices(n_bins, k=-1)

        for v in partial_k_verts:
            sv = int(surface_indices[v])
            nc = vertex_to_col[neighbors[sv]]
            nc = nc[nc >= 0]
            if len(nc) < 2:
                continue

            # Raw RDMs from each subject
            raw_list = []
            for f in fmri_cpu:
                hood = f[:, nc].astype(np.float64)              # (n_bins, k)
                mu   = hood.mean(axis=1, keepdims=True)
                hc   = hood - mu
                nm   = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
                nm[nm < 1e-10] = 1.0
                hn   = hc / nm
                flat = (1.0 - (hn @ hn.T))[tril_np].astype(np.float32)
                raw_list.append(flat)

            raw_arr  = np.stack(raw_list, axis=0)               # (N, n_pairs)
            gm       = raw_arr.mean(axis=0)                     # (n_pairs,)

            def _rank_1d(x):
                return np.argsort(np.argsort(x)).astype(np.float32)

            def _pearson_1d(a, b):
                ac, bc = a - a.mean(), b - b.mean()
                d = np.linalg.norm(ac) * np.linalg.norm(bc)
                return float(np.dot(ac, bc) / d) if d > 1e-10 else 0.0

            r_gm = _rank_1d(gm) if method == "spearman" else gm

            nc_up_sum = nc_lo_sum_cpu = 0.0
            for i in range(N):
                ri = _rank_1d(raw_arr[i]) if method == "spearman" else raw_arr[i]
                nc_up_sum += _pearson_1d(ri, r_gm)

                loo = (N * gm - raw_arr[i]) / (N - 1)
                r_loo = _rank_1d(loo) if method == "spearman" else loo
                nc_lo_sum_cpu += _pearson_1d(ri, r_loo)

            nc_upper[v] = nc_up_sum / N
            nc_lower[v] = nc_lo_sum_cpu / N

    return nc_upper, nc_lower


def _compute_nc_cpu(
    fmri_subs: list[np.ndarray],
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    method: str,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-CPU fallback using joblib."""
    from joblib import Parallel, delayed

    N       = len(fmri_subs)
    n_verts = fmri_subs[0].shape[1]
    n_bins  = fmri_subs[0].shape[0]
    tril    = np.tril_indices(n_bins, k=-1)

    def _rank_1d(x):
        return np.argsort(np.argsort(x)).astype(np.float32)

    def _pearson_1d(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        return float(np.dot(ac, bc) / d) if d > 1e-10 else 0.0

    def _one_vertex(v: int) -> tuple[float, float]:
        sv = int(surface_indices[v])
        nc = vertex_to_col[neighbors[sv]]
        nc = nc[nc >= 0]
        if len(nc) < 2:
            return 0.0, 0.0

        raw_list = []
        for f in fmri_subs:
            hood = f[:, nc].astype(np.float64)
            mu   = hood.mean(axis=1, keepdims=True)
            hc   = hood - mu
            nm   = np.sqrt((hc ** 2).sum(axis=1, keepdims=True))
            nm[nm < 1e-10] = 1.0
            hn   = hc / nm
            raw_list.append((1.0 - (hn @ hn.T))[tril].astype(np.float32))

        raw  = np.stack(raw_list, axis=0)   # (N, n_pairs)
        gm   = raw.mean(axis=0)
        r_gm = _rank_1d(gm) if method == "spearman" else gm

        nc_up_s = nc_lo_s = 0.0
        for i in range(N):
            ri    = _rank_1d(raw[i]) if method == "spearman" else raw[i]
            nc_up_s += _pearson_1d(ri, r_gm)
            loo    = (N * gm - raw[i]) / (N - 1)
            r_loo  = _rank_1d(loo) if method == "spearman" else loo
            nc_lo_s += _pearson_1d(ri, r_loo)

        return nc_up_s / N, nc_lo_s / N

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_vertex)(v) for v in range(n_verts)
    )
    nc_upper = np.array([r[0] for r in results], dtype=np.float32)
    nc_lower = np.array([r[1] for r in results], dtype=np.float32)
    return nc_upper, nc_lower


def _dispatch_hemisphere(
    hem: str,
    fmri_subs_hem: list[np.ndarray],
    surf_path: str,
    surf_indices: np.ndarray,
    workbench: str,
    subject: str,
    k: int,
    cache_dir: Path,
    method: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load neighbors, build vertex→col map, dispatch GPU or CPU."""
    log.info(f"  Hemisphere: {hem}  ({fmri_subs_hem[0].shape[1]} vertices, "
             f"{fmri_subs_hem[0].shape[0]} bins, {len(fmri_subs_hem)} subjects)")

    neighbors = get_neighbors(surf_path, workbench, subject, hem, k, cache_dir)
    n_surf    = neighbors.shape[0]
    v2c       = np.full(n_surf, -1, dtype=np.int32)
    v2c[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)

    try:
        import torch as _torch
        if _torch.cuda.is_available():
            try:
                return _compute_nc_gpu(
                    fmri_subs_hem, neighbors, surf_indices, v2c,
                    method=method, batch_size=batch_size, device="cuda",
                )
            except _torch.cuda.OutOfMemoryError:
                log.warning("  GPU OOM — retrying with half batch size")
                _torch.cuda.empty_cache()
                return _compute_nc_gpu(
                    fmri_subs_hem, neighbors, surf_indices, v2c,
                    method=method, batch_size=max(1, batch_size // 2), device="cuda",
                )
        else:
            log.info("  CUDA not available — using CPU")
    except ImportError:
        log.info("  torch not installed — using CPU")

    return _compute_nc_cpu(fmri_subs_hem, neighbors, surf_indices, v2c, method=method)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    N = len(args.subjects)
    bin_int  = int(args.bin_sec)
    skip_int = int(args.skip_sec)
    config   = (f"k{args.k}_delay{int(args.delay_sec)}s"
                f"_bin{bin_int}s_skip{skip_int}s_{args.method}")

    log.info("=" * 70)
    log.info(f"Noise ceiling  N={N} subjects  {config}")
    log.info(f"  subjects: {args.subjects}")
    log.info("=" * 70)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_ceiling_{N}subs_{config}.dscalar.nii"

    if out_path.exists():
        log.info(f"Already exists: {out_path.name} — skipping.")
        return

    # ── Load + bin fMRI for all subjects ──────────────────────────────────────
    timing_df = pd.read_csv(args.timing_csv)
    fmri_all  = []
    for sub in args.subjects:
        log.info(f"Loading {sub} ...")
        binned = _load_and_bin(
            sub, args.preprocessed_root, args.fmri_suffix,
            timing_df, args.bin_sec, args.skip_sec, args.delay_sec, args.tr,
        )
        log.info(f"  {sub}: binned shape = {binned.shape}")
        fmri_all.append(binned)   # (n_bins, n_verts_total)

    n_bins  = fmri_all[0].shape[0]
    n_total = fmri_all[0].shape[1]
    log.info(f"n_bins={n_bins}  n_pairs={n_bins*(n_bins-1)//2:,}  n_grays={n_total:,}")

    # ── Grayordinate split ─────────────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    cache_dir = (Path(args.geodesic_cache_dir) if args.geodesic_cache_dir
                 else out_dir / "_geodesic_cache")

    nc_full_upper = np.zeros(n_total, dtype=np.float32)
    nc_full_lower = np.zeros(n_total, dtype=np.float32)

    for hem, surf_path, h_indices, col_start in [
        ("left",  args.left_surface,  left_indices,  0),
        ("right", args.right_surface, right_indices, n_left),
    ]:
        n_hem = len(h_indices)
        fmri_hem = [f[:, col_start : col_start + n_hem] for f in fmri_all]

        nc_up, nc_lo = _dispatch_hemisphere(
            hem, fmri_hem, surf_path, h_indices,
            workbench=args.workbench,
            subject=args.subjects[0],
            k=args.k, cache_dir=cache_dir,
            method=args.method, batch_size=args.batch_size,
        )
        nc_full_upper[col_start : col_start + n_hem] = nc_up
        nc_full_lower[col_start : col_start + n_hem] = nc_lo

        log.info(f"  [{hem}] NC_upper: [{nc_up.min():.3f}, {nc_up.max():.3f}]  "
                 f"NC_lower: [{nc_lo.min():.3f}, {nc_lo.max():.3f}]")
        del fmri_hem, nc_up, nc_lo
        gc.collect()

    # ── Save CIFTI ─────────────────────────────────────────────────────────────
    maps = np.stack([nc_full_lower, nc_full_upper], axis=0)
    map_names = ["nc_lower", "nc_upper"]
    save_cifti_multimap(maps, map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path.name}")
    log.info(
        f"Done. NC_lower=[{nc_full_lower.min():.3f}, {nc_full_lower.max():.3f}]  "
        f"NC_upper=[{nc_full_upper.min():.3f}, {nc_full_upper.max():.3f}]"
    )


if __name__ == "__main__":
    main()
