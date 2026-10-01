"""
Vertex-wise classical partial RSA.

For every cortical searchlight, correlate the target and brain RDM upper
triangles after ordinary least-squares residualisation against the configured
nuisance-model RDMs.  With ``--method spearman`` (the project default), every
RDM vector is rank transformed before residualisation, so the result is a
classical partial Spearman correlation.  No ridge penalty is used.

The script supports the same disk and streaming fMRI inputs as searchlight.py
and can save temporal block maps for the corrected two-factor bootstrap used by
group_stats.py.
"""

import argparse
import gc
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap
from rsa.shared.model_registry import (
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    PARTIAL_RSA_RUNS,
    TR_DEFAULT,
    check_embeddings_exist,
    validate_run,
)
from rsa.shared.rsa_utils import (
    align_and_assert_bins,
    compute_rdm,
    get_run_bin_counts,
    load_fmri_cifti,
    preprocess_fmri,
    process_model_embeddings,
)

# Reuse the established geodesic-neighbour cache implementation.
sys.path.insert(0, str(Path(__file__).parent))
from searchlight import get_neighbors  # noqa: E402
from rsa.shared.naming import add_model_norm_arg, searchlight_config, searchlight_stem  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Vertex-wise classical partial RSA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", required=True,
                   help=f"Partial RSA run key. Available: {list(PARTIAL_RSA_RUNS.keys())}")

    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="Directory containing {subject}_{fmri_suffix}_cortex_59k.dtseries.nii.")
    inp.add_argument("--raw-dir", default=None,
                     help="Raw per-run HCP CIFTI directory for streaming preprocessing.")
    inp.add_argument("--fmri-suffix", default="raw", dest="fmri_suffix")

    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", default="group_average")
    p.add_argument("--bin-sec", type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec", type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr", type=float, default=TR_DEFAULT)
    p.add_argument("--k", type=int, required=True)
    add_model_norm_arg(p)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"],
                   help="Partial Spearman ranks all RDM vectors; Pearson uses raw distances.")
    p.add_argument("--left-surface", required=True)
    p.add_argument("--right-surface", required=True)
    p.add_argument("--workbench", required=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir")
    p.add_argument("--gpu-batch-size", type=int, default=256, dest="gpu_batch_size")
    p.add_argument("--n-jobs", type=int, default=-1, dest="n_jobs")
    p.add_argument("--n-blocks", type=int, default=1, dest="n_blocks",
                   help="Non-overlapping temporal blocks saved for C2F bootstrap; 1 disables.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even when the full and requested block maps exist.")

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true")
    prep.add_argument("--psc", default=False, action="store_true")
    prep.add_argument("--gsr", default=False, action=argparse.BooleanOptionalAction)
    return p.parse_args()


def _fmri_tag(args) -> str:
    if not args.raw_dir:
        return args.fmri_suffix
    parts = []
    if args.sg_filter:
        parts.append("sg")
    if args.psc:
        parts.append("psc")
    if args.gsr:
        parts.append("gsr")
    return "_".join(parts) if parts else "raw"


def _rdm_upper_tri(emb: np.ndarray, method: str = "spearman") -> np.ndarray:
    """Return the strict upper triangle of a correlation-distance RDM.

    For partial Spearman, average ranks are used so tied distances receive the
    conventional tied rank.  Pearson retains the raw distance values.
    """
    rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    vec = rdm[np.triu_indices(rdm.shape[0], k=1)].astype(np.float64)
    if method == "spearman":
        vec = rankdata(vec, method="average")
    return vec.astype(np.float32)


def fit_ols_projection(X_nuis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an intercept-inclusive OLS nuisance design and its pseudoinverse."""
    design = np.column_stack([
        np.ones(X_nuis.shape[0], dtype=np.float64),
        X_nuis.astype(np.float64),
    ])
    projection = np.linalg.pinv(design)
    return design.astype(np.float32), projection.astype(np.float32)


def residualize(y: np.ndarray, design: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Return OLS residuals from the intercept-inclusive nuisance design."""
    y = np.asarray(y, dtype=np.float32)
    return y - design @ (projection @ y)


def partial_corr(y_brain: np.ndarray, e_target: np.ndarray,
                 design: np.ndarray, projection: np.ndarray) -> float:
    """Classical partial correlation between brain and target RDM vectors."""
    e_brain = residualize(y_brain, design, projection)
    denom = np.linalg.norm(e_brain) * np.linalg.norm(e_target)
    if not np.isfinite(denom) or denom <= 1e-10:
        return 0.0
    return float(np.dot(e_brain, e_target) / denom)


def _rank_rows_average_torch(values):
    """Average-rank each row of a torch tensor, including exact ties."""
    import torch

    sorted_values, order = torch.sort(values, dim=1)
    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    groups = starts.cumsum(dim=1) - 1
    positions = torch.arange(
        1, values.shape[1] + 1, dtype=values.dtype, device=values.device
    ).unsqueeze(0).expand_as(values)
    sums = torch.zeros_like(values).scatter_add_(1, groups, positions)
    counts = torch.zeros_like(values).scatter_add_(1, groups, torch.ones_like(values))
    sorted_ranks = (sums / counts.clamp_min(1)).gather(1, groups)
    ranks = torch.empty_like(values)
    ranks.scatter_(1, order, sorted_ranks)
    return ranks


def run_partial_searchlight_gpu(
    fmri: np.ndarray,
    design: np.ndarray,
    projection: np.ndarray,
    e_target: np.ndarray,
    neighbors: np.ndarray,
    surface_indices: np.ndarray,
    vertex_to_col: np.ndarray,
    method: str = "spearman",
    batch_size: int = 256,
    n_jobs: int = -1,
    device: str = "cuda",
) -> np.ndarray:
    """GPU-batched partial RSA for full-k vertices, with exact CPU fallback."""
    import torch

    n_verts, n_bins = fmri.shape[1], fmri.shape[0]
    triu_idx = np.triu_indices(n_bins, k=1)
    triu_row = torch.tensor(triu_idx[0], dtype=torch.long, device=device)
    triu_col = torch.tensor(triu_idx[1], dtype=torch.long, device=device)

    design_t = torch.from_numpy(design).to(device)
    projection_t = torch.from_numpy(projection).to(device)
    e_target_t = torch.from_numpy(e_target).to(device)
    e_target_t = e_target_t / e_target_t.norm().clamp_min(1e-10)
    fmri_t = torch.from_numpy(fmri.T.astype(np.float32)).to(device)

    neighbor_cols = vertex_to_col[neighbors][surface_indices.astype(np.int32)]
    full_mask = np.all(neighbor_cols >= 0, axis=1)
    full_verts = np.where(full_mask)[0]
    partial_verts = np.where(~full_mask)[0]
    log.info("  GPU: %s full-k vertices (batch=%s), %s boundary vertices",
             f"{len(full_verts):,}", batch_size, f"{len(partial_verts):,}")

    corr_map = np.zeros(n_verts, dtype=np.float32)
    for start in range(0, len(full_verts), batch_size):
        batch_v = full_verts[start:start + batch_size]
        cols = torch.from_numpy(neighbor_cols[batch_v].astype(np.int64)).to(device)
        hood = fmri_t[cols].permute(0, 2, 1).float()
        centered = hood - hood.mean(dim=2, keepdim=True)
        normalized = centered / torch.linalg.norm(centered, dim=2, keepdim=True).clamp_min(1e-10)
        rdm = 1.0 - torch.bmm(normalized, normalized.transpose(1, 2))
        brain = rdm[:, triu_row, triu_col]
        if method == "spearman":
            brain = _rank_rows_average_torch(brain)

        coefficients = projection_t @ brain.T
        residuals = brain - (design_t @ coefficients).T
        residuals = residuals / torch.linalg.norm(residuals, dim=1, keepdim=True).clamp_min(1e-10)
        corr_map[batch_v] = (residuals * e_target_t.unsqueeze(0)).sum(dim=1).cpu().numpy()

    if len(partial_verts):
        corr_map[partial_verts] = _run_partial_vertices_cpu(
            fmri, design, projection, e_target, neighbors, surface_indices,
            vertex_to_col, partial_verts, method, n_jobs=n_jobs,
        )
    return corr_map


def _run_partial_vertices_cpu(
    fmri, design, projection, e_target, neighbors, surface_indices,
    vertex_to_col, vertices, method, n_jobs,
) -> np.ndarray:
    from joblib import Parallel, delayed

    triu_idx = np.triu_indices(fmri.shape[0], k=1)

    def one_vertex(v):
        surface_vertex = int(surface_indices[v])
        cols = vertex_to_col[neighbors[surface_vertex]]
        cols = cols[cols >= 0]
        if len(cols) < 2:
            return 0.0
        hood = fmri[:, cols].astype(np.float64)
        centered = hood - hood.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        brain = (1.0 - (centered / norms) @ (centered / norms).T)[triu_idx]
        if method == "spearman":
            brain = rankdata(brain, method="average")
        return partial_corr(brain.astype(np.float32), e_target, design, projection)

    return np.asarray(
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one_vertex)(int(v)) for v in vertices),
        dtype=np.float32,
    )


def run_partial_searchlight(
    fmri, design, projection, e_target, neighbors, surface_indices,
    vertex_to_col, method="spearman", batch_size=256, n_jobs=-1,
) -> np.ndarray:
    try:
        import torch
        if torch.cuda.is_available():
            try:
                return run_partial_searchlight_gpu(
                    fmri, design, projection, e_target, neighbors,
                    surface_indices, vertex_to_col, method, batch_size, n_jobs,
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("GPU OOM; retrying partial RSA on CPU")
                torch.cuda.empty_cache()
    except ImportError:
        pass

    log.info("  CUDA unavailable; running partial RSA on CPU with n_jobs=%s", n_jobs)
    vertices = np.arange(fmri.shape[1], dtype=np.int32)
    return _run_partial_vertices_cpu(
        fmri, design, projection, e_target, neighbors, surface_indices,
        vertex_to_col, vertices, method, n_jobs,
    )


def _prepare_model_partial(target_emb, nuisance_embs, method):
    target = _rdm_upper_tri(target_emb, method)
    nuisance = np.column_stack([
        _rdm_upper_tri(embedding, method) for embedding in nuisance_embs
    ]).astype(np.float32)
    design, projection = fit_ols_projection(nuisance)
    e_target = residualize(target, design, projection)
    return design, projection, e_target


def _searchlight_map(args, fmri, target_emb, nuisance_embs, surfaces, cache_dir):
    design, projection, e_target = _prepare_model_partial(
        target_emb, nuisance_embs, args.method
    )
    n_total = fmri.shape[1]
    result = np.zeros(n_total, dtype=np.float32)
    offset = 0
    for hem, (surface_path, surface_indices) in surfaces.items():
        n_hem = len(surface_indices)
        fmri_hem = fmri[:, offset:offset + n_hem]
        neighbors = get_neighbors(
            surface_path, args.workbench, args.subject, hem, args.k, cache_dir
        )
        surface_indices = surface_indices.astype(np.int32)
        vertex_to_col = np.full(neighbors.shape[0], -1, dtype=np.int32)
        vertex_to_col[surface_indices] = np.arange(n_hem, dtype=np.int32)
        result[offset:offset + n_hem] = run_partial_searchlight(
            fmri_hem, design, projection, e_target, neighbors,
            surface_indices, vertex_to_col, args.method,
            args.gpu_batch_size, args.n_jobs,
        )
        offset += n_hem
        del neighbors, vertex_to_col
        gc.collect()
    return result


def _load_fmri(args):
    if args.raw_dir:
        if args.subject == "group_average":
            raise ValueError("Streaming mode requires an individual subject, not group_average")
        from preprocess_individual import preprocess_subject
        prep_args = types.SimpleNamespace(
            sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr
        )
        data, _bm_axis, run_trs = preprocess_subject(
            args.subject, Path(args.raw_dir), args.tr, prep_args
        )
        return data.astype(np.float32, copy=False), run_trs

    base = Path(args.preprocessed_dir)
    cifti = base / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    run_trs_path = base / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    return load_fmri_cifti(str(cifti)), np.load(str(run_trs_path))


def run_analysis(args):
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    if bool(args.preprocessed_dir) == bool(args.raw_dir):
        raise ValueError("Specify exactly one of --preprocessed-dir or --raw-dir")
    if args.n_blocks < 1:
        raise ValueError("--n-blocks must be at least 1")

    cfg = validate_run(args.run)
    timing_df = pd.read_csv(args.timing_csv)
    fmri_tag = _fmri_tag(args)
    config = searchlight_config(args.k, args.delay_sec, args.bin_sec, args.skip_sec,
                                args.method, args.model_norm)
    subject_root = (
        Path(args.output_dir) / "group_average"
        if args.subject == "group_average"
        else Path(args.output_dir) / "subject_data" / args.subject
    )
    out_dir = subject_root / cfg.label / config
    full_npy = out_dir / (
        searchlight_stem(fmri_tag, args.k, args.delay_sec, args.bin_sec, args.skip_sec,
                         args.method, args.model_norm) + "_searchlight.npy"
    )
    blocks_npy = full_npy.with_name(
        full_npy.name.replace("_searchlight.npy", f"_searchlight_nblocks{args.n_blocks}.npy")
    )
    if (not args.force and full_npy.exists()
            and (args.n_blocks == 1 or blocks_npy.exists())):
        log.info("[%s] Full and requested block maps already exist; skipping", args.subject)
        return

    log.info("Classical partial %s RSA: %s, subject=%s", args.method, args.run, args.subject)
    log.info("Target=%s nuisance=%s", cfg.target, cfg.nuisance)
    fmri_raw, run_trs = _load_fmri(args)
    fmri_binned = preprocess_fmri(
        fmri_raw, timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec,
        skip_sec=args.skip_sec,
    )
    del fmri_raw
    gc.collect()

    def load_embedding(model, modality):
        path = check_embeddings_exist(
            args.embeddings_dir, model, modality, args.bin_sec, args.skip_sec
        )
        embedding = process_model_embeddings(
            str(path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
            model_norm=args.model_norm,
        )
        _, embedding = align_and_assert_bins(fmri_binned, embedding)
        return embedding

    target_emb = load_embedding(*cfg.target)
    nuisance_embs = [load_embedding(*item) for item in cfg.nuisance]
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    surfaces = {
        "left": (args.left_surface, left_indices),
        "right": (args.right_surface, right_indices),
    }
    cache_dir = (Path(args.geodesic_cache_dir) if args.geodesic_cache_dir
                 else Path(args.output_dir) / "_geodesic_cache")

    corr_full = _searchlight_map(
        args, fmri_binned, target_emb, nuisance_embs, surfaces, cache_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(full_npy), corr_full)

    if args.n_blocks > 1:
        n_bins = fmri_binned.shape[0]
        if n_bins < args.n_blocks * 6:
            raise ValueError(
                f"Too few bins ({n_bins}) for {args.n_blocks} blocks; need at least six per block"
            )
        if args.n_blocks == len(run_trs):
            counts = get_run_bin_counts(
                timing_df, run_trs, args.bin_sec, args.tr,
                args.delay_sec, args.skip_sec,
            )
            edges = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        else:
            edges = np.round(np.linspace(0, n_bins, args.n_blocks + 1)).astype(int)

        block_maps = np.zeros((args.n_blocks, corr_full.size), dtype=np.float32)
        for block, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            log.info("Block %d/%d: bins [%d,%d)", block + 1, args.n_blocks, start, stop)
            block_maps[block] = _searchlight_map(
                args,
                fmri_binned[start:stop],
                target_emb[start:stop],
                [embedding[start:stop] for embedding in nuisance_embs],
                surfaces,
                cache_dir,
            )
        np.save(str(blocks_npy), block_maps)
        log.info("Saved block maps: %s shape=%s", blocks_npy, block_maps.shape)

    leaf = ("integration_partial_r_searchlight.dscalar.nii"
            if cfg.kind == "integration" else "partial_corr_r_searchlight.dscalar.nii")
    cifti_path = out_dir / leaf
    save_cifti_multimap(
        corr_full.reshape(1, -1),
        [f"partial_{args.method}_{cfg.label}"],
        args.template_cifti,
        str(cifti_path),
    )
    log.info("Saved full map: %s mean=%.5f max=%.5f", cifti_path,
             float(corr_full.mean()), float(corr_full.max()))


def main():
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
