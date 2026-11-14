"""
rsa/searchlight.py
===================================
Vertex-wise searchlight RSA on HCP 7T movie-watching fMRI data.

For each vertex, a geodesic neighbourhood of k nearest vertices is assembled.
The fMRI RDM within that neighbourhood is correlated (Spearman or Pearson)
with the model RDM. Output: {output}/group_average/{model}/{stem}_joint_only_models.dscalar.nii
({x}_only for a modality other than av) with rho_j and its sigmaps and FDR mask; per subject,
{output}/subject_data/{subject}/{model}/ with rho_j and the block maps rho_j_block01 ...

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
  python searchlight.py \
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
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy import stats
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "deprecated")]
from rsa.shared.rsa_utils import (
    load_fmri_cifti, preprocess_fmri, get_run_bin_counts,
    process_model_embeddings, align_and_assert_bins,
    assert_segment_timing,
)
from shared.naming import (
    add_model_norm_arg, rsa_method, searchlight_config, searchlight_path, searchlight_stem, single_model,
)
from shared.cifti_io import (
    get_bm_axis, get_cortex_vertex_indices, get_combined_map_names, save_cifti_map, save_cifti_multimap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REPEATED_CLIPS = ("video5", "video9", "video14", "video18")

# n_jobs for joblib parallelism inside each subject's searchlight.
# Read from env var _RSA_N_JOBS (set by analysis.sh based on BATCH_SIZE).
# Default -1 (all CPUs) is fine when running a single subject, but MUST be
# reduced when many subjects run in parallel to avoid thread oversubscription.
# Rule of thumb: _RSA_N_JOBS = max(1, nproc // BATCH_SIZE)
N_JOBS = int(os.environ.get("_RSA_N_JOBS", -1))

from shared.searchlight import (  # noqa: F401
    _DCONN_CHUNK,
    _compute_geodesic_dconn,
    _dconn_is_complete,
    _extract_knn_from_dconn,
    _fdr_sigmap,
    _parse_k_from_npy,
    _precompute_model_rdm,
    _rho_sigmap,
    _searchlight_vertex_fast,
    get_neighbors,
    run_searchlight,
    run_searchlight_gpu,
)


# =============================================================================
# Group-average significance helpers
# =============================================================================


def _write_border_file(
    cluster_mask: np.ndarray,
    cifti_vertex_indices: np.ndarray,
    n_full_verts: int,
    surface_path: str,
    out_border_path: str,
    workbench: str,
    class_name: str = "fdr",
) -> None:
    """Expand cluster_mask to full surface and trace a border with wb_command."""
    full_mask = np.zeros(n_full_verts, dtype=np.float32)
    full_mask[cifti_vertex_indices] = cluster_mask

    tmp_metric = Path(out_border_path).with_suffix(".tmp.func.gii")
    arr = GiftiDataArray(data=full_mask, intent=0, datatype="NIFTI_TYPE_FLOAT32")
    nib.save(GiftiImage(darrays=[arr]), str(tmp_metric))
    try:
        subprocess.run(
            [workbench, "-metric-rois-to-border",
             surface_path, str(tmp_metric), class_name, out_border_path],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning(f"wb_command border failed for {out_border_path}:\n  {e.stderr.strip()}")
    finally:
        tmp_metric.unlink(missing_ok=True)


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
    p.add_argument("--modality", required=True, choices=["v", "a", "av", "at", "vt", "avt", "t", "caption_t", "transcript_t", "event_t", "transcript_avt", "event_avt"])
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Applied when slicing stimulus blocks from continuous fMRI.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF.")
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec",
                   help="Window stride in seconds (default: bin-sec, i.e. no overlap).")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"],
                   help="RDM comparator for the GPU/CPU searchlight kernel. "
                        "Note: rho_a (Kendall's tau_a, Schütt et al. 2023 §3.5) is NOT "
                        "supported here. Spearman reduces to a rank-vector dot product, "
                        "which is fully vectorizable on GPU (thousands of vertices in parallel). "
                        "Kendall's tau requires pairwise concordance counting with no equivalent "
                        "GPU batch formulation — scipy's O(n log n) CPU path is still too slow "
                        "for 60k vertices × ~1M pairs each. Use spearman for the searchlight; "
                        "rho_a is used at the parcel level (glasser.py) where it is feasible "
                        "and the noise-ceiling advantage matters most.")
    p.add_argument("--drop-repeated-clips", action="store_true", dest="drop_repeated_clips",
                   help="Drop the windows of the repeated clips after per-run normalization "
                        "(diagnostic map; group-average disk mode only).")
    p.add_argument("--n-blocks", type=int, default=4, dest="n_blocks",
                   help="Number of non-overlapping temporal blocks for the condition bootstrap "
                        "(Schütt et al. 2023, Eq. 5). Each block gets its own RSA map "
                        "(rho_x_block01 ...) in the output file. Set to 1 to disable.")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir",
                   help="Shared directory for geodesic k-NN .npy caches (preprocessing-"
                        "agnostic). Defaults to {output_dir}/_geodesic_cache if not set. "
                        "All analysis variants should point here so the k_max derivation "
                        "logic works across preprocessing configs.")

    p.add_argument("--gpu-batch-size", type=int, default=512, dest="gpu_batch_size",
                   help="Vertices per GPU batch (default 512; reduce if GPU OOM).")
    add_model_norm_arg(p)

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true")
    prep.add_argument("--psc", default=False, action="store_true")
    prep.add_argument("--gsr", default=False, action=argparse.BooleanOptionalAction)

    return p.parse_args()


# =============================================================================
# Geodesic distance — two-level cache
# =============================================================================


# =============================================================================
# Searchlight RSA
# =============================================================================


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
    return searchlight_config(args.k, args.delay_sec, args.bin_sec, args.skip_sec,
                              rsa_method(args.method), args.model_norm, hrf=args.hrf)


def _file_stem(args, fmri_tag: str) -> str:
    return searchlight_stem(fmri_tag, args.k, args.delay_sec, args.bin_sec, args.skip_sec,
                            rsa_method(args.method), args.model_norm)


def _streaming_fmri_tag(args) -> str:
    parts = []
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    if args.gsr:       parts.append("gsr")
    return "_".join(parts) if parts else "raw"


# =============================================================================
# Core analysis
# =============================================================================

def _output(args, fmri_tag: str):
    """Output file of the run, its map name (rho_j for av) and the maps a finished run has written."""
    stem = _file_stem(args, fmri_tag)
    root = Path(args.output_dir) / ("group_average" if args.subject == "group_average"
                                    else Path("subject_data") / args.subject)
    path = searchlight_path(root, args.model, args.modality, stem)
    name = single_model(args.modality)[1]
    if args.drop_repeated_clips:
        norepeats = "_norepeats" if args.drop_repeated_clips else ""
        return path.parent / "diagnostics" / f"{stem}{norepeats}_maps.dscalar.nii", name, {name}
    needed = {name}
    if args.subject == "group_average":
        needed.add(f"{name}_fdr_mask")
    if args.n_blocks > 1:
        needed.add(f"{name}_block{args.n_blocks:02d}")
    return path, name, needed


def _significance_maps(corr_full, n_bins, name, out_path, args) -> dict:
    """Uncorrected and BH-FDR sigmaps and the FDR mask of a group-average rho map, with FDR borders beside out_path."""
    p_uncorr, sigmap_uncorr = _rho_sigmap(corr_full, n_bins)
    sigmap_fdr, fdr_mask, n_sig_fdr = _fdr_sigmap(p_uncorr, corr_full)
    log.info(f"  n_bins={n_bins}  FDR significant (p<0.05): {n_sig_fdr:,} / {corr_full.shape[0]:,}")

    coverage_fdr = fdr_mask.mean()
    if fdr_mask.sum() == 0:
        log.info("  No FDR-significant vertices; skipping border files.")
    elif coverage_fdr > 0.90:
        log.warning(f"  FDR mask covers {coverage_fdr*100:.1f}% of cortex — skipping border files.")
    else:
        border = str(out_path).removesuffix(".dscalar.nii") + "_fdr_{}.border"
        lh_idx, rh_idx = get_cortex_vertex_indices(get_bm_axis(args.template_cifti))
        n_left = len(lh_idx)
        for hem, surf, mask, idx in (("lh", args.left_surface, fdr_mask[:n_left], lh_idx),
                                     ("rh", args.right_surface, fdr_mask[n_left:], rh_idx)):
            _write_border_file(mask, idx, nib.load(surf).darrays[0].data.shape[0], surf, border.format(hem), args.workbench)
        log.info(f"  FDR borders: {Path(border.format('{lh,rh}')).name}")
    return {f"{name}_sigmap_uncorr": sigmap_uncorr, f"{name}_sigmap_fdr": sigmap_fdr, f"{name}_fdr_mask": fdr_mask}


def _run_analysis(args, fmri_continuous: np.ndarray, run_trs: np.ndarray,
                  timing_df: pd.DataFrame, config: str, fmri_tag: str):

    bin_sec_int   = int(args.bin_sec)
    skip_int      = int(args.skip_sec)
    diagnostic    = args.drop_repeated_clips
    out_path, name, needed = _output(args, fmri_tag)
    if needed <= set(get_combined_map_names(out_path)):
        log.info(f"Output already exists — skipping: {out_path.name}")
        return

    assert_segment_timing(timing_df, args.bin_sec, args.tr,
                          args.delay_sec, args.skip_sec, run_trs)

    fmri_binned = preprocess_fmri(
        fmri_continuous, timing_df, run_trs, args.bin_sec, args.tr,
        args.delay_sec, skip_sec=args.skip_sec, normalize=True,
    )
    log.info(f"  fMRI binned z-scored: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"bin{bin_sec_int}s_skip{skip_int}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")

    emb = process_model_embeddings(
        str(emb_file), timing_df, bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, hrf=args.hrf,
        skip_sec=args.skip_sec, model_norm=args.model_norm,
    )
    log.info(f"  Model binned ({args.model_norm}): {emb.shape}")

    # Enforce exact temporal alignment
    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)
    if args.drop_repeated_clips:
        from cka.shared.kernels import window_index
        videos, _ = window_index(timing_df, run_trs, args.bin_sec, args.skip_sec, args.delay_sec, args.tr)
        assert len(videos) == fmri_binned.shape[0], (len(videos), fmri_binned.shape)
        keep = ~np.isin(videos, REPEATED_CLIPS)
        fmri_binned, emb = fmri_binned[keep], emb[keep]
    n_bins = fmri_binned.shape[0]
    log.info(f"  n_bins={n_bins}  n_pairs={(n_bins*(n_bins-1)//2):,}")

    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    # Shared geodesic cache: use explicit override if provided, else fall back to
    # the preprocessing-specific subdir.  All callers in analysis.sh pass
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if diagnostic:
        save_cifti_map(corr_full, args.template_cifti, str(out_path), name)
        log.info(f"  Saved: {out_path.name}  mean_r={corr_full.mean():.4f}  max_r={corr_full.max():.4f}")
        return
    maps = {name: corr_full}
    if args.subject == "group_average":
        maps.update(_significance_maps(corr_full, n_bins, name, out_path, args))

    # ── Block RSA pass (Schütt et al. 2023, Eq. 5 condition bootstrap) ──────
    # Split the time series into n_blocks non-overlapping segments and compute
    # a separate RSA correlation per block per vertex.  group_stats.py uses
    # these to run the corrected 2-factor bootstrap (subjects × time segments)
    # which generalizes to both new subjects AND new movie content.
    n_blocks = args.n_blocks
    if n_blocks > 1:
        n_bins_total = fmri_binned.shape[0]
        if n_bins_total < n_blocks * 6:
            log.warning(
                f"  Too few bins ({n_bins_total}) for {n_blocks} blocks "
                f"(need ≥{n_blocks * 6}); skipping block pass."
            )
        else:
            n_runs = len(run_trs)
            if n_blocks == n_runs:
                # Align blocks to actual fMRI run boundaries (best practice per
                # Schütt et al. 2023: blocks should correspond to independent
                # data acquisition segments, not arbitrary equal-length splits).
                per_run_counts = get_run_bin_counts(
                    timing_df, run_trs, args.bin_sec, args.tr,
                    args.delay_sec, args.skip_sec,
                )
                block_edges = np.concatenate([[0], np.cumsum(per_run_counts)]).astype(int)
                log.info(
                    f"  Block RSA: {n_blocks} blocks aligned to run boundaries "
                    f"(bins per run: {per_run_counts.tolist()})"
                )
            else:
                block_edges = np.round(
                    np.linspace(0, n_bins_total, n_blocks + 1)
                ).astype(int)
                log.info(
                    f"  Block RSA: {n_blocks} blocks of ~{n_bins_total // n_blocks} bins each"
                )
            block_maps = np.zeros((n_blocks, n_total), dtype=np.float32)
            for b, (b_start, b_end) in enumerate(
                zip(block_edges[:-1], block_edges[1:])
            ):
                if b_end - b_start < 6:
                    log.warning(
                        f"  Block {b}: only {b_end - b_start} bins — skipping."
                    )
                    continue
                fmri_block = fmri_binned[b_start:b_end]
                emb_block  = emb[b_start:b_end]
                blk_offset = 0
                for hem, (surf_path, _, surf_indices) in surfaces.items():
                    neighbors_blk = get_neighbors(
                        surf_path, args.workbench, args.subject,
                        hem, args.k, cache_dir,
                    )
                    surf_indices_blk = surf_indices.astype(np.int32)
                    n_surf_verts_blk = neighbors_blk.shape[0]
                    vtc_blk = np.full(n_surf_verts_blk, -1, dtype=np.int32)
                    vtc_blk[surf_indices_blk] = np.arange(
                        len(surf_indices_blk), dtype=np.int32
                    )
                    n_hem_blk = len(surf_indices_blk)
                    fmri_hem_blk = fmri_block[:, blk_offset:blk_offset + n_hem_blk]
                    corr_hem_blk = run_searchlight(
                        fmri_hem_blk, emb_block, neighbors_blk,
                        surface_indices=surf_indices_blk,
                        vertex_to_col=vtc_blk,
                        method=args.method, n_jobs=N_JOBS,
                        batch_size=args.gpu_batch_size,
                    )
                    block_maps[b, blk_offset:blk_offset + n_hem_blk] = corr_hem_blk
                    blk_offset += n_hem_blk
                    del neighbors_blk, vtc_blk, corr_hem_blk
                    gc.collect()
                log.info(
                    f"  Block {b}: bins [{b_start},{b_end})  "
                    f"mean_r={block_maps[b].mean():.4f}"
                )
            maps.update({f"{name}_block{b + 1:02d}": block_maps[b] for b in range(n_blocks)})

    del fmri_binned, emb
    gc.collect()

    save_cifti_multimap(np.stack(list(maps.values())), list(maps), args.template_cifti, str(out_path))
    log.info(f"  Saved: {out_path.name}  mean_r={corr_full.mean():.4f}  max_r={corr_full.max():.4f}")

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

    cifti = _cifti_path(args)
    trs_path = _run_trs_path(args)

    log.info(f"Searchlight RSA: {args.model}/{args.modality}/{config}")
    log.info(f"  fMRI: {cifti}")
    
    fmri_data = load_fmri_cifti(cifti)
    run_trs = np.load(trs_path)
    
    log.info(f"  fMRI loaded: {fmri_data.shape}  run_trs: {run_trs.tolist()}")

    _run_analysis(args, fmri_data, run_trs, timing_df, config, fmri_tag)
    log.info("Done.")


# =============================================================================
# Streaming mode
# =============================================================================

def _run_streaming(args):
    from preprocessing.preprocess_individual import preprocess_subject

    timing_df   = pd.read_csv(args.timing_csv)
    config      = _config_label(args)
    fmri_tag    = _streaming_fmri_tag(args)
    sub         = args.subject

    out_path, _, needed = _output(args, fmri_tag)
    if needed <= set(get_combined_map_names(out_path)):
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

    _run_analysis(args, data, run_trs, timing_df, config, fmri_tag)
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

    if args.raw_dir and args.drop_repeated_clips:
        log.error("--drop-repeated-clips needs --preprocessed-dir.")
        sys.exit(1)

    if args.raw_dir:
        _run_streaming(args)
    else:
        _run_disk(args)


if __name__ == "__main__":
    main()
