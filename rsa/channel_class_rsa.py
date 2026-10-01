#!/usr/bin/env python3
"""DEPRECATED — Run searchlight RSA on AV-sensitivity-restricted embedding channels.

Deprecated 2026-09-02. Depends on the legacy 4-class ``channel_sensitivity``
labelling (``cf_modeling.channel_cca_analysis.CLASS_ORDER``), superseded by the
two continuous preference axes (modality A-V, CCA anterior-posterior). Do not
run. May be revived later to target the top-right / bottom-left quadrants of
the two-axis preference scatterplots, at which point channel selection must be
rewritten against those axes rather than class labels.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import (  # noqa: E402
    get_bm_axis, get_cortex_vertex_indices, save_cifti_multimap,
)
from rsa.perm_searchlight import (  # noqa: E402
    _run_hemisphere, within_run_shift_pair_indices,
)
from rsa.searchlight import (  # noqa: E402
    get_neighbors,
)
from rsa.shared.rsa_utils import (  # noqa: E402
    align_and_assert_bins, assert_segment_timing, load_fmri_cifti,
    get_run_bin_counts, preprocess_fmri, process_model_embeddings,
)


log = logging.getLogger("channel_class_rsa")

# File is RETIRED (see module docstring) -- cf_modeling.channel_cca_analysis
# no longer computes or exports a 4-class channel labelling, so CLASS_ORDER
# is redefined locally rather than imported. Do not revive this constant
# elsewhere; a revival of this script must select channels from the two
# continuous preference axes instead (see the deprecation note above).
CLASS_ORDER = ("audio_only", "video_only", "both", "neither")


def _parse_analysis(values: list[str]) -> dict[str, Path | str]:
    model_tag, roi_set, representation_tag, embedding, sensitivity = values
    return {
        "model_tag": model_tag, "roi_set": roi_set,
        "representation_tag": representation_tag,
        "embedding": Path(embedding), "sensitivity": Path(sensitivity),
    }


def _map_base(analysis: dict[str, Path | str], channel_class: str) -> str:
    model = str(analysis["model_tag"])
    representation = str(analysis["representation_tag"])
    prefix = model if model.endswith(f"_{representation}") else f"{model}_{representation}"
    return f"{prefix}_{analysis['roi_set']}_significance_{channel_class}"


def _load_classes(path: Path, n_channels: int) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("channel").reset_index(drop=True)
    required = {"channel", "modality_significance_class"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks columns {sorted(required - set(frame.columns))}")
    if len(frame) != n_channels or not np.array_equal(frame.channel, np.arange(n_channels)):
        raise ValueError(f"{path} does not define exactly channels 0..{n_channels - 1}")
    unknown = set(frame.modality_significance_class) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unknown channel classes in {path}: {sorted(unknown)}")
    missing = [name for name in CLASS_ORDER
               if not (frame.modality_significance_class == name).any()]
    if missing:
        raise ValueError(f"Empty channel classes in {path}: {missing}")
    return frame


def _surface_context(args: argparse.Namespace, fmri: np.ndarray) -> list[dict]:
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    contexts = []
    offset = 0
    for hemisphere, surface, indices in (
        ("left", args.left_surface, left_indices),
        ("right", args.right_surface, right_indices),
    ):
        neighbors = get_neighbors(
            str(surface), str(args.workbench), "group_average", hemisphere,
            args.k, args.geodesic_cache_dir)
        indices = indices.astype(np.int32)
        vertex_to_col = np.full(neighbors.shape[0], -1, dtype=np.int32)
        vertex_to_col[indices] = np.arange(len(indices), dtype=np.int32)
        contexts.append({
            "hemisphere": hemisphere, "fmri": fmri[:, offset:offset + len(indices)],
            "indices": indices, "neighbors": neighbors,
            "vertex_to_col": vertex_to_col, "offset": offset,
        })
        offset += len(indices)
    return contexts


def _rsa_permutation_maps(
    embedding: np.ndarray, contexts: list[dict], n_grayordinates: int,
    permutation_indices: np.ndarray, args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rho, raw p, max-T p, and the synchronized null maxima."""
    rho = np.zeros(n_grayordinates, dtype=np.float32)
    p_uncorrected = np.ones(n_grayordinates, dtype=np.float32)
    null_max = np.full(len(permutation_indices), -np.inf, dtype=np.float32)
    for context in contexts:
        rho_values, p_values, hemisphere_max = _run_hemisphere(
            context["fmri"], embedding, context["neighbors"],
            context["indices"], context["vertex_to_col"], args.method,
            permutation_indices, args.gpu_batch_size, args.perm_batch_size)
        start = context["offset"]
        stop = start + len(rho_values)
        rho[start:stop] = rho_values
        p_uncorrected[start:stop] = p_values
        null_max = np.maximum(null_max, hemisphere_max)
        del rho_values, p_values, hemisphere_max
        gc.collect()
    exceedances = (null_max[:, None] >= rho[None, :]).sum(axis=0)
    p_maxT = ((exceedances + 1) / (len(null_max) + 1)).astype(np.float32)
    return rho, p_uncorrected, p_maxT, null_max


def _maxT_critical_rho(null_max: np.ndarray, alpha: float = 0.05) -> float:
    """Exact strict rho cutoff for pseudo-count max-T p < alpha."""
    n_perm = len(null_max)
    max_allowed_exceed = int(np.ceil(alpha * (n_perm + 1) - 1) - 1)
    if max_allowed_exceed < 0:
        return float("inf")
    if max_allowed_exceed >= n_perm:
        return float("-inf")
    return float(np.sort(null_max)[::-1][max_allowed_exceed])


def _atomic_multimap(data: np.ndarray, names: list[str], template: Path,
                     output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp.dscalar.nii")
    save_cifti_multimap(data, names, str(template), str(temporary))
    temporary.replace(output)


def _cortex_mask(template: Path) -> np.ndarray:
    axis = get_bm_axis(template)
    mask = np.zeros(len(axis), dtype=bool)
    for name, structure_slice, _ in axis.iter_structures():
        if "CORTEX" in name:
            mask[structure_slice] = True
    if not mask.any():
        raise ValueError(f"No cortical grayordinates in {template}")
    return mask


def _spatial_agreement(first: np.ndarray, second: np.ndarray,
                       cortex: np.ndarray) -> dict[str, float]:
    one, two = first[cortex], second[cortex]
    threshold_one = np.quantile(one, 0.99)
    threshold_two = np.quantile(two, 0.99)
    top_one, top_two = one >= threshold_one, two >= threshold_two
    intersection = int(np.sum(top_one & top_two))
    union = int(np.sum(top_one | top_two))
    denominator = int(top_one.sum() + top_two.sum())
    return {
        "pearson_r": float(np.corrcoef(one, two)[0, 1]),
        "mean_absolute_difference": float(np.mean(np.abs(one - two))),
        "top1pct_dice": float(2 * intersection / denominator),
        "top1pct_jaccard": float(intersection / union),
    }


def _signed_sigmap(p_values: np.ndarray, effect: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).tiny
    return (np.sign(effect) * -np.log10(
        np.maximum(p_values, eps))).astype(np.float32)


def _inference_maps(
    rho: np.ndarray, p_uncorrected: np.ndarray, p_maxT: np.ndarray,
    alpha: float = 0.05, family_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return empirical raw, BH-FDR, and max-T FWER maps and masks."""
    rho = np.asarray(rho, dtype=np.float32)
    p_uncorrected = np.asarray(p_uncorrected, dtype=np.float32)
    p_maxT = np.asarray(p_maxT, dtype=np.float32)
    if not (rho.shape == p_uncorrected.shape == p_maxT.shape):
        raise ValueError("rho, p_uncorrected, and p_maxT must have the same shape")
    if family_mask is None:
        family_mask = np.ones(rho.shape, dtype=bool)
    family_mask = np.asarray(family_mask, dtype=bool)
    if family_mask.shape != rho.shape:
        raise ValueError("family_mask must have the same shape as rho")
    q_fdr = np.ones(rho.shape, dtype=np.float32)
    q_fdr[family_mask] = stats.false_discovery_control(
        p_uncorrected[family_mask].astype(np.float64), method="bh").astype(np.float32)
    return {
        "searchlight_rho": rho,
        "searchlight_p_perm_uncorrected": p_uncorrected,
        "searchlight_sigmap_perm_uncorrected": _signed_sigmap(p_uncorrected, rho),
        "searchlight_perm_uncorrected_mask_0p05":
            (p_uncorrected < alpha).astype(np.float32),
        "searchlight_q_bh_fdr": q_fdr,
        "searchlight_sigmap_bh_fdr": _signed_sigmap(q_fdr, rho),
        "searchlight_bh_fdr_mask_0p05": (q_fdr < alpha).astype(np.float32),
        "searchlight_p_maxT_fwer": p_maxT,
        "searchlight_sigmap_maxT_fwer": _signed_sigmap(p_maxT, rho),
        "searchlight_maxT_fwer_mask_0p05": (p_maxT < alpha).astype(np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", nargs=5, action="append", required=True,
        metavar=("MODEL_TAG", "ROI_SET", "REPRESENTATION_TAG", "EMBEDDING", "SENSITIVITY"),
        help="Repeat once per model; ROI_SET is provenance for channel selection.")
    parser.add_argument(
        "--reference-rsa", nargs=2, action="append", default=[],
        metavar=("MODEL_TAG", "RHO_NPY"),
        help="Optional matching full-embedding rho map for spatial agreement.")
    parser.add_argument("--dtseries", type=Path, required=True)
    parser.add_argument("--run-trs", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--template-cifti", type=Path, required=True)
    parser.add_argument("--left-surface", type=Path, required=True)
    parser.add_argument("--right-surface", type=Path, required=True)
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--geodesic-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-cifti", type=Path, required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--method", choices=("spearman", "pearson"), default="spearman")
    parser.add_argument("--gpu-batch-size", type=int, default=64)
    parser.add_argument("--perm-batch-size", type=int, default=50)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--permutation-seed", type=int, default=20260831)
    parser.add_argument(
        "--keep-inference-cache", action="store_true",
        help="Keep restart caches after the combined CIFTI is written (default: remove).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyses = [_parse_analysis(values) for values in args.analysis]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_cifti.parent.mkdir(parents=True, exist_ok=True)

    timing = pd.read_csv(args.timing_csv)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    assert_segment_timing(
        timing, args.bin_sec, args.tr, args.delay_sec, args.skip_sec, run_trs)
    fmri = preprocess_fmri(
        load_fmri_cifti(str(args.dtseries)), timing, run_trs,
        args.bin_sec, args.tr, args.delay_sec, skip_sec=args.skip_sec,
        normalize=True)
    n_bins, n_grayordinates = fmri.shape
    run_bins = get_run_bin_counts(
        timing, run_trs, args.bin_sec, args.tr, args.delay_sec, args.skip_sec)
    if int(run_bins.sum()) != n_bins:
        raise ValueError(f"Run-bin sum {run_bins.sum()} does not match {n_bins} fMRI bins")
    permutation_indices = within_run_shift_pair_indices(
        run_bins, args.n_permutations, args.permutation_seed)
    log.info("Binned fMRI: %s", fmri.shape)
    contexts = _surface_context(args, fmri)
    cortex = _cortex_mask(args.template_cifti)

    maps, names, map_reports = [], [], []
    inference_caches: set[Path] = set()
    rho_by_analysis: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for analysis in analyses:
        source = np.load(analysis["embedding"], mmap_mode="r")
        classes = _load_classes(analysis["sensitivity"], source.shape[1])
        embedding = process_model_embeddings(
            str(analysis["embedding"]), timing, args.bin_sec, args.tr, run_trs,
            delay_sec=args.delay_sec, hrf=False, skip_sec=args.skip_sec,
            model_norm="zscore")
        _, embedding = align_and_assert_bins(fmri, embedding)
        embedding = np.nan_to_num(embedding, copy=False)

        for channel_class in CLASS_ORDER:
            selected = (classes.modality_significance_class.to_numpy() == channel_class)
            base = _map_base(analysis, channel_class)
            cache = args.output_dir / (
                f"{base}_k{args.k}_{args.method}_perm{args.n_permutations}_inference.npz")
            inference_caches.add(cache)
            if cache.exists():
                cached = np.load(cache)
                rho = cached["rho"].astype(np.float32)
                p_uncorrected = cached["p_uncorrected"].astype(np.float32)
                p_maxT = cached["p_maxT"].astype(np.float32)
                null_max = cached["null_max"].astype(np.float32)
                if any(values.shape != (n_grayordinates,) for values in
                       (rho, p_uncorrected, p_maxT)):
                    raise ValueError(f"Unexpected cached map shape in {cache}")
                if null_max.shape != (args.n_permutations,):
                    raise ValueError(f"Unexpected null-max shape in {cache}")
                log.info("Loaded cached inference: %s", cache.name)
            else:
                log.info(
                    "RSA %s: %d channels, %d within-run shifts",
                    base, int(selected.sum()), args.n_permutations)
                rho, p_uncorrected, p_maxT, null_max = _rsa_permutation_maps(
                    embedding[:, selected], contexts, n_grayordinates,
                    permutation_indices, args)
                np.savez_compressed(
                    cache, rho=rho, p_uncorrected=p_uncorrected,
                    p_maxT=p_maxT, null_max=null_max)

            inference = _inference_maps(
                rho, p_uncorrected, p_maxT, family_mask=cortex)
            raw_mask = inference["searchlight_perm_uncorrected_mask_0p05"] > 0
            fdr_mask = inference["searchlight_bh_fdr_mask_0p05"] > 0
            maxT_mask = inference["searchlight_maxT_fwer_mask_0p05"] > 0
            for suffix, values in inference.items():
                names.append(f"{base}_{suffix}")
                maps.append(values.astype(np.float32, copy=False))
            map_reports.append({
                "model_tag": analysis["model_tag"], "roi_set": analysis["roi_set"],
                "representation_tag": analysis["representation_tag"],
                "channel_class": channel_class, "n_channels": int(selected.sum()),
                "mean_rho_all_grayordinates": float(np.mean(rho)),
                "mean_rho_cortex": float(np.mean(rho[cortex])),
                "median_rho_cortex": float(np.median(rho[cortex])),
                "max_rho": float(np.max(rho[cortex])),
                "minimum_rho": float(np.min(rho[cortex])),
                "minimum_p_perm_uncorrected_cortex":
                    float(np.min(p_uncorrected[cortex])),
                "minimum_q_bh_fdr_cortex":
                    float(np.min(inference["searchlight_q_bh_fdr"][cortex])),
                "minimum_p_maxT_fwer_cortex": float(np.min(p_maxT[cortex])),
                "minimum_reachable_permutation_p": 1 / (args.n_permutations + 1),
                "maxT_fwer_critical_rho_0p05": _maxT_critical_rho(null_max),
                "n_uncorrected_significant_cortex": int(raw_mask[cortex].sum()),
                "n_bh_fdr_significant_cortex": int(fdr_mask[cortex].sum()),
                "n_maxT_fwer_significant_cortex": int(maxT_mask[cortex].sum()),
                "uncorrected_fraction_cortex": float(raw_mask[cortex].mean()),
                "bh_fdr_fraction_cortex": float(fdr_mask[cortex].mean()),
                "maxT_fwer_fraction_cortex": float(maxT_mask[cortex].mean()),
                "inference_cache": str(cache) if args.keep_inference_cache else None,
            })
            analysis_key = (
                str(analysis["model_tag"]), str(analysis["roi_set"]),
                str(analysis["representation_tag"]),
            )
            rho_by_analysis.setdefault(analysis_key, {})[channel_class] = rho
        del embedding
        gc.collect()

    _atomic_multimap(np.stack(maps), names, args.template_cifti, args.output_cifti)
    references = {tag: Path(path) for tag, path in args.reference_rsa}
    agreement_rows = []
    for analysis in analyses:
        model_tag = str(analysis["model_tag"])
        roi_set = str(analysis["roi_set"])
        representation_tag = str(analysis["representation_tag"])
        category_maps = rho_by_analysis[(model_tag, roi_set, representation_tag)]
        if model_tag in references:
            reference = np.load(references[model_tag]).astype(np.float32)
            if reference.shape != (n_grayordinates,):
                raise ValueError(f"Reference shape {reference.shape}: {references[model_tag]}")
            for channel_class, rho in category_maps.items():
                agreement_rows.append({
                    "model_tag": model_tag, "roi_set": roi_set,
                    "representation_tag": representation_tag,
                    "comparison": "category_vs_full",
                    "map_a": channel_class, "map_b": "full_embedding",
                    **_spatial_agreement(rho, reference, cortex),
                })
        for first, second in combinations(CLASS_ORDER, 2):
            agreement_rows.append({
                "model_tag": model_tag, "roi_set": roi_set,
                "representation_tag": representation_tag,
                "comparison": "category_pair",
                "map_a": first, "map_b": second,
                **_spatial_agreement(category_maps[first], category_maps[second], cortex),
            })

    metadata = {
        "analysis": "AV-sensitivity four-class channel searchlight RSA",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_cifti": str(args.output_cifti), "map_count": len(names),
        "category_rho_map_count": len(map_reports), "map_names": names,
        "maps": map_reports, "n_bins": n_bins, "n_grayordinates": n_grayordinates,
        "n_cortical_grayordinates": int(cortex.sum()),
        "spatial_agreement": agreement_rows,
        "parameters": {"k": args.k, "bin_sec": args.bin_sec,
                       "skip_sec": args.skip_sec, "delay_sec": args.delay_sec,
                       "method": args.method, "normalization": "within-run z-score",
                       "n_permutations": args.n_permutations,
                       "permutation_seed": args.permutation_seed,
                       "permutation_scheme":
                           "independent nonzero circular shifts within each movie run"},
        "significance": ("One-tailed empirical permutation p-values; BH-FDR q-values "
                         "across cortical searchlights; and single-step max-T FWER "
                         "p-values from the cortex-wide maximum rho per synchronized "
                         "within-run circular-shift permutation. Alpha is 0.05."),
        "scope": ("Group-average descriptive RSA; ROI-set tags identify the channel-analysis "
                  "provenance and do not spatially restrict the searchlight."),
    }
    metadata_path = args.output_dir / "channel_significance_class_rsa_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame(map_reports).to_csv(
        args.output_dir / "channel_significance_class_rsa_map_summary.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(
        args.output_dir / "channel_significance_class_rsa_spatial_agreement.csv", index=False)
    if not args.keep_inference_cache:
        for cache in inference_caches:
            cache.unlink(missing_ok=True)
        log.info("Removed %d temporary inference caches after successful CIFTI write",
                 len(inference_caches))
    log.info("Saved %d maps: %s", len(names), args.output_cifti)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
