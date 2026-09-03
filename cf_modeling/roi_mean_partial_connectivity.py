#!/usr/bin/env python3
"""Compute bilateral and hemispheric ROI-mean partial-correlation maps."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np


log = logging.getLogger("roi_mean_partial_connectivity")


def _runwise_standardize(values: np.ndarray, run_trs: np.ndarray) -> np.ndarray:
    """Z-score every column independently within each run (ddof=0)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if int(np.sum(run_trs)) != values.shape[0]:
        raise ValueError(
            f"run_trs sum to {int(np.sum(run_trs))}, but data have {values.shape[0]} TRs")
    result = np.empty_like(values)
    start = 0
    for length in np.asarray(run_trs, dtype=int):
        stop = start + int(length)
        chunk = values[start:stop]
        mean = np.nanmean(chunk, axis=0, keepdims=True)
        std = np.nanstd(chunk, axis=0, keepdims=True)
        safe = np.isfinite(std) & (std > np.finfo(np.float64).eps)
        result[start:stop] = np.where(safe, (chunk - mean) / std, 0.0)
        start = stop
    return np.nan_to_num(result, copy=False)


def paired_partial_correlations(
    target_z: np.ndarray,
    first_z: np.ndarray,
    second_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return corr(Y, first|second) and corr(Y, second|first) together.

    Inputs must already be centered/standardized over the same observations.
    The implementation uses the exact three-correlation identity and performs
    no ridge regularization.
    """
    target_z = np.asarray(target_z, dtype=np.float64)
    first_z = np.asarray(first_z, dtype=np.float64).reshape(-1)
    second_z = np.asarray(second_z, dtype=np.float64).reshape(-1)
    if target_z.ndim == 1:
        target_z = target_z[:, None]
    if target_z.shape[0] != first_z.size or first_z.size != second_z.size:
        raise ValueError("Targets and both ROI series must have the same number of TRs")

    n_obs = float(first_z.size)
    r_y_first = (target_z.T @ first_z) / n_obs
    r_y_second = (target_z.T @ second_z) / n_obs
    r_first_second = float((first_z @ second_z) / n_obs)
    r_y_first = np.clip(r_y_first, -1.0, 1.0)
    r_y_second = np.clip(r_y_second, -1.0, 1.0)
    r_first_second = float(np.clip(r_first_second, -1.0, 1.0))

    eps = np.finfo(np.float64).eps
    first_den = np.sqrt(np.maximum(
        (1.0 - r_y_second ** 2) * (1.0 - r_first_second ** 2), eps))
    second_den = np.sqrt(np.maximum(
        (1.0 - r_y_first ** 2) * (1.0 - r_first_second ** 2), eps))
    first_given_second = (
        (r_y_first - r_y_second * r_first_second) / first_den)
    second_given_first = (
        (r_y_second - r_y_first * r_first_second) / second_den)
    return (
        np.clip(first_given_second, -1.0, 1.0).astype(np.float32),
        np.clip(second_given_first, -1.0, 1.0).astype(np.float32),
    )


def _cortex_hemisphere_indices(brain_axis) -> dict[str, np.ndarray]:
    indices: dict[str, np.ndarray] = {}
    for structure, slc, _ in brain_axis.iter_structures():
        if "CORTEX_LEFT" in structure:
            indices["L"] = np.arange(len(brain_axis))[slc]
        elif "CORTEX_RIGHT" in structure:
            indices["R"] = np.arange(len(brain_axis))[slc]
    if set(indices) != {"L", "R"}:
        raise ValueError("Template must contain left and right cortical structures")
    return indices


def _load_mask(path: Path, expected: int) -> np.ndarray:
    image = nib.load(path)
    mask = np.asarray(image.dataobj)[0].astype(bool)
    if mask.size != expected:
        raise ValueError(f"{path} has {mask.size} grayordinates; expected {expected}")
    return mask


def _stream_mask_means(dataobj, masks: dict[str, np.ndarray], n_trs: int,
                       n_gray: int, batch_size: int) -> dict[str, np.ndarray]:
    """Accumulate several sparse mask means without ArrayProxy fancy indexing."""
    sums = {name: np.zeros(n_trs, dtype=np.float64) for name in masks}
    counts = {name: 0 for name in masks}
    for start in range(0, n_gray, batch_size):
        stop = min(start + batch_size, n_gray)
        active = {name: mask[start:stop] for name, mask in masks.items()
                  if np.any(mask[start:stop])}
        if not active:
            continue
        block = np.asarray(dataobj[:, start:stop], dtype=np.float64)
        for name, local_mask in active.items():
            sums[name] += block[:, local_mask].sum(axis=1)
            counts[name] += int(local_mask.sum())
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"Empty ROI masks: {empty}")
    return {name: sums[name] / counts[name] for name in masks}


def _save_dscalar(data: np.ndarray, names: list[str], template, path: Path) -> None:
    scalar_axis = nib.cifti2.ScalarAxis(names)
    brain_axis = template.header.get_axis(1)
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    image = nib.Cifti2Image(
        np.asarray(data, dtype=np.float32), header=header,
        nifti_header=template.nifti_header)
    nib.save(image, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dtseries", type=Path, required=True)
    parser.add_argument("--run-trs", type=Path, required=True)
    parser.add_argument("--roi-a", required=True,
                        help="Anterior CCA ROI name (or generic first ROI)")
    parser.add_argument("--roi-p", required=True,
                        help="Posterior CCA ROI name (or generic second ROI)")
    parser.add_argument("--mask-a", type=Path, required=True)
    parser.add_argument("--mask-p", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.dtseries, args.run_trs, args.mask_a, args.mask_p):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = nib.load(args.dtseries)
    if len(image.shape) != 2:
        raise ValueError(f"Expected a 2D dtseries, got {image.shape}")
    n_trs, n_gray = image.shape
    brain_axis = image.header.get_axis(1)
    hemis = _cortex_hemisphere_indices(brain_axis)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    if run_trs.sum() != n_trs:
        raise ValueError(f"run_trs sum {run_trs.sum()} != dtseries TRs {n_trs}")

    mask_a = _load_mask(args.mask_a, n_gray)
    mask_p = _load_mask(args.mask_p, n_gray)
    if np.any(mask_a & mask_p):
        raise ValueError("Anterior and posterior ROI masks overlap")

    mean_masks = {"a_bilateral": mask_a, "p_bilateral": mask_p}
    for hem, indices in hemis.items():
        hem_mask = np.zeros(n_gray, dtype=bool)
        hem_mask[indices] = True
        for role, mask in (("a", mask_a), ("p", mask_p)):
            selected = mask & hem_mask
            if not selected.any():
                raise ValueError(f"{role.upper()} mask is empty in hemisphere {hem}")
            mean_masks[f"{role}_{hem}"] = selected
    roi_series = _stream_mask_means(
        image.dataobj, mean_masks, n_trs, n_gray, args.batch_size)
    roi_z = {key: _runwise_standardize(value, run_trs)[:, 0]
             for key, value in roi_series.items()}

    bilateral_a = np.empty(n_gray, dtype=np.float32)
    bilateral_p = np.empty(n_gray, dtype=np.float32)
    within_a = np.full(n_gray, np.nan, dtype=np.float32)
    within_p = np.full(n_gray, np.nan, dtype=np.float32)

    for start in range(0, n_gray, args.batch_size):
        stop = min(start + args.batch_size, n_gray)
        target_z = _runwise_standardize(
            np.asarray(image.dataobj[:, start:stop], dtype=np.float64), run_trs)
        bilateral_a[start:stop], bilateral_p[start:stop] = paired_partial_correlations(
            target_z, roi_z["a_bilateral"], roi_z["p_bilateral"])
        for hem in ("L", "R"):
            local = hemis[hem][(hemis[hem] >= start) & (hemis[hem] < stop)]
            if local.size == 0:
                continue
            batch_cols = local - start
            a_vals, p_vals = paired_partial_correlations(
                target_z[:, batch_cols], roi_z[f"a_{hem}"], roi_z[f"p_{hem}"])
            within_a[local] = a_vals
            within_p[local] = p_vals
        log.info("Processed %d/%d grayordinates", stop, n_gray)

    maps = [bilateral_a, bilateral_p, within_a, within_p]
    names = [
        f"partial_r_{args.roi_a}_given_{args.roi_p}_bilateral",
        f"partial_r_{args.roi_p}_given_{args.roi_a}_bilateral",
        f"partial_r_{args.roi_a}_given_{args.roi_p}_within_hemisphere",
        f"partial_r_{args.roi_p}_given_{args.roi_a}_within_hemisphere",
    ]
    for hem in ("L", "R"):
        outside = np.ones(n_gray, dtype=bool)
        outside[hemis[hem]] = False
        for role, source, condition, values in (
            (args.roi_a, args.roi_a, args.roi_p, within_a),
            (args.roi_p, args.roi_p, args.roi_a, within_p),
        ):
            isolated = values.copy()
            isolated[outside] = np.nan
            maps.append(isolated)
            names.append(f"partial_r_{source}_given_{condition}_{hem}")

    out_path = args.output_dir / f"roi_mean_partial_connectivity_{args.roi_a}_{args.roi_p}.dscalar.nii"
    _save_dscalar(np.stack(maps), names, image, out_path)
    for name, values in zip(names[:4], maps[:4]):
        np.save(args.output_dir / f"{name}.npy", values)

    metadata = {
        "analysis": "exact_pairwise_partial_correlation",
        "dtseries": str(args.dtseries),
        "run_trs": run_trs.tolist(),
        "n_timepoints": int(n_trs),
        "roi_a": args.roi_a,
        "roi_p": args.roi_p,
        "mask_a": str(args.mask_a),
        "mask_p": str(args.mask_p),
        "roi_vertices": {
            args.roi_a: {"bilateral": int(mask_a.sum()),
                         "L": int(mask_a[hemis["L"]].sum()),
                         "R": int(mask_a[hemis["R"]].sum())},
            args.roi_p: {"bilateral": int(mask_p.sum()),
                         "L": int(mask_p[hemis["L"]].sum()),
                         "R": int(mask_p[hemis["R"]].sum())},
        },
        "standardization": "within-run z-score, all timepoints retained",
        "bilateral_definition": "ROI mean pools vertices across both hemispheres",
        "within_hemisphere_definition": (
            "left ROI means predict left targets; right ROI means predict right targets"),
        "distinction_from_cf_null": (
            "CF null uses one bilateral ROI mean at a time on held-out TRs and reports R2; "
            "these maps use both means simultaneously and report exact partial r."),
        "cifti": str(out_path),
        "map_names": names,
    }
    metadata_path = args.output_dir / f"roi_mean_partial_connectivity_{args.roi_a}_{args.roi_p}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
