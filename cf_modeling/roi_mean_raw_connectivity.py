#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from cf_modeling.cf_naming import ROI_MEAN_PREPROCESSING, ROI_MEAN_PREPROCESSING_OPTIONS, roi_mean_preprocessing_label

try:
    from cf_modeling.roi_mean_partial_connectivity import (
        _cortex_hemisphere_indices,
        _load_mask,
        _runwise_standardize,
        _save_dscalar,
        _stream_mask_means,
    )
except ModuleNotFoundError:
    from roi_mean_partial_connectivity import (
        _cortex_hemisphere_indices,
        _load_mask,
        _runwise_standardize,
        _save_dscalar,
        _stream_mask_means,
    )


log = logging.getLogger("roi_mean_raw_connectivity")


def correlations(target_z: np.ndarray, source_z: np.ndarray) -> np.ndarray:
    target_z = np.asarray(target_z, dtype=np.float64)
    source_z = np.asarray(source_z, dtype=np.float64).reshape(-1)
    if target_z.ndim == 1:
        target_z = target_z[:, None]
    if target_z.shape[0] != source_z.size:
        raise ValueError("Targets and source must have the same observations")
    return np.clip(target_z.T @ source_z / source_z.size, -1.0, 1.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dtseries", type=Path, required=True)
    parser.add_argument("--run-trs", type=Path, required=True)
    parser.add_argument("--roi-a", required=True)
    parser.add_argument("--roi-p", required=True)
    parser.add_argument("--mask-a", type=Path, required=True)
    parser.add_argument("--mask-p", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocessing", choices=sorted(ROI_MEAN_PREPROCESSING_OPTIONS),
                        default="raw")
    parser.add_argument(
        "--supporting-dir", type=Path, default=None,
        help=("Directory for NPY component backups and JSON provenance. Default: "
              "the pair directory (parent of cifti_maps)"),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessing_label = roi_mean_preprocessing_label(args.preprocessing)
    for path in (args.dtseries, args.run_trs, args.mask_a, args.mask_p):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supporting_dir = args.supporting_dir or args.output_dir.parent
    supporting_dir.mkdir(parents=True, exist_ok=True)

    image = nib.load(args.dtseries)
    if len(image.shape) != 2:
        raise ValueError(f"Expected a 2D dtseries, got {image.shape}")
    n_trs, n_gray = image.shape
    brain_axis = image.header.get_axis(1)
    hemis = _cortex_hemisphere_indices(brain_axis)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    if int(run_trs.sum()) != n_trs:
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
        bilateral_a[start:stop] = correlations(target_z, roi_z["a_bilateral"])
        bilateral_p[start:stop] = correlations(target_z, roi_z["p_bilateral"])
        for hem in ("L", "R"):
            local = hemis[hem][(hemis[hem] >= start) & (hemis[hem] < stop)]
            if not local.size:
                continue
            cols = local - start
            within_a[local] = correlations(target_z[:, cols], roi_z[f"a_{hem}"])
            within_p[local] = correlations(target_z[:, cols], roi_z[f"p_{hem}"])
        log.info("Processed %d/%d grayordinates", stop, n_gray)

    maps = [bilateral_a, bilateral_p, within_a, within_p]
    names = [
        f"roi_mean_timeseries_zero_order_pearson_r_{args.roi_a}_bilateral_{preprocessing_label}",
        f"roi_mean_timeseries_zero_order_pearson_r_{args.roi_p}_bilateral_{preprocessing_label}",
        f"roi_mean_timeseries_zero_order_pearson_r_{args.roi_a}_within_hemisphere_{preprocessing_label}",
        f"roi_mean_timeseries_zero_order_pearson_r_{args.roi_p}_within_hemisphere_{preprocessing_label}",
    ]
    for hem in ("L", "R"):
        outside = np.ones(n_gray, dtype=bool)
        outside[hemis[hem]] = False
        for roi, values in ((args.roi_a, within_a), (args.roi_p, within_p)):
            isolated = values.copy()
            isolated[outside] = np.nan
            maps.append(isolated)
            names.append(f"roi_mean_timeseries_zero_order_pearson_r_{roi}_{hem}_{preprocessing_label}")

    out_path = args.output_dir / (
        f"roi_mean_timeseries_zero_order_pearson_r_{args.roi_a}_{args.roi_p}_{preprocessing_label}.dscalar.nii")
    _save_dscalar(np.stack(maps), names, image, out_path)
    for name, values in zip(names[:4], maps[:4]):
        np.save(supporting_dir / f"{name}.npy", values)

    metadata = {
        "analysis": "roi_mean_timeseries_zero_order_pearson_correlation",
        "analysis_scope": (
            "ROI-mean fMRI time-series connectivity; not a connective-field model"),
        "preprocessing_option": args.preprocessing,
        "preprocessing_label": preprocessing_label,
        "preprocessing": ROI_MEAN_PREPROCESSING[preprocessing_label],
        "distinction_from_nc": (
            "These are correlations between ROI means and cortical timecourses; "
            "CF _nc is split-model R2 minus a one-regressor OLS null R2."),
        "dtseries": str(args.dtseries),
        "run_trs": run_trs.tolist(),
        "n_timepoints": int(n_trs),
        "standardization": "within-run z-score, all timepoints retained",
        "roi_a": args.roi_a,
        "roi_p": args.roi_p,
        "mask_a": str(args.mask_a),
        "mask_p": str(args.mask_p),
        "map_names": names,
        "cifti": str(out_path),
    }
    metadata_path = supporting_dir / (
        f"roi_mean_timeseries_zero_order_pearson_r_{args.roi_a}_{args.roi_p}_{preprocessing_label}.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
