#!/usr/bin/env python3
"""Export zero-order ROI-mean Pearson-r pairs as four Workbench 2-D dlabels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from cf_modeling.cf_naming import ROI_MEAN_PREPROCESSING, validate_roi_mean_preprocessing

try:
    from cf_modeling.bivariate_cifti import (
        find_pycortex_colormap, load_rgba_texture, quantize_bivariate,
        save_bivariate_dlabel, save_legend,
    )
except ModuleNotFoundError:
    from bivariate_cifti import (
        find_pycortex_colormap, load_rgba_texture, quantize_bivariate,
        save_bivariate_dlabel, save_legend,
    )


VIEW_MAPS = {"bilateral": (0, 1), "within_hemisphere": (2, 3), "L": (4, 5), "R": (6, 7)}


def expected_names(roi_a: str, roi_p: str, preprocessing_label: str | None = None) -> list[str]:
    suffix = f"_{preprocessing_label}" if preprocessing_label else ""
    return [
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_a}_bilateral{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_p}_bilateral{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_a}_within_hemisphere{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_p}_within_hemisphere{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_a}_L{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_p}_L{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_a}_R{suffix}",
        f"roi_mean_timeseries_zero_order_pearson_r_{roi_p}_R{suffix}",
    ]


def export_raw_bivariate_views(
    source: Path, output_dir: Path, roi_a: str, roi_p: str, *,
    bins: int = 32, vmin: float = 0.0, vmax: float = 0.4,
    colormap_png: str | None = None, supporting_dir: Path | None = None,
    preprocessing_label: str | None = None,
) -> dict[str, Path]:
    """Write zero-order ROI-mean Pearson-r dlabels for four spatial scopes."""
    image = nib.load(str(source))
    if len(image.shape) != 2 or image.shape[0] != 8:
        raise ValueError(f"Expected an 8-map zero-order Pearson-r dscalar, got {image.shape}")
    names = [str(name) for name in image.header.get_axis(0).name]
    if preprocessing_label is not None:
        validate_roi_mean_preprocessing(preprocessing_label)
    expected = expected_names(roi_a, roi_p, preprocessing_label)
    if names != expected:
        raise ValueError(f"Zero-order Pearson-r map names/order mismatch\nExpected: {expected}\nFound: {names}")
    maps = np.asarray(image.dataobj, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    supporting_dir = supporting_dir or output_dir.parent
    supporting_dir.mkdir(parents=True, exist_ok=True)
    texture_path = find_pycortex_colormap(colormap_png)
    texture = load_rgba_texture(texture_path)
    suffix = f"_{preprocessing_label}" if preprocessing_label else ""
    stem = f"bivariate_roi_mean_timeseries_zero_order_pearson_r_{roi_a}_{roi_p}{suffix}"
    outputs: dict[str, Path] = {}
    for view, (a_index, p_index) in VIEW_MAPS.items():
        keys, labels = quantize_bivariate(
            maps[p_index], maps[a_index], texture,
            bins=bins, vmin=vmin, vmax=vmax)
        path = output_dir / f"{stem}_{view}_{bins}bin.dlabel.nii"
        save_bivariate_dlabel(
            keys, labels, source, path,
            f"ROI-mean zero-order Pearson r {roi_a} x {roi_p} ({view}, {bins}x{bins})")
        outputs[view] = path
    legend = output_dir / f"{stem}_legend.png"
    save_legend(
        texture, legend, roi_a, roi_p, vmin, vmax,
        xlabel="ROI-P zero-order Pearson r (blue axis)",
        ylabel="ROI-A zero-order Pearson r (red axis)",
        title="Bivariate positive ROI-mean zero-order Pearson-r colour key")
    outputs["legend"] = legend
    metadata_path = supporting_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps({
        "analysis": "bivariate_visualization_of_roi_mean_timeseries_zero_order_pearson_correlations",
        "analysis_scope": "ROI-mean fMRI time-series connectivity; not a connective-field model",
        "preprocessing_label": preprocessing_label or "legacy_unspecified",
        "preprocessing": (ROI_MEAN_PREPROCESSING[preprocessing_label]
                          if preprocessing_label else "legacy output: provenance unavailable"),
        "source_cifti": str(source), "roi_a": roi_a, "roi_p": roi_p,
        "views": {key: str(value) for key, value in outputs.items() if key != "legend"},
        "legend": str(legend), "bins_per_axis": bins, "vmin": vmin, "vmax": vmax,
        "colormap": str(texture_path),
        "negative_value_display": "Clipped to the zero-colour edge in dlabels only; source values remain signed.",
    }, indent=2) + "\n")
    outputs["metadata"] = metadata_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-order-cifti", "--raw-cifti", dest="zero_order_cifti",
                        type=Path, required=True,
                        help="8-map zero-order ROI-mean Pearson-r dscalar (--raw-cifti is a deprecated alias).")
    parser.add_argument("--roi-a", required=True)
    parser.add_argument("--roi-p", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supporting-dir", type=Path, default=None,
                        help="Directory for JSON provenance (default: parent of cifti_maps).")
    parser.add_argument("--preprocessing-label", required=True,
                        choices=sorted(ROI_MEAN_PREPROCESSING))
    parser.add_argument("--colormap-png")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.4)
    args = parser.parse_args()
    outputs = export_raw_bivariate_views(
        args.zero_order_cifti, args.output_dir, args.roi_a, args.roi_p,
        bins=args.bins, vmin=args.vmin, vmax=args.vmax,
        colormap_png=args.colormap_png, supporting_dir=args.supporting_dir,
        preprocessing_label=args.preprocessing_label)
    for view, path in outputs.items():
        print(f"{view}: {path}")


if __name__ == "__main__":
    main()
