#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_modeling.bivariate_cifti import find_pycortex_colormap, load_rgba_texture, quantize_bivariate, save_bivariate_dlabel, save_legend
from cf_modeling.cf_naming import ROI_MEAN_PREPROCESSING, roi_mean_connectivity_names, validate_roi_mean_preprocessing


def export_connectivity_bivariate(
    connectivity_cifti: Path,
    output_dir: Path,
    roi_a: str,
    roi_b: str,
    preprocessing_label: str,
    *,
    bins: int = 32,
    vmin: float = 0.0,
    vmax: float = 0.4,
    colormap_png: str | None = None,
    supporting_dir: Path | None = None,
) -> dict[str, Path]:
    validate_roi_mean_preprocessing(preprocessing_label)
    stem, expected = roi_mean_connectivity_names(roi_a, roi_b, preprocessing_label)
    image = nib.load(str(connectivity_cifti))
    names = [str(name) for name in image.header.get_axis(0).name]
    if len(image.shape) != 2 or names != expected:
        raise ValueError(f"Connectivity CIFTI map names/order do not match the expected two-map layout.\nExpected: {expected}\nFound: {names}")
    a_given_b, b_given_a = np.asarray(image.dataobj, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    supporting_dir = supporting_dir or output_dir.parent
    supporting_dir.mkdir(parents=True, exist_ok=True)
    texture_path = find_pycortex_colormap(colormap_png)
    texture = load_rgba_texture(texture_path)
    keys, labels = quantize_bivariate(b_given_a, a_given_b, texture, bins=bins, vmin=vmin, vmax=vmax)
    base = f"bivariate_{stem}"
    dlabel_path = output_dir / f"{base}_{bins}bin.dlabel.nii"
    legend_path = output_dir / f"{base}_legend.png"
    save_bivariate_dlabel(keys, labels, connectivity_cifti, dlabel_path,
                          f"ROI-mean partial Pearson r {roi_a}|{roi_b} x {roi_b}|{roi_a} ({bins}x{bins})")
    save_legend(texture, legend_path, roi_a, roi_b, vmin, vmax,
                xlabel=f"{roi_b} | {roi_a} partial Pearson r (blue axis)",
                ylabel=f"{roi_a} | {roi_b} partial Pearson r (red axis)",
                title="Bivariate positive ROI-mean partial Pearson-r colour key")
    metadata_path = supporting_dir / f"{base}.json"
    metadata_path.write_text(json.dumps({
        "analysis": "bivariate_visualization_of_roi_mean_timeseries_leave_one_run_out_partial_pearson_correlations",
        "analysis_scope": "ROI-mean fMRI time-series connectivity; not a connective-field model",
        "preprocessing_label": preprocessing_label,
        "preprocessing": ROI_MEAN_PREPROCESSING[preprocessing_label],
        "source_cifti": str(connectivity_cifti),
        "roi_a": roi_a,
        "roi_b": roi_b,
        "orientation": {"horizontal_blue": expected[1], "vertical_red": expected[0]},
        "dlabel": str(dlabel_path),
        "legend": str(legend_path),
        "bins_per_axis": bins,
        "vmin": vmin,
        "vmax": vmax,
        "colormap": str(texture_path),
        "negative_value_display": "Values below vmin are clipped to the zero-colour edge only in the dlabel visualization; signed values are preserved in source_cifti.",
    }, indent=2) + "\n")
    return {"dlabel": dlabel_path, "legend": legend_path, "metadata": metadata_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the two ROI-mean partial-r maps as a pycortex-style Workbench dlabel.")
    parser.add_argument("--connectivity-cifti", type=Path, required=True)
    parser.add_argument("--roi-a", required=True)
    parser.add_argument("--roi-b", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supporting-dir", type=Path, default=None, help="Directory for JSON provenance (default: parent of cifti_maps).")
    parser.add_argument("--preprocessing-label", required=True, choices=sorted(ROI_MEAN_PREPROCESSING))
    parser.add_argument("--colormap-png", default=None)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.connectivity_cifti.exists():
        raise FileNotFoundError(args.connectivity_cifti)
    outputs = export_connectivity_bivariate(
        args.connectivity_cifti, args.output_dir, args.roi_a, args.roi_b, args.preprocessing_label, bins=args.bins,
        vmin=args.vmin, vmax=args.vmax, colormap_png=args.colormap_png, supporting_dir=args.supporting_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
