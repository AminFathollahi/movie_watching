#!/usr/bin/env python3
"""Export ordinary CCA correlation pairs as four Workbench 2-D dlabels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

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


def expected_names(roi_a: str, roi_p: str) -> list[str]:
    return [
        f"r_{roi_a}_raw_corr_bilateral", f"r_{roi_p}_raw_corr_bilateral",
        f"r_{roi_a}_raw_corr_within_hemisphere", f"r_{roi_p}_raw_corr_within_hemisphere",
        f"r_{roi_a}_raw_corr_L", f"r_{roi_p}_raw_corr_L",
        f"r_{roi_a}_raw_corr_R", f"r_{roi_p}_raw_corr_R",
    ]


def export_raw_bivariate_views(
    source: Path, output_dir: Path, roi_a: str, roi_p: str, *,
    bins: int = 32, vmin: float = 0.0, vmax: float = 0.4,
    colormap_png: str | None = None,
) -> dict[str, Path]:
    """Write bilateral, within-hemisphere, left, and right raw-r dlabels."""
    image = nib.load(str(source))
    if len(image.shape) != 2 or image.shape[0] != 8:
        raise ValueError(f"Expected an 8-map raw-correlation dscalar, got {image.shape}")
    names = [str(name) for name in image.header.get_axis(0).name]
    expected = expected_names(roi_a, roi_p)
    if names != expected:
        raise ValueError(f"Raw-correlation map names/order mismatch\nExpected: {expected}\nFound: {names}")
    maps = np.asarray(image.dataobj, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    texture_path = find_pycortex_colormap(colormap_png)
    texture = load_rgba_texture(texture_path)
    stem = f"bivariate_raw_corr_{roi_a}_{roi_p}"
    outputs: dict[str, Path] = {}
    for view, (a_index, p_index) in VIEW_MAPS.items():
        keys, labels = quantize_bivariate(
            maps[p_index], maps[a_index], texture,
            bins=bins, vmin=vmin, vmax=vmax)
        path = output_dir / f"{stem}_{view}_{bins}bin.dlabel.nii"
        save_bivariate_dlabel(
            keys, labels, source, path,
            f"raw r {roi_a} x {roi_p} ({view}, {bins}x{bins})")
        outputs[view] = path
    legend = output_dir / f"{stem}_legend.png"
    save_legend(
        texture, legend, roi_a, roi_p, vmin, vmax,
        xlabel="CCA-P raw Pearson r (blue axis)",
        ylabel="CCA-A raw Pearson r (red axis)",
        title="Bivariate positive raw-correlation colour key")
    outputs["legend"] = legend
    metadata_path = output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps({
        "analysis": "bivariate_visualization_of_ordinary_correlations",
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
    parser.add_argument("--raw-cifti", type=Path, required=True)
    parser.add_argument("--roi-a", required=True)
    parser.add_argument("--roi-p", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colormap-png")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.4)
    args = parser.parse_args()
    outputs = export_raw_bivariate_views(
        args.raw_cifti, args.output_dir, args.roi_a, args.roi_p,
        bins=args.bins, vmin=args.vmin, vmax=args.vmax,
        colormap_png=args.colormap_png)
    for view, path in outputs.items():
        print(f"{view}: {path}")


if __name__ == "__main__":
    main()
