#!/usr/bin/env python3
"""Export bilateral and hemispheric CCA partial-r pairs as 2-D dlabels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from cf_modeling.bivariate_cifti import (
        find_pycortex_colormap,
        load_rgba_texture,
        quantize_bivariate,
        save_bivariate_dlabel,
        save_legend,
    )
except ModuleNotFoundError:
    from bivariate_cifti import (
        find_pycortex_colormap,
        load_rgba_texture,
        quantize_bivariate,
        save_bivariate_dlabel,
        save_legend,
    )


VIEW_MAPS = {
    "bilateral": (0, 1),
    "within_hemisphere": (2, 3),
    "L": (4, 5),
    "R": (6, 7),
}


def _validate_map_names(names: list[str], roi_a: str, roi_p: str) -> None:
    """Reject a dscalar whose map order is incompatible with this exporter."""
    expected = [
        f"partial_r_{roi_a}_given_{roi_p}_bilateral",
        f"partial_r_{roi_p}_given_{roi_a}_bilateral",
        f"partial_r_{roi_a}_given_{roi_p}_within_hemisphere",
        f"partial_r_{roi_p}_given_{roi_a}_within_hemisphere",
        f"partial_r_{roi_a}_given_{roi_p}_L",
        f"partial_r_{roi_p}_given_{roi_a}_L",
        f"partial_r_{roi_a}_given_{roi_p}_R",
        f"partial_r_{roi_p}_given_{roi_a}_R",
    ]
    if names != expected:
        raise ValueError(
            "Partial-correlation CIFTI map names/order do not match the expected "
            f"eight-map layout.\nExpected: {expected}\nFound: {names}"
        )


def export_partial_bivariate_views(
    partial_cifti: Path,
    output_dir: Path,
    roi_a: str,
    roi_p: str,
    *,
    bins: int = 32,
    vmin: float = 0.0,
    vmax: float = 0.4,
    colormap_png: str | None = None,
) -> dict[str, Path]:
    """Write bilateral, combined-within-hemi, left, and right 2-D dlabels."""
    image = nib.load(str(partial_cifti))
    if len(image.shape) != 2 or image.shape[0] != 8:
        raise ValueError(f"Expected an 8-map partial dscalar, got {image.shape}")
    names = [str(name) for name in image.header.get_axis(0).name]
    _validate_map_names(names, roi_a, roi_p)
    maps = np.asarray(image.dataobj, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    texture_path = find_pycortex_colormap(colormap_png)
    texture = load_rgba_texture(texture_path)
    outputs: dict[str, Path] = {}
    stem = f"bivariate_partial_r_{roi_a}_{roi_p}"

    for view, (a_index, p_index) in VIEW_MAPS.items():
        # Same orientation as Figure 3a: P on dim1/blue (horizontal),
        # A on dim2/red (vertical).
        keys, labels = quantize_bivariate(
            maps[p_index], maps[a_index], texture,
            bins=bins, vmin=vmin, vmax=vmax,
        )
        path = output_dir / f"{stem}_{view}_{bins}bin.dlabel.nii"
        save_bivariate_dlabel(
            keys,
            labels,
            partial_cifti,
            path,
            f"partial r {roi_a}|{roi_p} x {roi_p}|{roi_a} ({view}, {bins}x{bins})",
        )
        outputs[view] = path

    legend_path = output_dir / f"{stem}_legend.png"
    save_legend(
        texture,
        legend_path,
        roi_a,
        roi_p,
        vmin,
        vmax,
        xlabel="CCA-P | CCA-A partial r (blue axis)",
        ylabel="CCA-A | CCA-P partial r (red axis)",
        title="Bivariate positive partial-correlation colour key",
    )
    outputs["legend"] = legend_path

    metadata_path = output_dir / f"{stem}.json"
    metadata = {
        "analysis": "bivariate_visualization_of_exact_partial_correlations",
        "source_cifti": str(partial_cifti),
        "roi_a": roi_a,
        "roi_p": roi_p,
        "orientation": {
            "horizontal_blue": f"partial_r_{roi_p}_given_{roi_a}",
            "vertical_red": f"partial_r_{roi_a}_given_{roi_p}",
        },
        "views": {name: str(path) for name, path in outputs.items()
                  if name != "legend"},
        "legend": str(legend_path),
        "bins_per_axis": bins,
        "vmin": vmin,
        "vmax": vmax,
        "colormap": str(texture_path),
        "negative_value_display": (
            "Values below vmin are clipped to the zero-colour edge only in the "
            "dlabel visualization; signed values are preserved in source_cifti."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    outputs["metadata"] = metadata_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exact partial-r pairs as pycortex-style Workbench dlabels."
    )
    parser.add_argument("--partial-cifti", type=Path, required=True)
    parser.add_argument("--roi-a", required=True)
    parser.add_argument("--roi-p", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colormap-png", default=None)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.partial_cifti.exists():
        raise FileNotFoundError(args.partial_cifti)
    outputs = export_partial_bivariate_views(
        args.partial_cifti,
        args.output_dir,
        args.roi_a,
        args.roi_p,
        bins=args.bins,
        vmin=args.vmin,
        vmax=args.vmax,
        colormap_png=args.colormap_png,
    )
    for view, path in outputs.items():
        print(f"{view}: {path}")


if __name__ == "__main__":
    main()
