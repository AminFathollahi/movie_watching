#!/usr/bin/env python3
"""Export a Hedger-style bivariate CF map for Workbench."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Support package imports and direct script execution.
try:
    from cf_modeling.bivariate_cifti import (
        find_pycortex_colormap,
        load_rgba_texture,
        quantize_bivariate,
        save_bivariate_dlabel,
        save_legend,
        save_pair_dscalar,
    )
except ModuleNotFoundError:
    from bivariate_cifti import (
        find_pycortex_colormap,
        load_rgba_texture,
        quantize_bivariate,
        save_bivariate_dlabel,
        save_legend,
        save_pair_dscalar,
    )


DEFAULT_OUTPUT_BASE = Path(
    "/home/amin/Research/Representation/Movie/outputs/cf_modeling"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a quantized Hedger/Pycortex 2-D CF map for wb_view."
    )
    parser.add_argument("--roi-a", required=True, help="dim2/red-axis ROI, e.g. cca_a")
    parser.add_argument("--roi-b", required=True, help="dim1/blue-axis ROI, e.g. cca_p")
    parser.add_argument(
        "--mode", choices=("group_average", "per_subject"), default="group_average"
    )
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--template-cifti", type=Path, required=True)
    parser.add_argument("--colormap-png", default=None)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument(
        "--vmax", type=float, default=0.4,
        help="Use 0.4 to reproduce the existing Figure 3a scale exactly.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: the ROI pair's cifti_maps directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_root = args.output_base / args.mode / f"{args.roi_a}_{args.roi_b}"
    source_dir = pair_root / ("prep" if args.mode == "group_average" else "group")
    suffix = "" if args.mode == "group_average" else "_avg"
    output_dir = args.output_dir or pair_root / "cifti_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match plot_fig3a exactly: ROI B is dim1/blue/horizontal and ROI A is
    # dim2/red/vertical.
    roi_a_values = np.load(source_dir / f"R2_{args.roi_a}{suffix}.npy")
    roi_b_values = np.load(source_dir / f"R2_{args.roi_b}{suffix}.npy")
    texture_path = find_pycortex_colormap(args.colormap_png)
    texture = load_rgba_texture(texture_path)
    keys, labels = quantize_bivariate(
        roi_b_values,
        roi_a_values,
        texture,
        bins=args.bins,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    stem = f"bivariate_{args.roi_a}_{args.roi_b}_raw"
    dlabel_path = output_dir / f"{stem}_{args.bins}bin.dlabel.nii"
    dscalar_path = output_dir / f"{stem}_axes.dscalar.nii"
    legend_path = output_dir / f"{stem}_legend.png"
    save_bivariate_dlabel(
        keys,
        labels,
        args.template_cifti,
        dlabel_path,
        f"{args.roi_a} x {args.roi_b} raw R2 ({args.bins}x{args.bins})",
    )
    save_pair_dscalar(
        roi_b_values,
        roi_a_values,
        (f"R2_{args.roi_b}", f"R2_{args.roi_a}"),
        args.template_cifti,
        dscalar_path,
    )
    save_legend(
        texture, legend_path, args.roi_a, args.roi_b, args.vmin, args.vmax
    )

    print(f"Bivariate Workbench map: {dlabel_path}")
    print(f"Continuous source axes:  {dscalar_path}")
    print(f"2-D colour key:          {legend_path}")
    print(f"Colormap source:         {texture_path}")
    print(
        f"Saturated at vmax={args.vmax:g}: "
        f"{args.roi_a}={np.mean(roi_a_values >= args.vmax):.1%}, "
        f"{args.roi_b}={np.mean(roi_b_values >= args.vmax):.1%}, "
        f"both={np.mean((roi_a_values >= args.vmax) & (roi_b_values >= args.vmax)):.1%}"
    )
    print("Open the dlabel in wb_view; add surfaces/parcels/borders as other overlays.")


if __name__ == "__main__":
    main()
