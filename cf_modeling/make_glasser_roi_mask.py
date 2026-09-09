#!/usr/bin/env python3
"""Save a bilateral Glasser-parcel grayordinate mask as a dscalar.nii."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from cf_modeling.roi_mean_partial_connectivity import _save_dscalar
from rsa.glasser import load_glasser_parcels


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glasser-dlabel", required=True)
    parser.add_argument("--template-cifti", required=True)
    parser.add_argument("--roi", required=True, help="Bilateral Glasser short name, e.g. A5")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args(argv)


def build_bilateral_mask(glasser_dlabel: str, bm_axis, roi: str) -> np.ndarray:
    parcels = load_glasser_parcels(glasser_dlabel, bm_axis)
    matching = [
        indices for name, indices in parcels.items()
        if name.removeprefix("L_").removeprefix("R_").removesuffix("_ROI") == roi
    ]
    if not matching:
        short = sorted({
            name.removeprefix("L_").removeprefix("R_").removesuffix("_ROI")
            for name in parcels
        })
        raise KeyError(f"Unknown Glasser parcel {roi!r}; known short names include {short[:10]}...")
    mask = np.zeros(len(bm_axis), dtype=bool)
    mask[np.unique(np.concatenate(matching))] = True
    return mask


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    template = nib.load(args.template_cifti)
    bm_axis = template.header.get_axis(1)
    mask = build_bilateral_mask(args.glasser_dlabel, bm_axis, args.roi)
    _save_dscalar(mask[None, :].astype(np.float32), [args.roi], template, args.output_path)
    print(f"{args.roi}: {int(mask.sum())} grayordinates -> {args.output_path}")


if __name__ == "__main__":
    main()
