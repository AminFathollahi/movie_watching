#!/usr/bin/env python3
"""Winner-take-all dominance map across several temporal seed ROIs.

Consumes the raw ROI-mean correlation maps already produced by
roi_mean_raw_connectivity.py for each seed and, for every candidate
grayordinate (the requested frontal parcel set, or the whole cortex if none
is given), labels it by whichever seed has the highest raw correlation
there. A grayordinate is left unlabelled (NaN) if every seed's correlation
is at or below zero there — a vertex least anti-correlated with a seed is
not evidence that seed dominates it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from cf_modeling.roi_mean_partial_connectivity import _load_mask, _save_dscalar
from rsa.glasser import load_glasser_parcels


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", action="append", required=True, metavar="NAME=RAW_CORR_NPY",
        help="Seed name and path to its r_{name}_raw_corr_bilateral.npy map; repeatable.",
    )
    parser.add_argument("--glasser-dlabel", help="Required only with --frontal-parcel.")
    parser.add_argument("--template-cifti", required=True)
    parser.add_argument(
        "--frontal-parcel", action="append", default=[],
        help="Bilateral Glasser short name restricting the candidate set to "
             "these parcels; repeatable. Omit to run over every cortical "
             "grayordinate.",
    )
    parser.add_argument(
        "--exclude-roi", action="append", default=[], metavar="NAME=MASK_NII",
        help="Named nonzero CIFTI mask (e.g. a seed's own ROI) dropped from "
             "the candidate set; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seed_paths = dict(item.split("=", 1) for item in args.seed)
    seeds = {name: np.load(path) for name, path in seed_paths.items()}
    names = list(seeds)

    template = nib.load(args.template_cifti)
    bm_axis = template.header.get_axis(1)
    n_gray = len(bm_axis)
    for name, values in seeds.items():
        if values.shape != (n_gray,):
            raise ValueError(f"Seed {name!r} map has shape {values.shape}, expected {(n_gray,)}")

    if args.frontal_parcel:
        if not args.glasser_dlabel:
            raise ValueError("--frontal-parcel requires --glasser-dlabel")
        parcels = load_glasser_parcels(args.glasser_dlabel, bm_axis)
        matching = [
            indices for name, indices in parcels.items()
            if name.removeprefix("L_").removeprefix("R_").removesuffix("_ROI") in args.frontal_parcel
        ]
        known = {name.removeprefix("L_").removeprefix("R_").removesuffix("_ROI") for name in parcels}
        missing = sorted(set(args.frontal_parcel) - known)
        if missing:
            raise KeyError(f"Unknown frontal parcels: {missing}")
        candidate_indices = np.unique(np.concatenate(matching))
    else:
        candidate_indices = np.arange(n_gray)

    exclude_paths = dict(item.split("=", 1) for item in args.exclude_roi)
    if exclude_paths:
        excluded = np.zeros(n_gray, dtype=bool)
        for path in exclude_paths.values():
            excluded |= _load_mask(Path(path), n_gray)
        candidate_indices = candidate_indices[~excluded[candidate_indices]]

    stacked = np.stack([seeds[name] for name in names])              # (n_seeds, n_gray)
    candidate_values = stacked[:, candidate_indices]                  # (n_seeds, n_candidates)
    winner = np.argmax(candidate_values, axis=0)
    winner_value = candidate_values[winner, np.arange(len(winner))]
    labelled = winner_value > 0.0
    n_unlabelled = int((~labelled).sum())

    label_map = np.full(n_gray, np.nan, dtype=np.float32)
    value_map = np.full(n_gray, np.nan, dtype=np.float32)
    label_map[candidate_indices[labelled]] = winner[labelled] + 1
    value_map[candidate_indices[labelled]] = winner_value[labelled]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_dscalar(
        np.stack([label_map, value_map]),
        ["winner_take_all_source_index", "winner_take_all_raw_r"],
        template, args.output_dir / "frontal_gradient_winner_take_all.dscalar.nii",
    )

    n_candidates = len(candidate_indices)
    rows = []
    for index, name in enumerate(names):
        mask = labelled & (winner == index)
        rows.append({
            "seed": name,
            "seed_index": index + 1,
            "n_vertices": int(mask.sum()),
            "fraction_of_candidate_cortex": float(mask.sum() / n_candidates),
            "mean_raw_r_where_dominant": (
                float(winner_value[mask].mean()) if mask.any() else float("nan")
            ),
        })
    rows.append({
        "seed": "unlabelled",
        "seed_index": 0,
        "n_vertices": n_unlabelled,
        "fraction_of_candidate_cortex": float(n_unlabelled / n_candidates),
        "mean_raw_r_where_dominant": float("nan"),
    })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "frontal_gradient_summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "seeds": names,
        "seed_maps": {name: str(Path(path).resolve()) for name, path in seed_paths.items()},
        "frontal_parcels": args.frontal_parcel or "all_cortex",
        "exclude_rois": {name: str(Path(path).resolve()) for name, path in exclude_paths.items()},
        "n_candidate_vertices": int(n_candidates),
        "n_unlabelled_vertices": n_unlabelled,
        "labeling": (
            "argmax raw Pearson r across the seed ROI-mean timecourses; "
            "unlabelled (NaN) where every seed's r is <= 0"
        ),
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
