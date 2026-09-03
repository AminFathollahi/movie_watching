#!/usr/bin/env python3
"""Summarize held-out PE-AV CCA fits across requested LBOE counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _finite_corr(first: np.ndarray, second: np.ndarray) -> float:
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 2:
        return float("nan")
    return float(np.corrcoef(first[keep], second[keep])[0, 1])


def _load_model(count: int, root: Path, roi_a: str, roi_p: str) -> dict:
    prep = root / "prep"
    maps = {
        "full": np.load(prep / "R2_full.npy"),
        "a": np.load(prep / f"R2_{roi_a}.npy"),
        "p": np.load(prep / f"R2_{roi_p}.npy"),
    }
    band_sizes = np.load(prep / "band_sizes.npy").astype(int)
    return {
        "requested_count": count,
        "root": str(root),
        "roi_a": roi_a,
        "roi_p": roi_p,
        "actual_lboe": {"a": int(band_sizes[0] // 2),
                        "p": int(band_sizes[1] // 2)},
        "mean_heldout_r2": {key: float(np.nanmean(value))
                            for key, value in maps.items()},
        "maps": maps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model", nargs=4, action="append", required=True,
        metavar=("COUNT", "ROOT", "ROI_A", "ROI_P"),
        help="Repeat once per fitted model")
    parser.add_argument("--reference-count", type=int, default=200)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [_load_model(int(count), Path(root), roi_a, roi_p)
              for count, root, roi_a, roi_p in args.model]
    by_count = {model["requested_count"]: model for model in models}
    if args.reference_count not in by_count:
        raise ValueError(f"Reference count {args.reference_count} was not supplied")
    reference = by_count[args.reference_count]

    for model in models:
        model["comparison_to_reference"] = {
            key: {
                "spatial_r": _finite_corr(model["maps"][key], reference["maps"][key]),
                "rmse": float(np.sqrt(np.nanmean(
                    (model["maps"][key] - reference["maps"][key]) ** 2))),
            }
            for key in ("full", "a", "p")
        }
        del model["maps"]

    winner = max(models, key=lambda item: item["mean_heldout_r2"]["full"])
    reference_mean = reference["mean_heldout_r2"]["full"]
    best_mean = winner["mean_heldout_r2"]["full"]
    if winner["requested_count"] == args.reference_count:
        interpretation = (
            f"The capped-{args.reference_count} model has the highest mean held-out "
            "full-model R², so the sensitivity analysis provides no evidence that the "
            "larger spatial basis overfits.")
    else:
        delta = best_mean - reference_mean
        interpretation = (
            f"The {winner['requested_count']}-LBOE model has the highest mean held-out "
            f"full-model R² (Δ={delta:+.6f} versus capped-{args.reference_count}); "
            "prefer it if the improvement is also spatially coherent rather than a "
            "negligible numerical difference.")

    payload = {
        "analysis": "PE-AV top-1% CCA LBOE-count sensitivity",
        "selection_rule": "maximize mean held-out cortical full-model R2",
        "reference_count": args.reference_count,
        "winner_by_rule": winner["requested_count"],
        "interpretation": interpretation,
        "models": sorted(models, key=lambda item: item["requested_count"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    rows = []
    for model in payload["models"]:
        comp = model["comparison_to_reference"]
        rows.append(
            f"| {model['requested_count']} | {model['actual_lboe']['a']} / "
            f"{model['actual_lboe']['p']} | {model['mean_heldout_r2']['full']:.6f} | "
            f"{comp['full']['spatial_r']:.5f} |")
    markdown = f"""# Validation of 200 LBOEs

**Theory.** Laplace-Beltrami eigenfunctions form a coarse-to-fine spatial basis;
more modes preserve finer connectivity structure. Banded ridge regularization is
selected by leave-one-run-out CV, so overfitting is judged on held-out movie TRs,
not by basis count alone.

| Requested | Actual A / P | Mean full R² | Full-map r vs {args.reference_count} |
|---:|---:|---:|---:|
{chr(10).join(rows)}

**Empirical result.** {interpretation} The requested {args.reference_count} is a
per-ROI, per-hemisphere maximum; the posterior ROI reaches its `n_vertices − 2`
safety cap, hence 200 anterior / 126 posterior.
"""
    args.output_markdown.write_text(markdown)


if __name__ == "__main__":
    main()
