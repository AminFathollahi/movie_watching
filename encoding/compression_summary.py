"""Compact G_d = R2(A,V,C_d(J)) - R2(A,V) summary across compression methods, ROIs, models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DIMS = (2, 4, 8, 16, 32, 64)


def build(output_root: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        inf_path = output_root / model / "runwise_incremental_av_bin5s_skip5s" / "inference.csv"
        inf = pd.read_csv(inf_path)
        for roi in sorted(inf["roi"].unique()):
            full = inf[(inf.method == "full") & (inf.roi == roi)].iloc[0]
            g_full = float(full["pooled_delta_r2_mean"])
            reliable = bool(full["bootstrap_ci_low"] > 0)
            by_dim = (
                inf[(inf.method != "full") & (inf.roi == roi)]
                .groupby(["method", "dimension"], as_index=False)["pooled_delta_r2_mean"]
                .mean().rename(columns={"pooled_delta_r2_mean": "g_d"})
            )
            for method, frame in by_dim.groupby("method"):
                frame = frame.sort_values("dimension")
                if not reliable or g_full <= 0:
                    d90 = "n/a (full G not reliably positive)"
                else:
                    hit = frame.loc[frame["g_d"] >= 0.9 * g_full, "dimension"]
                    d90 = str(int(hit.min())) if len(hit) else "not reached up to d=64"
                row = {
                    "model": model, "roi": roi, "method": method,
                    "g_full": g_full, "full_g_reliable": reliable,
                    "d_reaching_90pct_full": d90,
                }
                for d in DIMS:
                    match = frame.loc[frame["dimension"] == d, "g_d"]
                    row[f"g_d{d}"] = float(match.iloc[0]) if len(match) else None
                rows.append(row)
    cols = ["model", "roi", "method", "g_full", "full_g_reliable", "d_reaching_90pct_full"]
    cols += [f"g_d{d}" for d in DIMS]
    return pd.DataFrame(rows)[cols]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True,
                         help="incremental_av group_average dir containing one subdir per model")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    table = build(output_root, args.models)
    path = output_root / "compression_summary.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print("saved:", path)


if __name__ == "__main__":
    main()
