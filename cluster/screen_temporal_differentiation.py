"""Cheap screen: for every candidate clustering in a model-selection sweep, compute the
mean/max off-diagonal |Pearson r| among that solution's own cluster mean profiles.

A solution whose clusters all correlate above ~0.9 is temporally redundant (one shared
stimulus-locked signal split into near-identical pieces), regardless of its selection_score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from paths import OUTPUTS  # noqa: E402
from vertex_clustering import (  # noqa: E402
    labels_path, cluster_profiles, profile_correlation_summary, vertex_exclude_labels,
)
from io_cluster import load_group_average  # noqa: E402
from channel_timeseries_clustering import (  # noqa: E402
    channel_model_selection_dir, load_channel_timeseries, parse_args as channel_parse_args,
)
from rsa.shared.rsa_utils import preprocess_fmri  # noqa: E402

FAMILIES = ("peav", "nemotron_layer18_mp")
OUTPUT_DIR = OUTPUTS / "cluster"


def screen_table(ms_dir: Path, filename: str, units_by_bins: np.ndarray,
                 side: str, family: str) -> pd.DataFrame:
    table = pd.read_csv(ms_dir / "selected_clusterings.csv")
    rows = []
    for _, row in table.iterrows():
        lf = labels_path(ms_dir, row, filename)
        if not lf.exists():
            continue
        labels = np.load(lf)
        if labels.shape[0] != units_by_bins.shape[0]:
            continue
        exclude = vertex_exclude_labels(lf) if side == "vertex" else (-1,)
        profiles, ids, counts, noise = cluster_profiles(units_by_bins, labels, exclude=exclude)
        summ = profile_correlation_summary(profiles)
        rows.append({
            "side": side, "family": family,
            "reducer_tag": row["reducer_tag"], "cluster_tag": row["full_fit_cluster_tag"],
            "reduction_method": row["reduction_method"], "n_components": int(row["n_components"]),
            "cluster_method": row["method"], "selection_role": row["selection_role"],
            "selection_score": float(row["selection_score"]),
            "n_clusters": len(ids), "n_noise": noise,
            "mean_abs_off_diag": summ["mean_abs_off_diag"], "max_abs_off_diag": summ["max_abs_off_diag"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = channel_parse_args(["--family", "peav"])

    vertex_ms_dir = OUTPUT_DIR / "group_average" / "_vertex" / "norm-zscore_raw"
    X, vertex_run_trs, _ = load_group_average()
    timing_df = pd.read_csv(args.timing_csv)
    fmri_binned = preprocess_fmri(
        fmri_continuous=X, timing_df=timing_df, run_trs=vertex_run_trs,
        bin_sec=args.bin_sec, tr=args.tr, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    del X
    tables = [screen_table(vertex_ms_dir, "spatial_vertex_labels.npy", fmri_binned.T, "vertex", "group_average")]

    for family in FAMILIES:
        fam_args = channel_parse_args(["--family", family])
        channel_ts, _ = load_channel_timeseries(fam_args)
        ch_ms_dir = channel_model_selection_dir(OUTPUT_DIR, family)
        tables.append(screen_table(ch_ms_dir, "channel_labels.npy", channel_ts, "channel", family))

    out = pd.concat(tables, ignore_index=True).sort_values(["side", "family", "n_clusters"])
    out_path = OUTPUT_DIR / "_channel_vertex_alignment_screen.csv"
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
