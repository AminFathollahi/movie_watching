"""Cheap screen: for every candidate clustering in a model-selection sweep, compute the
mean/max off-diagonal |Pearson r| among that solution's own cluster mean profiles.

A solution whose clusters all correlate above ~0.9 is temporally redundant (one shared
stimulus-locked signal split into near-identical pieces) and cannot support a differential
channel<->vertex alignment test, regardless of its silhouette-based selection_score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from channel_vertex_alignment import (  # noqa: E402
    parse_args, labels_path, cluster_profiles, profile_correlation_summary, vertex_exclude_labels,
)
from io_cluster import load_group_average  # noqa: E402
from channel_timeseries_clustering import channel_model_selection_dir, load_channel_timeseries  # noqa: E402
from rsa.shared.rsa_utils import preprocess_fmri  # noqa: E402

FAMILIES = ("peav", "nemotron_layer18_mp", "topoomni_layer18_sheet_mp")
OUTPUT_DIR = Path("/home/amin/Research/Representation/Movie/outputs/cluster")


def screen_table(ms_dir: Path, filename: str, units_by_bins: np.ndarray,
                 side: str, family: str) -> tuple[pd.DataFrame, list[dict]]:
    table = pd.read_csv(ms_dir / "selected_clusterings.csv")
    rows = []
    store = []
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
        store.append({"profiles": profiles, "ids": ids, "counts": counts})
    return pd.DataFrame(rows), store


def plot_profile_correlation(corr: np.ndarray, ids: list[int], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(ids)
    figure, axis = plt.subplots(figsize=(max(4, 0.5 * n + 2), max(3.5, 0.5 * n + 1.5)), dpi=160)
    image = axis.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(n)); axis.set_xticklabels([f"c{c}" for c in ids], rotation=90)
    axis.set_yticks(range(n)); axis.set_yticklabels([f"c{c}" for c in ids])
    for i in range(n):
        for j in range(n):
            axis.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Pearson r")
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def write_profile_correlation_outputs(side: str, family: str, reducer_tag: str, cluster_tag: str,
                                      n_clusters: int, profiles: np.ndarray, ids: list[int],
                                      counts: dict[int, int]) -> Path:
    out_dir = OUTPUT_DIR / "profile_correlations" / side / family / f"{reducer_tag}_{cluster_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_bins = profiles.shape[1]
    corr = (profiles @ profiles.T) / n_bins
    labels = [f"cluster_{c}" for c in ids]
    pd.DataFrame(corr, index=labels, columns=labels).to_csv(out_dir / "profile_correlation.csv")

    title = f"{family} | {reducer_tag} | {cluster_tag} | n_clusters={n_clusters}"
    plot_profile_correlation(corr, ids, title, out_dir / "profile_correlation.png")

    np.save(out_dir / "cluster_profiles.npy", profiles)
    pd.DataFrame({"cluster_id": ids, "n_units": [counts[c] for c in ids]}).to_csv(
        out_dir / "cluster_sizes.csv", index=False
    )
    return out_dir


def save_top_n(df: pd.DataFrame, store: list[dict], side: str, family: str, top_n: int) -> None:
    candidates = df[df["n_clusters"] >= 3].sort_values("mean_abs_off_diag")
    for position in candidates.index[:top_n]:
        row = df.loc[position]
        entry = store[position]
        out_dir = write_profile_correlation_outputs(
            side, family, row["reducer_tag"], row["cluster_tag"], int(row["n_clusters"]),
            entry["profiles"], entry["ids"], entry["counts"],
        )
        print(f"  top-n [{side}/{family}] mean_abs_off_diag={row['mean_abs_off_diag']:.3f} -> {out_dir}")


def parse_screen_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=10,
                        help="Per (side, family), save full profile-correlation matrices for the "
                             "N solutions (n_clusters>=3) with the lowest mean_abs_off_diag")
    return parser.parse_args(argv)


def main() -> None:
    screen_args = parse_screen_args()
    args = parse_args(["--family", "peav"])

    vertex_ms_dir = OUTPUT_DIR / "group_average" / "_vertex" / "norm-zscore_raw"
    X, vertex_run_trs, _ = load_group_average(args.vertex_cifti, args.vertex_run_trs)
    timing_df = pd.read_csv(args.timing_csv)
    fmri_binned = preprocess_fmri(
        fmri_continuous=X, timing_df=timing_df, run_trs=vertex_run_trs,
        bin_sec=args.bin_sec, tr=args.tr, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    del X
    groups = [("vertex", "group_average",
              *screen_table(vertex_ms_dir, "spatial_vertex_labels.npy", fmri_binned.T, "vertex", "group_average"))]

    for family in FAMILIES:
        fam_args = parse_args(["--family", family])
        channel_ts, _ = load_channel_timeseries(fam_args)
        ch_ms_dir = channel_model_selection_dir(OUTPUT_DIR, family)
        groups.append((
            "channel", family,
            *screen_table(ch_ms_dir, "channel_labels.npy", channel_ts, "channel", family),
        ))

    out = pd.concat([g[2] for g in groups], ignore_index=True).sort_values(["side", "family", "n_clusters"])
    out_path = OUTPUT_DIR / "_channel_vertex_alignment_screen.csv"
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")

    print(f"\nSaving top-{screen_args.top_n} profile-correlation matrices per (side, family)...")
    for side, family, df, store in groups:
        save_top_n(df, store, side, family, screen_args.top_n)


if __name__ == "__main__":
    main()
