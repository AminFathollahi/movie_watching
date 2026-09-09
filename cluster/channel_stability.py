"""Evaluate channel-cluster stability across independent movie runs."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.incremental_av import REPEATED_VALIDATION_CLIPS, _bh_qvalues, sample_metadata
from encoding.shared.compression import (
    heldout_cluster_diagnostics,
    spherical_channel_labels,
)


def permuted_channel_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random channel partition with the same cluster sizes as `labels`."""
    return rng.permutation(labels)


def randomized_membership_null(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_permutations: int,
    random_state: int,
) -> np.ndarray:
    """Randomize channel membership while preserving cluster sizes."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    rng = np.random.default_rng(random_state)
    return np.array([
        adjusted_rand_score(labels_a, permuted_channel_labels(labels_b, rng))
        for _ in range(n_permutations)
    ])


def independent_run_stability(
    embedding: np.ndarray,
    run_ids: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_null: int = 1000,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit clusters independently in each run and compare memberships."""
    run_ids = np.asarray(run_ids)
    labels = {
        str(run): spherical_channel_labels(
            embedding[run_ids == run], n_clusters, random_state
        )
        for run in np.unique(run_ids)
    }
    rows = []
    for pair_index, (run_a, run_b) in enumerate(combinations(labels, 2)):
        observed = adjusted_rand_score(labels[run_a], labels[run_b])
        null = randomized_membership_null(
            labels[run_a], labels[run_b], n_null,
            random_state + 1009 * (pair_index + 1),
        )
        rows.append({
            "run_a": run_a,
            "run_b": run_b,
            "ari": float(observed),
            "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)),
            "permutation_p": float((1 + np.sum(null >= observed)) / (n_null + 1)),
        })
    return rows, labels


def heldout_run_diagnostics(
    embedding: np.ndarray,
    run_ids: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit memberships on other runs and score their held-out coherence."""
    run_ids = np.asarray(run_ids)
    rows = []
    labels = {}
    for run in np.unique(run_ids):
        test = run_ids == run
        fold_labels = spherical_channel_labels(
            embedding[~test], n_clusters, random_state
        )
        labels[str(run)] = fold_labels
        row = heldout_cluster_diagnostics(embedding[test], fold_labels)
        row["test_run"] = str(run)
        rows.append(row)
    return rows, labels


def select_resolution_by_heldout_reconstruction(
    summary: pd.DataFrame, coverage: float = 0.9
) -> dict:
    """Smallest k whose held-out reconstruction reaches `coverage` of the best achievable
    MSE reduction on the grid. Training-only: memberships are frozen on training runs and
    only scored, never refit, on the held-out run. Replaces silhouette-style selection,
    which is maximized by few well-separated clusters and picks k=2 regardless of signal."""
    ordered = summary.sort_values("n_clusters")
    n_clusters = ordered["n_clusters"].to_numpy()
    mse = ordered["heldout_reconstruction_mse"].to_numpy()
    total_gain = mse[0] - mse.min()
    if total_gain <= 1e-12:
        chosen = int(n_clusters[0])
    else:
        gain_fraction = (mse[0] - mse) / total_gain
        chosen = int(n_clusters[np.argmax(gain_fraction >= coverage)])
    return {
        "selected_n_clusters": chosen,
        "criterion": (
            f"smallest k reaching {coverage:.0%} of the held-out reconstruction MSE "
            "reduction achieved on the grid (frozen training memberships)"
        ),
        "coverage": coverage,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embedding-path", required=True)
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument(
        "--exclude-video-ids", default=",".join(REPEATED_VALIDATION_CLIPS)
    )
    parser.add_argument("--clusters", nargs="+", type=int, default=(2, 4, 8, 16, 32, 64))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--n-null", type=int, default=1000)
    return parser.parse_args(argv)


def summarize_results(output: Path) -> pd.DataFrame:
    independent_path = output / "independent_run_stability.csv"
    independent = pd.read_csv(independent_path)
    independent["permutation_q_bh"] = _bh_qvalues(independent["permutation_p"])
    independent["significant_fdr_05"] = independent["permutation_q_bh"] <= 0.05
    independent.to_csv(independent_path, index=False)

    heldout = pd.read_csv(output / "heldout_diagnostics.csv")
    seed = pd.read_csv(output / "seed_stability.csv")
    summary = independent.groupby("n_clusters", as_index=False).agg(
        independent_ari_mean=("ari", "mean"),
        independent_ari_sd=("ari", "std"),
        null_q95_mean=("null_q95", "mean"),
        n_run_comparisons=("ari", "size"),
        n_fdr_significant=("significant_fdr_05", "sum"),
    )
    seed_summary = seed.groupby("n_clusters", as_index=False).agg(
        seed_ari_mean=("ari", "mean"), seed_ari_sd=("ari", "std"),
    )
    heldout_summary = heldout.groupby("n_clusters", as_index=False).agg(
        heldout_within_correlation=("mean_within_cluster_correlation", "mean"),
        heldout_between_abs_correlation=("mean_absolute_between_cluster_correlation", "mean"),
        heldout_reconstruction_mse=("reconstruction_mse", "mean"),
        heldout_effective_dimension=("effective_dimension", "mean"),
    )
    summary = summary.merge(seed_summary, on="n_clusters").merge(
        heldout_summary, on="n_clusters"
    )
    summary.to_csv(output / "summary.csv", index=False)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = summary["n_clusters"].to_numpy()
    axes[0, 0].errorbar(
        x, summary["independent_ari_mean"], yerr=summary["independent_ari_sd"],
        color="#3266a8", marker="o", label="Observed ARI",
    )
    axes[0, 0].plot(x, summary["null_q95_mean"], "--", color="#6f7478", label="Null 95th percentile")
    axes[0, 0].set_title("Independent-run membership stability")
    axes[0, 0].set_ylabel("Adjusted Rand index")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].errorbar(
        x, summary["seed_ari_mean"], yerr=summary["seed_ari_sd"],
        color="#b17800", marker="s",
    )
    axes[0, 1].set_title("Training-seed stability")
    axes[0, 1].set_ylabel("Adjusted Rand index")

    axes[1, 0].plot(
        x, summary["heldout_within_correlation"], color="#3266a8",
        marker="o", label="Within cluster",
    )
    axes[1, 0].plot(
        x, summary["heldout_between_abs_correlation"], color="#b17800",
        marker="s", linestyle="--", label="Between clusters (absolute)",
    )
    axes[1, 0].set_title("Held-out channel coherence")
    axes[1, 0].set_ylabel("Mean correlation")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        x, summary["heldout_effective_dimension"], color="#3266a8", marker="o",
        label="Effective dimension",
    )
    axes[1, 1].plot(x, x, "--", color="#6f7478", label="Nominal dimension")
    axes[1, 1].set_title("Held-out effective dimensionality")
    axes[1, 1].set_ylabel("Dimensions")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, labels=[str(value) for value in x])
        axis.set_xlabel("Number of clusters")
        axis.grid(axis="y", color="#d8dde3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "channel_stability.png", dpi=180)
    plt.close(figure)
    return summary


def run(args) -> Path:
    embedding = np.load(args.embedding_path)
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, args.bin_sec, args.skip_sec)
    if embedding.ndim != 2 or embedding.shape[0] != len(metadata):
        raise ValueError("Embedding rows do not align with timing metadata")

    excluded = {item.strip() for item in args.exclude_video_ids.split(",") if item.strip()}
    keep = ~metadata["video_id"].isin(excluded).to_numpy()
    embedding = embedding[keep]
    metadata = metadata.loc[keep].reset_index(drop=True)
    run_ids = metadata["run_id"].to_numpy()
    if np.unique(run_ids).size < 2:
        raise ValueError("At least two runs are required")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output / "samples.csv", index=False)
    independent_rows = []
    heldout_rows = []
    seed_rows = []
    saved_labels = {}

    for n_clusters in args.clusters:
        fold_labels_by_seed = {}
        for seed in args.random_seeds:
            stability, run_labels = independent_run_stability(
                embedding, run_ids, n_clusters, seed, args.n_null
            )
            for row in stability:
                row.update(n_clusters=n_clusters, seed=seed)
            independent_rows.extend(stability)

            diagnostics, fold_labels = heldout_run_diagnostics(
                embedding, run_ids, n_clusters, seed
            )
            for row in diagnostics:
                row.update(n_clusters=n_clusters, seed=seed)
            heldout_rows.extend(diagnostics)
            fold_labels_by_seed[seed] = fold_labels

            for run, labels in run_labels.items():
                saved_labels[f"independent_k{n_clusters}_seed{seed}_run{run}"] = labels
            for run, labels in fold_labels.items():
                saved_labels[f"train_other_k{n_clusters}_seed{seed}_run{run}"] = labels

        for run in map(str, np.unique(run_ids)):
            for seed_a, seed_b in combinations(args.random_seeds, 2):
                seed_rows.append({
                    "n_clusters": n_clusters,
                    "test_run": run,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "ari": float(adjusted_rand_score(
                        fold_labels_by_seed[seed_a][run],
                        fold_labels_by_seed[seed_b][run],
                    )),
                })

    pd.DataFrame(independent_rows).to_csv(output / "independent_run_stability.csv", index=False)
    pd.DataFrame(heldout_rows).to_csv(output / "heldout_diagnostics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "seed_stability.csv", index=False)
    np.savez_compressed(output / "channel_labels.npz", **saved_labels)
    (output / "manifest.json").write_text(json.dumps({
        "embedding_path": str(Path(args.embedding_path).resolve()),
        "n_samples": int(len(metadata)),
        "n_channels": int(embedding.shape[1]),
        "runs": [str(run) for run in np.unique(run_ids)],
        "excluded_video_ids": sorted(excluded),
        "clusters": args.clusters,
        "random_seeds": args.random_seeds,
        "n_null": args.n_null,
        "null_definition": "Permutation of channel memberships with cluster sizes fixed",
    }, indent=2) + "\n")
    summary = summarize_results(output)
    selection = select_resolution_by_heldout_reconstruction(summary)
    (output / "selected_resolution.json").write_text(json.dumps(selection, indent=2) + "\n")
    return output


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
