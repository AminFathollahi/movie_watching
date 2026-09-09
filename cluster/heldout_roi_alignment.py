from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.channel_stability import permuted_channel_labels
from encoding.incremental_av import REPEATED_VALIDATION_CLIPS, _bh_qvalues, sample_metadata
from encoding.shared.compression import spherical_channel_labels
from encoding.shared.encoding_utils import build_fmri_arrays
from rsa.glasser import load_glasser_parcels


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--fmri-path", type=Path, required=True)
    parser.add_argument("--run-trs", type=Path, required=True)
    parser.add_argument("--glasser-dlabel", type=Path, required=True)
    parser.add_argument("--roi", action="append", required=True, metavar="NAME=PARCELS")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--n-shifts", type=int, default=5000)
    parser.add_argument("--n-random-partitions", type=int, default=2000)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument(
        "--exclude-video-ids", default=",".join(REPEATED_VALIDATION_CLIPS)
    )
    return parser.parse_args(argv)


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return centered / scale


def _cluster_profiles(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:
    profiles = np.stack([
        embedding[:, labels == cluster].mean(axis=1)
        for cluster in range(int(labels.max()) + 1)
    ])
    return _zscore_rows(profiles)


def _select_and_score(
    brain_train: np.ndarray,
    brain_test: np.ndarray,
    channel_train: np.ndarray,
    channel_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick each ROI's best-matching channel cluster on training data and
    score that fixed match on held-out data. No refitting on test data."""
    train_r = brain_train @ channel_train.T / brain_train.shape[1]
    test_matrix = brain_test @ channel_test.T / brain_test.shape[1]
    selected = np.argmax(np.abs(train_r), axis=1)
    observed = test_matrix[np.arange(len(selected)), selected]
    return selected, train_r[np.arange(len(selected)), selected], observed


def select_and_test(
    brain_train: np.ndarray,
    brain_test: np.ndarray,
    channel_train: np.ndarray,
    channel_test: np.ndarray,
    n_shifts: int,
    random_state: int,
) -> list[dict]:
    selected, train_r_selected, observed = _select_and_score(
        brain_train, brain_test, channel_train, channel_test
    )
    rng = np.random.default_rng(random_state)
    exceed = np.zeros(len(selected), dtype=int)
    for _ in range(n_shifts):
        shift = int(rng.integers(1, brain_test.shape[1]))
        shifted = np.roll(channel_test, shift, axis=1)
        null_matrix = brain_test @ shifted.T / brain_test.shape[1]
        null_selected = null_matrix[np.arange(len(selected)), selected]
        exceed += np.abs(null_selected) >= np.abs(observed)
    return [{
        "selected_channel_cluster": int(channel),
        "train_r": float(train_r_selected[index]),
        "test_r": float(observed[index]),
        "shift_p_two_sided": float((exceed[index] + 1) / (n_shifts + 1)),
    } for index, channel in enumerate(selected)]


def random_partition_control(
    embedding_train: np.ndarray,
    embedding_test: np.ndarray,
    brain_train: np.ndarray,
    brain_test: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    random_state: int,
) -> np.ndarray:
    """|test_r| from size-matched random channel partitions, same selection
    and held-out scoring as the real clustering. Rows are permutations,
    columns are ROIs."""
    rng = np.random.default_rng(random_state)
    abs_r = np.empty((n_permutations, brain_train.shape[0]))
    for draw in range(n_permutations):
        random_labels = permuted_channel_labels(labels, rng)
        channel_train = _cluster_profiles(embedding_train, random_labels)
        channel_test = _cluster_profiles(embedding_test, random_labels)
        _, _, observed = _select_and_score(brain_train, brain_test, channel_train, channel_test)
        abs_r[draw] = np.abs(observed)
    return abs_r


def _roi_indices(args, fmri_axis) -> dict[str, np.ndarray]:
    parcels = load_glasser_parcels(args.glasser_dlabel, fmri_axis)
    short = {
        name.removeprefix("L_").removeprefix("R_").removesuffix("_ROI"): indices
        for name, indices in parcels.items()
    }
    output = {}
    for item in args.roi:
        name, payload = item.split("=", 1)
        requested = [value.strip() for value in payload.split(",") if value.strip()]
        missing = sorted(set(requested) - set(short))
        if missing:
            raise KeyError(f"Unknown parcels for {name}: {missing}")
        matching = [indices for parcel, indices in parcels.items()
                    if parcel.removeprefix("L_").removeprefix("R_").removesuffix("_ROI")
                    in requested]
        output[name] = np.unique(np.concatenate(matching))
    return output


def _plot_results(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    colors = dict(zip(pd.unique(table["roi"]), ("#2878B5", "#D95319")))
    for roi, rows in table.groupby("roi", sort=False):
        axes[0].scatter(
            rows["train_r"], rows["test_r"], s=24, alpha=0.75,
            color=colors[roi], label=roi,
        )
        axes[1].scatter(
            rows["test_run"] + (0.05 if roi == "auditory" else -0.05),
            np.abs(rows["test_r"]), s=24, alpha=0.75, color=colors[roi], label=roi,
        )
    axes[0].axhline(0, color="0.7", linewidth=0.8)
    axes[0].axvline(0, color="0.7", linewidth=0.8)
    axes[0].set(xlabel="Training correlation", ylabel="Held-out correlation",
                title="Selected match generalization")
    axes[1].set(xlabel="Held-out run", ylabel="Absolute held-out correlation",
                title="Effect size by run", xticks=(1, 2, 3, 4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args) -> Path:
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, args.bin_sec, args.skip_sec)
    embedding = np.load(args.embedding_path)
    if embedding.shape[0] != len(metadata):
        raise ValueError("Embedding rows do not match timing metadata")
    excluded = {value.strip() for value in args.exclude_video_ids.split(",") if value.strip()}
    keep = ~metadata["video_id"].isin(excluded).to_numpy()
    embedding = embedding[keep]
    metadata = metadata.loc[keep].sort_values(
        ["run_id", "row_index"], kind="stable"
    ).reset_index(drop=True)

    fmri, _, _ = build_fmri_arrays(
        str(args.fmri_path), str(args.run_trs), timing, sorted(excluded),
        args.bin_sec, args.tr, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    fmri_axis = nib.load(str(args.fmri_path)).header.get_axis(1)
    roi_indices = _roi_indices(args, fmri_axis)
    roi_names = list(roi_indices)
    brain = np.column_stack([fmri[:, indices].mean(axis=1) for indices in roi_indices.values()])
    run_ids = metadata["run_id"].to_numpy()
    rows = []
    memberships = {}
    for test_run in pd.unique(run_ids):
        test = run_ids == test_run
        for seed in args.random_seeds:
            labels = spherical_channel_labels(
                embedding[~test], args.n_clusters, seed,
            )
            memberships[f"run{test_run}_seed{seed}"] = labels
            brain_train = _zscore_rows(brain[~test].T)
            brain_test = _zscore_rows(brain[test].T)
            channel_train = _cluster_profiles(embedding[~test], labels)
            channel_test = _cluster_profiles(embedding[test], labels)
            results = select_and_test(
                brain_train, brain_test,
                channel_train, channel_test, args.n_shifts,
                seed + 1009 * int(test_run),
            )
            random_abs_r = random_partition_control(
                embedding[~test], embedding[test], brain_train, brain_test,
                labels, args.n_random_partitions,
                seed + 7919 * int(test_run),
            )
            for roi_index, (roi, result) in enumerate(zip(roi_names, results)):
                null = random_abs_r[:, roi_index]
                observed = abs(result["test_r"])
                result.update(
                    roi=roi, test_run=test_run, seed=seed,
                    random_partition_null_mean_abs_r=float(null.mean()),
                    random_partition_null_q95_abs_r=float(np.quantile(null, 0.95)),
                    random_partition_p_two_sided=float(
                        (1 + np.sum(null >= observed)) / (args.n_random_partitions + 1)
                    ),
                )
                rows.append(result)
    table = pd.DataFrame(rows)
    table["same_sign"] = np.sign(table["train_r"]) == np.sign(table["test_r"])
    table["shift_q_bh"] = _bh_qvalues(table["shift_p_two_sided"])
    table["significant_fdr_05"] = table["shift_q_bh"] <= 0.05
    table["random_partition_q_bh"] = _bh_qvalues(table["random_partition_p_two_sided"])
    table["significant_vs_random_partition_fdr_05"] = table["random_partition_q_bh"] <= 0.05
    summary = table.groupby("roi", as_index=False).agg(
        mean_train_r=("train_r", "mean"),
        mean_test_r=("test_r", "mean"),
        mean_abs_test_r=("test_r", lambda values: float(np.mean(np.abs(values)))),
        test_r_sd=("test_r", "std"),
        sign_agreement=("same_sign", "mean"),
        n_tests=("test_r", "size"),
        n_fdr_significant=("significant_fdr_05", "sum"),
        random_partition_null_mean_abs_r=("random_partition_null_mean_abs_r", "mean"),
        n_significant_vs_random_partition=("significant_vs_random_partition_fdr_05", "sum"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "heldout_alignment.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    np.savez_compressed(args.output_dir / "channel_memberships.npz", **memberships)
    _plot_results(table, args.output_dir / "heldout_alignment.png")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "embedding_path": str(args.embedding_path.resolve()),
        "fmri_path": str(args.fmri_path.resolve()),
        "n_clusters": args.n_clusters,
        "random_seeds": args.random_seeds,
        "n_shifts": args.n_shifts,
        "n_random_partitions": args.n_random_partitions,
        "excluded_video_ids": sorted(excluded),
        "selection": "maximum absolute training-run correlation per ROI",
        "evaluation": "fixed channel membership and match on the held-out run",
        "null": "one synchronized nonzero circular shift for all channel profiles",
        "random_partition_null": (
            "size-matched random channel partitions (cluster sizes fixed to the "
            "real clustering) run through the identical training-run selection "
            "and held-out scoring"
        ),
    }, indent=2) + "\n")
    return args.output_dir


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
