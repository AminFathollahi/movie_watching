"""Tune channel-timeseries embeddings, then tune clustering on the winners.

Channel analogue of ``vertex_model_selection.py``: the sweep,
selection-score, and elbow-selection math (reducer grid, dimension
criteria, elbow selection, clustering grid, clustering selection score) are
all imported unchanged from that module and from
``channel_timeseries_clustering.py``. Only the two genuinely
grayordinate-specific pieces differ: there is no CIFTI/BrainModelAxis input
to validate against, and the final full-channel-set cluster-label maps are
written to CSV instead of ``.dlabel.nii``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from io_cluster import write_channel_labels_csv  # noqa: E402
from channel_timeseries_clustering import (  # noqa: E402
    FAMILIES, OUTPUT_DIR, EMBEDDINGS_DIR, TIMING_CSV, RUN_TRS, _label_names,
    channel_model_selection_dir, load_channel_timeseries,
)
from vertex_model_selection import (  # noqa: E402
    REDUCTION_METHODS, CLUSTER_METHODS, _save_table, run_reducer_sweep,
    select_reducers, cluster_grid, cluster_tag, _fit_cluster,
    cluster_selection_score,
)
from vertex_clustering import clustering_report, regress_out_global, _zscore_1d  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _csv_values(text: str, cast) -> list:
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid values cannot be empty")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--timing-csv", default=TIMING_CSV)
    parser.add_argument("--run-trs", default=RUN_TRS)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--features", default=None,
                        help="Defaults to this family's baseline channel-features cache "
                             "(run channel_timeseries_clustering.py first)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--reductions", nargs="+", choices=REDUCTION_METHODS,
                        default=list(REDUCTION_METHODS))
    parser.add_argument("--components", type=lambda x: _csv_values(x, int), default=[2, 3],
                        help="Comma-separated display dimensions; only 2 and 3 are supported")
    parser.add_argument(
        "--latent-components-grid", type=lambda x: _csv_values(x, int),
        default=[2, 3, 4, 5, 6, 8, 10],
        help="Dimensions evaluated for each method's native latent-dimension criterion",
    )
    parser.add_argument("--landmarks-grid", type=lambda x: _csv_values(x, int),
                        default=[1_000, 2_000])
    parser.add_argument("--extension-neighbors-grid", type=lambda x: _csv_values(x, int),
                        default=[4, 8])
    parser.add_argument("--isomap-neighbors-grid", type=lambda x: _csv_values(x, int),
                        default=[5, 15, 30, 50])
    parser.add_argument("--tsne-perplexity-grid", type=lambda x: _csv_values(x, float),
                        default=[15.0, 30.0, 50.0])
    parser.add_argument("--fastica-algorithms", type=lambda x: _csv_values(x, str),
                        default=["parallel", "deflation"])
    parser.add_argument("--fastica-functions", type=lambda x: _csv_values(x, str),
                        default=["logcosh", "exp", "cube"])
    parser.add_argument("--umap-neighbors-grid", type=lambda x: _csv_values(x, int),
                        default=[15, 30, 50])
    parser.add_argument("--umap-min-dist-grid", type=lambda x: _csv_values(x, float),
                        default=[0.0, 0.1, 0.5])
    parser.add_argument("--umap-metrics", type=lambda x: _csv_values(x, str),
                        default=["euclidean"])
    parser.add_argument("--quality-sample-size", type=int, default=2_000)
    parser.add_argument("--quality-neighbors", type=lambda x: _csv_values(x, int),
                        default=[10, 30])
    parser.add_argument("--cluster-sample-size", type=int, default=12_000)
    parser.add_argument("--kmeans-k-grid", type=lambda x: _csv_values(x, int),
                        default=[2, 3, 4, 5, 6, 8, 10, 12, 16, 20])
    parser.add_argument("--hdbscan-min-cluster-size-grid",
                        type=lambda x: _csv_values(x, int),
                        default=[10, 25, 50, 100, 250])
    parser.add_argument("--hdbscan-min-samples-grid",
                        type=lambda x: _csv_values(x, int), default=[5, 10, 25, 50])
    parser.add_argument("--birch-threshold-grid", type=lambda x: _csv_values(x, float),
                        default=[0.2, 0.35, 0.5, 0.75, 1.0, 1.5])
    parser.add_argument("--birch-branching-factor-grid",
                        type=lambda x: _csv_values(x, int), default=[25, 50, 100])
    parser.add_argument("--max-selected-clusters", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-reduction-sweep", action="store_true")
    parser.add_argument("--skip-clustering-sweep", action="store_true")
    parser.add_argument("--regress-global", action="store_true",
                        help="Regress the across-channel mean bin time series out of every "
                             "channel before z-scoring (only applies when --features "
                             "is not given explicitly; cached under a distinct tag)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if any(value not in (2, 3) for value in args.components):
        parser.error("--components may contain only 2 and/or 3")
    for name, values in vars(args).items():
        if name.endswith("_grid") or name in ("components", "quality_neighbors"):
            if any(value < 0 if name == "umap_min_dist_grid" else value <= 0 for value in values):
                parser.error(f"All values in --{name.replace('_', '-')} must be positive")
    return args


def run_channel_clustering_sweeps(selected: pd.DataFrame, features: np.ndarray,
                                  channel_ids: list[str], root: Path,
                                  args: argparse.Namespace) -> pd.DataFrame:
    """Channel analogue of run_clustering_sweeps: same sweep, CSV output for winners."""
    eval_path = root / "clustering_evaluation_indices.npy"
    if eval_path.exists() and not args.force:
        eval_indices = np.load(eval_path)
    else:
        rng = np.random.default_rng(args.random_state + 1)
        eval_indices = np.sort(rng.choice(
            len(features), size=min(args.cluster_sample_size, len(features)), replace=False
        ))
        np.save(eval_path, eval_indices)

    sweep_path = root / "clustering_sweep.csv"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if sweep_path.exists() and not args.force:
        for row in pd.read_csv(sweep_path).to_dict("records"):
            existing[(str(row["reducer_tag"]), str(row["cluster_tag"]))] = row
    all_rows = list(existing.values())
    grid = cluster_grid(args)

    for _, reducer in selected.iterrows():
        reducer_name = str(reducer["tag"])
        embedding = np.load(str(reducer["embedding"]))
        scaled = StandardScaler().fit_transform(embedding).astype(np.float32, copy=False)
        evaluation = scaled[eval_indices]
        log.info("Clustering sweep for %s: %d configurations", reducer_name, len(grid))
        for position, params in enumerate(grid, start=1):
            ctag = cluster_tag(params)
            key = (reducer_name, ctag)
            if key in existing:
                continue
            log.info("  cluster %d/%d: %s", position, len(grid), ctag)
            row: dict[str, Any] = {
                "reducer_tag": reducer_name,
                "reduction_method": reducer["method"],
                "n_components": int(reducer["n_components"]),
                "selection_role": reducer["selection_role"],
                "cluster_tag": ctag,
                **params,
                "status": "ok",
            }
            try:
                labels = _fit_cluster(evaluation, params, args.random_state)
                report = clustering_report(evaluation, labels, args.random_state)
                score = cluster_selection_score(report, args.max_selected_clusters)
                row.update(report, selection_score=score)
            except Exception as exc:
                log.exception("Clustering candidate failed: %s / %s", reducer_name, ctag)
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                           selection_score=float("nan"))
            all_rows.append(row)
            existing[key] = row
            _save_table(all_rows, sweep_path)

    table = pd.DataFrame(all_rows)
    selected_tags = set(selected["tag"].astype(str))
    valid = table[
        (table["status"] == "ok")
        & np.isfinite(table["selection_score"])
        & table["reducer_tag"].astype(str).isin(selected_tags)
    ].copy()
    if valid.empty:
        raise RuntimeError("No clustering configuration completed successfully")
    expected_pairs = {
        (tag, method) for tag in selected_tags for method in CLUSTER_METHODS
    }
    observed_pairs = set(zip(valid["reducer_tag"].astype(str), valid["method"].astype(str)))
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        # A wide grid can select a low-signal reducer for which every hyperparameter
        # of one clusterer degenerates (e.g. BIRCH finds zero valid clusters at any
        # threshold). That reducer/method pair is skipped, not fatal to the whole sweep.
        log.warning(
            "No valid clustering candidate for selected reducer/method pairs (skipped): "
            + ", ".join(f"{tag}/{method}" for tag, method in missing_pairs)
        )
    winners_idx = valid.groupby(["reducer_tag", "method"])["selection_score"].idxmax()
    winners = valid.loc[winners_idx].copy().sort_values(["n_components", "reduction_method", "method"])
    role_lookup = selected.set_index("tag")["selection_role"].to_dict()
    winners["selection_role"] = winners["reducer_tag"].map(role_lookup)
    # Unlike grayordinates, the evaluation sample is the full channel set
    # whenever cluster_sample_size >= n_channels, so HDBSCAN's min_cluster_size
    # never needs full-fit rescaling for these datasets.
    scale_factor = len(features) / len(eval_indices)
    winners["full_fit_min_cluster_size"] = np.nan
    hdbscan_mask = winners["method"] == "hdbscan"
    winners.loc[hdbscan_mask, "full_fit_min_cluster_size"] = (
        winners.loc[hdbscan_mask, "min_cluster_size"] * scale_factor
    ).round().clip(lower=2)
    full_fit_tags = []
    for _, winner in winners.iterrows():
        method = str(winner["method"])
        if method == "kmeans":
            full_params = {"method": method, "n_clusters": int(winner["n_clusters"])}
        elif method == "hdbscan":
            full_params = {
                "method": method,
                "min_cluster_size": int(winner["full_fit_min_cluster_size"]),
                "min_samples": int(winner["min_samples"]),
            }
        else:
            full_params = {
                "method": method, "threshold": float(winner["threshold"]),
                "branching_factor": int(winner["branching_factor"]),
            }
        full_fit_tags.append(cluster_tag(full_params))
    winners["full_fit_cluster_tag"] = full_fit_tags
    winners.to_csv(root / "selected_clusterings.csv", index=False)

    selected_map_records = []
    combined_labels: dict[str, pd.Series] = {}
    for _, winner in winners.iterrows():
        reducer_name = str(winner["reducer_tag"])
        selected_reducer = selected[selected["tag"] == reducer_name].iloc[0]
        embedding = np.load(str(selected_reducer["embedding"]))
        scaled = StandardScaler().fit_transform(embedding).astype(np.float32, copy=False)
        method = str(winner["method"])
        if method == "kmeans":
            params = {"method": method, "n_clusters": int(winner["n_clusters"])}
        elif method == "hdbscan":
            params = {"method": method,
                      "min_cluster_size": int(winner["full_fit_min_cluster_size"]),
                      "min_samples": int(winner["min_samples"])}
        else:
            params = {"method": method, "threshold": float(winner["threshold"]),
                      "branching_factor": int(winner["branching_factor"])}
        ctag = cluster_tag(params)
        config_tag = f"{reducer_name}_{ctag}"
        output_dir = root / "selected_maps" / reducer_name / config_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = output_dir / "channel_labels.npy"
        csv_path = output_dir / "channel_labels.csv"
        report_path = output_dir / "channel_report.json"
        if labels_path.exists() and csv_path.exists() and report_path.exists() and not args.force:
            labels = np.load(labels_path)
        else:
            log.info("Fitting selected full map: %s", config_tag)
            labels = _fit_cluster(scaled, params, args.random_state)
            report = clustering_report(scaled, labels, args.random_state)
            remapped = write_channel_labels_csv(
                labels, channel_ids, str(csv_path),
                label_names=_label_names(labels), map_name=config_tag,
            )
            np.save(labels_path, labels)
            report.update(
                reduction_tag=reducer_name,
                reduction_quality_score=float(selected_reducer["quality_score"]),
                selection_role=str(selected_reducer["selection_role"]),
                cluster_tag=ctag,
                selection_score=float(winner["selection_score"]),
                selection_sample_size=int(len(eval_indices)),
                selection_parameters={
                    "cluster_tag": str(winner["cluster_tag"]),
                    "min_cluster_size": (
                        None if method != "hdbscan" else int(winner["min_cluster_size"])
                    ),
                },
                hdbscan_full_fit_scale_factor=(scale_factor if method == "hdbscan" else None),
                parameters=params,
                csv_unique_keys=[int(value) for value in np.unique(remapped)],
            )
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        selected_map_records.append({
            "reducer_tag": reducer_name, "cluster_method": method,
            "reduction_method": str(selected_reducer["method"]),
            "n_components": int(selected_reducer["n_components"]),
            "selection_role": str(selected_reducer["selection_role"]),
            "full_fit_cluster_tag": ctag, "csv": str(csv_path),
            "report": str(report_path),
        })
        best = "_bestdim" if "latent_best" in str(selected_reducer["selection_role"]) else ""
        column = f"{int(selected_reducer['n_components'])}d{best}_{selected_reducer['method']}_{method}"
        combined_labels[column] = pd.read_csv(csv_path).set_index("channel_index")["cluster_key"]

    (root / "selected_maps_manifest.json").write_text(
        json.dumps(selected_map_records, indent=2, sort_keys=True) + "\n"
    )
    combined = pd.DataFrame({"channel_id": channel_ids})
    combined.index.name = "channel_index"
    for column, series in combined_labels.items():
        combined[column] = series.reindex(combined.index).to_numpy()
    combined.to_csv(root / "selected_clusterings_combined.csv")
    return winners


def _build_global_regressed_features(args: argparse.Namespace) -> Path:
    """Cache channel features with the across-channel mean regressed out first."""
    out_dir = (Path(args.output_dir) / args.family / "_channel_timeseries" /
               "norm-zscore_raw_globalregressed")
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "channel_timeseries_features.npy"
    channel_ids_path = out_dir / "channel_ids.json"
    if features_path.exists() and channel_ids_path.exists() and not args.force:
        return features_path
    timeseries, channel_ids = load_channel_timeseries(args)
    regressor = _zscore_1d(timeseries.mean(axis=0))
    residual = regress_out_global(timeseries, regressor)
    np.save(features_path, residual)
    channel_ids_path.write_text(json.dumps(channel_ids, indent=2) + "\n")
    (out_dir / "channel_timeseries_features_report.json").write_text(
        json.dumps({"family": args.family, "regress_global": True}, indent=2, sort_keys=True) + "\n"
    )
    return features_path


def run(args: argparse.Namespace) -> Path:
    if args.features is None:
        if args.regress_global:
            args.features = str(_build_global_regressed_features(args))
        else:
            default_tag = "norm-zscore_nc3_landmarks2000"
            args.features = str(
                Path(args.output_dir) / args.family / "_channel_timeseries" / default_tag /
                "channel_timeseries_features.npy"
            )
    features_path = Path(args.features)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Channel features not found: {features_path}. Run "
            "channel_timeseries_clustering.py for this family first."
        )
    channel_ids_path = features_path.parent / "channel_ids.json"
    channel_ids = json.loads(channel_ids_path.read_text())
    features = np.load(features_path)
    if features.ndim != 2 or features.shape[0] != len(channel_ids):
        raise ValueError("Channel features rows do not match cached channel_ids")
    root = channel_model_selection_dir(
        args.output_dir, args.family, regress_global=args.regress_global,
    )
    root.mkdir(parents=True, exist_ok=True)

    reduction_table_path = root / "reduction_sweep.csv"
    if args.skip_reduction_sweep:
        reduction_table = pd.read_csv(reduction_table_path)
    else:
        reduction_table = run_reducer_sweep(features, root, args)
    selected = select_reducers(reduction_table, root, args)

    winners = None
    if not args.skip_clustering_sweep:
        winners = run_channel_clustering_sweeps(selected, features, channel_ids, root, args)
    manifest = {
        "analysis": "channel_timeseries_hyperparameter_model_selection",
        "family": args.family,
        "features": str(features_path.resolve()),
        "features_shape": list(features.shape),
        "reduction_objective": {
            "trustworthiness": 0.4, "continuity": 0.4,
            "distance_rank_agreement": 0.2,
        },
        "latent_dimension_objectives": {
            "pca": "explained-variance elbow",
            "mds": "normalized-stress elbow",
            "isomap": "geodesic-reconstruction-error elbow",
            "tsne": "KL-divergence elbow",
            "fastica": "normalized-reconstruction-MSE elbow",
            "umap": "geometry-quality elbow",
        },
        "clustering_objective": {
            "silhouette": 0.35, "inverse_davies_bouldin": 0.20,
            "calinski_harabasz": 0.15, "cluster_balance": 0.15,
            "assigned_fraction": 0.15,
            "valid_cluster_count_range": [2, args.max_selected_clusters],
        },
        "arguments": {k: v for k, v in vars(args).items()},
        "n_reduction_candidates": int(len(reduction_table)),
        "n_selected_reducers": int(len(selected)),
        "n_selected_clusterings": None if winners is None else int(len(winners)),
    }
    (root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    log.info("Channel model-selection analysis complete: %s", root)
    return root


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


def demo() -> None:
    """Self-check for --regress-global: a dominant shared component should collapse
    after regress_out_global, both in the across-channel mean and its variance."""
    rng = np.random.default_rng(0)
    n_channels, n_bins = 40, 200
    global_signal = rng.normal(size=n_bins)
    betas = rng.uniform(0.5, 2.0, size=n_channels)
    timeseries = (betas[:, None] * global_signal[None, :]
                 + 0.05 * rng.normal(size=(n_channels, n_bins))).astype(np.float32)

    mean_before = timeseries.mean(axis=0)
    regressor = _zscore_1d(timeseries.mean(axis=0))
    residual = regress_out_global(timeseries, regressor)
    mean_after = residual.mean(axis=0)

    assert np.abs(mean_after).mean() < 0.1 * np.abs(mean_before).mean(), (
        f"across-channel mean not suppressed: {np.abs(mean_before).mean():.4f} -> "
        f"{np.abs(mean_after).mean():.4f}"
    )
    assert mean_after.var() < 0.1 * mean_before.var(), (
        f"across-channel mean variance not reduced: {mean_before.var():.4f} -> "
        f"{mean_after.var():.4f}"
    )
    print(f"[demo] regress_global OK: mean|across-channel mean| "
         f"{np.abs(mean_before).mean():.4f} -> {np.abs(mean_after).mean():.4f}; "
         f"var {mean_before.var():.4f} -> {mean_after.var():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
