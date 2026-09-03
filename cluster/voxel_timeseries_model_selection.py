"""Tune voxel-timeseries embeddings, then tune clustering on the winners.

Stage 1 evaluates PCA, landmark MDS, landmark Isomap, landmark t-SNE, FastICA,
and landmark UMAP. Reducers are ranked independently
within each (method, dimensionality) pair using geometry-preservation metrics
computed on one fixed evaluation sample:

    40% trustworthiness + 40% continuity + 20% distance-rank agreement.

For each method the same sweep also selects a method-specific latent dimension
using explained variance, normalized stress, reconstruction error, KL
divergence, or geometry preservation as appropriate. Stage 2 sweeps k for
k-means and density/threshold parameters for HDBSCAN and BIRCH on every unique
selected 2-D, 3-D, and latent-best embedding. It selects one configuration per clustering
algorithm from silhouette, Davies-Bouldin, Calinski-Harabasz, cluster balance,
and assigned-data coverage, then fits that configuration to all grayordinates
and writes a convention-labelled CIFTI dlabel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import Birch, HDBSCAN, KMeans
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from io_cluster import GROUP_AVG_CIFTI, write_dlabel  # noqa: E402
from voxel_timeseries_clustering import (  # noqa: E402
    OUTPUT_DIR,
    choose_landmarks,
    clustering_report,
    reduce_grayordinates,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_PRE_PCA = (
    Path(OUTPUT_DIR) / "group_average" / "_voxel_timeseries" /
    "norm-zscore_prepca50_nc3_landmarks2000" / "voxel_timeseries_prepca50.npy"
)
REDUCTION_METHODS = ("pca", "mds", "isomap", "tsne", "fastica", "umap")
CLUSTER_METHODS = ("kmeans", "hdbscan", "birch")


def _csv_values(text: str, cast) -> list:
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid values cannot be empty")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pre-pca", default=str(DEFAULT_PRE_PCA))
    parser.add_argument("--template-cifti", default=GROUP_AVG_CIFTI)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--analysis-label", default="group_average")
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
                        default=[50, 100, 250, 500, 1_000])
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if any(value not in (2, 3) for value in args.components):
        parser.error("--components may contain only 2 and/or 3")
    for name, values in vars(args).items():
        if name.endswith("_grid") or name in ("components", "quality_neighbors"):
            if any(value < 0 if name == "umap_min_dist_grid" else value <= 0 for value in values):
                parser.error(f"All values in --{name.replace('_', '-')} must be positive")
    return args


def _float_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def reducer_tag(method: str, n_components: int, params: dict[str, Any]) -> str:
    base = f"sreduce-{method}_snc{n_components}"
    if method == "mds":
        return (f"{base}_landmarks{params['n_landmarks']}_extk{params['extension_neighbors']}"
                f"_iter{params['mds_max_iterations']}")
    if method == "isomap":
        return (f"{base}_landmarks{params['n_landmarks']}"
                f"_nn{params['isomap_neighbors']}")
    if method == "tsne":
        return (f"{base}_landmarks{params['n_landmarks']}"
                f"_perp{_float_tag(params['tsne_perplexity'])}"
                f"_extk{params['extension_neighbors']}_iter{params['tsne_iterations']}")
    if method == "fastica":
        return f"{base}_alg-{params['fastica_algorithm']}_fun-{params['fastica_fun']}"
    if method == "umap":
        return (
            f"{base}_landmarks{params['n_landmarks']}"
            f"_nn{params['umap_neighbors']}"
            f"_mindist{_float_tag(params['umap_min_dist'])}"
            f"_metric-{params['umap_metric']}"
            f"_extk{params['extension_neighbors']}"
        )
    return base


def reducer_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    dimensions = sorted(set(args.components) | set(args.latent_components_grid))
    for n_components in dimensions:
        if "pca" in args.reductions:
            candidates.append({"method": "pca", "n_components": n_components})
        if "mds" in args.reductions:
            for landmarks, extension in itertools.product(
                args.landmarks_grid, args.extension_neighbors_grid
            ):
                candidates.append({
                    "method": "mds", "n_components": n_components,
                    "n_landmarks": landmarks, "extension_neighbors": extension,
                    "mds_max_iterations": 300,
                })
        if "isomap" in args.reductions:
            for landmarks, neighbors in itertools.product(
                args.landmarks_grid, args.isomap_neighbors_grid
            ):
                candidates.append({
                    "method": "isomap", "n_components": n_components,
                    "n_landmarks": landmarks, "isomap_neighbors": neighbors,
                })
        if "tsne" in args.reductions:
            for landmarks, perplexity, extension in itertools.product(
                args.landmarks_grid, args.tsne_perplexity_grid,
                args.extension_neighbors_grid,
            ):
                candidates.append({
                    "method": "tsne", "n_components": n_components,
                    "n_landmarks": landmarks, "tsne_perplexity": perplexity,
                    "extension_neighbors": extension, "tsne_iterations": 1_000,
                })
        if "fastica" in args.reductions:
            for algorithm, fun in itertools.product(
                args.fastica_algorithms, args.fastica_functions
            ):
                candidates.append({
                    "method": "fastica", "n_components": n_components,
                    "fastica_algorithm": algorithm, "fastica_fun": fun,
                    "fastica_max_iterations": 1_000,
                })
        if "umap" in args.reductions:
            for landmarks, neighbors, min_dist, metric in itertools.product(
                args.landmarks_grid, args.umap_neighbors_grid,
                args.umap_min_dist_grid, args.umap_metrics,
            ):
                candidates.append({
                    "method": "umap", "n_components": n_components,
                    "n_landmarks": landmarks, "umap_neighbors": neighbors,
                    "umap_min_dist": min_dist, "umap_metric": metric,
                    "extension_neighbors": args.extension_neighbors_grid[0],
                })
    return candidates


def embedding_quality(high_dim: np.ndarray, embedding: np.ndarray,
                      indices: np.ndarray, neighbor_counts: list[int]) -> dict[str, float]:
    """Measure local and global geometric fidelity on one fixed sample."""
    source = np.asarray(high_dim[indices], dtype=np.float32)
    reduced = np.asarray(embedding[indices], dtype=np.float32)
    valid_neighbors = [k for k in neighbor_counts if k < len(indices) / 2]
    if not valid_neighbors:
        raise ValueError("Quality sample is too small for requested neighborhood sizes")
    trusts = [float(trustworthiness(source, reduced, n_neighbors=k))
              for k in valid_neighbors]
    # Swapping source/embedding is the standard rank-based continuity analogue.
    continuities = [float(trustworthiness(reduced, source, n_neighbors=k))
                    for k in valid_neighbors]
    high_dist = pdist(source, metric="euclidean")
    low_dist = pdist(reduced, metric="euclidean")
    rho = float(spearmanr(high_dist, low_dist).statistic)
    rho_unit = (rho + 1.0) / 2.0
    score = 0.4 * np.mean(trusts) + 0.4 * np.mean(continuities) + 0.2 * rho_unit
    result = {
        "trustworthiness_mean": float(np.mean(trusts)),
        "continuity_mean": float(np.mean(continuities)),
        "distance_spearman": rho,
        "quality_score": float(score),
    }
    for k, value in zip(valid_neighbors, trusts):
        result[f"trustworthiness_k{k}"] = value
    for k, value in zip(valid_neighbors, continuities):
        result[f"continuity_k{k}"] = value
    return result


def dimension_criterion(
    high_dim: np.ndarray,
    embedding: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
    native_report: dict[str, Any],
    quality: dict[str, float],
    random_state: int,
) -> tuple[str, float, str]:
    """Return each reducer's method-appropriate latent-dimension objective."""
    method = config["method"]
    if method == "pca":
        variances = np.var(high_dim, axis=0, ddof=1, dtype=np.float64)
        value = float(variances[: config["n_components"]].sum() / variances.sum())
        return "explained_variance_fraction_of_prepca", value, "maximize"
    if method == "mds":
        value = native_report.get("normalized_stress")
        if value is None:
            landmark_indices = choose_landmarks(
                len(high_dim), int(config["n_landmarks"]), random_state
            )
            denominator = np.sum(pdist(high_dim[landmark_indices]) ** 2)
            value = np.sqrt(float(native_report["stress"]) / denominator)
        return "normalized_stress", float(value), "minimize"
    if method == "isomap":
        return "geodesic_reconstruction_error", float(
            native_report["reconstruction_error"]
        ), "minimize"
    if method == "tsne":
        return "kl_divergence", float(native_report["kl_divergence"]), "minimize"
    if method == "fastica":
        value = native_report.get("normalized_reconstruction_mse")
        if value is None:
            source = np.asarray(high_dim[indices], dtype=np.float64)
            reduced = np.asarray(embedding[indices], dtype=np.float64)
            design = np.column_stack([reduced, np.ones(len(reduced))])
            coefficients = np.linalg.lstsq(design, source, rcond=None)[0]
            reconstructed = design @ coefficients
            value = np.mean((source - reconstructed) ** 2) / np.var(source)
        return "normalized_reconstruction_mse", float(value), "minimize"
    if method == "umap":
        return "geometry_quality_score", float(quality["quality_score"]), "maximize"
    raise ValueError(f"Unknown reduction method: {method}")


def elbow_dimension(dimensions: np.ndarray, values: np.ndarray,
                    direction: str) -> tuple[int, np.ndarray]:
    """Select the diminishing-returns elbow against the endpoint chord."""
    order = np.argsort(dimensions)
    dims = np.asarray(dimensions, dtype=float)[order]
    raw = np.asarray(values, dtype=float)[order]
    if len(dims) == 1:
        return int(dims[0]), np.zeros(1, dtype=float)
    if direction == "maximize":
        monotonic = np.maximum.accumulate(raw)
        span = monotonic[-1] - monotonic[0]
        progress = ((monotonic - monotonic[0]) / span
                    if span > 0 else np.ones_like(monotonic))
    elif direction == "minimize":
        monotonic = np.minimum.accumulate(raw)
        span = monotonic[0] - monotonic[-1]
        progress = ((monotonic[0] - monotonic) / span
                    if span > 0 else np.ones_like(monotonic))
    else:
        raise ValueError(f"Unknown direction: {direction}")
    x_span = dims[-1] - dims[0]
    x = (dims - dims[0]) / x_span if x_span > 0 else np.zeros_like(dims)
    distances = progress - x
    best = int(np.argmax(distances))
    return int(dims[best]), distances


def _reduce(pre_pca: np.ndarray, config: dict[str, Any], args: argparse.Namespace):
    defaults = {
        "n_landmarks": args.landmarks_grid[0],
        "extension_neighbors": args.extension_neighbors_grid[0],
        "isomap_neighbors": args.isomap_neighbors_grid[0],
        "tsne_perplexity": args.tsne_perplexity_grid[0],
        "tsne_iterations": 1_000,
        "mds_max_iterations": 300,
        "fastica_algorithm": "parallel",
        "fastica_fun": "logcosh",
        "fastica_max_iterations": 1_000,
        "umap_neighbors": args.umap_neighbors_grid[0],
        "umap_min_dist": args.umap_min_dist_grid[0],
        "umap_metric": args.umap_metrics[0],
    }
    defaults.update(config)
    return reduce_grayordinates(
        pre_pca, method=config["method"], n_components=config["n_components"],
        n_landmarks=defaults["n_landmarks"],
        extension_neighbors=defaults["extension_neighbors"],
        isomap_neighbors=defaults["isomap_neighbors"],
        tsne_perplexity=defaults["tsne_perplexity"],
        tsne_iterations=defaults["tsne_iterations"],
        mds_max_iterations=defaults["mds_max_iterations"],
        fastica_algorithm=defaults["fastica_algorithm"],
        fastica_fun=defaults["fastica_fun"],
        fastica_max_iterations=defaults["fastica_max_iterations"],
        umap_neighbors=defaults["umap_neighbors"],
        umap_min_dist=defaults["umap_min_dist"],
        umap_metric=defaults["umap_metric"],
        random_state=args.random_state, n_jobs=args.n_jobs,
    )


def _save_table(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_reducer_sweep(pre_pca: np.ndarray, root: Path,
                      args: argparse.Namespace) -> pd.DataFrame:
    eval_path = root / "reduction_quality_indices.npy"
    if eval_path.exists() and not args.force:
        quality_indices = np.load(eval_path)
    else:
        rng = np.random.default_rng(args.random_state)
        quality_indices = np.sort(rng.choice(
            len(pre_pca), size=min(args.quality_sample_size, len(pre_pca)), replace=False
        ))
        np.save(eval_path, quality_indices)

    rows: list[dict[str, Any]] = []
    table_path = root / "reduction_sweep.csv"
    candidates = reducer_grid(args)
    log.info("Reducer sweep: %d configurations", len(candidates))
    for position, config in enumerate(candidates, start=1):
        tag = reducer_tag(config["method"], config["n_components"], config)
        candidate_dir = root / "reduction_candidates" / tag
        candidate_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = candidate_dir / "spatial_vertex_components.npy"
        report_path = candidate_dir / "reduction_report.json"
        log.info("Reducer %d/%d: %s", position, len(candidates), tag)
        row: dict[str, Any] = {"tag": tag, **config, "status": "ok"}
        try:
            if embedding_path.exists() and report_path.exists() and not args.force:
                embedding = np.load(embedding_path)
                cached_report = json.loads(report_path.read_text())
                native_report = cached_report.get("native_diagnostics", cached_report)
            else:
                embedding, native_report = _reduce(pre_pca, config, args)
                np.save(embedding_path, embedding)
            quality = embedding_quality(
                pre_pca, embedding, quality_indices, args.quality_neighbors
            )
            criterion_name, criterion_value, criterion_direction = dimension_criterion(
                pre_pca, embedding, quality_indices, config, native_report,
                quality, args.random_state,
            )
            full_report = {
                "tag": tag, "parameters": config, "native_diagnostics": native_report,
                "selection_metrics": quality,
                "dimension_selection": {
                    "criterion": criterion_name,
                    "value": criterion_value,
                    "direction": criterion_direction,
                },
                "quality_sample_size": int(len(quality_indices)),
                "quality_neighbors": args.quality_neighbors,
                "embedding": str(embedding_path),
            }
            report_path.write_text(json.dumps(full_report, indent=2, sort_keys=True) + "\n")
            row.update(
                quality,
                dimension_criterion=criterion_name,
                dimension_criterion_value=criterion_value,
                dimension_criterion_direction=criterion_direction,
                embedding=str(embedding_path), report=str(report_path),
            )
        except Exception as exc:
            log.exception("Reducer failed: %s", tag)
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        _save_table(rows, table_path)
    return pd.DataFrame(rows)


def select_reducers(table: pd.DataFrame, root: Path,
                    args: argparse.Namespace) -> pd.DataFrame:
    valid = table[(table["status"] == "ok") & np.isfinite(table["quality_score"])]
    if valid.empty:
        raise RuntimeError("No reducer configuration completed successfully")

    display = valid[valid["n_components"].isin(args.components)]
    display_indices = display.groupby(
        ["method", "n_components"]
    )["quality_score"].idxmax()
    display = valid.loc[display_indices].copy()
    display["selection_role"] = display["n_components"].map(
        lambda value: f"display_{int(value)}d"
    )

    native_rows = []
    for (_, _), group in valid.groupby(["method", "n_components"]):
        direction = str(group["dimension_criterion_direction"].iloc[0])
        column = group["dimension_criterion_value"]
        index = column.idxmax() if direction == "maximize" else column.idxmin()
        native_rows.append(valid.loc[index].copy())
    native_by_dimension = pd.DataFrame(native_rows).sort_values(
        ["method", "n_components"]
    )
    native_by_dimension.to_csv(root / "dimension_selection_sweep.csv", index=False)

    latent_rows = []
    latent_summary = []
    for method, group in native_by_dimension.groupby("method"):
        group = group.sort_values("n_components")
        direction = str(group["dimension_criterion_direction"].iloc[0])
        best_dimension, elbow_distances = elbow_dimension(
            group["n_components"].to_numpy(),
            group["dimension_criterion_value"].to_numpy(),
            direction,
        )
        winner = group[group["n_components"] == best_dimension].iloc[0].copy()
        winner["selection_role"] = "latent_best"
        winner["latent_dimension_selection"] = "diminishing_returns_elbow"
        latent_rows.append(winner)
        for (_, candidate), distance in zip(group.iterrows(), elbow_distances):
            latent_summary.append({
                "method": method,
                "n_components": int(candidate["n_components"]),
                "reducer_tag": candidate["tag"],
                "criterion": candidate["dimension_criterion"],
                "criterion_value": float(candidate["dimension_criterion_value"]),
                "direction": direction,
                "elbow_distance": float(distance),
                "selected": int(candidate["n_components"]) == best_dimension,
            })
    latent = pd.DataFrame(latent_rows)
    pd.DataFrame(latent_summary).to_csv(
        root / "selected_latent_dimensions.csv", index=False
    )

    combined = pd.concat([display, latent], ignore_index=True)
    role_by_tag = combined.groupby("tag")["selection_role"].agg(
        lambda roles: "|".join(sorted(set(roles)))
    )
    selected = combined.drop_duplicates("tag").copy()
    selected["selection_role"] = selected["tag"].map(role_by_tag)
    selected["is_latent_best"] = selected["selection_role"].str.contains("latent_best")
    selected = selected.sort_values(["n_components", "method"])
    selected.to_csv(root / "selected_reducers.csv", index=False)
    return selected


def cluster_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [{"method": "kmeans", "n_clusters": k} for k in args.kmeans_k_grid]
    rows += [
        {"method": "hdbscan", "min_cluster_size": size, "min_samples": samples}
        for size, samples in itertools.product(
            args.hdbscan_min_cluster_size_grid, args.hdbscan_min_samples_grid
        )
    ]
    rows += [
        {"method": "birch", "threshold": threshold, "branching_factor": branching}
        for threshold, branching in itertools.product(
            args.birch_threshold_grid, args.birch_branching_factor_grid
        )
    ]
    return rows


def cluster_tag(params: dict[str, Any]) -> str:
    if params["method"] == "kmeans":
        return f"scluster-kmeans_k{params['n_clusters']}"
    if params["method"] == "hdbscan":
        return (f"scluster-hdbscan_mcs{params['min_cluster_size']}"
                f"_ms{params['min_samples']}")
    return (f"scluster-birch_threshold{_float_tag(params['threshold'])}"
            f"_bf{params['branching_factor']}")


def _fit_cluster(features: np.ndarray, params: dict[str, Any], random_state: int):
    if params["method"] == "kmeans":
        model = KMeans(n_clusters=params["n_clusters"], n_init=20,
                       random_state=random_state)
    elif params["method"] == "hdbscan":
        model = HDBSCAN(min_cluster_size=params["min_cluster_size"],
                        min_samples=params["min_samples"])
    else:
        model = Birch(threshold=params["threshold"],
                      branching_factor=params["branching_factor"], n_clusters=None)
    return model.fit_predict(features).astype(np.int32, copy=False)


def cluster_selection_score(report: dict[str, Any], max_clusters: int) -> float:
    """Comparable bounded score with guards against degenerate fragmentations."""
    n_clusters = report["n_clusters"]
    if n_clusters < 2 or n_clusters > max_clusters or report["n_assigned"] < 2:
        return float("nan")
    values = [report["silhouette"], report["calinski_harabasz"],
              report["davies_bouldin"]]
    if any(value is None or not np.isfinite(value) for value in values):
        return float("nan")
    sizes = np.array(list(report["cluster_sizes_raw_labels"].values()), dtype=float)
    proportions = sizes / sizes.sum()
    balance = float(-(proportions * np.log(proportions)).sum() / np.log(n_clusters))
    assigned_fraction = 1.0 - report["noise_fraction"]
    silhouette_unit = (float(report["silhouette"]) + 1.0) / 2.0
    db_unit = 1.0 / (1.0 + float(report["davies_bouldin"]))
    ch_unit = float(report["calinski_harabasz"]) / (
        float(report["calinski_harabasz"]) + 1_000.0
    )
    return float(
        0.35 * silhouette_unit + 0.20 * db_unit + 0.15 * ch_unit
        + 0.15 * balance + 0.15 * assigned_fraction
    )


def _label_names(labels: np.ndarray) -> dict[int, str]:
    count = len(np.unique(labels[labels != -1]))
    return {0: "unassigned", **{key: f"cluster_{key}" for key in range(1, count + 1)}}


def run_clustering_sweeps(selected: pd.DataFrame, pre_pca: np.ndarray,
                          template: Path, root: Path,
                          args: argparse.Namespace) -> pd.DataFrame:
    eval_path = root / "clustering_evaluation_indices.npy"
    if eval_path.exists() and not args.force:
        eval_indices = np.load(eval_path)
    else:
        rng = np.random.default_rng(args.random_state + 1)
        eval_indices = np.sort(rng.choice(
            len(pre_pca), size=min(args.cluster_sample_size, len(pre_pca)), replace=False
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
        raise RuntimeError(
            "No valid clustering candidate for selected reducer/method pairs: "
            + ", ".join(f"{tag}/{method}" for tag, method in missing_pairs)
        )
    winners_idx = valid.groupby(["reducer_tag", "method"])["selection_score"].idxmax()
    winners = valid.loc[winners_idx].copy().sort_values(["n_components", "reduction_method", "method"])
    role_lookup = selected.set_index("tag")["selection_role"].to_dict()
    winners["selection_role"] = winners["reducer_tag"].map(role_lookup)
    scale_factor = len(pre_pca) / len(eval_indices)
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
        labels_path = output_dir / "spatial_vertex_labels.npy"
        dlabel_path = output_dir / "spatial_vertex_labels.dlabel.nii"
        report_path = output_dir / "spatial_report.json"
        if labels_path.exists() and dlabel_path.exists() and report_path.exists() and not args.force:
            selected_map_records.append({
                "reducer_tag": reducer_name, "cluster_method": method,
                "reduction_method": str(selected_reducer["method"]),
                "n_components": int(selected_reducer["n_components"]),
                "selection_role": str(selected_reducer["selection_role"]),
                "full_fit_cluster_tag": ctag, "dlabel": str(dlabel_path),
                "report": str(report_path),
            })
            continue
        log.info("Fitting selected full map: %s", config_tag)
        labels = _fit_cluster(scaled, params, args.random_state)
        report = clustering_report(scaled, labels, args.random_state)
        remapped = write_dlabel(
            labels, str(template), str(dlabel_path),
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
            dlabel_unique_keys=[int(value) for value in np.unique(remapped)],
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        selected_map_records.append({
            "reducer_tag": reducer_name, "cluster_method": method,
            "reduction_method": str(selected_reducer["method"]),
            "n_components": int(selected_reducer["n_components"]),
            "selection_role": str(selected_reducer["selection_role"]),
            "full_fit_cluster_tag": ctag, "dlabel": str(dlabel_path),
            "report": str(report_path),
        })
    (root / "selected_maps_manifest.json").write_text(
        json.dumps(selected_map_records, indent=2, sort_keys=True) + "\n"
    )
    return winners


def run(args: argparse.Namespace) -> Path:
    pre_pca_path = Path(args.pre_pca)
    template_path = Path(args.template_cifti)
    if not pre_pca_path.exists():
        raise FileNotFoundError(f"Preliminary PCA not found: {pre_pca_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template CIFTI not found: {template_path}")
    pre_pca = np.load(pre_pca_path)
    bm_axis = nib.load(str(template_path)).header.get_axis(1)
    if pre_pca.ndim != 2 or pre_pca.shape[0] != len(bm_axis):
        raise ValueError("Preliminary PCA rows do not match template BrainModelAxis")
    root = (Path(args.output_dir) / args.analysis_label /
            "_voxel_timeseries_model_selection" /
            f"norm-zscore_prepca{pre_pca.shape[1]}")
    root.mkdir(parents=True, exist_ok=True)

    reduction_table_path = root / "reduction_sweep.csv"
    if args.skip_reduction_sweep:
        reduction_table = pd.read_csv(reduction_table_path)
    else:
        reduction_table = run_reducer_sweep(pre_pca, root, args)
    selected = select_reducers(reduction_table, root, args)

    winners = None
    if not args.skip_clustering_sweep:
        winners = run_clustering_sweeps(
            selected, pre_pca, template_path, root, args
        )
    manifest = {
        "analysis": "voxel_timeseries_hyperparameter_model_selection",
        "preliminary_pca": str(pre_pca_path.resolve()),
        "template_cifti": str(template_path.resolve()),
        "pre_pca_shape": list(pre_pca.shape),
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
        "arguments": vars(args),
        "n_reduction_candidates": int(len(reduction_table)),
        "n_selected_reducers": int(len(selected)),
        "n_selected_clusterings": None if winners is None else int(len(winners)),
    }
    (root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    log.info("Model-selection analysis complete: %s", root)
    return root


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
