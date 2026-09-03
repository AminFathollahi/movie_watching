"""Tests for reducer and cluster hyperparameter selection."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.voxel_timeseries_model_selection import (
    cluster_grid,
    cluster_selection_score,
    embedding_quality,
    elbow_dimension,
    parse_args,
    reducer_grid,
    reducer_tag,
)


def test_embedding_quality_is_near_perfect_for_identical_geometry():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 3)).astype(np.float32)
    metrics = embedding_quality(features, features.copy(), np.arange(80), [5, 10])
    assert metrics["trustworthiness_mean"] == 1.0
    assert metrics["continuity_mean"] == 1.0
    assert metrics["distance_spearman"] > 0.999
    assert metrics["quality_score"] > 0.999


def test_default_reducer_grid_covers_both_dimensions_and_all_methods():
    args = parse_args([])
    grid = reducer_grid(args)
    assert len(grid) == 343
    assert {row["method"] for row in grid} == {
        "pca", "mds", "isomap", "tsne", "fastica", "umap"
    }
    assert {row["n_components"] for row in grid} == {2, 3, 4, 5, 6, 8, 10}
    mds = next(row for row in grid if row["method"] == "mds")
    assert reducer_tag("mds", mds["n_components"], mds).startswith("sreduce-mds_snc")
    umap = next(row for row in grid if row["method"] == "umap")
    assert "_mindist" in reducer_tag("umap", umap["n_components"], umap)


def test_elbow_dimension_finds_diminishing_returns_for_gain_and_loss():
    dimensions = np.array([2, 3, 4, 5, 6, 8, 10])
    gain = np.array([0.40, 0.62, 0.76, 0.82, 0.85, 0.87, 0.88])
    loss = 1.0 - gain
    assert elbow_dimension(dimensions, gain, "maximize")[0] == 4
    assert elbow_dimension(dimensions, loss, "minimize")[0] == 4


def test_default_cluster_grid_sweeps_k_and_density_parameters():
    args = parse_args([])
    grid = cluster_grid(args)
    assert len(grid) == 48
    k_values = [row["n_clusters"] for row in grid if row["method"] == "kmeans"]
    assert k_values == [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    assert sum(row["method"] == "hdbscan" for row in grid) == 20
    assert sum(row["method"] == "birch" for row in grid) == 18


def test_cluster_selection_score_rejects_degenerate_and_scores_valid_report():
    base = {
        "n_clusters": 3,
        "n_assigned": 100,
        "noise_fraction": 0.0,
        "silhouette": 0.5,
        "calinski_harabasz": 2_000.0,
        "davies_bouldin": 0.5,
        "cluster_sizes_raw_labels": {"0": 30, "1": 35, "2": 35},
    }
    assert np.isfinite(cluster_selection_score(base, max_clusters=100))
    invalid = dict(base, n_clusters=1)
    assert np.isnan(cluster_selection_score(invalid, max_clusters=100))
