"""Synthetic tests for the standalone voxel-timeseries clustering sweep."""

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.io_cluster import write_dlabel
from cluster.voxel_timeseries_clustering import (
    CLUSTERERS,
    REDUCTIONS,
    analysis_tag,
    cluster_embedding,
    clustering_tag,
    parse_args,
    reduce_grayordinates,
    reduction_tag,
    run,
    zscore_timeseries_inplace,
)


def _synthetic_features(seed=0, n_per_cluster=30):
    rng = np.random.default_rng(seed)
    centers = np.array([[-5, -5], [5, -5], [-5, 5], [5, 5]], dtype=np.float32)
    return np.vstack([
        center + rng.normal(scale=0.35, size=(n_per_cluster, 2))
        for center in centers
    ]).astype(np.float32)


def _write_template(path: Path, n_grayordinates=24, n_timepoints=12):
    bm_axis = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(n_grayordinates, dtype=bool), name="CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    series_axis = nib.cifti2.SeriesAxis(start=0.0, step=1.0, size=n_timepoints)
    rng = np.random.default_rng(4)
    data = rng.normal(size=(n_timepoints, n_grayordinates)).astype(np.float32)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, bm_axis))
    nib.save(nib.Cifti2Image(data, header=header), path)
    return data


def test_zscore_timeseries_handles_constant_rows_in_place():
    values = np.array([[1, 2, 3, 4], [8, 8, 8, 8]], dtype=np.float32)
    n_constant = zscore_timeseries_inplace(values)
    assert n_constant == 1
    np.testing.assert_allclose(values[0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(values[0].std(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(values[1], 0.0)


@pytest.mark.parametrize("method", REDUCTIONS)
def test_each_reduction_embeds_every_row(method):
    rng = np.random.default_rng(10)
    pre_pca = rng.normal(size=(48, 6)).astype(np.float32)
    embedding, info = reduce_grayordinates(
        pre_pca,
        method=method,
        n_components=2,
        n_landmarks=24,
        extension_neighbors=4,
        isomap_neighbors=5,
        tsne_perplexity=5,
        tsne_iterations=250,
        mds_max_iterations=20,
        random_state=3,
        n_jobs=1,
    )
    assert embedding.shape == (48, 2)
    assert np.isfinite(embedding).all()
    assert info["method"] == method


@pytest.mark.parametrize("method", CLUSTERERS)
def test_each_clusterer_labels_every_row(method):
    features = _synthetic_features()
    args = parse_args([
        "--n-components", "2",
        "--kmeans-clusters", "4",
        "--min-cluster-size", "8",
        "--min-samples", "3",
        "--birch-threshold", "0.25",
    ])
    labels, report = cluster_embedding(features, method, args)
    assert labels.shape == (len(features),)
    assert np.issubdtype(labels.dtype, np.integer)
    assert report["method"] == method
    assert report["n_clusters"] >= 1
    if method == "kmeans":
        assert report["n_clusters"] == 4
    if method == "birch":
        assert report["parameters"]["n_clusters"] is None


def test_config_tags_follow_existing_spatial_naming_convention():
    args = parse_args([
        "--n-components", "3", "--kmeans-clusters", "4",
        "--birch-threshold", "0.5",
    ])
    assert reduction_tag("pca", args) == "sreduce-pca_snc3"
    assert reduction_tag("mds", args) == "sreduce-mds_snc3_landmarks2000_extk8_iter300"
    assert reduction_tag("isomap", args) == "sreduce-isomap_snc3_landmarks2000_nn15"
    assert reduction_tag("tsne", args) == (
        "sreduce-tsne_snc3_landmarks2000_perp30_extk8_iter1000"
    )
    assert reduction_tag("umap", args) == (
        "sreduce-umap_snc3_landmarks2000_nn30_mindist0p1_metric-euclidean_extk8"
    )
    assert clustering_tag("kmeans", args) == "scluster-kmeans_k4"
    assert clustering_tag("hdbscan", args) == "scluster-hdbscan_mcs100_ms10"
    assert clustering_tag("birch", args) == "scluster-birch_threshold0p5_bf50"
    assert analysis_tag(args) == "norm-zscore_prepca50_nc3_landmarks2000"


def test_write_dlabel_round_trip_has_named_map_and_labels(tmp_path):
    template = tmp_path / "template.dtseries.nii"
    _write_template(template, n_grayordinates=6)
    output = tmp_path / "nested" / "spatial_vertex_labels.dlabel.nii"
    raw = np.array([-1, 8, 8, 3, 3, 3], dtype=np.int32)
    remapped = write_dlabel(
        raw,
        str(template),
        str(output),
        label_names={0: "unassigned", 1: "cluster_1", 2: "cluster_2"},
        map_name="sreduce-pca_snc3_scluster-hdbscan_mcs3_ms2",
    )
    np.testing.assert_array_equal(remapped, [0, 2, 2, 1, 1, 1])
    image = nib.load(output)
    axis = image.header.get_axis(0)
    assert axis.name[0] == "sreduce-pca_snc3_scluster-hdbscan_mcs3_ms2"
    assert axis.label[0][0][0] == "unassigned"
    assert axis.label[0][1][0] == "cluster_1"
    assert axis.label[0][2][0] == "cluster_2"

    with pytest.raises(ValueError, match="BrainModelAxis"):
        write_dlabel(np.zeros(5), str(template), str(tmp_path / "bad.dlabel.nii"))


def test_small_end_to_end_run_uses_config_tagged_directories(tmp_path):
    template = tmp_path / "tiny.dtseries.nii"
    _write_template(template, n_grayordinates=32, n_timepoints=14)
    args = parse_args([
        "--input-cifti", str(template),
        "--output-dir", str(tmp_path / "outputs"),
        "--reductions", "pca",
        "--clusterers", "kmeans",
        "--n-components", "2",
        "--pre-pca-components", "6",
        "--n-landmarks", "16",
        "--kmeans-clusters", "4",
    ])
    output_root = run(args)
    result = (
        output_root / "sreduce-pca_snc2" /
        "sreduce-pca_snc2_scluster-kmeans_k4"
    )
    assert (result / "spatial_vertex_labels.npy").exists()
    dlabel = result / "spatial_vertex_labels.dlabel.nii"
    assert dlabel.exists()
    assert nib.load(dlabel).header.get_axis(0).name[0] == (
        "sreduce-pca_snc2_scluster-kmeans_k4"
    )
    assert (result / "spatial_report.json").exists()
    assert (output_root / "manifest.json").exists()
    # A resume run must accept and reuse the matching preprocessing/result caches.
    assert run(args) == output_root
