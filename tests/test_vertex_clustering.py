"""Tests for the standalone vertex dimensionality-reduction/clustering sweep."""

import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import save_cifti_map
from cluster.io_cluster import write_dlabel
from cluster.vertex_clustering import (
    CLUSTERERS,
    REDUCTIONS,
    cluster_embedding,
    cluster_profiles as vertex_cluster_profiles,
    clustering_tag,
    expand_masked_labels,
    load_stimulus_mask,
    parse_args,
    profile_correlation_summary,
    reduce_grayordinates,
    reduction_tag,
    run,
    vertex_exclude_labels,
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
    raw = rng.normal(size=(48, 6)).astype(np.float32)
    embedding, info = reduce_grayordinates(
        raw,
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


def test_load_stimulus_mask_thresholds_correctly(tmp_path):
    template = tmp_path / "template.dtseries.nii"
    _write_template(template, n_grayordinates=10)
    values = np.array([1, -1, 0, 0.5, -0.2, 2, 0, 0, 3, -5], dtype=np.float32)
    mask_path = tmp_path / "mask.dscalar.nii"
    save_cifti_map(values, str(template), str(mask_path))

    mask = load_stimulus_mask(str(mask_path), 0.0, 10)
    assert mask.dtype == np.bool_
    assert mask.sum() == 4
    np.testing.assert_array_equal(
        mask, [True, False, False, True, False, True, False, False, True, False]
    )
    with pytest.raises(ValueError, match="expected"):
        load_stimulus_mask(str(mask_path), 0.0, 11)


def test_expand_masked_labels_reserves_noise_key_above_clusters():
    # 8 grayordinates, 6 masked-in (mask False at indices 3 and 6).
    mask = np.array([True, True, True, False, True, True, False, True])
    raw_labels = np.array([0, 0, 1, -1, 1, -1])  # two HDBSCAN clusters + noise

    full_raw, label_names, noise_key = expand_masked_labels(raw_labels, mask)
    assert full_raw.shape == (8,)
    assert (full_raw[~mask] == -1).all()
    assert noise_key == 3
    assert label_names[0] == "not_stimulus_driven"
    assert label_names[noise_key] == "noise"

    template_dir = Path(__file__).resolve().parent
    dlabel_path = template_dir / "_tmp_expand_masked_labels_test.dlabel.nii"
    try:
        template = template_dir / "_tmp_expand_masked_labels_template.dtseries.nii"
        _write_template(template, n_grayordinates=8)
        remapped = write_dlabel(
            full_raw, str(template), str(dlabel_path),
            label_names=label_names, map_name="test",
        )
    finally:
        for path in (dlabel_path, template_dir / "_tmp_expand_masked_labels_template.dtseries.nii"):
            path.unlink(missing_ok=True)

    # Masked-out vertices are always 0.
    assert (remapped[~mask] == 0).all()
    # Real cluster keys start at 1 and never touch 0 or the noise key.
    in_mask_clusters = remapped[mask][raw_labels != -1]
    assert set(in_mask_clusters) == {1, 2}
    assert in_mask_clusters.min() == 1
    # HDBSCAN noise among masked-in vertices gets its own reserved key,
    # distinct from 0 (masked-out) and from every real cluster id.
    in_mask_noise = remapped[mask][raw_labels == -1]
    assert (in_mask_noise == noise_key).all()
    assert noise_key not in {0, 1, 2}


def test_small_end_to_end_run_uses_config_tagged_directories(tmp_path):
    template = tmp_path / "tiny.dtseries.nii"
    _write_template(template, n_grayordinates=32, n_timepoints=14)
    mask_values = np.zeros(32, dtype=np.float32)
    mask_values[:24] = 1.0  # 24/32 vertices stimulus-driven
    mask_path = tmp_path / "mask.dscalar.nii"
    save_cifti_map(mask_values, str(template), str(mask_path))

    args = parse_args([
        "--input-cifti", str(template),
        "--output-dir", str(tmp_path / "outputs"),
        "--mask-cifti", str(mask_path),
        "--mask-threshold", "0.5",
        "--reductions", "pca",
        "--clusterers", "kmeans",
        "--n-components", "2",
        "--n-landmarks", "16",
        "--kmeans-clusters", "4",
    ])
    output_root = run(args)
    result = (
        output_root / "sreduce-pca_snc2" /
        "sreduce-pca_snc2_scluster-kmeans_k4"
    )
    labels = np.load(result / "spatial_vertex_labels.npy")
    assert labels.shape == (32,)
    assert (labels == 0).sum() == 8
    assert set(np.unique(labels)) <= {0, 1, 2, 3, 4}
    dlabel = result / "spatial_vertex_labels.dlabel.nii"
    assert dlabel.exists()
    assert nib.load(dlabel).header.get_axis(0).name[0] == (
        "sreduce-pca_snc2_scluster-kmeans_k4"
    )
    assert (result / "spatial_report.json").exists()
    assert (output_root / "manifest.json").exists()
    # A resume run must accept and reuse the matching preprocessing/result caches.
    assert run(args) == output_root


def test_cluster_profiles_excludes_stimulus_mask_and_noise_labels():
    # Label convention: 0 = outside the stimulus mask, 1/2 = real clusters,
    # 3 = reserved HDBSCAN-noise key above the highest cluster id.
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    units_by_bins = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 9, 9, 9],
        [1, 1, 1, 1],
        [1, 1, 1, 3],
        [10, 20, 30, 40],
        [10, 20, 30, 44],
        [-1, -2, -3, -4],
    ], dtype=np.float32)

    profiles, ids, counts, n_excluded = vertex_cluster_profiles(
        units_by_bins, labels, exclude=(0, 3)
    )
    assert ids == [1, 2]
    assert counts == {1: 2, 2: 2}
    assert n_excluded == 4
    assert profiles.shape == (2, 4)

    # Without excluding 0, label 0 (39556-vertex-scale "outside mask" block in
    # production) would silently score as its own cluster.
    profiles_unexcluded, ids_unexcluded, _, _ = vertex_cluster_profiles(
        units_by_bins, labels, exclude=(3,)
    )
    assert ids_unexcluded == [0, 1, 2]
    assert profiles_unexcluded.shape == (3, 4)


def test_vertex_exclude_labels_reads_noise_key_from_sibling_report(tmp_path):
    with_noise = tmp_path / "with_noise"
    with_noise.mkdir()
    (with_noise / "spatial_report.json").write_text(json.dumps({"noise_label": 5}))
    assert vertex_exclude_labels(with_noise / "spatial_vertex_labels.npy") == (0, 5)

    no_report = tmp_path / "no_report"
    no_report.mkdir()
    assert vertex_exclude_labels(no_report / "spatial_vertex_labels.npy") == (0,)

    no_noise = tmp_path / "no_noise"
    no_noise.mkdir()
    (no_noise / "spatial_report.json").write_text(json.dumps({"noise_label": None}))
    assert vertex_exclude_labels(no_noise / "spatial_vertex_labels.npy") == (0,)


def test_profile_correlation_summary_ignores_masked_out_vertices():
    # Two real, perfectly anti-correlated clusters; an "outside mask" block with
    # an unrelated random profile must not leak into their off-diagonal r.
    n_bins = 50
    rng = np.random.default_rng(1)
    cluster_a = np.tile(np.linspace(-1, 1, n_bins), (3, 1))
    cluster_b = -cluster_a[:2]
    outside_mask = rng.normal(size=(5, n_bins))
    labels = np.array([1] * 3 + [2] * 2 + [0] * 5)
    units_by_bins = np.vstack([cluster_a, cluster_b, outside_mask]).astype(np.float32)

    profiles, ids, _, _ = vertex_cluster_profiles(units_by_bins, labels, exclude=(0,))
    assert ids == [1, 2]
    summary = profile_correlation_summary(profiles)
    assert summary["mean_abs_off_diag"] == pytest.approx(1.0, abs=1e-4)
