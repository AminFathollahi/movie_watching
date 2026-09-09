import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.channel_stability import randomized_membership_null
from encoding.shared.compression import (
    fit_compression,
    heldout_cluster_diagnostics,
    spherical_channel_labels,
)


def test_compression_is_fit_only_on_training_samples():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(30, 12))
    test = rng.normal(size=(8, 12))
    shifted_test = test + 1000

    first = fit_compression(train, test, "pca", 4)
    second = fit_compression(train, shifted_test, "pca", 4)

    np.testing.assert_allclose(first.train, second.train)
    np.testing.assert_allclose(
        first.arrays["feature_mean"], second.arrays["feature_mean"]
    )
    assert not np.allclose(first.test, second.test)


def test_cluster_compressions_have_requested_dimension():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(40, 18))
    test = rng.normal(size=(10, 18))

    for method in ("cluster_mean", "cluster_pc1", "random_projection"):
        fitted = fit_compression(train, test, method, 5, random_state=7)
        assert fitted.train.shape == (40, 5)
        assert fitted.test.shape == (10, 5)


def test_cluster_diagnostics_favor_known_channel_groups():
    rng = np.random.default_rng(4)
    sources = rng.normal(size=(60, 3))
    heldout = np.column_stack([
        sources[:, group, None] + rng.normal(scale=0.05, size=(60, 4))
        for group in range(3)
    ]).reshape(60, 12)
    labels = np.repeat(np.arange(3), 4)

    diagnostics = heldout_cluster_diagnostics(heldout, labels)

    assert diagnostics["mean_within_cluster_correlation"] > 0.9
    assert diagnostics["reconstruction_mse"] < 0.02


def test_ari_is_label_name_invariant_but_not_membership_invariant():
    labels = np.repeat(np.arange(4), 20)
    renamed = np.array([3, 1, 0, 2])[labels]
    null = randomized_membership_null(labels, renamed, 100, random_state=9)

    assert adjusted_rand_score(labels, renamed) == 1.0
    assert np.quantile(null, 0.95) < 0.2


def test_spherical_channel_labels_does_not_mutate_input():
    rng = np.random.default_rng(6)
    train = rng.normal(size=(40, 10))
    original = train.copy()

    spherical_channel_labels(train, 3, random_state=0)

    np.testing.assert_array_equal(train, original)


def test_cluster_compression_test_scale_matches_train_scale():
    # Weakly correlated, high-channel-count data reproduces the near-degenerate
    # cluster means that previously blew up on held-out samples (see
    # spherical_channel_labels mutating its input via an aliased transpose).
    rng = np.random.default_rng(7)
    train = rng.normal(size=(60, 200))
    test = rng.normal(size=(20, 200))

    for method in ("cluster_mean", "cluster_pc1"):
        fitted = fit_compression(train, test, method, 40, random_state=0)
        train_std = fitted.train.std(axis=0)
        test_std = fitted.test.std(axis=0)
        np.testing.assert_allclose(train_std, 1.0, atol=1e-4)
        assert test_std.max() < 5.0


def test_spherical_clustering_recovers_shared_profiles():
    rng = np.random.default_rng(5)
    profiles = rng.normal(size=(80, 3))
    embedding = np.column_stack([
        profiles[:, group, None] + rng.normal(scale=0.03, size=(80, 5))
        for group in range(3)
    ]).reshape(80, 15)
    expected = np.repeat(np.arange(3), 5)

    labels = spherical_channel_labels(embedding, 3, random_state=0)

    assert adjusted_rand_score(expected, labels) > 0.95
