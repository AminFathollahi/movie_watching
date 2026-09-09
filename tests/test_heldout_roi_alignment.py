import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.heldout_roi_alignment import (
    _cluster_profiles,
    _select_and_score,
    _zscore_rows,
    random_partition_control,
    select_and_test,
)


def test_select_and_test_uses_training_match_on_test_data():
    rng = np.random.default_rng(4)
    train_shared = rng.normal(size=120)
    test_shared = rng.normal(size=80)
    brain_train = _zscore_rows(np.stack([train_shared, rng.normal(size=120)]))
    brain_test = _zscore_rows(np.stack([test_shared, rng.normal(size=80)]))
    channel_train = _zscore_rows(np.stack([
        train_shared + 0.05 * rng.normal(size=120),
        rng.normal(size=120),
    ]))
    channel_test = _zscore_rows(np.stack([
        test_shared + 0.05 * rng.normal(size=80),
        rng.normal(size=80),
    ]))

    result = select_and_test(
        brain_train, brain_test, channel_train, channel_test,
        n_shifts=1000, random_state=2,
    )

    assert result[0]["selected_channel_cluster"] == 0
    assert result[0]["test_r"] > 0.9
    assert result[0]["shift_p_two_sided"] < 0.01


def test_row_zscore_handles_constant_profiles():
    transformed = _zscore_rows(np.array([[1.0, 1.0], [1.0, 2.0]]))
    assert np.all(transformed[0] == 0)
    assert np.isclose(transformed[1].mean(), 0)


def test_random_partition_control_null_is_weaker_than_real_clustering():
    rng = np.random.default_rng(7)
    n_train, n_test = 150, 100
    n_signal, n_noise = 6, 6
    train_shared = rng.normal(size=n_train)
    test_shared = rng.normal(size=n_test)

    signal_train = train_shared[None, :] + 0.3 * rng.normal(size=(n_signal, n_train))
    signal_test = test_shared[None, :] + 0.3 * rng.normal(size=(n_signal, n_test))
    noise_train = rng.normal(size=(n_noise, n_train))
    noise_test = rng.normal(size=(n_noise, n_test))
    embedding_train = np.concatenate([signal_train, noise_train]).T
    embedding_test = np.concatenate([signal_test, noise_test]).T
    labels = np.array([0] * n_signal + [1] * n_noise)

    brain_train = _zscore_rows(train_shared[None, :])
    brain_test = _zscore_rows(test_shared[None, :])
    channel_train = _cluster_profiles(embedding_train, labels)
    channel_test = _cluster_profiles(embedding_test, labels)
    selected, _, observed = _select_and_score(brain_train, brain_test, channel_train, channel_test)
    assert selected[0] == 0
    real_abs_r = abs(observed[0])
    assert real_abs_r > 0.5

    null = random_partition_control(
        embedding_train, embedding_test, brain_train, brain_test,
        labels, n_permutations=200, random_state=3,
    )
    assert null.shape == (200, 1)
    assert real_abs_r > np.quantile(null[:, 0], 0.95)
