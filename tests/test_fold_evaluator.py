import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding.shared.fold_evaluator import (
    ALL_SUBSETS,
    _r2_per_target,
    evaluate_split,
    _pearson_per_target,
    partition_variance,
)

ALPHAS = np.logspace(-3, 2, 7)


def test_r2_retains_negative_values_and_marks_constant_targets():
    y_true = np.column_stack([np.arange(5), np.ones(5)])
    y_pred = np.column_stack([np.arange(5)[::-1], np.ones(5)])
    score = _r2_per_target(y_true, y_pred)
    assert score[0] < 0
    assert np.isnan(score[1])



def test_partition_regions_sum_to_full_model_and_match_unique_definitions():
    rng = np.random.default_rng(0)
    r2 = {key: rng.normal(size=6) for key in ALL_SUBSETS}
    regions = partition_variance(r2)
    np.testing.assert_allclose(sum(regions.values()), r2["avj"])
    np.testing.assert_allclose(regions["unique_j"], r2["avj"] - r2["av"])
    np.testing.assert_allclose(regions["unique_a"], r2["avj"] - r2["vj"])
    np.testing.assert_allclose(regions["unique_v"], r2["avj"] - r2["aj"])


def test_partition_recovers_a_purely_joint_signal():
    rng = np.random.default_rng(11)
    runs = np.repeat(np.arange(4), 20)
    audio, video, joint = (rng.normal(size=(80, k)) for k in (3, 3, 5))
    targets = joint @ rng.normal(size=(5, 2)) + rng.normal(scale=0.05, size=(80, 2))
    result = evaluate_split(
        audio, video, joint, targets, runs, runs != 3, runs == 3, ALPHAS,
        subsets=ALL_SUBSETS, n_iter=5, backend="numpy",
    )
    regions = partition_variance(result.r2)
    assert regions["unique_j"].mean() > 0.5
    assert abs(regions["unique_a"].mean()) < 0.1
    assert abs(regions["unique_v"].mean()) < 0.1



def test_fold_masks_cover_both_split_variants():
    import pandas as pd

    from encoding.shared.splits import make_folds

    run_ids = np.repeat([1, 2, 3, 4], 5)
    keep = np.tile([True, True, True, True, False], 4)
    data = {"keep": keep, "metadata": pd.DataFrame({"run_id": run_ids})}
    (name, train, test), = make_folds(data, "fixed")
    assert name == "fixed" and train.sum() == 16 and test.sum() == 4 and not (train & test).any()
    folds = make_folds(data, "loro")
    assert [label for label, *_ in folds] == ["1", "2", "3", "4"]
    for label, train, test in folds:
        assert not (train & test).any()
        assert not (train | test)[~keep].any()
        assert (run_ids[test] == int(label)).all() and (run_ids[train] != int(label)).all()
    clips = np.array([f"video{i}" for i in np.repeat(np.arange(10), 2)])
    data["metadata"]["video_id"] = clips
    folds = make_folds(data, "loco")
    assert [label for label, *_ in folds] == list(pd.unique(clips[keep]))
    for label, train, test in folds:
        assert (clips[test] == label).all() and (clips[train] != label).all()
        assert not (train | test)[~keep].any() and (train | test).sum() == keep.sum()



def test_pearson_per_target_matches_numpy_and_marks_constant_targets():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 3))
    prediction = y * 2 + rng.normal(size=(30, 3))
    y[:, 2] = 1.0
    score = _pearson_per_target(y, prediction)
    np.testing.assert_allclose(score[:2], [np.corrcoef(y[:, i], prediction[:, i])[0, 1] for i in range(2)])
    assert np.isnan(score[2])
