import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding.shared.splits import load_inputs
from encoding.shared.fold_evaluator import ALL_SUBSETS, evaluate_split

ALPHAS = np.logspace(-2, 3, 6)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    runs = np.repeat(np.arange(4), 15)
    audio, video = rng.normal(size=(60, 6)), rng.normal(size=(60, 6))
    joint = np.hstack([audio[:, :3] + video[:, :3], rng.normal(size=(60, 3))])
    targets = joint[:, :2] + rng.normal(scale=0.3, size=(60, 2))
    return runs, audio, video, joint, targets


def _fit(runs, audio, video, joint, targets):
    return evaluate_split(
        audio, video, joint, targets, runs, runs != 3, runs == 3, ALPHAS,
        subsets=ALL_SUBSETS, n_iter=4, backend="numpy",
    )


def _same_fit(first, second):
    for key, value in first.arrays.items():
        np.testing.assert_array_equal(value, second.arrays[key], err_msg=key)


def test_perturbing_test_features_leaves_every_fitted_quantity_unchanged():
    runs, audio, video, joint, targets = _data()
    first = _fit(runs, audio, video, joint, targets)
    noisy = [band.copy() for band in (audio, video, joint)]
    for band in noisy:
        band[runs == 3] += np.random.default_rng(1).normal(scale=5, size=band[runs == 3].shape)
    second = _fit(runs, *noisy, targets)
    _same_fit(first, second)
    assert not np.allclose(first.predictions["avj"], second.predictions["avj"])


def test_perturbing_test_responses_leaves_predictions_and_fit_unchanged():
    runs, audio, video, joint, targets = _data()
    first = _fit(runs, audio, video, joint, targets)
    changed = targets.copy()
    changed[runs == 3] += np.random.default_rng(2).normal(scale=5, size=changed[runs == 3].shape)
    second = _fit(runs, audio, video, joint, changed)
    _same_fit(first, second)
    for key in ALL_SUBSETS:
        np.testing.assert_array_equal(first.predictions[key], second.predictions[key])


def _residual_of_joint(audio, video, joint, rows):
    design = np.hstack([audio, video])[rows]
    weights = np.linalg.solve(design.T @ design + 1.0 * np.eye(design.shape[1]), design.T @ joint[rows])
    return joint - np.hstack([audio, video]) @ weights


def test_residual_fitted_before_the_split_is_detected_as_leaking():
    runs, audio, video, joint, _ = _data()
    shifted_audio = audio.copy()
    shifted_audio[runs == 3] += 5
    everything = np.ones(len(runs), dtype=bool)
    leaky_first = _residual_of_joint(audio, video, joint, everything)
    leaky_second = _residual_of_joint(shifted_audio, video, joint, everything)
    assert not np.allclose(leaky_first[runs != 3], leaky_second[runs != 3])
    train_only = runs != 3
    clean_first = _residual_of_joint(audio, video, joint, train_only)
    clean_second = _residual_of_joint(shifted_audio, video, joint, train_only)
    np.testing.assert_allclose(clean_first[train_only], clean_second[train_only])


@pytest.mark.parametrize("field", ["model", "audio_model", "video_model", "joint_model_template"])
def test_global_scramble_models_are_rejected(field):
    args = SimpleNamespace(model="m", audio_model=None, video_model=None, joint_model_template=None)
    setattr(args, field, "m_avscramble")
    with pytest.raises(ValueError, match="avscramble"):
        load_inputs(args)


def test_fixed_split_response_normalization_uses_training_clips_only():
    import pandas as pd

    from encoding.shared.encoding_utils import _bin_and_split_fmri

    timing = pd.DataFrame({
        "video_id": ["video1", "video2", "video3", "video4"],
        "onset_sec": [0, 30, 60, 90],
        "duration_sec": [20, 20, 20, 20],
        "run_id": [1, 1, 2, 2],
    })
    run_trs = np.array([60, 60])
    rng = np.random.default_rng(0)
    fmri = rng.normal(size=(3, 120)).astype(np.float32)
    perturbed = fmri.copy()
    for start in (30, 90):
        perturbed[:, start:start + 20] += 100
    args = (timing, ["video2", "video4"], 5.0, 1.0, run_trs)
    train_a, test_a, _ = _bin_and_split_fmri(fmri, *args)
    train_b, test_b, _ = _bin_and_split_fmri(perturbed, *args)
    np.testing.assert_array_equal(train_a, train_b)
    assert not np.allclose(test_a, test_b)
    np.testing.assert_allclose(test_a.mean(axis=0), 0, atol=1e-5)


def test_training_row_response_scaling_ignores_the_held_out_clip():
    from encoding.shared.splits import scale_on_training_rows

    rng = np.random.default_rng(0)
    runs = np.repeat([1, 2], 6)
    targets = (rng.normal(size=(12, 3)) + 100 * runs[:, None]).astype(np.float32)
    test = np.zeros(12, bool)
    test[:2] = True
    scaled = scale_on_training_rows(targets, runs, ~test, test)
    for run in (1, 2):
        rows = (runs == run) & ~test
        np.testing.assert_allclose(scaled[rows].mean(0), 0, atol=1e-4)
        np.testing.assert_allclose(scaled[rows].std(0), 1, atol=1e-4)
    changed = targets.copy()
    changed[test] += 50
    again = scale_on_training_rows(changed, runs, ~test, test)
    np.testing.assert_array_equal(again[~test], scaled[~test])
    expected = np.broadcast_to(50 / targets[(runs == 1) & ~test].std(0), (2, 3))
    np.testing.assert_allclose(again[test] - scaled[test], expected, rtol=1e-3)
    with pytest.raises(ValueError, match="no training rows"):
        scale_on_training_rows(targets, runs, runs == 2, runs == 1)


def test_run_scaling_uses_each_run_of_the_features_only():
    from encoding.shared.splits import scale_groups

    rng = np.random.default_rng(4)
    runs = np.repeat([1, 2, 3], 5)
    x = rng.normal(3, 2, (15, 4))
    demeaned = scale_groups(x, runs, "demean")
    zscored = scale_groups(x, runs, "zscore")
    for run in (1, 2, 3):
        part = x[runs == run]
        np.testing.assert_allclose(demeaned[runs == run], part - part.mean(0), atol=1e-5)
        np.testing.assert_allclose(zscored[runs == run], (part - part.mean(0)) / part.std(0), atol=1e-5)
    shifted = x.copy()
    shifted[runs == 3] += 7.0
    np.testing.assert_array_equal(scale_groups(shifted, runs, "demean")[runs != 3], demeaned[runs != 3])


def test_training_row_feature_scaling_matches_the_responses_and_ignores_the_held_out_clip():
    from encoding.shared.splits import scale_on_training_rows

    rng = np.random.default_rng(5)
    runs = np.repeat([1, 2], 6)
    x = rng.normal(3, 2, (12, 4))
    test = np.zeros(12, bool)
    test[:2] = True
    demeaned = scale_on_training_rows(x, runs, ~test, test, "demean")
    for run in (1, 2):
        train = x[(runs == run) & ~test]
        np.testing.assert_allclose(demeaned[runs == run], x[runs == run] - train.mean(0), atol=1e-5)
    np.testing.assert_array_equal(scale_on_training_rows(x, runs, ~test, test, "zscore"), scale_on_training_rows(x, runs, ~test, test))
    changed = x.copy()
    changed[test] += 50
    np.testing.assert_array_equal(scale_on_training_rows(changed, runs, ~test, test, "demean")[~test], demeaned[~test])


@pytest.mark.parametrize("split,response,feature", [("loco", "run", "demean"), ("loro", "clip", "zscore"), ("loro", "none", "demean")])
def test_scalings_without_a_leak_free_or_matching_counterpart_are_rejected(split, response, feature):
    from encoding.shared.splits import check_scalings

    with pytest.raises(ValueError):
        check_scalings(SimpleNamespace(split=split, response_scaling=response, feature_scaling=feature))


def test_principal_components_come_from_training_rows_only():
    from encoding.shared.splits import training_components

    rng = np.random.default_rng(6)
    x = rng.normal(size=(40, 6)) @ rng.normal(size=(6, 6))
    train = np.arange(40) < 30
    scores = training_components(x, train, 3)
    _, _, axes = np.linalg.svd(x[train] - x[train].mean(0), full_matrices=False)
    np.testing.assert_allclose(np.abs(scores), np.abs(x @ axes[:3].T), rtol=1e-4, atol=1e-4)
    changed = x.copy()
    changed[~train] += 100
    np.testing.assert_allclose(training_components(changed, train, 3)[train], scores[train], atol=1e-4)
