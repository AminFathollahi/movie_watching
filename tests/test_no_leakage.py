import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding.shared.splits import load_inputs
from encoding.shared.fold_evaluator import ALL_SUBSETS, FEATURE_SCALINGS, evaluate_split, standardize_bands

ALPHAS = np.logspace(-2, 3, 6)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    runs = np.repeat(np.arange(4), 15)
    audio, video = rng.normal(size=(60, 6)), rng.normal(size=(60, 6))
    joint = np.hstack([audio[:, :3] + video[:, :3], rng.normal(size=(60, 3))])
    targets = joint[:, :2] + rng.normal(scale=0.3, size=(60, 2))
    return runs, audio, video, joint, targets


def _fit(runs, audio, video, joint, targets, feature_scaling="zscore"):
    return evaluate_split(
        audio, video, joint, targets, runs, runs != 3, runs == 3, ALPHAS,
        subsets=ALL_SUBSETS, n_iter=4, backend="numpy", feature_scaling=feature_scaling,
    )


def _same_fit(first, second):
    for key, value in first.arrays.items():
        np.testing.assert_array_equal(value, second.arrays[key], err_msg=key)


@pytest.mark.parametrize("scaling", FEATURE_SCALINGS)
def test_perturbing_test_features_leaves_every_fitted_quantity_unchanged(scaling):
    runs, audio, video, joint, targets = _data()
    first = _fit(runs, audio, video, joint, targets, scaling)
    noisy = [band.copy() for band in (audio, video, joint)]
    for band in noisy:
        band[runs == 3] += np.random.default_rng(1).normal(scale=5, size=band[runs == 3].shape)
    second = _fit(runs, *noisy, targets, scaling)
    _same_fit(first, second)
    assert not np.allclose(first.predictions["avj"], second.predictions["avj"])


@pytest.mark.parametrize("scaling", FEATURE_SCALINGS)
def test_perturbing_test_responses_leaves_predictions_and_fit_unchanged(scaling):
    runs, audio, video, joint, targets = _data()
    first = _fit(runs, audio, video, joint, targets, scaling)
    changed = targets.copy()
    changed[runs == 3] += np.random.default_rng(2).normal(scale=5, size=changed[runs == 3].shape)
    second = _fit(runs, audio, video, joint, changed, scaling)
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


def test_center_scaling_subtracts_the_training_mean_without_dividing():
    rng = np.random.default_rng(0)
    train, test = rng.normal(loc=3.0, scale=2.0, size=(30, 4)), rng.normal(loc=8.0, size=(10, 4))
    (train_c,), (test_c,), _ = standardize_bands([train], [test], scale=False)
    np.testing.assert_allclose(train_c.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(train_c.std(axis=0), train.std(axis=0), rtol=1e-4)
    np.testing.assert_allclose(test_c, test - train.mean(axis=0), atol=1e-4)


def test_features_are_standardized_with_training_statistics_only():
    rng = np.random.default_rng(0)
    train, test = rng.normal(loc=3.0, scale=2.0, size=(30, 4)), rng.normal(loc=8.0, size=(10, 4))
    (train_z,), (test_z,), _ = standardize_bands([train], [test])
    np.testing.assert_allclose(train_z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(train_z.std(axis=0), 1, atol=1e-5)
    np.testing.assert_allclose(test_z, (test - train.mean(axis=0)) / train.std(axis=0), atol=1e-4)
    perturbed = test + rng.normal(scale=9, size=test.shape)
    (train_p,), _, _ = standardize_bands([train], [perturbed])
    np.testing.assert_array_equal(train_z, train_p)


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
