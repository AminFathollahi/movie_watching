import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.shared.fold_evaluator import (
    _r2_per_target,
    evaluate_compression_efficiency_fold,
    evaluate_outer_fold,
    paired_clip_inference,
)
from encoding.incremental_av import _bh_qvalues


def test_r2_retains_negative_values_and_marks_constant_targets():
    y_true = np.column_stack([np.arange(5), np.ones(5)])
    y_pred = np.column_stack([np.arange(5)[::-1], np.ones(5)])

    score = _r2_per_target(y_true, y_pred)

    assert score[0] < 0
    assert np.isnan(score[1])


def test_bh_qvalues_are_monotone_in_pvalue_order():
    p_values = np.array([0.04, 0.001, 0.02, 0.8])
    q_values = _bh_qvalues(p_values)

    assert np.all(np.diff(q_values[np.argsort(p_values)]) >= 0)
    assert np.all(q_values >= p_values)


def test_outer_fold_never_fits_compression_on_test_samples():
    rng = np.random.default_rng(10)
    runs = np.repeat(np.arange(4), 10)
    audio = rng.normal(size=(40, 4))
    video = rng.normal(size=(40, 4))
    joint = rng.normal(size=(40, 8))
    targets = rng.normal(size=(40, 2))
    changed_joint = joint.copy()
    changed_joint[runs == 3] += 500
    kwargs = dict(
        targets=targets,
        run_ids=runs,
        test_run=3,
        alphas=np.logspace(-2, 2, 5),
        method="pca",
        dimension=3,
        n_iter=3,
        backend="numpy",
    )

    first = evaluate_outer_fold(audio, video, joint, **kwargs)
    second = evaluate_outer_fold(audio, video, changed_joint, **kwargs)

    np.testing.assert_allclose(first.compression.train, second.compression.train)
    np.testing.assert_allclose(
        first.compression.arrays["feature_mean"],
        second.compression.arrays["feature_mean"],
    )
    assert not np.allclose(first.compression.test, second.compression.test)


def test_joint_features_improve_heldout_prediction_when_signal_is_joint():
    rng = np.random.default_rng(11)
    runs = np.repeat(np.arange(4), 20)
    audio = rng.normal(size=(80, 3))
    video = rng.normal(size=(80, 3))
    joint = rng.normal(size=(80, 5))
    weights = rng.normal(size=(5, 2))
    targets = joint @ weights + rng.normal(scale=0.05, size=(80, 2))

    result = evaluate_outer_fold(
        audio, video, joint, targets, runs, 3,
        np.logspace(-3, 2, 7), method="full", n_iter=5, backend="numpy"
    )

    assert np.nanmean(result.delta_r2) > 0.5


def test_cluster_compression_with_collinear_band_does_not_collapse_heldout_fit():
    """Regression: an aliased in-place transpose inside spherical_channel_labels
    used to silently corrupt the training features it clustered, leaving the
    held-out cluster_mean/cluster_pc1 features on a different scale than
    training (~11x here) and driving extended_r2 to -77 on this exact
    scenario. It must now track the baseline within a small tolerance."""
    rng = np.random.default_rng(13)
    runs = np.repeat(np.arange(4), 30)
    audio = rng.normal(size=(120, 4))
    video = rng.normal(size=(120, 4))
    shared = rng.normal(size=(120, 3))
    joint = np.repeat(shared, 20, axis=1) + rng.normal(scale=0.05, size=(120, 60))
    targets = 2.0 * shared[:, :2] + 0.1 * audio[:, :2] + rng.normal(scale=0.2, size=(120, 2))
    alphas = np.logspace(-2, 9, 23)

    for method in ("cluster_mean", "cluster_pc1"):
        result = evaluate_outer_fold(
            audio, video, joint, targets, runs, 3,
            alphas, method=method, dimension=16, n_iter=20, backend="numpy",
        )
        assert result.compression.test.std(axis=0).max() < 2.0
        assert np.nanmean(result.extended_r2) > np.nanmean(result.baseline_r2) - 0.05


def test_paired_clip_inference_detects_consistent_error_reduction():
    rng = np.random.default_rng(12)
    y_true = rng.normal(size=(60, 3))
    baseline = y_true + 1.0
    extended = y_true + 0.1
    clip_ids = np.repeat(np.arange(6), 10)

    result = paired_clip_inference(
        y_true, baseline, extended, clip_ids,
        n_bootstrap=500, n_permutations=500, random_state=0,
    )

    assert result["mean_mse_reduction"] > 0
    assert result["bootstrap_ci_low"] > 0
    assert result["sign_flip_p_greater"] < 0.05


def test_matched_budget_efficiency_uses_requested_total_dimension():
    rng = np.random.default_rng(13)
    runs = np.repeat(np.arange(4), 20)
    audio = rng.normal(size=(80, 8))
    video = rng.normal(size=(80, 8))
    latent = rng.normal(size=(80, 4))
    joint = np.column_stack([latent, latent + rng.normal(scale=0.02, size=(80, 4))])
    targets = latent @ rng.normal(size=(4, 2))

    result = evaluate_compression_efficiency_fold(
        audio, video, joint, targets, runs, 3, np.logspace(-3, 2, 7),
        method="pca", total_dimension=4, n_iter=5, backend="numpy",
    )

    assert result.compressions["audio"].dimension == 2
    assert result.compressions["video"].dimension == 2
    assert result.compressions["joint"].dimension == 4
    assert np.nanmean(result.joint_minus_additive_r2) > 0.5
