import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.roi_av_profile import (
    decomposition_metrics, partial_unique_correlations, participation_ratio, _resample_stats,
    roi_mean_r2, roi_mean_timecourse,
)
from encoding.shared.fold_evaluator import _r2_per_target


def test_purely_synergistic_case_has_positive_synergy():
    # A and V alone explain little; only the native joint embedding captures
    # the interaction, so R2_joint far exceeds the additive model.
    metrics = decomposition_metrics(
        r2_a=0.1, r2_v=0.1, r2_additive=0.15, r2_joint=0.4, unique_a=0.0, unique_v=0.0
    )

    assert metrics["synergy"] > 0.2


def test_asymmetric_uniques_pass_through():
    metrics = decomposition_metrics(
        r2_a=0.4, r2_v=0.0, r2_additive=0.4, r2_joint=0.4, unique_a=0.8, unique_v=0.0
    )

    assert np.isclose(metrics["unique_A"], 0.8)
    assert np.isclose(metrics["unique_V"], 0.0)


def test_roi_mean_r2_matches_r2_of_the_averaged_timecourses():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(200, 4))
    prediction = target + rng.normal(scale=0.1, size=(200, 4))

    expected = float(_r2_per_target(
        roi_mean_timecourse(target)[:, None], roi_mean_timecourse(prediction)[:, None]
    )[0])

    assert np.isclose(roi_mean_r2(target, prediction), expected)


def test_roi_mean_r2_differs_from_mean_of_per_vertex_r2_under_shared_signal():
    # Each vertex carries the same signal plus large independent noise, and
    # the prediction is the noiseless shared signal broadcast to every
    # vertex. Per-vertex R2 is swamped by each vertex's own noise, but
    # averaging across vertices before scoring cancels the noise and
    # recovers the shared signal almost exactly -- the two definitions must
    # disagree sharply on data like this.
    rng = np.random.default_rng(4)
    n_samples, n_vertices = 500, 200
    signal = rng.normal(size=(n_samples, 1))
    noise = rng.normal(scale=5.0, size=(n_samples, n_vertices))
    target = signal + noise
    prediction = np.repeat(signal, n_vertices, axis=1)

    per_vertex_mean_r2 = float(np.nanmean(_r2_per_target(target, prediction)))
    roi_r2 = roi_mean_r2(target, prediction)

    assert roi_r2 > 0.8
    assert per_vertex_mean_r2 < 0.3
    assert roi_r2 - per_vertex_mean_r2 > 0.5


def test_partial_unique_correlations_isolates_the_driving_band():
    # Target is exactly the audio prediction; video is independent noise, so
    # controlling for video should not change corr(target, audio) much, and
    # corr(target, video | audio) should collapse toward zero.
    rng = np.random.default_rng(0)
    audio_pred = rng.normal(size=(2000, 5))
    video_pred = rng.normal(size=(2000, 5))
    target = audio_pred + rng.normal(scale=0.01, size=(2000, 5))

    unique_a, unique_v = partial_unique_correlations(target, audio_pred, video_pred)

    assert unique_a > 0.99
    assert abs(unique_v) < 0.1


def test_partial_unique_correlations_bounded_when_bands_are_collinear():
    # Video is an exact linear function of audio: the two nuisance-removal
    # denominators (1 - r_first_second**2) go to zero. The result must stay a
    # finite, clipped correlation, not NaN/inf.
    rng = np.random.default_rng(1)
    audio_pred = rng.normal(size=(200, 3))
    video_pred = 2.0 * audio_pred + 1.0
    target = audio_pred + rng.normal(scale=0.05, size=(200, 3))

    unique_a, unique_v = partial_unique_correlations(target, audio_pred, video_pred)

    assert np.isfinite(unique_a) and -1.0 <= unique_a <= 1.0
    assert np.isfinite(unique_v) and -1.0 <= unique_v <= 1.0


def test_resample_stats_reports_defined_fraction_and_drops_nan_before_resampling():
    effects = np.array([0.5, 0.6, np.nan, 0.4, np.nan])
    stats = _resample_stats(effects, n_bootstrap=200, n_permutations=200, random_state=0)

    assert stats["n_blocks"] == 5
    assert stats["n_blocks_defined"] == 3
    assert np.isclose(stats["fraction_defined"], 0.6)
    assert np.isclose(stats["mean"], 0.5)
    assert not np.isnan(stats["sign_flip_p_greater"])


def test_resample_stats_all_nan_yields_nan_statistic_not_a_crash():
    effects = np.array([np.nan, np.nan, 0.3])
    stats = _resample_stats(effects, n_bootstrap=200, n_permutations=200, random_state=0)

    assert stats["n_blocks_defined"] == 1
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["sign_flip_p_greater"])


def test_participation_ratio_is_one_for_rank_one_signal():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(1, 20))
    x = rng.normal(size=(500, 1)) @ direction

    assert np.isclose(participation_ratio(x), 1.0, atol=1e-6)


def test_participation_ratio_approaches_dimension_for_independent_features():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 10))

    assert participation_ratio(x) > 8.0


if __name__ == "__main__":
    test_purely_synergistic_case_has_positive_synergy()
    test_asymmetric_uniques_pass_through()
    test_roi_mean_r2_matches_r2_of_the_averaged_timecourses()
    test_roi_mean_r2_differs_from_mean_of_per_vertex_r2_under_shared_signal()
    test_partial_unique_correlations_isolates_the_driving_band()
    test_partial_unique_correlations_bounded_when_bands_are_collinear()
    test_resample_stats_reports_defined_fraction_and_drops_nan_before_resampling()
    test_resample_stats_all_nan_yields_nan_statistic_not_a_crash()
    test_participation_ratio_is_one_for_rank_one_signal()
    test_participation_ratio_approaches_dimension_for_independent_features()
    print("demo OK")
