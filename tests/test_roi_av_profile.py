import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.roi_av_profile import decomposition_metrics, participation_ratio, _resample_stats


def test_purely_redundant_case_has_zero_synergy_and_full_redundancy():
    # Video explains nothing A doesn't already: additive = A = V. Fusion (J)
    # adds nothing beyond the additive model either.
    metrics = decomposition_metrics(r2_a=0.4, r2_v=0.4, r2_additive=0.4, r2_joint=0.4)

    assert metrics["unique_A"] == 0.0
    assert metrics["unique_V"] == 0.0
    assert np.isclose(metrics["redundancy"], 1.0)
    assert np.isclose(metrics["synergy"], 0.0)


def test_purely_synergistic_case_has_positive_synergy():
    # A and V alone explain little; only the native joint embedding captures
    # the interaction, so R2_joint far exceeds the additive model.
    metrics = decomposition_metrics(r2_a=0.1, r2_v=0.1, r2_additive=0.15, r2_joint=0.4)

    assert metrics["synergy"] > 0.2


def test_audio_dominant_asymmetry():
    metrics = decomposition_metrics(r2_a=0.4, r2_v=0.0, r2_additive=0.4, r2_joint=0.4)

    assert np.isclose(metrics["audio_dominance"], 1.0)
    assert np.isclose(metrics["unique_A"], 0.4)
    assert np.isclose(metrics["unique_V"], 0.0)


def test_zero_denominators_return_nan_not_a_crash():
    metrics = decomposition_metrics(r2_a=0.0, r2_v=0.0, r2_additive=0.0, r2_joint=0.0)

    assert np.isnan(metrics["audio_dominance"])
    assert np.isnan(metrics["redundancy"])


def test_redundancy_is_nan_when_either_band_r2_is_negative_or_near_zero():
    # A noisy ROI with a slightly negative cross-validated R2 for one band
    # must not produce a numeric (garbage) redundancy ratio.
    metrics = decomposition_metrics(r2_a=0.05, r2_v=-0.02, r2_additive=0.05, r2_joint=0.05)
    assert np.isnan(metrics["redundancy"])

    metrics = decomposition_metrics(r2_a=0.05, r2_v=1e-5, r2_additive=0.05, r2_joint=0.05)
    assert np.isnan(metrics["redundancy"])


def test_redundancy_is_defined_when_both_bands_clear_the_threshold():
    metrics = decomposition_metrics(r2_a=0.2, r2_v=0.1, r2_additive=0.25, r2_joint=0.25)
    assert not np.isnan(metrics["redundancy"])


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
    test_purely_redundant_case_has_zero_synergy_and_full_redundancy()
    test_purely_synergistic_case_has_positive_synergy()
    test_audio_dominant_asymmetry()
    test_zero_denominators_return_nan_not_a_crash()
    test_redundancy_is_nan_when_either_band_r2_is_negative_or_near_zero()
    test_redundancy_is_defined_when_both_bands_clear_the_threshold()
    test_resample_stats_reports_defined_fraction_and_drops_nan_before_resampling()
    test_resample_stats_all_nan_yields_nan_statistic_not_a_crash()
    test_participation_ratio_is_one_for_rank_one_signal()
    test_participation_ratio_approaches_dimension_for_independent_features()
    print("demo OK")
