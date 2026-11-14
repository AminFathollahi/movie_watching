import numpy as np

from rsa.av_scramble_permutation_inference import (
    _alpha_tag,
    _fdr_maps,
    _max_stat_critical_rho,
    _max_stat_fwe_maps,
    _signed_sigmap,
    _thresholded_sigmap,
)


def test_signed_sigmap_uses_effect_direction():
    p = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    effect = np.array([1.0, -1.0, 0.0], dtype=np.float32)

    actual = _signed_sigmap(p, effect)

    np.testing.assert_allclose(actual, [2.0, -1.0, 0.0], atol=1e-6)


def test_thresholded_sigmap_and_alpha_tag_are_explicit():
    p = np.array([0.001, 0.004, 0.006, 0.02], dtype=np.float32)
    effect = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float32)

    sigmap, mask = _thresholded_sigmap(p, effect, alpha=0.005)

    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        sigmap, [3.0, np.log10(0.004), 0.0, 0.0], rtol=1e-6)
    assert _alpha_tag(0.01) == "0p01"
    assert _alpha_tag(0.005) == "0p005"


def test_fdr_maps_return_adjusted_p_sigmap_and_mask():
    p = np.array([0.001, 0.01, 0.2, 0.8], dtype=np.float32)
    effect = np.ones(4, dtype=np.float32)

    p_fdr, sigmap_fdr, mask = _fdr_maps(p, effect, alpha=0.05)

    np.testing.assert_allclose(p_fdr, [0.004, 0.02, 0.26666668, 0.8], rtol=1e-6)
    np.testing.assert_allclose(sigmap_fdr, -np.log10(p_fdr), rtol=1e-6)
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0])


def test_max_stat_fwe_uses_each_permutation_map_maximum():
    null = np.array([
        [0.10, 0.20, 0.30],
        [0.20, 0.40, 0.10],
        [0.30, 0.10, 0.20],
    ], dtype=np.float32)
    intact = np.array([0.35, 0.45, 0.25], dtype=np.float32)
    effect = np.ones(3, dtype=np.float32)

    p_fwe, sigmap, mask, null_max, critical = _max_stat_fwe_maps(
        null, intact, effect, alpha=0.30)

    np.testing.assert_allclose(null_max, [0.30, 0.40, 0.30])
    # One null-map maximum exceeds .35; none exceeds .45; all exceed .25.
    np.testing.assert_allclose(p_fwe, [0.50, 0.25, 1.00])
    np.testing.assert_allclose(sigmap, -np.log10(p_fwe), rtol=1e-6)
    np.testing.assert_array_equal(mask, [0.0, 1.0, 0.0])
    assert np.isclose(critical, 0.40)


def test_max_stat_critical_rho_respects_discrete_pseudocount_threshold():
    null_max = np.arange(1, 501, dtype=np.float32)

    # At n=500 and strict p<.01, up to four null maxima may exceed observed:
    # observed rho must therefore exceed the fifth-largest null maximum.
    assert _max_stat_critical_rho(null_max, 0.01) == 496.0
    # At strict p<.005, only one null maximum may exceed observed, so the
    # exact cutoff is the second-largest null maximum (not a 99.5% quantile).
    assert _max_stat_critical_rho(null_max, 0.005) == 499.0
