import numpy as np

from rsa.av_scramble_permutation_inference import (
    _fdr_maps,
    _max_stat_fwe_maps,
    _signed_sigmap,
)


def test_signed_sigmap_uses_effect_direction():
    p = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    effect = np.array([1.0, -1.0, 0.0], dtype=np.float32)

    actual = _signed_sigmap(p, effect)

    np.testing.assert_allclose(actual, [2.0, -1.0, 0.0], atol=1e-6)


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
