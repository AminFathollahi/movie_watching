"""Tests for the per-channel modality-axis vs CCA-axis correlation.

Covers the entire surviving analysis: two continuous per-channel axes
(modality = audio-minus-video, CCA = anterior-minus-posterior), each in a
zero-order and a partial variant, and one block-bootstrapped Spearman +
Pearson correlation between them. No classes, no permutation testing, no
FDR, no winner-take-all group test -- see the module docstring in
cf_modeling/deprecated/channel_cca_analysis.py for what was deleted and why.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PATH = ROOT / "cf_modeling" / "deprecated" / "channel_cca_analysis.py"
SPEC = importlib.util.spec_from_file_location("channel_cca_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_no_class_or_significance_machinery_remains():
    """The whole point of this rebuild: no labelling/inference surface."""
    forbidden = {
        "CLASS_ORDER", "channel_sensitivity", "channel_index_sets",
        "adjust_fdr", "modality_selectivity", "sufficiency_region",
        "axis_reliability", "disattenuated_correlation",
        "group_association_bootstrap", "_sufficiency_null_test",
        "_circular_shift_indices", "_cliffs_delta",
    }
    present = forbidden & set(dir(MODULE))
    assert not present, f"Retired names still present: {present}"


def test_profile_correlation_is_offset_and_scale_invariant():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 3))
    shifted = 2.5 * base + 10.0
    corr = MODULE._profile_correlation(base, shifted)
    np.testing.assert_allclose(corr, 1.0, atol=1e-9)


def test_profile_correlation_broadcasts_a_shared_vector():
    rng = np.random.default_rng(1)
    vector = rng.normal(size=80)
    matrix = np.column_stack([vector + rng.normal(scale=0.01, size=80),
                              rng.normal(size=80)])
    corr = MODULE._profile_correlation(matrix, vector[:, None])
    assert corr[0] > 0.99
    assert abs(corr[1]) < 0.3


def test_elementwise_partial_corr_matches_fixed_vector_reference():
    rng = np.random.default_rng(2)
    n = 300
    zscore = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    a = zscore(rng.normal(size=n))
    p = zscore(rng.normal(size=n))
    x = zscore(0.6 * a + 0.3 * p + rng.normal(scale=0.2, size=n))
    manual = MODULE._elementwise_partial_corr(x[:, None], a[:, None], p[:, None])[0]
    reference, _ = MODULE.paired_partial_correlations(x[:, None], a, p)
    assert abs(manual - float(reference[0])) < 1e-5


def test_block_indices_stay_inside_runs():
    runs = np.array([11, 13, 9])
    rng = np.random.default_rng(3)
    indices = MODULE._block_indices(runs, 5, rng)
    start = 0
    for count in runs:
        block = indices[start:start + count]
        assert np.all((block >= start) & (block < start + count))
        start += count


def _synthetic(seed: int):
    rng = np.random.default_rng(seed)
    n = 240
    run_bins = np.array([n])
    base_a = np.sin(np.linspace(0, 6 * np.pi, n))
    base_v = np.cos(np.linspace(0, 10 * np.pi, n))
    noise = lambda: rng.normal(size=n)  # noqa: E731
    # channel 0: audio- and anterior-driven; channel 1: video- and
    # posterior-driven; channel 2: neither (pure noise on every leg).
    intact = np.column_stack([base_a, base_v, noise()])
    from_a = np.column_stack([base_a + 0.05 * noise(), noise(), noise()])
    from_v = np.column_stack([noise(), base_v + 0.05 * noise(), noise()])
    cca_a = base_a + 0.05 * noise()
    cca_p = base_v + 0.05 * noise()
    return intact, from_a, from_v, cca_a, cca_p, run_bins


def test_axis_pair_sign_conventions_both_variants():
    intact, from_a, from_v, cca_a, cca_p, run_bins = _synthetic(4)
    for variant in MODULE.VARIANTS:
        modality, cca = MODULE.axis_pair(
            variant, intact, from_a, from_v, cca_a, cca_p, run_bins)
        assert modality.shape == cca.shape == (3,)
        assert modality[0] > 0.5 and modality[1] < -0.5   # audio- vs video-driven
        assert cca[0] > 0.5 and cca[1] < -0.5              # anterior- vs posterior-driven
        assert abs(modality[2]) < 0.5 and abs(cca[2]) < 0.5  # neither


def test_block_bootstrap_axis_correlation_shape_and_ci_contains_observed_direction():
    intact, from_a, from_v, cca_a, cca_p, run_bins = _synthetic(5)
    for variant in MODULE.VARIANTS:
        stats, modality, cca = MODULE.block_bootstrap_axis_correlation(
            variant, intact, from_a, from_v, cca_a, cca_p, run_bins,
            n_boot=100, block_length=5, seed=6)
        assert {"spearman_rho", "spearman_ci_low", "spearman_ci_high", "pearson_r",
               "pearson_ci_low", "pearson_ci_high", "n_channels", "n_boot",
               "block_length_bins"} == set(stats)
        assert modality.shape == cca.shape == (3,)
        assert -1.0 <= stats["spearman_rho"] <= 1.0
        assert -1.0 <= stats["pearson_r"] <= 1.0
        assert stats["spearman_ci_low"] <= stats["spearman_rho"] <= stats["spearman_ci_high"]
        assert stats["pearson_ci_low"] <= stats["pearson_r"] <= stats["pearson_ci_high"]
        # Modality- and CCA-preferring channels co-vary by construction (both
        # driven by the same audio/anterior vs video/posterior split), so the
        # two axes should be strongly positively correlated here.
        assert stats["spearman_rho"] > 0.8
        assert stats["pearson_r"] > 0.8


def test_model_files_uses_correct_ablation_suffixes():
    intact, from_a, from_v = MODULE._model_files(Path("/root"), "peav")
    assert intact == Path("/root/peav/bin5s_skip5s/peav_av.npy")
    assert from_a == Path("/root/peav_clsav_from_a/bin5s_skip5s/peav_clsav_from_a_av.npy")
    assert from_v == Path("/root/peav_clsav_from_v/bin5s_skip5s/peav_clsav_from_v_av.npy")


def test_figures_are_written(tmp_path):
    intact, from_a, from_v, cca_a, cca_p, run_bins = _synthetic(7)
    for variant in MODULE.VARIANTS:
        stats, modality, cca = MODULE.block_bootstrap_axis_correlation(
            variant, intact, from_a, from_v, cca_a, cca_p, run_bins,
            n_boot=50, block_length=5, seed=8)
        path = MODULE.save_scatter(modality, cca, stats, variant, tmp_path, "demo")
        assert path.exists() and path.stat().st_size > 0


def test_selfcheck_runs():
    MODULE._selfcheck()
