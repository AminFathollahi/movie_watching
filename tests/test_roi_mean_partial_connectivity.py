"""Tests for exact paired ROI-mean partial correlations."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


MODULE_PATH = (Path(__file__).resolve().parents[1] / "cf_modeling" /
               "roi_mean_partial_connectivity.py")
SPEC = importlib.util.spec_from_file_location("roi_mean_partial_connectivity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _residualize(x, nuisance):
    design = np.column_stack([np.ones(len(x)), nuisance])
    return x - design @ np.linalg.lstsq(design, x, rcond=None)[0]


def test_paired_partial_correlations_match_residual_definition():
    rng = np.random.default_rng(7)
    first = rng.normal(size=200)
    second = 0.4 * first + rng.normal(size=200)
    targets = np.column_stack([
        0.7 * first + 0.2 * second + rng.normal(size=200),
        -0.3 * first + 0.8 * second + rng.normal(size=200),
    ])
    run_trs = np.array([100, 100])
    first_z = MODULE._runwise_standardize(first, run_trs)[:, 0]
    second_z = MODULE._runwise_standardize(second, run_trs)[:, 0]
    targets_z = MODULE._runwise_standardize(targets, run_trs)
    got_first, got_second = MODULE.paired_partial_correlations(
        targets_z, first_z, second_z)
    expected_first = [pearsonr(_residualize(targets_z[:, i], second_z),
                               _residualize(first_z, second_z)).statistic
                      for i in range(targets.shape[1])]
    expected_second = [pearsonr(_residualize(targets_z[:, i], first_z),
                                _residualize(second_z, first_z)).statistic
                       for i in range(targets.shape[1])]
    np.testing.assert_allclose(got_first, expected_first, atol=1e-6)
    np.testing.assert_allclose(got_second, expected_second, atol=1e-6)


def test_runwise_standardization_removes_run_offsets():
    values = np.array([1.0, 2.0, 3.0, 101.0, 102.0, 103.0])
    standardized = MODULE._runwise_standardize(values, np.array([3, 3]))[:, 0]
    np.testing.assert_allclose(standardized[:3], standardized[3:])
    assert abs(standardized[:3].mean()) < 1e-12


def test_stream_mask_means_uses_contiguous_batches():
    data = np.arange(24, dtype=float).reshape(4, 6)
    masks = {
        "first": np.array([1, 0, 1, 0, 0, 0], dtype=bool),
        "second": np.array([0, 0, 0, 1, 1, 1], dtype=bool),
    }
    got = MODULE._stream_mask_means(data, masks, 4, 6, batch_size=2)
    np.testing.assert_allclose(got["first"], data[:, [0, 2]].mean(axis=1))
    np.testing.assert_allclose(got["second"], data[:, [3, 4, 5]].mean(axis=1))
