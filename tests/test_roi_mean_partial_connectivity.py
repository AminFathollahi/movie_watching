"""Tests for leave-one-run-out ROI-mean partial correlations."""

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (Path(__file__).resolve().parents[1] / "cf_modeling" /
               "roi_mean_partial_connectivity.py")
SPEC = importlib.util.spec_from_file_location("roi_mean_partial_connectivity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _residualize(x, nuisance):
    design = np.column_stack([np.ones(len(x)), nuisance])
    return x - design @ np.linalg.lstsq(design, x, rcond=None)[0]


def test_loro_partial_correlations_match_training_coefficient_definition():
    rng = np.random.default_rng(7)
    run_trs = np.array([60, 70, 50])
    n = run_trs.sum()
    first = rng.normal(size=n)
    second = 0.4 * first + rng.normal(size=n)
    targets = np.column_stack([0.7 * first + 0.2 * second + rng.normal(size=n), -0.3 * first + 0.8 * second + rng.normal(size=n)])
    first_z = MODULE._runwise_standardize(first, run_trs)[:, 0]
    second_z = MODULE._runwise_standardize(second, run_trs)[:, 0]
    targets_z = MODULE._runwise_standardize(targets, run_trs)
    got_first, got_second = MODULE.loro_partial_correlations(targets_z, first_z, second_z, run_trs)
    run = np.repeat(np.arange(3), run_trs)

    def expected(role, control):
        total = np.zeros(targets.shape[1])
        for held_out in range(3):
            train, test = run != held_out, run == held_out
            design = np.column_stack([np.ones(train.sum()), control[train]])
            test_design = np.column_stack([np.ones(test.sum()), control[test]])
            residual = lambda y: y[test] - test_design @ np.linalg.lstsq(design, y[train], rcond=None)[0]
            x = residual(role)
            total += test.sum() * np.array([np.corrcoef(residual(targets_z[:, i]), x)[0, 1] for i in range(targets.shape[1])])
        return total / n

    np.testing.assert_allclose(got_first, expected(first_z, second_z), atol=1e-6)
    np.testing.assert_allclose(got_second, expected(second_z, first_z), atol=1e-6)


def test_geodesic_exclusion_follows_mesh_edges():
    coords = np.column_stack([np.arange(8, dtype=float), np.zeros(8), np.zeros(8)])
    faces = np.array([[i, i + 1, min(i + 2, 7)] for i in range(7)])
    got = MODULE.geodesic_exclusion(np.arange(8), coords, faces, np.array([0]), 2.5)
    assert got.tolist() == [True, True, True, False, False, False, False, False]


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
