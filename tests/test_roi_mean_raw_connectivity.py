"""Tests for ordinary ROI-mean cortical correlations."""

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "cf_modeling" / "roi_mean_raw_connectivity.py"
SPEC = importlib.util.spec_from_file_location("roi_mean_raw_connectivity", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_correlations_match_numpy_reference():
    rng = np.random.default_rng(4)
    source = rng.normal(size=41)
    target = rng.normal(size=(41, 7))
    source = (source - source.mean()) / source.std()
    target = (target - target.mean(axis=0)) / target.std(axis=0)
    got = MODULE.correlations(target, source)
    expected = np.array([np.corrcoef(source, target[:, i])[0, 1]
                         for i in range(target.shape[1])])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_correlations_reject_mismatched_observations():
    try:
        MODULE.correlations(np.zeros((4, 2)), np.zeros(3))
    except ValueError as exc:
        assert "same observations" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
