"""Unit tests for CCA threshold variants and legacy naming compatibility."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "cf_modeling" / "run_cca_islands.py"
SPEC = importlib.util.spec_from_file_location("run_cca_islands", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_top_two_percent_resolves_to_98th_percentile():
    values = np.arange(1000, dtype=np.float32)
    threshold, method = MODULE.resolve_threshold(values, None, 2.0, 0.12)
    assert method == "top_percent"
    assert threshold == pytest.approx(np.percentile(values, 98))


def test_fixed_and_registry_thresholds_are_preserved():
    values = np.arange(10, dtype=np.float32)
    assert MODULE.resolve_threshold(values, 0.2, None, 0.12) == (0.2, "fixed")
    assert MODULE.resolve_threshold(values, None, None, 0.12) == (0.12, "fixed")


def test_threshold_modes_are_mutually_exclusive_at_api_level():
    with pytest.raises(ValueError, match="either"):
        MODULE.resolve_threshold(np.arange(10), 0.1, 2.0, 0.12)


def test_variant_suffix_is_applied_to_both_rois():
    names = {"anterior": "cca_a", "posterior": "cca_p"}
    assert MODULE.suffixed_roi_names(names, "peav_2pct") == {
        "anterior": "cca_a_peav_2pct",
        "posterior": "cca_p_peav_2pct",
    }
    assert MODULE.suffixed_roi_names(names, None) == names
    with pytest.raises(ValueError, match="letters"):
        MODULE.suffixed_roi_names(names, "bad-suffix")


def test_cca_anterior_posterior_suffix_names():
    names = {"anterior": "cca_a", "posterior": "cca_p"}
    assert MODULE.suffixed_roi_names(names, "peav_1pct_lboe50") == {
        "anterior": "cca_a_peav_1pct_lboe50",
        "posterior": "cca_p_peav_1pct_lboe50",
    }


def test_merge_saddle_returns_strict_premerge_level():
    # Linear surface: seed peaks at vertices 0 and 4 connect through vertex 2.
    values = np.array([5.0, 4.0, 2.0, 3.0, 6.0])
    adjacency = [[1], [0, 2], [1, 3], [2, 4], [3]]
    threshold = MODULE.merge_saddle_threshold(
        values,
        np.array([0], dtype=np.int32),
        np.array([4], dtype=np.int32),
        adjacency,
    )
    assert threshold == 2.0
    separated = MODULE.components(values > threshold, adjacency)
    assert len(separated) == 2


def test_merge_saddle_rejects_empty_seed():
    with pytest.raises(ValueError, match="at least one"):
        MODULE.merge_saddle_threshold(
            np.array([2.0, 1.0]),
            np.array([], dtype=np.int32),
            np.array([1], dtype=np.int32),
            [[1], [0]],
        )


def test_minimum_degree_core_enforces_final_mesh_support():
    # Vertices 0--3 form a tetrahedral 3-core; vertex 4 is a one-edge tail.
    adjacency = [
        [1, 2, 3, 4], [0, 2, 3], [0, 1, 3], [0, 1, 2], [0]
    ]
    result = MODULE.minimum_degree_core(np.ones(5, dtype=bool), adjacency, 3)
    assert np.array_equal(result, [True, True, True, True, False])


def test_surface_ample_uses_relative_peak_threshold():
    # Two triangular peak plateaus joined by a low bridge at vertex 3.
    adjacency = [
        [1, 2], [0, 2], [0, 1, 3], [2, 4], [3, 5, 6], [4, 6], [4, 5]
    ]
    values = np.array([10.0, 9.0, 8.0, 2.0, 7.0, 8.0, 9.0])
    regions = MODULE.surface_ample_regions(
        values,
        [np.array([0, 1, 2]), np.array([4, 5, 6])],
        adjacency,
        peak_fraction=0.5,
        min_degree=2,
    )
    assert [item["n_vertices_before_support_filter"] for item in regions] == [3, 3]
    assert [item["ample_threshold"] for item in regions] == [5.0, 4.5]
    assert np.intersect1d(regions[0]["vertices"], regions[1]["vertices"]).size == 0


def test_surface_ample_rejects_overlapping_peak_regions():
    values = np.array([9.0, 8.0, 7.0, 8.0, 9.0])
    adjacency = [[1], [0, 2], [1, 3], [2, 4], [3]]
    with pytest.raises(RuntimeError, match="overlap at 5 vertices"):
        MODULE.surface_ample_regions(
            values,
            [np.array([0, 1]), np.array([3, 4])],
            adjacency,
            peak_fraction=0.7,
            min_degree=0,
        )
