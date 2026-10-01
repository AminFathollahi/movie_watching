"""
tests/test_preprocess.py
========================
Unit tests for preprocessing helper functions in preprocess_individual.py.

All tests operate on synthetic in-memory numpy arrays — no file I/O needed.
Run with: conda run -n vicsompy_av pytest tests/test_preprocess.py -v
"""
import sys
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocess_individual import (
    apply_gsr,
    apply_psc,
    apply_sg_filter,
    SG_WINDOW,
    SG_ORDER,
)

# Data convention throughout: (n_verts, T)


# =============================================================================
# apply_gsr
# =============================================================================

def test_apply_gsr_removes_global_signal():
    """After GSR, mean across vertices (global signal) must be ≈0 per timepoint."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((100, 50)).astype(np.float32)
    apply_gsr(data)
    # axis=0 = vertices; mean per timepoint should be 0
    np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-5)


def test_apply_gsr_preserves_differences():
    """Vertex differences are unchanged by global signal subtraction."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((10, 20)).astype(np.float64)
    diff_before = data[0] - data[1]
    apply_gsr(data)
    diff_after = data[0] - data[1]
    np.testing.assert_allclose(diff_after, diff_before, atol=1e-10)


# =============================================================================
# apply_psc
# =============================================================================

def test_apply_psc_centering():
    """After PSC, mean per vertex is 0 (data is centered on baseline)."""
    rng = np.random.default_rng(5)
    # Positive baseline so mean is well-defined
    data = rng.standard_normal((50, 40)).astype(np.float64) + 1000.0
    apply_psc(data)
    np.testing.assert_allclose(data.mean(axis=1), 0.0, atol=1e-6)


def test_apply_psc_unit_amplitude():
    """A vertex with constant amplitude ±A around mean M should give ±(A/M)*100."""
    data = np.zeros((1, 4), dtype=np.float64)
    data[0] = [100.0, 110.0, 90.0, 100.0]   # mean=100, deviations=0,+10,-10,0
    apply_psc(data)
    expected = np.array([0.0, 10.0, -10.0, 0.0])
    np.testing.assert_allclose(data[0], expected, atol=1e-8)


def test_apply_psc_no_nan_zero_mean():
    """Zero-mean vertex must not produce NaN (protected by mean[mean==0]=1)."""
    data = np.zeros((3, 20), dtype=np.float32)
    data[1] = 1.0   # non-zero vertex
    apply_psc(data)
    assert not np.any(np.isnan(data))
    assert not np.any(np.isinf(data))


# =============================================================================
# apply_sg_filter
# =============================================================================

def test_apply_sg_filter_preserves_shape():
    rng = np.random.default_rng(6)
    data = rng.standard_normal((30, 600)).astype(np.float32)
    orig_shape = data.shape
    apply_sg_filter(data)
    assert data.shape == orig_shape


def test_apply_sg_filter_removes_dc():
    """SG filter is a high-pass; a constant signal should be driven to ~0."""
    data = np.ones((5, 600), dtype=np.float64) * 42.0
    apply_sg_filter(data)
    np.testing.assert_allclose(data, 0.0, atol=1e-8)


def test_sg_window_order_matches_vicsompy():
    """SG window=201, order=3 must match vicsompy config.yml."""
    assert SG_WINDOW == 201
    assert SG_ORDER == 3


# =============================================================================
# Filtered mode: delay-shifted TR index calculation
# =============================================================================

def test_delay_shifted_tr_indices_basic():
    """Verify delay-shifted TR window matches expected start/end."""
    tr = 1.0
    delay_sec = 5.0
    onset_sec, end_sec = 10.0, 20.0
    start_tr = int(round((onset_sec + delay_sec) / tr))
    end_tr   = int(round((end_sec   + delay_sec) / tr))
    assert start_tr == 15
    assert end_tr   == 25
    assert end_tr - start_tr == 10


def test_delay_shifted_tr_indices_sub_tr_precision():
    """Delay-shifted indices must round correctly with sub-TR onset precision."""
    tr = 1.0
    delay_sec = 5.0
    onset_sec, end_sec = 10.4, 20.6   # non-integer onsets
    start_tr = int(round((onset_sec + delay_sec) / tr))
    end_tr   = int(round((end_sec   + delay_sec) / tr))
    assert start_tr == 15   # round(15.4) = 15
    assert end_tr   == 26   # round(25.6) = 26
