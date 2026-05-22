"""
tests/test_rsa.py
=================
Unit tests for rsa/shared/rsa_utils.py.

All tests use synthetic numpy arrays — no CIFTI I/O needed.
Run with: conda run -n vicsompy_av pytest tests/test_rsa.py -v
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rsa", "shared"))

from rsa_utils import compute_rdm, correlate_rdms, preprocess_fmri


# =============================================================================
# compute_rdm
# =============================================================================

def test_compute_rdm_shape():
    rng = np.random.default_rng(10)
    data = rng.standard_normal((8, 100))
    rdm = compute_rdm(data)
    assert rdm.shape == (8, 8)


def test_compute_rdm_diagonal_zero():
    rng = np.random.default_rng(11)
    data = rng.standard_normal((6, 50))
    rdm = compute_rdm(data)
    np.testing.assert_allclose(np.diag(rdm), 0.0, atol=1e-10)


def test_compute_rdm_symmetric():
    rng = np.random.default_rng(12)
    data = rng.standard_normal((5, 40))
    rdm = compute_rdm(data)
    np.testing.assert_allclose(rdm, rdm.T, atol=1e-12)


def test_compute_rdm_range():
    """Correlation distance ∈ [0, 2]."""
    rng = np.random.default_rng(13)
    data = rng.standard_normal((10, 30))
    rdm = compute_rdm(data)
    assert np.all(rdm >= -1e-10)
    assert np.all(rdm <= 2.0 + 1e-10)


def test_compute_rdm_identical_rows():
    """Identical rows must have distance 0 (after diagonal)."""
    data = np.tile(np.arange(50, dtype=float), (4, 1))
    rdm = compute_rdm(data)
    np.testing.assert_allclose(rdm, 0.0, atol=1e-10)


def test_compute_rdm_cosine():
    rng = np.random.default_rng(14)
    data = rng.standard_normal((5, 20))
    rdm = compute_rdm(data, method="cosine")
    assert rdm.shape == (5, 5)
    np.testing.assert_allclose(np.diag(rdm), 0.0, atol=1e-10)


# =============================================================================
# correlate_rdms
# =============================================================================

def test_correlate_rdms_identical():
    """Correlation of an RDM with itself must be 1."""
    rng = np.random.default_rng(20)
    data = rng.standard_normal((10, 50))
    rdm = compute_rdm(data)
    r, p = correlate_rdms(rdm, rdm, method="spearman")
    assert r == pytest.approx(1.0, abs=1e-8)


def test_correlate_rdms_range():
    rng = np.random.default_rng(21)
    d1 = rng.standard_normal((8, 30))
    d2 = rng.standard_normal((8, 30))
    rdm1 = compute_rdm(d1)
    rdm2 = compute_rdm(d2)
    r, p = correlate_rdms(rdm1, rdm2, method="spearman")
    assert -1.0 <= r <= 1.0
    assert 0.0 <= p <= 1.0


def test_correlate_rdms_pearson():
    rng = np.random.default_rng(22)
    rdm = compute_rdm(rng.standard_normal((6, 20)))
    r, p = correlate_rdms(rdm, rdm, method="pearson")
    assert r == pytest.approx(1.0, abs=1e-8)


def test_correlate_rdms_uses_lower_triangle():
    """Changing the upper triangle must not change the correlation."""
    rng = np.random.default_rng(23)
    data = rng.standard_normal((5, 20))
    rdm1 = compute_rdm(data)
    rdm2 = rdm1.copy()
    # Corrupt upper triangle of rdm1
    rdm1_corrupted = rdm1.copy()
    rdm1_corrupted[0, 3] = 999.0
    rdm1_corrupted[1, 4] = -999.0

    r_orig, _ = correlate_rdms(rdm1, rdm2)
    r_corrupt, _ = correlate_rdms(rdm1_corrupted, rdm2)
    # Upper triangle corruption should not affect the result
    assert r_orig == pytest.approx(r_corrupt, abs=1e-8)


# =============================================================================
# preprocess_fmri
# Signature: preprocess_fmri(fmri, timing_df, run_trs, bin_sec, tr, delay_sec=0)
# Notes:
#   - timing_df with no run/run_id column is treated as a single run.
#   - timing_df with no onset_sec column has onsets computed from cumulative duration.
#   - Output is per-run z-scored; a constant signal → z-score of 0 (not original value).
# =============================================================================

def _make_timing(durations, onset_sec=None):
    """Helper: build a single-run timing DataFrame.

    onset_sec values are GLOBAL (cumulative across runs), matching the real
    movie_timing.csv format.  For a single run, global == within-run since
    run_start_sec == 0.
    """
    if onset_sec is None:
        # For a single run: within-run == global since run_start_sec = 0
        onset_sec = [0.0] + list(np.cumsum(durations[:-1]))
    return pd.DataFrame({"onset_sec": onset_sec, "duration_sec": durations})


def test_preprocess_fmri_output_shape():
    rng = np.random.default_rng(30)
    tr = 1.0
    bin_sec = 2.0   # 2 TRs per bin
    n_verts = 20
    durations = [10.0, 14.0, 8.0]   # → 5, 7, 4 bins
    T_total = int(sum(durations))
    fmri = rng.standard_normal((n_verts, T_total)).astype(np.float32)
    run_trs = np.array([T_total])
    timing_df = _make_timing(durations)

    out = preprocess_fmri(fmri, timing_df, run_trs, bin_sec, tr)
    assert out.shape == (5 + 7 + 4, n_verts)


def test_preprocess_fmri_output_dtype():
    rng = np.random.default_rng(31)
    T = 20
    fmri = rng.standard_normal((10, T)).astype(np.float32)
    run_trs = np.array([T])
    timing_df = _make_timing([20.0])
    out = preprocess_fmri(fmri, timing_df, run_trs, bin_sec=2.0, tr=1.0)
    assert out.dtype == np.float32


def test_preprocess_fmri_z_scored():
    """Output must be z-scored per run — mean≈0, std≈1 over bins."""
    rng = np.random.default_rng(33)
    n_verts, T = 10, 40
    fmri = rng.standard_normal((n_verts, T)).astype(np.float32)
    run_trs = np.array([T])
    # Single clip of 40s = 20 bins
    timing_df = _make_timing([40.0])
    out = preprocess_fmri(fmri, timing_df, run_trs, bin_sec=2.0, tr=1.0)
    assert out.shape == (20, n_verts)
    # Each vertex should be approximately z-scored
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=0.05)


def test_preprocess_fmri_bin_shape_multi_clip():
    """Multi-clip timing: output row count equals sum of floor(dur/bin) per clip."""
    rng = np.random.default_rng(34)
    bin_sec, tr = 2.0, 1.0
    # Two clips with a rest gap: clip1 onset=0 dur=10, clip2 onset=20 dur=12
    # Single run so global == within-run (run_start_sec=0)
    durations  = [10.0, 12.0]
    onset_secs = [0.0, 20.0]
    T_total = 40
    fmri = rng.standard_normal((5, T_total)).astype(np.float32)
    run_trs = np.array([T_total])
    timing_df = pd.DataFrame({"onset_sec": onset_secs, "duration_sec": durations})

    out = preprocess_fmri(fmri, timing_df, run_trs, bin_sec, tr)
    expected_bins = int(np.floor(10.0 / 2.0)) + int(np.floor(12.0 / 2.0))  # 5 + 6 = 11
    assert out.shape == (expected_bins, 5)


def test_preprocess_fmri_multi_run():
    """Two-run timing: each run is z-scored independently."""
    rng = np.random.default_rng(35)
    n_verts = 8
    # Run 1: 20 TRs, 1 clip 20s (onset=0 global)
    # Run 2: 20 TRs, 1 clip 20s (onset=20 global — run_2 starts at TR=20)
    T = 40
    fmri = rng.standard_normal((n_verts, T)).astype(np.float32)
    run_trs = np.array([20, 20])
    timing_df = pd.DataFrame({
        "run_id":       [1, 2],
        "onset_sec":    [0.0, 20.0],   # global: run2 at t=20 = run1_end + 0 within-run
        "duration_sec": [20.0, 20.0],
    })
    out = preprocess_fmri(fmri, timing_df, run_trs, bin_sec=2.0, tr=1.0)
    assert out.shape == (20, n_verts)  # 10 bins per run
    # Verify each run's contribution is independently z-scored
    run1_out = out[:10]
    run2_out = out[10:]
    np.testing.assert_allclose(run1_out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(run2_out.mean(axis=0), 0.0, atol=1e-5)


def test_preprocess_fmri_delay():
    """Positive delay shifts the fMRI window forward, reducing available bins."""
    rng = np.random.default_rng(36)
    n_verts, T = 6, 30
    fmri = rng.standard_normal((n_verts, T)).astype(np.float32)
    run_trs = np.array([T])
    bin_sec, tr = 2.0, 1.0
    # Clip at onset=0, duration=20 → 10 bins; with delay=4: window=[4, 24], still 10 bins
    timing_df = _make_timing([20.0])
    out_no_delay   = preprocess_fmri(fmri, timing_df, run_trs, bin_sec, tr, delay_sec=0.0)
    out_with_delay = preprocess_fmri(fmri, timing_df, run_trs, bin_sec, tr, delay_sec=4.0)
    assert out_no_delay.shape   == (10, n_verts)
    assert out_with_delay.shape == (10, n_verts)
    # The two outputs must use different fMRI columns (delay shifts window)
    assert not np.allclose(out_no_delay, out_with_delay)
