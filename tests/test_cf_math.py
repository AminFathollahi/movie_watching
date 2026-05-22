"""
tests/test_cf_math.py
=====================
Unit tests for cf_modeling/shared/ridge_utils.py math.

All tests use synthetic data — no real fMRI or external drive needed.
Run with: conda run -n vicsompy_av pytest tests/test_cf_math.py -v
"""
import sys
import os
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cf_modeling", "shared"))

from ridge_utils import (
    project_onto_lboes,
    fit_null_r2,
    generate_leave_one_run_out,
    build_pipeline,
)


def _make_fake_subsurface(rng, n_verts_L, n_verts_R, n_lboe,
                           offset_L=0, offset_R=59292):
    """Minimal SimpleNamespace mimicking a Subsurface59k for projection tests."""
    sub = SimpleNamespace()
    sub.subsurface_verts_L = rng.choice(59292, size=n_verts_L, replace=False) + offset_L
    sub.subsurface_verts_R = rng.choice(59292, size=n_verts_R, replace=False) + offset_R
    sub.L_eigenvectors = rng.standard_normal((n_verts_L, n_lboe))
    sub.R_eigenvectors = rng.standard_normal((n_verts_R, n_lboe))
    sub.n_lboe = n_lboe
    return sub


# =============================================================================
# project_onto_lboes
# =============================================================================

def test_project_onto_lboes_shape():
    rng = np.random.default_rng(42)
    T = 50
    data_118k = rng.standard_normal((118584, T)).astype(np.float32)
    sub_a = _make_fake_subsurface(rng, 30, 28, 10)
    sub_b = _make_fake_subsurface(rng, 20, 18, 8)

    dm, band_sizes = project_onto_lboes(data_118k, [sub_a, sub_b])

    assert dm.shape == (T, 2 * 10 + 2 * 8)   # (50, 36)
    assert band_sizes == [20, 16]
    assert sum(band_sizes) == dm.shape[1]


def test_project_onto_lboes_math():
    """Projection matches manual matrix multiply."""
    rng = np.random.default_rng(43)
    T, n_lboe = 30, 5
    vL = np.array([0, 1, 2])
    vR = np.array([59292, 59293])
    eig_L = rng.standard_normal((3, n_lboe))
    eig_R = rng.standard_normal((2, n_lboe))

    sub = SimpleNamespace(
        subsurface_verts_L=vL,
        subsurface_verts_R=vR,
        L_eigenvectors=eig_L,
        R_eigenvectors=eig_R,
        n_lboe=n_lboe,
    )
    data = rng.standard_normal((118584, T)).astype(np.float32)

    dm, _ = project_onto_lboes(data, [sub])
    expected_L = data[vL, :].T @ eig_L        # (T, n_lboe)
    expected_R = data[vR, :].T @ eig_R        # (T, n_lboe)
    expected = np.hstack([expected_L, expected_R])  # (T, 2*n_lboe)

    np.testing.assert_allclose(dm, expected, atol=1e-4)


# =============================================================================
# fit_null_r2
# =============================================================================

def test_fit_null_r2_trivial():
    """R² must be 1 when the regressor is identical to the target column."""
    rng = np.random.default_rng(0)
    T, n = 100, 5
    Y = rng.standard_normal((T, n)).astype(np.float32)
    R2 = fit_null_r2(Y[:, 0], Y)
    assert R2.shape == (n,)
    assert R2[0] == pytest.approx(1.0, abs=1e-4)


def test_fit_null_r2_random_shape_and_range():
    rng = np.random.default_rng(7)
    T, n = 80, 20
    Y = rng.standard_normal((T, n)).astype(np.float32)
    reg = rng.standard_normal(T).astype(np.float32)
    R2 = fit_null_r2(reg, Y)
    assert R2.shape == (n,)
    assert R2.dtype == np.float32
    assert np.all(R2 >= -1.5) and np.all(R2 <= 1.0 + 1e-5)


def test_fit_null_r2_matches_vicsompy_formula():
    """Our OLS R² must match vicsompy formula: 1 - (Y-Yhat).var(0) / Y.var(0).

    vicsompy uses .var(-1) on (n_targets, T) shaped arrays; we use (T, n_targets)
    with axis=0, which is mathematically identical.
    """
    rng = np.random.default_rng(99)
    T, n = 60, 10
    Y = rng.standard_normal((T, n))
    reg = rng.standard_normal(T)

    # Our implementation
    R2_ours = fit_null_r2(reg, Y.astype(np.float32)).astype(np.float64)

    # Reproduce the OLS fit and compute vicsompy-style R²
    dm = np.column_stack([reg, np.ones(T)])
    betas, _, _, _ = np.linalg.lstsq(dm, Y, rcond=None)
    Y_hat = dm @ betas
    R2_vic = 1.0 - (Y - Y_hat).var(axis=0) / Y.var(axis=0)

    np.testing.assert_allclose(R2_ours, R2_vic, atol=1e-5)


def test_fit_null_r2_constant_target():
    """Constant Y column has ss_tot=0 → R² should be 0, not NaN."""
    Y = np.zeros((50, 3), dtype=np.float32)
    Y[:, 1] = 1.0          # constant but non-zero
    reg = np.random.default_rng(5).standard_normal(50).astype(np.float32)
    R2 = fit_null_r2(reg, Y)
    assert not np.any(np.isnan(R2))


# =============================================================================
# generate_leave_one_run_out (vicsompy version — uses random permutations)
# =============================================================================

def test_generate_loro_yields_n_runs_splits():
    n_runs, run_len = 4, 100
    n_samples = n_runs * run_len
    run_onsets = np.arange(n_runs) * run_len
    splits = list(generate_leave_one_run_out(n_samples, run_onsets,
                                              random_state=0))
    assert len(splits) == n_runs


def test_generate_loro_train_val_partition():
    n_runs, run_len = 4, 100
    n_samples = n_runs * run_len
    run_onsets = np.arange(n_runs) * run_len
    for train, val in generate_leave_one_run_out(n_samples, run_onsets,
                                                  random_state=1):
        assert len(train) + len(val) == n_samples
        assert len(np.intersect1d(train, val)) == 0


def test_generate_loro_covers_all_samples():
    """Union of all val sets = all sample indices."""
    n_runs, run_len = 4, 50
    n_samples = n_runs * run_len
    run_onsets = np.arange(n_runs) * run_len
    all_val = set()
    for _, val in generate_leave_one_run_out(n_samples, run_onsets,
                                              random_state=2):
        all_val |= set(val.tolist())
    assert all_val == set(range(n_samples))


# =============================================================================
# build_pipeline
# =============================================================================

def test_build_pipeline_structure():
    from sklearn.pipeline import Pipeline
    from himalaya.kernel_ridge import MultipleKernelRidgeCV

    n_train = 200
    run_onsets = np.array([0, 50, 100, 150])
    band_sizes = [20, 16]
    roi_names = ["roi_a", "roi_b"]

    pipeline, backend = build_pipeline(
        n_train, run_onsets, band_sizes, roi_names,
        backend_engine="numpy",
        n_iter=2, alpha_vals=3,
    )
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline[-1], MultipleKernelRidgeCV)


# =============================================================================
# Integration score formula (05_integration_maps.py math, tested standalone)
# =============================================================================

def test_integration_score_negative_clips_to_zero():
    R2_a = np.array([-0.1, 0.0, 0.5])
    R2_b = np.array([0.3, 0.0, 0.4])
    score = np.sqrt(np.clip(R2_a, 0, None) * np.clip(R2_b, 0, None))
    assert score[0] == pytest.approx(0.0)   # R2_a < 0
    assert score[1] == pytest.approx(0.0)   # both zero


def test_integration_score_positive_is_geometric_mean():
    R2_a = np.array([0.5, 0.81])
    R2_b = np.array([0.4, 0.25])
    score = np.sqrt(np.clip(R2_a, 0, None) * np.clip(R2_b, 0, None))
    np.testing.assert_allclose(score[0], np.sqrt(0.5 * 0.4), atol=1e-8)
    np.testing.assert_allclose(score[1], np.sqrt(0.81 * 0.25), atol=1e-8)
