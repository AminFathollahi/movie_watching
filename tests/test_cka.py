import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cka.shared import kernels
from rsa.shared.rsa_utils import preprocess_fmri


def _contrasts(n):
    pairs = [(i, j) for i in range(n) for j in range(i)]
    c = np.zeros((len(pairs), n))
    for row, (i, j) in enumerate(pairs):
        c[row, i], c[row, j] = 1, -1
    return c


def _ar1_covariance(rng, n, rho=0.6):
    sigma = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    scale = (1 + 0.3 * rng.random(n)) ** 0.5
    return sigma * scale[:, None] * scale[None, :]


def test_low_signal_likelihood_is_linear_cka_numerator():
    rng = np.random.default_rng(0)
    n = 7
    c = _contrasts(n)
    v = (c @ c.T) ** 2
    h = np.eye(n) - 1 / n
    grams = [(lambda u: u @ u.T)(rng.standard_normal((n, p)) + 3.0) for p in (30, 12)]
    d = [np.diag(c @ g @ c.T) for g in grams]
    lhs = d[0] @ np.linalg.pinv(v) @ d[1]
    rhs = np.sum((h @ grams[0] @ h) * (h @ grams[1] @ h))
    assert abs(lhs - rhs) / abs(rhs) < 1e-8


def test_whitened_likelihood_identity_and_inverse_square_root():
    rng = np.random.default_rng(1)
    n = 7
    sigma = _ar1_covariance(rng, n)
    sigma = (sigma + sigma.T) / 2
    c = _contrasts(n)
    h = np.eye(n) - 1 / n
    inverse = np.linalg.pinv(h @ sigma @ h)
    grams = [(lambda u: u @ u.T)(rng.standard_normal((n, p)) + 3.0) for p in (30, 12)]
    d = [np.diag(c @ g @ c.T) for g in grams]
    lhs = d[0] @ np.linalg.pinv((c @ sigma @ c.T) ** 2) @ d[1]
    assert abs(lhs - np.trace(inverse @ grams[0] @ inverse @ grams[1])) / abs(lhs) < 1e-8
    whiten, stats = kernels.whitener(sigma)
    assert np.abs(whiten @ whiten - inverse).max() < 1e-8
    assert stats["n_floored"] == 0 and abs(stats["dropped_null_eigenvalue"]) < 1e-10


def test_cv_gram_matches_sum_over_group_pairs():
    rng = np.random.default_rng(2)
    groups, windows, vertices = 5, 7, 4
    u = rng.standard_normal((groups, windows, vertices))
    naive = sum(u[m] @ u[n].T for m in range(groups) for n in range(groups) if m != n) / (groups * (groups - 1) * vertices)
    y = torch.from_numpy(u.transpose(2, 0, 1).reshape(vertices * groups, windows)[None])
    assert np.abs(kernels.cv_gram(y, groups, vertices)[0].numpy() - naive).max() < 1e-10


def test_loglik_and_difference():
    r = np.array([0.0, 0.3, -0.3, 1.0], np.float32)
    ll = kernels.loglik(r, 11)
    assert ll[0] == 0 and np.isclose(ll[1], ll[2]) and np.isclose(ll[1], -(55 / 2) * np.log(1 - 0.09), rtol=1e-6)
    assert np.isfinite(ll[3])
    assert np.allclose(kernels.loglik_diff(r[:3], r[[1, 0, 2]], 11), ll[:3] - ll[[1, 0, 2]])


def test_window_index_matches_preprocess_fmri():
    rng = np.random.default_rng(3)
    timing = pd.DataFrame({
        "video_id": ["video1", "video2", "video3", "video4"], "run_id": [1, 1, 2, 2],
        "onset_sec": [0, 23, 40, 65], "duration_sec": [20, 22, 24, 40],
    })
    run_trs = np.array([40, 100])
    fmri = rng.standard_normal((3, 140)).astype(np.float32)
    binned = preprocess_fmri(fmri, timing, run_trs, 5.0, 1.0, 5.0, skip_sec=5.0)
    videos, runs = kernels.window_index(timing, run_trs, 5.0, 5.0, 5.0)
    assert len(videos) == len(runs) == binned.shape[0]
    assert list(np.bincount(runs)) == [6, 12]
    assert set(videos) == {"video1", "video2", "video3", "video4"}


def _neighbourhoods():
    rng = np.random.default_rng(4)
    ncols = np.stack([rng.permutation(12)[:3] for _ in range(12)]).astype(np.int32)
    ncols[3, 1] = -1
    ncols[7, 0] = -1
    return rng, ncols


def _valid(row):
    return row[row >= 0]


def _cka(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_noncv_pass_matches_direct_cka():
    rng, ncols = _neighbourhoods()
    windows = 8
    fmri = rng.standard_normal((windows, 12)).astype(np.float32)
    embedding = rng.standard_normal((windows, 5))
    centered = embedding - embedding.mean(0)
    verts = np.arange(12)
    out = kernels.noncv_pass(fmri, {"m": kernels.model_gram(embedding)}, ncols, verts, "cpu")["m"]
    for v in verts:
        hood = fmri[:, _valid(ncols[v])].astype(np.float64)
        hood = hood - hood.mean(0)
        assert abs(out[v] - _cka(hood @ hood.T, centered @ centered.T)) < 1e-5
    subset = kernels.noncv_pass(fmri, {"m": kernels.model_gram(embedding)}, ncols, np.array([2, 5]), "cpu")["m"]
    assert np.allclose(subset[[2, 5]], out[[2, 5]]) and not subset[[0, 1, 3]].any()


def test_cv_pass_matches_direct_cka():
    rng, ncols = _neighbourhoods()
    groups, windows = 4, 6
    ut = rng.standard_normal((12, groups, windows)).astype(np.float32)
    embeddings = {"a": rng.standard_normal((windows, 5)), "b": rng.standard_normal((windows, 3))}
    whiten = _ar1_covariance(rng, windows)
    whiten = np.linalg.inv(np.linalg.cholesky(whiten)).T
    whiten = (whiten + whiten.T) / 2
    out = kernels.cv_pass(ut, ncols, embeddings, whiten, np.arange(12), "cpu", batch=5)
    h = np.eye(windows) - 1 / windows
    for v in range(12):
        hood = ut[_valid(ncols[v])].astype(np.float64)
        gram = sum(hood[:, m].T @ hood[:, n] for m in range(groups) for n in range(groups) if m != n)
        gram /= groups * (groups - 1) * hood.shape[0]
        for name, x in embeddings.items():
            xc = x - x.mean(0)
            assert abs(out[(name, "cv")][v] - _cka(h @ gram @ h, xc @ xc.T)) < 1e-4
            xw = whiten @ xc
            assert abs(out[(name, "cv-ar")][v] - _cka(whiten @ gram @ whiten, xw @ xw.T)) < 1e-4
