import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    v = (c @ sigma @ c.T) ** 2
    wuc = d[0] @ np.linalg.pinv(v) @ d[1] / np.sqrt((d[0] @ np.linalg.pinv(v) @ d[0]) * (d[1] @ np.linalg.pinv(v) @ d[1]))
    assert abs(wuc - _cka(whiten @ grams[0] @ whiten, whiten @ grams[1] @ whiten)) < 1e-8
    assert stats["n_floored"] == 0 and abs(stats["dropped_null_eigenvalue"]) < 1e-10


def test_cv_gram_matches_sum_over_group_pairs():
    rng = np.random.default_rng(2)
    groups, windows, vertices = 5, 7, 4
    u = rng.standard_normal((groups, windows, vertices))
    naive = sum(u[m] @ u[n].T for m in range(groups) for n in range(groups) if m != n) / (groups * (groups - 1) * vertices)
    y = torch.from_numpy(u.transpose(2, 0, 1).reshape(vertices * groups, windows)[None])
    assert np.abs(kernels.cv_gram(y, groups, vertices)[0].numpy() - naive).max() < 1e-10


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


def _group_matrices(hood):
    return [hood[:, m, :].T.astype(np.float64) for m in range(hood.shape[1])]


def _cv_matrix(ys):
    groups, p = len(ys), ys[0].shape[1]
    return sum(ys[m] @ ys[n].T for m in range(groups) for n in range(groups) if m != n) / (groups * (groups - 1) * p)


def _signal_problem(seed, signal, windows=9, groups=6, vertices=8, size=3):
    rng = np.random.default_rng(seed)
    xa, xv, xj = (rng.standard_normal((windows, d)) for d in (5, 4, 6))
    base = np.stack([xa @ rng.standard_normal((5, vertices)), xv @ rng.standard_normal((4, vertices)), xj @ rng.standard_normal((6, vertices))])
    truth = signal * np.einsum("c,cwv->wv", rng.uniform(0.2, 1.0, 3), base)
    ut = (truth.T[:, None, :] + rng.standard_normal((vertices, groups, windows))).astype(np.float32)
    ncols = np.stack([rng.permutation(vertices)[:size] for _ in range(vertices)]).astype(np.int32)
    whiten = np.linalg.inv(np.linalg.cholesky(_ar1_covariance(rng, windows))).T
    return ut, ncols, (xa, xv, xj), (whiten + whiten.T) / 2


def test_subject_maps_average_to_the_cross_validated_gram():
    ut, ncols, xs, whiten = _signal_problem(11, 1.0, vertices=10)
    groups = ut.shape[1]
    (plain, white), _ = kernels.component_grams(xs, whiten)
    passes = [kernels.subject_pass(ut[:, m], ut.sum(1) - ut[:, m], ncols, (plain, white), whiten, np.arange(len(ut)), "cpu", batch=4)
              for m in range(groups)]
    numerator = np.mean([cosines * norms[:, None] for cosines, norms in passes], 0) / (groups - 1)
    out = kernels.cv_pass(ut, ncols, dict(zip("avj", xs)), whiten, np.arange(len(ut)), "cpu", batch=4)
    for i, measure in enumerate(("cv", "cv-ar")):
        for v in range(len(ut)):
            hood = torch.from_numpy(ut[_valid(ncols[v])]).reshape(1, -1, ut.shape[2])
            gram = kernels.cv_gram(hood, groups, hood.shape[1] // groups)
            norm = np.linalg.norm(kernels.double_center(gram).numpy() if measure == "cv" else whiten @ gram[0].numpy().astype(np.float64) @ whiten)
            assert np.abs(numerator[i][:, v] / norm - [out[(c, measure)][v] for c in "avj"]).max() < 1e-4
    cosines, norms = passes[0]
    hood = ut[_valid(ncols[0])]
    cross = hood[:, 0].T @ (hood.sum(1) - hood[:, 0]) / len(hood)
    target = (cross + cross.T) / 2
    target = target - target.mean(0) - target.mean(1)[:, None] + target.mean()
    assert np.abs(cosines[0][:, 0] - plain.T @ target.ravel() / np.linalg.norm(target)).max() < 1e-5


def _brute_cosines(hood, whiten, xs, measure):
    target = _cv_matrix(_group_matrices(hood))
    windows = target.shape[0]
    left = np.eye(windows) - 1 / windows if measure == "cv" else whiten
    grams = []
    for x in xs:
        z = left @ (x - x.mean(0)) if measure == "cv-ar" else (np.eye(windows) - 1 / windows) @ (x - x.mean(0))
        gram = z @ z.T
        grams.append((gram / np.linalg.norm(gram)).ravel())
    target = (left @ target @ left).ravel()
    return target / np.linalg.norm(target), np.stack(grams, 1)


def _brute_semipartial(target, grams):
    out = []
    for x in range(3):
        others = [i for i in range(3) if i != x]
        beta = np.linalg.lstsq(grams[:, others], grams[:, x], rcond=None)[0]
        residual = grams[:, x] - grams[:, others] @ beta
        out.append(target @ residual / np.linalg.norm(residual))
    return np.array(out)


def test_semipartial_matches_explicit_residualization_on_the_matrices():
    ut, ncols, xs, whiten = _signal_problem(10, 1.0)
    out = kernels.cv_pass(ut, ncols, dict(zip("avj", xs)), whiten, np.arange(len(ut)), "cpu", batch=3)
    _, cosines = kernels.component_grams(xs, whiten)
    for measure, cosine in zip(("cv", "cv-ar"), cosines):
        sp = kernels.semipartial(np.stack([out[(c, measure)] for c in "avj"]), cosine)
        for v in range(len(ut)):
            target, grams = _brute_cosines(ut[_valid(ncols[v])], whiten, xs, measure)
            assert np.abs(grams.T @ grams - cosine).max() < 1e-6
            assert np.abs(sp[:, v] - _brute_semipartial(target, grams)).max() < 1e-4


def test_semipartial_is_zero_for_a_component_in_the_span_of_the_others():
    ut, ncols, (xa, xv, _), whiten = _signal_problem(11, 1.0)
    scaled = [(x - x.mean(0)) / np.sqrt(np.linalg.norm((x - x.mean(0)) @ (x - x.mean(0)).T)) for x in (xa, xv)]
    xj = np.concatenate(scaled, 1)
    out = kernels.cv_pass(ut, ncols, {"a": xa, "v": xv, "j": xj}, whiten, np.arange(len(ut)), "cpu", batch=3)
    for measure, cosine in zip(("cv", "cv-ar"), kernels.component_grams((xa, xv, xj), whiten)[1]):
        sp = kernels.semipartial(np.stack([out[(c, measure)] for c in "avj"]), cosine)
        assert measure == "cv-ar" or np.all(sp[2] == 0)
        assert measure == "cv-ar" or kernels.semipartial_coefficients(cosine)[1][2] < 1e-6


def test_noncv_semipartial_matches_explicit_residualization_on_the_matrices():
    ut, ncols, xs, _ = _signal_problem(12, 1.0)
    fmri = ut.mean(1).T
    grams = {c: kernels.model_gram(x) for c, x in zip("avj", xs)}
    out = kernels.noncv_pass(fmri, grams, ncols, np.arange(len(ut)), "cpu")
    columns = np.stack(list(grams.values()), 1).astype(np.float64)
    assert np.abs(columns.T @ columns - kernels.component_grams(xs, np.eye(len(fmri)))[1][0]).max() < 1e-6
    sp = kernels.semipartial(np.stack([out[c] for c in "avj"]), columns.T @ columns)
    for v in range(len(ut)):
        hood = fmri[:, _valid(ncols[v])].astype(np.float64)
        hood = hood - hood.mean(0)
        target = (hood @ hood.T).ravel()
        assert np.abs(sp[:, v] - _brute_semipartial(target / np.linalg.norm(target), columns)).max() < 1e-4


def test_random_effects_match_scipy_t_tests():
    from scipy import stats

    from cka.cka_searchlight import random_effects

    values = np.random.default_rng(0).uniform(-0.2, 0.6, (12, 3, 5)).astype(np.float32)
    out = random_effects(values, ["cka_a", "cka_v", "cka_j"])
    z = np.arctanh(values.astype(np.float64))
    np.testing.assert_allclose(out["t_cka_v"], stats.ttest_1samp(z[:, 1], 0).statistic, rtol=1e-5)
    np.testing.assert_allclose(out["t_j_minus_a"], stats.ttest_rel(z[:, 2], z[:, 0]).statistic, rtol=1e-5)
    np.testing.assert_allclose(out["diff_j_minus_v"], (z[:, 2] - z[:, 1]).mean(0), rtol=1e-5)
    assert list(out) == ["t_cka_a", "t_cka_v", "t_cka_j", "diff_j_minus_a", "t_j_minus_a", "diff_j_minus_v", "t_j_minus_v"]
    sp = random_effects(values, ["sp_a", "sp_v", "sp_j"])
    assert list(sp) == ["t_sp_a", "t_sp_v", "t_sp_j"]
    np.testing.assert_allclose(sp["t_sp_j"], stats.ttest_1samp(z[:, 2], 0).statistic, rtol=1e-5)


def test_commonality_r2_is_the_least_squares_fit_of_the_brain_matrix_and_its_unique_j_is_sp_j_squared():
    rng = np.random.default_rng(12)
    columns = rng.normal(size=(50, 3))
    columns /= np.linalg.norm(columns, axis=0)
    brain = columns @ np.array([0.5, 0.3, 0.4]) + 0.5 * rng.normal(size=50)
    brain /= np.linalg.norm(brain)
    cka, cosines = (columns.T @ brain)[:, None], columns.T @ columns
    parts = kernels.commonality(cka, cosines)
    for subset in ("a", "av", "vj", "avj"):
        design = columns[:, ["avj".index(b) for b in subset]]
        fitted = design @ np.linalg.lstsq(design, brain, rcond=None)[0]
        np.testing.assert_allclose(parts[f"r2_{subset}"][0], fitted @ fitted, rtol=1e-8)
    np.testing.assert_allclose(parts["unique_j"][0], kernels.semipartial(cka, cosines)[2, 0] ** 2, rtol=1e-8)
    regions = ("unique_a", "unique_v", "unique_j", "shared_av_only", "shared_aj_only", "shared_vj_only", "shared_avj")
    np.testing.assert_allclose(sum(parts[k][0] for k in regions), parts["r2_avj"][0], rtol=1e-8)
    np.testing.assert_allclose(parts["shared_av"], parts["shared_av_only"] + parts["shared_avj"], rtol=1e-8)


def test_commonality_random_effects_use_raw_values():
    from scipy import stats

    from cka.cka_searchlight import random_effects

    values = np.random.default_rng(1).normal(0.01, 0.02, (12, 2, 5))
    out = random_effects(values, ["unique_j", "shared_av"], correlations=False)
    np.testing.assert_allclose(out["t_shared_av"], stats.ttest_1samp(values[:, 1], 0).statistic, rtol=1e-5)
