import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cka.shared import kernels
from rsa.shared.rsa_utils import preprocess_fmri


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


def _signal_problem(seed, windows=9, vertices=8, size=3):
    rng = np.random.default_rng(seed)
    xa, xv, xj = (rng.standard_normal((windows, d)) for d in (5, 4, 6))
    base = np.stack([xa @ rng.standard_normal((5, vertices)), xv @ rng.standard_normal((4, vertices)), xj @ rng.standard_normal((6, vertices))])
    fmri = np.einsum("c,cwv->wv", rng.uniform(0.2, 1.0, 3), base) + rng.standard_normal((windows, vertices))
    ncols = np.stack([rng.permutation(vertices)[:size] for _ in range(vertices)]).astype(np.int32)
    return fmri.astype(np.float32), ncols, (xa, xv, xj)


def _brute_semipartial(target, grams):
    out = []
    for x in range(3):
        others = [i for i in range(3) if i != x]
        beta = np.linalg.lstsq(grams[:, others], grams[:, x], rcond=None)[0]
        residual = grams[:, x] - grams[:, others] @ beta
        out.append(target @ residual / np.linalg.norm(residual))
    return np.array(out)


def test_semipartial_matches_explicit_residualization_on_the_matrices():
    fmri, ncols, xs = _signal_problem(12)
    grams = {c: kernels.model_gram(x) for c, x in zip("avj", xs)}
    out = kernels.noncv_pass(fmri, grams, ncols, np.arange(fmri.shape[1]), "cpu")
    columns = np.stack(list(grams.values()), 1).astype(np.float64)
    sp = kernels.semipartial(np.stack([out[c] for c in "avj"]), columns.T @ columns)
    for v in range(fmri.shape[1]):
        hood = fmri[:, _valid(ncols[v])].astype(np.float64)
        hood = hood - hood.mean(0)
        target = (hood @ hood.T).ravel()
        assert np.abs(sp[:, v] - _brute_semipartial(target / np.linalg.norm(target), columns)).max() < 1e-4


def test_semipartial_is_zero_for_a_component_in_the_span_of_the_others():
    fmri, ncols, (xa, xv, _) = _signal_problem(11)
    scaled = [(x - x.mean(0)) / np.sqrt(np.linalg.norm((x - x.mean(0)) @ (x - x.mean(0)).T)) for x in (xa, xv)]
    grams = {c: kernels.model_gram(x) for c, x in zip("avj", (xa, xv, np.concatenate(scaled, 1)))}
    out = kernels.noncv_pass(fmri, grams, ncols, np.arange(fmri.shape[1]), "cpu")
    columns = np.stack(list(grams.values()), 1).astype(np.float64)
    sp = kernels.semipartial(np.stack([out[c] for c in "avj"]), columns.T @ columns)
    assert np.all(sp[2] == 0) and kernels.semipartial_coefficients(columns.T @ columns)[1][2] < 1e-6


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


def test_pool_gives_mean_standard_error_and_fractions():
    from cka.cka_searchlight import pool

    values = np.random.default_rng(1).uniform(0, 0.2, (12, 3, 5))
    out = pool(values, ["cka_a", "cka_v", "cka_j"])
    assert list(out[""]) == ["mean_cka_a", "mean_cka_v", "mean_cka_j", "frac_j_gt_a", "frac_j_gt_v"]
    np.testing.assert_allclose(out[""]["mean_cka_v"], values[:, 1].mean(0))
    np.testing.assert_allclose(out["_sem"]["sem_cka_j"], values[:, 2].std(0, ddof=1) / np.sqrt(12))
    np.testing.assert_allclose(out[""]["frac_j_gt_a"], (values[:, 2] > values[:, 0]).sum(0) / 12)
    assert list(pool(values, ["sp_a", "sp_v", "sp_j"])[""]) == ["mean_sp_a", "mean_sp_v", "mean_sp_j"]


def test_noise_ceiling_matches_brute_force():
    rng, ncols = _neighbourhoods()
    windows, n = 7, 5
    shared = rng.standard_normal((12, windows))
    subjects = [(shared + rng.standard_normal((12, windows))).astype(np.float16) for _ in range(n)]
    lower, upper = kernels.noise_ceiling_pass(subjects, ncols, np.arange(12), "cpu", batch=4)
    for v in range(12):
        grams = []
        for y in subjects:
            hood = y[_valid(ncols[v])].T.astype(np.float64)
            hood = hood - hood.mean(0)
            gram = hood @ hood.T
            grams.append(gram / np.linalg.norm(gram))
        mean = np.mean(grams, 0)
        assert abs(upper[v] - np.mean([_cka(g, mean) for g in grams])) < 1e-5
        others = [_cka(g, (n * mean - g) / (n - 1)) for g in grams]
        assert abs(lower[v] - np.mean(others)) < 1e-5
