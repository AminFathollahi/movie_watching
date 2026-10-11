import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import mahalanobis, pdist
from scipy.stats import kendalltau, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsa.rdm_searchlight import model_rdm, partial_correlations, residual, searchlight, unit_ranks


def _setup(seed, n_bins=24, n_vert=6, width=4):
    rng = np.random.default_rng(seed)
    fmri = rng.standard_normal((n_bins, n_vert)).astype(np.float32)
    neighbors = np.stack([rng.permutation(n_vert)[:width] for _ in range(n_vert)])
    return rng, fmri, neighbors, np.tril_indices(n_bins, k=-1)


def _rsatoolbox_shrinkage_diag(matrix, dof):
    # Transcribed from rsatoolbox.data.noise._covariance_diag.
    s_sum = np.zeros((matrix.shape[1], matrix.shape[1]))
    s2_sum = np.zeros((matrix.shape[1], matrix.shape[1]))
    for m_line in matrix:
        xt_x = np.outer(m_line, m_line)
        s_sum += xt_x
        s2_sum += xt_x ** 2
    s = s_sum / dof
    var = np.diag(s)
    std = np.sqrt(var)
    s_mean = s_sum / np.expand_dims(std, 0) / np.expand_dims(std, 1) / (matrix.shape[0] - 1)
    s2_mean = s2_sum / np.expand_dims(var, 0) / np.expand_dims(var, 1) / (matrix.shape[0] - 1)
    var_hat = matrix.shape[0] / dof ** 2 * (s2_mean - s_mean ** 2)
    mask = ~np.eye(s.shape[0], dtype=bool)
    lamb = max(min(np.sum(var_hat[mask]) / np.sum(s_mean[mask] ** 2), 1), 0)
    return s * (np.eye(s.shape[0]) + (1 - lamb) * mask)


def test_kendall_cosine_matches_scipy_with_ties():
    rng, fmri, neighbors, tril = _setup(0)
    fmri[5] = fmri[2]
    emb = rng.standard_normal((24, 6))
    emb[[7, 9]] = emb[3]
    targets = [model_rdm(emb, tril, "cosine"), np.round(model_rdm(emb, tril, "cosine"), 1)]
    taus = searchlight(fmri, targets, neighbors, np.arange(6), 24, "cosine", "kendall")
    for v in range(6):
        hood = fmri[:, neighbors[v]].astype(np.float64)
        hood /= np.linalg.norm(hood, axis=1, keepdims=True)
        brain = (1 - hood @ hood.T)[tril]
        for t, target in enumerate(targets):
            assert np.isclose(taus[t, v], kendalltau(brain, target).statistic, atol=1e-10)


def test_spearman_semipartial_and_partial_euclidean():
    rng, fmri, neighbors, tril = _setup(1)
    emb = rng.standard_normal((24, 5))
    rdms = [model_rdm(emb + rng.standard_normal(emb.shape) * s, tril, "euclidean") for s in (0.5, 0.8, 0.3)]
    assert np.allclose(np.sort(model_rdm(emb, tril, "euclidean")), np.sort(pdist(emb, "euclidean")))
    resid = [residual(rdms[x], [rdms[i] for i in range(3) if i != x]) for x in range(3)]
    out = searchlight(fmri, rdms + resid, neighbors, np.arange(6), 24, "euclidean")
    ranks = np.stack([unit_ranks(r) for r in rdms])
    pc = partial_correlations(out[:3], ranks @ ranks.T)
    model = [rankdata(r) for r in rdms]
    for v in range(6):
        hood = fmri[:, neighbors[v]].astype(np.float64)
        brain = ((hood[:, None] - hood[None]) ** 2).sum(-1)[tril]
        for x in range(3):
            others = [model[i] for i in range(3) if i != x]
            assert np.isclose(out[x, v], spearmanr(brain, rdms[x]).statistic)
            assert np.isclose(out[3 + x, v], spearmanr(brain, resid[x]).statistic)
            assert np.isclose(pc[x, v], np.corrcoef(residual(rankdata(brain), others), residual(model[x], others))[0, 1])


def test_mahalanobis_matches_rsatoolbox_shrinkage():
    rng, fmri, neighbors, tril = _setup(2, n_bins=30, width=5)
    rep_rows = np.r_[np.arange(0, 4), np.arange(10, 14), np.arange(20, 24)]
    rep_cond = np.tile(np.arange(4), 3)
    target = model_rdm(rng.standard_normal((30, 4)), tril, "euclidean")
    rho = searchlight(fmri, [target], neighbors, np.arange(6), 30, "mahalanobis", repeats=(rep_rows, rep_cond))
    for v in range(6):
        hood = fmri[:, neighbors[v]].astype(np.float64)
        rep = hood[rep_rows]
        for c in range(4):
            rep[rep_cond == c] -= rep[rep_cond == c].mean(0)
        precision = np.linalg.inv(_rsatoolbox_shrinkage_diag(rep, len(rep_rows) - 4))
        brain = np.array([mahalanobis(hood[i], hood[j], precision) ** 2 for i, j in zip(*tril)])
        assert np.isclose(rho[0, v], spearmanr(brain, target).statistic)


def test_subject_gpu_path_matches_searchlight():
    from rsa.rdm_searchlight import hood_columns, subject_spearman_gpu
    rng, fmri, neighbors, tril = _setup(3, n_bins=30, n_vert=8, width=5)
    targets = [model_rdm(rng.standard_normal((30, 4)), tril, "euclidean") for _ in range(3)]
    expected = searchlight(fmri, targets, neighbors, np.arange(8), 30, "euclidean")
    cols, ncols = hood_columns(neighbors, np.arange(8))
    got = subject_spearman_gpu(fmri, cols, ncols, np.stack([unit_ranks(t) for t in targets]), batch=3)
    assert np.allclose(got, expected, atol=1e-5)


def test_crossnobis_matches_explicit_cross_validation():
    rng, fmri, neighbors, _ = _setup(4, n_bins=40, width=5)
    n_cond, n_part = 6, 4
    rep_rows = np.concatenate([np.arange(10 * m, 10 * m + n_cond) for m in range(n_part)])
    rep_cond = np.tile(np.arange(n_cond), n_part)
    tril = np.tril_indices(n_cond, k=-1)
    target = model_rdm(rng.standard_normal((n_cond, 4)), tril, "euclidean")
    rho = searchlight(fmri, [target], neighbors, np.arange(6), n_cond, "crossnobis", repeats=(rep_rows, rep_cond))
    for v in range(6):
        hood = fmri[:, neighbors[v]].astype(np.float64)
        parts = [hood[10 * m:10 * m + n_cond] for m in range(n_part)]
        resid = np.concatenate([p - np.mean(parts, axis=0) for p in parts])
        P = np.linalg.inv(_rsatoolbox_shrinkage_diag(resid, n_part * n_cond - n_cond))
        d = np.zeros(len(tril[0]))
        for a in range(n_part):
            for b in range(n_part):
                if a != b:
                    da = parts[a][tril[0]] - parts[a][tril[1]]
                    db = parts[b][tril[0]] - parts[b][tril[1]]
                    d += np.einsum("pi,ij,pj->p", da, P, db)
        assert np.isclose(rho[0, v], spearmanr(d, target).statistic)
