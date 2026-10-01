import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.searchlight import _precompute_model_rdm, _searchlight_vertex_fast, run_searchlight_gpu

METRIC = {"correlation": "correlation", "euclidean": "sqeuclidean"}


def make_data(n_bins=24, n_verts=40, k=6, n_features=11, seed=0):
    rng = np.random.default_rng(seed)
    fmri = rng.standard_normal((n_bins, n_verts)).astype(np.float32)
    emb = rng.standard_normal((n_bins, n_features))
    emb[:, :3] += fmri[:, :3] @ rng.standard_normal((3, 3))
    neighbors = np.stack([rng.choice(np.delete(np.arange(n_verts), v), k, replace=False) for v in range(n_verts)]).astype(np.int32)
    return fmri, emb, neighbors


def direct(fmri, emb, neighbors, vertex, distance):
    brain = pdist(fmri[:, neighbors[vertex]].astype(np.float64), METRIC[distance])
    model = pdist(emb, METRIC[distance])
    return spearmanr(brain, model)[0]


@pytest.mark.parametrize("distance", ["correlation", "euclidean"])
def test_vertex_and_batched_paths_match_direct_rsa(distance):
    fmri, emb, neighbors = make_data()
    n_bins, n_verts = fmri.shape
    tril = np.tril_indices(n_bins, k=-1)
    _, model_norm = _precompute_model_rdm(emb, n_bins, tril, "spearman", distance)
    identity = np.arange(n_verts, dtype=np.int32)
    batched = run_searchlight_gpu(fmri, emb, neighbors, identity, identity, "spearman", batch_size=7, device="cpu", distance=distance)
    for v in range(n_verts):
        expected = direct(fmri, emb, neighbors, v, distance)
        single = _searchlight_vertex_fast(v, fmri, model_norm, neighbors, identity, tril, "spearman", distance)
        assert single == pytest.approx(expected, abs=2e-4)
        assert batched[v] == pytest.approx(expected, abs=2e-4)


def test_euclidean_differs_from_correlation():
    fmri, emb, neighbors = make_data(seed=1)
    fmri[:, :] *= np.linspace(0.5, 3, fmri.shape[0])[:, None].astype(np.float32)
    assert abs(direct(fmri, emb, neighbors, 0, "correlation") - direct(fmri, emb, neighbors, 0, "euclidean")) > 1e-3
