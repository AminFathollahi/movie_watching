import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.sheet_rsa import (
    N_UNITS, SHEET_COLS, TOWER_AUDIO, TOWER_NAMES, TOWER_THINKER, TOWER_VISION,
    characterize, knn_on_sheet, random_neighbors_within_tower, tower_id,
)


def test_tower_id_boundaries_and_counts():
    tid = tower_id()
    assert tid.shape == (N_UNITS,)
    assert (tid == TOWER_VISION).sum() == 160 * 256
    assert (tid == TOWER_AUDIO).sum() == 160 * 256
    assert (tid == TOWER_THINKER).sum() == 144 * 512
    assert tid[159 * SHEET_COLS + 255] == TOWER_VISION
    assert tid[159 * SHEET_COLS + 256] == TOWER_AUDIO
    assert tid[160 * SHEET_COLS + 0] == TOWER_THINKER


def test_knn_on_sheet_tiny_lattice():
    coords = np.array([(r, c) for r in range(2) for c in range(4)])
    nn = knn_on_sheet(coords, k=2)
    assert set(nn[0].tolist()) == {1, 4}


def test_random_neighbors_within_tower_respects_groups_and_self_exclusion():
    tid = np.array([0] * 50 + [1] * 50)
    rng = np.random.default_rng(0)
    nb = random_neighbors_within_tower(tid, k=10, rng=rng)
    assert nb.shape == (100, 10)
    for i in range(100):
        assert i not in nb[i]
        assert (tid[nb[i]] == tid[i]).all()
        assert len(set(nb[i].tolist())) == 10


def test_characterize_per_tower_and_hotspot_overlap():
    rng = np.random.default_rng(1)
    n = 300
    tid = np.repeat([0, 1, 2], n // 3)
    coords = np.stack([np.arange(n), np.arange(n)], axis=1)
    a = rng.standard_normal(n).astype(np.float32)
    p = a + rng.standard_normal(n).astype(np.float32) * 0.1

    class Args:
        seed_a_name, seed_p_name, seed, pref_top_decile = "cca_a", "cca_p", 42, 0.1

    results = {"cca_a_av": {"rho": a}, "cca_p_av": {"rho": p}}
    out = characterize(results, coords, tid, Args())

    for t in sorted(TOWER_NAMES):
        m = tid == t
        entry = out["rho_mean_by_tower"][TOWER_NAMES[t]]
        assert np.isclose(entry["cca_a_rho_mean"], a[m].mean())
        assert np.isclose(entry["diff_mean"],
                          entry["cca_a_rho_mean"] - entry["cca_p_rho_mean"])

    overlap = out["hotspot_overlap"]
    assert 0.0 <= overlap["jaccard"] <= 1.0
    assert overlap["overlap_n"] <= overlap["union_n"]
