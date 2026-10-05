"""
rsa/spatial_stats.py
=======================
2D spatial-compactness metrics for a set of TopoOmni cortical-sheet units,
used to test whether a localizer cluster's selected units also sit together
on TopoOmni's own internal 2D sheet coordinates (the paper's own internal
validation via Island Moran's I), or are scattered.

Grid-position formula (derived from the model source, not guessed)
--------------------------------------------------------------------
TopoOmni's `unified_sheet` tensor (Qwen2_5OmniThinkerForConditionalGeneration.
forward(), qwen2_5_omni.py:1230-1253, Model Repos/topo-omni) has shape
(T, 304, 512): rows 0-159 hold the visual+audio cortical sheet, rows 160-303
hold the text-decoder ("multimodal") cortical sheet, laid out as

    multimodal_cortical_sheet  # [L=36, T, D=2048]
    .permute(1, 0, 2)          # [T, L, D]
    .reshape(-1, 144, 512)     # [T, 144, 512], since L*D = 144*512

i.e. layer L's D=2048 channels are flattened row-major into 4 rows of 512
columns each (2048 / 512 = 4), placed at global rows [160 + 4L, 160 + 4L + 4).
A channel index d in [0, 2047) at text-decoder layer L therefore sits at

    global_row = 160 + 4*L + d // 512
    col        =            d %  512

`coords.npy` (the model's real trained coordinate registry) is confirmed
absent from both the cloned repo and the downloaded HF snapshot, so this
uses the paper's own documented fallback: the trivial row-major (304, 512)
grid (`load_positions()`, qwen2_5_omni.py:782-805, when coords.npy is
missing). This formula needs no forward pass and no `unified_sheet` runtime
tensor -- a unit's grid position is fixed by (layer, channel) alone, so it
works for our 5s segments even though `unified_sheet` itself collapses to
empty at that clip length.

Only the TEXT-DECODER layers (9, 18, 27 -- our sheet models) are supported;
visual/audio cortical-sheet positions are not needed here.
"""

import numpy as np

N_ROWS, N_COLS = 304, 512
TEXT_ROW_OFFSET = 160
ROWS_PER_LAYER = 4  # 2048 channels / 512 cols


def topoomni_grid_positions(unit_indices: np.ndarray, layer_idx: int) -> np.ndarray:
    """Map cortical-sheet channel indices (0-2047, at text-decoder layer
    `layer_idx`) to (row, col) on the (304, 512) unified_sheet grid.
    Returns an (n_units, 2) int array of [row, col]."""
    unit_indices = np.asarray(unit_indices)
    row = TEXT_ROW_OFFSET + ROWS_PER_LAYER * layer_idx + unit_indices // 512
    col = unit_indices % 512
    return np.stack([row, col], axis=1)


def _local_grid_from_units(unit_indices: np.ndarray) -> np.ndarray:
    """Binary indicator map on the local (ROWS_PER_LAYER, 512) subgrid a single
    layer's units occupy. Global row offset is a pure translation and does not
    affect any adjacency/distance-based statistic below, so we work locally."""
    grid = np.zeros((ROWS_PER_LAYER, N_COLS), dtype=np.float64)
    local_row = unit_indices // 512
    col = unit_indices % 512
    grid[local_row, col] = 1.0
    return grid


def morans_i(field: np.ndarray, contiguity: str = "rook") -> float:
    """Moran's I spatial autocorrelation for a 2D field (no wraparound).
    contiguity: 'rook' (4-neighbor) or 'queen' (8-neighbor)."""
    if contiguity == "rook":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif contiguity == "queen":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError(f"Unknown contiguity: {contiguity!r}")

    z = field - field.mean()
    n = field.size
    numer = 0.0
    w_sum = 0.0
    padded = np.pad(z, 1, mode="constant", constant_values=0.0)
    valid = np.pad(np.ones_like(z), 1, mode="constant", constant_values=0.0)
    for dr, dc in offsets:
        shifted = padded[1 + dr: 1 + dr + z.shape[0], 1 + dc: 1 + dc + z.shape[1]]
        shifted_valid = valid[1 + dr: 1 + dr + z.shape[0], 1 + dc: 1 + dc + z.shape[1]]
        numer += float(np.sum(z * shifted))
        w_sum += float(np.sum(shifted_valid))
    denom = float(np.sum(z ** 2))
    if denom == 0.0 or w_sum == 0.0:
        return 0.0
    return (n / w_sum) * (numer / denom)


def island_morans_i(unit_indices: np.ndarray, n_perm: int = 2000, seed: int = 0,
                     contiguity: str = "rook") -> dict:
    """Island Moran's I: is this set of sheet units spatially compact (an
    'island') on the local (ROWS_PER_LAYER, 512) subgrid its layer occupies,
    vs. a random same-size subset of that layer's 2048 units?

    Returns dict with observed I, permutation-null mean/std, z-score, and a
    two-sided empirical p-value (fraction of null |I| >= observed |I|)."""
    unit_indices = np.asarray(unit_indices)
    n_units = len(unit_indices)
    field = _local_grid_from_units(unit_indices)
    i_obs = morans_i(field, contiguity=contiguity)

    rng = np.random.default_rng(seed)
    all_units = np.arange(2048)
    null_i = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        sample = rng.choice(all_units, size=n_units, replace=False)
        null_i[p] = morans_i(_local_grid_from_units(sample), contiguity=contiguity)

    null_mean, null_std = float(null_i.mean()), float(null_i.std())
    z = (i_obs - null_mean) / null_std if null_std > 0 else 0.0
    p_value = float(np.mean(np.abs(null_i) >= abs(i_obs)))
    return dict(
        morans_i=float(i_obs), null_mean=null_mean, null_std=null_std,
        z_score=float(z), p_value=p_value, n_units=int(n_units), n_perm=n_perm,
    )


def compactness_index(unit_indices: np.ndarray, n_perm: int = 2000, seed: int = 0) -> dict:
    """Companion 2D metric: mean pairwise Euclidean distance between selected
    units' grid positions (local subgrid), vs. a random-subset permutation
    null. Smaller than the null -> units sit closer together than chance
    (compact); larger -> more spread out than chance."""
    unit_indices = np.asarray(unit_indices)
    n_units = len(unit_indices)
    if n_units < 2:
        return dict(mean_pairwise_dist=0.0, null_mean=0.0, null_std=0.0,
                     z_score=0.0, p_value=1.0, n_units=int(n_units), n_perm=n_perm)

    def _mean_pairwise(indices):
        local_row = indices // 512
        col = indices % 512
        pts = np.stack([local_row, col], axis=1).astype(np.float64)
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        iu = np.triu_indices(len(indices), k=1)
        return float(dists[iu].mean())

    d_obs = _mean_pairwise(unit_indices)
    rng = np.random.default_rng(seed)
    all_units = np.arange(2048)
    null_d = np.array([_mean_pairwise(rng.choice(all_units, size=n_units, replace=False))
                        for _ in range(n_perm)])
    null_mean, null_std = float(null_d.mean()), float(null_d.std())
    z = (d_obs - null_mean) / null_std if null_std > 0 else 0.0
    p_value = float(np.mean(null_d <= d_obs))  # one-sided: compact = smaller-than-null
    return dict(
        mean_pairwise_dist=d_obs, null_mean=null_mean, null_std=null_std,
        z_score=float(z), p_value=p_value, n_units=int(n_units), n_perm=n_perm,
    )


def demo():
    """Self-check: a tight 2x2 block should look compact (positive Moran's I,
    below-null mean pairwise distance); units scattered one-per-row-per-far-
    apart-column should not."""
    compact_units = np.array([0, 1, 512, 513])  # local rows 0-1, cols 0-1: a 2x2 block
    r = island_morans_i(compact_units, n_perm=500, seed=1)
    assert r["morans_i"] > 0, f"expected positive Moran's I for a compact block, got {r}"
    c = compactness_index(compact_units, n_perm=500, seed=1)
    assert c["mean_pairwise_dist"] < c["null_mean"], f"expected below-null spread, got {c}"

    scattered_units = np.array([0, 200, 400, 511, 1000 + 300, 1500 + 100])
    cs = compactness_index(scattered_units, n_perm=500, seed=1)
    assert cs["mean_pairwise_dist"] > c["mean_pairwise_dist"]

    pos = topoomni_grid_positions(np.array([0, 511, 512, 2047]), layer_idx=9)
    expected = np.array([[196, 0], [196, 511], [197, 0], [199, 511]])
    assert np.array_equal(pos, expected), f"grid position formula mismatch: {pos}"

    print("spatial_stats.py: all checks passed")


if __name__ == "__main__":
    demo()
