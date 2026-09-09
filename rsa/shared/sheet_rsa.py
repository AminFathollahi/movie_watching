#!/usr/bin/env python3
"""
rsa/shared/sheet_rsa.py
========================
Shared geometry, plotting, and hotspot/topography characterization for RSA
between CCA seed ROIs and Topo-Omni's cortical sheet (rsa/full_sheet_rsa.py).

Sheet geometry: `unified_sheet` is [304, 512] -- vision encoder (32 layers,
5 rows/layer) at rows 0-159/cols 0-255, audio encoder (32 layers, 5
rows/layer) at rows 0-159/cols 256-511, and the Thinker text stack (36
layers, 4 rows/layer) at rows 160-303/all cols. This third tower is the
autoregressive language-model backbone consuming fused audio+video tokens
(`Model Repos/topo-omni/src/models/qwen2_5_omni.py`'s `multimodal_cortical_
sheet`) -- it is not a Whisper-style audio decoder, so it is called "thinker"
here, never "decoder".

Coordinates: no trained coords.npy ships with the released checkpoint or its
HF cache. `load_true_coords` is a deterministic regeneration of
`init_coords.permute_coordinates(seed=42)`, validated bit-identical across
two torch builds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SHEET_ROWS = 304
SHEET_COLS = 512
N_UNITS = SHEET_ROWS * SHEET_COLS  # 155648
ENCODER_ROWS = 160  # rows 0-159: vision/audio encoders; 160-303: thinker stack

TOWER_VISION, TOWER_AUDIO, TOWER_THINKER = 0, 1, 2
TOWER_NAMES = {TOWER_VISION: "vision", TOWER_AUDIO: "audio", TOWER_THINKER: "thinker"}

TOPO_REPO = "/home/amin/Research/Representation/Movie/Model Repos/topo-omni"


# =============================================================================
# Sheet geometry
# =============================================================================

def tower_id() -> np.ndarray:
    """Per-unit tower id (0=vision, 1=audio, 2=thinker) from each unit's
    RASTER position -- see module docstring for the row/col boundaries."""
    k = np.arange(N_UNITS)
    row, col = k // SHEET_COLS, k % SHEET_COLS
    enc = row < ENCODER_ROWS
    tid = np.where(enc, np.where(col < 256, TOWER_VISION, TOWER_AUDIO), TOWER_THINKER)
    return tid.astype(np.int32)


def load_true_coords(cache_path: Path) -> np.ndarray:
    """(N_UNITS, 2) int64 true (row, col) per raster flat index k=row*512+col."""
    if cache_path.exists():
        return np.load(cache_path)
    import torch
    sys.path.insert(0, TOPO_REPO)
    sys.path.insert(0, f"{TOPO_REPO}/src")
    import init_coords as ic  # type: ignore
    ic.N_col = SHEET_COLS
    rng = torch.Generator()
    rng.manual_seed(42)
    coordinates = torch.Tensor([(i, j) for i in range(SHEET_ROWS) for j in range(SHEET_COLS)])
    coordinates = ic.permute_coordinates(coordinates, rng)
    arr = coordinates.numpy().astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def knn_on_sheet(coords: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours per unit in (row, col) space, self excluded,
    nearest-first. cKDTree (O(n log n)) -- brute-force cdist needs ~180 GB
    of distance matrix at N_UNITS scale."""
    tree = cKDTree(coords.astype(np.float64))
    _, idx = tree.query(coords.astype(np.float64), k=k + 1, workers=-1)
    return idx[:, 1:].astype(np.int32)


def random_neighbors_within_tower(tid: np.ndarray, k: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """k random neighbour indices per unit, drawn uniformly from the SAME
    tower only, coordinates ignored -- the topography control's null: does
    the true k-NN spatial neighbourhood outperform an arbitrary same-tower,
    same-size sample?"""
    n = tid.size
    out = np.empty((n, k), dtype=np.int32)
    pools = {t: np.flatnonzero(tid == t) for t in np.unique(tid)}
    for i in range(n):
        pool = pools[tid[i]]
        pool = pool[pool != i]
        out[i] = rng.choice(pool, size=k, replace=False)
    return out


# =============================================================================
# RSA helpers
# =============================================================================

def sanity_corr(sheet_emb: np.ndarray, seed_emb: np.ndarray) -> np.ndarray:
    """Plain Pearson r between the seed ROI's mean time series and each unit."""
    seed_mean = seed_emb.mean(axis=1)
    seed_z = (seed_mean - seed_mean.mean()) / (seed_mean.std() + 1e-12)
    unit_z = (sheet_emb - sheet_emb.mean(axis=0, keepdims=True)) / (
        sheet_emb.std(axis=0, keepdims=True) + 1e-12)
    return (unit_z * seed_z[:, None]).mean(axis=0).astype(np.float32)


def mean_nn_dist(xy: np.ndarray) -> float:
    """Mean nearest-OTHER-point Euclidean distance, via cKDTree."""
    return float(cKDTree(xy).query(xy, k=2)[0][:, 1].mean())


# =============================================================================
# Figures
# =============================================================================

def _canvas(values: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    rmin, rmax = int(coords[:, 0].min()), int(coords[:, 0].max())
    canvas = np.full((rmax - rmin + 1, SHEET_COLS), np.nan, dtype=np.float32)
    canvas[coords[:, 0] - rmin, coords[:, 1]] = values
    return canvas, rmin, rmax


def plot_sheet_map(ax, values: np.ndarray, coords: np.ndarray, title: str,
                   vlim: float, fdr_mask: np.ndarray | None, cmap: str):
    canvas, rmin, rmax = _canvas(values, coords)
    im = ax.imshow(canvas, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim,
                    extent=[0, SHEET_COLS, rmax + 0.5, rmin - 0.5])
    if fdr_mask is not None and fdr_mask.any():
        rows, cols = coords[fdr_mask, 0], coords[fdr_mask, 1]
        ax.scatter(cols + 0.5, rows + 0.5, s=4, facecolors="none",
                   edgecolors="black", linewidths=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sheet col")
    ax.set_ylabel("sheet row (true)")
    return im


# =============================================================================
# Cross-seed / hotspot characterization
# =============================================================================

def characterize(results: dict, coords: np.ndarray, tid: np.ndarray, args) -> dict:
    """cca_a vs. cca_p contrast, per-tower rho means, and hotspot composition
    + cross-seed hotspot overlap (Jaccard) for the top decile of each seed."""
    seed_a, seed_p = args.seed_a_name, args.seed_p_name
    a, p = results[f"{seed_a}_av"]["rho"], results[f"{seed_p}_av"]["rho"]
    n = a.size
    diff = a - p
    cross_r = float(np.corrcoef(a, p)[0, 1])
    max_i, min_i = int(np.argmax(diff)), int(np.argmin(diff))

    def _loc(i):
        return {"unit_index": i, "tower": TOWER_NAMES[int(tid[i])],
                "true_row": int(coords[i, 0]), "true_col": int(coords[i, 1])}

    per_tower = {}
    for t in sorted(TOWER_NAMES):
        m = tid == t
        a_mean, p_mean = float(a[m].mean()), float(p[m].mean())
        per_tower[TOWER_NAMES[t]] = {
            f"{seed_a}_rho_mean": a_mean, f"{seed_p}_rho_mean": p_mean,
            "diff_mean": a_mean - p_mean, "n_units": int(m.sum()),
        }

    rng = np.random.default_rng(args.seed)
    hotspots, top_masks = {}, {}
    for name, rho in ((seed_a, a), (seed_p, p)):
        n_top = int(np.ceil(args.pref_top_decile * n))
        mask = np.zeros(n, dtype=bool)
        mask[np.argsort(rho)[::-1][:n_top]] = True
        top_masks[name] = mask

        tower_counts = {TOWER_NAMES[t]: int(((tid == t) & mask).sum())
                        for t in sorted(TOWER_NAMES)}

        # Spatial contiguity: mean nearest-OTHER-hotspot-unit distance in true
        # coord space, vs. a null of random same-size unit subsets.
        top_xy = coords[mask].astype(np.float64)
        observed_nn = mean_nn_dist(top_xy)
        n_top_n = int(mask.sum())
        all_xy = coords.astype(np.float64)
        null_nn = np.empty(2000, dtype=np.float64)
        for i in range(2000):
            sel = rng.permutation(n)[:n_top_n]
            null_nn[i] = mean_nn_dist(all_xy[sel])
        p_contig = float((np.sum(null_nn <= observed_nn) + 1) / (2000 + 1))

        hotspots[name] = dict(
            n_top=n_top_n, tower_composition=tower_counts,
            observed_mean_nn_dist=observed_nn,
            null_mean_nn_dist_mean=float(null_nn.mean()),
            null_mean_nn_dist_p5=float(np.quantile(null_nn, 0.05)),
            p_more_contiguous_than_random=p_contig,
            interpretation=(
                "p < 0.05 means hotspot units sit closer together (in true "
                "coord space) than a random same-size subset of all units. "
                "Coordinates were generated under a spatial-smoothness training "
                "objective, so this is close to true by construction and should "
                "not be over-read as a fMRI-side finding."
            ),
        )

    overlap_n = int((top_masks[seed_a] & top_masks[seed_p]).sum())
    union_n = int((top_masks[seed_a] | top_masks[seed_p]).sum())
    jaccard = overlap_n / union_n if union_n else 0.0
    n_top_n = int(top_masks[seed_a].sum())
    null_overlap = np.empty(2000, dtype=np.int64)
    for i in range(2000):
        s1 = rng.permutation(n)[:n_top_n]
        s2 = rng.permutation(n)[:n_top_n]
        null_overlap[i] = np.intersect1d(s1, s2, assume_unique=True).size
    p_overlap = float((np.sum(null_overlap >= overlap_n) + 1) / (2000 + 1))

    return dict(
        cross_seed_spatial_pearson_r=cross_r,
        diff_mean=float(diff.mean()), diff_std=float(diff.std()),
        diff_positive_fraction=float((diff > 0).mean()),
        diff_max_loc=_loc(max_i), diff_max_value=float(diff[max_i]),
        diff_min_loc=_loc(min_i), diff_min_value=float(diff[min_i]),
        rho_mean_by_tower=per_tower,
        hotspots=hotspots,
        hotspot_overlap={
            "n_top_decile": n_top_n, "overlap_n": overlap_n, "union_n": union_n,
            "jaccard": jaccard, "p_more_overlap_than_random": p_overlap,
            "interpretation": (
                "p < 0.05 means cca_a's and cca_p's top-decile units share "
                "more sheet units than expected by chance -- the two seeds' "
                "hotspots are not independent."
            ),
        },
    )
