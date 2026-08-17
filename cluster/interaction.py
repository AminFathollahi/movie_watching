"""
cluster/interaction.py
========================
Relates stimulus-derived temporal states to brain-derived spatial networks:
for each (state k, network g) cell, how much does mean fMRI activity in
network g rise/fall while the stimulus is in state k, relative to network
g's grand mean over all segments? Significance is via a block-circular
permutation null that shifts state labels within each run (preserving run
boundaries and state autocorrelation).
"""

import logging

import numpy as np
from scipy.stats import false_discovery_control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _network_segment_means(fmri_binned: np.ndarray, vertex_labels: np.ndarray, networks):
    """(n_seg, V), (V,) -> (n_seg, G): per-segment mean fMRI within each network."""
    n_seg = fmri_binned.shape[0]
    out = np.empty((n_seg, len(networks)), dtype=np.float64)
    for gi, g in enumerate(networks):
        cols = vertex_labels == g
        out[:, gi] = fmri_binned[:, cols].mean(axis=1)
    return out


def interaction_matrix(fmri_binned: np.ndarray, state_labels: np.ndarray, vertex_labels: np.ndarray):
    """M[k, g] = mean over segments in state k of (mean fMRI in network g).

    Excludes vertex_labels == -1 (unassigned/noise) vertices.

    Returns
    -------
    M         : (K, G) float64
    states    : sorted unique state labels (row order)
    networks  : sorted unique non-noise network labels (col order)
    """
    states = np.unique(state_labels)
    networks = np.unique(vertex_labels[vertex_labels != -1])
    net_means = _network_segment_means(fmri_binned, vertex_labels, networks)  # (n_seg, G)

    M = np.empty((len(states), len(networks)), dtype=np.float64)
    for ki, k in enumerate(states):
        mask = state_labels == k
        M[ki] = net_means[mask].mean(axis=0)

    log.info(f"interaction_matrix: fmri_binned={fmri_binned.shape}  "
             f"K={len(states)}  G={len(networks)}  M={M.shape}")
    return M, states, networks


def block_permutation_test(fmri_binned: np.ndarray, state_labels: np.ndarray, vertex_labels: np.ndarray,
                           run_lengths, n_perm: int = 1000, correction: str = "fdr",
                           random_state: int = 0):
    """Block-circular-shift permutation test on interaction_matrix cells.

    Deviation[k, g] = M[k, g] - grand_mean[g], where grand_mean[g] is network
    g's mean over ALL segments (not just state k). Null: circularly shift
    state_labels within each run by an independent random offset (preserves
    run boundaries + within-state temporal autocorrelation), recompute the
    deviation, repeat n_perm times.

    Returns
    -------
    M              : (K, G) float64 — observed, descriptive, always meaningful
    pvals_corrected: (K, G) float64
    stats          : dict — correction, n_perm, null summary
    """
    run_lengths = np.asarray(run_lengths)
    n_seg = int(run_lengths.sum())
    assert n_seg == len(state_labels), (
        f"run_lengths sum {n_seg} != len(state_labels) {len(state_labels)}")
    run_starts = np.concatenate([[0], np.cumsum(run_lengths)])[:-1]

    states = np.unique(state_labels)
    networks = np.unique(vertex_labels[vertex_labels != -1])
    K, G = len(states), len(networks)
    net_means = _network_segment_means(fmri_binned, vertex_labels, networks)  # (n_seg, G)
    grand_mean = net_means.mean(axis=0)  # (G,)

    M, _, _ = interaction_matrix(fmri_binned, state_labels, vertex_labels)
    observed_dev = M - grand_mean[None, :]  # (K, G)

    rng = np.random.default_rng(random_state)
    null_devs = np.empty((n_perm, K, G), dtype=np.float64)
    state_to_row = {s: i for i, s in enumerate(states)}

    for p in range(n_perm):
        shifted = state_labels.copy()
        for start, length in zip(run_starts, run_lengths):
            length = int(length)
            offset = rng.integers(0, length)
            run_slice = slice(int(start), int(start) + length)
            shifted[run_slice] = np.roll(state_labels[run_slice], offset)

        M_null = np.empty((K, G), dtype=np.float64)
        for k in states:
            mask = shifted == k
            row = state_to_row[k]
            if mask.sum() == 0:
                M_null[row] = grand_mean  # deviation 0 if a state vanishes under this shift
            else:
                M_null[row] = net_means[mask].mean(axis=0)
        null_devs[p] = M_null - grand_mean[None, :]

    if correction == "fdr":
        # Two-sided per-cell p-value, then BH-FDR across all K*G cells.
        pvals_raw = (np.abs(null_devs) >= np.abs(observed_dev)[None]).mean(axis=0)
        pvals_raw = np.clip(pvals_raw, 1.0 / n_perm, 1.0)
        pvals_corrected = false_discovery_control(pvals_raw.ravel(), method="bh").reshape(K, G)
    elif correction == "maxstat":
        null_max = np.abs(null_devs).max(axis=(1, 2))  # (n_perm,)
        pvals_corrected = np.array(
            [[(null_max >= abs(observed_dev[ki, gi])).mean() for gi in range(G)] for ki in range(K)]
        )
        pvals_corrected = np.clip(pvals_corrected, 1.0 / n_perm, 1.0)
    else:
        raise ValueError(f"Unknown correction: {correction}")

    n_sig = int((pvals_corrected < 0.05).sum())
    stats = {
        "correction": correction, "n_perm": n_perm, "K": K, "G": G,
        "n_sig_q05": n_sig,
        "null_dev_abs_mean": float(np.abs(null_devs).mean()),
        "null_dev_abs_std": float(np.abs(null_devs).std()),
    }
    log.info(f"block_permutation_test: {stats}")
    return M, pvals_corrected, stats


def demo():
    rng = np.random.default_rng(0)
    run_lengths = [40, 35, 38]
    n_seg = sum(run_lengths)
    n_verts = 200

    # Two networks (network 0: verts 0-99, network 1: verts 100-199).
    vertex_labels = np.array([0] * 100 + [1] * 100)
    state_labels = np.concatenate([rng.integers(0, 2, l) for l in run_lengths])

    # Plant a real effect: network 0 activity is +2 higher during state 1.
    fmri_binned = rng.normal(scale=1.0, size=(n_seg, n_verts))
    fmri_binned[state_labels == 1, :100] += 2.0

    M, states, networks = interaction_matrix(fmri_binned, state_labels, vertex_labels)
    assert M.shape == (2, 2)

    M2, pvals, stats = block_permutation_test(
        fmri_binned, state_labels, vertex_labels, run_lengths, n_perm=500, correction="fdr")
    assert np.allclose(M, M2)
    row1 = np.where(states == 1)[0][0]
    assert pvals[row1, 0] < 0.05, f"planted state1 x network0 effect should survive FDR, p={pvals[row1, 0]}"
    print(f"[demo] block_permutation_test OK: planted effect p={pvals[row1, 0]:.4f}  stats={stats}")


if __name__ == "__main__":
    demo()
