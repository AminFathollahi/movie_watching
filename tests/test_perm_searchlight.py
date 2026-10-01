import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.perm_searchlight import (
    _perm_vertex_cpu, _perm_vertex_cpu_paired, within_run_shift_pair_indices,
)
from rsa.searchlight import _precompute_model_rdm


def test_paired_null_matches_single_seed_and_shares_permutation():
    """_perm_vertex_cpu_paired must (1) reproduce the existing against-zero
    rho/exceedance for each seed exactly, and (2) apply the SAME perm_idx_all
    to both seeds when forming the difference null -- the paired-null
    assumption full_sheet_rsa.py's difference test depends on."""
    rng = np.random.default_rng(0)
    n_bins, n_units = 16, 5
    n_perm = 25
    perm_idx = within_run_shift_pair_indices(np.array([n_bins]), n_perm, seed=7)

    fmri = rng.standard_normal((n_bins, n_units)).astype(np.float32)
    emb_a = rng.standard_normal((n_bins, 4)).astype(np.float32)
    emb_p = rng.standard_normal((n_bins, 4)).astype(np.float32)
    neighbors = np.array([[1, 2, 3]], dtype=np.int32)  # neighbours for surf_v=0
    vertex_to_col = np.arange(n_units, dtype=np.int32)
    tril_idx = np.tril_indices(n_bins, k=-1)

    _, model_norm_a = _precompute_model_rdm(emb_a, n_bins, tril_idx, "spearman")
    _, model_norm_p = _precompute_model_rdm(emb_p, n_bins, tril_idx, "spearman")

    rho_a, exc_a, rho_p, exc_p, exc_diff = _perm_vertex_cpu_paired(
        0, fmri, model_norm_a, model_norm_p, neighbors, vertex_to_col,
        tril_idx, "spearman", perm_idx)

    rho_a_ref, exc_a_ref, _ = _perm_vertex_cpu(
        0, fmri, model_norm_a, neighbors, vertex_to_col, tril_idx, "spearman", perm_idx)
    rho_p_ref, exc_p_ref, _ = _perm_vertex_cpu(
        0, fmri, model_norm_p, neighbors, vertex_to_col, tril_idx, "spearman", perm_idx)
    assert np.isclose(rho_a, rho_a_ref) and exc_a == exc_a_ref
    assert np.isclose(rho_p, rho_p_ref) and exc_p == exc_p_ref

    # Manually recompute the shared brain RDM and the paired diff null using
    # the SAME perm_idx for both seeds, and check the exceedance matches.
    hood = fmri[:, [1, 2, 3]]
    mu = hood.mean(axis=1, keepdims=True)
    hn = (hood - mu) / np.sqrt(((hood - mu) ** 2).sum(axis=1, keepdims=True))
    fmri_flat = (1.0 - hn @ hn.T)[tril_idx]
    order = np.argsort(fmri_flat)
    fr = np.empty(len(fmri_flat), dtype=np.float32)
    fr[order] = np.arange(len(fmri_flat), dtype=np.float32)
    fc = fr - fr.mean()
    brain_norm = fc / np.linalg.norm(fc)

    null_rho_a = model_norm_a[perm_idx] @ brain_norm
    null_rho_p = model_norm_p[perm_idx] @ brain_norm   # same perm_idx as null_rho_a
    null_diff = null_rho_a - null_rho_p
    observed_diff = rho_a - rho_p
    exc_diff_expected = int((np.abs(null_diff) >= abs(observed_diff)).sum())
    assert exc_diff == exc_diff_expected

    # An UNPAIRED null (independently shuffled perm streams per seed) would
    # generally give a different exceedance count -- confirms the test above
    # is actually sensitive to the pairing, not vacuously true.
    perm_idx_p_shuffled = perm_idx[rng.permutation(n_perm)]
    null_rho_p_unpaired = model_norm_p[perm_idx_p_shuffled] @ brain_norm
    null_diff_unpaired = null_rho_a - null_rho_p_unpaired
    exc_diff_unpaired = int((np.abs(null_diff_unpaired) >= abs(observed_diff)).sum())
    assert exc_diff_unpaired != exc_diff_expected or n_perm < 5  # tiny n_perm could tie


if __name__ == "__main__":
    test_paired_null_matches_single_seed_and_shares_permutation()
    print("OK")
