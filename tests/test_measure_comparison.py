import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "measure_comparison"))

import compare_measures as cm


def test_neighbourhood_homogeneity_matches_direct_mean_pairwise_correlation():
    rng = np.random.default_rng(0)
    fmri = rng.standard_normal((30, 12)).astype(np.float32)
    ncols = np.stack([rng.choice(np.delete(np.arange(12), v), 5, replace=False) for v in range(12)]).astype(np.int32)
    ncols[3, 3:] = -1
    ncols[4, 1:] = -1
    h = cm.neighbourhood_homogeneity(fmri, ncols, chunk=5)
    for v in (0, 3, 7):
        cols = ncols[v][ncols[v] >= 0]
        r = np.corrcoef(fmri[:, cols].T)
        assert abs(h[v] - r[np.triu_indices(len(cols), 1)].mean()) < 1e-5
    assert h[4] == 0.0


def test_core_bins_follow_the_count_of_neighbours_in_the_core():
    core = np.zeros(200, bool)
    core[:100] = True
    rows = [np.r_[np.arange(c), np.arange(100, 200 - c)] for c in (0, 1, 10, 11, 100)]
    rows.append(np.r_[np.arange(10), np.arange(100, 185), np.full(5, -1)])
    assert cm.core_bins(core, np.stack(rows)).tolist() == [0, 1, 1, 2, 10, 1]


def test_map_paths_use_the_diagnostics_folders_and_cka_names():
    args = Namespace(k=100, delay_sec=5.0, bin_sec=5.0, skip_sec=5.0, scaling="center", fmri_suffix="raw", subject="group_average",
                     rsa_dir="rsa", cka_dir="cka")
    paths = cm.map_paths(args, "m")
    assert paths["euclid-spearman"].as_posix() == "rsa/raw/group_average/m_av/diagnostics/rsa_59k_raw_k100_delay5s_bin5s_skip5s_euclid-spearman_center_maps.dscalar.nii"
    assert paths["corr-spearman_nr"].name.endswith("_corr-spearman_center_norepeats_maps.dscalar.nii")
    assert paths["cka"].as_posix() == "cka/raw/group_average/m_av/diagnostics/cka_59k_raw_k100_delay5s_bin5s_skip5s_noncv_center_maps.dscalar.nii"
    assert paths["cka-cv-ar"].name == "cka_59k_raw_k100_delay5s_bin5s_skip5s_cv-ar_center_maps.dscalar.nii"
    assert cm.difference_path(args, "a", "b", "cv").name == "cka_59k_raw_k100_delay5s_bin5s_skip5s_cv-loglik-diff_a_av_minus_b_av_center_maps.dscalar.nii"
