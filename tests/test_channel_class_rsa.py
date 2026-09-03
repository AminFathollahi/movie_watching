"""Tests for category-restricted RSA input validation.

``rsa/channel_class_rsa.py`` is DEPRECATED (2026-09-02, see its module
docstring) -- it depends on the retired 4-class channel-sensitivity scheme
and should not be run. Its own CLASS_ORDER is now a local constant (no
longer imported from cf_modeling.channel_cca_analysis, which does not define
one any more). Every test below exercises only this module's self-contained
helpers (shift-index generation, class-table validation, map-naming,
spatial-agreement, and FDR/max-T map assembly) -- none of them run the
retired RSA pipeline itself, so none needed to be skipped; they remain valid
regression coverage for this module's plumbing even while the module is not
meant to be run.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "rsa" / "channel_class_rsa.py"
SPEC = importlib.util.spec_from_file_location("channel_class_rsa", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_within_run_shift_indices_are_reproducible_pair_bijections():
    first = MODULE.within_run_shift_pair_indices(np.array([4, 5]), 12, seed=7)
    second = MODULE.within_run_shift_pair_indices(np.array([4, 5]), 12, seed=7)
    assert first.shape == (12, 36)
    np.testing.assert_array_equal(first, second)
    expected = np.arange(36)
    assert all(np.array_equal(np.sort(row), expected) for row in first)
    assert all(not np.array_equal(row, expected) for row in first)


def test_load_classes_requires_complete_nonempty_partition(tmp_path):
    path = tmp_path / "classes.csv"
    pd.DataFrame({
        "channel": np.arange(8),
        "modality_significance_class": np.repeat(MODULE.CLASS_ORDER, 2),
    }).to_csv(path, index=False)
    result = MODULE._load_classes(path, 8)
    assert result.modality_significance_class.value_counts().to_dict() == {
        name: 2 for name in MODULE.CLASS_ORDER}


def test_load_classes_rejects_missing_class(tmp_path):
    path = tmp_path / "classes.csv"
    pd.DataFrame({
        "channel": np.arange(4),
        "modality_significance_class": ["audio_only", "video_only", "both", "both"],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Empty channel classes"):
        MODULE._load_classes(path, 4)


def test_analysis_parser_preserves_mask_provenance():
    parsed = MODULE._parse_analysis([
        "peav", "top1pct_peav", "clsav", "embedding.npy", "classes.csv"])
    assert parsed["roi_set"] == "top1pct_peav"
    assert parsed["representation_tag"] == "clsav"
    assert MODULE._map_base(parsed, "audio_only") == \
        "peav_clsav_top1pct_peav_significance_audio_only"


def test_map_base_does_not_duplicate_representation_suffix():
    parsed = MODULE._parse_analysis([
        "nemotron_layer18_mp", "top1pct_nemotron_layer_18_mp", "mp",
        "embedding.npy", "classes.csv"])
    assert MODULE._map_base(parsed, "video_only") == \
        "nemotron_layer18_mp_top1pct_nemotron_layer_18_mp_significance_video_only"


def test_spatial_agreement_is_exact_for_identical_maps():
    values = np.linspace(-1, 1, 1000)
    cortex = np.ones(1000, dtype=bool)
    result = MODULE._spatial_agreement(values, values.copy(), cortex)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["mean_absolute_difference"] == 0.0
    assert result["top1pct_dice"] == 1.0
    assert result["top1pct_jaccard"] == 1.0


def test_inference_bundle_contains_raw_fdr_and_maxT_values_and_masks():
    rho = np.linspace(-0.3, 0.5, 100)
    p = np.linspace(0.001, 1.0, 100)
    p_maxT = np.minimum(1.0, p * 4)
    maps = MODULE._inference_maps(rho, p, p_maxT)
    assert set(maps) == {
        "searchlight_rho", "searchlight_p_perm_uncorrected",
        "searchlight_sigmap_perm_uncorrected",
        "searchlight_perm_uncorrected_mask_0p05",
        "searchlight_q_bh_fdr", "searchlight_sigmap_bh_fdr",
        "searchlight_bh_fdr_mask_0p05", "searchlight_p_maxT_fwer",
        "searchlight_sigmap_maxT_fwer", "searchlight_maxT_fwer_mask_0p05",
    }
    assert all(values.shape == rho.shape for values in maps.values())
    assert np.all((maps["searchlight_p_perm_uncorrected"] >= 0) &
                  (maps["searchlight_p_perm_uncorrected"] <= 1))
    assert np.all((maps["searchlight_q_bh_fdr"] >= 0) &
                  (maps["searchlight_q_bh_fdr"] <= 1))
    assert np.array_equal(
        maps["searchlight_bh_fdr_mask_0p05"],
        maps["searchlight_q_bh_fdr"] < 0.05)
    assert np.array_equal(
        maps["searchlight_maxT_fwer_mask_0p05"],
        maps["searchlight_p_maxT_fwer"] < 0.05)


def test_maxT_critical_rho_matches_strict_pseudocount_threshold():
    null_max = np.arange(500, dtype=float)
    critical = MODULE._maxT_critical_rho(null_max, alpha=0.05)
    p = ((null_max[:, None] >= np.array([critical - 0.1, critical + 0.1]))
         .sum(axis=0) + 1) / 501
    assert p[0] >= 0.05
    assert p[1] < 0.05


def test_bh_family_mask_excludes_non_cortical_positions():
    rho = np.array([0.4, 0.3, 0.0, 0.0])
    raw = np.array([0.01, 0.04, 1.0, 1.0])
    maxT = np.array([0.03, 0.2, 1.0, 1.0])
    maps = MODULE._inference_maps(
        rho, raw, maxT, family_mask=np.array([True, True, False, False]))
    np.testing.assert_allclose(maps["searchlight_q_bh_fdr"][:2], [0.02, 0.04])
    np.testing.assert_allclose(maps["searchlight_q_bh_fdr"][2:], 1.0)
