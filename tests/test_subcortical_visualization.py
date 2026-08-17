import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subcortical.subcortical_io import SUBCORTICAL_STRUCTURES
from subcortical.subcortical_visualization import (
    ANATOMICAL_GROUPS,
    discover_group_results,
    result_category_and_slug,
    _assert_vertex_count,
    _save_multimetric,
)


def test_anatomical_groups_cover_each_structure_once():
    parts = [part for cfg in ANATOMICAL_GROUPS.values() for part in cfg["parts"]]
    assert sorted(parts) == sorted(SUBCORTICAL_STRUCTURES)
    assert len(parts) == len(set(parts))


def test_output_units_are_bilateral_except_cerebellum_and_brainstem():
    assert ANATOMICAL_GROUPS["CEREBELLUM"]["parts"] == (
        "CEREBELLUM_LEFT", "CEREBELLUM_RIGHT"
    )
    assert ANATOMICAL_GROUPS["BRAIN_STEM"]["parts"] == ("BRAIN_STEM",)
    for name, cfg in ANATOMICAL_GROUPS.items():
        if name not in {"CEREBELLUM", "BRAIN_STEM"}:
            assert name.startswith("BILATERAL_")
            assert len(cfg["parts"]) == 2
            assert cfg["parts"][0].endswith("_LEFT")
            assert cfg["parts"][1].endswith("_RIGHT")


def test_discovery_excludes_subject_results(tmp_path: Path):
    expected = []
    for folder in ("group_average", "groupstats", "noise_ceiling"):
        path = tmp_path / folder / "model" / "result.dscalar.nii"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        expected.append(path)
    subject = tmp_path / "subject_data" / "100610" / "result.dscalar.nii"
    subject.parent.mkdir(parents=True)
    subject.touch()
    assert discover_group_results(tmp_path) == sorted(expected)


def test_result_slug_encodes_full_analysis_identity(tmp_path: Path):
    output_dir = tmp_path
    source = (output_dir / "group_average" / "nemotron_layer18_lt_av" /
              "k100_delay5s_bin5s_skip5s_spearman" /
              "rsa_subcortical_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii")
    category, slug = result_category_and_slug(source, output_dir)
    assert category == "group_average"
    assert slug == ("nemotron_layer18_lt_av__k100_delay5s_bin5s_skip5s_spearman"
                     "__rsa_subcortical_raw_k100_delay5s_bin5s_skip5s_spearman_maps")
    filename = f"{slug}__BILATERAL_THALAMUS.func.gii"
    # Different analyses never collide on filename alone.
    other_source = source.with_name("integration_partial_r_searchlight.dscalar.nii")
    _, other_slug = result_category_and_slug(other_source, output_dir)
    assert other_slug != slug


def test_vertex_count_self_check_catches_mismatch(tmp_path: Path):
    path = tmp_path / "bad.func.gii"
    _save_multimetric(path, [("rho", np.zeros(10, dtype=np.float32))])
    _assert_vertex_count(path, 10)  # matches -- no raise
    with pytest.raises(AssertionError):
        _assert_vertex_count(path, 11)  # mismatched carrier -- must fail loud
