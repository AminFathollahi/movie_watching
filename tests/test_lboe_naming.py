"""Tests for requested-LBOE naming."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMING_SPEC = importlib.util.spec_from_file_location(
    "cf_naming", ROOT / "cf_modeling" / "cf_naming.py")
NAMING = importlib.util.module_from_spec(NAMING_SPEC)
sys.modules["cf_naming"] = NAMING
NAMING_SPEC.loader.exec_module(NAMING)

lboe_count = NAMING.lboe_count
qualify_lboe = NAMING.qualify_lboe
strip_lboe_suffix = NAMING.strip_lboe_suffix
persubject_output_root = NAMING.persubject_output_root
validate_roi_mean_preprocessing = NAMING.validate_roi_mean_preprocessing


def test_lboe_names_are_terminal_and_idempotent():
    assert qualify_lboe("cca_a_peav_1pct", 100) == "cca_a_peav_1pct_lboe100"
    assert qualify_lboe("cca_a_peav_1pct_lboe100", 100) == "cca_a_peav_1pct_lboe100"
    assert strip_lboe_suffix("cca_a_peav_1pct_lboe100") == "cca_a_peav_1pct"
    assert lboe_count("cca_a_peav_1pct_lboe100") == 100
    assert lboe_count("cca_a_peav_1pct") is None


def test_roi_mean_preprocessing_labels_are_explicit_and_validated():
    assert validate_roi_mean_preprocessing("sg_psc_per_run_zscore") == \
        "sg_psc_per_run_zscore"
    assert validate_roi_mean_preprocessing("raw_per_run_zscore") == \
        "raw_per_run_zscore"


def test_persubject_output_root_honours_override_and_default(monkeypatch, tmp_path: Path):
    override = tmp_path / "external"
    override.mkdir()
    monkeypatch.setenv("MOVIE_PERSUBJECT_ROOT", str(override))
    assert persubject_output_root() == override
    monkeypatch.delenv("MOVIE_PERSUBJECT_ROOT")

    # No override: only meaningful when the external drive is mounted.
    if NAMING.PERSUBJECT_MOUNT.is_dir():
        root = persubject_output_root()
        assert root == NAMING.PERSUBJECT_MOUNT / NAMING.PERSUBJECT_SUBPATH
        assert root.is_dir()


def test_persubject_output_root_raises_on_unreachable_override(monkeypatch, tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setenv("MOVIE_PERSUBJECT_ROOT", str(missing))
    try:
        persubject_output_root()
        assert False, "expected RuntimeError for a nonexistent override path"
    except RuntimeError as exc:
        assert str(missing) in str(exc)
