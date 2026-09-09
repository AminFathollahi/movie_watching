"""Tests for requested-LBOE naming and collision-safe artifact migration."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMING_SPEC = importlib.util.spec_from_file_location(
    "cf_naming", ROOT / "cf_modeling" / "cf_naming.py")
NAMING = importlib.util.module_from_spec(NAMING_SPEC)
sys.modules["cf_naming"] = NAMING
NAMING_SPEC.loader.exec_module(NAMING)
MIGRATION_SPEC = importlib.util.spec_from_file_location(
    "migrate_lboe_artifacts", ROOT / "cf_modeling" / "migrate_lboe_artifacts.py")
MIGRATION = importlib.util.module_from_spec(MIGRATION_SPEC)
sys.modules["migrate_lboe_artifacts"] = MIGRATION
MIGRATION_SPEC.loader.exec_module(MIGRATION)

lboe_count = NAMING.lboe_count
qualify_lboe = NAMING.qualify_lboe
strip_lboe_suffix = NAMING.strip_lboe_suffix
persubject_output_root = NAMING.persubject_output_root
apply_plan = MIGRATION.apply_plan
plan_migration = MIGRATION.plan_migration


def test_lboe_names_are_terminal_and_idempotent():
    assert qualify_lboe("cca_a_peav_1pct", 100) == "cca_a_peav_1pct_lboe100"
    assert qualify_lboe("cca_a_peav_1pct_lboe100", 100) == "cca_a_peav_1pct_lboe100"
    assert strip_lboe_suffix("cca_a_peav_1pct_lboe100") == "cca_a_peav_1pct"
    assert lboe_count("cca_a_peav_1pct_lboe100") == 100
    assert lboe_count("cca_a_peav_1pct") is None


def test_persubject_output_root_honours_override_and_default(monkeypatch, tmp_path: Path):
    override = tmp_path / "external"
    override.mkdir()
    monkeypatch.setenv("MOVIE_PERSUBJECT_ROOT", str(override))
    assert persubject_output_root() == override
    monkeypatch.delenv("MOVIE_PERSUBJECT_ROOT")

    # Real drive, no override: only meaningful when it's actually mounted.
    if NAMING.PERSUBJECT_MOUNT.is_mount():
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


def test_migration_qualifies_pair_files_but_preserves_source_mask_paths(tmp_path: Path):
    root = tmp_path / "group_average" / "cca_a_test_cca_p_test"
    prep = root / "prep"
    prep.mkdir(parents=True)
    (prep / "R2_cca_a_test_nc.npy").write_bytes(b"array-a")
    (prep / "R2_cca_p_test_nc.npy").write_bytes(b"array-p")
    metadata = root / "result_cca_a_test_cca_p_test.json"
    metadata.write_text(
        '{"roi_a":"cca_a_test","output":"' + str(root)
        + '","mask":"/some/masks/cca_a_test_mask.dscalar.nii"}\n',
        encoding="utf-8")

    plans = plan_migration(tmp_path, default_lboe=200)
    assert len(plans) == 1
    record = apply_plan(plans[0])
    destination = Path(plans[0].destination)
    assert record["status"] == "migrated"
    assert (destination / "prep" / "R2_cca_a_test_lboe200_nc.npy").is_file()
    rewritten = (destination /
                 "result_cca_a_test_lboe200_cca_p_test_lboe200.json").read_text()
    assert '"roi_a":"cca_a_test_lboe200"' in rewritten
    assert "/masks/cca_a_test_mask.dscalar.nii" in rewritten
