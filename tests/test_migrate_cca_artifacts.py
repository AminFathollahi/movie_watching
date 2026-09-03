import json
from pathlib import Path

import pytest

from cf_modeling.migrate_cca_artifacts import (
    apply_in_place,
    apply_migration,
    cca_name,
    plan_in_place,
    plan_migration,
)


def test_cca_name_uses_anatomical_mapping_without_touching_suffixes():
    assert cca_name("mca_v_peav_1pct") == "cca_a_peav_1pct"
    assert cca_name("mca_d_nemotron_layer18") == "cca_p_nemotron_layer18"
    assert cca_name("mca_islands_masks") == "cca_islands_masks"
    assert cca_name("MCA_V") == "CCA_A"


def test_plan_is_additive_and_dry_run_does_not_write(tmp_path: Path):
    source = tmp_path / "old"
    destination = tmp_path / "new"
    (source / "masks").mkdir(parents=True)
    (source / "masks" / "mca_v_peav.json").write_text(
        '{"roi": "mca_v", "other": "mca_d"}\n', encoding="utf-8"
    )

    operations = plan_migration(source, destination)
    assert operations[0].destination == destination / "masks" / "cca_a_peav.json"
    assert not destination.exists()


def test_apply_copies_and_rewrites_text_but_keeps_source(tmp_path: Path):
    source = tmp_path / "old"
    destination = tmp_path / "new"
    source.mkdir()
    old = source / "mca_v_mask.csv"
    old.write_text("name,mca_d\n", encoding="utf-8")

    operations = plan_migration(source, destination)
    apply_migration(operations, destination, source)

    assert old.read_text(encoding="utf-8") == "name,mca_d\n"
    assert (destination / "cca_a_mask.csv").read_text(encoding="utf-8") == "name,cca_p\n"
    assert (destination / "cca_migration_manifest.json").is_file()


def test_plan_rejects_destination_inside_source(tmp_path: Path):
    source = tmp_path / "old"
    source.mkdir()
    with pytest.raises(ValueError):
        plan_migration(source, source / "cca")


def test_in_place_rewrites_paths_and_text_and_removes_empty_dirs(tmp_path: Path):
    root = tmp_path / "outputs"
    old = root / "group_average" / "mca_v_x_mca_d_x" / "mca_v.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"first": "mca_v", "second": "mca_d"}\n', encoding="utf-8")

    apply_in_place(root, plan_in_place(root))

    new = root / "group_average" / "cca_a_x_cca_p_x" / "cca_a.json"
    assert new.read_text(encoding="utf-8") == '{"first": "cca_a", "second": "cca_p"}\n'
    assert not old.exists()
    assert not old.parent.exists()
    manifest = root / "cca_inplace_migration_manifest.json"
    assert manifest.is_file()
    assert json.loads(manifest.read_text())["summary"]["renamed"] == 1


def test_in_place_deduplicates_equal_canonical_text(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()
    legacy = root / "mca_v.txt"
    canonical = root / "cca_a.txt"
    legacy.write_text("mca_d\n", encoding="utf-8")
    canonical.write_text("cca_p\n", encoding="utf-8")

    records = apply_in_place(root)

    assert not legacy.exists()
    assert canonical.read_text(encoding="utf-8") == "cca_p\n"
    assert any(record.status == "deduplicated" for record in records)


def test_in_place_rewrites_cifti_scalar_and_label_names(tmp_path: Path):
    import nibabel as nib
    import numpy as np

    root = tmp_path / "outputs"
    root.mkdir()
    brain = nib.cifti2.BrainModelAxis.from_mask(
        np.array([True, True]), name="CIFTI_STRUCTURE_CORTEX_LEFT")

    scalar = nib.Cifti2Image(
        np.array([[1.0, 2.0]], dtype=np.float32),
        header=nib.cifti2.Cifti2Header.from_axes((
            nib.cifti2.ScalarAxis(["mca_v_given_mca_d"]), brain)))
    nib.save(scalar, root / "mca_v.dscalar.nii")
    canonical_path = root / "already_cca.dscalar.nii"
    nib.save(scalar, canonical_path)

    label = nib.Cifti2Image(
        np.array([[1, 2]], dtype=np.float32),
        header=nib.cifti2.Cifti2Header.from_axes((
            nib.cifti2.LabelAxis(
                ["mca_islands"],
                [{0: ("background", (0, 0, 0, 0)),
                  1: ("mca_v", (1, 0, 0, 1)),
                  2: ("mca_d", (0, 0, 1, 1))}]),
            brain)))
    nib.save(label, root / "mca_islands.dlabel.nii")

    apply_in_place(root)

    scalar_new = nib.load(root / "cca_a.dscalar.nii")
    assert list(scalar_new.header.get_axis(0).name) == ["cca_a_given_cca_p"]
    np.testing.assert_array_equal(np.asanyarray(scalar_new.dataobj), [[1.0, 2.0]])
    canonical_new = nib.load(canonical_path)
    assert list(canonical_new.header.get_axis(0).name) == ["cca_a_given_cca_p"]
    label_new = nib.load(root / "cca_islands.dlabel.nii")
    label_axis = label_new.header.get_axis(0)
    assert list(label_axis.name) == ["cca_islands"]
    assert label_axis.label[0][1][0] == "cca_a"
    assert label_axis.label[0][2][0] == "cca_p"
