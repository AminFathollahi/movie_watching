"""Tests for Workbench bivariate raw-correlation exports."""

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cf_modeling"))
PATH = ROOT / "cf_modeling" / "export_raw_corr_bivariate_cifti.py"
SPEC = importlib.util.spec_from_file_location("export_raw_corr_bivariate_cifti", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exports_four_raw_views(tmp_path, monkeypatch):
    roi_a, roi_p = "cca_a_test", "cca_p_test"
    scalar = nib.cifti2.ScalarAxis(MODULE.expected_names(roi_a, roi_p))
    brain = nib.cifti2.BrainModelAxis.from_mask(np.ones(5, bool), name="CortexLeft")
    source = tmp_path / "raw.dscalar.nii"
    nib.save(nib.Cifti2Image(
        np.linspace(-0.2, 0.5, 40, dtype=np.float32).reshape(8, 5),
        header=nib.cifti2.Cifti2Header.from_axes((scalar, brain))), source)
    texture = np.ones((4, 4, 4), dtype=np.float32)
    monkeypatch.setattr(MODULE, "find_pycortex_colormap", lambda _: tmp_path / "cmap.png")
    monkeypatch.setattr(MODULE, "load_rgba_texture", lambda _: texture)
    monkeypatch.setattr(MODULE, "save_legend", lambda *args, **kwargs: None)
    outputs = MODULE.export_raw_bivariate_views(
        source, tmp_path, roi_a, roi_p, bins=4)
    assert set(("bilateral", "within_hemisphere", "L", "R")) <= outputs.keys()
    for view in ("bilateral", "within_hemisphere", "L", "R"):
        assert nib.load(outputs[view]).shape == (1, 5)
    assert outputs["metadata"].parent == tmp_path.parent
    assert "remain signed" in outputs["metadata"].read_text()


def test_preprocessing_label_is_part_of_new_map_names():
    names = MODULE.expected_names(
        "cca_a_test", "cca_p_test", "raw_per_run_zscore")
    assert all(name.endswith("_raw_per_run_zscore") for name in names)
