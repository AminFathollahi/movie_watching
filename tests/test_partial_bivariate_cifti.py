"""Tests for the Workbench 2-D export of the ROI-mean partial correlations."""

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cf_modeling"))
MODULE_PATH = ROOT / "cf_modeling" / "export_partial_bivariate_cifti.py"
SPEC = importlib.util.spec_from_file_location("export_partial_bivariate_cifti", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _connectivity_cifti(path: Path, roi_a: str, roi_b: str, label: str) -> None:
    _, names = MODULE.roi_mean_connectivity_names(roi_a, roi_b, label)
    brain = nib.cifti2.BrainModelAxis.from_mask(np.ones(6, dtype=bool), name="CortexLeft")
    data = np.linspace(-0.2, 0.5, 12, dtype=np.float32).reshape(2, 6)
    data[0, 0] = np.nan
    nib.save(nib.Cifti2Image(data, header=nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(names), brain))), path)


def test_export_writes_one_dlabel_with_nan_unlabelled(tmp_path, monkeypatch):
    source = tmp_path / "connectivity.dscalar.nii"
    _connectivity_cifti(source, "A5", "FFC", "raw_per_run_zscore")
    texture = np.ones((4, 4, 4), dtype=np.float32)
    monkeypatch.setattr(MODULE, "find_pycortex_colormap", lambda _: tmp_path / "cmap.png")
    monkeypatch.setattr(MODULE, "load_rgba_texture", lambda _: texture)
    monkeypatch.setattr(MODULE, "save_legend", lambda *args, **kwargs: None)
    outputs = MODULE.export_connectivity_bivariate(source, tmp_path, "A5", "FFC", "raw_per_run_zscore", bins=4)
    assert outputs["dlabel"].name == "bivariate_roi_mean_timeseries_connectivity_A5_FFC_raw_per_run_zscore_4bin.dlabel.nii"
    image = nib.load(outputs["dlabel"])
    assert image.shape == (1, 6)
    assert np.asarray(image.dataobj)[0, 0] == 0
    assert outputs["metadata"].parent == tmp_path.parent
    assert "signed values are preserved" in outputs["metadata"].read_text()


def test_rejects_wrong_map_order(tmp_path):
    source = tmp_path / "connectivity.dscalar.nii"
    _connectivity_cifti(source, "A5", "FFC", "raw_per_run_zscore")
    image = nib.load(source)
    names = list(image.header.get_axis(0).name)[::-1]
    header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(names), image.header.get_axis(1)))
    nib.save(nib.Cifti2Image(np.asarray(image.dataobj), header=header), source)
    with pytest.raises(ValueError, match="map names/order"):
        MODULE.export_connectivity_bivariate(source, tmp_path, "A5", "FFC", "raw_per_run_zscore")
