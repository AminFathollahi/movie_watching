"""Tests for additive Workbench 2-D partial-correlation exports."""

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


def _partial_cifti(path: Path, roi_a: str, roi_p: str) -> None:
    names = [
        f"partial_r_{roi_a}_given_{roi_p}_bilateral",
        f"partial_r_{roi_p}_given_{roi_a}_bilateral",
        f"partial_r_{roi_a}_given_{roi_p}_within_hemisphere",
        f"partial_r_{roi_p}_given_{roi_a}_within_hemisphere",
        f"partial_r_{roi_a}_given_{roi_p}_L",
        f"partial_r_{roi_p}_given_{roi_a}_L",
        f"partial_r_{roi_a}_given_{roi_p}_R",
        f"partial_r_{roi_p}_given_{roi_a}_R",
    ]
    scalar = nib.cifti2.ScalarAxis(names)
    brain = nib.cifti2.BrainModelAxis.from_mask(
        np.ones(6, dtype=bool), name="CortexLeft"
    )
    data = np.linspace(-0.2, 0.5, 48, dtype=np.float32).reshape(8, 6)
    nib.save(
        nib.Cifti2Image(
            data,
            header=nib.cifti2.Cifti2Header.from_axes((scalar, brain)),
        ),
        path,
    )


def test_export_writes_all_four_views_with_shared_scale(tmp_path, monkeypatch):
    roi_a, roi_p = "cca_a_test", "cca_p_test"
    source = tmp_path / "partial.dscalar.nii"
    _partial_cifti(source, roi_a, roi_p)
    texture = np.ones((4, 4, 4), dtype=np.float32)
    texture[..., 3] = 1.0
    monkeypatch.setattr(MODULE, "find_pycortex_colormap", lambda _: tmp_path / "cmap.png")
    monkeypatch.setattr(MODULE, "load_rgba_texture", lambda _: texture)
    monkeypatch.setattr(MODULE, "save_legend", lambda *args, **kwargs: None)

    outputs = MODULE.export_partial_bivariate_views(
        source, tmp_path, roi_a, roi_p, bins=4, vmin=0.0, vmax=0.4
    )
    for view in ("bilateral", "within_hemisphere", "L", "R"):
        image = nib.load(outputs[view])
        assert image.shape == (1, 6)
        assert image.header.get_axis(0).name[0].endswith(f"({view}, 4x4)")
    metadata = outputs["metadata"].read_text()
    assert "signed values are preserved" in metadata


def test_rejects_wrong_map_order(tmp_path):
    roi_a, roi_p = "cca_a_test", "cca_p_test"
    source = tmp_path / "partial.dscalar.nii"
    _partial_cifti(source, roi_a, roi_p)
    image = nib.load(source)
    names = list(image.header.get_axis(0).name)
    names[0], names[1] = names[1], names[0]
    header = nib.cifti2.Cifti2Header.from_axes(
        (nib.cifti2.ScalarAxis(names), image.header.get_axis(1))
    )
    nib.save(nib.Cifti2Image(np.asarray(image.dataobj), header=header), source)
    with pytest.raises(ValueError, match="map names/order"):
        MODULE.export_partial_bivariate_views(source, tmp_path, roi_a, roi_p)
