"""Tests for the additive Workbench bivariate-map exporter."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "cf_modeling" / "export_bivariate_cifti.py"
)
SPEC = importlib.util.spec_from_file_location("export_bivariate_cifti", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _coordinate_texture(size=8):
    texture = np.zeros((size, size, 4), dtype=np.float32)
    texture[..., 0] = np.arange(size)[None, :] / (size - 1)
    texture[..., 1] = np.arange(size)[:, None] / (size - 1)
    texture[..., 3] = 1.0
    return texture


def test_quantization_keys_track_both_axes():
    dim1 = np.array([0.0, 0.4, 0.0, 0.4])
    dim2 = np.array([0.0, 0.0, 0.4, 0.4])
    keys, labels = MODULE.quantize_bivariate(
        dim1, dim2, _coordinate_texture(), bins=2, vmin=0.0, vmax=0.4
    )
    np.testing.assert_array_equal(keys, [1, 2, 3, 4])
    assert len(labels) == 5  # four bivariate cells plus INVALID


def test_quantization_clips_like_pycortex():
    keys_outside, _ = MODULE.quantize_bivariate(
        np.array([-10.0, 10.0]),
        np.array([-10.0, 10.0]),
        _coordinate_texture(),
        bins=4,
        vmin=0.0,
        vmax=0.4,
    )
    keys_edges, _ = MODULE.quantize_bivariate(
        np.array([0.0, 0.4]),
        np.array([0.0, 0.4]),
        _coordinate_texture(),
        bins=4,
        vmin=0.0,
        vmax=0.4,
    )
    np.testing.assert_array_equal(keys_outside, keys_edges)


def test_quantization_preserves_invalid_as_unlabeled():
    keys, labels = MODULE.quantize_bivariate(
        np.array([0.1, np.nan]),
        np.array([0.2, 0.2]),
        _coordinate_texture(),
        bins=4,
    )
    assert keys[0] != 0
    assert keys[1] == 0
    assert labels[0][1][3] == 0.0


def test_quantization_rejects_bad_parameters():
    texture = _coordinate_texture()
    with pytest.raises(ValueError, match="Axis shapes differ"):
        MODULE.quantize_bivariate(np.zeros(2), np.zeros(3), texture)
    with pytest.raises(ValueError, match="bins"):
        MODULE.quantize_bivariate(np.zeros(2), np.zeros(2), texture, bins=1)
