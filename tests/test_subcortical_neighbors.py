import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subcortical.subcortical_io import neighbor_columns


def _slices(sizes):
    out, start = {}, 0
    for name, side in zip(("THALAMUS_LEFT", "PUTAMEN_LEFT"), sizes):
        ijk = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)
        out[name] = {"start": start, "stop": start + len(ijk), "voxel_ijk": ijk, "world_xyz": ijk * 1.6}
        start += len(ijk)
    return out


def test_neighbor_columns_stay_within_each_structure(tmp_path):
    slices = _slices((3, 4))
    cols = neighbor_columns(slices, 5, tmp_path, "unused")
    assert cols.shape == (27 + 64, 5)
    assert np.array_equal(cols[:, 0], np.arange(91))
    assert cols[:27].max() < 27 and cols[27:].min() >= 27


def test_neighbor_columns_require_contiguous_structures(tmp_path):
    slices = _slices((3, 3))
    slices["PUTAMEN_LEFT"]["start"] += 1
    with pytest.raises(AssertionError):
        neighbor_columns(slices, 5, tmp_path, "unused")
