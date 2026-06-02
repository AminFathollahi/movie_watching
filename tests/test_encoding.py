"""
tests/test_encoding.py
======================
Unit tests for encoding/shared/encoding_utils.py.

build_fmri_arrays mocks nibabel so no real CIFTI is needed.
Run with: conda run -n vicsompy_av pytest tests/test_encoding.py -v
"""
import sys
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "encoding", "shared"))

import encoding_utils
from encoding_utils import generate_leave_one_run_out


# =============================================================================
# Helpers
# =============================================================================

def _make_timing_df(n_videos=4, run_size=2, duration_sec=10.0):
    """Build a minimal timing DataFrame with string video_ids.

    String video_ids avoid the pandas iterrows() float-upcast issue:
    iterrows() upcasts numeric columns to float64 when the row contains a float
    (duration_sec), turning int video_id 3 into "3.0" via str(). Using string
    IDs from the start matches real timing CSVs and keeps str() a no-op.

    onset_sec is cumulative (global), matching the real movie_timing.csv format.
    """
    rows = []
    cumulative_onset = 0.0
    for i in range(n_videos):
        rows.append({
            "video_id":     str(i + 1),       # "1", "2", "3", "4" — already strings
            "run_id":       (i // run_size) + 1,
            "onset_sec":    cumulative_onset,
            "duration_sec": duration_sec,
        })
        cumulative_onset += duration_sec
    return pd.DataFrame(rows)


def _make_run_trs(timing_df: pd.DataFrame, tr: float = 1.0) -> np.ndarray:
    """Compute (n_runs,) int array of TRs per run from timing_df."""
    run_col = "run_id" if "run_id" in timing_df.columns else "run"
    trs = (timing_df.groupby(run_col, sort=True)["duration_sec"]
                    .sum() / tr).astype(int).values
    return trs.astype(np.int32)


@contextmanager
def _patch_fmri(cifti_data, run_trs):
    """Patch nib.load (returns fake CIFTI) and np.load (returns run_trs)."""
    mock_img = MagicMock()
    mock_img.get_fdata.return_value = cifti_data
    mock_nib = MagicMock()
    mock_nib.load.return_value = mock_img
    with patch.object(encoding_utils, "nib", mock_nib), \
         patch("numpy.load", return_value=run_trs):
        yield


# =============================================================================
# build_fmri_arrays
# =============================================================================

def test_build_fmri_arrays_shapes():
    rng = np.random.default_rng(20)
    timing_df = _make_timing_df()   # 4 videos, 10 s each, 2 per run
    tr, bin_sec, n_verts = 1.0, 2.0, 30
    T = 40   # 4 × 10 TRs
    test_ids = ["3"]   # video 3 is test; 1, 2, 4 are train
    run_trs = _make_run_trs(timing_df, tr)  # [20, 20]

    data = rng.standard_normal((T, n_verts)).astype(np.float32)
    with _patch_fmri(data, run_trs):
        Y_train, Y_test, run_onsets = encoding_utils.build_fmri_arrays(
            "fake.nii", "fake_run_trs.npy", timing_df, test_ids, bin_sec, tr
        )

    # 5 bins × 3 train videos = 15; 5 bins × 1 test video = 5
    assert Y_train.shape == (15, n_verts)
    assert Y_test.shape  == (5,  n_verts)
    assert Y_train.dtype == np.float32
    assert Y_test.dtype  == np.float32


def test_build_fmri_arrays_run_onsets():
    """run_onsets must have one entry per training run; first entry is 0."""
    rng = np.random.default_rng(21)
    timing_df = _make_timing_df()   # 2 runs, video 3 is test
    run_trs = _make_run_trs(timing_df)  # [20, 20]
    data = rng.standard_normal((40, 10)).astype(np.float32)
    with _patch_fmri(data, run_trs):
        _, _, run_onsets = encoding_utils.build_fmri_arrays(
            "fake.nii", "fake_run_trs.npy", timing_df, ["3"], bin_sec=2.0, tr=1.0
        )
    # 2 training runs → 2 onsets; first always 0
    assert len(run_onsets) == 2
    assert run_onsets[0] == 0


def test_build_fmri_arrays_total_bins():
    """Y_train + Y_test row count must equal total included bins."""
    rng = np.random.default_rng(22)
    timing_df = _make_timing_df()
    T, n_verts, tr, bin_sec = 40, 5, 1.0, 2.0
    run_trs = _make_run_trs(timing_df, tr)  # [20, 20]
    data = rng.standard_normal((T, n_verts)).astype(np.float32)
    with _patch_fmri(data, run_trs):
        Y_train, Y_test, _ = encoding_utils.build_fmri_arrays(
            "fake.nii", "fake_run_trs.npy", timing_df, ["3"], bin_sec, tr
        )
    # 4 videos × 5 bins = 20 total bins
    assert Y_train.shape[0] + Y_test.shape[0] == 20


def test_build_fmri_arrays_all_test():
    """If all videos are test, Y_train should have 0 rows, Y_test covers all bins."""
    rng = np.random.default_rng(23)
    timing_df = _make_timing_df(n_videos=2, run_size=1, duration_sec=10.0)
    run_trs = _make_run_trs(timing_df)  # [10, 10]
    data = rng.standard_normal((20, 8)).astype(np.float32)
    test_ids = ["1", "2"]
    with _patch_fmri(data, run_trs):
        Y_train, Y_test, run_onsets = encoding_utils.build_fmri_arrays(
            "fake.nii", "fake_run_trs.npy", timing_df, test_ids, bin_sec=2.0, tr=1.0
        )
    assert Y_train.shape[0] == 0
    assert Y_test.shape[0]  == 10   # 2 videos × 5 bins
    assert run_onsets == []


# =============================================================================
# generate_leave_one_run_out (encoding version — deterministic, sequential)
# =============================================================================

def test_encoding_loro_yields_n_runs():
    splits = list(generate_leave_one_run_out(n_samples=300,
                                              run_onsets=[0, 100, 200]))
    assert len(splits) == 3


def test_encoding_loro_partition():
    """Train + val must cover all samples without overlap."""
    n_samples, run_onsets = 400, [0, 100, 200, 300]
    for train, val in generate_leave_one_run_out(n_samples, run_onsets):
        assert len(train) + len(val) == n_samples
        assert len(np.intersect1d(train, val)) == 0


def test_encoding_loro_sequential_folds():
    """Fold i validates exactly the samples in run i (deterministic)."""
    run_onsets = [0, 10, 20]
    n_samples = 30
    expected_val_sets = [
        set(range(0,  10)),
        set(range(10, 20)),
        set(range(20, 30)),
    ]
    for (_, val), expected in zip(
        generate_leave_one_run_out(n_samples, run_onsets), expected_val_sets
    ):
        assert set(val.tolist()) == expected


def test_encoding_loro_covers_all_samples():
    """Union of all val sets must equal the full training index set."""
    n_samples, run_onsets = 300, [0, 100, 200]
    all_val = set()
    for _, val in generate_leave_one_run_out(n_samples, run_onsets):
        all_val |= set(val.tolist())
    assert all_val == set(range(n_samples))
