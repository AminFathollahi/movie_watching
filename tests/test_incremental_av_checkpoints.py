import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.incremental_av import (
    _compact_fold_outputs,
    _load_npz,
    _records_for_configuration,
    _savez_atomic,
)


def test_atomic_npz_save_replaces_existing_file_and_loads_required_arrays(tmp_path):
    path = tmp_path / "metrics.npz"
    _savez_atomic(path, stale=np.array([0]))
    _savez_atomic(path, test_indices=np.array([2, 5]), delta_r2=np.array([0.2]))

    loaded = _load_npz(path, ("test_indices", "delta_r2"))

    assert not (tmp_path / ".metrics.npz.tmp").exists()
    np.testing.assert_array_equal(loaded["test_indices"], [2, 5])
    np.testing.assert_array_equal(loaded["delta_r2"], [0.2])
    with np.load(path) as archive:
        assert archive.files == ["test_indices", "delta_r2"]


def test_load_npz_rejects_corrupt_and_incomplete_caches(tmp_path):
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz archive")
    incomplete = tmp_path / "incomplete.npz"
    _savez_atomic(incomplete, test_indices=np.array([1]))

    assert _load_npz(corrupt, ("test_indices", "delta_r2")) is None
    assert _load_npz(incomplete, ("test_indices", "delta_r2")) is None


def test_compact_fold_outputs_retains_only_declared_metric_maps(tmp_path):
    fold = tmp_path / "full" / "run3"
    fold.mkdir(parents=True)
    _savez_atomic(
        fold / "metrics.npz",
        test_indices=np.array([1, 4]), baseline_r2=np.array([0.1]),
        extended_r2=np.array([0.3]), delta_r2=np.array([0.2]),
        baseline_prediction=np.ones((2, 1)), y_true=np.ones((2, 1)),
    )
    _savez_atomic(
        fold / "efficiency_metrics.npz",
        test_indices=np.array([1, 4]), additive_r2=np.array([0.1]),
        joint_r2=np.array([0.3]), joint_minus_additive_r2=np.array([0.2]),
        joint_prediction=np.ones((2, 1)),
    )

    _compact_fold_outputs(tmp_path / "full")

    with np.load(fold / "metrics.npz") as archive:
        assert set(archive.files) == {
            "test_indices", "baseline_r2", "extended_r2", "delta_r2",
        }
    with np.load(fold / "efficiency_metrics.npz") as archive:
        assert set(archive.files) == {
            "test_indices", "additive_r2", "joint_r2", "joint_minus_additive_r2",
        }


def test_records_for_configuration_recovers_only_complete_configuration_rows():
    frame = pd.DataFrame([
        {"configuration": "pca_d4", "run": 1, "value": 0.1},
        {"configuration": "pca_d4", "run": 2, "value": 0.2},
        {"configuration": "full", "run": 1, "value": 0.3},
    ])

    recovered = _records_for_configuration(frame, "pca_d4", expected_rows=2)

    assert recovered == [
        {"configuration": "pca_d4", "run": 1, "value": 0.1},
        {"configuration": "pca_d4", "run": 2, "value": 0.2},
    ]
    assert _records_for_configuration(frame, "pca_d4", expected_rows=3) == []
    assert _records_for_configuration(None, "pca_d4", expected_rows=2) == []
