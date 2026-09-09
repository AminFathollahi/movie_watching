import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.channel_subset_searchlight import subset_columns


def test_subset_columns_selects_cluster_channels():
    full = np.arange(20).reshape(4, 5).astype(np.float32)  # (bins, channels)
    labels = np.array([0, 1, 1, 0, 1])
    subset, indices = subset_columns(full, labels, cluster_id=1)
    assert list(indices) == [1, 2, 4]
    assert subset.shape == (4, 3)
    np.testing.assert_array_equal(subset, full[:, [1, 2, 4]])


def test_subset_columns_rejects_length_mismatch():
    full = np.zeros((4, 5), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int32)
    try:
        subset_columns(full, labels, cluster_id=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on channel-count mismatch")


if __name__ == "__main__":
    test_subset_columns_selects_cluster_channels()
    test_subset_columns_rejects_length_mismatch()
    print("[demo] OK")
