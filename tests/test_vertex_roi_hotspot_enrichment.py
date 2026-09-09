import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.vertex_roi_hotspot_enrichment import enrichment_table


def test_enrichment_table_detects_planted_overlap():
    labels = np.array([0, 0] + [1] * 10 + [2] * 10)
    category = np.arange(2, 12)  # exactly cluster 1's vertices
    population_mask = labels != 0
    table = enrichment_table(labels, [1, 2], population_mask, {"cat": category})

    row1 = table[table["cluster_id"] == 1].iloc[0]
    row2 = table[table["cluster_id"] == 2].iloc[0]
    assert row1["overlap"] == 10
    assert row1["p"] < 1e-4
    assert row2["overlap"] == 0
    assert row2["p"] == 1.0
    assert "q_bh" in table.columns


if __name__ == "__main__":
    test_enrichment_table_detects_planted_overlap()
    print("[demo] OK")
