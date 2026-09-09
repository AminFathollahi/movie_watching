import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cluster.channel_stability import select_resolution_by_heldout_reconstruction


def test_selection_picks_smallest_k_after_diminishing_returns():
    summary = pd.DataFrame({
        "n_clusters": [2, 4, 8, 16, 32],
        "heldout_reconstruction_mse": [1.0, 0.5, 0.46, 0.45, 0.44],
    })

    selection = select_resolution_by_heldout_reconstruction(summary, coverage=0.9)

    assert selection["selected_n_clusters"] == 8


def test_selection_falls_back_to_smallest_k_when_reconstruction_never_improves():
    summary = pd.DataFrame({
        "n_clusters": [2, 4, 8],
        "heldout_reconstruction_mse": [0.5, 0.5, 0.5],
    })

    selection = select_resolution_by_heldout_reconstruction(summary)

    assert selection["selected_n_clusters"] == 2
