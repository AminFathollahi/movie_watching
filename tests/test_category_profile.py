import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.category_profile import (
    bin_label_windows,
    category_block_metrics,
    decomposition_metrics,
    dominant_visual_labels,
    fit_category_decomposition,
)


def _fit_args():
    return SimpleNamespace(n_iter=5, backend="numpy", model_random_state=0)


def _synthetic_bands(rng, n_per_run=40, n_runs=4):
    n = n_per_run * n_runs
    audio = rng.uniform(size=(n, 3))
    visual = rng.uniform(size=(n, 5))
    conjunction = (audio[:, :, None] * visual[:, None, :]).reshape(n, 15)
    run_ids = np.repeat(np.arange(n_runs), n_per_run)
    return audio, visual, conjunction, run_ids


def test_additive_signal_yields_near_zero_conjunction_gain():
    rng = np.random.default_rng(0)
    audio, visual, conjunction, run_ids = _synthetic_bands(rng)
    w_a = rng.normal(scale=1.0, size=(3, 2))
    w_v = rng.normal(scale=1.0, size=(5, 2))
    targets = audio @ w_a + visual @ w_v + rng.normal(scale=0.05, size=(audio.shape[0], 2))

    oof, _ = fit_category_decomposition(
        audio, visual, conjunction, targets, run_ids,
        np.logspace(-2, 4, 13), _fit_args(),
    )
    block_frame = category_block_metrics(oof, targets, {"all": np.arange(2)}, run_ids)

    assert block_frame["conjunction_gain"].mean() < 0.05
    assert block_frame["unique_audio"].mean() > 0.1
    assert block_frame["unique_visual"].mean() > 0.1


def test_conjunctive_signal_yields_clearly_positive_conjunction_gain():
    rng = np.random.default_rng(1)
    audio, visual, conjunction, run_ids = _synthetic_bands(rng)
    w_c = rng.normal(scale=1.0, size=(15, 2))
    targets = conjunction @ w_c + rng.normal(scale=0.05, size=(audio.shape[0], 2))

    oof, _ = fit_category_decomposition(
        audio, visual, conjunction, targets, run_ids,
        np.logspace(-2, 4, 13), _fit_args(),
    )
    block_frame = category_block_metrics(oof, targets, {"all": np.arange(2)}, run_ids)

    # Raw (uncentered) elementwise products correlate somewhat with their own
    # factors, so the additive model absorbs part of a purely conjunctive
    # signal too -- 0.08 is well above the ~0 seen in the additive case above
    # while staying below what this specific construction can actually reach.
    assert block_frame["conjunction_gain"].mean() > 0.08


def test_decomposition_metrics_arithmetic():
    metrics = decomposition_metrics(r2_audio=0.3, r2_visual=0.1, r2_additive=0.35, r2_full=0.5)

    assert np.isclose(metrics["unique_audio"], 0.25)
    assert np.isclose(metrics["unique_visual"], 0.05)
    assert np.isclose(metrics["shared"], 0.05)
    assert np.isclose(metrics["conjunction_gain"], 0.15)
    assert np.isclose(metrics["audio_categorical_dominance"], 0.2)


def test_dominant_visual_labels_rejects_ties_and_all_zero_rows():
    visual = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0, 0.0],
    ])
    labels, valid = dominant_visual_labels(visual)

    assert list(valid) == [True, False, False, True]
    assert labels[0] == 0
    assert labels[3] == 1


def test_bin_label_windows_matches_manual_window_average():
    run_trs = np.array([10, 10])
    raw = np.arange(20, dtype=np.float64).reshape(20, 1)
    metadata = pd.DataFrame({
        "row_index": [0, 1],
        "run_id": [1, 2],
        "window_onset_sec": [2.0, 13.0],  # run 2 onset is global: 10 (run 1 length) + 3
    })

    binned = bin_label_windows(raw, run_trs, metadata, bin_sec=3.0, tr=1.0)

    assert np.isclose(binned[0, 0], raw[2:5].mean())
    assert np.isclose(binned[1, 0], raw[13:16].mean())


if __name__ == "__main__":
    test_additive_signal_yields_near_zero_conjunction_gain()
    test_conjunctive_signal_yields_clearly_positive_conjunction_gain()
    test_decomposition_metrics_arithmetic()
    test_dominant_visual_labels_rejects_ties_and_all_zero_rows()
    test_bin_label_windows_matches_manual_window_average()
    print("demo OK")
