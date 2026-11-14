import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.naming import (
    DEFAULT_MODEL_NORM, MODEL_NORMS, add_model_norm_arg, method_label, searchlight_config,
)
from shared.binning import preprocess_fmri, process_model_embeddings


@pytest.fixture
def two_run_inputs(tmp_path):
    rng = np.random.default_rng(0)
    timing = pd.DataFrame({
        "run": [1, 1, 2, 2],
        "onset_sec": [0.0, 20.0, 40.0, 60.0],
        "duration_sec": [20.0, 20.0, 20.0, 20.0],
    })
    run_trs = np.array([40, 40])
    embedding = rng.standard_normal((16, 3)) * np.array([1.0, 5.0, 0.2]) + np.array([3.0, -2.0, 10.0])
    path = tmp_path / "emb.npy"
    np.save(path, embedding.astype(np.float32))
    return path, timing, run_trs, embedding


def _embed(inputs, model_norm):
    path, timing, run_trs, _ = inputs
    return process_model_embeddings(str(path), timing, bin_sec=5.0, tr=1.0, run_trs=run_trs,
                                    model_norm=model_norm)


def test_zscore_gives_unit_variance_per_run_and_dimension(two_run_inputs):
    out = _embed(two_run_inputs, "zscore")
    for run in (out[:8], out[8:]):
        np.testing.assert_allclose(run.mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(run.std(axis=0), 1, atol=1e-5)


def test_center_removes_run_mean_and_keeps_scale(two_run_inputs):
    out = _embed(two_run_inputs, "center")
    raw = two_run_inputs[3].astype(np.float32)
    for sl in (slice(0, 8), slice(8, 16)):
        np.testing.assert_allclose(out[sl].mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(out[sl], raw[sl] - raw[sl].mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(out[sl].std(axis=0), raw[sl].std(axis=0), rtol=1e-5)


def test_unknown_model_norm_is_rejected(two_run_inputs):
    with pytest.raises(ValueError):
        _embed(two_run_inputs, "none")


def test_fmri_is_z_scored_regardless_of_model_norm():
    rng = np.random.default_rng(1)
    fmri = (rng.standard_normal((4, 40)) * 3 + 7).astype(np.float32)
    timing = pd.DataFrame({"onset_sec": [0.0, 20.0], "duration_sec": [20.0, 20.0]})
    out = preprocess_fmri(fmri, timing, np.array([40]), bin_sec=5.0, tr=1.0)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-5)


def test_cli_default_is_center_and_only_two_values_are_accepted():
    parser = argparse.ArgumentParser()
    add_model_norm_arg(parser)
    assert MODEL_NORMS == ("zscore", "center")
    assert parser.parse_args([]).model_norm == DEFAULT_MODEL_NORM == "center"
    assert parser.parse_args(["--model-norm", "zscore"]).model_norm == "zscore"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-norm", "demean"])


def test_crossnobis_method_label_has_no_model_norm():
    assert method_label("spearman", "center") == "pearson_spearman_center"
    assert method_label("rho_a", "center") == "rho_a"
