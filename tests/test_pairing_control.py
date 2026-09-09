import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.pairing_control import effect_inference


def test_script_runs_directly_without_module_shadowing():
    # Regression: `python encoding/pairing_control.py` puts encoding/ at
    # sys.path[0], which used to shadow the `encoding` package with the
    # sibling encoding/encoding.py module and raise ModuleNotFoundError.
    result = subprocess.run(
        [sys.executable, str(ROOT / "encoding" / "pairing_control.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_effect_inference_detects_consistent_positive_effect():
    result = effect_inference(
        np.linspace(0.1, 0.2, 14), n_bootstrap=1000,
        n_permutations=1000, random_state=3,
    )

    assert result["mean_pairing_advantage"] > 0
    assert result["bootstrap_ci_low"] > 0
    assert result["sign_flip_p_greater"] < 0.01


def test_effect_inference_requires_multiple_finite_clips():
    for effects in (np.array([1.0]), np.array([0.0, np.nan])):
        try:
            effect_inference(effects, 10, 10, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid clip effects to be rejected")
