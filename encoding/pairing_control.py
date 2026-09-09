from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = str(Path(__file__).resolve().parent)
ROOT = str(Path(__file__).resolve().parents[1])
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, ROOT)

from encoding.incremental_av import _bh_qvalues


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--intact-output", type=Path, required=True)
    parser.add_argument("--mismatch-output", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args(argv)


def effect_inference(
    effects: np.ndarray, n_bootstrap: int, n_permutations: int, random_state: int,
) -> dict:
    effects = np.asarray(effects, dtype=float)
    if effects.ndim != 1 or effects.size < 2 or not np.isfinite(effects).all():
        raise ValueError("effects must contain at least two finite clip values")
    rng = np.random.default_rng(random_state)
    bootstrap = effects[
        rng.integers(0, len(effects), size=(n_bootstrap, len(effects)))
    ].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(effects)))
    null = (signs * effects).mean(axis=1)
    observed = float(effects.mean())
    return {
        "n_clips": int(len(effects)),
        "mean_pairing_advantage": observed,
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p_greater": float(
            (1 + np.sum(null >= observed)) / (n_permutations + 1)
        ),
    }


def _load(path: Path) -> pd.DataFrame:
    table = path / "full" / "roi_clip_metrics.csv"
    if not table.exists():
        raise FileNotFoundError(table)
    frame = pd.read_csv(table)
    required = {"roi", "clip_id", "mse_reduction"}
    if not required.issubset(frame):
        raise ValueError(f"{table} is missing {sorted(required - set(frame))}")
    return frame


def run(args) -> Path:
    intact = _load(args.intact_output)[["roi", "clip_id", "mse_reduction"]].rename(
        columns={"mse_reduction": "intact_mse_reduction"}
    )
    mismatch_frames = []
    for seed_index, path in enumerate(args.mismatch_output):
        frame = _load(path)[["roi", "clip_id", "mse_reduction"]].copy()
        frame["mismatch_index"] = seed_index
        frame["mismatch_output"] = str(path.resolve())
        mismatch_frames.append(frame)
    mismatch = pd.concat(mismatch_frames, ignore_index=True)
    paired = mismatch.merge(intact, on=["roi", "clip_id"], validate="many_to_one")
    paired["pairing_advantage"] = (
        paired["intact_mse_reduction"] - paired["mse_reduction"]
    )

    seed_summary = paired.groupby(
        ["mismatch_index", "mismatch_output", "roi"], as_index=False
    ).agg(
        n_clips=("clip_id", "nunique"),
        mean_pairing_advantage=("pairing_advantage", "mean"),
    )
    clip_effects = paired.groupby(["roi", "clip_id"], as_index=False).agg(
        intact_mse_reduction=("intact_mse_reduction", "first"),
        mean_mismatch_mse_reduction=("mse_reduction", "mean"),
        pairing_advantage=("pairing_advantage", "mean"),
        n_mismatches=("mismatch_index", "nunique"),
    )
    inference_rows = []
    for roi, frame in clip_effects.groupby("roi", sort=False):
        result = effect_inference(
            frame["pairing_advantage"].to_numpy(), args.n_bootstrap,
            args.n_permutations, args.random_state,
        )
        result["roi"] = roi
        result["n_mismatches"] = len(args.mismatch_output)
        inference_rows.append(result)
    inference = pd.DataFrame(inference_rows)
    inference["sign_flip_q_bh"] = _bh_qvalues(inference["sign_flip_p_greater"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_seed_clip_effects.csv", index=False)
    clip_effects.to_csv(args.output_dir / "paired_clip_effects.csv", index=False)
    seed_summary.to_csv(args.output_dir / "seed_summary.csv", index=False)
    inference.to_csv(args.output_dir / "inference.csv", index=False)
    manifest = {
        "intact_output": str(args.intact_output.resolve()),
        "mismatch_outputs": [str(path.resolve()) for path in args.mismatch_output],
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "random_state": args.random_state,
        "effect": "intact incremental MSE reduction minus mismatched incremental MSE reduction",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return args.output_dir


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
