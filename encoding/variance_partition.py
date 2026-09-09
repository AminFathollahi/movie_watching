"""Compare held-out A+V and A+V+J encoding on fixed validation clips.

This compatibility analysis uses the historical clip split. The primary
generalization analysis is encoding/incremental_av.py, which holds out runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from encoding.shared.encoding_utils import build_fmri_arrays, save_cifti, split_embedding_array
from encoding.shared.fold_evaluator import (
    _r2_per_target,
    _to_numpy,
    fit_group_ridge,
    standardize_bands,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--fmri-suffix", default="raw")
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--template-cifti", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject", default="group_average")
    parser.add_argument("--model", required=True)
    parser.add_argument("--nuisance-model")
    parser.add_argument("--nuisance-modalities", default="a,v")
    parser.add_argument("--bin-sec", type=float, required=True)
    parser.add_argument("--skip-sec", type=float)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--alpha-min", type=float, required=True)
    parser.add_argument("--alpha-max", type=float, required=True)
    parser.add_argument("--n-alphas", type=int, required=True)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--test-video-ids", required=True)
    return parser.parse_args(argv)


def _config_label(args) -> str:
    mode = "norm" if args.normalize else "demean"
    return (
        f"delay{args.delay_sec:.0f}s_{mode}_bin{args.bin_sec:.0f}s_"
        f"skip{args.skip_sec:.0f}s"
    )


def _training_run_ids(
    timing: pd.DataFrame,
    test_ids: set[str],
    bin_sec: float,
    skip_sec: float,
) -> np.ndarray:
    groups = []
    run_column = "run_id" if "run_id" in timing.columns else "run"
    for run, rows in timing.groupby(run_column, sort=True):
        for _, row in rows.iterrows():
            if str(row["video_id"]) in test_ids:
                continue
            duration = float(row["duration_sec"])
            n_windows = (
                max(0, int(np.floor((duration - bin_sec) / skip_sec)) + 1)
                if duration >= bin_sec else 0
            )
            groups.extend([run] * n_windows)
    return np.asarray(groups)


def run_analysis(args) -> Path:
    if "avscramble" in args.model.lower():
        raise ValueError(
            "Canonical avscramble embeddings globally permute audio across the "
            "train/test split and cannot be used for this analysis"
        )
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    timing = pd.read_csv(args.timing_csv)
    test_ids = {item.strip() for item in args.test_video_ids.split(",") if item.strip()}
    nuisance_model = args.nuisance_model or args.model
    nuisance_modalities = [
        item.strip() for item in args.nuisance_modalities.split(",") if item.strip()
    ]
    if not nuisance_modalities:
        raise ValueError("At least one nuisance modality is required")

    fmri_path = (
        Path(args.preprocessed_dir)
        / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    )
    run_trs_path = (
        Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    )
    y_train, y_test, _ = build_fmri_arrays(
        str(fmri_path), str(run_trs_path), timing, sorted(test_ids),
        args.bin_sec, args.tr, args.delay_sec, args.skip_sec,
    )
    train_runs = _training_run_ids(timing, test_ids, args.bin_sec, args.skip_sec)
    if len(train_runs) != len(y_train):
        raise ValueError("Training run labels do not align with fMRI samples")

    bin_dir = f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"
    nuisance_root = Path(args.embeddings_dir) / nuisance_model / bin_dir
    joint_root = Path(args.embeddings_dir) / args.model / bin_dir
    raw_bands = [
        np.load(nuisance_root / f"{nuisance_model}_{modality}.npy")
        for modality in nuisance_modalities
    ]
    raw_bands.append(np.load(joint_root / f"{args.model}_av.npy"))

    train_bands, test_bands = [], []
    for band in raw_bands:
        train, test = split_embedding_array(
            band, timing, sorted(test_ids), args.bin_sec,
            hrf=False, normalize=args.normalize, skip_sec=args.skip_sec,
        )
        train_bands.append(train)
        test_bands.append(test)
    train_bands, test_bands, _ = standardize_bands(train_bands, test_bands)
    if any(len(band) != len(y_train) for band in train_bands):
        raise ValueError("Training embeddings do not align with fMRI samples")
    if any(len(band) != len(y_test) for band in test_bands):
        raise ValueError("Test embeddings do not align with fMRI samples")

    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    baseline_prediction, baseline_model, backend, _ = fit_group_ridge(
        train_bands[:-1], y_train, test_bands[:-1], train_runs, alphas,
        n_iter=args.n_iter, backend=args.backend, random_state=0,
    )
    extended_prediction, extended_model, _, _ = fit_group_ridge(
        train_bands, y_train, test_bands, train_runs, alphas,
        n_iter=args.n_iter, backend=args.backend, random_state=0,
    )
    baseline_r2 = _r2_per_target(y_test, baseline_prediction).astype(np.float32)
    extended_r2 = _r2_per_target(y_test, extended_prediction).astype(np.float32)
    delta_r2 = (extended_r2 - baseline_r2).astype(np.float32)

    output = Path(args.output_dir) / args.subject / args.model / _config_label(args)
    output.mkdir(parents=True, exist_ok=True)
    maps = {
        "incremental_av_baseline_r2": baseline_r2,
        "incremental_av_extended_r2": extended_r2,
        "incremental_av_delta_r2": delta_r2,
    }
    for name, values in maps.items():
        save_cifti(
            values, args.template_cifti, str(output / f"{name}.dscalar.nii"),
            map_name=name,
        )
    np.savez_compressed(
        output / "incremental_av_predictions.npz",
        y_true=y_test,
        baseline_prediction=baseline_prediction,
        extended_prediction=extended_prediction,
        baseline_error=y_test - baseline_prediction,
        extended_error=y_test - extended_prediction,
        baseline_r2=baseline_r2,
        extended_r2=extended_r2,
        delta_r2=delta_r2,
        baseline_best_alphas=_to_numpy(baseline_model.best_alphas_),
        extended_best_alphas=_to_numpy(extended_model.best_alphas_),
    )
    (output / "incremental_av_provenance.json").write_text(json.dumps({
        "comparison": "A+V+J minus A+V",
        "split": "fixed held-out validation clips",
        "test_video_ids": sorted(test_ids),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "nuisance_model": nuisance_model,
        "nuisance_modalities": nuisance_modalities,
        "joint_model": args.model,
        "backend": backend,
        "alphas": alphas.tolist(),
        "n_iter": args.n_iter,
        "warning": "Use encoding/incremental_av.py for primary unseen-run inference.",
    }, indent=2) + "\n")
    log.info(
        "Saved direct incremental AV maps to %s (mean delta R2 %.5f)",
        output, np.nanmean(delta_r2),
    )
    return output


if __name__ == "__main__":
    run_analysis(parse_args())
