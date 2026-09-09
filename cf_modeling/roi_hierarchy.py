"""Directionality tests for whether frontal parcels inherit stimulus
representation from four temporal/occipitotemporal source ROIs.

Three converging, label-free tests on group-average data, all evaluated on
held-out movie runs with training-only fitting:

  mediation         Attenuation asymmetry: regress each source ROI's
                     ROI-mean timecourse out of each frontal target's (and
                     vice versa), refit the audio+video+joint encoding
                     model, and compare how much held-out R2 each direction
                     destroys. Positive asymmetry means the target is
                     downstream of the source.
  latency           Sweep --delay-sec and record the delay maximizing
                     held-out R2 per ROI.
  integration_window Sweep the bin lengths that actually exist on disk for
                     this model's embeddings and record held-out R2 per ROI
                     as a function of bin length.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from encoding.incremental_av import (
    REPEATED_VALIDATION_CLIPS,
    _bh_qvalues,
    apply_hrf_by_clip,
    load_target_rois,
    sample_metadata,
)
from encoding.roi_av_profile import MIN_R2_FOR_REDUNDANCY
from encoding.shared.encoding_utils import build_fmri_arrays
from encoding.shared.fold_evaluator import _r2_per_target, fit_group_ridge, standardize_bands


log = logging.getLogger("roi_hierarchy")

_BIN_DIR_PATTERN = re.compile(r"^bin(\d+(?:\.\d+)?)s_skip(\d+(?:\.\d+)?)s$")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--fmri-suffix", default="raw")
    parser.add_argument("--subject", default="group_average")
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--hrf", action="store_true")
    parser.add_argument(
        "--exclude-video-ids", default=",".join(REPEATED_VALIDATION_CLIPS),
    )
    parser.add_argument(
        "--roi-mask", action="append", default=[], metavar="NAME=PATH",
        help="Named nonzero CIFTI mask; may be repeated.",
    )
    parser.add_argument("--glasser-dlabel")
    parser.add_argument(
        "--parcel-roi", action="append", default=[], metavar="NAME=PARCELS",
        help="Named comma-separated bilateral Glasser parcel set; may be repeated.",
    )
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument(
        "--source-roi", action="append", required=True, metavar="NAME",
        help="ROI name (from --roi-mask/--parcel-roi) treated as an upstream "
             "source; every other loaded ROI is a frontal target. Repeatable.",
    )
    parser.add_argument(
        "--stage", choices=("mediation", "latency", "integration_window", "all"),
        default="all",
    )
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=9.0)
    parser.add_argument("--n-alphas", type=int, default=23)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--model-random-state", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument(
        "--delay-grid", nargs="+", type=float,
        default=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    )
    return parser.parse_args(argv)


# =============================================================================
# Data loading — mirrors encoding/incremental_av.py's run() setup, but reduces
# every target ROI to its mean timecourse instead of keeping per-vertex targets.
# =============================================================================

def _load_bundle(args, bin_sec: float, skip_sec: float, delay_sec: float, hrf: bool):
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, bin_sec, skip_sec)
    root = Path(args.embeddings_dir) / args.model / f"bin{int(bin_sec)}s_skip{int(skip_sec)}s"
    embeddings = {
        "a": np.load(root / f"{args.model}_a.npy"),
        "v": np.load(root / f"{args.model}_v.npy"),
        "av": np.load(root / f"{args.model}_av.npy"),
    }
    expected = len(metadata)
    for name, values in embeddings.items():
        if values.shape[0] != expected:
            raise ValueError(
                f"{name} embeddings have {values.shape[0]} rows; timing describes {expected}")
        if hrf:
            embeddings[name] = apply_hrf_by_clip(values, metadata, bin_sec)

    excluded = {item.strip() for item in args.exclude_video_ids.split(",") if item.strip()}
    keep = ~metadata["video_id"].isin(excluded).to_numpy()
    fmri_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    run_trs_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    targets, _, _ = build_fmri_arrays(
        str(fmri_path), str(run_trs_path), timing, sorted(excluded), bin_sec, args.tr,
        delay_sec=delay_sec, skip_sec=skip_sec,
    )
    if targets.shape[0] != int(keep.sum()):
        raise ValueError(
            f"fMRI rows {targets.shape[0]} do not align to {int(keep.sum())} retained rows")

    target_indices, roi_columns = load_target_rois(args, fmri_path, targets.shape[1])
    targets = targets[:, target_indices]
    metadata = metadata.loc[keep].sort_values(
        ["run_id", "row_index"], kind="stable",
    ).reset_index(drop=True)
    retained_rows = metadata["row_index"].to_numpy(dtype=int)
    for name in embeddings:
        embeddings[name] = embeddings[name][retained_rows]

    roi_means = {name: targets[:, cols].mean(axis=1).astype(np.float64)
                 for name, cols in roi_columns.items()}
    missing_sources = sorted(set(args.source_roi) - set(roi_means))
    if missing_sources:
        raise KeyError(f"--source-roi not among loaded ROIs: {missing_sources}")
    frontal_names = [name for name in roi_means if name not in args.source_roi]
    if not frontal_names:
        raise ValueError("No frontal target ROIs remain after removing --source-roi entries")
    roi_names = list(args.source_roi) + frontal_names
    roi_matrix = np.column_stack([roi_means[name] for name in roi_names])
    run_ids = metadata["run_id"].to_numpy()
    video_ids = metadata["video_id"].to_numpy()
    return embeddings["a"], embeddings["v"], embeddings["av"], roi_matrix, roi_names, run_ids, video_ids


# =============================================================================
# Fold-level encoding fits
# =============================================================================

def _fit_fold_predict(audio, video, joint, target_matrix, run_ids, test_run, alphas,
                       n_iter, backend, model_random_state):
    """Fit on every run but ``test_run``, predict on it. Returns (test_mask, prediction)."""
    run_ids = np.asarray(run_ids)
    test_mask = run_ids == test_run
    train_mask = ~test_mask
    train_bands, test_bands, _ = standardize_bands(
        [audio[train_mask], video[train_mask], joint[train_mask]],
        [audio[test_mask], video[test_mask], joint[test_mask]],
    )
    y_train = np.asarray(target_matrix[train_mask], dtype=np.float32)
    prediction, _, _, _ = fit_group_ridge(
        train_bands, y_train, test_bands, run_ids[train_mask], alphas,
        n_iter=n_iter, backend=backend, random_state=model_random_state,
    )
    return test_mask, prediction


def _fit_fold_r2(audio, video, joint, target_matrix, run_ids, test_run, alphas,
                  n_iter, backend, model_random_state):
    test_mask, prediction = _fit_fold_predict(
        audio, video, joint, target_matrix, run_ids, test_run, alphas,
        n_iter, backend, model_random_state,
    )
    y_test = np.asarray(target_matrix[test_mask], dtype=np.float32)
    return _r2_per_target(y_test, prediction), np.flatnonzero(test_mask)


def _oof_predictions(audio, video, joint, target_matrix, run_ids, folds, alphas,
                      n_iter, backend, model_random_state):
    """Out-of-fold predictions for every row, one LORO fold at a time."""
    oof = np.full(target_matrix.shape, np.nan, dtype=np.float64)
    for test_run in folds:
        test_mask, prediction = _fit_fold_predict(
            audio, video, joint, target_matrix, run_ids, test_run, alphas,
            n_iter, backend, model_random_state,
        )
        oof[test_mask] = prediction
    return oof


def _block_r2(true_values, predictions, video_ids, blocks):
    """Held-out R2 per movie block, one row per block (columns = targets)."""
    table = np.full((len(blocks), true_values.shape[1]), np.nan, dtype=np.float64)
    for row, block in enumerate(blocks):
        mask = video_ids == block
        table[row] = _r2_per_target(true_values[mask], predictions[mask])
    return table


def _block_stats_two_sided(effects: np.ndarray, n_bootstrap: int, n_permutations: int,
                            random_state: int) -> dict:
    """Bootstrap CI and two-sided sign-flip test over one effect per movie block.

    Same NaN-block gating as encoding.roi_av_profile._resample_stats (a block
    with an undefined ratio is dropped, not zero-filled; the stat is NaN if
    fewer than 2 blocks remain), but two-sided: the null here is
    |sign-flipped mean| >= |observed mean|, since mediation asymmetry is
    meaningful in either sign (source-downstream vs target-downstream).
    """
    effects = np.asarray(effects, dtype=np.float64)
    defined = effects[~np.isnan(effects)]
    if len(defined) < 2:
        return {
            "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "sign_flip_p_two_sided": float("nan"),
            "n_blocks": int(len(effects)), "n_blocks_defined": int(len(defined)),
        }
    rng = np.random.default_rng(random_state)
    bootstrap = defined[rng.integers(0, len(defined), (n_bootstrap, len(defined)))].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(defined)))
    null = (signs * defined).mean(axis=1)
    observed = float(defined.mean())
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1))
    return {
        "mean": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p_two_sided": p_value,
        "n_blocks": int(len(effects)), "n_blocks_defined": int(len(defined)),
    }


def _roi_r2_table(audio, video, joint, target_matrix, run_ids, alphas,
                   n_iter, backend, model_random_state):
    """Held-out R2 for every target column, one row per leave-one-run-out fold."""
    folds = sorted(np.unique(run_ids).tolist())
    table = np.full((len(folds), target_matrix.shape[1]), np.nan, dtype=np.float64)
    for row, test_run in enumerate(folds):
        r2_vals, _ = _fit_fold_r2(
            audio, video, joint, target_matrix, run_ids, test_run, alphas,
            n_iter, backend, model_random_state,
        )
        table[row] = r2_vals
    return table, folds


def _residualize_matrix(target_matrix: np.ndarray, nuisance_col: np.ndarray,
                         train_mask: np.ndarray) -> np.ndarray:
    """Regress ``nuisance_col`` out of every column, weights fit on train only."""
    train_mask = np.asarray(train_mask, dtype=bool)
    design_train = np.column_stack(
        [np.ones(int(train_mask.sum())), nuisance_col[train_mask]])
    beta, *_ = np.linalg.lstsq(design_train, target_matrix[train_mask], rcond=None)
    design_full = np.column_stack([np.ones(len(nuisance_col)), nuisance_col])
    return target_matrix - design_full @ beta


# =============================================================================
# (a) Mediation asymmetry
# =============================================================================

def mediation_asymmetry(
    audio: np.ndarray, video: np.ndarray, joint: np.ndarray,
    roi_matrix: np.ndarray, roi_names: list[str], sources: list[str],
    run_ids: np.ndarray, video_ids: np.ndarray, alphas: np.ndarray,
    n_iter: int = 20, backend: str = "torch", model_random_state: int = 0,
    n_bootstrap: int = 10000, n_permutations: int = 10000, random_state: int = 0,
) -> pd.DataFrame:
    """Attenuation A(X->Y) = 1 - R2_Y_given_X / R2_Y for every source/target pair.

    Asymmetry = A(source->target) - A(target->source); positive means the
    target is downstream of the source. A(X->Y) is undefined (NaN) on any
    block where R2_Y is at or below MIN_R2_FOR_REDUNDANCY: dividing by a
    near-zero or negative R2 there is meaningless, not a real attenuation.

    Inference resamples over movie blocks (one non-validation video segment
    each), not outer LORO folds: with only len(folds) folds a sign-flip test
    floors at 1/2**len(folds) and no result could ever be significant.
    """
    name_to_col = {name: i for i, name in enumerate(roi_names)}
    frontal = [name for name in roi_names if name not in sources]
    if not sources or not frontal:
        raise ValueError("mediation_asymmetry needs at least one source and one frontal target")

    run_ids = np.asarray(run_ids)
    video_ids = np.asarray(video_ids)
    folds = sorted(np.unique(run_ids).tolist())
    blocks = sorted(np.unique(video_ids).tolist())

    frontal_matrix = roi_matrix[:, [name_to_col[name] for name in frontal]]
    source_matrix = roi_matrix[:, [name_to_col[name] for name in sources]]

    raw_oof = _oof_predictions(
        audio, video, joint, roi_matrix, run_ids, folds, alphas, n_iter, backend, model_random_state,
    )
    raw_block_r2 = _block_r2(roi_matrix, raw_oof, video_ids, blocks)

    given_s_true = np.full((len(sources), *frontal_matrix.shape), np.nan)
    given_s_pred = np.full_like(given_s_true, np.nan)
    for s_index, s_name in enumerate(sources):
        s_col = roi_matrix[:, name_to_col[s_name]]
        for test_run in folds:
            train_mask = run_ids != test_run
            residual = _residualize_matrix(frontal_matrix, s_col, train_mask)
            test_mask, prediction = _fit_fold_predict(
                audio, video, joint, residual, run_ids, test_run, alphas,
                n_iter, backend, model_random_state,
            )
            given_s_true[s_index, test_mask] = residual[test_mask]
            given_s_pred[s_index, test_mask] = prediction

    given_t_true = np.full((len(frontal), *source_matrix.shape), np.nan)
    given_t_pred = np.full_like(given_t_true, np.nan)
    for t_index, t_name in enumerate(frontal):
        t_col = roi_matrix[:, name_to_col[t_name]]
        for test_run in folds:
            train_mask = run_ids != test_run
            residual = _residualize_matrix(source_matrix, t_col, train_mask)
            test_mask, prediction = _fit_fold_predict(
                audio, video, joint, residual, run_ids, test_run, alphas,
                n_iter, backend, model_random_state,
            )
            given_t_true[t_index, test_mask] = residual[test_mask]
            given_t_pred[t_index, test_mask] = prediction

    rows = []
    for s_index, s_name in enumerate(sources):
        r2_s_block = raw_block_r2[:, name_to_col[s_name]]
        r2_t_given_s_block = _block_r2(given_s_true[s_index], given_s_pred[s_index], video_ids, blocks)
        for t_index, t_name in enumerate(frontal):
            r2_t_block = raw_block_r2[:, name_to_col[t_name]]
            r2_s_given_t_block = _block_r2(given_t_true[t_index], given_t_pred[t_index], video_ids, blocks)

            a_s_to_t = np.where(
                r2_t_block > MIN_R2_FOR_REDUNDANCY,
                1.0 - np.clip(r2_t_given_s_block[:, t_index], 0.0, None) / r2_t_block,
                np.nan,
            )
            a_t_to_s = np.where(
                r2_s_block > MIN_R2_FOR_REDUNDANCY,
                1.0 - np.clip(r2_s_given_t_block[:, s_index], 0.0, None) / r2_s_block,
                np.nan,
            )
            asymmetry = a_s_to_t - a_t_to_s

            stats = _block_stats_two_sided(asymmetry, n_bootstrap, n_permutations, random_state)

            rows.append({
                "source": s_name,
                "target": t_name,
                "n_blocks": stats["n_blocks"],
                "n_blocks_defined": stats["n_blocks_defined"],
                "r2_source_mean": float(np.nanmean(r2_s_block)),
                "r2_target_mean": float(np.nanmean(r2_t_block)),
                "attenuation_source_to_target": float(np.nanmean(a_s_to_t)),
                "attenuation_target_to_source": float(np.nanmean(a_t_to_s)),
                "asymmetry": stats["mean"],
                "asymmetry_ci_low": stats["ci_low"],
                "asymmetry_ci_high": stats["ci_high"],
                "sign_flip_p_two_sided": stats["sign_flip_p_two_sided"],
                "downstream_direction": (
                    "target_downstream_of_source" if stats["mean"] > 0 else "source_downstream_of_target"
                ) if np.isfinite(stats["mean"]) else "undefined",
            })
    table = pd.DataFrame(rows)
    table["sign_flip_q_bh"] = np.nan
    valid = table["sign_flip_p_two_sided"].notna()
    if valid.any():
        table.loc[valid, "sign_flip_q_bh"] = _bh_qvalues(table.loc[valid, "sign_flip_p_two_sided"])
    return table


def run_mediation(args) -> Path:
    audio, video, joint, roi_matrix, roi_names, run_ids, video_ids = _load_bundle(
        args, args.bin_sec, args.skip_sec, args.delay_sec, args.hrf,
    )
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    table = mediation_asymmetry(
        audio, video, joint, roi_matrix, roi_names, args.source_roi, run_ids, video_ids, alphas,
        n_iter=args.n_iter, backend=args.backend, model_random_state=args.model_random_state,
        n_bootstrap=args.n_bootstrap, n_permutations=args.n_permutations,
        random_state=args.model_random_state,
    )
    output_root = Path(args.output_dir) / args.subject / args.model / "mediation"
    output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_root / "mediation_asymmetry.csv", index=False)
    (output_root / "manifest.json").write_text(json.dumps({
        "model": args.model,
        "sources": list(args.source_roi),
        "frontal_targets": [name for name in roi_names if name not in args.source_roi],
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec, "delay_sec": args.delay_sec,
        "hrf": args.hrf,
        "n_folds": int(len(np.unique(run_ids))),
        "n_blocks": int(len(np.unique(video_ids))),
        "inference_unit": "movie block (one non-validation video segment)",
        "min_r2_for_redundancy": MIN_R2_FOR_REDUNDANCY,
        "definition": (
            "Attenuation A(X->Y) = 1 - R2_Y_given_X / R2_Y, where R2_Y_given_X is the "
            "held-out encoding R2 of Y after regressing X's training-fold-estimated "
            "contribution out of Y, both computed per movie block from out-of-fold "
            "LORO predictions. A(X->Y) is NaN on a block where R2_Y <= "
            "min_r2_for_redundancy. Asymmetry = A(S->T) - A(T->S); positive means T is "
            "downstream of S."
        ),
    }, indent=2) + "\n")
    log.info("Saved mediation asymmetry table to %s", output_root)
    return output_root


# =============================================================================
# (b) Response latency gradient
# =============================================================================

def run_latency(args) -> Path:
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    rows = []
    roi_names_ref = None
    for delay_sec in args.delay_grid:
        audio, video, joint, roi_matrix, roi_names, run_ids, _ = _load_bundle(
            args, args.bin_sec, args.skip_sec, delay_sec, args.hrf,
        )
        if roi_names_ref is None:
            roi_names_ref = roi_names
        elif roi_names != roi_names_ref:
            raise RuntimeError("ROI set changed across the delay grid")
        table, _ = _roi_r2_table(
            audio, video, joint, roi_matrix, run_ids, alphas,
            args.n_iter, args.backend, args.model_random_state,
        )
        mean_r2 = table.mean(axis=0)
        for name, value in zip(roi_names, mean_r2):
            rows.append({
                "delay_sec": delay_sec, "roi": name, "r2_mean": float(value),
                "role": "source" if name in args.source_roi else "frontal_target",
            })
        log.info("Latency sweep delay=%.1fs done", delay_sec)

    frame = pd.DataFrame(rows)
    best = frame.loc[frame.groupby("roi")["r2_mean"].idxmax()].reset_index(drop=True)
    best = best.rename(columns={"delay_sec": "peak_delay_sec", "r2_mean": "peak_r2_mean"})

    output_root = Path(args.output_dir) / args.subject / args.model / "latency"
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "latency_curve.csv", index=False)
    best.to_csv(output_root / "latency_peaks.csv", index=False)
    (output_root / "manifest.json").write_text(json.dumps({
        "model": args.model, "delay_grid": list(args.delay_grid),
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec, "hrf": args.hrf,
    }, indent=2) + "\n")
    log.info("Saved latency gradient to %s", output_root)
    return output_root


# =============================================================================
# (c) Temporal integration-window gradient
# =============================================================================

def discover_bin_grid(embeddings_dir, model: str) -> list[float]:
    """bin{N}s_skip{N}s directories that actually hold a/v/av embeddings."""
    root = Path(embeddings_dir) / model
    if not root.is_dir():
        raise FileNotFoundError(f"No embeddings directory for model {model!r}: {root}")
    grid = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _BIN_DIR_PATTERN.match(entry.name)
        if not match or match.group(1) != match.group(2):
            continue
        bin_sec = float(match.group(1))
        paths = [entry / f"{model}_{modality}.npy" for modality in ("a", "v", "av")]
        if not all(path.exists() for path in paths):
            continue
        rows = {np.load(path, mmap_mode="r").shape[0] for path in paths}
        if len(rows) == 1 and rows.pop() > 1:
            grid.append(bin_sec)
    if not grid:
        raise FileNotFoundError(
            f"No usable bin{{N}}s_skip{{N}}s embedding directories under {root}")
    return sorted(grid)


def run_integration_window(args) -> Path:
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    grid = discover_bin_grid(args.embeddings_dir, args.model)
    log.info("Integration-window grid for %s: %s", args.model, grid)
    rows = []
    roi_names_ref = None
    for bin_sec in grid:
        audio, video, joint, roi_matrix, roi_names, run_ids, _ = _load_bundle(
            args, bin_sec, bin_sec, args.delay_sec, args.hrf,
        )
        if roi_names_ref is None:
            roi_names_ref = roi_names
        elif roi_names != roi_names_ref:
            raise RuntimeError("ROI set changed across the bin-length grid")
        table, _ = _roi_r2_table(
            audio, video, joint, roi_matrix, run_ids, alphas,
            args.n_iter, args.backend, args.model_random_state,
        )
        mean_r2 = table.mean(axis=0)
        for name, value in zip(roi_names, mean_r2):
            rows.append({
                "bin_sec": bin_sec, "roi": name, "r2_mean": float(value),
                "role": "source" if name in args.source_roi else "frontal_target",
            })
        log.info("Integration-window sweep bin=%.1fs done", bin_sec)

    frame = pd.DataFrame(rows)
    best = frame.loc[frame.groupby("roi")["r2_mean"].idxmax()].reset_index(drop=True)
    best = best.rename(columns={"bin_sec": "peak_bin_sec", "r2_mean": "peak_r2_mean"})

    output_root = Path(args.output_dir) / args.subject / args.model / "integration_window"
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "integration_window_curve.csv", index=False)
    best.to_csv(output_root / "integration_window_peaks.csv", index=False)
    (output_root / "manifest.json").write_text(json.dumps({
        "model": args.model, "bin_grid_sec": grid, "delay_sec": args.delay_sec, "hrf": args.hrf,
    }, indent=2) + "\n")
    log.info("Saved integration-window gradient to %s", output_root)
    return output_root


# =============================================================================
# Dispatch
# =============================================================================

def run(args) -> None:
    if args.stage in ("mediation", "all"):
        run_mediation(args)
    if args.stage in ("latency", "all"):
        run_latency(args)
    if args.stage in ("integration_window", "all"):
        run_integration_window(args)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
