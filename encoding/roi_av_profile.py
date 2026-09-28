"""Per-ROI audio/video/joint variance decomposition on held-out movie runs.

For each ROI, fits nested leave-one-run-out ridge models on the group-average
response using two proxy bands (A, V) plus the model's native joint
audiovisual embedding (J). The ridge is fit per vertex, but both evaluation
metrics are computed on the ROI-mean timecourse: held-out R2 (giving synergy)
is the coefficient of determination between the
ROI-mean observed and ROI-mean out-of-fold predicted timecourses, and unique
audio/video variance is a genuine partial correlation between the same
ROI-mean observed timecourse and each band's ROI-mean out-of-fold
prediction, controlling for the other band's. Two band configurations are
supported via --band-config:

  av    A = audio embedding, V = video embedding (native modality bands)
  text  A = transcript embedding (speech content, audio-semantic proxy),
        V = caption embedding (scene description, visual-semantic proxy)

J is always the model's native "av" embedding regardless of band
configuration: for --band-config text this asks whether the model's actual
fused audiovisual signal explains anything beyond the transcript+caption
linguistic proxies.
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

from cf_modeling.roi_mean_partial_connectivity import paired_partial_correlations
from encoding.incremental_av import (
    REPEATED_VALIDATION_CLIPS,
    _bh_qvalues,
    apply_hrf_by_clip,
    load_target_rois,
    sample_metadata,
)
from encoding.shared.encoding_utils import build_fmri_arrays
from encoding.shared.fold_evaluator import _r2_per_target, fit_group_ridge, standardize_bands

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BAND_CONFIGS = {
    "av": ("a", "v"),
    "text": ("transcript_t", "caption_t"),
}
JOINT_MODALITY = "av"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--fmri-suffix", default="raw")
    parser.add_argument("--subject", default="group_average")
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--band-config", choices=sorted(BAND_CONFIGS), default="av")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--hrf", action="store_true")
    parser.add_argument(
        "--exclude-video-ids",
        default=",".join(REPEATED_VALIDATION_CLIPS),
        help="Clips excluded from all fitting and evaluation.",
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
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=9.0)
    parser.add_argument("--n-alphas", type=int, default=23)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--model-random-state", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    return parser.parse_args(argv)


# =============================================================================
# Decomposition arithmetic (pure functions, unit-tested directly)
# =============================================================================

def decomposition_metrics(r2_a: float, r2_v: float, r2_additive: float, r2_joint: float,
                           unique_a: float, unique_v: float) -> dict:
    """Assemble the R2-based and partial-correlation-based decomposition quantities.

    unique_a/unique_v are genuine partial correlations -- corr(ROI, A | V) and
    corr(ROI, V | A) -- computed by the caller from out-of-fold predictions
    (see partial_unique_correlations); this function only derives the
    remaining R2-based quantity (synergy) and assembles the record.
    """
    synergy = r2_joint - r2_additive
    return {
        "R2_A": r2_a, "R2_V": r2_v, "R2_additive": r2_additive, "R2_joint": r2_joint,
        "unique_A": unique_a, "unique_V": unique_v, "synergy": synergy,
    }


def roi_mean_timecourse(values: np.ndarray) -> np.ndarray:
    """Collapse an (n_samples, n_roi_targets) array to its per-sample ROI mean."""
    return np.asarray(values, dtype=np.float64).mean(axis=1)


def roi_mean_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    """Out-of-fold R2 between the ROI-mean observed and predicted timecourses.

    target/prediction are (n_samples, n_roi_targets) arrays; both are
    collapsed to their un-z-scored ROI mean via roi_mean_timecourse before
    the standard R2 identity (1 - SS_res/SS_tot) is applied, so this stays a
    genuine coefficient of determination -- unbounded below, not a squared
    correlation -- evaluated on the same ROI-mean object as
    partial_unique_correlations, just without that function's z-scoring.
    """
    observed = roi_mean_timecourse(target)
    predicted = roi_mean_timecourse(prediction)
    return float(_r2_per_target(observed[:, None], predicted[:, None])[0])


def partial_unique_correlations(target: np.ndarray, audio_pred: np.ndarray,
                                 video_pred: np.ndarray) -> tuple[float, float]:
    """corr(ROI, A | V) and corr(ROI, V | A) from out-of-fold predictions.

    target/audio_pred/video_pred are (n_samples, n_roi_targets) arrays over the
    same observations (typically one movie block, one ROI); each is collapsed
    to an ROI-mean timecourse via roi_mean_timecourse, z-scored over those
    observations, and passed to paired_partial_correlations's exact
    partial-correlation identity.
    """
    def roi_mean_z(values: np.ndarray) -> np.ndarray:
        mean = roi_mean_timecourse(values)
        std = mean.std()
        return (mean - mean.mean()) / std if std > np.finfo(np.float64).eps else np.zeros_like(mean)

    unique_a, unique_v = paired_partial_correlations(
        roi_mean_z(target), roi_mean_z(audio_pred), roi_mean_z(video_pred)
    )
    return float(unique_a[0]), float(unique_v[0])


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality of x's (samples x features) covariance eigenspectrum."""
    centered = np.asarray(x, dtype=np.float64) - x.mean(axis=0, keepdims=True)
    eigvals = np.clip(np.linalg.eigvalsh(centered.T @ centered), 0, None)
    total = eigvals.sum()
    return float(total ** 2 / np.square(eigvals).sum()) if total > 1e-12 else float("nan")


def _bootstrap_sign_flip(effects: np.ndarray, n_bootstrap: int, n_permutations: int,
                          random_state: int) -> dict:
    """Bootstrap CI and sign-flip permutation test over a vector of per-block effects.

    Same pattern as encoding.shared.fold_evaluator.paired_clip_inference, but
    over a caller-supplied effect vector (here: one scalar per movie block)
    rather than per-clip paired prediction errors. Resampling unit is the
    movie block (one non-validation video segment), not the outer LORO fold:
    with only 4 outer folds a sign-flip test floors at 1/2**4 and can never
    reach significance regardless of effect size.
    """
    effects = np.asarray(effects, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    bootstrap = effects[rng.integers(0, len(effects), (n_bootstrap, len(effects)))].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(effects)))
    null = (signs * effects).mean(axis=1)
    observed = float(effects.mean())
    return {
        "mean": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p_greater": float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
    }


def _resample_stats(effects: np.ndarray, n_bootstrap: int, n_permutations: int,
                     random_state: int) -> dict:
    """Block bootstrap/sign-flip over a metric vector, one entry per movie block.

    NaN entries (an undefined metric on that block) are dropped before
    resampling; their fraction is reported rather than silently discarded,
    and the whole statistic is NaN if fewer than 2 blocks remain.
    """
    effects = np.asarray(effects, dtype=np.float64)
    defined = effects[~np.isnan(effects)]
    fraction_defined = float(len(defined) / len(effects)) if len(effects) else float("nan")
    if len(defined) < 2:
        stats = {
            "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "sign_flip_p_greater": float("nan"),
        }
    else:
        stats = _bootstrap_sign_flip(defined, n_bootstrap, n_permutations, random_state)
    stats.update({
        "n_blocks": int(len(effects)),
        "n_blocks_defined": int(len(defined)),
        "fraction_defined": fraction_defined,
    })
    return stats


def _with_bh_qvalues(frame: pd.DataFrame) -> pd.DataFrame:
    frame["q_bh"] = np.nan
    valid = frame["sign_flip_p_greater"].notna()
    if valid.any():
        frame.loc[valid, "q_bh"] = _bh_qvalues(frame.loc[valid, "sign_flip_p_greater"])
    return frame


# =============================================================================
# Fitting
# =============================================================================

def _load_band(embeddings_dir: str, model: str, modality: str, bin_sec: float,
                skip_sec: float) -> np.ndarray:
    path = (
        Path(embeddings_dir) / model / f"bin{int(bin_sec)}s_skip{int(skip_sec)}s" /
        f"{model}_{modality}.npy"
    )
    return np.load(path).astype(np.float64)


def _fit_fold(embeddings: dict, targets: np.ndarray, run_ids: np.ndarray, test_run,
              alphas: np.ndarray, args) -> dict:
    """Fit A, V, A+V, A+V+J on one held-out run; return predictions and R2 arrays."""
    test_mask = run_ids == test_run
    train_mask = ~test_mask
    train_runs = run_ids[train_mask]
    y_train, y_test = targets[train_mask], targets[test_mask]

    raw_train = [embeddings[name][train_mask] for name in ("a", "b", "j")]
    raw_test = [embeddings[name][test_mask] for name in ("a", "b", "j")]
    std_train, std_test, _ = standardize_bands(raw_train, raw_test)
    a_train, b_train, j_train = std_train
    a_test, b_test, j_test = std_test

    fits = {}
    for tag, train_bands, test_bands in (
        ("A", [a_train], [a_test]),
        ("V", [b_train], [b_test]),
        ("additive", [a_train, b_train], [a_test, b_test]),
        ("joint", [a_train, b_train, j_train], [a_test, b_test, j_test]),
    ):
        prediction, _, _, _ = fit_group_ridge(
            train_bands, y_train, test_bands, train_runs, alphas,
            n_iter=args.n_iter, backend=args.backend, random_state=args.model_random_state,
        )
        fits[tag] = (prediction, _r2_per_target(y_test, prediction))
    return {"test_mask": test_mask, "fits": fits}


def run(args) -> Path:
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, args.bin_sec, args.skip_sec)

    band_a_modality, band_b_modality = BAND_CONFIGS[args.band_config]
    modality_by_key = {"a": band_a_modality, "b": band_b_modality, "j": JOINT_MODALITY}
    embeddings = {}
    for key, modality in modality_by_key.items():
        values = _load_band(args.embeddings_dir, args.model, modality, args.bin_sec, args.skip_sec)
        if values.shape[0] != len(metadata):
            raise ValueError(
                f"{modality} has {values.shape[0]} rows; timing describes {len(metadata)}"
            )
        embeddings[key] = values
    if args.hrf:
        for key in embeddings:
            embeddings[key] = apply_hrf_by_clip(embeddings[key], metadata, args.bin_sec)

    excluded = {item.strip() for item in args.exclude_video_ids.split(",") if item.strip()}
    keep = ~metadata["video_id"].isin(excluded).to_numpy()

    fmri_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    run_trs_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    targets, _, _ = build_fmri_arrays(
        str(fmri_path), str(run_trs_path), timing, sorted(excluded), args.bin_sec, args.tr,
        delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    if targets.shape[0] != int(keep.sum()):
        raise ValueError(
            f"fMRI rows {targets.shape[0]} do not align to {int(keep.sum())} retained rows"
        )

    target_indices, roi_columns = load_target_rois(args, fmri_path, targets.shape[1])
    targets = targets[:, target_indices]

    metadata = metadata.loc[keep].sort_values(["run_id", "row_index"], kind="stable").reset_index(drop=True)
    retained_rows = metadata["row_index"].to_numpy(dtype=int)
    for key in embeddings:
        embeddings[key] = embeddings[key][retained_rows]

    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    run_ids = metadata["run_id"].to_numpy()
    folds = sorted(pd.unique(run_ids))
    n_samples, n_all_targets = targets.shape
    oof = {
        tag: np.full((n_samples, n_all_targets), np.nan, dtype=np.float32)
        for tag in ("A", "V", "additive", "joint")
    }
    for test_run in folds:
        log.info("Fitting held-out run %s", test_run)
        result = _fit_fold(embeddings, targets, run_ids, test_run, alphas, args)
        test_mask = result["test_mask"]
        for tag in oof:
            oof[tag][test_mask] = result["fits"][tag][0]

    # Inference resamples over movie blocks (non-validation video segments),
    # not outer LORO folds: with only len(folds) folds a sign-flip test floors
    # at 1/2**len(folds) and no result could ever be significant.
    video_ids = metadata["video_id"].to_numpy()
    blocks = sorted(pd.unique(video_ids))
    block_records = []
    for roi, columns in roi_columns.items():
        for block in blocks:
            block_mask = video_ids == block
            r2 = {
                tag: roi_mean_r2(
                    targets[block_mask][:, columns], oof[tag][block_mask][:, columns]
                )
                for tag in oof
            }
            unique_a, unique_v = partial_unique_correlations(
                targets[block_mask][:, columns],
                oof["A"][block_mask][:, columns], oof["V"][block_mask][:, columns],
            )
            record = decomposition_metrics(
                r2["A"], r2["V"], r2["additive"], r2["joint"], unique_a, unique_v
            )
            record["roi"] = roi
            record["video_id"] = str(block)
            record["n_block_samples"] = int(block_mask.sum())
            block_records.append(record)
    block_frame = pd.DataFrame(block_records)

    metric_names = [
        "R2_A", "R2_V", "R2_additive", "R2_joint",
        "unique_A", "unique_V", "synergy",
    ]
    inference_rows = []
    for roi in roi_columns:
        roi_frame = block_frame.loc[block_frame["roi"] == roi]
        for metric in metric_names:
            stats = _resample_stats(
                roi_frame[metric].to_numpy(dtype=float),
                args.n_bootstrap, args.n_permutations, args.model_random_state,
            )
            stats.update({"roi": roi, "metric": metric})
            inference_rows.append(stats)
    inference_frame = _with_bh_qvalues(pd.DataFrame(inference_rows))

    contrast_rois = ("cca_a", "cca_p")
    contrast_frame = None
    if all(name in roi_columns for name in contrast_rois):
        left = block_frame.loc[block_frame["roi"] == contrast_rois[0]].set_index("video_id")
        right = block_frame.loc[block_frame["roi"] == contrast_rois[1]].set_index("video_id")
        common_blocks = left.index.intersection(right.index)
        contrast_rows = []
        for metric in metric_names:
            diff = (left.loc[common_blocks, metric] - right.loc[common_blocks, metric]).to_numpy(dtype=float)
            stats = _resample_stats(
                diff, args.n_bootstrap, args.n_permutations, args.model_random_state
            )
            stats.update({"roi_a": contrast_rois[0], "roi_b": contrast_rois[1], "metric": metric})
            contrast_rows.append(stats)
        contrast_frame = _with_bh_qvalues(pd.DataFrame(contrast_rows))

    dimensionality_rows = []
    for roi, columns in roi_columns.items():
        dimensionality_rows.append({
            "roi": roi,
            "n_roi_targets": int(len(columns)),
            "n_samples": int(n_samples),
            "audio_predicted_dim": participation_ratio(oof["A"][:, columns]),
            "video_predicted_dim": participation_ratio(oof["V"][:, columns]),
        })
    dimensionality_frame = pd.DataFrame(dimensionality_rows)

    output_root = (
        Path(args.output_dir) / args.subject / args.model /
        f"roi_av_profile_{args.band_config}_bin{args.bin_sec:g}s_skip{args.skip_sec:g}s"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    block_frame.to_csv(output_root / "block_metrics.csv", index=False)
    inference_frame.to_csv(output_root / "inference.csv", index=False)
    if contrast_frame is not None:
        contrast_frame.to_csv(output_root / "region_contrast_cca_a_vs_cca_p.csv", index=False)
    dimensionality_frame.to_csv(output_root / "effective_dimensionality.csv", index=False)
    (output_root / "rois.json").write_text(json.dumps(
        {name: target_indices[columns].tolist() for name, columns in roi_columns.items()}, indent=2
    ) + "\n")
    (output_root / "manifest.json").write_text(json.dumps({
        "subject": args.subject,
        "model": args.model,
        "band_config": args.band_config,
        "band_a_modality": band_a_modality,
        "band_b_modality": band_b_modality,
        "joint_modality": JOINT_MODALITY,
        "runs": [str(run_id) for run_id in folds],
        "excluded_video_ids": sorted(excluded),
        "inference_unit": "movie block (one non-validation video segment)",
        "n_blocks": len(blocks),
        "blocks": [str(block) for block in blocks],
        "rois": {name: int(len(columns)) for name, columns in roi_columns.items()},
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec, "delay_sec": args.delay_sec,
        "tr": args.tr, "hrf": args.hrf,
        "alphas": alphas.tolist(), "n_iter": args.n_iter, "backend": args.backend,
        "model_random_state": args.model_random_state,
        "n_bootstrap": args.n_bootstrap, "n_permutations": args.n_permutations,
        "multiplicity": "Benjamini-Hochberg across roi x metric in this invocation",
        "arguments": vars(args),
    }, indent=2) + "\n")
    log.info("Saved ROI AV profile to %s", output_root)
    return output_root


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
