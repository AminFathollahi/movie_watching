"""Per-ROI stimulus-category encoding profiles: audio/visual category
variance decomposition, isolated-source audio decoding, and cross-modal
(audio-context) generalization of visual-category decoding.

Manually annotated per-second stimulus regressors (audio: environmental
sounds/music/speech; visual: body/face/animal/place/object) are routed
through the identical bin/skip/delay stimulus-side path used for model
embeddings elsewhere in this repo (sample_metadata + build_fmri_arrays), so
category R2 is directly comparable to the AV embedding R2 reported by
roi_av_profile.py.
"""

from __future__ import annotations

import argparse
import itertools
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

from encoding.incremental_av import REPEATED_VALIDATION_CLIPS, _bh_qvalues, load_target_rois, sample_metadata
from encoding.roi_av_profile import _resample_stats, _with_bh_qvalues
from encoding.shared.encoding_utils import build_fmri_arrays
from encoding.shared.fold_evaluator import _r2_per_target, fit_group_ridge, standardize_bands

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AUDIO_CATEGORIES = ("environmental_sounds", "music", "speech")
ISOLATED_CATEGORIES = ("speech", "music", "environmental_sounds")
# Column order taken from the filename; corroborated (not independently
# confirmed) by matching sums to this analysis's own "face/animal sparse"
# framing -- see the manifest's visual_category_order note.
VISUAL_CATEGORIES = ("body", "face", "animal", "place", "object")
CONJUNCTION_CATEGORIES = tuple(f"{a}_x_{v}" for a in AUDIO_CATEGORIES for v in VISUAL_CATEGORIES)

DEFAULT_LABELS_DIR = "/home/amin/Research/Representation/Movie/data/per_segment_setare_labels"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--fmri-suffix", default="raw")
    parser.add_argument("--subject", default="group_average")
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
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
    parser.add_argument(
        "--n-label-permutations", type=int, default=200,
        help="Permutations for the classification (isolated-audio, cross-modal) nulls; "
             "kept far smaller than --n-permutations because each one refits a classifier.",
    )
    parser.add_argument(
        "--min-cell-n", type=int, default=15,
        help="Minimum bins required to report a decoding/generalization cell.",
    )
    parser.add_argument("--classifier-c", type=float, default=1.0)
    return parser.parse_args(argv)


# =============================================================================
# Label loading and binning
# =============================================================================

def load_raw_labels(labels_dir: str, run_trs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-second audio/visual/isolated-audio regressors, aligned to run_trs order."""
    labels_dir = Path(labels_dir)
    audio_frame = pd.read_excel(
        labels_dir / "full_timecourse_regressors_standardized.xlsx", sheet_name="Regressors"
    ).sort_values(["run_id", "timepoint_in_run"], kind="stable").reset_index(drop=True)
    isolated_frame = pd.read_excel(
        labels_dir / "isolated_speech_music_environment_sounds_standardized_timepoint.xlsx",
        sheet_name="Isolated_Regressors",
    ).sort_values(["run_id", "timepoint_in_run"], kind="stable").reset_index(drop=True)
    visual_raw = np.load(labels_dir / "body_face_animal_place_object.npy").astype(np.float64)

    run_counts = audio_frame.groupby("run_id")["timepoint_in_run"].count().sort_index().to_numpy()
    if not np.array_equal(run_counts, np.asarray(run_trs)):
        raise ValueError(f"Label run TR counts {run_counts.tolist()} != fMRI run_trs {list(run_trs)}")
    if not np.array_equal(isolated_frame["global_timepoint"].to_numpy(), audio_frame["global_timepoint"].to_numpy()):
        raise ValueError("Isolated-regressor sheet is not aligned to the full-timecourse sheet")
    if visual_raw.shape != (len(audio_frame), len(VISUAL_CATEGORIES)):
        raise ValueError(f"body_face_animal_place_object.npy has shape {visual_raw.shape}")

    audio_raw = audio_frame[list(AUDIO_CATEGORIES)].to_numpy(dtype=np.float64)
    isolated_raw = isolated_frame[[f"isolated_{name}" for name in ISOLATED_CATEGORIES]].to_numpy(dtype=np.float64)
    return audio_raw, visual_raw, isolated_raw


def bin_label_windows(raw: np.ndarray, run_trs: np.ndarray, metadata: pd.DataFrame, bin_sec: float, tr: float) -> np.ndarray:
    """Average per-second regressor columns within each stimulus-side window.

    Mirrors the window arithmetic in encoding.shared.encoding_utils._bin_and_split_fmri
    (global onset_sec minus cumulative run-start offset), but skips its per-run
    z-scoring and returns one row per metadata window in metadata['row_index'] order
    so results align 1:1 with model embeddings built from the same metadata.
    """
    bin_trs = max(1, int(round(bin_sec / tr)))
    cum_trs = np.concatenate([[0], np.cumsum(run_trs)]).astype(int)
    n_runs = len(run_trs)
    out = np.full((len(metadata), raw.shape[1]), np.nan, dtype=np.float64)
    for row in metadata.itertuples():
        run_id = int(row.run_id)
        if not (1 <= run_id <= n_runs):
            raise ValueError(f"run_id {run_id} outside 1..{n_runs}")
        run_start_sec = cum_trs[run_id - 1] * tr
        start_tr = int(round((row.window_onset_sec - run_start_sec) / tr))
        global_start = cum_trs[run_id - 1] + start_tr
        if start_tr < 0 or global_start + bin_trs > cum_trs[run_id]:
            raise ValueError(f"window at row {row.row_index} falls outside run {run_id}")
        out[row.row_index] = raw[global_start: global_start + bin_trs].mean(axis=0)
    if np.isnan(out).any():
        raise RuntimeError("Unfilled rows after label binning")
    return out


# =============================================================================
# Part 1: category variance decomposition
# =============================================================================

def decomposition_metrics(r2_audio: float, r2_visual: float, r2_additive: float, r2_full: float) -> dict:
    return {
        "R2_audio": r2_audio, "R2_visual": r2_visual,
        "R2_additive": r2_additive, "R2_full": r2_full,
        "unique_audio": r2_additive - r2_visual,
        "unique_visual": r2_additive - r2_audio,
        "shared": r2_audio + r2_visual - r2_additive,
        "conjunction_gain": r2_full - r2_additive,
        "audio_categorical_dominance": r2_audio - r2_visual,
    }


METRIC_NAMES = (
    "R2_audio", "R2_visual", "R2_additive", "R2_full",
    "unique_audio", "unique_visual", "shared",
    "conjunction_gain", "audio_categorical_dominance",
)


def fit_category_decomposition(audio_bins, visual_bins, conjunction_bins, targets, run_ids, alphas, args):
    folds = sorted(pd.unique(run_ids))
    oof = {tag: np.full(targets.shape, np.nan, dtype=np.float32) for tag in ("audio", "visual", "additive", "full")}
    for test_run in folds:
        log.info("Category decomposition: held-out run %s", test_run)
        test_mask = run_ids == test_run
        train_mask = ~test_mask
        train_runs = run_ids[train_mask]
        std_train, std_test, _ = standardize_bands(
            [audio_bins[train_mask], visual_bins[train_mask], conjunction_bins[train_mask]],
            [audio_bins[test_mask], visual_bins[test_mask], conjunction_bins[test_mask]],
        )
        a_tr, v_tr, c_tr = std_train
        a_te, v_te, c_te = std_test
        y_train, y_test = targets[train_mask], targets[test_mask]
        for tag, train_bands, test_bands in (
            ("audio", [a_tr], [a_te]),
            ("visual", [v_tr], [v_te]),
            ("additive", [a_tr, v_tr], [a_te, v_te]),
            ("full", [a_tr, v_tr, c_tr], [a_te, v_te, c_te]),
        ):
            prediction, _, _, _ = fit_group_ridge(
                train_bands, y_train, test_bands, train_runs, alphas,
                n_iter=args.n_iter, backend=args.backend, random_state=args.model_random_state,
            )
            oof[tag][test_mask] = prediction
    return oof, folds


def category_block_metrics(oof, targets, roi_columns, video_ids) -> pd.DataFrame:
    blocks = sorted(pd.unique(video_ids))
    rows = []
    for roi, columns in roi_columns.items():
        for block in blocks:
            block_mask = video_ids == block
            r2 = {
                tag: float(np.nanmean(_r2_per_target(
                    targets[block_mask][:, columns], oof[tag][block_mask][:, columns]
                )))
                for tag in oof
            }
            record = decomposition_metrics(r2["audio"], r2["visual"], r2["additive"], r2["full"])
            record.update({"roi": roi, "video_id": str(block), "n_block_samples": int(block_mask.sum())})
            rows.append(record)
    return pd.DataFrame(rows)


def block_inference(block_frame: pd.DataFrame, roi_columns, args) -> pd.DataFrame:
    rows = []
    for roi in roi_columns:
        roi_frame = block_frame.loc[block_frame["roi"] == roi]
        for metric in METRIC_NAMES:
            stats = _resample_stats(
                roi_frame[metric].to_numpy(dtype=float),
                args.n_bootstrap, args.n_permutations, args.model_random_state,
            )
            stats.update({"roi": roi, "metric": metric})
            rows.append(stats)
    return _with_bh_qvalues(pd.DataFrame(rows))


def region_contrast(block_frame: pd.DataFrame, roi_columns, args) -> pd.DataFrame | None:
    contrast_rois = ("cca_a", "cca_p")
    if not all(name in roi_columns for name in contrast_rois):
        return None
    left = block_frame.loc[block_frame["roi"] == contrast_rois[0]].set_index("video_id")
    right = block_frame.loc[block_frame["roi"] == contrast_rois[1]].set_index("video_id")
    common = left.index.intersection(right.index)
    rows = []
    for metric in METRIC_NAMES:
        diff = (left.loc[common, metric] - right.loc[common, metric]).to_numpy(dtype=float)
        stats = _resample_stats(diff, args.n_bootstrap, args.n_permutations, args.model_random_state)
        stats.update({"roi_a": contrast_rois[0], "roi_b": contrast_rois[1], "metric": metric})
        rows.append(stats)
    return _with_bh_qvalues(pd.DataFrame(rows))


# =============================================================================
# Parts 2/3: classification helpers
# =============================================================================

def _fit_predict_loro(X, y, run_ids, folds, classifier_c, random_state):
    from sklearn.linear_model import LogisticRegression

    prediction = np.full(len(y), -1, dtype=int)
    for test_run in folds:
        test_mask = run_ids == test_run
        train_mask = ~test_mask
        if test_mask.sum() == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        std_train, std_test, _ = standardize_bands([X[train_mask]], [X[test_mask]])
        clf = LogisticRegression(C=classifier_c, max_iter=2000, random_state=random_state)
        clf.fit(std_train[0], y[train_mask])
        prediction[test_mask] = clf.predict(std_test[0])
    return prediction


def cross_validated_balanced_accuracy(X, y, run_ids, classifier_c, random_state, min_cell_n):
    from sklearn.metrics import balanced_accuracy_score

    folds = sorted(pd.unique(run_ids))
    if len(y) < min_cell_n or len(np.unique(y)) < 2:
        return None
    prediction = _fit_predict_loro(X, y, run_ids, folds, classifier_c, random_state)
    scored = prediction >= 0
    if scored.sum() < min_cell_n or len(np.unique(y[scored])) < 2:
        return None
    return float(balanced_accuracy_score(y[scored], prediction[scored])), int(scored.sum())


def isolated_audio_decoding(targets, roi_columns, isolated_bins, run_ids, args) -> pd.DataFrame:
    from sklearn.metrics import balanced_accuracy_score

    pure = np.isclose(isolated_bins, 1.0)
    onehot = pure.sum(axis=1) == 1
    labels = np.argmax(pure, axis=1)
    labels = np.where(onehot, labels, -1)
    valid = onehot
    class_counts = {
        name: int(((labels == index) & valid).sum())
        for index, name in enumerate(ISOLATED_CATEGORIES)
    }

    rows = []
    for roi, columns in roi_columns.items():
        row = {"roi": roi, "n_bins": int(valid.sum()), **{f"n_{name}": class_counts[name] for name in ISOLATED_CATEGORIES}}
        X = targets[valid][:, columns]
        y = labels[valid]
        rid = run_ids[valid]
        result = cross_validated_balanced_accuracy(X, y, rid, args.classifier_c, args.model_random_state, args.min_cell_n)
        if result is None:
            row.update({
                "balanced_accuracy": np.nan, "n_scored": 0,
                "null_mean": np.nan, "null_ci_low": np.nan, "null_ci_high": np.nan,
                "p_perm": np.nan, "refused": True,
                "reason": f"fewer than {args.min_cell_n} scorable bins or <2 classes",
            })
            rows.append(row)
            continue
        observed, n_scored = result
        rng = np.random.default_rng(args.model_random_state)
        null = []
        for _ in range(args.n_label_permutations):
            y_perm = rng.permutation(y)
            permuted = cross_validated_balanced_accuracy(
                X, y_perm, rid, args.classifier_c, args.model_random_state, min_cell_n=2
            )
            if permuted is not None:
                null.append(permuted[0])
        null = np.asarray(null, dtype=float)
        p_perm = float((1 + np.sum(null >= observed)) / (len(null) + 1)) if len(null) else np.nan
        row.update({
            "balanced_accuracy": observed, "n_scored": n_scored,
            "null_mean": float(null.mean()) if len(null) else np.nan,
            "null_ci_low": float(np.quantile(null, 0.025)) if len(null) else np.nan,
            "null_ci_high": float(np.quantile(null, 0.975)) if len(null) else np.nan,
            "p_perm": p_perm, "refused": False, "reason": "",
        })
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["q_bh"] = np.nan
    defined = ~frame["refused"]
    if defined.any():
        frame.loc[defined, "q_bh"] = _bh_qvalues(frame.loc[defined, "p_perm"])
    return frame


def dominant_visual_labels(visual_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = visual_bins.max(axis=1)
    is_max = visual_bins == max_value[:, None]
    unique_max = is_max.sum(axis=1) == 1
    valid = unique_max & (max_value > 0)
    labels = np.argmax(visual_bins, axis=1)
    return labels, valid


def cross_modal_generalization(targets, roi_columns, isolated_bins, visual_bins, run_ids, args) -> pd.DataFrame:
    pure = np.isclose(isolated_bins, 1.0)
    onehot = pure.sum(axis=1) == 1
    audio_context = np.where(onehot, np.argmax(pure, axis=1), -1)
    visual_labels, visual_valid = dominant_visual_labels(visual_bins)
    joint_valid = onehot & visual_valid

    rows = []
    for roi, columns in roi_columns.items():
        X = targets[joint_valid][:, columns]
        y = visual_labels[joint_valid]
        ctx = audio_context[joint_valid]
        rid = run_ids[joint_valid]

        within = {}
        for context_index in range(len(ISOLATED_CATEGORIES)):
            mask = ctx == context_index
            result = cross_validated_balanced_accuracy(
                X[mask], y[mask], rid[mask], args.classifier_c, args.model_random_state, args.min_cell_n
            )
            within[context_index] = result

        for train_index, test_index in itertools.permutations(range(len(ISOLATED_CATEGORIES)), 2):
            train_mask = ctx == train_index
            test_mask = ctx == test_index
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            row = {
                "roi": roi,
                "train_context": ISOLATED_CATEGORIES[train_index],
                "test_context": ISOLATED_CATEGORIES[test_index],
                "n_train": n_train, "n_test": n_test,
                "n_train_classes": int(len(np.unique(y[train_mask]))) if n_train else 0,
                "n_test_classes": int(len(np.unique(y[test_mask]))) if n_test else 0,
            }
            within_result = within[train_index]
            row["within_context_balanced_accuracy"] = within_result[0] if within_result else np.nan
            if (
                n_train < args.min_cell_n or n_test < args.min_cell_n
                or row["n_train_classes"] < 2 or row["n_test_classes"] < 1
                or within_result is None
            ):
                row.update({
                    "cross_context_balanced_accuracy": np.nan, "generalization_drop": np.nan,
                    "refused": True, "reason": f"below --min-cell-n={args.min_cell_n} or too few classes",
                })
                rows.append(row)
                continue
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score

            std_train, std_test, _ = standardize_bands([X[train_mask]], [X[test_mask]])
            clf = LogisticRegression(C=args.classifier_c, max_iter=2000, random_state=args.model_random_state)
            clf.fit(std_train[0], y[train_mask])
            cross_accuracy = float(balanced_accuracy_score(y[test_mask], clf.predict(std_test[0])))
            row.update({
                "cross_context_balanced_accuracy": cross_accuracy,
                "generalization_drop": within_result[0] - cross_accuracy,
                "refused": False, "reason": "",
            })
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Run
# =============================================================================

def run(args) -> Path:
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, args.bin_sec, args.skip_sec)

    fmri_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    run_trs_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    run_trs = np.load(run_trs_path)

    audio_raw, visual_raw, isolated_raw = load_raw_labels(args.labels_dir, run_trs)
    conjunction_raw = (
        audio_raw[:, :, None] * visual_raw[:, None, :]
    ).reshape(len(audio_raw), len(AUDIO_CATEGORIES) * len(VISUAL_CATEGORIES))
    raw_all = np.concatenate([audio_raw, visual_raw, conjunction_raw, isolated_raw], axis=1)
    label_bins = bin_label_windows(raw_all, run_trs, metadata, args.bin_sec, args.tr)
    if label_bins.shape[0] != len(metadata):
        raise ValueError("Binned label matrix row count does not match stimulus-side metadata")

    n_audio, n_visual, n_conjunction, n_isolated = (
        len(AUDIO_CATEGORIES), len(VISUAL_CATEGORIES), len(CONJUNCTION_CATEGORIES), len(ISOLATED_CATEGORIES),
    )
    audio_bins = label_bins[:, :n_audio]
    visual_bins = label_bins[:, n_audio: n_audio + n_visual]
    conjunction_bins = label_bins[:, n_audio + n_visual: n_audio + n_visual + n_conjunction]
    isolated_bins = label_bins[:, n_audio + n_visual + n_conjunction:]
    assert isolated_bins.shape[1] == n_isolated

    excluded = {item.strip() for item in args.exclude_video_ids.split(",") if item.strip()}
    keep = ~metadata["video_id"].isin(excluded).to_numpy()
    targets, _, _ = build_fmri_arrays(
        str(fmri_path), str(run_trs_path), timing, sorted(excluded), args.bin_sec, args.tr,
        delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    if targets.shape[0] != int(keep.sum()):
        raise ValueError(f"fMRI rows {targets.shape[0]} do not align to {int(keep.sum())} retained rows")

    target_indices, roi_columns = load_target_rois(args, fmri_path, targets.shape[1])
    targets = targets[:, target_indices]

    metadata = metadata.loc[keep].sort_values(["run_id", "row_index"], kind="stable").reset_index(drop=True)
    retained_rows = metadata["row_index"].to_numpy(dtype=int)
    audio_bins = audio_bins[retained_rows]
    visual_bins = visual_bins[retained_rows]
    conjunction_bins = conjunction_bins[retained_rows]
    isolated_bins = isolated_bins[retained_rows]
    if audio_bins.shape[0] != targets.shape[0]:
        raise ValueError("Binned labels and fMRI targets have different sample counts after alignment")

    run_ids = metadata["run_id"].to_numpy()
    video_ids = metadata["video_id"].to_numpy()
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)

    oof, folds = fit_category_decomposition(audio_bins, visual_bins, conjunction_bins, targets, run_ids, alphas, args)
    block_frame = category_block_metrics(oof, targets, roi_columns, video_ids)
    inference_frame = block_inference(block_frame, roi_columns, args)
    contrast_frame = region_contrast(block_frame, roi_columns, args)

    isolated_frame = isolated_audio_decoding(targets, roi_columns, isolated_bins, run_ids, args)
    cross_modal_frame = cross_modal_generalization(targets, roi_columns, isolated_bins, visual_bins, run_ids, args)

    output_root = Path(args.output_dir) / args.subject
    output_root.mkdir(parents=True, exist_ok=True)
    block_frame.to_csv(output_root / "block_metrics.csv", index=False)
    inference_frame.to_csv(output_root / "inference.csv", index=False)
    if contrast_frame is not None:
        contrast_frame.to_csv(output_root / "region_contrast_cca_a_vs_cca_p.csv", index=False)
    isolated_frame.to_csv(output_root / "isolated_audio_decoding.csv", index=False)
    cross_modal_frame.to_csv(output_root / "cross_modal_generalization.csv", index=False)
    (output_root / "rois.json").write_text(json.dumps(
        {name: target_indices[columns].tolist() for name, columns in roi_columns.items()}, indent=2
    ) + "\n")
    (output_root / "manifest.json").write_text(json.dumps({
        "subject": args.subject,
        "runs": [str(run_id) for run_id in folds],
        "excluded_video_ids": sorted(excluded),
        "inference_unit": "movie block (one non-validation video segment)",
        "n_blocks": int(pd.unique(video_ids).size),
        "blocks": [str(block) for block in sorted(pd.unique(video_ids))],
        "rois": {name: int(len(columns)) for name, columns in roi_columns.items()},
        "audio_categories": AUDIO_CATEGORIES,
        "visual_categories": VISUAL_CATEGORIES,
        "visual_category_order": {
            "source": "filename body_face_animal_place_object.npy",
            "confidence": (
                "not independently confirmed against a labeled source; corroborated only by "
                "this analysis's own observation that face and animal are the sparse categories, "
                "which matches per-column TR sums of 219 and 175 respectively under this ordering"
            ),
        },
        "conjunction_categories": CONJUNCTION_CATEGORIES,
        "isolated_categories": ISOLATED_CATEGORIES,
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec, "delay_sec": args.delay_sec, "tr": args.tr,
        "alphas": alphas.tolist(), "n_iter": args.n_iter, "backend": args.backend,
        "model_random_state": args.model_random_state,
        "n_bootstrap": args.n_bootstrap, "n_permutations": args.n_permutations,
        "n_label_permutations": args.n_label_permutations, "min_cell_n": args.min_cell_n,
        "classifier_c": args.classifier_c,
        "multiplicity": "Benjamini-Hochberg across roi x metric in this invocation",
        "arguments": vars(args),
    }, indent=2) + "\n")
    log.info("Saved category profile to %s", output_root)
    return output_root


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
