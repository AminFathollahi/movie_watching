"""Inputs and split logic shared by the seven-model variance partition."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from encoding.shared.encoding_utils import apply_hrf_to_segment, build_fmri_arrays, spm_hrf
from encoding.shared.fold_evaluator import ALL_SUBSETS, FEATURE_SCALINGS, evaluate_split

log = logging.getLogger(__name__)
REPEATED_VALIDATION_CLIPS = ("video5", "video9", "video14", "video18")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--fmri-suffix", default="raw")
    parser.add_argument("--subject", default="group_average")
    parser.add_argument(
        "--split", choices=("fixed", "runwise"), default="runwise",
        help="fixed: train on the other clips, test on the repeated clips; "
             "runwise: drop the repeated clips, outer leave-one-run-out.",
    )
    parser.add_argument(
        "--feature-scaling", choices=FEATURE_SCALINGS, default="center",
        help="zscore: subtract the training mean and divide by the training standard deviation "
             "(Hedger et al.); center: subtract the training mean only (Gallant-lab tutorials).",
    )
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--model", help="Model supplying J (and A and V unless overridden).")
    parser.add_argument(
        "--subsets", nargs="+", choices=ALL_SUBSETS, default=list(ALL_SUBSETS),
        help="Band subsets to fit; bands not involved are not loaded.",
    )
    parser.add_argument("--output-name")
    parser.add_argument(
        "--joint-model-template",
        help="Joint-feature model name with a {run} placeholder for outer-fold-specific inputs.",
    )
    parser.add_argument("--audio-model", help="Model supplying A; defaults to --model.")
    parser.add_argument("--video-model", help="Model supplying V; defaults to --model.")
    parser.add_argument("--tag", help="Inserted after the scaling in every output file name.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--hrf", action="store_true")
    parser.add_argument(
        "--exclude-video-ids",
        default=",".join(REPEATED_VALIDATION_CLIPS),
        help="Clips excluded from all fitting and main evaluation.",
    )
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=9.0)
    parser.add_argument("--n-alphas", type=int, default=23)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--model-random-state", type=int, default=0)
    return parser


def sample_metadata(timing: pd.DataFrame, bin_sec: float, skip_sec: float) -> pd.DataFrame:
    rows = []
    for _, clip in timing.iterrows():
        duration = float(clip["duration_sec"])
        n_windows = (
            max(0, int(np.floor((duration - bin_sec) / skip_sec)) + 1)
            if duration >= bin_sec else 0
        )
        for window in range(n_windows):
            rows.append({
                "row_index": len(rows),
                "video_id": str(clip["video_id"]),
                "run_id": clip["run_id"],
                "window_index": window,
                "window_onset_sec": float(clip["onset_sec"]) + window * skip_sec,
            })
    return pd.DataFrame(rows)


def apply_hrf_by_clip(
    embeddings: np.ndarray, metadata: pd.DataFrame, bin_sec: float
) -> np.ndarray:
    output = np.empty_like(embeddings, dtype=np.float64)
    kernel = spm_hrf(bin_sec)
    for _, clip_rows in metadata.groupby("video_id", sort=False):
        indices = clip_rows["row_index"].to_numpy(dtype=int)
        output[indices] = apply_hrf_to_segment(
            np.asarray(embeddings[indices], dtype=np.float64), kernel,
        )
    return output


def reject_global_scramble(*names):
    if any(name and "avscramble" in name.lower() for name in names):
        raise ValueError(
            "Global avscramble embeddings cross run boundaries; use joint features "
            "permuted separately within each training and test partition"
        )


def load_inputs(args) -> dict:
    reject_global_scramble(args.model, args.audio_model, args.video_model, args.joint_model_template)
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, args.bin_sec, args.skip_sec)
    bands = set("".join(args.subsets))
    names = {"a": args.audio_model or args.model, "v": args.video_model or args.model}
    if any(band in bands and not names[band] for band in "av") or ("j" in bands and not args.model):
        raise ValueError("--model (or --audio-model / --video-model) is required for the requested subsets")
    bin_dir = f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"
    root = Path(args.embeddings_dir)
    embeddings = {
        band: np.load(root / names[band] / bin_dir / f"{names[band]}_{band}.npy")
        for band in "av" if band in bands
    }
    joint_names = {}
    if "j" in bands and args.joint_model_template:
        if args.split == "fixed":
            raise ValueError("--joint-model-template is defined per held-out run; use --split runwise")
        joint_names = {
            run: args.joint_model_template.format(run=run)
            for run in sorted(pd.unique(metadata["run_id"]))
        }
    elif "j" in bands:
        joint_names = {"default": args.model}
    joint = {
        run: np.load(root / name / bin_dir / f"{name}_av.npy")
        for run, name in joint_names.items()
    }
    for label, values in [*embeddings.items(), *((f"av_run{r}", v) for r, v in joint.items())]:
        if values.shape[0] != len(metadata):
            raise ValueError(f"{label} has {values.shape[0]} rows; timing describes {len(metadata)}")
    if args.hrf:
        embeddings = {k: apply_hrf_by_clip(v, metadata, args.bin_sec) for k, v in embeddings.items()}
        joint = {k: apply_hrf_by_clip(v, metadata, args.bin_sec) for k, v in joint.items()}

    excluded = {item.strip() for item in args.exclude_video_ids.split(",") if item.strip()}
    if args.split == "fixed" and not excluded:
        raise ValueError("--split fixed needs the repeated clips in --exclude-video-ids")
    keep = ~metadata["video_id"].isin(excluded).to_numpy()
    fmri_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    run_trs_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    y_train, y_test, run_onsets = build_fmri_arrays(
        str(fmri_path), str(run_trs_path), timing, sorted(excluded), args.bin_sec, args.tr,
        delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    if y_train.shape[0] != int(keep.sum()) or y_test.shape[0] != int((~keep).sum()):
        raise ValueError("fMRI rows do not align with embedding rows")

    metadata["keep"] = keep
    metadata = metadata.sort_values(["run_id", "row_index"], kind="stable").reset_index(drop=True)
    keep = metadata["keep"].to_numpy()
    kept_runs = pd.factorize(metadata.loc[keep, "run_id"])[0]
    if not np.array_equal(np.flatnonzero(np.r_[True, np.diff(kept_runs) != 0]), run_onsets):
        raise ValueError("Embedding run boundaries do not match fMRI run boundaries")
    targets = np.empty((len(metadata), y_train.shape[1]), dtype=np.float32)
    targets[keep] = y_train
    targets[~keep] = y_test
    rows = metadata["row_index"].to_numpy(dtype=int)
    if len({values.shape[1] for values in joint.values()}) > 1:
        raise ValueError("Joint embeddings have inconsistent feature dimensions")
    return {
        "a": embeddings["a"][rows] if "a" in embeddings else None,
        "v": embeddings["v"][rows] if "v" in embeddings else None,
        "joint": {run: values[rows] for run, values in joint.items()},
        "targets": targets,
        "metadata": metadata,
        "keep": keep,
        "excluded": excluded,
        "joint_names": joint_names,
        "audio_model": names["a"] if "a" in bands else None,
        "video_model": names["v"] if "v" in bands else None,
    }


def make_folds(data: dict, split: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    keep, run_ids = data["keep"], data["metadata"]["run_id"].to_numpy()
    if split == "fixed":
        return [("fixed", keep, ~keep)]
    return [
        (str(run), keep & (run_ids != run), keep & (run_ids == run))
        for run in pd.unique(run_ids[keep])
    ]


def fit_folds(data: dict, args, subsets: tuple[str, ...]):
    run_ids = data["metadata"]["run_id"].to_numpy()
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    for label, train_mask, test_mask in make_folds(data, args.split):
        joint = None
        if data["joint"]:
            joint = data["joint"].get(int(label) if label.isdigit() else label, data["joint"].get("default"))
            if joint is None:
                raise KeyError(f"No joint embeddings configured for held-out run {label}")
        log.info("Fitting %s split, fold %s", args.split, label)
        yield label, test_mask, evaluate_split(
            data["a"], data["v"], joint, data["targets"], run_ids, train_mask, test_mask,
            alphas, label=label, subsets=subsets, n_iter=args.n_iter, backend=args.backend,
            model_random_state=args.model_random_state, feature_scaling=args.feature_scaling,
        )
