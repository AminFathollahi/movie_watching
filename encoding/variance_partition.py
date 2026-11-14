"""Gallant-lab variance partition of audio (A), video (V) and joint (J) features."""

from __future__ import annotations

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

from encoding.shared.splits import build_parser, fit_folds, load_inputs
from encoding.shared.tags import JOINT_NAMES, pair_tag
from encoding.shared.encoding_utils import load_cifti_maps, save_cifti_maps, write_text_atomic
from encoding.shared.fold_evaluator import ALL_SUBSETS, _pearson_per_target, _r2_per_target

log = logging.getLogger(__name__)
SAVED_PREDICTIONS = {"a", "v", "av", "j", "avj"}
SOURCES = ("audio_model", "video_model", "joint_models")
RESPONSE_NAMES = {"run": "", "train": "_trainscaled", "clip": "_clipdemeaned", "none": "_rawresponse"}
RESPONSE_RULES = {
    "run": "z-scored per run, held-out clips of the fixed split on their own statistics",
    "train": "z-scored per run with the mean and standard deviation of the run's training rows of each fold",
    "clip": "each clip's own mean subtracted",
    "none": "unscaled",
}


def parse_args(argv=None):
    parser = build_parser()
    parser.add_argument("--template-cifti", required=True)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save the pooled held-out predictions of the subsets a, v, av, j and avj, the scaled observed responses and the row table.")
    args = parser.parse_args(argv)
    if not model_dir(args):
        parser.error("--model or --output-name is required for the requested subsets")
    if args.tag is None:
        audio, video = args.audio_model or args.model, args.video_model or args.model
        if not (audio and video):
            parser.error("--tag is required unless both an audio and a video model are given")
        args.tag = pair_tag(audio, video)
    return args


def model_dir(args) -> str | None:
    bands = set("".join(args.subsets))
    if args.output_name:
        return args.output_name
    if bands == {"a"}:
        return args.audio_model or args.model
    if bands == {"v"}:
        return args.video_model or args.model
    return args.model


def joint_name(args) -> str | None:
    """Short name of J for the pair fits of PE-AV and Nemotron, which share the partitions folder; None for single-encoder and renamed fits."""
    return None if args.output_name or set("".join(args.subsets)) in ({"a"}, {"v"}) else JOINT_NAMES.get(args.model)


def stem(args, metric: str) -> str:
    scaling = args.feature_scaling + RESPONSE_NAMES[getattr(args, "response_scaling", "run")]
    scaling += f"_pca{args.n_components}" if getattr(args, "n_components", 0) else ""
    joint = joint_name(args)
    return f"encoding_{metric}_{args.split}_{scaling}" + (f"_{joint}" if joint else "") + (f"_{args.tag}" if args.tag else "")


def clip_r2(y, predictions, clip_ids) -> dict[str, np.ndarray]:
    clips = list(dict.fromkeys(clip_ids))
    out = {"clips": np.asarray(clips)}
    for key, prediction in predictions.items():
        out[f"clip_r2_{key}"] = np.stack([
            _r2_per_target(y[clip_ids == clip], prediction[clip_ids == clip]).astype(np.float32)
            for clip in clips
        ])
    return out


def save_predictions(args, data, oof, oof_y, tested, fold_of, out: Path, stem_r2: str) -> None:
    metadata = data["metadata"]
    run_trs = np.load(Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    run_start = dict(zip(sorted(metadata["run_id"].unique()), np.r_[0, np.cumsum(run_trs[:-1])] * args.tr))
    rows = metadata[tested]
    pd.DataFrame({
        "run": rows["run_id"].to_numpy(), "clip": rows["video_id"].to_numpy(),
        "onset_sec": rows["window_onset_sec"].to_numpy() - rows["run_id"].map(run_start).to_numpy(),
        "fold": fold_of[tested],
    }).to_csv(out / f"{stem_r2}_rows.csv", index=False)
    np.save(out / f"{stem_r2}_observed.npy", oof_y[tested])
    for key in SAVED_PREDICTIONS & oof.keys():
        np.save(out / f"{stem_r2}_predictions_{key}.npy", oof[key][tested])


def output_folder(args) -> Path:
    window = f"delay{args.delay_sec:.0f}s_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    folder = "partitions" if joint_name(args) else model_dir(args)
    return Path(args.output_dir) / args.subject / folder / window


def requested_settings(args) -> dict:
    """Provenance entries the command line determines; a stored fit with different values is not the requested fit."""
    bands = set("".join(args.subsets))
    return {
        "split": args.split,
        "feature_scaling": args.feature_scaling,
        "response_scaling": args.response_scaling,
        "audio_model": (args.audio_model or args.model) if "a" in bands else None,
        "video_model": (args.video_model or args.model) if "v" in bands else None,
        "tag": args.tag,
        "n_iter": args.n_iter,
        "n_components": args.n_components or None,
        "model_random_state": args.model_random_state,
        "alphas": np.logspace(args.alpha_min, args.alpha_max, args.n_alphas).tolist(),
    }


def fit_done(args) -> bool:
    """True when the fit's provenance, written last, holds the requested subsets, settings, window and joint model."""
    out = output_folder(args)
    try:
        old = json.loads((out / f"{stem(args, 'r2')}_provenance.json").read_text())
    except (OSError, ValueError):
        return False
    window = {"bin_sec": args.bin_sec, "skip_sec": args.skip_sec, "delay_sec": args.delay_sec}
    joint = {} if args.joint_model_template or "j" not in "".join(args.subsets) else {"default": args.model}
    return (
        all((out / f"{stem(args, metric)}_models.dscalar.nii").exists() for metric in ("r2", "pearson_r"))
        and set(args.subsets) <= set(old.get("subsets", []))
        and all(old.get(key) == value for key, value in requested_settings(args).items())
        and all(old.get(key, value) == value for key, value in window.items())
        and (args.joint_model_template or "j" not in "".join(args.subsets) or old.get("joint_models") == joint)
    )


def earlier_fit(path: Path, settings: dict, subsets) -> dict:
    """Provenance of subsets already stored under this name that the current fit does not replace."""
    old = json.loads(path.read_text()) if path.exists() else {}
    if set(old.get("subsets", [])) <= set(subsets):
        return {}
    changed = sorted(
        key for key, value in settings.items()
        if old.get(key) != value and not (key in SOURCES and not (old.get(key) and value))
    )
    if changed:
        raise ValueError(
            f"{path.name} holds subsets fitted with different {changed}; "
            "refit all of them or use another --tag"
        )
    return old


def run(args) -> Path:
    subsets = tuple(key for key in ALL_SUBSETS if key in args.subsets)
    data = load_inputs(args)
    targets, keep = data["targets"], data["keep"]
    out = output_folder(args)
    stem_r2 = stem(args, "r2")
    settings = {
        **requested_settings(args),
        "n_train_pool": int(keep.sum()),
        "excluded_video_ids": sorted(data["excluded"]),
        "joint_models": {str(k): v for k, v in data["joint_names"].items()},
    }
    old = earlier_fit(out / f"{stem_r2}_provenance.json", settings, subsets)

    oof = {key: np.full(targets.shape, np.nan, dtype=np.float32) for key in subsets}
    oof_y = np.full(targets.shape, np.nan, dtype=np.float32)
    tested = np.zeros(len(targets), dtype=bool)
    fold_of = np.full(len(targets), "", dtype=object)
    per_fold, folds = {key: [] for key in subsets}, []
    fitted = {f"{kind}_{key}": [] for key in subsets for kind in ("alphas", "deltas")}
    clip_ids = data["metadata"]["video_id"].to_numpy()
    extra = {}
    for label, test_mask, result in fit_folds(data, args, subsets):
        folds.append(label)
        tested[result.test_indices] = True
        oof_y[result.test_indices] = result.y_test
        fold_of[result.test_indices] = str(label)
        for key in subsets:
            oof[key][result.test_indices] = result.predictions[key]
            per_fold[key].append(result.r2[key])
            fitted[f"alphas_{key}"].append(result.arrays[f"{key}_best_alphas"])
            fitted[f"deltas_{key}"].append(result.arrays[f"{key}_deltas"])
        if args.split == "fixed":
            extra = clip_r2(result.y_test, result.predictions, clip_ids[result.test_indices])

    scores = {
        "r2": {key: _r2_per_target(oof_y[tested], oof[key][tested]).astype(np.float32) for key in subsets},
        "r": {key: _pearson_per_target(oof_y[tested], oof[key][tested]).astype(np.float32) for key in subsets},
    }
    stored = tuple(key for key in ALL_SUBSETS if key in {*subsets, *old.get("subsets", [])})
    full = set(stored) == set(ALL_SUBSETS)
    out.mkdir(parents=True, exist_ok=True)
    for prefix, metric in (("r2", "r2"), ("r", "pearson_r")):
        path = str(out / f"{stem(args, metric)}_models.dscalar.nii")
        maps = {f"{prefix}_{key}": value for key, value in scores[prefix].items()}
        if old:
            maps = {**load_cifti_maps(path), **maps}
        maps = {f"{prefix}_{key}": maps[f"{prefix}_{key}"] for key in stored}
        save_cifti_maps(maps, args.template_cifti, path)
    if args.save_predictions:
        save_predictions(args, data, oof, oof_y, tested, fold_of, out, stem_r2)
    fold_path = out / f"{stem_r2}_per_fold.npz"
    np.savez_compressed(fold_path, **{
        **(dict(np.load(fold_path)) if old else {}), "folds": np.asarray(folds),
        **{key: np.stack(values) for key, values in per_fold.items()},
        **{key: np.stack(values) for key, values in fitted.items()}, **extra,
    })
    write_text_atomic(out / f"{stem_r2}_provenance.json", json.dumps({
        **settings,
        "bin_sec": args.bin_sec,
        "skip_sec": args.skip_sec,
        "delay_sec": args.delay_sec,
        **{key: settings[key] or old.get(key) for key in SOURCES},
        "folds": folds,
        "n_test_rows": int(tested.sum()),
        "subsets": list(stored),
        "r2": "pooled held-out 1 - SSR/SST per vertex, negatives retained",
        "pearson_r": "pooled held-out Pearson correlation per vertex",
        "normalization": (
            "responses: " + RESPONSE_RULES[args.response_scaling] + "; embeddings: "
            + ("unscaled" if args.feature_scaling == "none" else
               {"demean": "mean subtracted", "zscore": "mean subtracted and divided by the standard deviation"}[args.feature_scaling]
               + ", over the same rows and statistics as the responses")
            + "; the training mean of the responses is the intercept"
        ),
        "partition": (
            "inclusion-exclusion over the seven subset R2 values; j_minus_av = R2(j) - R2(av); for tags other than "
            "the own-tower pair, own_avj_minus_av = R2(avj) of the own-tower fit with the same split and scaling - R2(av) of this fit; "
            "shared_av = R2(a) + R2(v) - R2(av); own_shared_av_minus_shared_av = shared_av of the own-tower fit - shared_av of this fit; "
            "two-set A-V Venn (J not involved): gain_a_over_v = R2(av) - R2(v), gain_v_over_a = R2(av) - R2(a), and "
            "gain_a_over_v_times_gain_v_over_a = max(gain_a_over_v, 0) * max(gain_v_over_a, 0), positive only where each sense adds held-out R2 beyond the other"
            if full else None
        ),
    }, indent=2) + "\n")
    log.info("Saved %s variance partition to %s", args.split, out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
