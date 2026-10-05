"""Gallant-lab variance partition of audio (A), video (V) and joint (J) features."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from encoding.shared.splits import build_parser, fit_folds, load_inputs
from encoding.shared.encoding_utils import load_cifti_maps, save_cifti_maps
from encoding.shared.fold_evaluator import (
    ALL_SUBSETS,
    _pearson_per_target,
    _r2_per_target,
    partition_variance,
)

log = logging.getLogger(__name__)
SOURCES = ("audio_model", "video_model", "joint_models")
RESPONSE_NAMES = {"run": "", "train": "_trainscaled", "clip": "_clipdemeaned", "none": "_rawresponse"}
OWN_TAG = "unimodal_own"
RESPONSE_RULES = {
    "run": "z-scored per run, held-out clips of the fixed split on their own statistics",
    "train": "z-scored per run with the mean and standard deviation of the run's training rows of each fold",
    "clip": "each clip's own mean subtracted",
    "none": "unscaled",
}


def parse_args(argv=None):
    parser = build_parser()
    parser.add_argument("--template-cifti", required=True)
    args = parser.parse_args(argv)
    if not model_dir(args):
        parser.error("--model or --output-name is required for the requested subsets")
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


def stem(args, metric: str) -> str:
    scaling = args.feature_scaling + RESPONSE_NAMES[getattr(args, "response_scaling", "run")]
    scaling += f"_pca{args.n_components}" if getattr(args, "n_components", 0) else ""
    return f"encoding_{metric}_{args.split}_{scaling}" + (f"_{args.tag}" if args.tag else "")


def clip_r2(y, predictions, clip_ids) -> dict[str, np.ndarray]:
    clips = list(dict.fromkeys(clip_ids))
    out = {"clips": np.asarray(clips)}
    for key, prediction in predictions.items():
        out[f"clip_r2_{key}"] = np.stack([
            _r2_per_target(y[clip_ids == clip], prediction[clip_ids == clip]).astype(np.float32)
            for clip in clips
        ])
    return out


def partition_maps(r2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    regions = {key: value.astype(np.float32) for key, value in partition_variance(r2).items()}
    gains = {
        "gain_j_over_a": r2["aj"] - r2["a"],
        "gain_j_over_v": r2["vj"] - r2["v"],
        "j_minus_av": r2["j"] - r2["av"],
        "shared_av": r2["a"] + r2["v"] - r2["av"],
    }
    unique = {key: value for key, value in regions.items() if key.startswith("unique")}
    overlaps = {key: value for key, value in regions.items() if key.startswith("shared")}
    return {**unique, **gains, **overlaps}


def shared_av(models: dict) -> np.ndarray:
    return models["r2_a"] + models["r2_v"] - models["r2_av"]


def refresh_own_gains(directory: Path, prefix: str) -> list[str]:
    """Add, to every other tag's partition file, own_avj_minus_av = R2(avj) of the unimodal_own fit - R2(av) of this
    fit and own_shared_av_minus_shared_av = shared_av of the unimodal_own fit - shared_av of this fit; summarize the
    second over the control tags (A and V from models other than J's) in {prefix}_own_vs_controls."""
    own = directory / f"{prefix}_{OWN_TAG}_models.dscalar.nii"
    if not own.exists():
        return []
    own_models = load_cifti_maps(str(own))
    updated, contrasts = [], []
    for partition in sorted(directory.glob(f"{prefix}_*_partition.dscalar.nii")):
        tag = partition.name[len(prefix) + 1:-len("_partition.dscalar.nii")]
        models = directory / f"{prefix}_{tag}_models.dscalar.nii"
        provenance_path = directory / f"{prefix}_{tag}_provenance.json"
        if tag == OWN_TAG or not models.exists() or not provenance_path.exists():
            continue
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("tag") != tag:
            continue
        other = load_cifti_maps(str(models))
        maps = load_cifti_maps(str(partition))
        maps["own_avj_minus_av"] = own_models["r2_avj"] - other["r2_av"]
        maps["own_shared_av_minus_shared_av"] = shared_av(own_models) - shared_av(other)
        save_cifti_maps(maps, str(partition), str(partition))
        updated.append(tag)
        joint = provenance["joint_models"].get("default", "")
        if joint and not any(provenance[key].startswith(joint) for key in ("audio_model", "video_model")):
            contrasts.append(maps["own_shared_av_minus_shared_av"])
    if contrasts:
        contrasts = np.stack(contrasts)
        save_cifti_maps({
            "mean_own_shared_av_minus_shared_av": contrasts.mean(0),
            "min_own_shared_av_minus_shared_av": contrasts.min(0),
            "n_controls_own_shared_av_greater": (contrasts > 0).sum(0).astype(np.float32),
        }, str(own), str(directory / f"{prefix}_own_vs_controls.dscalar.nii"))
    return updated


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
    tag = f"delay{args.delay_sec:.0f}s_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    out = Path(args.output_dir) / args.subject / model_dir(args) / tag
    stem_r2 = stem(args, "r2")
    settings = {
        "split": args.split,
        "feature_scaling": args.feature_scaling,
        "response_scaling": args.response_scaling,
        "n_train_pool": int(keep.sum()),
        "repeated_video_ids": sorted(data["excluded"]),
        "audio_model": data["audio_model"],
        "video_model": data["video_model"],
        "tag": args.tag,
        "joint_models": {str(k): v for k, v in data["joint_names"].items()},
        "n_iter": args.n_iter,
        "n_components": args.n_components or None,
        "model_random_state": args.model_random_state,
        "alphas": np.logspace(args.alpha_min, args.alpha_max, args.n_alphas).tolist(),
    }
    old = earlier_fit(out / f"{stem_r2}_provenance.json", settings, subsets)

    oof = {key: np.full(targets.shape, np.nan, dtype=np.float32) for key in subsets}
    oof_y = np.full(targets.shape, np.nan, dtype=np.float32)
    tested = np.zeros(len(targets), dtype=bool)
    per_fold, folds = {key: [] for key in subsets}, []
    fitted = {f"{kind}_{key}": [] for key in subsets for kind in ("alphas", "deltas")}
    clip_ids = data["metadata"]["video_id"].to_numpy()
    extra = {}
    for label, test_mask, result in fit_folds(data, args, subsets):
        folds.append(label)
        tested[result.test_indices] = True
        oof_y[result.test_indices] = result.y_test
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
        if full and prefix == "r2":
            save_cifti_maps(
                partition_maps({key: maps[f"r2_{key}"] for key in ALL_SUBSETS}),
                args.template_cifti, str(out / f"{stem_r2}_partition.dscalar.nii"),
            )
    fold_path = out / f"{stem_r2}_per_fold.npz"
    np.savez_compressed(fold_path, **{
        **(dict(np.load(fold_path)) if old else {}), "folds": np.asarray(folds),
        **{key: np.stack(values) for key, values in per_fold.items()},
        **{key: np.stack(values) for key, values in fitted.items()}, **extra,
    })
    (out / f"{stem_r2}_provenance.json").write_text(json.dumps({
        **settings,
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
            "unimodal_own, own_avj_minus_av = R2(avj) of the unimodal_own fit with the same split and scaling - R2(av) of this fit; "
            "shared_av = R2(a) + R2(v) - R2(av); own_shared_av_minus_shared_av = shared_av of the unimodal_own fit - shared_av of this fit"
            if full else None
        ),
    }, indent=2) + "\n")
    if full:
        refresh_own_gains(out, stem_r2[:-len(f"_{args.tag}")] if args.tag else stem_r2)
    log.info("Saved %s variance partition to %s", args.split, out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
