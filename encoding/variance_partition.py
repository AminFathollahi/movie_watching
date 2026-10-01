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
from encoding.shared.encoding_utils import save_cifti_maps
from encoding.shared.fold_evaluator import (
    ALL_SUBSETS,
    _pearson_per_target,
    _r2_per_target,
    partition_variance,
)

log = logging.getLogger(__name__)


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
    return f"encoding_{metric}_{args.split}_{args.feature_scaling}" + (f"_{args.tag}" if args.tag else "")


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
    }
    unique = {key: value for key, value in regions.items() if key.startswith("unique")}
    overlaps = {key: value for key, value in regions.items() if key.startswith("shared")}
    return {**unique, **gains, **overlaps}


def run(args) -> Path:
    subsets = tuple(key for key in ALL_SUBSETS if key in args.subsets)
    data = load_inputs(args)
    targets, keep = data["targets"], data["keep"]
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

    r2 = {key: _r2_per_target(oof_y[tested], oof[key][tested]).astype(np.float32) for key in subsets}
    pearson = {key: _pearson_per_target(oof_y[tested], oof[key][tested]).astype(np.float32) for key in subsets}
    full = set(subsets) == set(ALL_SUBSETS)
    tag = f"delay{args.delay_sec:.0f}s_bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s"
    out = Path(args.output_dir) / args.subject / model_dir(args) / tag
    out.mkdir(parents=True, exist_ok=True)
    stem_r2 = stem(args, "r2")
    maps = [
        (stem_r2, "models", {f"r2_{key}": value for key, value in r2.items()}),
        (stem(args, "pearson_r"), "models", {f"r_{key}": value for key, value in pearson.items()}),
    ]
    if full:
        maps.append((stem_r2, "partition", partition_maps(r2)))
    for name, part, values in maps:
        save_cifti_maps(values, args.template_cifti, str(out / f"{name}_{part}.dscalar.nii"))
    np.savez_compressed(
        out / f"{stem_r2}_per_fold.npz", folds=np.asarray(folds),
        **{key: np.stack(values) for key, values in per_fold.items()},
        **{key: np.stack(values) for key, values in fitted.items()}, **extra,
    )
    (out / f"{stem_r2}_provenance.json").write_text(json.dumps({
        "split": args.split,
        "feature_scaling": args.feature_scaling,
        "folds": folds,
        "n_train_pool": int(keep.sum()),
        "n_test_rows": int(tested.sum()),
        "repeated_video_ids": sorted(data["excluded"]),
        "audio_model": data["audio_model"],
        "video_model": data["video_model"],
        "tag": args.tag,
        "joint_models": {str(k): v for k, v in data["joint_names"].items()},
        "subsets": list(subsets),
        "r2": "pooled held-out 1 - SSR/SST per vertex, negatives retained",
        "pearson_r": "pooled held-out Pearson correlation per vertex",
        "normalization": "embeddings: training-row mean (and standard deviation if zscore); responses: z-scored per run, held-out clips on their own statistics",
        "partition": (
            "inclusion-exclusion over the seven subset R2 values; j_minus_av = R2(j) - R2(av)"
            if full else None
        ),
        "n_iter": args.n_iter,
        "model_random_state": args.model_random_state,
        "alphas": np.logspace(args.alpha_min, args.alpha_max, args.n_alphas).tolist(),
    }, indent=2) + "\n")
    log.info("Saved %s variance partition to %s", args.split, out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
