#!/usr/bin/env python3
"""Append native-AV comparison maps to group-average encoding r_AV CIFTIs.

For a model with matched group-average Pearson-r maps for audio, video, and
joint audiovisual embeddings, this script adds two maps to
``encoding_pearson_r_audiovisual.dscalar.nii``:

* ``av_superadditivity`` = AV - (A + V)
* ``av_conjunction`` = AV inside ``(AV > 0) & (AV > A) & (AV > V)``, else 0

The binary conjunction support is also written as
``encoding_pearson_r_audiovisual_conjunction.mask.nii``.  All comparisons are vertexwise and
require finite AV, A, and V values.

The positive part of the spatial stimulus-regressor map is binarized and
multiplied vertexwise with every AV map.  Thus only stimulus-relevant vertices
survive while their encoding values remain unchanged.  These maps are appended
as ``stim_r``, ``stim_superadditivity``, and ``stim_conjunction``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import (  # noqa: E402
    get_combined_map_names,
    load_named_map,
    merge_into_combined,
    save_cifti_map,
)
from rsa.shared.model_registry import MODELS  # noqa: E402


log = logging.getLogger(__name__)

SUPERADDITIVITY_NAME = "av_superadditivity"
CONJUNCTION_NAME = "av_conjunction"
CONJUNCTION_MASK_NAME = "av_conjunction_mask"
DERIVED_MAP_NAMES = (SUPERADDITIVITY_NAME, CONJUNCTION_NAME)
STIM_MAP_NAMES = ("stim_r", "stim_superadditivity", "stim_conjunction")
TARGET_PREFIXES = ("pe-av", "cav-mae", "omni3b", "topoomni", "nemotron")
DEFAULT_STIMULUS_MAP = (
    ROOT.parent / "outputs" / "sitmulus_regressor_cifti"
    / "HCP_movie_stimulus_correlation_5sdelay_normalized.dscalar.nii"
)


def is_supported_target(model: str) -> bool:
    """Whether a registry model has genuine, matched A/V/AV representations."""
    info = MODELS.get(model)
    if info is None or not model.startswith(TARGET_PREFIXES):
        return False
    return info.get("joint", False) and {"a", "v", "av"}.issubset(
        info.get("modalities", [])
    )


def compute_av_derived_maps(
    av: np.ndarray,
    audio: np.ndarray,
    video: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (superadditivity, conjunction-weighted AV, conjunction mask)."""
    av = np.asarray(av, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    video = np.asarray(video, dtype=np.float32)
    if av.ndim != 1:
        raise ValueError(f"AV map must be 1-D, got {av.shape}")
    if audio.shape != av.shape or video.shape != av.shape:
        raise ValueError(
            f"Encoding-map shape mismatch: AV={av.shape}, "
            f"A={audio.shape}, V={video.shape}"
        )

    finite = np.isfinite(av) & np.isfinite(audio) & np.isfinite(video)
    superadditivity = np.full(av.shape, np.nan, dtype=np.float32)
    superadditivity[finite] = av[finite] - (audio[finite] + video[finite])

    conjunction_mask = finite & (av > 0) & (av > audio) & (av > video)
    conjunction = np.zeros(av.shape, dtype=np.float32)
    conjunction[conjunction_mask] = av[conjunction_mask]
    return superadditivity, conjunction, conjunction_mask.astype(np.float32)


def append_av_derived_maps(
    config_dir: Path,
    *,
    force: bool = False,
    stimulus_map: Path = DEFAULT_STIMULUS_MAP,
) -> list[str]:
    """Append missing derived maps and save the standalone binary mask."""
    paths = {
        "a": config_dir / "encoding_pearson_r_audio.dscalar.nii",
        "v": config_dir / "encoding_pearson_r_visual.dscalar.nii",
        "av": config_dir / "encoding_pearson_r_audiovisual.dscalar.nii",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing encoding map(s):\n  " + "\n  ".join(missing))

    av = load_named_map(paths["av"], "encoding_pearson_r_audiovisual")
    audio = load_named_map(paths["a"], "encoding_pearson_r_audio")
    video = load_named_map(paths["v"], "encoding_pearson_r_visual")
    superadditivity, conjunction, conjunction_mask = compute_av_derived_maps(
        av, audio, video
    )

    existing = set(get_combined_map_names(paths["av"]))
    maps = {
        SUPERADDITIVITY_NAME: superadditivity,
        CONJUNCTION_NAME: conjunction,
    }
    if not stimulus_map.is_file():
        raise FileNotFoundError(f"Missing stimulus-regressor map: {stimulus_map}")
    stimulus_names = get_combined_map_names(stimulus_map)
    if len(stimulus_names) != 1:
        raise ValueError(
            f"Expected one stimulus-regressor map in {stimulus_map}, "
            f"found {stimulus_names}"
        )
    stimulus = np.asarray(
        load_named_map(stimulus_map, stimulus_names[0]), dtype=np.float32
    )
    if stimulus.shape != av.shape:
        raise ValueError(
            f"Stimulus/encoding shape mismatch: stimulus={stimulus.shape}, AV={av.shape}"
        )
    stimulus_positive = (np.isfinite(stimulus) & (stimulus > 0)).astype(np.float32)
    maps.update({
        "stim_r": av * stimulus_positive,
        "stim_superadditivity": superadditivity * stimulus_positive,
        "stim_conjunction": conjunction * stimulus_positive,
    })
    added: list[str] = []
    for map_name, data in maps.items():
        if not force and map_name in existing:
            log.info("  %-24s already present; skipping", map_name)
            continue
        merge_into_combined(data, map_name, paths["av"], str(paths["av"]))
        added.append(map_name)

    mask_path = config_dir / "encoding_pearson_r_audiovisual_conjunction.mask.nii"
    if force or not mask_path.is_file():
        save_cifti_map(
            conjunction_mask,
            str(paths["av"]),
            str(mask_path),
            CONJUNCTION_MASK_NAME,
        )
        log.info("  %-24s %d vertices", CONJUNCTION_MASK_NAME, int(conjunction_mask.sum()))
    else:
        log.info("  %-24s already present; skipping", CONJUNCTION_MASK_NAME)
    return added


def iter_existing_targets(encoding_root: Path, config: str | None):
    """Yield supported (model, config directory) pairs with an AV r map."""
    for model_dir in sorted(path for path in encoding_root.iterdir() if path.is_dir()):
        if not is_supported_target(model_dir.name):
            continue
        config_dirs = [model_dir / config] if config else sorted(
            path for path in model_dir.iterdir() if path.is_dir()
        )
        for config_dir in config_dirs:
            if (config_dir / "encoding_pearson_r_audiovisual.dscalar.nii").is_file():
                yield model_dir.name, config_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoding-root",
        required=True,
        help="Group-average encoding root containing {model}/{config}/.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-model")
    target.add_argument("--all-existing", action="store_true")
    parser.add_argument(
        "--config",
        help="Config directory. Required with --target-model; optional scan filter "
             "with --all-existing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing derived maps and standalone mask.",
    )
    parser.add_argument(
        "--stimulus-map",
        type=Path,
        default=DEFAULT_STIMULUS_MAP,
        help="Spatial stimulus-regressor CIFTI; only its positive values are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    encoding_root = Path(args.encoding_root)

    if args.target_model:
        if not args.config:
            raise SystemExit("--config is required with --target-model")
        if not is_supported_target(args.target_model):
            raise SystemExit(
                f"{args.target_model!r} is not a supported native AV model with "
                "registered a, v, and av modalities"
            )
        targets = [(args.target_model, encoding_root / args.target_model / args.config)]
    else:
        targets = list(iter_existing_targets(encoding_root, args.config))

    if not targets:
        raise FileNotFoundError("No supported group-average AV encoding maps found")

    completed = 0
    skipped_missing = 0
    for model, config_dir in targets:
        log.info("AV derived maps: %s / %s", model, config_dir.name)
        try:
            append_av_derived_maps(
                config_dir, force=args.force, stimulus_map=args.stimulus_map
            )
            completed += 1
        except FileNotFoundError as exc:
            if not args.all_existing:
                raise
            skipped_missing += 1
            log.warning("  SKIP incomplete modalities: %s", str(exc).replace("\n", "; "))

    log.info(
        "Derived-map pass complete: %d target/configs completed, %d incomplete skipped",
        completed,
        skipped_missing,
    )


if __name__ == "__main__":
    main()
