"""Append descriptive AV comparison maps to a group-average AV CIFTI.

For each natural joint-AV group-average searchlight result, three audio/video
reference families are used:

``own``
    PE-AV/CAV-MAE cls-a and cls-v maps, or the matching Omni-family
    pre-fusion encoder A/V maps.
``unimodal``
    AudioMAE and VideoMAEv2 maps.
``text_aligned``
    WavLM and PE-Core maps.

Three maps are made per family (nine total):

* ``av_conjunction_<family>`` is a binary map for
  ``(AV > 0) & (AV > A) & (AV > V)``.
* ``av_superadditivity_<family>`` is the signed continuous contrast
  ``AV - (A + V)``.
* ``av_max_uni_<family>`` is the signed continuous contrast
  ``AV - max(A, V)``.

All inputs are ordinary group-average searchlight rho ``.npy`` maps computed
with exactly the same preprocessing, searchlight, delay, bin/stride, and RSA
method. The maps are added to the target model's normal combined AV dscalar.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import get_combined_map_names, merge_into_combined  # noqa: E402
from rsa.shared.model_registry import av_derived_baselines  # noqa: E402


log = logging.getLogger(__name__)

FAMILY_ORDER = ("own", "unimodal", "text_aligned")


def expected_map_names() -> list[str]:
    """Return the nine scalar-axis names in their append order."""
    return [
        *(f"av_conjunction_{family}" for family in FAMILY_ORDER),
        *(f"av_superadditivity_{family}" for family in FAMILY_ORDER),
        *(f"av_max_uni_{family}" for family in FAMILY_ORDER),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-model")
    target.add_argument(
        "--all-existing",
        action="store_true",
        help="Backfill every supported natural *_av combined CIFTI matching the "
             "selected config/rho filename under --rsa-root.",
    )
    parser.add_argument(
        "--list-dependencies",
        action="store_true",
        help="Print unique 'model modality' dependencies for missing maps and exit. "
             "When --combined-output is supplied, families whose two maps already "
             "exist are omitted.",
    )
    parser.add_argument(
        "--rsa-root",
        help="Group-average RSA root containing {model}_{modality}/{config}/.",
    )
    parser.add_argument("--config", help="Searchlight config directory name.")
    parser.add_argument(
        "--rho-filename",
        help="Searchlight rho .npy filename shared by target and baselines.",
    )
    parser.add_argument(
        "--combined-output",
        help="Target model's normal combined AV .dscalar.nii file.",
    )
    parser.add_argument(
        "--template-cifti",
        help="CIFTI defining the BrainModelAxis if combined-output is new.",
    )
    return parser.parse_args()


def _target_baselines(
    target_model: str,
) -> dict[str, tuple[tuple[str, str], tuple[str, str]]]:
    baselines = av_derived_baselines(target_model)
    if baselines is None:
        raise ValueError(
            f"{target_model!r} is not a supported natural joint-AV target"
        )
    return baselines


def missing_map_names(combined_output: Path) -> list[str]:
    """Return expected names absent from an existing (or new) combined CIFTI."""
    existing = set(get_combined_map_names(combined_output))
    return [name for name in expected_map_names() if name not in existing]


def dependency_pairs(
    target_model: str,
    maps_needed: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return unique A/V RSA dependencies for absent derived maps."""
    baselines = av_derived_baselines(target_model)
    if baselines is None:
        return []
    if maps_needed is None:
        families_needed = set(FAMILY_ORDER)
    else:
        families_needed = {
            family for family in FAMILY_ORDER
            if any(name.endswith(f"_{family}") for name in maps_needed)
        }
    pairs: list[tuple[str, str]] = []
    for family in FAMILY_ORDER:
        if family not in families_needed:
            continue
        for pair in baselines[family]:
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def compute_derived_maps(
    av: np.ndarray,
    baseline_maps: dict[str, tuple[np.ndarray, np.ndarray]],
    families: tuple[str, ...] = FAMILY_ORDER,
) -> dict[str, np.ndarray]:
    """Compute selected-family maps from aligned 1-D AV/A/V rho arrays."""
    av = np.asarray(av, dtype=np.float32)
    if av.ndim != 1:
        raise ValueError(f"AV map must be 1-D, got shape {av.shape}")

    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family in families:
        if family not in FAMILY_ORDER:
            raise ValueError(f"Unknown baseline family: {family}")
        if family not in baseline_maps:
            raise KeyError(f"Missing baseline family: {family}")
        audio, video = (
            np.asarray(arr, dtype=np.float32) for arr in baseline_maps[family]
        )
        if audio.shape != av.shape or video.shape != av.shape:
            raise ValueError(
                f"{family} map shape mismatch: AV={av.shape}, "
                f"A={audio.shape}, V={video.shape}"
            )
        normalized[family] = (audio, video)

    result: dict[str, np.ndarray] = {}
    for family in families:
        audio, video = normalized[family]
        finite = np.isfinite(av) & np.isfinite(audio) & np.isfinite(video)
        result[f"av_conjunction_{family}"] = (
            finite & (av > 0) & (av > audio) & (av > video)
        ).astype(np.float32)

    for family in families:
        audio, video = normalized[family]
        result[f"av_superadditivity_{family}"] = (
            av - (audio + video)
        ).astype(np.float32)

    for family in families:
        audio, video = normalized[family]
        result[f"av_max_uni_{family}"] = (
            av - np.maximum(audio, video)
        ).astype(np.float32)

    return result


def _rho_path(
    rsa_root: Path,
    model: str,
    modality: str,
    config: str,
    rho_filename: str,
) -> Path:
    return rsa_root / f"{model}_{modality}" / config / rho_filename


def append_derived_maps(
    *,
    target_model: str,
    rsa_root: Path,
    config: str,
    rho_filename: str,
    combined_output: Path,
    template_cifti: Path,
) -> list[str]:
    """Load matching rho maps and append only absent derived CIFTI maps."""
    baselines = _target_baselines(target_model)
    maps_needed = missing_map_names(combined_output)
    for map_name in expected_map_names():
        if map_name not in maps_needed:
            log.info("  %-39s already present; skipping", map_name)
    if not maps_needed:
        log.info("All nine AV derived maps already exist; nothing to do")
        return []

    families_needed = tuple(
        family for family in FAMILY_ORDER
        if any(name.endswith(f"_{family}") for name in maps_needed)
    )
    av_path = _rho_path(rsa_root, target_model, "av", config, rho_filename)
    source_paths = {"target AV": av_path}
    for family in FAMILY_ORDER:
        if family not in families_needed:
            continue
        (audio_model, audio_mod), (video_model, video_mod) = baselines[family]
        source_paths[f"{family} A"] = _rho_path(
            rsa_root, audio_model, audio_mod, config, rho_filename
        )
        source_paths[f"{family} V"] = _rho_path(
            rsa_root, video_model, video_mod, config, rho_filename
        )

    missing = [(label, path) for label, path in source_paths.items() if not path.is_file()]
    if missing:
        detail = "\n".join(f"  {label}: {path}" for label, path in missing)
        raise FileNotFoundError(f"Missing RSA rho map(s):\n{detail}")

    av = np.load(av_path).astype(np.float32)
    baseline_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family in families_needed:
        baseline_maps[family] = (
            np.load(source_paths[f"{family} A"]).astype(np.float32),
            np.load(source_paths[f"{family} V"]).astype(np.float32),
        )

    maps = compute_derived_maps(av, baseline_maps, families_needed)
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    for map_name, data in maps.items():
        if map_name not in maps_needed:
            continue
        merge_into_combined(
            data,
            map_name,
            combined_output,
            str(template_cifti),
        )
        if map_name.startswith("av_conjunction_"):
            log.info("  %-39s %d vertices", map_name, int(data.sum()))
        else:
            log.info("  %-39s mean=%+.5f", map_name, float(np.nanmean(data)))
    return maps_needed


def main() -> None:
    args = parse_args()

    if args.list_dependencies:
        if args.all_existing:
            raise SystemExit("--list-dependencies requires --target-model")
        maps_needed = (
            missing_map_names(Path(args.combined_output))
            if args.combined_output else None
        )
        for model, modality in dependency_pairs(args.target_model, maps_needed):
            print(model, modality)
        return

    required = ["rsa_root", "config", "rho_filename", "template_cifti"]
    if not args.all_existing:
        required.append("combined_output")
    missing_args = [f"--{name.replace('_', '-')}" for name in required if not getattr(args, name)]
    if missing_args:
        raise SystemExit(f"Missing required build arguments: {', '.join(missing_args)}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rsa_root = Path(args.rsa_root)
    template_cifti = Path(args.template_cifti)
    if args.all_existing:
        suffix = "_searchlight.npy"
        if not args.rho_filename.endswith(suffix):
            raise SystemExit(f"--rho-filename must end with {suffix!r}")
        combined_name = args.rho_filename.removesuffix(suffix) + "_maps.dscalar.nii"
        targets = []
        for combined_output in sorted(rsa_root.glob(f"*_av/{combined_name}")):
            model = combined_output.parent.name.removesuffix("_av")
            if av_derived_baselines(model) is not None:
                targets.append((model, combined_output))
        if not targets:
            raise FileNotFoundError(
                f"No supported AV combined CIFTIs named {combined_name!r} under {rsa_root}"
            )
        total_added = 0
        for model, combined_output in targets:
            log.info("AV derived-map backfill: %s", model)
            total_added += len(append_derived_maps(
                target_model=model,
                rsa_root=rsa_root,
                config=args.config,
                rho_filename=args.rho_filename,
                combined_output=combined_output,
                template_cifti=template_cifti,
            ))
        log.info(
            "Backfill complete: %d target CIFTIs, %d maps appended",
            len(targets), total_added,
        )
    else:
        names = append_derived_maps(
            target_model=args.target_model,
            rsa_root=rsa_root,
            config=args.config,
            rho_filename=args.rho_filename,
            combined_output=Path(args.combined_output),
            template_cifti=template_cifti,
        )
        log.info("Added %d missing AV derived maps in %s", len(names), args.combined_output)


if __name__ == "__main__":
    main()
