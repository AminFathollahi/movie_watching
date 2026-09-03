"""Consolidate group-average residualized RSA results for one AV model.

The output contains one searchlight scalar from each complementary analysis:

``partial_correlation``
    Brain/model partial Spearman correlation controlling for AudioMAE and
    VideoMAEv2 RDMs.
``linear_resid``
    RSA of the AV embedding after cross-validated linear residualization
    against AudioMAE and VideoMAEv2 embeddings.
``projection_resid``
    RSA of the AV embedding after removing its per-sample projection onto the
    model's dimension-compatible own A/V embedding span.

Each scalar is appended independently, so interrupted and repeated runs only
load and add maps that are absent from the destination CIFTI.
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
    load_single_map,
    merge_into_combined,
)


log = logging.getLogger(__name__)

MAP_ORDER = ("partial_correlation", "linear_resid", "projection_resid")
SEARCHLIGHT_MAP_NAME = "searchlight_spearman_rho"


def residualized_filename(normal_maps_filename: str) -> str:
    """Return the residual-suite filename corresponding to a normal RSA CIFTI."""
    suffix = "_maps.dscalar.nii"
    if not normal_maps_filename.endswith(suffix):
        raise ValueError(f"Normal maps filename must end with {suffix!r}")
    return normal_maps_filename.removesuffix(suffix) + "_residualized_maps.dscalar.nii"


def source_paths(
    rsa_root: Path,
    model: str,
    config: str,
    normal_maps_filename: str,
) -> dict[str, Path]:
    """Return the three conventional source paths for a model."""
    return {
        "partial_correlation": (
            rsa_root
            / f"{model}_av_partial_corr"
            / config
            / "partial_corr_r_searchlight.dscalar.nii"
        ),
        "linear_resid": (
            rsa_root / f"{model}_av_linear_resid_av" / normal_maps_filename
        ),
        "projection_resid": (
            rsa_root / f"{model}_av_projection_resid_av" / normal_maps_filename
        ),
    }


def output_path(
    rsa_root: Path,
    model: str,
    normal_maps_filename: str,
) -> Path:
    """Return the consolidated residual-suite CIFTI path."""
    return (
        rsa_root
        / f"{model}_av"
        / residualized_filename(normal_maps_filename)
    )


def missing_map_names(destination: Path) -> list[str]:
    """Return residual scalars absent from an existing or new CIFTI."""
    existing = set(get_combined_map_names(destination))
    return [name for name in MAP_ORDER if name not in existing]


def append_residualized_maps(
    *,
    model: str,
    rsa_root: Path,
    config: str,
    normal_maps_filename: str,
    template_cifti: Path,
) -> list[str]:
    """Append only missing residual-analysis scalars and return their names."""
    destination = output_path(rsa_root, model, normal_maps_filename)
    maps_needed = missing_map_names(destination)
    for map_name in MAP_ORDER:
        if map_name not in maps_needed:
            log.info("  %-24s already present; skipping", map_name)
    if not maps_needed:
        log.info("All residualized RSA maps already exist; nothing to do")
        return []

    sources = source_paths(rsa_root, model, config, normal_maps_filename)
    absent = [sources[name] for name in maps_needed if not sources[name].is_file()]
    if absent:
        detail = "\n".join(f"  {path}" for path in absent)
        raise FileNotFoundError(f"Missing residualized RSA source map(s):\n{detail}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    for map_name in maps_needed:
        source = sources[map_name]
        if map_name == "partial_correlation":
            data = load_single_map(source)
        else:
            data = load_named_map(source, SEARCHLIGHT_MAP_NAME)
        merge_into_combined(
            np.asarray(data, dtype=np.float32),
            map_name,
            destination,
            str(template_cifti),
        )
        log.info("  %-24s <- %s", map_name, source)
    return maps_needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--rsa-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--normal-maps-filename", required=True)
    parser.add_argument("--template-cifti", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    added = append_residualized_maps(
        model=args.model,
        rsa_root=Path(args.rsa_root),
        config=args.config,
        normal_maps_filename=args.normal_maps_filename,
        template_cifti=Path(args.template_cifti),
    )
    log.info("Added %d residualized RSA maps for %s", len(added), args.model)


if __name__ == "__main__":
    main()
