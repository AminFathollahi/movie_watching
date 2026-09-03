#!/usr/bin/env python3
"""Add LBOE qualifiers to fitted CF trees; dry-run and refuse collisions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from cf_naming import lboe_count, qualify_lboe
except ModuleNotFoundError:  # package-style imports in tests and notebooks
    from cf_modeling.cf_naming import lboe_count, qualify_lboe


TEXT_SUFFIXES = {".csv", ".json", ".log", ".md", ".sh", ".txt", ".yaml", ".yml"}
CIFTI_SUFFIXES = (".dscalar.nii", ".dlabel.nii")
MANIFEST = "lboe_naming_migration_manifest.json"


@dataclass(frozen=True)
class PairPlan:
    source: str
    destination: str
    roi_a: str
    roi_b: str
    qualified_roi_a: str
    qualified_roi_b: str
    requested_lboe: int


def _is_cifti(path: Path) -> bool:
    return path.name.endswith(CIFTI_SUFFIXES)


def _roi_names_from_root(root: Path) -> tuple[str, str]:
    """Recover the two exact ROI tokens without guessing at underscores."""
    name = root.name
    if name.startswith("cca_a") and "_cca_p" in name:
        boundary = name.index("_cca_p")
        return name[:boundary], name[boundary + 1:]

    candidates = []
    prep = root / "prep"
    if prep.is_dir():
        for path in prep.glob("R2_*_nc.npy"):
            token = path.name[len("R2_"):-len("_nc.npy")]
            if token not in {"full", "product_map"}:
                candidates.append(token)
    if len(set(candidates)) < 2:
        subsurfaces = root / "subsurfaces"
        if subsurfaces.is_dir():
            for path in subsurfaces.glob("sub_*.pkl"):
                candidates.append(path.stem[len("sub_"):])
    candidates = sorted(set(candidates), key=len, reverse=True)
    for first in candidates:
        for second in candidates:
            if first != second and name == f"{first}_{second}":
                return first, second
    raise ValueError(f"Cannot recover the two ROI names from {root}")


def plan_migration(output_base: Path, default_lboe: int = 200) -> list[PairPlan]:
    output_base = output_base.resolve()
    plans = []
    for mode in ("group_average", "per_subject"):
        mode_dir = output_base / mode
        if not mode_dir.is_dir():
            continue
        for root in sorted(path for path in mode_dir.iterdir() if path.is_dir()):
            roi_a, roi_b = _roi_names_from_root(root)
            count_a, count_b = lboe_count(roi_a), lboe_count(roi_b)
            if count_a is not None or count_b is not None:
                if count_a is None or count_b is None or count_a != count_b:
                    raise ValueError(f"Inconsistent LBOE qualifiers in {root}")
                continue
            qa, qb = qualify_lboe(roi_a, default_lboe), qualify_lboe(roi_b, default_lboe)
            destination = root.with_name(f"{qa}_{qb}")
            plans.append(PairPlan(
                str(root), str(destination), roi_a, roi_b, qa, qb, default_lboe))
    return plans


def _replace(value: str, plan: PairPlan) -> str:
    """Rewrite analysis names while preserving LBOE-independent mask paths."""
    replacements = sorted(
        ((plan.roi_a, plan.qualified_roi_a), (plan.roi_b, plan.qualified_roi_b)),
        key=lambda pair: len(pair[0]), reverse=True)
    protected = {}
    for index, (old, _) in enumerate(replacements):
        for suffix in ("_mask.dscalar.nii", "_L_mask.csv", "_R_mask.csv"):
            original = f"/masks/{old}{suffix}"
            marker = f"/__LBOE_SOURCE_MASK_{index}_{suffix.replace('/', '_')}__"
            if original in value:
                value = value.replace(original, marker)
                protected[marker] = original
    value = value.replace(plan.source, plan.destination)
    for old, new in replacements:
        value = value.replace(old, new)
    for marker, original in protected.items():
        value = value.replace(marker, original)
    return value


def _rewrite_cifti(path: Path, plan: PairPlan) -> bool:
    image = nib.load(path)
    axes = []
    changed = False
    for index in range(len(image.shape)):
        axis = image.header.get_axis(index)
        if isinstance(axis, nib.cifti2.ScalarAxis):
            names = [_replace(str(name), plan) for name in axis.name]
            changed |= names != list(axis.name)
            axis = nib.cifti2.ScalarAxis(names, meta=[dict(item) for item in axis.meta])
        elif isinstance(axis, nib.cifti2.LabelAxis):
            names = [_replace(str(name), plan) for name in axis.name]
            labels = [{
                int(key): (_replace(str(value[0]), plan), tuple(value[1]))
                for key, value in table.items()
            } for table in axis.label]
            changed |= names != list(axis.name) or labels != list(axis.label)
            axis = nib.cifti2.LabelAxis(
                names, labels, meta=[dict(item) for item in axis.meta])
        axes.append(axis)
    if not changed:
        return False
    header = nib.cifti2.Cifti2Header.from_axes(axes)
    rewritten = nib.Cifti2Image(
        np.asanyarray(image.dataobj), header=header, nifti_header=image.nifti_header)
    with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".nii", dir=path.parent,
            delete=False) as handle:
        temporary = Path(handle.name)
    try:
        nib.save(rewritten, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _rewrite_metadata(path: Path, plan: PairPlan) -> bool:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    rewritten = _replace(text, plan)
    if rewritten == text:
        return False
    path.write_text(rewritten, encoding="utf-8")
    return True


def apply_plan(plan: PairPlan) -> dict:
    source, destination = Path(plan.source), Path(plan.destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    os.replace(source, destination)

    renamed_paths = 1
    mappings = sorted(
        ((plan.roi_a, plan.qualified_roi_a), (plan.roi_b, plan.qualified_roi_b)),
        key=lambda pair: len(pair[0]), reverse=True)
    for path in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        new_name = path.name
        for old, new in mappings:
            new_name = new_name.replace(old, new)
        if new_name != path.name:
            target = path.with_name(new_name)
            if target.exists():
                raise FileExistsError(target)
            os.replace(path, target)
            renamed_paths += 1

    metadata_rewritten = 0
    cifti_rewritten = 0
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if _is_cifti(path):
            cifti_rewritten += int(_rewrite_cifti(path, plan))
        elif path.suffix.lower() in TEXT_SUFFIXES:
            metadata_rewritten += int(_rewrite_metadata(path, plan))
    return {
        "plan": asdict(plan),
        "status": "migrated",
        "renamed_paths": renamed_paths,
        "metadata_files_rewritten": metadata_rewritten,
        "cifti_files_rewritten": cifti_rewritten,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_base", type=Path)
    parser.add_argument("--default-lboe", type=int, default=200,
                        help="Requested count assigned to historical unqualified fits")
    parser.add_argument("--apply", action="store_true",
                        help="Apply the in-place migration; default is a dry run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = plan_migration(args.output_base, args.default_lboe)
    for plan in plans:
        print(f"{plan.source} -> {plan.destination}")
    if not args.apply:
        print(f"Dry run: {len(plans)} pair trees; no files changed.")
        return
    records = []
    for plan in plans:
        records.append(apply_plan(plan))
    manifest = {
        "scope": "fitted group_average/per_subject pair trees only",
        "source_masks": "deliberately LBOE-independent and not renamed",
        "default_historical_requested_lboe": args.default_lboe,
        "records": records,
    }
    path = args.output_base.resolve() / MANIFEST
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Migrated {len(records)} pair trees; manifest: {path}")


if __name__ == "__main__":
    main()
