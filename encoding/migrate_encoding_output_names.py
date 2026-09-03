#!/usr/bin/env python3
"""Migrate legacy encoding artifact names to the explicit naming schema.

The migration renames files and updates CIFTI scalar-map labels. It is
collision-safe: if a destination already exists, it aborts before changing
anything unless the source and destination are the same path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
STEM_RENAMES = {
    "encoding_r2_unique_av": "variance_partition_r2_av_residual_band",
    "encoding_r2_full": "variance_partition_r2_full",
    "encoding_r2_A": "variance_partition_r2_audio_band",
    "encoding_r2_V": "variance_partition_r2_visual_band",
    "encoding_r2_av": "encoding_r2_audiovisual",
    "encoding_r2_a": "encoding_r2_audio",
    "encoding_r2_v": "encoding_r2_visual",
    "encoding_r_av": "encoding_pearson_r_audiovisual",
    "encoding_r_a": "encoding_pearson_r_audio",
    "encoding_r_v": "encoding_pearson_r_visual",
}


def renamed(value: str) -> str:
    for old, new in STEM_RENAMES.items():
        if value == old or value.startswith((old + "_", old + ".")):
            return new + value[len(old):]
    return value


def migration_plan(root: Path) -> list[tuple[Path, Path]]:
    plan = []
    for source in root.rglob("*"):
        if not source.is_file():
            continue
        destination_name = renamed(source.name)
        if destination_name != source.name:
            plan.append((source, source.with_name(destination_name)))
    return sorted(plan)


def rewrite_cifti_labels(path: Path) -> None:
    if not (path.name.endswith(".dscalar.nii") or path.name.endswith(".mask.nii")):
        return
    image = nib.load(str(path))
    index_map = image.header.matrix.get_index_map(0)
    named_maps = list(index_map.named_maps)
    if not named_maps:
        return
    old_names = [named_map.map_name for named_map in named_maps]
    names = [renamed(name) for name in old_names]
    if names == old_names:
        return
    for named_map, name in zip(named_maps, names):
        named_map.map_name = name
    if path.name.endswith(".dscalar.nii"):
        temporary = path.with_name(path.name[:-12] + ".migration-tmp.dscalar.nii")
    else:
        temporary = path.with_name(path.name[:-9] + ".migration-tmp.dscalar.nii")
    temporary.unlink(missing_ok=True)
    nib.save(image, str(temporary))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    parser.add_argument("--quiet", action="store_true", help="omit individual paths")
    args = parser.parse_args()
    plan = migration_plan(args.root)
    collisions = [(src, dst) for src, dst in plan if dst.exists()]
    if collisions:
        details = "\n".join(f"{src} -> {dst}" for src, dst in collisions)
        raise FileExistsError(f"Refusing {len(collisions)} collision(s):\n{details}")
    print(f"Planned renames: {len(plan)}")
    if not args.quiet:
        for source, destination in plan:
            print(f"{source} -> {destination}")
    if not args.apply:
        return
    for source, destination in plan:
        source.rename(destination)
    # Scan the entire tree so an interrupted prior run also repairs labels in
    # files that were renamed just before interruption.
    new_stems = tuple(STEM_RENAMES.values())
    for path in args.root.rglob("*.nii"):
        if path.is_file() and path.name.startswith(new_stems):
            rewrite_cifti_labels(path)
    print(f"Completed renames: {len(plan)}")


if __name__ == "__main__":
    main()
