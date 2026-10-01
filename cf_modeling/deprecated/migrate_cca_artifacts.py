#!/usr/bin/env python3
"""Migrate cf_modeling artifacts to anatomically correct CCA names.

The canonical mapping is ``mca_v -> cca_a`` (anterior) and
``mca_d -> cca_p`` (posterior). Two deliberately explicit modes are offered:

* additive copy to a separate destination (the historical safe default);
* manifest-backed in-place rename, intended for the large existing output tree.

Both modes rewrite UTF-8 metadata and CIFTI scalar/label names without changing
numeric map data. In-place migration never overwrites a different file. An
existing equivalent canonical file is retained and the legacy duplicate is
removed; a genuine collision is kept under a ``_legacy_renamed`` suffix and
recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}
CIFTI_SUFFIXES = (".dscalar.nii", ".dlabel.nii")
MANIFEST_NAME = "cca_inplace_migration_manifest.json"


def cca_name(value: str) -> str:
    """Translate legacy MCA tokens while preserving all unrelated suffixes."""
    return (value.replace("MCA_V", "CCA_A")
                 .replace("MCA_D", "CCA_P")
                 .replace("mca_v", "cca_a")
                 .replace("mca_d", "cca_p")
                 .replace("MCA", "CCA")
                 .replace("mca", "cca"))


def _is_cifti(path: Path) -> bool:
    return path.name.endswith(CIFTI_SUFFIXES)


@dataclass(frozen=True)
class Migration:
    source: Path
    destination: Path
    kind: str


@dataclass
class MigrationRecord:
    source: str
    destination: str
    kind: str
    status: str
    note: str = ""


def _kind(path: Path) -> str:
    if _is_cifti(path):
        return "cifti"
    if path.suffix.lower() in TEXT_SUFFIXES:
        return "text"
    return "binary"


def plan_migration(source_root: Path, destination_root: Path) -> list[Migration]:
    """Return deterministic additive-copy operations without touching disk."""
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if destination_root == source_root or source_root in destination_root.parents:
        raise ValueError("Destination must be separate from the source tree")

    operations: list[Migration] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        destination = destination_root.joinpath(*(cca_name(part) for part in relative.parts))
        operations.append(Migration(source, destination, _kind(source)))
    return operations


def plan_in_place(root: Path) -> list[Migration]:
    """Return a snapshot plan for every file whose path or contents may change."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    operations = []
    for source in sorted(path for path in root.rglob("*") if path.is_file()):
        if source.name in {MANIFEST_NAME, "cca_migration_manifest.json"}:
            continue
        relative = source.relative_to(root)
        destination = root.joinpath(*(cca_name(part) for part in relative.parts))
        kind = _kind(source)
        if destination != source or kind in {"text", "cifti"}:
            operations.append(Migration(source, destination, kind))
    return operations


def _canonical_text_bytes(source: Path) -> bytes:
    raw = source.read_bytes()
    try:
        return cca_name(raw.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:
        return raw


def _rewrite_text(source: Path, destination: Path) -> None:
    destination.write_bytes(_canonical_text_bytes(source))


def _canonical_axis(axis):
    if isinstance(axis, nib.cifti2.ScalarAxis):
        meta = [dict(item) for item in axis.meta]
        return nib.cifti2.ScalarAxis(
            [cca_name(str(name)) for name in axis.name], meta=meta)
    if isinstance(axis, nib.cifti2.LabelAxis):
        labels = []
        for table in axis.label:
            labels.append({
                int(key): (cca_name(str(value[0])), tuple(value[1]))
                for key, value in table.items()
            })
        meta = [dict(item) for item in axis.meta]
        return nib.cifti2.LabelAxis(
            [cca_name(str(name)) for name in axis.name], labels, meta=meta)
    return axis


def _rewrite_cifti(source: Path, destination: Path) -> None:
    image = nib.load(source)
    axes = [_canonical_axis(image.header.get_axis(i)) for i in range(len(image.shape))]
    header = nib.cifti2.Cifti2Header.from_axes(axes)
    rewritten = nib.Cifti2Image(
        np.asanyarray(image.dataobj), header=header, nifti_header=image.nifti_header)
    nib.save(rewritten, destination)


def _materialize_canonical(source: Path, destination: Path, kind: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "text":
        _rewrite_text(source, destination)
    elif kind == "cifti":
        _rewrite_cifti(source, destination)
    else:
        shutil.copy2(source, destination)


def apply_migration(operations: list[Migration], destination_root: Path,
                    source_root: Path | None = None) -> None:
    """Copy operations to a new tree; refuse collisions rather than overwrite."""
    destination_root.mkdir(parents=True, exist_ok=True)
    targets = [operation.destination for operation in operations]
    if len(targets) != len(set(targets)):
        raise FileExistsError("Migration has colliding destination paths")
    for operation in operations:
        if operation.destination.exists():
            raise FileExistsError(operation.destination)
    for operation in operations:
        _materialize_canonical(operation.source, operation.destination, operation.kind)

    manifest = {
        "mapping": {"mca_v": "cca_a", "mca_d": "cca_p", "mca": "cca"},
        "source_root": str(source_root.resolve()) if source_root is not None else None,
        "files": [
            {"source": str(operation.source), "destination": str(operation.destination),
             "kind": operation.kind}
            for operation in operations
        ],
    }
    (destination_root / "cca_migration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equivalent(first: Path, second: Path, kind: str) -> bool:
    if kind == "text":
        return _canonical_text_bytes(first) == second.read_bytes()
    if kind == "cifti":
        first_image, second_image = nib.load(first), nib.load(second)
        if first_image.shape != second_image.shape:
            return False
        if not np.array_equal(np.asanyarray(first_image.dataobj),
                              np.asanyarray(second_image.dataobj), equal_nan=True):
            return False
        first_axis = _canonical_axis(first_image.header.get_axis(0))
        second_axis = second_image.header.get_axis(0)
        if type(first_axis) is not type(second_axis):
            return False
        if hasattr(first_axis, "name") and list(first_axis.name) != list(second_axis.name):
            return False
        if isinstance(first_axis, nib.cifti2.LabelAxis):
            return list(first_axis.label) == list(second_axis.label)
        return True
    return first.stat().st_size == second.stat().st_size and _sha256(first) == _sha256(second)


def _conflict_path(destination: Path) -> Path:
    name = destination.name
    for suffix in CIFTI_SUFFIXES:
        if name.endswith(suffix):
            stem, extension = name[:-len(suffix)], suffix
            break
    else:
        stem, extension = destination.stem, destination.suffix
    candidate = destination.with_name(f"{stem}_legacy_renamed{extension}")
    counter = 2
    while candidate.exists():
        candidate = destination.with_name(f"{stem}_legacy_renamed_{counter}{extension}")
        counter += 1
    return candidate


def _atomic_rewrite_in_place(path: Path, kind: str) -> bool:
    """Canonicalize mutable metadata at one path; return whether bytes changed."""
    if kind not in {"text", "cifti"}:
        return False
    temporary_suffix = "".join(path.suffixes) if kind == "cifti" else ".tmp"
    with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=temporary_suffix,
            dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        _materialize_canonical(path, temporary, kind)
        changed = not _equivalent(path, temporary, "binary")
        if changed:
            os.replace(temporary, path)
        else:
            temporary.unlink()
        return changed
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def apply_in_place(root: Path,
                   operations: list[Migration] | None = None) -> list[MigrationRecord]:
    """Apply a collision-safe in-place migration and write a rollback manifest."""
    root = root.resolve()
    operations = plan_in_place(root) if operations is None else operations
    records: list[MigrationRecord] = []

    for operation in operations:
        source, destination, kind = operation.source, operation.destination, operation.kind
        if not source.exists():
            records.append(MigrationRecord(
                str(source), str(destination), kind, "skipped_missing"))
            continue
        try:
            if source == destination:
                changed = _atomic_rewrite_in_place(source, kind)
                records.append(MigrationRecord(
                    str(source), str(destination), kind,
                    "metadata_rewritten" if changed else "unchanged"))
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _equivalent(source, destination, kind):
                    source.unlink()
                    records.append(MigrationRecord(
                        str(source), str(destination), kind, "deduplicated"))
                else:
                    conflict = _conflict_path(destination)
                    _materialize_canonical(source, conflict, kind)
                    source.unlink()
                    records.append(MigrationRecord(
                        str(source), str(conflict), kind, "collision_preserved",
                        f"Canonical destination already existed: {destination}"))
                continue

            if kind in {"text", "cifti"}:
                _materialize_canonical(source, destination, kind)
                source.unlink()
            else:
                os.replace(source, destination)
            records.append(MigrationRecord(
                str(source), str(destination), kind, "renamed"))
        except Exception as error:
            records.append(MigrationRecord(
                str(source), str(destination), kind, "error",
                f"{type(error).__name__}: {error}"))

    for directory in sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    manifest = {
        "mapping": {
            "mca_v": "cca_a (anterior)",
            "mca_d": "cca_p (posterior)",
            "mca": "cca",
        },
        "root": str(root),
        "rollback": "Use each record's source/destination pair; no differing file was overwritten.",
        "summary": {
            status: sum(record.status == status for record in records)
            for status in sorted({record.status for record in records})
        },
        "files": [asdict(record) for record in records],
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any(record.status == "error" for record in records):
        errors = sum(record.status == "error" for record in records)
        raise RuntimeError(
            f"Migration completed with {errors} error(s); inspect {root / MANIFEST_NAME}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing artifact tree")
    parser.add_argument("destination", nargs="?", type=Path,
                        help="Separate destination for additive-copy mode")
    parser.add_argument("--in-place", action="store_true",
                        help="Rename within source instead of copying to a destination")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the operation; otherwise only print the plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_place:
        if args.destination is not None:
            raise SystemExit("Do not provide destination with --in-place")
        operations = plan_in_place(args.source)
    else:
        if args.destination is None:
            raise SystemExit("Additive-copy mode requires destination")
        operations = plan_migration(args.source, args.destination)

    for operation in operations:
        print(f"{operation.source} -> {operation.destination} [{operation.kind}]")
    if not args.apply:
        print("Dry run: no files were changed (pass --apply to proceed).")
        return
    if args.in_place:
        records = apply_in_place(args.source.resolve(), operations)
        print(f"Processed {len(records)} files in {args.source}; manifest: "
              f"{args.source.resolve() / MANIFEST_NAME}")
    else:
        apply_migration(operations, args.destination.resolve(), args.source)
        print(f"Wrote {len(operations)} files to {args.destination}")


if __name__ == "__main__":
    main()
