"""Name LBOE-specific CF analyses without renaming source ROI masks."""

from __future__ import annotations

import os
import re
from pathlib import Path


LBOE_SUFFIX = re.compile(r"_lboe(?P<count>[1-9][0-9]*)$")

# External drive per-subject CF outputs were moved to (root SSD is chronically
# near-full; see 2026-09-04 migration). Mirrors OUTPUTS/cf_modeling/ layout.
PERSUBJECT_MOUNT = Path("/media/amin/ADATA HD710 PRO")
PERSUBJECT_SUBPATH = "Research/Representation/Movie/outputs/cf_modeling"


def persubject_output_root() -> Path:
    """Root directory for per-subject CF outputs (external drive, not the SSD).

    Honours ``MOVIE_PERSUBJECT_ROOT`` as an override. Raises if the target
    isn't reachable: writing per-subject outputs through a symlink to an
    unmounted drive would otherwise silently fall back onto the root
    filesystem, which is exactly the near-full-disk failure this move exists
    to prevent.
    """
    override = os.environ.get("MOVIE_PERSUBJECT_ROOT")
    if override:
        root = Path(override)
        if not root.exists():
            raise RuntimeError(
                f"MOVIE_PERSUBJECT_ROOT={root} does not exist. Point it at a "
                "reachable directory.")
        return root
    if not PERSUBJECT_MOUNT.is_mount():
        raise RuntimeError(
            f"External drive not mounted at {PERSUBJECT_MOUNT}. Mount it "
            "before running this (or set MOVIE_PERSUBJECT_ROOT to bypass).")
    root = PERSUBJECT_MOUNT / PERSUBJECT_SUBPATH
    root.mkdir(parents=True, exist_ok=True)
    return root


def lboe_count(name: str) -> int | None:
    """Return the terminal requested LBOE count, if one is present."""
    match = LBOE_SUFFIX.search(name)
    return int(match.group("count")) if match else None


def strip_lboe_suffix(name: str) -> str:
    """Map an analysis ROI name back to its LBOE-independent mask name."""
    return LBOE_SUFFIX.sub("", name)


def qualify_lboe(name: str, count: int) -> str:
    """Append ``_lboeN`` idempotently; reject a conflicting existing tag."""
    if count < 1:
        raise ValueError("LBOE count must be positive")
    existing = lboe_count(name)
    if existing is None:
        return f"{name}_lboe{count}"
    if existing != count:
        raise ValueError(
            f"ROI {name!r} already requests {existing} LBOEs, not {count}")
    return name
