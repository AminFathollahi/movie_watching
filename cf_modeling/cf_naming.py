"""Name LBOE-specific CF analyses without renaming source ROI masks."""

from __future__ import annotations

import re


LBOE_SUFFIX = re.compile(r"_lboe(?P<count>[1-9][0-9]*)$")


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
