# vendor/ — DEPRECATED

This directory contains verbatim copies of code from [vicsompy](https://github.com/nicholashedger/vicsompy)
(Hedger et al. 2025, MIT License) that were previously used when direct imports were not possible.

## Status: no longer imported

As of the current pipeline refactor, **all code is imported directly from the vicsompy source
repository** without `pip install` (using `sys.path` injection with `VICSOMPY_REPO`).

- `vendor/hedger_cf/subsurface.py` — superseded by `from vicsompy.surface import Subsurface`
  (used in `01_extract_geometry.py`)
- `vendor/hedger_cf/utils.py` — superseded by `from vicsompy.utils import generate_leave_one_run_out`
  (used in `shared/ridge_utils.py`)

## Why kept

The files are retained for:
1. Attribution and provenance transparency (see `vendor/hedger_cf/ATTRIBUTION.md`)
2. Fallback reference if the source repo path changes

## Do not import from vendor/

All pipeline scripts (`01_extract_geometry.py`, `02_fit_cf_model.py`, `shared/ridge_utils.py`)
now import from vicsompy directly.  Do not add new imports from this directory.
