from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.paths import EXTERNAL  # noqa: E402

LBOE_SUFFIX = re.compile(r"_lboe(?P<count>[1-9][0-9]*)$")

# Per-subject CF outputs are written under this mount, in the layout of OUTPUTS/cf_modeling/.
PERSUBJECT_MOUNT = EXTERNAL
PERSUBJECT_SUBPATH = "outputs/cf_modeling"

ROI_MEAN_PREPROCESSING = {
    "sg_psc_per_run_zscore": {
        "continuous_input": "sg_psc group-average dtseries",
        "continuous_preprocessing": (
            "Per run: Savitzky-Golay detrending, then percent-signal-change "
            "scaling using the pre-detrending run mean; no continuous z-score."),
        "correlation_standardization": "Per-run TR-level z-score of every vertex and ROI mean.",
    },
    "raw_per_run_zscore": {
        "continuous_input": "raw group-average dtseries",
        "continuous_preprocessing": "None (raw continuous BOLD; no SG, PSC, or GSR).",
        "correlation_standardization": "Per-run TR-level z-score of every vertex and ROI mean.",
    },
}

ROI_MEAN_PREPROCESSING_OPTIONS = {
    "raw": "raw_per_run_zscore",
    "sg_psc": "sg_psc_per_run_zscore",
}


def validate_roi_mean_preprocessing(label: str) -> str:
    if label not in ROI_MEAN_PREPROCESSING:
        raise ValueError(
            f"Unknown ROI-mean preprocessing {label!r}; choose one of "
            f"{sorted(ROI_MEAN_PREPROCESSING)}")
    return label


def roi_mean_preprocessing_label(preprocessing: str) -> str:
    if preprocessing not in ROI_MEAN_PREPROCESSING_OPTIONS:
        raise ValueError(
            f"Unknown ROI-mean preprocessing {preprocessing!r}; choose one of "
            f"{sorted(ROI_MEAN_PREPROCESSING_OPTIONS)}")
    return ROI_MEAN_PREPROCESSING_OPTIONS[preprocessing]


def roi_mean_connectivity_names(roi_a: str, roi_b: str, label: str) -> tuple[str, list[str]]:
    stem = f"roi_mean_timeseries_connectivity_{roi_a}_{roi_b}_{label}"
    return stem, [f"roi_mean_timeseries_connectivity_{roi_a}_given_{roi_b}_{label}",
                  f"roi_mean_timeseries_connectivity_{roi_b}_given_{roi_a}_{label}"]


def cf_model_map_stems(roi_a: str, roi_b: str) -> dict[str, str]:
    return {
        "full_r2": "cf_model_full_r2",
        "split_r2_a": f"cf_model_split_r2_{roi_a}",
        "split_r2_b": f"cf_model_split_r2_{roi_b}",
        "roi_mean_null_r2_a": f"cf_model_roi_mean_null_r2_{roi_a}",
        "roi_mean_null_r2_b": f"cf_model_roi_mean_null_r2_{roi_b}",
        "null_corrected_split_r2_a": (
            f"cf_model_null_corrected_split_r2_{roi_a}"),
        "null_corrected_split_r2_b": (
            f"cf_model_null_corrected_split_r2_{roi_b}"),
        "joint_split_r2_geomean": "cf_model_joint_split_r2_geomean",
        "joint_null_corrected_split_r2_geomean": (
            "cf_model_joint_null_corrected_split_r2_geomean"),
    }


CF_ESTIMATES = {"group_average": "fit_to_group_mean_timeseries", "per_subject": "mean_of_subject_maps"}


def cf_maps_stem(roi_a: str, roi_b: str, mode: str) -> str:
    return f"cf_model_maps_{roi_a}_{roi_b}_{CF_ESTIMATES[mode]}"


def per_subject_mean_stem(cf_model_stem: str) -> str:
    return f"per_subject_mean_{cf_model_stem}"


def persubject_output_root() -> Path:
    """Root directory for per-subject CF outputs (external drive).

    Honours ``MOVIE_PERSUBJECT_ROOT`` as an override. Raises if the target
    isn't reachable: writing through a symlink to an unmounted drive would
    otherwise silently fall back onto the root filesystem.
    """
    override = os.environ.get("MOVIE_PERSUBJECT_ROOT")
    if override:
        root = Path(override)
        if not root.exists():
            raise RuntimeError(
                f"MOVIE_PERSUBJECT_ROOT={root} does not exist. Point it at a "
                "reachable directory.")
        return root
    if not PERSUBJECT_MOUNT.is_dir():
        raise RuntimeError(
            f"External drive not reachable at {PERSUBJECT_MOUNT}. Mount it "
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
