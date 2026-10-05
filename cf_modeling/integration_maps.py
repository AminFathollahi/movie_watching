"""
cf_modeling/integration_maps.py
=================================
Derive integration maps and save a single combined CIFTI from CF model results.

For ``group_average`` mode: loads the CF fit to one group-mean fMRI time
series from the prep directory.  This is not an average of fitted maps.
For ``per_subject`` mode: aggregates subject-level CF R² maps (nanmean).

Single CIFTI output: cf_model_maps_{roi_a}_{roi_b}_*.dscalar.nii
  All scalar maps in one file for wb_view / viz_cf_modeling.ipynb:
    cf_model_full_r2, cf_model_split_r2_{a}, cf_model_split_r2_{b},
    cf_model_roi_mean_null_r2_{a}, cf_model_roi_mean_null_r2_{b},
    cf_model_null_corrected_split_r2_{a},
    cf_model_null_corrected_split_r2_{b},
    cf_model_joint_split_r2_geomean,
    cf_model_joint_null_corrected_split_r2_geomean,
    modality_balance

The 2D bimodal map (R²_a_nc vs R²_b_nc colour wheel) is generated in
viz_cf_modeling.ipynb via pycortex — not saved as a CIFTI.
"""

import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import save_cifti_multimap
from paths import DATA, OUTPUTS
from cf_modeling.cf_naming import (
    cf_model_map_stems,
    legacy_cf_model_map_stems,
    per_subject_mean_stem,
    resolve_cf_model_map_path,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_OUT_BASE       = str(OUTPUTS / "cf_modeling")
_PYCORTEX_STORE = str(DATA / "hedger2026")


# =============================================================================
# Per-subject map collection
# =============================================================================

def collect_maps(subjects_dir, map_name, min_subjects, legacy_name=None):
    """Load a canonical subject map, accepting legacy names during migration."""
    sub_dirs = sorted(glob(os.path.join(subjects_dir, "*")))
    arrays, missing = [], []
    for sd in sub_dirs:
        path = resolve_cf_model_map_path(sd, map_name, legacy_name)
        if os.path.exists(path):
            arrays.append(np.load(path))
        else:
            missing.append(os.path.basename(sd))
    if missing:
        log.warning("  %s: missing for %d subjects", map_name, len(missing))
    if len(arrays) < min_subjects:
        raise RuntimeError(
            f"Only {len(arrays)} subjects have {map_name}.npy — "
            f"need at least {min_subjects}.")
    return np.stack(arrays, axis=0)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute integration maps and CIFTI outputs from CF model results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",           required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi-a",          dest="roi_a",          default="A1")
    p.add_argument("--roi-b",          dest="roi_b",          default="V1")
    p.add_argument("--output-base",    dest="output_base",    default=_OUT_BASE)
    p.add_argument("--pycortex-store", dest="pycortex_store", default=_PYCORTEX_STORE)
    p.add_argument("--template-cifti", dest="template_cifti", default=None,
                   help="59k cortex-only CIFTI template (required).")
    p.add_argument("--min-subjects",   dest="min_subjects",   type=int, default=1)
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if args.template_cifti is None:
        raise ValueError("--template-cifti is required.")

    os.environ["PYCORTEX_FILESTORE"] = args.pycortex_store

    roi_root = f"{args.output_base}/{args.mode}/{args.roi_a}_{args.roi_b}"
    roi_a    = args.roi_a
    roi_b    = args.roi_b
    stems = cf_model_map_stems(roi_a, roi_b)
    legacy_stems = legacy_cf_model_map_stems(roi_a, roi_b)

    log.info("=" * 60)
    log.info("Integration maps — %s  %s × %s", args.mode, roi_a, roi_b)
    log.info("=" * 60)

    # ── Load maps ─────────────────────────────────────────────────────────────
    if args.mode == "group_average":
        prep_dir      = f"{roi_root}/prep"
        out_cifti_dir = f"{roi_root}/cifti_maps"
        os.makedirs(out_cifti_dir, exist_ok=True)

        log.info("Loading held-out CF R² maps …")
        def _load(key):
            return np.load(resolve_cf_model_map_path(
                prep_dir, stems[key], legacy_stems[key]))
        R2_full    = _load("full_r2")
        R2_a       = _load("split_r2_a")
        R2_b       = _load("split_r2_b")
        R2_null_a  = _load("roi_mean_null_r2_a")
        R2_null_b  = _load("roi_mean_null_r2_b")
        R2_a_nc    = _load("null_corrected_split_r2_a")
        R2_b_nc    = _load("null_corrected_split_r2_b")
        product    = _load("joint_split_r2_geomean")
        product_nc = _load("joint_null_corrected_split_r2_geomean")

    else:  # per_subject
        subjects_dir  = f"{roi_root}/subjects"
        group_dir     = f"{roi_root}/group"
        out_cifti_dir = f"{group_dir}/cifti_maps"
        os.makedirs(group_dir,     exist_ok=True)
        os.makedirs(out_cifti_dir, exist_ok=True)

        log.info("Aggregating subject-level held-out CF R² maps …")
        def _avg(key):
            return np.nanmean(collect_maps(
                subjects_dir, stems[key], args.min_subjects, legacy_stems[key]),
                              axis=0).astype(np.float32)

        R2_full    = _avg("full_r2")
        R2_a       = _avg("split_r2_a")
        R2_b       = _avg("split_r2_b")
        R2_null_a  = _avg("roi_mean_null_r2_a")
        R2_null_b  = _avg("roi_mean_null_r2_b")
        R2_a_nc    = _avg("null_corrected_split_r2_a")
        R2_b_nc    = _avg("null_corrected_split_r2_b")
        product    = _avg("joint_split_r2_geomean")
        product_nc = _avg("joint_null_corrected_split_r2_geomean")
        log.info("  Averaged %d subjects.", R2_a.shape[0] if R2_a.ndim > 1 else 1)

        for key, arr in [
            ("full_r2", R2_full),
            ("split_r2_a", R2_a), ("split_r2_b", R2_b),
            ("roi_mean_null_r2_a", R2_null_a),
            ("roi_mean_null_r2_b", R2_null_b),
            ("null_corrected_split_r2_a", R2_a_nc),
            ("null_corrected_split_r2_b", R2_b_nc),
            ("joint_split_r2_geomean", product),
            ("joint_null_corrected_split_r2_geomean", product_nc),
        ]:
            np.save(os.path.join(group_dir, f"{per_subject_mean_stem(stems[key])}.npy"), arr)

    # ── Derived maps ──────────────────────────────────────────────────────────
    modality_balance = (
        (R2_a_nc - R2_b_nc) / (np.abs(R2_a_nc) + np.abs(R2_b_nc) + 1e-8)
    ).astype(np.float32)

    # ── Single combined CIFTI with all maps ───────────────────────────────────
    map_names = [
        stems["full_r2"],
        stems["split_r2_a"], stems["split_r2_b"],
        stems["roi_mean_null_r2_a"], stems["roi_mean_null_r2_b"],
        stems["null_corrected_split_r2_a"], stems["null_corrected_split_r2_b"],
        stems["joint_split_r2_geomean"],
        stems["joint_null_corrected_split_r2_geomean"],
        "cf_model_null_corrected_split_r2_balance"
    ]
    map_arrays = [
        R2_full,
        R2_a,          R2_b,
        R2_null_a,     R2_null_b,
        R2_a_nc,       R2_b_nc,
        product,       product_nc,
        modality_balance
    ]

    estimate = ("fit_to_group_mean_timeseries" if args.mode == "group_average"
                else "mean_of_subject_maps")
    combined_path = os.path.join(
        out_cifti_dir,
        f"cf_model_maps_{roi_a.lower()}_{roi_b.lower()}_{estimate}.dscalar.nii")
    save_cifti_multimap(
        np.vstack([a.reshape(1, -1) for a in map_arrays]),
        map_names,
        args.template_cifti,
        combined_path,
    )
    log.info("Saved: %s  (%d maps)", os.path.basename(combined_path), len(map_names))
    for name, arr in zip(map_names, map_arrays):
        log.info("  %-25s mean=%+.4f  frac>0=%.1f%%",
                 name, float(np.nanmean(arr)), 100.0 * float(np.mean(arr > 0)))

    # Keep cifti_maps/ view-only: CIFTIs and legends, no JSON sidecars.
    metadata_path = Path(out_cifti_dir).parent / (
        f"cf_model_maps_{roi_a.lower()}_{roi_b.lower()}_{estimate}.json")
    metadata_path.write_text(json.dumps({
        "analysis": "connective_field_model_heldout_r2_maps",
        "estimate": estimate,
        "estimate_definition": (
            "One CF model fit to the group-mean fMRI time series."
            if args.mode == "group_average" else
            "Arithmetic mean of independently fitted subject-level CF maps."),
        "map_names": map_names,
        "cifti": combined_path,
    }, indent=2) + "\n")

    log.info("\nintegration_maps.py complete.")


if __name__ == "__main__":
    main()
