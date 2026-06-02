"""
cf_modeling/integration_maps.py
=================================
Derive integration maps and save a single combined CIFTI from CF model results.

For group_average mode: loads R² maps from the prep directory.
For per_subject mode: aggregates R² maps across subjects (nanmean).

Single CIFTI output: cf_result_{roi_a}_{roi_b}.dscalar.nii
  All scalar maps in one file for wb_view / viz_cf_modeling.ipynb:
    R2_full, R2_{a}, R2_{b}, R2_null_{a}, R2_null_{b},
    R2_{a}_nc, R2_{b}_nc, product_map, product_map_nc,
    modality_balance

The 2D bimodal map (R²_a_nc vs R²_b_nc colour wheel) is generated in
viz_cf_modeling.ipynb via pycortex — not saved as a CIFTI.
"""

import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cifti_io import save_cifti_multimap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_OUT_BASE       = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"
_PYCORTEX_STORE = "/home/amin/Research/Representation/Movie/data/hedger2026"


# =============================================================================
# Per-subject map collection
# =============================================================================

def collect_maps(subjects_dir, map_name, min_subjects):
    """Load map_name.npy from all completed subject directories → (N, n_verts)."""
    sub_dirs = sorted(glob(os.path.join(subjects_dir, "*")))
    arrays, missing = [], []
    for sd in sub_dirs:
        path = os.path.join(sd, f"{map_name}.npy")
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

    log.info("=" * 60)
    log.info("Integration maps — %s  %s × %s", args.mode, roi_a, roi_b)
    log.info("=" * 60)

    # ── Load maps ─────────────────────────────────────────────────────────────
    if args.mode == "group_average":
        prep_dir      = f"{roi_root}/prep"
        out_cifti_dir = f"{roi_root}/cifti_maps"
        os.makedirs(out_cifti_dir, exist_ok=True)

        log.info("Loading R² maps …")
        R2_full    = np.load(os.path.join(prep_dir, "R2_full.npy"))
        R2_a       = np.load(os.path.join(prep_dir, f"R2_{roi_a}.npy"))
        R2_b       = np.load(os.path.join(prep_dir, f"R2_{roi_b}.npy"))
        R2_null_a  = np.load(os.path.join(prep_dir, f"R2_null_{roi_a}.npy"))
        R2_null_b  = np.load(os.path.join(prep_dir, f"R2_null_{roi_b}.npy"))
        R2_a_nc    = np.load(os.path.join(prep_dir, f"R2_{roi_a}_nc.npy"))
        R2_b_nc    = np.load(os.path.join(prep_dir, f"R2_{roi_b}_nc.npy"))
        product    = np.load(os.path.join(prep_dir, "product_map.npy"))
        product_nc = np.load(os.path.join(prep_dir, "product_map_nc.npy"))

    else:  # per_subject
        subjects_dir  = f"{roi_root}/subjects"
        group_dir     = f"{roi_root}/group"
        out_cifti_dir = f"{group_dir}/cifti_maps"
        os.makedirs(group_dir,     exist_ok=True)
        os.makedirs(out_cifti_dir, exist_ok=True)

        log.info("Aggregating per-subject R² maps …")
        def _avg(name):
            return np.nanmean(collect_maps(subjects_dir, name, args.min_subjects),
                              axis=0).astype(np.float32)

        R2_full    = _avg("R2_full")
        R2_a       = _avg(f"R2_{roi_a}")
        R2_b       = _avg(f"R2_{roi_b}")
        R2_null_a  = _avg(f"R2_null_{roi_a}")
        R2_null_b  = _avg(f"R2_null_{roi_b}")
        R2_a_nc    = _avg(f"R2_{roi_a}_nc")
        R2_b_nc    = _avg(f"R2_{roi_b}_nc")
        product    = _avg("product_map")
        product_nc = _avg("product_map_nc")
        log.info("  Averaged %d subjects.", R2_a.shape[0] if R2_a.ndim > 1 else 1)

        suffix = "_avg" if args.mode == "per_subject" else ""
        for name, arr in [
            (f"R2_full{suffix}",          R2_full),
            (f"R2_{roi_a}{suffix}",        R2_a),
            (f"R2_{roi_b}{suffix}",        R2_b),
            (f"R2_null_{roi_a}{suffix}",   R2_null_a),
            (f"R2_null_{roi_b}{suffix}",   R2_null_b),
            (f"R2_{roi_a}_nc{suffix}",     R2_a_nc),
            (f"R2_{roi_b}_nc{suffix}",     R2_b_nc),
            (f"product_map{suffix}",       product),
            (f"product_map_nc{suffix}",    product_nc),
        ]:
            np.save(os.path.join(group_dir, f"{name}.npy"), arr)

    # ── Derived maps ──────────────────────────────────────────────────────────
    modality_balance = (
        (R2_a_nc - R2_b_nc) / (np.abs(R2_a_nc) + np.abs(R2_b_nc) + 1e-8)
    ).astype(np.float32)

    # ── Single combined CIFTI with all maps ───────────────────────────────────
    map_names = [
        "R2_full",
        f"R2_{roi_a}",       f"R2_{roi_b}",
        f"R2_null_{roi_a}",  f"R2_null_{roi_b}",
        f"R2_{roi_a}_nc",    f"R2_{roi_b}_nc",
        "product_map",       "product_map_nc",
        "modality_balance"
    ]
    map_arrays = [
        R2_full,
        R2_a,          R2_b,
        R2_null_a,     R2_null_b,
        R2_a_nc,       R2_b_nc,
        product,       product_nc,
        modality_balance
    ]

    combined_path = os.path.join(
        out_cifti_dir, f"cf_result_{roi_a.lower()}_{roi_b.lower()}.dscalar.nii")
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

    log.info("\nintegration_maps.py complete.")


if __name__ == "__main__":
    main()
