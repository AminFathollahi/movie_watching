"""
cf_modeling/integration_maps.py
=================================
Derive Figure 3a display maps from null-corrected split R².

For group_average mode: loads R2_nc from the prep directory and computes
integration maps directly.

For per_subject mode: first aggregates null-corrected R² maps across subjects
(nanmean), then computes integration maps on the group average.

Integration maps
----------------
  integration_score   = √(clip(R2_A_nc, 0) × clip(R2_B_nc, 0))
                        high where BOTH are positive → bimodal zone
  modality_balance    = R2_A_nc − R2_B_nc  (diverging; + = ROI_A dominant)
  bimodal_map.dlabel  = 4-category label: neither / ROI_A_only / ROI_B_only / bimodal

All outputs saved as CIFTI dscalar.nii using --template_cifti.

Run
---
    python integration_maps.py --mode group_average --roi_a A1 --roi_b V1
    python integration_maps.py --mode per_subject   --roi_a A5 --roi_b FFC
                                  --min_subjects 5
"""

import argparse
import logging
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
from nibabel.cifti2.cifti2_axes import LabelAxis

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"
_HCP_DIR   = f"{_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_OUT_BASE  = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"

MASK_PERCENTILE  = 90    # top 10% of vertices → integration mask
DLABEL_THRESHOLD = 0.0   # vertices above this count as "hot" for dlabel


# =============================================================================
# CIFTI saving
# =============================================================================

def _bm_axis_from_template(template_cifti):
    return nib.load(template_cifti).header.get_axis(1)


def _save_dscalar(arr, name, bm_axis, out_dir):
    arr_f32   = arr.astype(np.float32)
    scalar_ax = nib.cifti2.ScalarAxis([name])
    header    = nib.cifti2.Cifti2Header.from_axes((scalar_ax, bm_axis))
    img       = nib.Cifti2Image(arr_f32.reshape(1, -1), header=header)
    path      = os.path.join(out_dir, f"{name}.dscalar.nii")
    nib.save(img, path)
    log.info(f"  {name}: min={arr.min():.4f}  max={arr.max():.4f}  "
             f"mean={arr.mean():.4f}  frac>0={np.mean(arr>0):.1%}")
    return path


def _save_dlabel(R2_a_nc, R2_b_nc, roi_a, roi_b, bm_axis, out_dir, threshold=0.0):
    a_hot = R2_a_nc > threshold
    b_hot = R2_b_nc > threshold
    label_arr = np.zeros(len(R2_a_nc), dtype=np.int32)
    label_arr[a_hot & ~b_hot] = 1
    label_arr[~a_hot & b_hot] = 2
    label_arr[a_hot  &  b_hot] = 3

    lt = {
        0: ("neither",       (0.5, 0.5, 0.5, 0.0)),
        1: (f"{roi_a}_only", (0.8, 0.1, 0.1, 1.0)),
        2: (f"{roi_b}_only", (0.1, 0.1, 0.8, 1.0)),
        3: ("bimodal",       (0.6, 0.0, 0.8, 1.0)),
    }
    label_col    = np.empty(1, dtype=object)
    label_col[0] = lt
    la     = LabelAxis(name=np.array(["bimodal_map"]), label=label_col)
    header = nib.Cifti2Header.from_axes((la, bm_axis))
    img    = nib.Cifti2Image(label_arr.reshape(1, -1).astype(np.float32), header=header)
    img.nifti_header["intent_code"] = 3007
    path = os.path.join(out_dir, "bimodal_map.dlabel.nii")
    nib.save(img, path)
    counts = {k: int((label_arr == k).sum()) for k in range(4)}
    log.info(f"  bimodal dlabel — neither={counts[0]}  {roi_a}_only={counts[1]}  "
             f"{roi_b}_only={counts[2]}  bimodal={counts[3]}")


# =============================================================================
# Map arithmetic
# =============================================================================

def compute_integration_maps(R2_a_nc, R2_b_nc):
    integration_score = np.sqrt(
        np.clip(R2_a_nc, 0, None) * np.clip(R2_b_nc, 0, None)
    ).astype(np.float32)
    modality_balance = (R2_a_nc - R2_b_nc).astype(np.float32)
    return integration_score, modality_balance


# =============================================================================
# Per-subject aggregation
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
        log.warning(f"  {map_name}: missing for {len(missing)} subjects "
                    f"(e.g. {missing[:3]})")
    if len(arrays) < min_subjects:
        raise RuntimeError(
            f"Only {len(arrays)} subjects have {map_name}.npy — "
            f"need at least {min_subjects}.")
    log.info(f"  {map_name}: loaded {len(arrays)} subjects")
    return np.stack(arrays, axis=0)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute integration maps from null-corrected R² (Figure 3a).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",          required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi_a",         default="A1")
    p.add_argument("--roi_b",         default="V1")
    p.add_argument("--output_base",   default=_OUT_BASE)
    p.add_argument("--template_cifti", default=None,
                   help="59k preprocessed CIFTI used as template for dscalar/dlabel "
                        "output headers (required). Pass the group-average dtseries "
                        "from preprocess_individual.py.")
    p.add_argument("--min_subjects", type=int, default=1,
                   help="Minimum subjects required (per_subject mode only).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.template_cifti is None:
        raise ValueError(
            "--template_cifti is required. Pass the preprocessed group-average 59k "
            "CIFTI (group_average_{suffix}_cortex_59k.dtseries.nii).")

    roi_root = f"{args.output_base}/{args.mode}/{args.roi_a}_{args.roi_b}"
    bm_axis  = _bm_axis_from_template(args.template_cifti)

    log.info("=" * 60)
    log.info(f"Script 05 — Integration maps ({args.mode})")
    log.info(f"  ROIs: {args.roi_a} × {args.roi_b}")
    log.info(f"  dlabel threshold: {DLABEL_THRESHOLD}")
    log.info("=" * 60)

    if args.mode == "group_average":
        prep_dir  = f"{roi_root}/prep"
        cifti_dir = f"{roi_root}/cifti_maps"
        os.makedirs(cifti_dir, exist_ok=True)

        log.info("\nLoading null-corrected R² maps …")
        R2_a_nc = np.load(os.path.join(prep_dir, f"R2_{args.roi_a}_nc.npy"))
        R2_b_nc = np.load(os.path.join(prep_dir, f"R2_{args.roi_b}_nc.npy"))
        out_npy_dir  = prep_dir
        out_cifti_dir = cifti_dir

    else:  # per_subject
        subjects_dir = f"{roi_root}/subjects"
        group_dir    = f"{roi_root}/group"
        out_npy_dir  = group_dir
        out_cifti_dir = f"{group_dir}/cifti_maps"
        os.makedirs(out_npy_dir,  exist_ok=True)
        os.makedirs(out_cifti_dir, exist_ok=True)

        log.info("\nAggregating per-subject null-corrected R² maps …")
        maps_a_nc = collect_maps(subjects_dir, f"R2_{args.roi_a}_nc", args.min_subjects)
        maps_b_nc = collect_maps(subjects_dir, f"R2_{args.roi_b}_nc", args.min_subjects)
        maps_full = collect_maps(subjects_dir, "R2_full",              args.min_subjects)
        maps_a    = collect_maps(subjects_dir, f"R2_{args.roi_a}",     args.min_subjects)
        maps_b    = collect_maps(subjects_dir, f"R2_{args.roi_b}",     args.min_subjects)
        n_subs    = maps_a_nc.shape[0]
        log.info(f"  Averaging across {n_subs} subjects …")

        R2_a_nc = np.nanmean(maps_a_nc, axis=0).astype(np.float32)
        R2_b_nc = np.nanmean(maps_b_nc, axis=0).astype(np.float32)
        R2_full_avg = np.nanmean(maps_full, axis=0).astype(np.float32)
        R2_a_avg    = np.nanmean(maps_a,    axis=0).astype(np.float32)
        R2_b_avg    = np.nanmean(maps_b,    axis=0).astype(np.float32)

        log.info(f"  R2_{args.roi_a}_nc: mean={R2_a_nc.mean():.4f}  "
                 f"frac>0={np.mean(R2_a_nc > 0):.1%}")
        log.info(f"  R2_{args.roi_b}_nc: mean={R2_b_nc.mean():.4f}  "
                 f"frac>0={np.mean(R2_b_nc > 0):.1%}")

        # Save group average R2 maps as .npy and CIFTI
        for name, arr in [
            (f"R2_{args.roi_a}_nc_avg", R2_a_nc),
            (f"R2_{args.roi_b}_nc_avg", R2_b_nc),
            ("R2_full_avg",              R2_full_avg),
            (f"R2_{args.roi_a}_avg",     R2_a_avg),
            (f"R2_{args.roi_b}_avg",     R2_b_avg),
        ]:
            np.save(os.path.join(out_npy_dir, f"{name}.npy"), arr)
            _save_dscalar(arr, name, bm_axis, out_cifti_dir)

    log.info(f"\n  R2_{args.roi_a}_nc: mean={R2_a_nc.mean():.4f}")
    log.info(f"  R2_{args.roi_b}_nc: mean={R2_b_nc.mean():.4f}")

    # ── Integration maps ──────────────────────────────────────────────────────
    log.info("\nComputing integration maps …")
    integration_score, modality_balance = compute_integration_maps(R2_a_nc, R2_b_nc)

    threshold      = np.percentile(integration_score, MASK_PERCENTILE)
    int_mask       = (integration_score >= threshold).astype(np.float32)
    mask_name      = f"top_{MASK_PERCENTILE}_percentile_integration"
    log.info(f"  Integration mask: threshold={threshold:.5f}  "
             f"n_verts={int(int_mask.sum())}  ({100-MASK_PERCENTILE}% of cortex)")

    suffix = "_avg" if args.mode == "per_subject" else ""
    for name, arr in [
        (f"integration_score{suffix}",  integration_score),
        (f"modality_balance{suffix}",   modality_balance),
        (f"{mask_name}{suffix}",        int_mask),
    ]:
        np.save(os.path.join(out_npy_dir, f"{name}.npy"), arr.astype(np.float32))
        _save_dscalar(arr, name, bm_axis, out_cifti_dir)

    log.info("\nBuilding bimodal dlabel …")
    _save_dlabel(R2_a_nc, R2_b_nc, args.roi_a, args.roi_b, bm_axis,
                 out_cifti_dir, threshold=DLABEL_THRESHOLD)

    p95_a = np.percentile(R2_a_nc[R2_a_nc > 0], 95) if (R2_a_nc > 0).any() else 0
    p95_b = np.percentile(R2_b_nc[R2_b_nc > 0], 95) if (R2_b_nc > 0).any() else 0
    log.info("\n--- wb_view: continuous 2D dual-overlay (Figure 3a equivalent) ---")
    log.info(f"  Overlay 1 (blue): R2_{args.roi_b}_nc dscalar  min=0  max≈{p95_b:.4f}  transparent below 0")
    log.info(f"  Overlay 2 (red):  R2_{args.roi_a}_nc dscalar  min=0  max≈{p95_a:.4f}  transparent below 0")
    log.info(f"  Discrete labels: bimodal_map.dlabel.nii")
    log.info("\nScript 05 complete.")
    log.info(f"Next: python summary.py --mode {args.mode} "
             f"--roi_a {args.roi_a} --roi_b {args.roi_b}")


if __name__ == "__main__":
    main()
