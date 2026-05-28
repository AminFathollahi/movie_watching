"""
cf_modeling/integration_maps.py
=================================
Derive integration maps and save CIFTI outputs from CF model results.

For group_average mode: loads R² maps from the prep directory.
For per_subject mode: aggregates R² maps across subjects (nanmean).

CIFTI outputs (all in {roi_root}/{mode}/cifti_maps/):
  1.  R2_audio_{roi_a}.dscalar.nii      — null-corrected R² for ROI A
  2.  R2_video_{roi_b}.dscalar.nii      — null-corrected R² for ROI B
  3.  R2_full.dscalar.nii               — full-model R²
  4.  integration_score.dscalar.nii     — √(R²_A × R²_B)  raw product
  5.  integration_score_nc.dscalar.nii  — √(clip(R²_A_nc,0) × clip(R²_B_nc,0))  nc product
  6.  modality_balance.dscalar.nii      — (R²_a_nc−R²_b_nc)/(|R²_a_nc|+|R²_b_nc|+ε)
  7.  bimodal_mask.dscalar.nii          — binary: both R²_nc > 0
  8.  bimodal_map.dlabel.nii            — 4-class discrete label
  9.  bimodal_continuous_2d.dscalar.nii — 2-map: [audio_R2_nc, video_R2_nc] for 2D overlay
  10. cf_result_{roi_a}_{roi_b}.dscalar.nii — all scalar maps combined
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
# Bimodal dlabel helper
# =============================================================================

def _save_bimodal_dlabel(R2_a_nc, R2_b_nc, roi_a, roi_b, template_cifti, cifti_dir):
    """Save bimodal dlabel CIFTI (0=neither, 1=roi_a only, 2=roi_b only, 3=bimodal)."""
    import nibabel as nib

    a_pos = R2_a_nc > 0
    b_pos = R2_b_nc > 0
    label_map = np.zeros(len(R2_a_nc), dtype=np.int32)
    label_map[a_pos & ~b_pos] = 1
    label_map[~a_pos & b_pos] = 2
    label_map[a_pos & b_pos]  = 3

    bm_axis    = nib.load(template_cifti).header.get_axis(1)
    label_table = nib.cifti2.Cifti2LabelTable()
    for key, (name, r, g, b, a) in {
        0: ("Neither",                  0.6,  0.6,  0.6,  1.0),
        1: (f"{roi_a}_dominant",        0.85, 0.15, 0.15, 1.0),
        2: (f"{roi_b}_dominant",        0.15, 0.15, 0.85, 1.0),
        3: (f"Bimodal_{roi_a}_{roi_b}", 0.65, 0.10, 0.75, 1.0),
    }.items():
        label_table[key] = nib.cifti2.Cifti2Label(key, name, r, g, b, a)

    map_name   = f"bimodal_{roi_a}_{roi_b}"
    label_axis = nib.cifti2.LabelAxis([map_name], [label_table])
    header     = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_axis))
    img        = nib.Cifti2Image(label_map.reshape(1, -1).astype(np.int32), header=header)
    out_path   = os.path.join(cifti_dir, f"{map_name}.dlabel.nii")
    nib.save(img, out_path)
    log.info("  Saved: %s  (bimodal=%.1f%%)", os.path.basename(out_path),
             100.0 * float(np.mean(label_map == 3)))


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
        R2_full       = np.load(os.path.join(prep_dir, "R2_full.npy"))
        R2_a          = np.load(os.path.join(prep_dir, f"R2_{roi_a}.npy"))
        R2_b          = np.load(os.path.join(prep_dir, f"R2_{roi_b}.npy"))
        R2_a_nc       = np.load(os.path.join(prep_dir, f"R2_{roi_a}_nc.npy"))
        R2_b_nc       = np.load(os.path.join(prep_dir, f"R2_{roi_b}_nc.npy"))
        product       = np.load(os.path.join(prep_dir, "product_map.npy"))
        product_nc    = np.load(os.path.join(prep_dir, "product_map_nc.npy"))

        scalar_names  = [
            "R2_full",
            f"R2_{roi_a}",      f"R2_{roi_b}",
            f"R2_{roi_a}_nc",   f"R2_{roi_b}_nc",
            "product_map",      "product_map_nc",
        ]
        scalar_arrays = [R2_full, R2_a, R2_b, R2_a_nc, R2_b_nc, product, product_nc]

    else:  # per_subject
        subjects_dir  = f"{roi_root}/subjects"
        group_dir     = f"{roi_root}/group"
        out_cifti_dir = f"{group_dir}/cifti_maps"
        os.makedirs(group_dir,     exist_ok=True)
        os.makedirs(out_cifti_dir, exist_ok=True)

        log.info("Aggregating per-subject R² maps …")
        maps_full      = collect_maps(subjects_dir, "R2_full",        args.min_subjects)
        maps_a         = collect_maps(subjects_dir, f"R2_{roi_a}",    args.min_subjects)
        maps_b         = collect_maps(subjects_dir, f"R2_{roi_b}",    args.min_subjects)
        maps_a_nc      = collect_maps(subjects_dir, f"R2_{roi_a}_nc", args.min_subjects)
        maps_b_nc      = collect_maps(subjects_dir, f"R2_{roi_b}_nc", args.min_subjects)
        maps_product   = collect_maps(subjects_dir, "product_map",    args.min_subjects)
        maps_product_nc = collect_maps(subjects_dir, "product_map_nc", args.min_subjects)
        log.info("  Averaging %d subjects …", maps_a.shape[0])

        R2_full    = np.nanmean(maps_full,       axis=0).astype(np.float32)
        R2_a       = np.nanmean(maps_a,          axis=0).astype(np.float32)
        R2_b       = np.nanmean(maps_b,          axis=0).astype(np.float32)
        R2_a_nc    = np.nanmean(maps_a_nc,       axis=0).astype(np.float32)
        R2_b_nc    = np.nanmean(maps_b_nc,       axis=0).astype(np.float32)
        product    = np.nanmean(maps_product,    axis=0).astype(np.float32)
        product_nc = np.nanmean(maps_product_nc, axis=0).astype(np.float32)

        for name, arr in [
            ("R2_full_avg",             R2_full),
            (f"R2_{roi_a}_avg",         R2_a),
            (f"R2_{roi_b}_avg",         R2_b),
            (f"R2_{roi_a}_nc_avg",      R2_a_nc),
            (f"R2_{roi_b}_nc_avg",      R2_b_nc),
            ("product_map_avg",         product),
            ("product_map_nc_avg",      product_nc),
        ]:
            np.save(os.path.join(group_dir, f"{name}.npy"), arr)

        scalar_names  = [
            "R2_full_avg",
            f"R2_{roi_a}_avg",      f"R2_{roi_b}_avg",
            f"R2_{roi_a}_nc_avg",   f"R2_{roi_b}_nc_avg",
            "product_map_avg",      "product_map_nc_avg",
        ]
        scalar_arrays = [R2_full, R2_a, R2_b, R2_a_nc, R2_b_nc, product, product_nc]

    # ── Derived maps ──────────────────────────────────────────────────────────

    # Modality balance: signed preference index in [−1, +1]
    modality_balance = (
        (R2_a_nc - R2_b_nc) /
        (np.abs(R2_a_nc) + np.abs(R2_b_nc) + 1e-8)
    ).astype(np.float32)

    # Bimodal mask: both null-corrected R² > 0
    bimodal_mask = ((R2_a_nc > 0) & (R2_b_nc > 0)).astype(np.float32)

    # ── Combined multi-map CIFTI (all scalar maps) ────────────────────────────
    combined_path = os.path.join(
        out_cifti_dir, f"cf_result_{roi_a.lower()}_{roi_b.lower()}.dscalar.nii")
    save_cifti_multimap(
        np.vstack(scalar_arrays),
        scalar_names,
        args.template_cifti,
        combined_path,
    )
    log.info("Saved combined CIFTI: %s  (%d maps)", os.path.basename(combined_path),
             len(scalar_names))

    # ── Individual CIFTI dscalar files ───────────────────────────────────────
    import nibabel as nib
    bm_ax = nib.load(args.template_cifti).header.get_axis(1)

    def _save_dscalar(arr, name):
        scalar_ax = nib.cifti2.ScalarAxis([name])
        hdr = nib.cifti2.Cifti2Header.from_axes((scalar_ax, bm_ax))
        img = nib.Cifti2Image(arr.reshape(1, -1).astype(np.float32), header=hdr)
        nib.save(img, os.path.join(out_cifti_dir, f"{name}.dscalar.nii"))
        log.info("  Saved: %s.dscalar.nii  (mean=%.4f  frac>0=%.1f%%)",
                 name, float(arr.mean()), 100.0 * float(np.mean(arr > 0)))

    _save_dscalar(R2_a_nc,          f"R2_audio_{roi_a}")
    _save_dscalar(R2_b_nc,          f"R2_video_{roi_b}")
    _save_dscalar(R2_full,          "R2_full")
    _save_dscalar(product,          "integration_score")
    _save_dscalar(product_nc,       "integration_score_nc")
    _save_dscalar(modality_balance, "modality_balance")
    _save_dscalar(bimodal_mask,     "bimodal_mask")

    # ── Bimodal dlabel (4-class discrete) ────────────────────────────────────
    _save_bimodal_dlabel(R2_a_nc, R2_b_nc, roi_a, roi_b,
                         args.template_cifti, out_cifti_dir)

    # ── Bimodal continuous 2D CIFTI (2 maps: R2_a_nc + R2_b_nc) ─────────────
    # Two scalar maps loaded side-by-side in wb_view yield the 2D colour wheel.
    scalar_ax_2d = nib.cifti2.ScalarAxis(
        [f"audio_R2_{roi_a}", f"video_R2_{roi_b}"])
    hdr_2d  = nib.cifti2.Cifti2Header.from_axes((scalar_ax_2d, bm_ax))
    data_2d = np.stack([R2_a_nc.astype(np.float32),
                        R2_b_nc.astype(np.float32)], axis=0)
    nib.save(nib.Cifti2Image(data_2d, header=hdr_2d),
             os.path.join(out_cifti_dir, "bimodal_continuous_2d.dscalar.nii"))
    log.info("  Saved: bimodal_continuous_2d.dscalar.nii")

    log.info("\nintegration_maps.py complete.")


if __name__ == "__main__":
    main()
