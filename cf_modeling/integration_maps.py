"""
cf_modeling/integration_maps.py
=================================
Derive continuous 2D dual-overlay maps and comprehensive CIFTI outputs.

For group_average mode: loads all R2 components from the prep directory.
For per_subject mode: aggregates all R² maps across subjects (nanmean).

Outputs:
  1. CIFTI dscalar.nii: A combined multi-map CIFTI containing the full model, 
     raw splits, null models, and null-corrected maps with dynamic ROI names.
  2. Pycortex Flatmap (PNG): A 2D colormap (Vertex2D) flatmap using the 
     null-corrected maps, reproducing the dual-overlay style of Hedger et al.
"""

import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

# Set ROOT to the 'movie_watching' parent directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import from root cifti_io
from cifti_io import get_bm_axis, save_cifti_multimap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"
_OUT_BASE  = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"
_PYCORTEX_STORE = f"{_DATA_BASE}/hedger2026"


# =============================================================================
# Bimodal dlabel helper
# =============================================================================

def _save_bimodal_dlabel_standalone(R2_a_nc, R2_b_nc, roi_a, roi_b,
                                     template_cifti, cifti_dir):
    """Save bimodal dlabel CIFTI (0=neither, 1=roi_a, 2=roi_b, 3=bimodal)."""
    import nibabel as nib

    a_pos = R2_a_nc > 0
    b_pos = R2_b_nc > 0
    label_map = np.zeros(len(R2_a_nc), dtype=np.int32)
    label_map[a_pos & ~b_pos] = 1
    label_map[~a_pos & b_pos] = 2
    label_map[a_pos & b_pos]  = 3

    bm_axis = nib.load(template_cifti).header.get_axis(1)
    label_table = nib.cifti2.Cifti2LabelTable()
    for key, (name, r, g, b, a) in {
        0: ("Neither",                  0.6,  0.6,  0.6,  1.0),
        1: (f"{roi_a}_dominant",        0.85, 0.15, 0.15, 1.0),
        2: (f"{roi_b}_dominant",        0.15, 0.15, 0.85, 1.0),
        3: (f"Bimodal_{roi_a}_{roi_b}", 0.65, 0.10, 0.75, 1.0),
    }.items():
        label_table[key] = nib.cifti2.Cifti2Label(key, name, r, g, b, a)

    map_name = f"bimodal_{roi_a}_{roi_b}"
    label_axis = nib.cifti2.LabelAxis([map_name], [label_table])
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_axis))
    img = nib.Cifti2Image(label_map.reshape(1, -1).astype(np.int32), header=header)
    out_path = os.path.join(cifti_dir, f"{map_name}.dlabel.nii")
    nib.save(img, out_path)
    log.info("  Saved bimodal dlabel: %s", os.path.basename(out_path))


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
    if len(arrays) < min_subjects:
        raise RuntimeError(
            f"Only {len(arrays)} subjects have {map_name}.npy — "
            f"need at least {min_subjects}.")
    return np.stack(arrays, axis=0)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and visualize continuous 2D integration maps and comprehensive CIFTIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",           required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi-a",          dest="roi_a",         default="A1")
    p.add_argument("--roi-b",          dest="roi_b",         default="V1")
    p.add_argument("--output-base",    dest="output_base",   default=_OUT_BASE)
    p.add_argument("--pycortex-store", dest="pycortex_store", default=_PYCORTEX_STORE,
                   help="Path to directory containing hcp_999999_draw_NH")
    p.add_argument("--template-cifti", dest="template_cifti", default=None,
                   help="59k preprocessed CIFTI used as template for dscalar output.")
    p.add_argument("--min-subjects",   dest="min_subjects",  type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    if args.template_cifti is None:
        raise ValueError("--template_cifti is required for mapping headers.")


    os.environ["PYCORTEX_FILESTORE"] = args.pycortex_store

    roi_root = f"{args.output_base}/{args.mode}/{args.roi_a}_{args.roi_b}"
    bm_axis  = get_bm_axis(args.template_cifti)

    log.info("=" * 60)
    log.info(f"Integration maps — Combined CIFTIs & Bimodal dlabel ({args.mode})")
    log.info(f"  ROIs: {args.roi_a} × {args.roi_b}")
    log.info("=" * 60)

    if args.mode == "group_average":
        prep_dir  = f"{roi_root}/prep"
        out_cifti_dir = f"{roi_root}/cifti_maps"
        out_fig_dir   = f"{roi_root}/figures"
        os.makedirs(out_cifti_dir, exist_ok=True)
        os.makedirs(out_fig_dir, exist_ok=True)

        log.info("\nLoading R² maps …")
        R2_full   = np.load(os.path.join(prep_dir, "R2_full.npy"))
        R2_a      = np.load(os.path.join(prep_dir, f"R2_{args.roi_a}.npy"))
        R2_b      = np.load(os.path.join(prep_dir, f"R2_{args.roi_b}.npy"))
        Shared_R2 = np.load(os.path.join(prep_dir, "Shared_R2.npy"))
        R2_null_a = np.load(os.path.join(prep_dir, f"R2_null_{args.roi_a}.npy"))
        R2_null_b = np.load(os.path.join(prep_dir, f"R2_null_{args.roi_b}.npy"))
        R2_a_nc   = np.load(os.path.join(prep_dir, f"R2_{args.roi_a}_nc.npy"))
        R2_b_nc   = np.load(os.path.join(prep_dir, f"R2_{args.roi_b}_nc.npy"))
        product_map = np.load(os.path.join(prep_dir, "product_map.npy"))
        map_names = [
            "R2_full", f"R2_{args.roi_a}", f"R2_{args.roi_b}",
            "Shared_R2",
            f"R2_null_{args.roi_a}", f"R2_null_{args.roi_b}",
            f"R2_{args.roi_a}_nc", f"R2_{args.roi_b}_nc",
            "product_map",
        ]

    else:  # per_subject
        subjects_dir = f"{roi_root}/subjects"
        group_dir    = f"{roi_root}/group"
        out_cifti_dir = f"{group_dir}/cifti_maps"
        out_fig_dir   = f"{group_dir}/figures"
        os.makedirs(group_dir,  exist_ok=True)
        os.makedirs(out_cifti_dir, exist_ok=True)
        os.makedirs(out_fig_dir, exist_ok=True)

        log.info("\nAggregating per-subject R² maps …")
        maps_full   = collect_maps(subjects_dir, "R2_full", args.min_subjects)
        maps_a      = collect_maps(subjects_dir, f"R2_{args.roi_a}", args.min_subjects)
        maps_b      = collect_maps(subjects_dir, f"R2_{args.roi_b}", args.min_subjects)
        maps_null_a = collect_maps(subjects_dir, f"R2_null_{args.roi_a}", args.min_subjects)
        maps_null_b = collect_maps(subjects_dir, f"R2_null_{args.roi_b}", args.min_subjects)
        maps_a_nc   = collect_maps(subjects_dir, f"R2_{args.roi_a}_nc", args.min_subjects)
        maps_b_nc   = collect_maps(subjects_dir, f"R2_{args.roi_b}_nc", args.min_subjects)
        maps_product = collect_maps(subjects_dir, "product_map", args.min_subjects)
        log.info(f"  Averaging across {maps_a_nc.shape[0]} subjects …")

        R2_full   = np.nanmean(maps_full, axis=0).astype(np.float32)
        R2_a      = np.nanmean(maps_a, axis=0).astype(np.float32)
        R2_b      = np.nanmean(maps_b, axis=0).astype(np.float32)
        R2_null_a = np.nanmean(maps_null_a, axis=0).astype(np.float32)
        R2_null_b = np.nanmean(maps_null_b, axis=0).astype(np.float32)
        R2_a_nc   = np.nanmean(maps_a_nc, axis=0).astype(np.float32)
        R2_b_nc   = np.nanmean(maps_b_nc, axis=0).astype(np.float32)
        product_map = np.nanmean(maps_product, axis=0).astype(np.float32)

        # Save group average .npy 
        for name, arr in [
            ("R2_full_avg", R2_full),
            (f"R2_{args.roi_a}_avg", R2_a),
            (f"R2_{args.roi_b}_avg", R2_b),
            (f"R2_null_{args.roi_a}_avg", R2_null_a),
            (f"R2_null_{args.roi_b}_avg", R2_null_b),
            (f"R2_{args.roi_a}_nc_avg", R2_a_nc),
            (f"R2_{args.roi_b}_nc_avg", R2_b_nc),
            ("product_map_avg", product_map),
        ]:
            np.save(os.path.join(group_dir, f"{name}.npy"), arr)
        
        R2_a_pos = np.clip(R2_a_nc, 0, None)
        R2_b_pos = np.clip(R2_b_nc, 0, None)
    
        # 2. Calculate the square root of the product
        integration_score = np.sqrt(R2_a_pos * R2_b_pos)
        
        # 3. Save it as a pure numpy array so summary.py can read it
        np.save(os.path.join(group_dir, "integration_score.npy"), integration_score)
        map_names = [
            "R2_full_avg",
            f"R2_{args.roi_a}_avg",
            f"R2_{args.roi_b}_avg",
            f"R2_null_{args.roi_a}_avg",
            f"R2_null_{args.roi_b}_avg",
            f"R2_{args.roi_a}_nc_avg",
            f"R2_{args.roi_b}_nc_avg",
            "product_map_avg",
        ]

    # Save Combined Multi-Map CIFTI
    combined_cifti_name = f"cf_result_{args.roi_a.lower()}_{args.roi_b.lower()}.dscalar.nii"
    combined_cifti_path = os.path.join(out_cifti_dir, combined_cifti_name)

    if args.mode == "group_average":
        data_2d = np.vstack([R2_full, R2_a, R2_b, Shared_R2,
                             R2_null_a, R2_null_b, R2_a_nc, R2_b_nc, product_map])
    else:
        data_2d = np.vstack([R2_full, R2_a, R2_b,
                             R2_null_a, R2_null_b, R2_a_nc, R2_b_nc, product_map])

    save_cifti_multimap(data_2d, map_names, args.template_cifti, combined_cifti_path)
    log.info(f"\nSaved combined CIFTI: {combined_cifti_path}")
    log.info(f"  Maps included: {map_names}")

    # Bimodal integration classification dlabel
    if args.mode == "group_average":
        _save_bimodal_dlabel_standalone(R2_a_nc, R2_b_nc, args.roi_a, args.roi_b,
                                        args.template_cifti, out_cifti_dir)

    log.info("\nintegration_maps.py complete.")

if __name__ == "__main__":
    main()