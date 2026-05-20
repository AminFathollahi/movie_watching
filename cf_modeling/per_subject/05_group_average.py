"""
05_group_average.py
===================
Aggregate per-subject R² maps, compute group-average integration maps, save
59k CIFTI files for wb_view, resample to 32k for pycortex, and show the
Vertex2D dual-colormap visualization identical to CF_modeling CLAUDE.md Option C.

Workflow
--------
1. Collect per-subject R2_{roi}_nc.npy maps (skip subjects with missing files)
2. Nanmean across subjects → group-average maps (108441,)
3. Compute integration_score and modality_balance
4. Save 59k CIFTI dscalar.nii using one subject's BrainModelAxis as template
5. Resample 59k → 32k via wb_command -cifti-resample
6. Pycortex Vertex2D visualization on hcp_999999_draw_NH

Run
---
    conda activate vicsompy_av
    python 05_group_average.py [--min_subjects N] [--no_pycortex] \\
        --roi_a A5 --roi_b FFC \\
        --output_base /path/to/outputs \\
        --hcp_dir /path/to/HCP_S1200_GroupAvg_v1 \\
        --cifti_dir /path/to/HCP/fMRI_CIFTI
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os
import sys

_DEFAULT_DATA_BASE   = "/home/amin/Research/Representation/Movie/data/Setareh"
_DEFAULT_HCP_DIR     = f"{_DEFAULT_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_DEFAULT_CIFTI_DIR   = "/media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI"
_DEFAULT_ROI_A       = "A5"
_DEFAULT_ROI_B       = "FFC"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/per_subject"

ROI_A         = _DEFAULT_ROI_A
ROI_B         = _DEFAULT_ROI_B
HCP_DIR       = _DEFAULT_HCP_DIR
CIFTI_RAW_DIR = _DEFAULT_CIFTI_DIR
OUTPUT_ROOT   = f"{_DEFAULT_OUTPUT_BASE}/{ROI_A}_{ROI_B}"
SUBJECTS_DIR  = f"{OUTPUT_ROOT}/subjects"
GROUP_DIR     = f"{OUTPUT_ROOT}/group"
CIFTI_59K_DIR = f"{GROUP_DIR}/cifti_59k"
CIFTI_32K_DIR = f"{GROUP_DIR}/cifti_32k"

TEMPLATE_32K = f"{HCP_DIR}/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
SPHERE_59K_L = f"{HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
SPHERE_59K_R = f"{HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"
SPHERE_32K_L = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
SPHERE_32K_R = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"

PYCORTEX_FILESTORE = HCP_DIR
CX_SUB             = "hcp_999999_draw_NH"

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging

import numpy as np
import nibabel as nib
from nibabel.cifti2.cifti2_axes import LabelAxis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.cifti_io import (
    collect_maps,
    get_59k_bm_axis,
    save_59k_cifti,
    resample_to_32k,
    cifti_59k_to_surface,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Analysis helpers
# =============================================================================

def compute_integration_maps(R2_a_nc, R2_b_nc):
    integration_score = np.sqrt(
        np.clip(R2_a_nc, 0, None) * np.clip(R2_b_nc, 0, None)
    ).astype(np.float32)
    modality_balance = (R2_a_nc - R2_b_nc).astype(np.float32)
    return integration_score, modality_balance


def save_dlabel_59k(R2_a_nc, R2_b_nc, bm_axis, threshold=0.0):
    """Save bimodal 4-category dlabel in 59k space."""
    audio_hot = R2_a_nc > threshold
    video_hot = R2_b_nc > threshold

    label_arr = np.zeros(108441, dtype=np.int32)
    label_arr[audio_hot & ~video_hot] = 1
    label_arr[~audio_hot & video_hot] = 2
    label_arr[audio_hot  &  video_hot] = 3

    lt = {
        0: ("neither",        (0.5, 0.5, 0.5, 0.0)),
        1: (f"{ROI_A}_only",  (0.8, 0.1, 0.1, 1.0)),
        2: (f"{ROI_B}_only",  (0.1, 0.1, 0.8, 1.0)),
        3: ("bimodal",        (0.6, 0.0, 0.8, 1.0)),
    }
    label_col    = np.empty(1, dtype=object)
    label_col[0] = lt
    la     = LabelAxis(name=np.array(["bimodal_map"]), label=label_col)
    header = nib.Cifti2Header.from_axes((la, bm_axis))
    img    = nib.Cifti2Image(label_arr.reshape(1, -1).astype(np.float32), header=header)
    img.nifti_header["intent_code"] = 3007
    path = os.path.join(CIFTI_59K_DIR, "bimodal_map.59k.dlabel.nii")
    nib.save(img, path)

    counts = {k: int((label_arr == k).sum()) for k in range(4)}
    log.info(f"  Saved bimodal dlabel — "
             f"neither={counts[0]}  {ROI_A}_only={counts[1]}  "
             f"{ROI_B}_only={counts[2]}  bimodal={counts[3]}")


def show_pycortex(R2_a_nc, R2_b_nc, bm_axis):
    """Display group Vertex2D in pycortex (interactive; not batch-safe)."""
    import cortex
    os.environ["PYCORTEX_FILESTORE"] = PYCORTEX_FILESTORE
    cortex.database.default_filestore = PYCORTEX_FILESTORE

    surf_a = cifti_59k_to_surface(R2_a_nc, bm_axis)
    surf_b = cifti_59k_to_surface(R2_b_nc, bm_axis)

    pos_a = surf_a[np.isfinite(surf_a) & (surf_a > 0)]
    pos_b = surf_b[np.isfinite(surf_b) & (surf_b > 0)]
    p95_a = float(np.percentile(pos_a, 95)) if len(pos_a) else 0.05
    p95_b = float(np.percentile(pos_b, 95)) if len(pos_b) else 0.05

    log.info(f"\nPycortex Vertex2D:")
    log.info(f"  {ROI_A} (red)  vmax = {p95_a:.4f}")
    log.info(f"  {ROI_B} (blue) vmax = {p95_b:.4f}")

    vx = cortex.Vertex2D(
        surf_b, surf_a,
        subject=CX_SUB,
        vmin=0, vmax=p95_b,
        vmin2=0, vmax2=p95_a,
        cmap="PU_RdBu_covar_alpha",
    )
    cortex.quickshow(vx)


# =============================================================================
# MAIN
# =============================================================================

def main(min_subjects=1, no_pycortex=False):
    log.info("=" * 60)
    log.info("Script 05 — Group average + visualization")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info("=" * 60)

    log.info("\nCollecting per-subject R² maps …")
    maps_a_nc = collect_maps(SUBJECTS_DIR, f"R2_{ROI_A}_nc", min_subjects)
    maps_b_nc = collect_maps(SUBJECTS_DIR, f"R2_{ROI_B}_nc", min_subjects)
    maps_full = collect_maps(SUBJECTS_DIR, "R2_full",         min_subjects)
    maps_a    = collect_maps(SUBJECTS_DIR, f"R2_{ROI_A}",     min_subjects)
    maps_b    = collect_maps(SUBJECTS_DIR, f"R2_{ROI_B}",     min_subjects)
    n_subs    = maps_a_nc.shape[0]
    log.info(f"  Averaging across {n_subs} subjects …")

    R2_a_nc = np.nanmean(maps_a_nc, axis=0).astype(np.float32)
    R2_b_nc = np.nanmean(maps_b_nc, axis=0).astype(np.float32)
    R2_full = np.nanmean(maps_full, axis=0).astype(np.float32)
    R2_a    = np.nanmean(maps_a,    axis=0).astype(np.float32)
    R2_b    = np.nanmean(maps_b,    axis=0).astype(np.float32)

    log.info(f"  R2_{ROI_A}_nc: mean={R2_a_nc.mean():.4f}  frac>0={np.mean(R2_a_nc > 0):.1%}")
    log.info(f"  R2_{ROI_B}_nc: mean={R2_b_nc.mean():.4f}  frac>0={np.mean(R2_b_nc > 0):.1%}")

    for name, arr in [
        (f"R2_{ROI_A}_nc_avg", R2_a_nc),
        (f"R2_{ROI_B}_nc_avg", R2_b_nc),
        ("R2_full_avg",        R2_full),
        (f"R2_{ROI_A}_avg",    R2_a),
        (f"R2_{ROI_B}_avg",    R2_b),
    ]:
        np.save(os.path.join(GROUP_DIR, f"{name}.npy"), arr)
    log.info(f"  Saved group .npy files to {GROUP_DIR}")

    integration_score, modality_balance = compute_integration_maps(R2_a_nc, R2_b_nc)
    np.save(os.path.join(GROUP_DIR, "integration_score_avg.npy"), integration_score)
    np.save(os.path.join(GROUP_DIR, "modality_balance_avg.npy"),  modality_balance)
    log.info(f"  integration_score: mean={integration_score.mean():.4f}  "
             f"frac>0={np.mean(integration_score > 0):.1%}")

    log.info("\nBuilding 59k CIFTI template …")
    try:
        bm_axis = get_59k_bm_axis(SUBJECTS_DIR, CIFTI_RAW_DIR)
        log.info("  59k BrainModelAxis acquired.")

        for name, arr in [
            (f"R2_{ROI_A}_nc_avg",   R2_a_nc),
            (f"R2_{ROI_B}_nc_avg",   R2_b_nc),
            ("R2_full_avg",           R2_full),
            ("integration_score_avg", integration_score),
            ("modality_balance_avg",  modality_balance),
        ]:
            save_59k_cifti(arr, name, bm_axis, CIFTI_59K_DIR)

        save_dlabel_59k(R2_a_nc, R2_b_nc, bm_axis)

    except Exception as e:
        log.warning(f"  Could not build 59k CIFTIs: {e}")
        bm_axis = None

    log.info("\nResampling group maps 59k → 32k …")
    maps_to_resample = [
        f"R2_{ROI_A}_nc_avg",
        f"R2_{ROI_B}_nc_avg",
        "R2_full_avg",
        "integration_score_avg",
        "modality_balance_avg",
    ]
    for name in maps_to_resample:
        cifti_59k = os.path.join(CIFTI_59K_DIR, f"{name}.59k.dscalar.nii")
        if os.path.exists(cifti_59k):
            resample_to_32k(cifti_59k, name, CIFTI_32K_DIR,
                            TEMPLATE_32K,
                            SPHERE_59K_L, SPHERE_59K_R,
                            SPHERE_32K_L, SPHERE_32K_R)

    log.info("\n--- wb_view (continuous 2D dual-overlay, matches vicsompy Figure 3a) ---")
    log.info(f"  Overlay 1 (blue):  {CIFTI_32K_DIR}/R2_{ROI_B}_nc_avg.32k.dscalar.nii")
    log.info(f"  Overlay 2 (red):   {CIFTI_32K_DIR}/R2_{ROI_A}_nc_avg.32k.dscalar.nii")

    if not no_pycortex and bm_axis is not None:
        log.info("\nLaunching pycortex Vertex2D visualization …")
        try:
            show_pycortex(R2_a_nc, R2_b_nc, bm_axis)
        except Exception as e:
            log.warning(f"  Pycortex visualization failed: {e}")
    elif no_pycortex:
        log.info("\n  --no_pycortex: skipping visualization.")
    else:
        log.info("\n  No BrainModelAxis available — pycortex visualization skipped.")

    log.info("\nScript 05 complete.")
    log.info(f"  59k CIFTIs : {CIFTI_59K_DIR}")
    log.info(f"  32k CIFTIs : {CIFTI_32K_DIR}")
    log.info(f"\nNext: python 06_group_stats.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group-average R² maps, integration maps, CIFTI output."
    )
    parser.add_argument("--min_subjects", type=int, default=1,
                        help="Minimum subjects required (default 1 for pilot)")
    parser.add_argument("--no_pycortex", action="store_true",
                        help="Skip pycortex visualization (for headless servers)")
    parser.add_argument("--roi_a",       default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",       default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base", default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--hcp_dir",     default=_DEFAULT_HCP_DIR,
                        help="HCP S1200 group-average atlas directory (default: %(default)s)")
    parser.add_argument("--cifti_dir",   default=_DEFAULT_CIFTI_DIR,
                        help="Directory containing subject CIFTI dtseries files "
                             "(default: %(default)s)")
    args = parser.parse_args()

    ROI_A         = args.roi_a
    ROI_B         = args.roi_b
    HCP_DIR       = args.hcp_dir
    CIFTI_RAW_DIR = args.cifti_dir
    OUTPUT_ROOT   = f"{args.output_base}/{ROI_A}_{ROI_B}"
    SUBJECTS_DIR  = f"{OUTPUT_ROOT}/subjects"
    GROUP_DIR     = f"{OUTPUT_ROOT}/group"
    CIFTI_59K_DIR = f"{GROUP_DIR}/cifti_59k"
    CIFTI_32K_DIR = f"{GROUP_DIR}/cifti_32k"
    TEMPLATE_32K  = f"{HCP_DIR}/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
    SPHERE_59K_L  = f"{HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
    SPHERE_59K_R  = f"{HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"
    SPHERE_32K_L  = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
    SPHERE_32K_R  = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"
    PYCORTEX_FILESTORE = HCP_DIR

    for d in [GROUP_DIR, CIFTI_59K_DIR, CIFTI_32K_DIR]:
        os.makedirs(d, exist_ok=True)

    main(args.min_subjects, args.no_pycortex)
