"""
cf_modeling/00_make_roi_masks.py
==================================
Generate per-ROI CSV mask files from the 59k_fs_LR Glasser HCP-MMP1 dlabel.

These CSV masks are the input format expected by vicsompy's MssCf when building
subsurfaces from scratch (force_new=True in make_subsurfaces()).  They are also
used by this pipeline's 01_extract_geometry.py as an alternative to loading masks
directly from the dlabel (both approaches produce identical Boolean arrays).

Output format (vicsompy convention)
-------------------------------------
One CSV per ROI per hemisphere:
    {masks_dir}/{roi}_L_mask.csv
    {masks_dir}/{roi}_R_mask.csv

Each CSV has a single column named 'mask' with 59292 rows (one per sphere vertex
in the 59k_fs_LR hemisphere), containing True/False.

Usage
-----
    python cf_modeling/00_make_roi_masks.py \\
        --glasser-dlabel /path/to/Q1-Q6...59k_fs_LR.dlabel.nii \\
        --rois 3b V1 A1 V2 TA2 MST FFC A5 \\
        --masks-dir /path/to/cf_modeling_outputs/masks

Or as part of analysis.sh:
    run_python 00_make_roi_masks.py --glasser-dlabel "$GLASSER_DLABEL" \\
        --rois 3b V1 --masks-dir "$MASKS_DIR"

Note: this script is OPTIONAL.  01_extract_geometry.py reads the dlabel directly
for ROI masks.  Run 00_make_roi_masks.py only if you need the CSV files for
other purposes (e.g., sharing ROI masks or using force_new=True subsurfaces).
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_VERTS_PER_HEM = 59292  # 59k_fs_LR: 59292 sphere vertices per hemisphere


# =============================================================================
# Glasser dlabel loading
# =============================================================================

def load_dlabel_masks(glasser_dlabel: str, roi_names: list) -> dict:
    """Extract Boolean vertex masks for each ROI from the Glasser dlabel.

    Parameters
    ----------
    glasser_dlabel : str — path to the 59k_fs_LR Glasser dlabel.nii.
    roi_names      : list of str — Glasser short names (e.g. ['3b', 'V1']).

    Returns
    -------
    masks : dict[str, (mask_L, mask_R)]
        mask_L : (59292,) bool — True at left-hemisphere ROI vertices.
        mask_R : (59292,) bool — True at right-hemisphere ROI vertices.
    """
    log.info("Loading Glasser dlabel: %s", os.path.basename(glasser_dlabel))
    img  = nib.load(glasser_dlabel)
    data = img.get_fdata(dtype=np.float32)[0]   # (118584,) integer label codes

    # Build name→label-key lookup from the dlabel's label table.
    ax0        = img.header.get_axis(0)
    name2key   = {name: key for key, (name, _) in ax0.label[0].items()}

    masks = {}
    for roi in roi_names:
        lk = name2key.get(f"L_{roi}_ROI")
        rk = name2key.get(f"R_{roi}_ROI")

        if lk is None or rk is None:
            available = sorted(
                n.replace("L_", "").replace("_ROI", "")
                for n in name2key
                if n.startswith("L_") and n.endswith("_ROI")
            )
            raise ValueError(
                f"ROI '{roi}' not found in dlabel.  "
                f"Available Glasser ROI names:\n  {available}"
            )

        mask_L = (data[:N_VERTS_PER_HEM] == lk)   # (59292,) bool
        mask_R = (data[N_VERTS_PER_HEM:] == rk)   # (59292,) bool

        masks[roi] = (mask_L, mask_R)
        log.info(
            "  %-6s  L=%4d verts (label=%d)  R=%4d verts (label=%d)",
            roi, int(mask_L.sum()), lk, int(mask_R.sum()), rk,
        )

    return masks


# =============================================================================
# CSV mask saving (vicsompy format)
# =============================================================================

def save_masks(masks: dict, masks_dir: str) -> None:
    """Write {roi}_{L/R}_mask.csv files in vicsompy's expected format.

    Format: single column 'mask' with 59292 rows, values True or False.

    Parameters
    ----------
    masks     : output of load_dlabel_masks().
    masks_dir : str — output directory (created if needed).
    """
    os.makedirs(masks_dir, exist_ok=True)
    for roi, (mask_L, mask_R) in masks.items():
        for hem, mask in [("L", mask_L), ("R", mask_R)]:
            out_path = os.path.join(masks_dir, f"{roi}_{hem}_mask.csv")
            pd.DataFrame({"mask": mask.astype(bool)}).to_csv(out_path, index=False)
            log.info("  Saved: %s  (%d True)", os.path.basename(out_path), int(mask.sum()))
    log.info("Masks saved to: %s", masks_dir)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate per-ROI CSV mask files from the Glasser HCP-MMP1 "
            "59k_fs_LR dlabel.nii.  Output: {roi}_{L/R}_mask.csv with a "
            "'mask' column (59292 rows, bool)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--glasser-dlabel", required=True, dest="glasser_dlabel",
        help="Path to Q1-Q6_RelatedParcellation210...59k_fs_LR.dlabel.nii",
    )
    p.add_argument(
        "--rois", required=True, nargs="+", dest="rois",
        help=(
            "One or more Glasser ROI short names (e.g. 3b V1 A1 TA2). "
            "Checked against L_{roi}_ROI and R_{roi}_ROI in the dlabel label table."
        ),
    )
    p.add_argument(
        "--masks-dir", required=True, dest="masks_dir",
        help="Output directory for CSV mask files.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("00 — Generate ROI CSV masks from Glasser dlabel")
    log.info("  ROIs     : %s", args.rois)
    log.info("  dlabel   : %s", os.path.basename(args.glasser_dlabel))
    log.info("  masks-dir: %s", args.masks_dir)
    log.info("=" * 60)

    masks = load_dlabel_masks(args.glasser_dlabel, args.rois)
    save_masks(masks, args.masks_dir)

    log.info("Done.  %d ROIs → %d CSV files in %s",
             len(args.rois), 2 * len(args.rois), args.masks_dir)


if __name__ == "__main__":
    main()
