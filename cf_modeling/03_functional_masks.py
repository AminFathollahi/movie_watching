"""
cf_modeling/03_functional_masks.py
=====================================
Derive functional ROI masks from a completed CF model run.

Workflow
--------
1.  Load R2_full, R2_{src_a}_nc, R2_{src_b}_nc from the source CF model's
    prep/ (group_average) or group/ (per_subject) directory.
2.  Restrict to valid vertices: R2_full > --min-r2-full (default 0).
3.  Apply a relative threshold within valid vertices to define:
      "{output_roi_a}" mask  — vertices where R2_{src_a}_nc is above the
                               q-th percentile (default 50th) of all valid
                               vertices; floor at R2_nc > 0.
      "{output_roi_b}" mask  — same logic for ROI B.
4.  Resolve any overlap between the two masks (configurable):
      winner  — overlapping vertex goes to whichever ROI has higher R2_nc.
      allow   — both masks keep the overlapping vertex.
      exclude — overlapping vertices are removed from both masks.
5.  Project grayordinate boolean masks (59412) → full-sphere space (118584)
    using the CIFTI BrainModelAxis, then split into L/R (59292 each).
6.  Save vicsompy-format CSV masks:
      {masks_dir}/{output_roi_a}_{L/R}_mask.csv
      {masks_dir}/{output_roi_b}_{L/R}_mask.csv
7.  Save an inspection CIFTI dscalar (2 maps: final mask_a + mask_b).

Usage
-----
  python cf_modeling/03_functional_masks.py \\
      --source-roi-a A1 --source-roi-b V1 \\
      --output-roi-a auditory_cx --output-roi-b visual_cx \\
      --mode group_average \\
      --output-base /path/to/outputs/cf_modeling \\
      --masks-dir   /path/to/outputs/cf_modeling/masks \\
      --template-cifti /path/to/cortex_59k.dtseries.nii \\
      --threshold-method percentile --threshold-q 50 \\
      --overlap-method winner
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Local lib imports
# ---------------------------------------------------------------------------
_CF_DIR = Path(__file__).resolve().parent
if str(_CF_DIR) not in sys.path:
    sys.path.insert(0, str(_CF_DIR))

from lib.data_adapter import grayord_to_sphere_space, N_VERTS_PER_HEM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_OUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"


# =============================================================================
# Helpers
# =============================================================================

def _load_r2_maps(output_base: str, mode: str, src_a: str, src_b: str) -> tuple:
    """Load R2_full, R2_a_nc, R2_b_nc from the appropriate prep/group dir."""
    roi_tag = f"{src_a}_{src_b}"
    if mode == "group_average":
        prep_dir = os.path.join(output_base, "group_average", roi_tag, "prep")
    else:
        prep_dir = os.path.join(output_base, "per_subject", roi_tag, "group")

    for name in ["R2_full", f"R2_{src_a}_nc", f"R2_{src_b}_nc"]:
        path = os.path.join(prep_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required map not found: {path}\n"
                f"Run 02_fit_cf_model.py (and integration_maps.py for per_subject) first."
            )

    R2_full = np.load(os.path.join(prep_dir, "R2_full.npy")).astype(np.float32)
    R2_a_nc = np.load(os.path.join(prep_dir, f"R2_{src_a}_nc.npy")).astype(np.float32)
    R2_b_nc = np.load(os.path.join(prep_dir, f"R2_{src_b}_nc.npy")).astype(np.float32)

    log.info("Loaded maps from: %s", prep_dir)
    log.info("  R2_full : shape=%s  mean=%.4f  frac>0=%.1f%%",
             R2_full.shape, float(np.nanmean(R2_full)),
             100.0 * float(np.mean(R2_full > 0)))
    log.info("  R2_%s_nc: shape=%s  mean=%.4f  frac>0=%.1f%%",
             src_a, R2_a_nc.shape, float(np.nanmean(R2_a_nc)),
             100.0 * float(np.mean(R2_a_nc > 0)))
    log.info("  R2_%s_nc: shape=%s  mean=%.4f  frac>0=%.1f%%",
             src_b, R2_b_nc.shape, float(np.nanmean(R2_b_nc)),
             100.0 * float(np.mean(R2_b_nc > 0)))
    return R2_full, R2_a_nc, R2_b_nc


def _compute_mask(R2_nc: np.ndarray, valid: np.ndarray,
                  method: str, q: float, abs_thr: float,
                  roi_name: str) -> np.ndarray:
    """Apply threshold within valid vertices; floor at R2_nc > 0.

    Returns
    -------
    mask : (n_grayord,) bool
    """
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RuntimeError("No valid vertices (R2_full > min_r2_full) found.")

    if method == "percentile":
        raw_thr = float(np.percentile(R2_nc[valid], q))
        if raw_thr < 0:
            log.warning(
                "  [%s] %d-th percentile threshold is %.4f < 0; clamping to 0.",
                roi_name, q, raw_thr,
            )
            raw_thr = 0.0
        thr = raw_thr
    else:
        thr = abs_thr

    mask = valid & (R2_nc > thr)
    log.info(
        "  [%s] threshold=%.4f (%s q=%.0f)  valid=%d  selected=%d (%.1f%%)",
        roi_name, thr, method, q, n_valid, int(mask.sum()),
        100.0 * float(mask.sum()) / max(n_valid, 1),
    )
    return mask


def _resolve_overlap(mask_a: np.ndarray, mask_b: np.ndarray,
                     R2_a_nc: np.ndarray, R2_b_nc: np.ndarray,
                     method: str) -> tuple:
    """Resolve overlap between mask_a and mask_b.

    Parameters
    ----------
    method : "winner" | "allow" | "exclude"
    """
    overlap = mask_a & mask_b
    n_overlap = int(overlap.sum())

    if n_overlap == 0:
        log.info("  Overlap: 0 vertices — no resolution needed.")
        return mask_a, mask_b

    log.info("  Overlap: %d vertices (%.1f%% of mask_a, %.1f%% of mask_b)",
             n_overlap,
             100.0 * n_overlap / max(int(mask_a.sum()), 1),
             100.0 * n_overlap / max(int(mask_b.sum()), 1))

    if method == "winner":
        a_wins = overlap & (R2_a_nc >= R2_b_nc)
        b_wins = overlap & (R2_b_nc > R2_a_nc)
        mask_a = (mask_a & ~overlap) | a_wins
        mask_b = (mask_b & ~overlap) | b_wins
        log.info("  winner: %d → a, %d → b", int(a_wins.sum()), int(b_wins.sum()))
    elif method == "exclude":
        mask_a = mask_a & ~overlap
        mask_b = mask_b & ~overlap
        log.info("  exclude: removed %d from both", n_overlap)
    else:  # allow
        log.info("  allow: keeping %d vertices in both masks", n_overlap)

    return mask_a, mask_b


def _grayord_mask_to_sphere_hemispheres(
    mask_grayord: np.ndarray, bm_axis
) -> tuple:
    """Convert (n_grayord,) bool mask → (mask_L, mask_R) each (N_VERTS_PER_HEM,) bool."""
    sphere = grayord_to_sphere_space(mask_grayord.astype(np.float32), bm_axis)
    mask_L = sphere[:N_VERTS_PER_HEM] > 0.5
    mask_R = sphere[N_VERTS_PER_HEM:] > 0.5
    return mask_L, mask_R


def _save_csv_masks(roi_name: str, mask_L: np.ndarray, mask_R: np.ndarray,
                    masks_dir: str) -> None:
    """Write {roi_name}_{L/R}_mask.csv in vicsompy format."""
    for hem, mask in [("L", mask_L), ("R", mask_R)]:
        out_path = os.path.join(masks_dir, f"{roi_name}_{hem}_mask.csv")
        pd.DataFrame({"mask": mask.astype(bool)}).to_csv(out_path, index=False)
        log.info("  Saved: %s  (%d True / %d total)",
                 os.path.basename(out_path), int(mask.sum()), len(mask))


def _save_inspection_cifti(
    mask_a_grayord: np.ndarray,
    mask_b_grayord: np.ndarray,
    valid_grayord: np.ndarray,
    out_roi_a: str,
    out_roi_b: str,
    template_cifti: str,
    masks_dir: str,
) -> None:
    """Save a 3-map dscalar and a dlabel for visual inspection in wb_view."""
    bm_ax = nib.load(template_cifti).header.get_axis(1)

    # 3-map dscalar: [mask_a, mask_b, valid]
    data_3 = np.stack([
        mask_a_grayord.astype(np.float32),
        mask_b_grayord.astype(np.float32),
        valid_grayord.astype(np.float32),
    ], axis=0)
    map_names = [f"mask_{out_roi_a}", f"mask_{out_roi_b}", "valid_R2_full"]
    scalar_ax = nib.cifti2.ScalarAxis(map_names)
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_ax, bm_ax))
    out_path = os.path.join(masks_dir, f"functional_masks_{out_roi_a}_{out_roi_b}.dscalar.nii")
    nib.save(nib.Cifti2Image(data_3, header=hdr), out_path)
    log.info("  Saved inspection CIFTI: %s", os.path.basename(out_path))

    # dlabel: 0=neither, 1=out_roi_a only, 2=out_roi_b only, 3=both
    label_map = np.zeros(len(mask_a_grayord), dtype=np.int32)
    label_map[mask_a_grayord & ~mask_b_grayord] = 1
    label_map[~mask_a_grayord & mask_b_grayord] = 2
    label_map[mask_a_grayord & mask_b_grayord]  = 3

    label_table = nib.cifti2.Cifti2LabelTable()
    for key, (name, r, g, b, a) in {
        0: ("Neither",                        0.6,  0.6,  0.6,  1.0),
        1: (f"{out_roi_a}_only",              0.85, 0.15, 0.15, 1.0),
        2: (f"{out_roi_b}_only",              0.15, 0.15, 0.85, 1.0),
        3: (f"Both_{out_roi_a}_{out_roi_b}",  0.65, 0.10, 0.75, 1.0),
    }.items():
        label_table[key] = nib.cifti2.Cifti2Label(key, name, r, g, b, a)

    map_name   = f"functional_masks_{out_roi_a}_{out_roi_b}"
    label_axis = nib.cifti2.LabelAxis([map_name], [label_table])
    hdr_lbl    = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_ax))
    dlabel_path = os.path.join(masks_dir, f"{map_name}.dlabel.nii")
    nib.save(nib.Cifti2Image(label_map.reshape(1, -1).astype(np.int32),
                              header=hdr_lbl), dlabel_path)
    log.info("  Saved inspection dlabel: %s  (bimodal=%.1f%%)",
             os.path.basename(dlabel_path),
             100.0 * float(np.mean(label_map == 3)))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Derive functional ROI masks from a completed CF model run "
            "(group_average or per_subject) and save in vicsompy CSV format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-roi-a",  required=True, dest="source_roi_a",
                   help="ROI A used in the source CF model run (e.g. A1).")
    p.add_argument("--source-roi-b",  required=True, dest="source_roi_b",
                   help="ROI B used in the source CF model run (e.g. V1).")
    p.add_argument("--output-roi-a",  default=None, dest="output_roi_a",
                   help="Name for the ROI-A-derived mask (default: {src_a}_fx).")
    p.add_argument("--output-roi-b",  default=None, dest="output_roi_b",
                   help="Name for the ROI-B-derived mask (default: {src_b}_fx).")
    p.add_argument("--mode",          required=True, choices=["group_average", "per_subject"],
                   help="Must match the mode used for the source CF model run.")
    p.add_argument("--output-base",   default=_OUT_BASE, dest="output_base",
                   help="CF modeling output root (source maps loaded from here).")
    p.add_argument("--masks-dir",     default=None, dest="masks_dir",
                   help="Output directory for CSV mask files "
                        "(default: {output_base}/masks).")
    p.add_argument("--template-cifti", required=True, dest="template_cifti",
                   help="59k cortex-only CIFTI template for BrainModelAxis.")
    p.add_argument("--min-r2-full",   default=0.0, type=float, dest="min_r2_full",
                   help="Minimum R2_full to consider a vertex valid.")
    p.add_argument("--threshold-method", default="percentile",
                   choices=["percentile", "absolute"], dest="threshold_method",
                   help="Threshold method for defining masks.")
    p.add_argument("--threshold-q",   default=50.0, type=float, dest="threshold_q",
                   help="Percentile of R2_nc among valid vertices (used if "
                        "--threshold-method percentile).  Floored to 0.")
    p.add_argument("--threshold-abs", default=0.0, type=float, dest="threshold_abs",
                   help="Absolute R2_nc threshold (used if --threshold-method absolute).")
    p.add_argument("--overlap-method", default="winner",
                   choices=["winner", "allow", "exclude"], dest="overlap_method",
                   help="How to resolve vertices selected by both masks: "
                        "winner=higher R2_nc wins, allow=keep in both, "
                        "exclude=remove from both.")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    out_roi_a = args.output_roi_a or f"{args.source_roi_a}_fx"
    out_roi_b = args.output_roi_b or f"{args.source_roi_b}_fx"
    masks_dir = args.masks_dir or os.path.join(args.output_base, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("03 — Functional masks from CF model results")
    log.info("  Source  : %s × %s  (%s)", args.source_roi_a, args.source_roi_b, args.mode)
    log.info("  Output  : %s  /  %s", out_roi_a, out_roi_b)
    log.info("  Thresh  : %s (q=%.0f, abs=%.4f)",
             args.threshold_method, args.threshold_q, args.threshold_abs)
    log.info("  Overlap : %s", args.overlap_method)
    log.info("  Masks → : %s", masks_dir)
    log.info("=" * 60)

    # ── Load source R² maps ───────────────────────────────────────────────────
    R2_full, R2_a_nc, R2_b_nc = _load_r2_maps(
        args.output_base, args.mode, args.source_roi_a, args.source_roi_b)

    # ── Valid vertices ────────────────────────────────────────────────────────
    valid = R2_full > args.min_r2_full
    log.info("Valid vertices (R2_full > %.4f): %d / %d  (%.1f%%)",
             args.min_r2_full, int(valid.sum()), len(valid),
             100.0 * float(valid.mean()))

    # ── Threshold → raw masks ─────────────────────────────────────────────────
    log.info("Thresholding ...")
    mask_a = _compute_mask(R2_a_nc, valid, args.threshold_method,
                           args.threshold_q, args.threshold_abs, out_roi_a)
    mask_b = _compute_mask(R2_b_nc, valid, args.threshold_method,
                           args.threshold_q, args.threshold_abs, out_roi_b)

    # ── Overlap resolution ────────────────────────────────────────────────────
    log.info("Resolving overlap (%s) ...", args.overlap_method)
    mask_a, mask_b = _resolve_overlap(mask_a, mask_b, R2_a_nc, R2_b_nc,
                                       args.overlap_method)

    # ── Grayordinate → sphere → hemispheres ───────────────────────────────────
    log.info("Projecting to sphere space ...")
    bm_axis = nib.load(args.template_cifti).header.get_axis(1)
    mask_a_L, mask_a_R = _grayord_mask_to_sphere_hemispheres(mask_a, bm_axis)
    mask_b_L, mask_b_R = _grayord_mask_to_sphere_hemispheres(mask_b, bm_axis)

    log.info("  %s: L=%d verts  R=%d verts", out_roi_a,
             int(mask_a_L.sum()), int(mask_a_R.sum()))
    log.info("  %s: L=%d verts  R=%d verts", out_roi_b,
             int(mask_b_L.sum()), int(mask_b_R.sum()))

    if mask_a_L.sum() + mask_a_R.sum() == 0:
        log.warning("  %s mask is empty — try lowering --threshold-q", out_roi_a)
    if mask_b_L.sum() + mask_b_R.sum() == 0:
        log.warning("  %s mask is empty — try lowering --threshold-q", out_roi_b)

    # ── Save CSV masks ────────────────────────────────────────────────────────
    log.info("Saving CSV masks to: %s", masks_dir)
    _save_csv_masks(out_roi_a, mask_a_L, mask_a_R, masks_dir)
    _save_csv_masks(out_roi_b, mask_b_L, mask_b_R, masks_dir)

    # ── Save inspection CIFTI ─────────────────────────────────────────────────
    log.info("Saving inspection CIFTI ...")
    _save_inspection_cifti(
        mask_a, mask_b, valid,
        out_roi_a, out_roi_b,
        args.template_cifti, masks_dir,
    )

    log.info("\nDone.  Functional masks ready.")
    log.info("Next: run 01_extract_geometry.py with --masks-dir %s "
             "--roi-a %s --roi-b %s", masks_dir, out_roi_a, out_roi_b)


if __name__ == "__main__":
    main()
