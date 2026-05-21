"""
rsa/run_glasser.py
==========================
Parcel-wise RSA using the Glasser MMP 360-parcel atlas.

**This script operates exclusively in 32k_fs_LR space.**
All inputs — fMRI CIFTI, template CIFTI, and Glasser dlabel — must be in
32k_fs_LR (32492 vertices per hemisphere).  Outputs from other analyses that
are in 59k_fs_LR (e.g. cf_modeling maps, encoding maps) must be resampled to
32k before use:

    wb_command -cifti-resample \\
        input_59k.dscalar.nii COLUMN \\
        template_32k.dscalar.nii COLUMN \\
        BARYCENTRIC BARYCENTRIC \\
        output_32k.dscalar.nii \\
        -left-spheres  S1200.L.sphere.59k_fs_LR.surf.gii S1200.L.sphere.32k_fs_LR.surf.gii \\
        -right-spheres S1200.R.sphere.59k_fs_LR.surf.gii S1200.R.sphere.32k_fs_LR.surf.gii

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Pre-filtered, pre-z-scored CIFTI produced by preprocess_individual.py
  with --timing-csv (filtered mode). The file is expected at:
    {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii

STREAMING MODE (--raw-dir):
  Raw 7T CIFTI dtseries files (59k space). The script preprocesses on-the-fly,
  saves only result maps, then frees all arrays before the next subject.

Both modes are parallel-safe: each call processes one (subject, model,
modality) tuple. GNU parallel in run_analysis.sh spawns N such processes
simultaneously.

Usage (disk mode):
  python run_glasser.py \\
      --preprocessed-dir <path> --fmri-suffix gsr_zscore_delay5s \\
      --subject group_average --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path_32k> \\
      --glasser-dlabel <path_32k> \\
      --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 2.0 --delay-sec 5.0 \\
      [--normalize] [--blockdiag] [--hrf] \\
      --method spearman [--subject avg]

Usage (streaming mode):
  python run_glasser.py \\
      --raw-dir <path> --subject 100610 --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path_32k> \\
      --glasser-dlabel <path_32k> \\
      --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 2.0 --delay-sec 5.0 --tr 1.0 \\
      [--sg-filter] [--psc] [--no-gsr] [--no-z-score] \\
      --method spearman
"""

import argparse
import logging
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.shared.rsa_utils import (
    load_fmri_cifti, preprocess_fmri,
    process_model_embeddings, compute_rdm, correlate_rdms,
)
from rsa.shared.cifti_io import get_bm_axis, save_cifti_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Glasser parcel-wise RSA on movie fMRI (32k_fs_LR space only — "
                    "see module docstring for resampling from 59k).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — mutually exclusive modes
    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk mode] Directory of pre-filtered, pre-z-scored CIFTIs "
                          "(output of preprocess_individual.py --timing-csv). "
                          "File: {dir}/{subject}_{fmri-suffix}_cortex_59k.dtseries.nii")
    inp.add_argument("--fmri-suffix", default="gsr_zscore_delay5s", dest="fmri_suffix",
                     help="[disk mode] Filename suffix that encodes preprocessing "
                          "(e.g. 'gsr_zscore_delay5s').")
    inp.add_argument("--raw-dir", default=None,
                     help="[streaming mode] Root directory of raw 7T CIFTI dtseries files. "
                          "The subject is preprocessed on-the-fly; no CIFTI is saved.")

    p.add_argument("--timing-csv", required=True,
                   help="Path to movie_timing.csv.")
    p.add_argument("--embeddings-dir", required=True,
                   help="Base directory for model embeddings.")
    p.add_argument("--template-cifti", required=True,
                   help="Template 32k_fs_LR dscalar.nii for CIFTI output header.")
    p.add_argument("--glasser-dlabel", required=True,
                   help="Glasser MMP parcellation 32k_fs_LR .dlabel.nii.")

    p.add_argument("--output-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID (used for output subdirectory; also selects "
                        "which subject to preprocess in streaming mode).")

    p.add_argument("--model", required=True,
                   help="Model name (must match subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True, choices=["v", "a", "av"],
                   help="Embedding modality: v=video, a=audio, av=joint.")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Hemodynamic delay in seconds. In disk mode: config label only. "
                        "In streaming mode: applied when filtering movie TPs.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF. Use only when fMRI was "
                        "preprocessed with --delay-sec 0.")
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score normalization of embeddings.")
    p.add_argument("--blockdiag", action="store_true",
                   help="Block-diagonal normalization (per-segment instead of global).")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"],
                   help="RDM correlation method.")
    p.add_argument("--tr", type=float, required=True,
                   help="TR in seconds.")

    # Streaming preprocessing flags (ignored in disk mode)
    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true",
                      help="Savitzky-Golay high-pass filter.")
    prep.add_argument("--psc", default=False, action="store_true",
                      help="Percent signal change normalization.")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction,
                      help="Global signal regression.")
    prep.add_argument("--z-score", default=True, action=argparse.BooleanOptionalAction,
                      dest="z_score", help="Z-score per vertex.")

    return p.parse_args()


# =============================================================================
# Glasser parcellation loading
# =============================================================================

def load_glasser_parcels(dlabel_path: str, fmri_bm_axis) -> dict:
    """Extract Glasser parcel membership mapped to fMRI grayordinate indices.

    Args:
        dlabel_path: str — path to Glasser 32k .dlabel.nii
        fmri_bm_axis: CIFTI BrainModelAxis of the fMRI file

    Returns:
        dict: {parcel_name: (n,) int32 array of fMRI grayordinate column indices}
    """
    img = nib.load(dlabel_path)
    label_data = img.get_fdata(dtype=np.float32).squeeze().astype(np.int32)
    dlabel_bm  = img.header.get_axis(1)
    label_axis = img.header.get_axis(0)

    vertex_label: dict = {}
    for name, sl, struct in dlabel_bm.iter_structures():
        if "CORTEX" not in name:
            continue
        for local_i, vidx in enumerate(struct.vertex):
            vertex_label[(name, int(vidx))] = int(label_data[sl.start + local_i])

    vertex_fmri: dict = {}
    for name, sl, struct in fmri_bm_axis.iter_structures():
        if "CORTEX" not in name:
            continue
        for local_i, vidx in enumerate(struct.vertex):
            vertex_fmri[(name, int(vidx))] = sl.start + local_i

    key_to_indices: dict = {}
    for (hem, vidx), lbl in vertex_label.items():
        if lbl == 0:
            continue
        fmri_pos = vertex_fmri.get((hem, vidx))
        if fmri_pos is not None:
            key_to_indices.setdefault(lbl, []).append(fmri_pos)

    parcels = {}
    for key, (name, _rgba) in label_axis.label[0].items():
        if key == 0:
            continue
        indices = key_to_indices.get(key, [])
        if indices:
            parcels[name] = np.array(sorted(indices), dtype=np.int32)

    log.info(f"  Loaded {len(parcels)} Glasser parcels from {Path(dlabel_path).name}")
    return parcels


# =============================================================================
# Parcel RSA
# =============================================================================

def compute_parcel_rsa(fmri: np.ndarray, model_rdm: np.ndarray,
                        parcels: dict, n_grayords: int,
                        method: str = "spearman") -> tuple[np.ndarray, np.ndarray]:
    """Compute RSA for every Glasser parcel.

    Args:
        fmri: (n_bins, n_grayords) float32
        model_rdm: (n_bins, n_bins) float64
        parcels: dict — {parcel_name: vertex_indices}
        n_grayords: int — total grayordinate count for the output maps
        method: str

    Returns:
        corr_map: (n_grayords,) float32
        pval_map: (n_grayords,) float32
    """
    corr_map = np.zeros(n_grayords, dtype=np.float32)
    pval_map = np.ones(n_grayords,  dtype=np.float32)

    for name, indices in parcels.items():
        if len(indices) < 2:
            continue
        if len(indices) >= 3:
            parcel_fmri = fmri[:, indices]
        else:
            parcel_fmri = fmri[:, indices].mean(axis=1, keepdims=True)
        parcel_rdm = compute_rdm(parcel_fmri.astype(np.float64), method="correlation")
        r, p = correlate_rdms(parcel_rdm, model_rdm, method=method)
        corr_map[indices] = r
        pval_map[indices] = p

    return corr_map, pval_map


# =============================================================================
# Output naming
# =============================================================================

def _cifti_path(args) -> str:
    """Construct the per-subject preprocessed CIFTI path from directory + suffix."""
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")


def _config_label(args) -> str:
    parts = [
        "norm" if args.normalize else "nonorm",
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        f"bin{args.bin_sec:.0f}s",
        args.method,
    ]
    if args.blockdiag:
        parts.append("blockdiag")
    return "_".join(parts)


# =============================================================================
# Core analysis (shared between disk and streaming modes)
# =============================================================================

def _run_analysis(args, fmri_data: np.ndarray, fmri_bm_axis,
                  timing_df: pd.DataFrame, config: str, out_root: Path):
    """Run Glasser RSA on pre-loaded (n_vertices, T_included) fMRI data."""
    corr_out   = out_root / f"glasser_rsa_{args.method}.dscalar.nii"
    report_out = out_root / "ranked_report.csv"
    if corr_out.exists() and report_out.exists():
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    fmri_binned = preprocess_fmri(fmri_data, timing_df, args.bin_sec, args.tr)
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"{int(args.bin_sec)}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, hrf=args.hrf,
        normalize=args.normalize, blockdiag=args.blockdiag,
        tr=args.tr,
    )

    n_bins      = min(fmri_binned.shape[0], emb.shape[0])
    fmri_binned = fmri_binned[:n_bins]
    emb         = emb[:n_bins]

    model_rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    log.info(f"  Model RDM: {model_rdm.shape}")

    parcels   = load_glasser_parcels(args.glasser_dlabel, fmri_bm_axis)
    n_grayords = fmri_binned.shape[1]
    corr_map, pval_map = compute_parcel_rsa(
        fmri_binned, model_rdm, parcels, n_grayords, method=args.method
    )

    out_root.mkdir(parents=True, exist_ok=True)
    save_cifti_map(corr_map, args.template_cifti, str(corr_out),
                   map_name=f"glasser_rsa_{args.method}")
    log.info(f"  Saved: {corr_out}")

    pval_out = out_root / "glasser_rsa_pval.dscalar.nii"
    save_cifti_map(pval_map, args.template_cifti, str(pval_out),
                   map_name="glasser_rsa_pval")

    rows = []
    for name, indices in parcels.items():
        if len(indices) == 0:
            continue
        r_val = float(corr_map[indices].mean())
        p_val = float(pval_map[indices[0]])
        rows.append({"parcel": name, "r": r_val, "p": p_val,
                     "n_vertices": len(indices)})

    report = (pd.DataFrame(rows)
              .sort_values("r", ascending=False)
              .reset_index(drop=True))
    report["rank"] = range(1, len(report) + 1)
    report.to_csv(str(report_out), index=False)
    log.info(f"  Ranked report: {report_out}")
    log.info(f"  Top-5 parcels:\n{report.head(5).to_string(index=False)}")


# =============================================================================
# Disk mode
# =============================================================================

def _run_disk(args):
    timing_df = pd.read_csv(args.timing_csv)
    config    = _config_label(args)
    out_root  = Path(args.output_dir) / args.subject / args.model / config

    cifti = _cifti_path(args)
    log.info(f"Glasser RSA: {args.model} / {args.modality} / {config}")
    log.info(f"  fMRI: {cifti}")
    log.info("  Loading fMRI ...")
    fmri_data    = load_fmri_cifti(cifti)   # (n_vertices, T_included)
    fmri_bm_axis = get_bm_axis(cifti)
    log.info(f"  fMRI loaded: {fmri_data.shape}")

    _run_analysis(args, fmri_data, fmri_bm_axis, timing_df, config, out_root)
    log.info("Done.")


# =============================================================================
# Streaming mode
# =============================================================================

def _run_streaming(args):
    from preprocess_individual import preprocess_subject_filtered

    timing_df = pd.read_csv(args.timing_csv)
    config    = _config_label(args)
    sub       = args.subject
    raw_dir   = Path(args.raw_dir)
    out_root  = Path(args.output_dir) / sub / args.model / config

    corr_out   = out_root / f"glasser_rsa_{args.method}.dscalar.nii"
    report_out = out_root / "ranked_report.csv"
    if corr_out.exists() and report_out.exists():
        log.info(f"[{sub}] Outputs already exist — skipping")
        return

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc,
        gsr=args.gsr, z_score=args.z_score,
    )

    log.info(f"[{sub}] Preprocessing raw CIFTI (delay={args.delay_sec}s) ...")
    data, bm_axis, _run_trs = preprocess_subject_filtered(
        sub, raw_dir, timing_df, args.delay_sec, args.tr, prep_args
    )
    log.info(f"[{sub}] Preprocessed: {data.shape}")

    _run_analysis(args, data, bm_axis, timing_df, config, out_root)
    del data
    log.info(f"[{sub}] Done.")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.preprocessed_dir and args.raw_dir:
        log.error("--preprocessed-dir and --raw-dir are mutually exclusive.")
        sys.exit(1)
    if not args.preprocessed_dir and not args.raw_dir:
        log.error("Either --preprocessed-dir (disk mode) or --raw-dir (streaming mode) is required.")
        sys.exit(1)

    if args.raw_dir:
        _run_streaming(args)
    else:
        _run_disk(args)


if __name__ == "__main__":
    main()
