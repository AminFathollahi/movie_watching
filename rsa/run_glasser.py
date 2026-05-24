"""
rsa/run_glasser.py
==========================
Parcel-wise RSA using the Glasser MMP 360-parcel atlas in 59k.

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Continuous, cleaned CIFTI produced by preprocess_individual.py.
  Expected at:
    {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii
    {preprocessed_dir}/{subject}_{fmri_suffix}_run_trs.npy

STREAMING MODE (--raw-dir):
  Raw 7T CIFTI dtseries files. The script preprocesses on-the-fly,
  saves only result maps, then frees all arrays before the next subject.

Both modes are parallel-safe: each call processes one (subject, model,
modality) tuple. GNU parallel in run_analysis.sh spawns N such processes
simultaneously.

Usage (disk mode):
  python run_glasser.py \
      --preprocessed-dir <path> --fmri-suffix sg_psc_gsr \
      --subject group_average --timing-csv <path> \
      --embeddings-dir <path> --template-cifti <path_59k> \
      --glasser-dlabel <path_59k> \
      --output-dir <path> \
      --model pe-av-small-16-frame --modality av \
      --bin-sec 2.0 --delay-sec 5.0 \
      --method spearman [--subject avg]
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
    align_and_assert_bins
)
from rsa.shared.cifti_io import (
    get_bm_axis, save_cifti_multimap,
    get_combined_map_names, merge_into_combined,
)

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
        description="Glasser parcel-wise RSA on movie fMRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — mutually exclusive modes
    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk mode] Directory of continuous, cleaned CIFTIs.")
    inp.add_argument("--fmri-suffix", default="sg_psc_gsr", dest="fmri_suffix",
                     help="[disk mode] Filename suffix that encodes preprocessing.")
    inp.add_argument("--raw-dir", default=None,
                     help="[streaming mode] Root directory of raw 7T CIFTI files.")

    p.add_argument("--timing-csv", required=True,
                   help="Path to movie_timing.csv.")
    p.add_argument("--embeddings-dir", required=True,
                   help="Base directory for model embeddings.")
    p.add_argument("--template-cifti", required=True,
                   help="Template dscalar.nii for CIFTI output header.")
    p.add_argument("--glasser-dlabel", required=True,
                   help="Glasser MMP parcellation .dlabel.nii.")

    p.add_argument("--output-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID.")

    p.add_argument("--model", required=True,
                   help="Model name (must match subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True, choices=["v", "a", "av"],
                   help="Embedding modality: v=video, a=audio, av=joint.")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Hemodynamic delay applied when slicing stimulus blocks.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF.")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"],
                   help="RDM correlation method.")
    p.add_argument("--tr", type=float, required=True,
                   help="TR in seconds.")
    p.add_argument("--combined-output", default=None, dest="combined_output",
                   help="Path to a combined .dscalar.nii shared with run_searchlight.py. "
                        "This script adds/replaces the 'glasser_{method}_rho' map. "
                        "The individual output file is still saved alongside.")

    # Streaming preprocessing flags
    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true",
                      help="Savitzky-Golay high-pass filter.")
    prep.add_argument("--psc", default=False, action="store_true",
                      help="Percent signal change normalization.")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction,
                      help="Global signal regression.")

    return p.parse_args()


# =============================================================================
# Glasser parcellation loading
# =============================================================================

def load_glasser_parcels(dlabel_path: str, fmri_bm_axis) -> dict:
    """Extract Glasser parcel membership mapped to fMRI grayordinate indices."""
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
    """Compute RSA for every Glasser parcel."""
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
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")

def _run_trs_path(args) -> str:
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_run_trs.npy")


def _config_label(args) -> str:
    parts = [
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        f"bin{args.bin_sec:.0f}s",
        args.method,
    ]
    return "_".join(parts)


def _streaming_fmri_tag(args) -> str:
    parts = []
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    if args.gsr:       parts.append("gsr")
    return "_".join(parts) if parts else "raw"


# =============================================================================
# Core analysis (shared between disk and streaming modes)
# =============================================================================

def _run_analysis(args, fmri_continuous: np.ndarray, run_trs: np.ndarray, fmri_bm_axis,
                  timing_df: pd.DataFrame, config: str, out_root: Path,
                  fmri_tag: str):
    """Run Glasser RSA on pre-loaded continuous fMRI data."""
    bin_sec_int   = int(args.bin_sec)
    delay_tag     = f"delay{int(args.delay_sec)}s"
    maps_out      = out_root / f"glasser_rsa_{fmri_tag}_{delay_tag}_bin{bin_sec_int}_{args.method}_maps.dscalar.nii"
    report_out    = out_root / "ranked_report.csv"
    map_name      = f"glasser_{args.method}_rho"
    combined_path = Path(args.combined_output) if args.combined_output else None

    # ── Skip / fast-merge logic ───────────────────────────────────────────────
    if maps_out.exists() and report_out.exists():
        if combined_path is None:
            log.info(f"Outputs already exist — skipping: {out_root}")
            return
        if map_name in get_combined_map_names(combined_path):
            log.info(f"Outputs already exist and combined up to date — skipping: {out_root}")
            return
        # Individual done, combined missing this map → merge without recomputing
        log.info(f"  Individual map exists; merging '{map_name}' into combined ...")
        corr_map = nib.load(str(maps_out)).get_fdata(dtype=np.float32).squeeze()
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        merge_into_combined(corr_map, map_name, combined_path, args.template_cifti)
        return

    fmri_binned = preprocess_fmri(
        fmri_continuous, timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec
    )
    log.info(f"  fMRI binned & z-scored: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"{int(args.bin_sec)}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec,
        tr=args.tr,
    )

    # Enforce exact temporal alignment
    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)

    model_rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    log.info(f"  Model RDM: {model_rdm.shape}")

    parcels    = load_glasser_parcels(args.glasser_dlabel, fmri_bm_axis)
    n_grayords = fmri_binned.shape[1]
    corr_map, pval_map = compute_parcel_rsa(
        fmri_binned, model_rdm, parcels, n_grayords, method=args.method
    )

    out_root.mkdir(parents=True, exist_ok=True)
    save_cifti_multimap(
        corr_map.reshape(1, -1),
        [f"{args.method}_rho"],
        args.template_cifti,
        str(maps_out),
    )
    log.info(f"  Saved: {maps_out.name}")

    # ── Merge into combined output ────────────────────────────────────────────
    if combined_path is not None:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        merge_into_combined(corr_map, map_name, combined_path, args.template_cifti)

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
    fmri_tag  = args.fmri_suffix
    out_root  = Path(args.output_dir) / args.subject / args.model / config

    cifti = _cifti_path(args)
    trs_path = _run_trs_path(args)
    
    log.info(f"Glasser RSA: {args.model} / {args.modality} / {config}")
    log.info(f"  fMRI: {cifti}")
    log.info("  Loading fMRI ...")
    
    fmri_data    = load_fmri_cifti(cifti)
    fmri_bm_axis = get_bm_axis(cifti)
    run_trs      = np.load(trs_path)
    
    log.info(f"  fMRI loaded: {fmri_data.shape}  run_trs: {run_trs.tolist()}")

    _run_analysis(args, fmri_data, run_trs, fmri_bm_axis, timing_df, config, out_root, fmri_tag)
    log.info("Done.")


# =============================================================================
# Streaming mode
# =============================================================================

def _run_streaming(args):
    from preprocess_individual import preprocess_subject

    timing_df = pd.read_csv(args.timing_csv)
    config    = _config_label(args)
    fmri_tag  = _streaming_fmri_tag(args)
    sub       = args.subject
    raw_dir   = Path(args.raw_dir)
    out_root  = Path(args.output_dir) / sub / args.model / config

    bin_sec_int = int(args.bin_sec)
    delay_tag   = f"delay{int(args.delay_sec)}s"
    maps_out    = out_root / f"glasser_rsa_{fmri_tag}_{delay_tag}_bin{bin_sec_int}_{args.method}_maps.dscalar.nii"
    report_out  = out_root / "ranked_report.csv"
    
    if maps_out.exists() and report_out.exists():
        log.info(f"[{sub}] Outputs already exist — skipping")
        return

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr
    )

    log.info(f"[{sub}] Preprocessing raw CIFTI (continuous mode) ...")
    data, bm_axis, run_trs = preprocess_subject(
        sub, raw_dir, args.tr, prep_args
    )
    log.info(f"[{sub}] Preprocessed continuous: {data.shape}  run_trs: {run_trs.tolist()}")

    _run_analysis(args, data, run_trs, bm_axis, timing_df, config, out_root, fmri_tag)
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