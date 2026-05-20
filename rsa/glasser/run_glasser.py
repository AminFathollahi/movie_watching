"""
rsa/glasser/run_glasser.py
==========================
Parcel-wise RSA using the Glasser MMP 360-parcel atlas.

For each parcel, the mean fMRI time series across vertices is computed and a
representational dissimilarity matrix (RDM) is built. That parcel RDM is then
correlated with the model RDM. Results are saved as CIFTI dscalar maps and a
ranked CSV report.

The Glasser atlas is provided as a dense-label CIFTI (.dlabel.nii). Both 32k
and 59k resolutions are supported — pass the file matching your fMRI data.

Usage:
  python run_glasser.py \\
      --fmri-cifti <path> --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path> \\
      --glasser-dlabel <path> \\
      --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 2.0 --delay-sec 5.0 \\
      [--normalize] [--blockdiag] [--hrf] \\
      --method spearman [--subject avg]
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Shared RSA utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Glasser parcel-wise RSA on movie fMRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--fmri-cifti", required=True,
                   help="Preprocessed 59k CIFTI dtseries (group-avg or per-subject).")
    p.add_argument("--timing-csv", required=True,
                   help="Path to movie_timing.csv.")
    p.add_argument("--embeddings-dir", required=True,
                   help="Base directory for model embeddings.")
    p.add_argument("--template-cifti", required=True,
                   help="Template dscalar.nii for CIFTI output header.")
    p.add_argument("--glasser-dlabel", required=True,
                   help="Glasser MMP parcellation .dlabel.nii (59k or 32k).")

    # Output
    p.add_argument("--output-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID or 'group_average' (used for output subdirectory).")

    # Analysis config
    p.add_argument("--model", required=True,
                   help="Model name (must match subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True, choices=["v", "a", "av"],
                   help="Embedding modality: v=video, a=audio, av=joint.")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, required=True,
                   help="Hemodynamic delay in seconds.")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF instead of boxcar delay.")
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score normalization of embeddings.")
    p.add_argument("--blockdiag", action="store_true",
                   help="Block-diagonal normalization (per-segment instead of global).")
    p.add_argument("--method", required=True, choices=["spearman", "pearson"],
                   help="RDM correlation method.")
    p.add_argument("--tr", type=float, required=True,
                   help="TR in seconds.")

    return p.parse_args()


# =============================================================================
# Glasser parcellation loading
# =============================================================================

def load_glasser_parcels(dlabel_path: str, fmri_bm_axis) -> dict:
    """Extract Glasser parcel membership mapped to fMRI grayordinate indices.

    The Glasser 59k atlas covers all surface vertices including the medial wall,
    while the fMRI CIFTI covers only non-medial-wall cortical vertices.  Parcels
    are matched by (hemisphere, vertex_index) so returned indices directly address
    columns of the fMRI time series array.

    Args:
        dlabel_path:  str — path to Glasser .dlabel.nii (59k atlas)
        fmri_bm_axis: CIFTI BrainModelAxis of the fMRI file

    Returns:
        dict: {parcel_name: (n,) int32 array of fMRI grayordinate column indices}
        Only parcels with at least one vertex in the fMRI are included.
        Parcel index 0 (medial wall / background) is excluded.
    """
    img = nib.load(dlabel_path)
    label_data = img.get_fdata(dtype=np.float32).squeeze().astype(np.int32)
    dlabel_bm  = img.header.get_axis(1)
    label_axis = img.header.get_axis(0)

    # (hemisphere_name, vertex_index) → parcel label key
    vertex_label: dict = {}
    for name, sl, struct in dlabel_bm.iter_structures():
        if "CORTEX" not in name:
            continue
        for local_i, vidx in enumerate(struct.vertex):
            vertex_label[(name, int(vidx))] = int(label_data[sl.start + local_i])

    # (hemisphere_name, vertex_index) → fMRI grayordinate column position
    vertex_fmri: dict = {}
    for name, sl, struct in fmri_bm_axis.iter_structures():
        if "CORTEX" not in name:
            continue
        for local_i, vidx in enumerate(struct.vertex):
            vertex_fmri[(name, int(vidx))] = sl.start + local_i

    # Accumulate fMRI column indices per parcel key
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
        method: str — "spearman" or "pearson"

    Returns:
        corr_map: (n_grayords,) float32 — each vertex assigned its parcel's r value
        pval_map: (n_grayords,) float32 — each vertex assigned its parcel's p value
    """
    corr_map = np.zeros(n_grayords, dtype=np.float32)
    pval_map = np.ones(n_grayords, dtype=np.float32)

    for name, indices in parcels.items():
        if len(indices) < 2:
            continue  # need at least 2 vertices for a meaningful RDM
        parcel_fmri = fmri[:, indices].mean(axis=1, keepdims=True)  # (n_bins, 1)
        # With a single averaged signal the parcel RDM is trivially all-zeros;
        # use the multi-vertex representation for a proper RDM
        if len(indices) >= 3:
            parcel_fmri = fmri[:, indices]  # (n_bins, n_vertices_in_parcel)
        parcel_rdm = compute_rdm(parcel_fmri.astype(np.float64), method="correlation")
        r, p = correlate_rdms(parcel_rdm, model_rdm, method=method)
        corr_map[indices] = r
        pval_map[indices] = p

    return corr_map, pval_map


# =============================================================================
# Output naming
# =============================================================================

def _config_label(args) -> str:
    """Build a descriptive label for the current analysis configuration."""
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
# Main
# =============================================================================

def main():
    args = parse_args()

    timing_df = pd.read_csv(args.timing_csv)
    config = _config_label(args)

    out_root = Path(args.output_dir) / args.subject / args.model / config
    out_root.mkdir(parents=True, exist_ok=True)

    # Skip if outputs already exist
    corr_out = out_root / f"glasser_rsa_{args.method}.dscalar.nii"
    report_out = out_root / "ranked_report.csv"
    if corr_out.exists() and report_out.exists():
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    log.info(f"Glasser RSA: {args.model} / {args.modality} / {config}")

    # Load and preprocess fMRI
    # When hrf=True, embeddings are HRF-convolved; fMRI window is not shifted.
    # When hrf=False, fMRI window is shifted by delay_sec; embeddings are unshifted.
    fmri_delay = 0.0 if args.hrf else args.delay_sec
    log.info("  Loading fMRI ...")
    fmri_raw = load_fmri_cifti(args.fmri_cifti)  # (n_vertices, T_total)
    fmri_binned = preprocess_fmri(fmri_raw, timing_df, args.bin_sec, fmri_delay,
                                   args.tr)  # (n_bins, n_vertices)
    log.info(f"  fMRI binned: {fmri_binned.shape}")

    # Load and process embeddings
    emb_file = (Path(args.embeddings_dir) / args.model /
                f"{int(args.bin_sec)}s" / f"{args.model}_{args.modality}.npy")
    log.info(f"  Embeddings: {emb_file}")
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, hrf=args.hrf,
        normalize=args.normalize, blockdiag=args.blockdiag,
        tr=args.tr,
    )

    # Safety trim in case of off-by-one from rounding
    n_bins = min(fmri_binned.shape[0], emb.shape[0])
    fmri_binned = fmri_binned[:n_bins]
    emb = emb[:n_bins]

    model_rdm = compute_rdm(emb.astype(np.float64), method="correlation")
    log.info(f"  Model RDM: {model_rdm.shape}")

    # Load Glasser parcels (vertex-index mapping to fMRI grayordinates)
    fmri_bm_axis = get_bm_axis(args.fmri_cifti)
    parcels = load_glasser_parcels(args.glasser_dlabel, fmri_bm_axis)

    n_grayords = fmri_binned.shape[1]
    corr_map, pval_map = compute_parcel_rsa(
        fmri_binned, model_rdm, parcels, n_grayords, method=args.method
    )

    # Save CIFTI maps
    save_cifti_map(corr_map, args.template_cifti, str(corr_out),
                   map_name=f"glasser_rsa_{args.method}")
    log.info(f"  Saved: {corr_out}")

    pval_out = out_root / f"glasser_rsa_pval.dscalar.nii"
    save_cifti_map(pval_map, args.template_cifti, str(pval_out),
                   map_name="glasser_rsa_pval")

    # Ranked CSV report — one row per parcel with its r and p values
    rows = []
    for name, indices in parcels.items():
        if len(indices) == 0:
            continue
        r_val = float(corr_map[indices].mean())
        p_val = float(pval_map[indices[0]])
        rows.append({"parcel": name, "r": r_val, "p": p_val,
                     "n_vertices": len(indices)})

    import pandas as _pd
    report = _pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)
    report["rank"] = range(1, len(report) + 1)
    report.to_csv(str(report_out), index=False)
    log.info(f"  Ranked report: {report_out}")

    log.info(f"  Top-5 parcels:\n{report.head(5).to_string(index=False)}")
    log.info("Done.")


if __name__ == "__main__":
    main()
