"""
rsa/run_rdm_diagonal.py
========================
Compute within-movie (block-diagonal) and cross-movie (off-diagonal) RDMs.

Given a set of temporal embeddings and a timing CSV, this script:
  1. Assigns each temporal bin to its source video (movie).
  2. Computes the full pairwise RDM from the model embeddings.
  3. Produces two masked variants:
       off-diagonal  : NaN on all pairs where both bins come from the *same*
                       movie — retains only cross-movie similarity structure.
       block-diagonal: NaN on all pairs where bins come from *different* movies
                       — retains only within-movie similarity structure.

The within-movie (block-diagonal) RDM measures narrative/scene-level temporal
similarity.  The cross-movie (off-diagonal) RDM probes semantic generalisation
across distinct narratives.

Outputs
-------
  {output_dir}/{model}_{modality}_bin{bin_sec}s_rdm_full.npy        (n_bins, n_bins)
  {output_dir}/{model}_{modality}_bin{bin_sec}s_rdm_offdiag.npy     (n_bins, n_bins) — NaN within-movie
  {output_dir}/{model}_{modality}_bin{bin_sec}s_rdm_blockdiag.npy   (n_bins, n_bins) — NaN cross-movie
  {output_dir}/{model}_{modality}_bin{bin_sec}s_segment_labels.npy  (n_bins,) int — video index per bin
  {output_dir}/{model}_{modality}_bin{bin_sec}s_rdm_summary.json

Usage
-----
  python rsa/run_rdm_diagonal.py \\
      --embeddings-dir /path/to/model_embeddings \\
      --timing-csv     /path/to/movie_timing.csv  \\
      --output-dir     /path/to/outputs/rsa/rdm_diagonal \\
      --model          pe-av-small-16-frame \\
      --modality       av \\
      --bin-sec        5.0 \\
      --delay-sec      5.0 \\
      --tr             1.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.rsa_utils import process_model_embeddings
from rsa.shared.model_registry import (
    BIN_SEC_DEFAULT,
    DELAY_SEC_DEFAULT,
    TR_DEFAULT,
    DIAGONAL_MASK_DEFAULT,
    check_embeddings_exist,
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
        description="Compute within-movie and cross-movie masked RDMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--timing-csv",     required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--model",
                   default=DIAGONAL_MASK_DEFAULT["model"],
                   help="Model name key (see rsa/shared/model_registry.py).")
    p.add_argument("--modality",
                   default=DIAGONAL_MASK_DEFAULT["modality"],
                   choices=["a", "v", "av"],
                   help="Modality code.")
    p.add_argument("--bin-sec",   type=float, default=BIN_SEC_DEFAULT)
    p.add_argument("--delay-sec", type=float, default=DELAY_SEC_DEFAULT)
    p.add_argument("--tr",        type=float, default=TR_DEFAULT)
    p.add_argument("--rdm-metric",
                   default="cosine",
                   choices=["cosine", "correlation", "euclidean"],
                   help="Pairwise distance metric for RDM construction.")
    return p.parse_args()


# =============================================================================
# Segment-label construction (mirrors preprocess_fmri floor-bin logic)
# =============================================================================

def build_segment_labels(
    timing_df: pd.DataFrame,
    bin_sec: float,
    tr: float,
    run_trs: np.ndarray | None,
    delay_sec: float = 0.0,
) -> np.ndarray:
    """Return an integer video-index label for every temporal bin.

    Replicates the floor-binning from rsa_utils.preprocess_fmri so that the
    segment labels align exactly with the processed embedding and fMRI arrays.

    Parameters
    ----------
    timing_df : DataFrame — must have columns: video_id, duration_sec, onset_sec,
                            and optionally run_id / run.
    bin_sec   : float — temporal bin width in seconds
    tr        : float — repetition time (seconds)
    run_trs   : (n_runs,) int | None — run lengths; when None the timing CSV is
                treated as a single run (e.g. for embeddings which have no run
                boundary boundary truncation)
    delay_sec : float — haemodynamic delay applied to onset (default 0)

    Returns
    -------
    labels : (n_total_bins,) int — video index (0-based) per bin
    """
    run_col = (
        "run" if "run" in timing_df.columns
        else ("run_id" if "run_id" in timing_df.columns else None)
    )

    if run_col is None:
        groups = [(None, timing_df)]
        run_trs_list = [int(timing_df["duration_sec"].sum() / tr) + 1]
    else:
        groups = list(timing_df.groupby(run_col, sort=False))
        if run_trs is None:
            run_trs_list = [
                int(g["duration_sec"].sum() / tr) + 1 for _, g in groups
            ]
        else:
            run_trs_list = list(run_trs)

    # Build a unique integer index for each video_id
    all_video_ids = timing_df["video_id"].tolist() if "video_id" in timing_df.columns else list(range(len(timing_df)))
    unique_ids    = list(dict.fromkeys(all_video_ids))   # preserves order
    id_to_idx     = {vid: i for i, vid in enumerate(unique_ids)}

    labels = []
    for run_idx, (_, run_df) in enumerate(groups):
        run_tr_count = run_trs_list[run_idx]
        run_start_sec = sum(run_trs_list[:run_idx]) * tr

        for _, row in run_df.iterrows():
            n_bins = int(np.floor(row["duration_sec"] / bin_sec))
            if n_bins == 0:
                continue

            # Mirror boundary truncation from preprocess_fmri
            within_run_onset = row["onset_sec"] - run_start_sec
            start_tr = int(np.round((within_run_onset + delay_sec) / tr))
            end_tr   = start_tr + n_bins * max(1, int(np.round(bin_sec / tr)))

            if start_tr >= run_tr_count or start_tr < 0:
                continue
            if end_tr > run_tr_count:
                n_bins = (run_tr_count - start_tr) // max(1, int(np.round(bin_sec / tr)))
                if n_bins == 0:
                    continue

            vid_id  = row.get("video_id", row.name)
            vid_idx = id_to_idx.get(vid_id, 0)
            labels.extend([vid_idx] * n_bins)

    return np.array(labels, dtype=np.int32)


# =============================================================================
# RDM masking
# =============================================================================

def mask_rdm_offdiagonal(rdm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Set within-movie pairs to NaN; return cross-movie RDM."""
    same = (labels[:, None] == labels[None, :])            # (n, n) bool
    out  = rdm.copy().astype(np.float64)
    out[same] = np.nan
    return out


def mask_rdm_blockdiagonal(rdm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Set cross-movie pairs to NaN; return within-movie RDM."""
    diff = (labels[:, None] != labels[None, :])            # (n, n) bool
    out  = rdm.copy().astype(np.float64)
    out[diff] = np.nan
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ── Resolve embedding path and load ─────────────────────────────────────
    emb_file  = check_embeddings_exist(args.embeddings_dir, args.model,
                                        args.modality, args.bin_sec)
    timing_df = pd.read_csv(args.timing_csv)
    log.info(f"Embeddings: {emb_file}")
    log.info(f"Timing CSV: {args.timing_csv}  ({len(timing_df)} clips)")

    # For embeddings we do not apply run-level boundary truncation.
    # Build run_trs from actual per-run durations so that the within-run onset
    # conversion in process_model_embeddings is correct and no clips are dropped.
    # +1 padding matches build_segment_labels so both functions produce identical
    # per-run run_start_sec values and therefore identical bin counts.
    run_col_local = (
        "run" if "run" in timing_df.columns
        else ("run_id" if "run_id" in timing_df.columns else None)
    )
    if run_col_local is None:
        _emb_run_trs = np.array(
            [int(timing_df["duration_sec"].sum() / args.tr) + 1], dtype=np.int64
        )
    else:
        _groups_local = list(timing_df.groupby(run_col_local, sort=False))
        _emb_run_trs = np.array(
            [int(g["duration_sec"].sum() / args.tr) + 1 for _, g in _groups_local],
            dtype=np.int64,
        )
    emb = process_model_embeddings(
        str(emb_file), timing_df,
        bin_sec=args.bin_sec, tr=args.tr,
        run_trs=_emb_run_trs,
        delay_sec=0.0,    # embeddings are not delayed — they are stimulus-aligned
    )
    n_bins = emb.shape[0]
    log.info(f"Embedding shape: {emb.shape}  (n_bins={n_bins})")

    # ── Build segment labels ─────────────────────────────────────────────────
    labels = build_segment_labels(
        timing_df, args.bin_sec, args.tr,
        run_trs=None, delay_sec=0.0,
    )
    if len(labels) != n_bins:
        raise ValueError(
            f"Label count ({len(labels)}) ≠ embedding bin count ({n_bins}).\n"
            f"Check that bin_sec and timing CSV match the embedding file."
        )
    n_videos = int(labels.max()) + 1
    log.info(f"Segment labels: {len(labels)} bins across {n_videos} videos")

    # ── Compute full RDM ─────────────────────────────────────────────────────
    log.info(f"Computing full RDM (metric={args.rdm_metric}) ...")
    rdm_full = squareform(pdist(emb.astype(np.float64), metric=args.rdm_metric))
    log.info(f"  RDM shape: {rdm_full.shape}  "
             f"mean={np.nanmean(rdm_full):.4f}  max={np.nanmax(rdm_full):.4f}")

    # ── Masked variants ──────────────────────────────────────────────────────
    rdm_off   = mask_rdm_offdiagonal(rdm_full, labels)
    rdm_block = mask_rdm_blockdiagonal(rdm_full, labels)

    off_nonnan   = np.sum(~np.isnan(rdm_off))
    block_nonnan = np.sum(~np.isnan(rdm_block))
    total_cells  = n_bins * n_bins
    log.info(
        f"  Off-diagonal (cross-movie) non-NaN cells: "
        f"{off_nonnan:,} / {total_cells:,} ({100*off_nonnan/total_cells:.1f}%)"
    )
    log.info(
        f"  Block-diagonal (within-movie) non-NaN cells: "
        f"{block_nonnan:,} / {total_cells:,} ({100*block_nonnan/total_cells:.1f}%)"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_sec_int = int(args.bin_sec)
    stem = f"{args.model}_{args.modality}_bin{bin_sec_int}s"

    np.save(out_dir / f"{stem}_rdm_full.npy",        rdm_full.astype(np.float32))
    np.save(out_dir / f"{stem}_rdm_offdiag.npy",     rdm_off.astype(np.float32))
    np.save(out_dir / f"{stem}_rdm_blockdiag.npy",   rdm_block.astype(np.float32))
    np.save(out_dir / f"{stem}_segment_labels.npy",  labels)

    summary = {
        "model":          args.model,
        "modality":       args.modality,
        "bin_sec":        args.bin_sec,
        "rdm_metric":     args.rdm_metric,
        "n_bins":         int(n_bins),
        "n_videos":       int(n_videos),
        "off_nonnan_pct": float(100 * off_nonnan / total_cells),
        "block_nonnan_pct": float(100 * block_nonnan / total_cells),
        "rdm_full_mean":  float(np.nanmean(rdm_full)),
        "rdm_off_mean":   float(np.nanmean(rdm_off)),
        "rdm_block_mean": float(np.nanmean(rdm_block)),
    }
    summary_path = out_dir / f"{stem}_rdm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info(f"Outputs written to: {out_dir}")
    log.info(f"  {stem}_rdm_full.npy")
    log.info(f"  {stem}_rdm_offdiag.npy")
    log.info(f"  {stem}_rdm_blockdiag.npy")
    log.info(f"  {stem}_segment_labels.npy")
    log.info(f"  {stem}_rdm_summary.json")


if __name__ == "__main__":
    main()
