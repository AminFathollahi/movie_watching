"""
encoding/run_encoding.py
=========================
Ridge encoding model for movie fMRI data.

Fits a ridge regression model (himalaya RidgeCV with SVD solver) that predicts
cortical fMRI responses from model embeddings. Alpha is selected independently
per vertex via leave-one-run-out cross-validation on the training set.

Train / test split matches the RSA notebooks:
  Test: video5, video9, video14, video18 (last video of each run, 82 TRs each)
  Train: the remaining 14 videos

Outputs one CIFTI dscalar.nii per modality containing Pearson r on the test set.

Usage:
  python run_encoding.py \\
      --fmri-cifti <path> --timing-csv <path> \\
      --embeddings-dir <path> --template-cifti <path> \\
      --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 2.0 --delay-sec 5.0 \\
      [--hrf] [--normalize] \\
      --alpha-min -2 --alpha-max 9 --n-alphas 23 \\
      --chunk-size 2000 --tr 1.0 \\
      --test-video-ids video5,video9,video14,video18 \\
      [--subject group_average]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoding.shared.encoding_utils import (
    build_fmri_arrays, build_embedding_arrays,
    run_encoding_model, save_cifti,
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
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Ridge encoding model for movie fMRI.",
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

    # Output
    p.add_argument("--output-dir", required=True,
                   help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID or 'group_average' (for output subdirectory naming).")

    # Analysis config
    p.add_argument("--model", required=True,
                   help="Model name (must match a subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True,
                   help="Modality: v | a | av | all (all runs each modality separately).")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Hemodynamic delay in seconds (applied to X; ignored if --hrf).")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF instead of boxcar delay.")
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score normalization of embeddings.")
    p.add_argument("--tr", type=float, required=True,
                   help="TR in seconds.")

    # Ridge parameters
    p.add_argument("--alpha-min", type=float, required=True,
                   help="Log10 of minimum ridge alpha.")
    p.add_argument("--alpha-max", type=float, required=True,
                   help="Log10 of maximum ridge alpha.")
    p.add_argument("--n-alphas", type=int, required=True,
                   help="Number of alpha values on the log scale.")
    p.add_argument("--chunk-size", type=int, required=True,
                   help="Number of vertices per batch for CV (memory/speed tradeoff).")

    # Train/test split
    p.add_argument("--test-video-ids", required=True,
                   help="Comma-separated list of test video IDs.")

    return p.parse_args()


# =============================================================================
# Output naming
# =============================================================================

def _config_label(args) -> str:
    """Build a descriptive label for the current analysis configuration."""
    parts = [
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        "norm" if args.normalize else "nonorm",
        f"bin{args.bin_sec:.0f}s",
    ]
    return "_".join(parts)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    timing_df = pd.read_csv(args.timing_csv)
    test_ids = [v.strip() for v in args.test_video_ids.split(",")]
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    # When hrf=True, embeddings are HRF-convolved; fMRI window is not shifted.
    # When hrf=False, fMRI window is shifted forward by delay_sec.
    fmri_delay = 0.0 if args.hrf else args.delay_sec
    config = _config_label(args)

    modalities = ["v", "a", "av"] if args.modality == "all" else [args.modality]
    out_root = Path(args.output_dir) / args.subject / args.model / config
    out_root.mkdir(parents=True, exist_ok=True)

    # Skip if all requested outputs already exist
    out_paths = {mod: out_root / f"encoding_r_{mod}.dscalar.nii" for mod in modalities}
    if all(p.exists() for p in out_paths.values()):
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    log.info(f"Encoding model: {args.model} / {args.modality} / {config}")
    log.info(f"  fMRI: {args.fmri_cifti}")

    # Build fMRI arrays (same for all modalities)
    log.info("  Building fMRI arrays ...")
    Y_train, Y_test, run_onsets = build_fmri_arrays(
        args.fmri_cifti, timing_df, test_ids, args.bin_sec, fmri_delay, args.tr
    )
    log.info(f"  Y_train: {Y_train.shape}  Y_test: {Y_test.shape}  "
             f"run_onsets: {run_onsets}")

    for mod in modalities:
        out_path = out_paths[mod]
        if out_path.exists():
            log.info(f"  Skipping {mod} — output exists")
            continue

        emb_file = (Path(args.embeddings_dir) / args.model /
                    f"{int(args.bin_sec)}s" / f"{args.model}_{mod}.npy")
        log.info(f"  [{mod}] Embeddings: {emb_file}")

        X_train, X_test = build_embedding_arrays(
            str(emb_file), timing_df, test_ids,
            bin_sec=args.bin_sec, hrf=args.hrf,
            normalize=args.normalize,
        )
        log.info(f"  [{mod}] X_train: {X_train.shape}  X_test: {X_test.shape}")

        r_vals = run_encoding_model(
            X_train, Y_train, X_test, Y_test,
            run_onsets=run_onsets, alphas=alphas,
            chunk_size=args.chunk_size,
        )

        save_cifti(r_vals, args.template_cifti, str(out_path),
                   map_name=f"encoding_r_{mod}")
        log.info(f"  [{mod}] Saved: {out_path}  "
                 f"(mean r={r_vals.mean():.4f}, max r={r_vals.max():.4f})")

    log.info("Done.")


if __name__ == "__main__":
    main()
