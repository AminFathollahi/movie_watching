"""
encoding/run_encoding.py
=========================
Ridge encoding model for movie fMRI data (59k grayordinate space).

Fits a ridge regression model (himalaya RidgeCV with SVD solver) that predicts
cortical fMRI responses from model embeddings. Alpha is selected independently
per vertex via leave-one-run-out cross-validation on the training set.

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Reads a continuous, cleaned CIFTI produced by preprocess_individual.py.
  The file is expected at:
    {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii

STREAMING MODE (--raw-dir):
  Preprocesses the subject's raw 7T CIFTI on-the-fly (filtered mode: movie TPs
  with hemodynamic delay, GSR, z-score). No preprocessed CIFTI is saved.

Both modes are parallel-safe: each call processes one (subject, model, modality)
tuple. GNU parallel in run_analysis.sh spawns N such processes simultaneously.

Train / test split:
  Test: video5, video9, video14, video18 (last video of each run, 82 TRs each)
  Train: the remaining 14 videos

Outputs one CIFTI dscalar.nii per modality containing Pearson r on the test set.

Usage (disk mode):
  python run_encoding.py \\
      --preprocessed-dir <path> --fmri-suffix raw \\
      --timing-csv <path> --embeddings-dir <path> --template-cifti <path> \\
      --output-dir <path> --subject group_average \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 \\
      --alpha-min -2 --alpha-max 9 --n-alphas 23 \\
      --test-video-ids video5,video9,video14,video18

Usage (streaming mode):
  python run_encoding.py \\
      --raw-dir <path> --subject 100610 \\
      --timing-csv <path> --embeddings-dir <path> --template-cifti <path> \\
      --output-dir <path> \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 \\
      [--sg-filter] [--psc] [--no-gsr] [--no-z-score] \\
      --alpha-min -2 --alpha-max 9 --n-alphas 23 \\
      --test-video-ids video5,video9,video14,video18
"""

import argparse
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from encoding.shared.encoding_utils import (
    _bin_and_split_fmri, build_fmri_arrays,
    build_embedding_arrays, run_encoding_model, save_cifti,
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
        description="Ridge encoding model for movie fMRI (59k grayordinate space).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk mode] Directory of pre-filtered, pre-z-scored CIFTIs "
                          "(output of preprocess_individual.py --timing-csv). "
                          "File: {dir}/{subject}_{fmri-suffix}_cortex_59k.dtseries.nii")
    inp.add_argument("--fmri-suffix", default="raw", dest="fmri_suffix",
                     help="[disk mode] Filename suffix that encodes preprocessing "
                          "(e.g. 'raw', 'sg_psc_gsr'). Must match preprocess_individual.py output.")
    inp.add_argument("--raw-dir", default=None, dest="raw_dir",
                     help="[streaming mode] Root directory of raw 7T CIFTI dtseries files. "
                          "The subject is preprocessed on-the-fly; no CIFTI is saved.")

    p.add_argument("--timing-csv", required=True, help="Path to movie_timing.csv.")
    p.add_argument("--embeddings-dir", required=True,
                   help="Base directory for model embeddings.")
    p.add_argument("--template-cifti", required=True,
                   help="Template 59k dscalar.nii for CIFTI output header.")
    p.add_argument("--output-dir", required=True, help="Root output directory.")
    p.add_argument("--subject", default="group_average",
                   help="Subject ID (used for output path and, in disk mode, to "
                        "construct the CIFTI filename).")

    p.add_argument("--model", required=True,
                   help="Model name (subdirectory in --embeddings-dir).")
    p.add_argument("--modality", required=True,
                   help="Modality: v | a | av | all.")
    p.add_argument("--bin-sec", type=float, required=True,
                   help="Temporal bin size in seconds.")
    p.add_argument("--delay-sec", type=float, default=5.0,
                   help="Hemodynamic delay in seconds (applied in both disk and streaming "
                        "modes when slicing stimulus blocks from continuous fMRI).")
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM HRF (use with delay-sec 0).")
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec",
                   help="Window stride in seconds (default: bin-sec, i.e. no overlap).")
    p.add_argument("--normalize", action="store_true",
                   help="Per-run z-score normalization of embeddings.")
    p.add_argument("--tr", type=float, required=True, help="TR in seconds.")

    p.add_argument("--alpha-min", type=float, required=True,
                   help="Log10 of minimum ridge alpha.")
    p.add_argument("--alpha-max", type=float, required=True,
                   help="Log10 of maximum ridge alpha.")
    p.add_argument("--n-alphas", type=int, required=True,
                   help="Number of alpha values on the log scale.")
    p.add_argument("--backend", default="torch_cuda", dest="backend",
                   help="himalaya backend for ridge fitting: torch_cuda | torch | numpy. "
                        "torch_cuda uses GPU if available, falls back to torch automatically.")
    p.add_argument("--test-video-ids", required=True,
                   help="Comma-separated test video IDs.")

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true")
    prep.add_argument("--psc",       default=False, action="store_true")
    prep.add_argument("--gsr",    default=False, action=argparse.BooleanOptionalAction)
    prep.add_argument("--z-score", default=True, action=argparse.BooleanOptionalAction,
                      dest="z_score")

    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def _config_label(args) -> str:
    parts = [
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        "norm" if args.normalize else "nonorm",
        f"bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s",
    ]
    return "_".join(parts)


def _cifti_path(args) -> str:
    """Construct the per-subject preprocessed CIFTI path from directory + suffix."""
    return str(Path(args.preprocessed_dir) /
               f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")


def _run_modalities(args, timing_df, test_ids, alphas, config,
                    modalities, out_root, Y_train, Y_test, run_onsets,
                    subject_tag=""):
    """Fit encoding model for each modality; save one CIFTI per modality."""
    prefix = f"[{subject_tag}] " if subject_tag else ""
    for mod in modalities:
        out_path = out_root / f"encoding_r_{mod}.dscalar.nii"
        if out_path.exists():
            log.info(f"{prefix}Skipping {mod} — output exists")
            continue

        emb_file = (Path(args.embeddings_dir) / args.model /
                    f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s" /
                    f"{args.model}_{mod}.npy")
        log.info(f"{prefix}[{mod}] Embeddings: {emb_file}")

        X_train, X_test = build_embedding_arrays(
            str(emb_file), timing_df, test_ids,
            bin_sec=args.bin_sec, hrf=args.hrf, normalize=args.normalize,
            skip_sec=args.skip_sec,
        )
        log.info(f"{prefix}[{mod}] X_train={X_train.shape}  X_test={X_test.shape}")

        r_vals = run_encoding_model(
            X_train, Y_train, X_test, Y_test,
            run_onsets=run_onsets, alphas=alphas,
            backend=args.backend,
        )
        save_cifti(r_vals, args.template_cifti, str(out_path),
                   map_name=f"encoding_r_{mod}")
        log.info(f"{prefix}[{mod}] Saved: {out_path.name}  "
                 f"(mean r={r_vals.mean():.4f}, max r={r_vals.max():.4f})")


# =============================================================================
# Disk mode
# =============================================================================

def _run_disk(args):
    timing_df  = pd.read_csv(args.timing_csv)
    test_ids   = [v.strip() for v in args.test_video_ids.split(",")]
    alphas     = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    config     = _config_label(args)
    modalities = ["v", "a", "av"] if args.modality == "all" else [args.modality]

    out_root  = Path(args.output_dir) / args.subject / args.model / config
    out_paths = {mod: out_root / f"encoding_r_{mod}.dscalar.nii" for mod in modalities}
    if all(p.exists() for p in out_paths.values()):
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    cifti = _cifti_path(args)
    log.info(f"Encoding model: {args.model} / {args.modality} / {config}")
    log.info(f"  fMRI: {cifti}")

    run_trs_path = str(Path(args.preprocessed_dir) /
                       f"{args.subject}_{args.fmri_suffix}_run_trs.npy")
    Y_train, Y_test, run_onsets = build_fmri_arrays(
        cifti, run_trs_path, timing_df, test_ids,
        args.bin_sec, args.tr, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    log.info(f"  Y_train={Y_train.shape}  Y_test={Y_test.shape}  "
             f"run_onsets={run_onsets}")

    out_root.mkdir(parents=True, exist_ok=True)
    _run_modalities(args, timing_df, test_ids, alphas, config,
                    modalities, out_root, Y_train, Y_test, run_onsets)
    log.info("Done.")


# =============================================================================
# Streaming mode
# =============================================================================

def _run_streaming(args):
    # preprocess_subject is the correct name in preprocess_individual.py.
    # It returns (n_cortex, T_total), bm_axis, run_trs — the full continuous
    # signal (SG→PSC→GSR). _bin_and_split_fmri then uses onset_sec + run_trs
    # to locate each clip and apply the haemodynamic delay.
    from preprocess_individual import preprocess_subject

    timing_df  = pd.read_csv(args.timing_csv)
    test_ids   = [v.strip() for v in args.test_video_ids.split(",")]
    alphas     = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    config     = _config_label(args)
    modalities = ["v", "a", "av"] if args.modality == "all" else [args.modality]
    sub        = args.subject
    raw_dir    = Path(args.raw_dir)

    out_root  = Path(args.output_dir) / sub / args.model / config
    out_paths = {mod: out_root / f"encoding_r_{mod}.dscalar.nii" for mod in modalities}
    if all(p.exists() for p in out_paths.values()):
        log.info(f"[{sub}] All outputs exist — skipping")
        return

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr,
    )
    log.info(f"[{sub}] Preprocessing raw CIFTI "
             f"(SG={args.sg_filter} PSC={args.psc} GSR={args.gsr}) ...")
    data, _bm_axis, run_trs = preprocess_subject(
        sub, raw_dir, tr=args.tr, args=prep_args
    )

    Y_train, Y_test, run_onsets = _bin_and_split_fmri(
        data, timing_df, test_ids, args.bin_sec, args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    del data
    log.info(f"[{sub}] Y_train={Y_train.shape}  Y_test={Y_test.shape}  "
             f"run_onsets={run_onsets}")

    out_root.mkdir(parents=True, exist_ok=True)
    _run_modalities(args, timing_df, test_ids, alphas, config,
                    modalities, out_root, Y_train, Y_test, run_onsets,
                    subject_tag=sub)
    del Y_train, Y_test
    log.info(f"[{sub}] Done.")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
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
