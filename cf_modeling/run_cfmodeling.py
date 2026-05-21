"""
cf_modeling/run_cfmodeling.py
==============================
Unified CF modeling pipeline (phases 02→04): prepare data, fit banded ridge,
apply null correction.

Covers data preparation, banded ridge fitting, and null correction in one
parallel-safe entry point.  The geometry script (extract_geometry.py),
integration maps (integration_maps.py), and summary (summary.py) remain
separate because they have different I/O patterns.

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Reads a pre-saved CIFTI (full-run mode: SG→PSC→GSR→zscore per run) and
  the companion run_trs.npy from disk.  The files are expected at:
    {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii
    {preprocessed_dir}/{subject}_{fmri_suffix}_run_trs.npy

STREAMING MODE (--raw-dir):
  Preprocesses the subject's raw 7T CIFTI on-the-fly (full-run mode:
  SG→PSC→GSR→zscore per run).  No preprocessed CIFTI is saved.
  Per-subject only — group_average requires pre-averaged data on disk.

Both modes are parallel-safe: each call processes one (subject, ROI pair)
tuple.  GNU parallel in run_analysis.sh spawns N such processes simultaneously.

Train / test split  [Hedger et al. 2025 / vicsompy convention]
  Last --n-test-trs TRs of each run → test (default: 103).
  Remaining TRs → train; run_onsets tracked for LORO-CV.

For group_average mode, CIFTI dscalar.nii maps are also written using
--template-cifti.  For per_subject, only .npy files are written (CIFTIs
are built by integration_maps.py).

Skip logic: if R2_{ROI_A}_nc.npy and R2_{ROI_B}_nc.npy already exist for the
subject / group, the run is skipped.

Usage (disk mode — group average):
  python run_cfmodeling.py \\
      --mode group_average \\
      --roi-a A1 --roi-b V1 \\
      --preprocessed-dir /path/to/preprocessed \\
      --fmri-suffix sg_psc_gsr_zscore \\
      --output-base /path/to/outputs/cf_modeling \\
      --template-cifti /path/to/group_average_..._cortex_59k.dtseries.nii

Usage (disk mode — per subject):
  python run_cfmodeling.py \\
      --mode per_subject \\
      --roi-a A5 --roi-b FFC \\
      --subject 100610 \\
      --preprocessed-dir /path/to/preprocessed \\
      --fmri-suffix sg_psc_gsr_zscore \\
      --output-base /path/to/outputs/cf_modeling

Usage (streaming mode — per subject):
  python run_cfmodeling.py \\
      --mode per_subject \\
      --roi-a A5 --roi-b FFC \\
      --subject 100610 \\
      --raw-dir /path/to/raw_ciftis \\
      --output-base /path/to/outputs/cf_modeling \\
      [--sg-filter] [--psc] [--no-gsr] [--no-z-score]
"""

import argparse
import logging
import os
import pickle
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
from himalaya.scoring import r2_score, r2_score_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.ridge_utils import project_onto_lboes, build_pipeline, fit_null_r2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_TEST_TRS_DEFAULT = 103


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified CF modeling pipeline (phases 02→04): "
                    "prepare data, fit banded ridge, apply null correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode", required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi-a", default="A1", dest="roi_a",
                   help="ROI A name (must match subsurface cache from script 01).")
    p.add_argument("--roi-b", default="V1", dest="roi_b",
                   help="ROI B name.")
    p.add_argument("--subject", default=None,
                   help="HCP subject ID (required for --mode per_subject).")
    p.add_argument("--output-base",
                   default="/home/amin/Research/Representation/Movie/outputs/cf_modeling",
                   dest="output_base",
                   help="Output base directory.")

    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk mode] Directory of pre-saved CIFTIs (full-run preprocessing). "
                          "Files: {dir}/{subject}_{fmri-suffix}_cortex_59k.dtseries.nii "
                          "and {dir}/{subject}_{fmri-suffix}_run_trs.npy")
    inp.add_argument("--fmri-suffix", default="sg_psc_gsr_zscore", dest="fmri_suffix",
                     help="[disk mode] Preprocessing suffix in the CIFTI filename.")
    inp.add_argument("--raw-dir", default=None, dest="raw_dir",
                     help="[streaming mode] Root directory of raw 7T CIFTI dtseries files. "
                          "Preprocesses on-the-fly; no CIFTI is saved.  "
                          "Per-subject only.")

    p.add_argument("--template-cifti", default=None, dest="template_cifti",
                   help="59k CIFTI template for dscalar output "
                        "(required for --mode group_average).")
    p.add_argument("--n-test-trs", type=int, default=N_TEST_TRS_DEFAULT, dest="n_test_trs",
                   help="TRs per run held out as test (last N TRs of each run).")
    p.add_argument("--tr", type=float, default=1.0,
                   help="TR in seconds.")

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true",
                      dest="sg_filter", help="Savitzky-Golay high-pass filter.")
    prep.add_argument("--psc", default=False, action="store_true",
                      help="Percent signal change normalization.")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction,
                      help="Global signal regression.")
    prep.add_argument("--z-score", default=True, action=argparse.BooleanOptionalAction,
                      dest="z_score", help="Z-score per vertex.")

    return p.parse_args()


# =============================================================================
# Shared helpers
# =============================================================================

def load_subsurface(cache_dir: str, roi_name: str):
    path = os.path.join(cache_dir, f"sub_{roi_name.lower()}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Subsurface cache not found: {path}\n"
            "Run extract_geometry.py first.")
    with open(path, "rb") as fh:
        sub = pickle.load(fh)
    if not hasattr(sub, "n_lboe"):
        sub.n_lboe = sub.L_eigenvectors.shape[1]
    if not hasattr(sub, "subsurface_verts"):
        sub.subsurface_verts = np.concatenate([
            sub.subsurface_verts_L, sub.subsurface_verts_R])
    return sub


def split_train_test(cortex_data: np.ndarray, run_trs: list, n_test_trs: int):
    """Split (n_cortex, T_total) into train/test by run boundary."""
    train_chunks, test_chunks = [], []
    run_onsets = []
    n_train    = 0
    offset     = 0

    for n_run in run_trs:
        if n_run <= n_test_trs:
            raise ValueError(
                f"Run has only {n_run} TRs but n_test_trs={n_test_trs}.")
        test_start = offset + n_run - n_test_trs
        train_chunks.append(cortex_data[:, offset:test_start])
        test_chunks.append(cortex_data[:, test_start:offset + n_run])
        run_onsets.append(n_train)
        n_train += n_run - n_test_trs
        offset  += n_run

    return (np.concatenate(train_chunks, axis=1),
            np.concatenate(test_chunks,  axis=1),
            run_onsets)


def _be_to_npy(var, backend):
    if isinstance(var, list):
        return [backend.to_numpy(v) for v in var]
    return backend.to_numpy(var)


def _save_npy(arr: np.ndarray, name: str, out_dir: str):
    np.save(os.path.join(out_dir, f"{name}.npy"), arr.astype(np.float32))
    log.info(f"  Saved {name}.npy")


def _save_cifti(arr: np.ndarray, name: str, template_cifti: str, cifti_dir: str):
    template = nib.load(template_cifti)
    img = nib.Cifti2Image(arr.astype(np.float32).reshape(1, -1),
                          header=template.header,
                          nifti_header=template.nifti_header)
    nib.save(img, os.path.join(cifti_dir, f"{name}.dscalar.nii"))
    log.info(f"  Saved {name}.dscalar.nii  "
             f"(mean={arr.mean():.4f}, frac>0={np.mean(arr > 0):.1%})")


# =============================================================================
# Core pipeline (shared between disk and streaming modes)
# =============================================================================

def _run_pipeline(args, cortex_data: np.ndarray, run_trs: list,
                  out_dir: str, cifti_dir: str | None,
                  subject_tag: str = ""):
    """Run 02→04 pipeline on pre-loaded (n_cortex, T_total) data."""
    prefix = f"[{subject_tag}] " if subject_tag else ""

    roi_root  = (f"{args.output_base}/{args.mode}/"
                 f"{args.roi_a}_{args.roi_b}")
    cache_dir = f"{roi_root}/subsurfaces"

    log.info(f"{prefix}Loading subsurfaces ...")
    sub_a = load_subsurface(cache_dir, args.roi_a)
    sub_b = load_subsurface(cache_dir, args.roi_b)
    log.info(f"{prefix}  [{args.roi_a}] L={sub_a.L_eigenvectors.shape}  "
             f"R={sub_a.R_eigenvectors.shape}")
    log.info(f"{prefix}  [{args.roi_b}] L={sub_b.L_eigenvectors.shape}  "
             f"R={sub_b.R_eigenvectors.shape}")

    log.info(f"{prefix}Splitting train / test (n_test_trs={args.n_test_trs}) ...")
    train_data, test_data, run_onsets = split_train_test(
        cortex_data, run_trs, args.n_test_trs)
    del cortex_data
    log.info(f"{prefix}  train: {train_data.shape}  test: {test_data.shape}  "
             f"run_onsets: {run_onsets}")

    log.info(f"{prefix}Projecting onto LBOEs ...")
    X_train, band_sizes = project_onto_lboes(train_data, [sub_a, sub_b])
    X_test,  _          = project_onto_lboes(test_data,  [sub_a, sub_b])
    log.info(f"{prefix}  X_train={X_train.shape}  X_test={X_test.shape}")
    log.info(f"{prefix}  band_sizes: {args.roi_a}={band_sizes[0]}  "
             f"{args.roi_b}={band_sizes[1]}")

    Y_train     = train_data.T.astype(np.float32)
    Y_test      = test_data.T.astype(np.float32)
    mean_a_test = test_data[sub_a.subsurface_verts, :].mean(axis=0).astype(np.float32)
    mean_b_test = test_data[sub_b.subsurface_verts, :].mean(axis=0).astype(np.float32)
    del train_data, test_data

    log.info(f"{prefix}Fitting banded ridge ...")
    pipeline, backend = build_pipeline(
        n_samples_train=X_train.shape[0],
        run_onsets=np.array(run_onsets),
        band_sizes=band_sizes,
        roi_names=[args.roi_a.lower(), args.roi_b.lower()],
    )
    pipeline.fit(X_train, Y_train)
    log.info(f"{prefix}  Fit complete.")

    log.info(f"{prefix}Computing variance partitioning on test set ...")
    Y_hat_full   = pipeline.predict(X_test)
    R2_full      = _be_to_npy(r2_score(Y_test, Y_hat_full), backend)
    Y_hat_split  = pipeline.predict(X_test, split=True)
    split_scores = _be_to_npy(r2_score_split(Y_test, Y_hat_split), backend)
    R2_a   = split_scores[0]
    R2_b   = split_scores[1]
    Shared = (R2_a + R2_b - R2_full).astype(np.float32)
    del X_train, X_test, Y_hat_full, Y_hat_split, pipeline

    log.info(f"{prefix}  R2_full={np.nanmean(R2_full):.4f}  "
             f"R2_{args.roi_a}={np.nanmean(R2_a):.4f}  "
             f"R2_{args.roi_b}={np.nanmean(R2_b):.4f}")

    log.info(f"{prefix}Null correction ...")
    R2_null_a = fit_null_r2(mean_a_test, Y_test)
    R2_null_b = fit_null_r2(mean_b_test, Y_test)
    del Y_test, mean_a_test, mean_b_test

    R2_a_nc = (R2_a - R2_null_a).astype(np.float32)
    R2_b_nc = (R2_b - R2_null_b).astype(np.float32)

    log.info(f"{prefix}  R2_{args.roi_a}_nc: mean={R2_a_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_a_nc > 0):.1%}")
    log.info(f"{prefix}  R2_{args.roi_b}_nc: mean={R2_b_nc.mean():.4f}  "
             f"frac>0={np.mean(R2_b_nc > 0):.1%}")

    os.makedirs(out_dir, exist_ok=True)
    if cifti_dir:
        os.makedirs(cifti_dir, exist_ok=True)

    log.info(f"{prefix}Saving to {out_dir} ...")
    for name, arr in [
        ("R2_full",              R2_full),
        (f"R2_{args.roi_a}",     R2_a),
        (f"R2_{args.roi_b}",     R2_b),
        ("Shared_R2",            Shared),
        (f"R2_null_{args.roi_a}", R2_null_a),
        (f"R2_null_{args.roi_b}", R2_null_b),
        (f"R2_{args.roi_a}_nc",   R2_a_nc),
        (f"R2_{args.roi_b}_nc",   R2_b_nc),
    ]:
        _save_npy(arr, name, out_dir)
        if cifti_dir:
            _save_cifti(arr, name, args.template_cifti, cifti_dir)

    # Save band_sizes and run_onsets alongside data (needed by downstream scripts)
    np.save(os.path.join(out_dir, "band_sizes.npy"),
            np.array(band_sizes, dtype=np.int32))
    np.save(os.path.join(out_dir, "run_onsets.npy"),
            np.array(run_onsets, dtype=np.int32))


# =============================================================================
# Disk mode
# =============================================================================

def _cifti_path(args, subject: str) -> str:
    return str(Path(args.preprocessed_dir) /
               f"{subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")


def _run_trs_path(args, subject: str) -> str:
    return str(Path(args.preprocessed_dir) /
               f"{subject}_{args.fmri_suffix}_run_trs.npy")


def _run_disk(args):
    subject = ("group_average"
                if args.mode == "group_average"
                else args.subject)
    roi_root = f"{args.output_base}/{args.mode}/{args.roi_a}_{args.roi_b}"
    out_dir  = (f"{roi_root}/prep"
                if args.mode == "group_average"
                else f"{roi_root}/subjects/{subject}")
    cifti_dir = (f"{roi_root}/cifti_maps"
                 if args.mode == "group_average" else None)

    # Skip if null-corrected maps already exist
    done_a = os.path.exists(os.path.join(out_dir, f"R2_{args.roi_a}_nc.npy"))
    done_b = os.path.exists(os.path.join(out_dir, f"R2_{args.roi_b}_nc.npy"))
    if done_a and done_b:
        log.info(f"[{subject}] R2_nc maps exist — skipping")
        return

    cifti = _cifti_path(args, subject)
    trs   = _run_trs_path(args, subject)
    log.info("=" * 60)
    log.info(f"CF modeling ({args.mode}) — {args.roi_a} × {args.roi_b}")
    if args.mode == "per_subject":
        log.info(f"  Subject: {subject}")
    log.info(f"  CIFTI: {os.path.basename(cifti)}")
    log.info("=" * 60)

    img       = nib.load(cifti)
    full_data = img.get_fdata(dtype=np.float32)   # (T, n_cortex)
    cortex_data = full_data.T                      # (n_cortex, T)
    del full_data
    log.info(f"  Loaded: {cortex_data.shape}")

    run_trs = np.load(trs).tolist()
    log.info(f"  run_trs: {run_trs}  (total: {sum(run_trs)} TRs)")

    _run_pipeline(args, cortex_data, run_trs, out_dir, cifti_dir,
                  subject_tag=subject if args.mode == "per_subject" else "")
    log.info(f"[{subject}] Done.")


# =============================================================================
# Streaming mode (per_subject only)
# =============================================================================

def _run_streaming(args):
    from preprocess_individual import preprocess_subject_fullrun

    sub      = args.subject
    roi_root = f"{args.output_base}/per_subject/{args.roi_a}_{args.roi_b}"
    out_dir  = f"{roi_root}/subjects/{sub}"

    done_a = os.path.exists(os.path.join(out_dir, f"R2_{args.roi_a}_nc.npy"))
    done_b = os.path.exists(os.path.join(out_dir, f"R2_{args.roi_b}_nc.npy"))
    if done_a and done_b:
        log.info(f"[{sub}] R2_nc maps exist — skipping")
        return

    log.info("=" * 60)
    log.info(f"CF modeling (per_subject, streaming) — {args.roi_a} × {args.roi_b}")
    log.info(f"  Subject: {sub}")
    log.info("=" * 60)

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter, psc=args.psc,
        gsr=args.gsr, z_score=args.z_score,
    )
    log.info(f"[{sub}] Preprocessing raw CIFTI ...")
    cortex_data, _bm_axis, run_trs_arr = preprocess_subject_fullrun(
        sub, Path(args.raw_dir), tr=args.tr, args=prep_args
    )
    run_trs = run_trs_arr.tolist()
    log.info(f"[{sub}] Preprocessed: {cortex_data.shape}  run_trs: {run_trs}")

    _run_pipeline(args, cortex_data, run_trs, out_dir, cifti_dir=None,
                  subject_tag=sub)
    del cortex_data
    log.info(f"[{sub}] Done.")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.mode == "per_subject" and args.subject is None:
        log.error("--subject is required for --mode per_subject.")
        sys.exit(1)
    if args.mode == "group_average" and args.raw_dir:
        log.error("Streaming mode (--raw-dir) is not supported for --mode group_average. "
                  "Use --preprocessed-dir with a pre-averaged CIFTI.")
        sys.exit(1)
    if args.preprocessed_dir and args.raw_dir:
        log.error("--preprocessed-dir and --raw-dir are mutually exclusive.")
        sys.exit(1)
    if not args.preprocessed_dir and not args.raw_dir:
        log.error("Either --preprocessed-dir (disk mode) or --raw-dir (streaming mode) "
                  "is required.")
        sys.exit(1)
    if args.mode == "group_average" and args.template_cifti is None:
        log.error("--template-cifti is required for --mode group_average.")
        sys.exit(1)

    if args.raw_dir:
        _run_streaming(args)
    else:
        _run_disk(args)


if __name__ == "__main__":
    main()
