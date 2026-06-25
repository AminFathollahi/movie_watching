"""
encoding/encoding.py
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
tuple. GNU parallel in analysis.sh spawns N such processes simultaneously.

Train / test split:
  Test: video5, video9, video14, video18 (last video of each run, 82 TRs each)
  Train: the remaining 14 videos

Outputs one CIFTI dscalar.nii per modality containing Pearson r on the test set.

Usage (disk mode):
  python encoding.py \\
      --preprocessed-dir <path> --fmri-suffix raw \\
      --timing-csv <path> --embeddings-dir <path> --template-cifti <path> \\
      --output-dir <path> --subject group_average \\
      --model pe-av-small-16-frame --modality av \\
      --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 \\
      --alpha-min -2 --alpha-max 9 --n-alphas 23 \\
      --test-video-ids video5,video9,video14,video18

Usage (streaming mode):
  python encoding.py \\
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
import subprocess
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# Remove the script's own directory from sys.path so that encoding.py doesn't
# shadow the encoding/ package when resolving "encoding.shared".
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
sys.path.insert(0, str(ROOT))
from encoding.shared.encoding_utils import (
    _bin_and_split_fmri, build_fmri_arrays,
    build_embedding_arrays, run_encoding_model, save_cifti,
)
from cifti_io import (
    get_bm_axis, get_cortex_vertex_indices,
    get_combined_map_names, merge_into_combined,
    save_cifti_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# Group-average significance helpers
# =============================================================================

def _r_sigmap(r: np.ndarray, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """One-tailed p-values and sign × −log10(p) for Pearson r with df = n_test − 2."""
    df   = max(n_test - 2, 1)
    r    = np.asarray(r, dtype=np.float64)
    t    = r * np.sqrt(df) / np.sqrt(np.maximum(1.0 - r**2, 1e-10))
    p_two = stats.t.sf(np.abs(t), df=df) * 2.0
    p_uncorr = np.where(t > 0, p_two / 2.0, 1.0 - p_two / 2.0).astype(np.float32)
    eps = np.finfo(np.float32).tiny
    sigmap = (np.sign(r) * (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)
    return p_uncorr, sigmap


def _fdr_sigmap(p_uncorr: np.ndarray, r: np.ndarray,
                alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, int]:
    """BH-FDR on p_uncorr; returns (sigmap_fdr, fdr_mask_f32, n_sig)."""
    p_fdr = stats.false_discovery_control(p_uncorr.astype(np.float64), method="bh")
    eps   = np.finfo(np.float32).tiny
    sigmap_fdr = (np.sign(r) * (-np.log10(np.maximum(p_fdr, eps)))).astype(np.float32)
    fdr_mask   = (p_fdr < alpha).astype(np.float32)
    return sigmap_fdr, fdr_mask, int(fdr_mask.sum())


def _write_border_file(
    cluster_mask: np.ndarray,
    cifti_vertex_indices: np.ndarray,
    n_full_verts: int,
    surface_path: str,
    out_border_path: str,
    workbench: str,
    class_name: str = "fdr",
) -> None:
    full_mask = np.zeros(n_full_verts, dtype=np.float32)
    full_mask[cifti_vertex_indices] = cluster_mask
    tmp_metric = Path(out_border_path).with_suffix(".tmp.func.gii")
    arr = GiftiDataArray(data=full_mask, intent=0, datatype="NIFTI_TYPE_FLOAT32")
    nib.save(GiftiImage(darrays=[arr]), str(tmp_metric))
    try:
        subprocess.run(
            [workbench, "-metric-rois-to-border",
             surface_path, str(tmp_metric), class_name, out_border_path],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning(f"wb_command border failed for {out_border_path}:\n  {e.stderr.strip()}")
    finally:
        tmp_metric.unlink(missing_ok=True)


def _save_group_avg_significance(r_vals, n_test, template_cifti,
                                  out_root, mod, args):
    """Compute and save p-values + FDR maps alongside the group-average r CIFTI."""
    p_uncorr, sigmap_uncorr = _r_sigmap(r_vals, n_test)
    sigmap_fdr, fdr_mask, n_sig = _fdr_sigmap(p_uncorr, r_vals)
    log.info(f"  [{mod}] n_test={n_test}  FDR significant (p<0.05): {n_sig:,} / {r_vals.shape[0]:,}")

    save_cifti_map(sigmap_uncorr, template_cifti,
                   str(out_root / f"encoding_r_{mod}_sigmap_uncorr.dscalar.nii"),
                   f"encoding_r_{mod}_sigmap_uncorr")
    save_cifti_map(sigmap_fdr,    template_cifti,
                   str(out_root / f"encoding_r_{mod}_sigmap_fdr.dscalar.nii"),
                   f"encoding_r_{mod}_sigmap_fdr")

    fdr_mask_path = out_root / f"encoding_r_{mod}_fdr_mask.dscalar.nii"
    save_cifti_map(fdr_mask, template_cifti, str(fdr_mask_path), f"encoding_r_{mod}_fdr_mask")
    log.info(f"  [{mod}] Saved sigmap_uncorr, sigmap_fdr, fdr_mask")

    workbench  = getattr(args, "workbench", None)
    left_surf  = getattr(args, "left_surface", None)
    right_surf = getattr(args, "right_surface", None)
    if workbench and left_surf and right_surf:
        n_grays   = r_vals.shape[0]
        coverage  = fdr_mask.sum() / n_grays if n_grays > 0 else 0.0
        if fdr_mask.sum() == 0:
            log.info(f"  [{mod}] No FDR-significant vertices; skipping border files.")
        elif coverage > 0.90:
            log.warning(f"  [{mod}] FDR mask covers {coverage*100:.1f}% — skipping border files.")
        else:
            bm = get_bm_axis(template_cifti)
            lh_idx, rh_idx = get_cortex_vertex_indices(bm)
            n_left     = len(lh_idx)
            n_verts_lh = nib.load(left_surf).darrays[0].data.shape[0]
            n_verts_rh = nib.load(right_surf).darrays[0].data.shape[0]
            _write_border_file(fdr_mask[:n_left], lh_idx, n_verts_lh, left_surf,
                               str(out_root / f"encoding_r_{mod}_fdr_lh.border"), workbench)
            _write_border_file(fdr_mask[n_left:], rh_idx, n_verts_rh, right_surf,
                               str(out_root / f"encoding_r_{mod}_fdr_rh.border"), workbench)
            log.info(f"  [{mod}] FDR border files saved.")


# =============================================================================
# Group-average significance fast-path (for existing r-CIFTIs)
# =============================================================================

def _compute_n_test(timing_df: pd.DataFrame, test_ids: list,
                     bin_sec: float, skip_sec: float) -> int:
    """Count test bins from timing CSV using the same formula as _bin_and_split_fmri.
    Uses per-video duration only — ignores run-boundary cutoff (negligible for large n)."""
    skip = skip_sec if skip_sec is not None else bin_sec
    n = 0
    for vid in test_ids:
        dur = float(timing_df.loc[timing_df["video_id"] == vid, "duration_sec"].iloc[0])
        n += max(0, int(np.floor((dur - bin_sec) / skip)) + 1) if dur >= bin_sec else 0
    return n


def _run_sigmap_fastpath(out_paths: dict, out_root: Path,
                          args, timing_df: pd.DataFrame, test_ids: list) -> None:
    """Compute missing significance maps for existing group-average r CIFTIs.
    Called when all r-CIFTIs exist but sigmaps may be absent."""
    n_test_path = out_root / "n_test.npy"
    if n_test_path.exists():
        n_test = int(np.load(str(n_test_path))[0])
    else:
        n_test = _compute_n_test(timing_df, test_ids, args.bin_sec, args.skip_sec)
        np.save(str(n_test_path), np.array([n_test]))
        log.info(f"  Saved n_test={n_test} → {n_test_path.name}")

    for mod, r_path in out_paths.items():
        sigmap_paths = [
            out_root / f"encoding_r_{mod}_sigmap_uncorr.dscalar.nii",
            out_root / f"encoding_r_{mod}_sigmap_fdr.dscalar.nii",
            out_root / f"encoding_r_{mod}_fdr_mask.dscalar.nii",
        ]
        if all(p.exists() for p in sigmap_paths):
            log.info(f"  [{mod}] Significance maps already up to date — skipping")
            continue
        r_vals = nib.load(str(r_path)).get_fdata(dtype=np.float32).squeeze()
        log.info(f"  [{mod}] Fast-path: computing significance maps (n_test={n_test})")
        _save_group_avg_significance(r_vals, n_test, args.template_cifti, out_root, mod, args)


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

    sig = p.add_argument_group("group-average significance maps (optional)")
    sig.add_argument("--left-surface",  default=None, dest="left_surface",
                     help="Left 59k midthickness .surf.gii — used to create FDR border files.")
    sig.add_argument("--right-surface", default=None, dest="right_surface",
                     help="Right 59k midthickness .surf.gii — used to create FDR border files.")
    sig.add_argument("--workbench",     default=None,
                     help="Path to wb_command. Required for border file creation.")

    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def _config_label(args) -> str:
    parts = [
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        "norm" if args.normalize else "demean",
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

        if args.subject == "group_average":
            n_test_path = out_root / "n_test.npy"
            if not n_test_path.exists():
                np.save(str(n_test_path), np.array([Y_test.shape[0]]))
            _save_group_avg_significance(
                r_vals, Y_test.shape[0], args.template_cifti, out_root, mod, args
            )


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
        if args.subject == "group_average":
            _run_sigmap_fastpath(out_paths, out_root, args, timing_df, test_ids)
        else:
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
        if args.subject == "group_average":
            _run_sigmap_fastpath(out_paths, out_root, args, timing_df, test_ids)
        else:
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
