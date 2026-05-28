"""
cf_modeling/02_fit_cf_model.py
================================
Fit a multi-source spectral CF model using vicsompy's MssCf pipeline.

This script is the main modeling entry point.  It replaces run_cfmodeling.py
and is designed to import directly from the vicsompy source repository rather
than vendored copies, ensuring maximum alignment with Hedger et al. (2025).

Pipeline (matching vicsompy's analyse_subject exactly, without splicing)
------------------------------------------------------------------------
1. Load preprocessed CIFTI (group-average or per-subject).
2. Split into train/test per run with z-scoring (Hedger convention: last
   --n-test-trs TRs of each run = test; z-score each portion independently).
3. Convert grayordinate data to sphere-space (118584 vertices, medial wall = 0).
4. Load pre-built Subsurface objects from the cache (built by 01_extract_geometry.py).
5. Build temp YAML config (via lib/config_builder.py) and mock subject adapter.
6. Instantiate CfModel (lib/cf_model.py) — subclass of vicsompy's MssCf.
7. nm.make_dm_grayord()      — LBOE design matrix from sphere-space train data.
8. nm.prep_pipeline()        — himalaya banded ridge with LORO-CV.
9. nm.fit_grayord()          — fit on grayordinate targets (T_train × 59412).
10. nm.get_params()           — betas, train split R², best alphas.
11. nm.test_xval_grayord()   — test R² (full + per-band split).
12. nm.compute_null_r2()     — OLS null model from ROI mean timecourses.
13. nm.save_outcomes()        — write npy files via MssCf.saveout().
14. nm.save_all_maps()        — R²_nc maps as npy + CIFTI.

No splicing
-----------
splice_lookups(), load_lookups(), save_spliced_lookups() are NOT called.
These require lookup-table CSVs that do not exist for custom ROI pairs.

GPU support
-----------
Himalaya automatically uses the torch_cuda backend when CUDA is available
(configured via --backend in the YAML config builder).  Set PYTORCH_CUDA_ALLOC_CONF
in run_analysis.sh for memory management.

Input modes
-----------
DISK MODE (--preprocessed-dir + --fmri-suffix):
  Reads a pre-saved concatenated CIFTI (SG→PSC→GSR per run, from
  preprocess_individual.py) and companion run_trs.npy from disk.
  Pattern: {preprocessed_dir}/{subject}_{fmri_suffix}_cortex_59k.dtseries.nii
           {preprocessed_dir}/{subject}_{fmri_suffix}_run_trs.npy

STREAMING MODE (--raw-dir):
  Preprocesses the subject's raw 7T CIFTI on-the-fly (SG→PSC→GSR per run).
  Per-subject only — group_average requires pre-processed data on disk.

Skip logic
----------
If R2_{roi_a}_nc.npy and R2_{roi_b}_nc.npy both exist in the output directory,
the run is skipped (safe for GNU parallel reruns).

Usage
-----
  # Group average (disk mode)
  python cf_modeling/02_fit_cf_model.py \\
      --mode group_average --roi-a 3b --roi-b V1 \\
      --preprocessed-dir /path/to/preprocessed/average_sub/sg_psc \\
      --fmri-suffix sg_psc \\
      --output-base /path/to/outputs/cf_modeling \\
      --template-cifti /path/to/.../group_average_sg_psc_cortex_59k.dtseries.nii

  # Per-subject (disk mode)
  python cf_modeling/02_fit_cf_model.py \\
      --mode per_subject --roi-a A5 --roi-b FFC --subject 100610 \\
      --preprocessed-dir /path/to/preprocessed/sg_psc \\
      --fmri-suffix sg_psc \\
      --output-base /path/to/outputs/cf_modeling

  # Streaming (per_subject only)
  python cf_modeling/02_fit_cf_model.py \\
      --mode per_subject --roi-a A5 --roi-b FFC --subject 100610 \\
      --raw-dir /path/to/raw_ciftis \\
      --output-base /path/to/outputs/cf_modeling \\
      [--sg-filter] [--psc] [--no-gsr]
"""

import argparse
import logging
import os
import pickle
import sys
import types
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import zscore

# ---------------------------------------------------------------------------
# Vicsompy import — must be before lib imports that re-export vicsompy classes
# ---------------------------------------------------------------------------
VICSOMPY_REPO = os.environ.get(
    "VICSOMPY_REPO",
    "/home/amin/Research/Representation/Movie/Vicarious_somatotopy",
)
if VICSOMPY_REPO not in sys.path:
    sys.path.insert(0, VICSOMPY_REPO)

# ---------------------------------------------------------------------------
# Local lib imports
# ---------------------------------------------------------------------------
_CF_DIR = Path(__file__).resolve().parent
if str(_CF_DIR) not in sys.path:
    sys.path.insert(0, str(_CF_DIR))

from lib.config_builder  import build_temp_yaml
from lib.subject_adapter import CfSubjectAdapter
from lib.cf_model        import CfModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_TEST_TRS_DEFAULT = 103    # last 103 TRs per run = test (Hedger et al. 2025)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit a multi-source spectral CF model using vicsompy's MssCf "
            "(Hedger et al. 2025, no splicing, modular ROI definition)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi-a", default="3b",  dest="roi_a",
                   help="ROI A Glasser short name (must match subsurface cache from 01).")
    p.add_argument("--roi-b", default="V1",  dest="roi_b",
                   help="ROI B Glasser short name.")
    p.add_argument("--subject", default=None,
                   help="HCP subject ID (required for --mode per_subject).")
    p.add_argument("--output-base",
                   default="/home/amin/Research/Representation/Movie/outputs/cf_modeling",
                   dest="output_base")
    p.add_argument("--vicsompy-repo", default=VICSOMPY_REPO, dest="vicsompy_repo",
                   help="Path to local vicsompy source repo.")

    inp = p.add_argument_group("input (choose one)")
    inp.add_argument("--preprocessed-dir", default=None, dest="preprocessed_dir",
                     help="[disk] Directory of pre-saved CIFTI + run_trs.npy.")
    inp.add_argument("--fmri-suffix", default="sg_psc", dest="fmri_suffix",
                     help="[disk] Preprocessing suffix in the CIFTI filename.")
    inp.add_argument("--raw-dir", default=None, dest="raw_dir",
                     help="[streaming] Directory of raw 7T CIFTI files. Per-subject only.")

    p.add_argument("--template-cifti", default=None, dest="template_cifti",
                   help="59k CIFTI for dscalar map headers (required for group_average).")
    p.add_argument("--n-test-trs", type=int, default=N_TEST_TRS_DEFAULT, dest="n_test_trs",
                   help="TRs per run held out as test (last N TRs).")
    p.add_argument("--tr", type=float, default=1.0, help="TR in seconds.")
    p.add_argument("--backend", default="torch_cuda", dest="backend",
                   choices=["torch_cuda", "torch", "numpy", "cupy"],
                   help="Himalaya compute backend.")
    p.add_argument("--n-iter", type=int, default=20, dest="n_iter",
                   help="Himalaya random_search solver iterations.")
    p.add_argument("--n-targets-batch", type=int, default=20000, dest="n_targets_batch",
                   help="Targets per GPU batch.")

    prep = p.add_argument_group("streaming preprocessing (ignored in disk mode)")
    prep.add_argument("--sg-filter", default=False, action="store_true", dest="sg_filter")
    prep.add_argument("--psc",       default=False, action="store_true")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction)

    return p.parse_args()


# =============================================================================
# Train / test split with per-run z-scoring (matches vicsompy exactly)
# =============================================================================

def split_train_test(cortex_data: np.ndarray, run_trs: list, n_test_trs: int):
    """Split (n_grayord, T_total) into per-run train/test with z-scoring.

    Replicates vicsompy HcpSubject.split_test_sequence():
      - Last n_test_trs TRs of each run → test.
      - Z-score each portion independently along the time axis.
      - Concatenate across runs.

    Parameters
    ----------
    cortex_data : (n_grayord, T_total) float32
    run_trs     : list of int — number of TRs per run.
    n_test_trs  : int — TRs per run held out as test.

    Returns
    -------
    train_data  : (n_grayord, T_train) float32 — z-scored train data.
    test_data   : (n_grayord, T_test)  float32 — z-scored test data.
    run_onsets  : list of int — onset of each training run in train_data.
    """
    train_chunks, test_chunks = [], []
    run_onsets = []
    n_train    = 0
    offset     = 0

    for n_run in run_trs:
        if n_run <= n_test_trs:
            raise ValueError(
                f"Run has {n_run} TRs but n_test_trs={n_test_trs}: no training samples."
            )
        test_start = offset + n_run - n_test_trs

        train_raw = cortex_data[:, offset:test_start]          # (n_grayord, T_train_run)
        test_raw  = cortex_data[:, test_start:offset + n_run]  # (n_grayord, T_test_run)

        # Z-score each run's train and test independently along the time axis.
        # axis=1 → time, matching vicsompy stats.zscore(v[:, slice], axis=1).
        train_z = zscore(train_raw, axis=1, nan_policy="omit")
        test_z  = zscore(test_raw,  axis=1, nan_policy="omit")

        train_chunks.append(train_z)
        test_chunks.append(test_z)
        run_onsets.append(n_train)
        n_train += n_run - n_test_trs
        offset  += n_run

    train_data = np.nan_to_num(np.concatenate(train_chunks, axis=1)).astype(np.float32)
    test_data  = np.nan_to_num(np.concatenate(test_chunks,  axis=1)).astype(np.float32)

    return train_data, test_data, run_onsets


# =============================================================================
# Subsurface loader
# =============================================================================

def load_subsurface(cache_dir: str, roi_name: str):
    """Load a Subsurface pkl from the cache built by 01_extract_geometry.py."""
    path = os.path.join(cache_dir, f"sub_{roi_name.lower()}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Subsurface cache not found: {path}\n"
            "Run 01_extract_geometry.py first."
        )
    with open(path, "rb") as fh:
        sub = pickle.load(fh)
    # Patch attributes added after potential older saves
    if not hasattr(sub, "n_lboe"):
        sub.n_lboe = sub.L_eigenvectors.shape[1]
    if not hasattr(sub, "subsurface_verts"):
        sub.subsurface_verts = np.concatenate([
            sub.subsurface_verts_L, sub.subsurface_verts_R])
    return sub


# =============================================================================
# Core pipeline
# =============================================================================

def _run_pipeline(
    args,
    cortex_data: np.ndarray,
    run_trs: list,
    bm_axis,
    out_dir: str,
    cifti_dir: str | None,
    subject_tag: str = "",
) -> None:
    """Run the full CF modeling pipeline on pre-loaded grayordinate data.

    Parameters
    ----------
    cortex_data : (n_grayord, T_total) float32 — CIFTI grayordinate data
                  (59412 for HCP 59k_fs_LR, medial wall excluded, runs concatenated).
    run_trs     : list of int — TRs per run (4 entries for HCP movie).
    bm_axis     : nibabel BrainModelAxis — from the CIFTI header.
    out_dir     : str — directory for npy files.
    cifti_dir   : str | None — directory for CIFTI maps (group_average only).
    subject_tag : str — label for log messages.
    """
    prefix = f"[{subject_tag}] " if subject_tag else ""

    roi_root  = os.path.join(args.output_base, args.mode, f"{args.roi_a}_{args.roi_b}")
    cache_dir = os.path.join(roi_root, "subsurfaces")

    # ── Skip if already done ──────────────────────────────────────────────
    done_a = os.path.exists(os.path.join(out_dir, f"R2_{args.roi_a}_nc.npy"))
    done_b = os.path.exists(os.path.join(out_dir, f"R2_{args.roi_b}_nc.npy"))
    if done_a and done_b:
        log.info("%sR2_nc maps already exist — skipping.", prefix)
        return

    # ── Load subsurfaces ──────────────────────────────────────────────────
    log.info("%sLoading subsurfaces …", prefix)
    sub_a = load_subsurface(cache_dir, args.roi_a)
    sub_b = load_subsurface(cache_dir, args.roi_b)
    log.info("%s  [%s] L_eig=%s  R_eig=%s  n_lboe=%d",
             prefix, args.roi_a,
             sub_a.L_eigenvectors.shape, sub_a.R_eigenvectors.shape, sub_a.n_lboe)
    log.info("%s  [%s] L_eig=%s  R_eig=%s  n_lboe=%d",
             prefix, args.roi_b,
             sub_b.L_eigenvectors.shape, sub_b.R_eigenvectors.shape, sub_b.n_lboe)

    # ── Train / test split with per-run z-scoring ─────────────────────────
    log.info("%sSplitting train/test (n_test_trs=%d) …", prefix, args.n_test_trs)
    train_data, test_data, run_onsets = split_train_test(
        cortex_data, run_trs, args.n_test_trs)
    del cortex_data
    log.info("%s  train=%s  test=%s  run_onsets=%s",
             prefix, train_data.shape, test_data.shape, run_onsets)

    # ── Build temp YAML config and mock subject ───────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    masks_dir      = os.path.join(roi_root, "masks")
    surfaces_dir   = cache_dir                        # subsurfaces already here

    # Determine backend: fall back to torch if CUDA not available
    backend = args.backend
    if backend == "torch_cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("%sCUDA not available; falling back to 'torch' backend.", prefix)
                backend = "torch"
        except ImportError:
            log.warning("%storch not available; falling back to 'numpy' backend.", prefix)
            backend = "numpy"

    temp_yaml = build_temp_yaml(
        roi_a=args.roi_a,
        roi_b=args.roi_b,
        n_lboe_a=sub_a.n_lboe,
        n_lboe_b=sub_b.n_lboe,
        masks_dir=masks_dir,
        surfaces_dir=surfaces_dir,
        out_dir=out_dir,
        backend_engine=backend,
        n_iter=args.n_iter,
        n_targets_batch=args.n_targets_batch,
        n_targets_batch_refit=args.n_targets_batch,
        tmp_dir=out_dir,
    )
    log.info("%sTemp YAML written: %s", prefix, temp_yaml)

    subject_id  = "group_average" if args.mode == "group_average" else subject_tag
    mock_subject = CfSubjectAdapter(subject_id, out_dir)

    # ── Initialise CfModel ────────────────────────────────────────────────
    log.info("%sInitialising CfModel (vicsompy.MssCf subclass) …", prefix)
    nm = CfModel(mock_subject, subject_id, yaml=temp_yaml)
    nm.inject_subsurfaces(
        subsurfaces=[sub_a, sub_b],
        roi_names=[args.roi_a, args.roi_b],
        n_lboes=[sub_a.n_lboe, sub_b.n_lboe],
    )

    # ── Design matrix from sphere-space train data ────────────────────────
    log.info("%sBuilding design matrix …", prefix)
    nm.make_dm_grayord(train_data, bm_axis)

    # ── Prepare banded ridge pipeline ─────────────────────────────────────
    log.info("%sPreparing pipeline (backend=%s, n_iter=%d) …",
             prefix, backend, args.n_iter)
    nm.prep_pipeline(run_durations=np.array(run_onsets, dtype=int))

    # ── Fit on grayordinate targets ───────────────────────────────────────
    log.info("%sFitting model …", prefix)
    nm.fit_grayord(train_data)
    del train_data

    # ── Get betas and training-set scores ─────────────────────────────────
    log.info("%sExtracting betas and train scores …", prefix)
    nm.get_params()

    # ── Test cross-validation ─────────────────────────────────────────────
    log.info("%sTest cross-validation …", prefix)
    nm.test_xval_grayord(test_data, bm_axis)

    # ── Null model ────────────────────────────────────────────────────────
    log.info("%sComputing null model R² …", prefix)
    null_r2 = nm.compute_null_r2(test_data, bm_axis)
    del test_data

    # ── Save npy outputs via MssCf.saveout() ─────────────────────────────
    log.info("%sSaving outcomes (betas, scores, alphas) …", prefix)
    nm.save_outcomes()

    # ── Save R² maps (npy + CIFTI) ────────────────────────────────────────
    log.info("%sSaving R² maps …", prefix)
    nm.save_all_maps(
        null_r2=null_r2,
        roi_a=args.roi_a,
        roi_b=args.roi_b,
        bm_axis=bm_axis,
        out_dir=out_dir,
        template_cifti=args.template_cifti,
        cifti_dir=cifti_dir,
    )

    # ── Clean up temp YAML ────────────────────────────────────────────────
    try:
        os.remove(temp_yaml)
    except OSError:
        pass

    log.info("%sDone.", prefix)


# =============================================================================
# Disk mode
# =============================================================================

def _run_disk(args) -> None:
    subject = "group_average" if args.mode == "group_average" else args.subject
    roi_tag  = f"{args.roi_a}_{args.roi_b}"
    roi_root = os.path.join(args.output_base, args.mode, roi_tag)
    out_dir  = (os.path.join(roi_root, "prep")
                if args.mode == "group_average"
                else os.path.join(roi_root, "subjects", subject))
    cifti_dir = (os.path.join(roi_root, "cifti_maps")
                 if args.mode == "group_average" else None)

    cifti_path = os.path.join(
        args.preprocessed_dir,
        f"{subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii",
    )
    trs_path = os.path.join(
        args.preprocessed_dir,
        f"{subject}_{args.fmri_suffix}_run_trs.npy",
    )

    if not os.path.exists(cifti_path):
        log.error("CIFTI not found: %s", cifti_path)
        sys.exit(1)

    log.info("=" * 60)
    log.info("02 — CF modeling (%s)  %s × %s", args.mode, args.roi_a, args.roi_b)
    log.info("  CIFTI   : %s", os.path.basename(cifti_path))
    log.info("  Output  : %s", out_dir)
    log.info("=" * 60)

    img         = nib.load(cifti_path)
    bm_axis     = img.header.get_axis(1)
    cortex_data = img.get_fdata(dtype=np.float32).T   # (n_grayord, T)
    del img
    log.info("  Loaded: %s", cortex_data.shape)

    run_trs = np.load(trs_path).tolist() if os.path.exists(trs_path) else None
    if run_trs is None:
        log.warning("run_trs.npy not found; inferring 4 equal runs.")
        T = cortex_data.shape[1]
        run_trs = [T // 4] * 4
    log.info("  run_trs: %s  (total: %d TRs)", run_trs, sum(run_trs))

    _run_pipeline(
        args, cortex_data, run_trs, bm_axis,
        out_dir, cifti_dir,
        subject_tag=subject if args.mode == "per_subject" else "",
    )


# =============================================================================
# Streaming mode (per_subject only)
# =============================================================================

def _run_streaming(args) -> None:
    """Preprocess raw CIFTIs on-the-fly and run the CF pipeline.

    Follows preprocess_individual.preprocess_subject():
      Per run: load → extract cortex → SG (optional) → PSC (optional) → GSR (optional)
      Then concatenate and pass to _run_pipeline().
    """
    # Import preprocess_subject from the repo root
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from preprocess_individual import preprocess_subject

    sub      = args.subject
    roi_tag  = f"{args.roi_a}_{args.roi_b}"
    roi_root = os.path.join(args.output_base, "per_subject", roi_tag)
    out_dir  = os.path.join(roi_root, "subjects", sub)

    log.info("=" * 60)
    log.info("02 — CF modeling (per_subject, streaming)  %s × %s",
             args.roi_a, args.roi_b)
    log.info("  Subject : %s", sub)
    log.info("  Flags   : SG=%s  PSC=%s  GSR=%s",
             args.sg_filter, args.psc, args.gsr)
    log.info("=" * 60)

    prep_args = types.SimpleNamespace(
        sg_filter=args.sg_filter,
        psc=args.psc,
        gsr=args.gsr,
    )
    cortex_data, bm_axis, run_trs_arr = preprocess_subject(
        sub, Path(args.raw_dir), tr=args.tr, args=prep_args)
    run_trs = run_trs_arr.tolist()
    log.info("  Preprocessed: %s  run_trs: %s", cortex_data.shape, run_trs)

    _run_pipeline(
        args, cortex_data, run_trs, bm_axis,
        out_dir, cifti_dir=None,
        subject_tag=sub,
    )
    del cortex_data


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Ensure vicsompy repo is on sys.path
    if args.vicsompy_repo not in sys.path:
        sys.path.insert(0, args.vicsompy_repo)

    # Validation
    if args.mode == "per_subject" and args.subject is None:
        log.error("--subject is required for --mode per_subject.")
        sys.exit(1)
    if args.mode == "group_average" and args.raw_dir:
        log.error("Streaming mode is not supported for group_average. "
                  "Use --preprocessed-dir with a pre-averaged CIFTI.")
        sys.exit(1)
    if args.preprocessed_dir and args.raw_dir:
        log.error("--preprocessed-dir and --raw-dir are mutually exclusive.")
        sys.exit(1)
    if not args.preprocessed_dir and not args.raw_dir:
        log.error("Either --preprocessed-dir (disk mode) or --raw-dir (streaming mode) required.")
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
