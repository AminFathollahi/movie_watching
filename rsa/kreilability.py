#!/usr/bin/env python3
"""
Evaluate split-half reliability of searchlight RSA maps across different
neighbourhood sizes (k).

For each random split the script:
  1. Averages preprocessed fMRI across subjects in each half.
  2. Saves the half-average as a CIFTI dtseries.nii (outdir/{split}/{half}_avg.dtseries.nii).
  3. Builds a half-specific average midthickness surface from the subjects' individual
     surface files (wb_command -surface-average).
  4. Runs searchlight RSA on the half-average map using that surface.
  5. Evaluates split-half reproducibility (global Pearson r + top-10 % Dice) per k.

Two subjects (126931, 745555) have no midthickness surfaces and are excluded
automatically → 175 subjects used for splitting.

Usage (all defaults set; only required args shown):
  conda activate analysis
  python rsa/kreilability.py \\
    --subjects-list /home/amin/Research/Representation/Movie/data/subjects.txt \\
    --timing-csv    /home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered/movie_timing.csv \\
    --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \\
    --template-cifti /home/amin/Research/Representation/Movie/data/average_sub/sg_psc/group_average_sg_psc_cortex_59k.dtseries.nii \\
    --model pe-av-small-16-frame --modality av \\
    --k 100 150 200 --n-splits 50

Override defaults:
  --preprocessed-dir  /home/amin/Research/Representation/Movie/data/preprocessed
  --outdir            /media/amin/ADATA HD710 PRO/Research/Representation/Movie/outputs/k_splithalf
  --indiv-surf-template ".../midthickness_1.6/{sub}.{hem}.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
  --bin-sec 2.0
  --workbench /opt/workbench/bin_linux64/wb_command
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

# Import existing pipeline components (run from repo root)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.searchlight import get_neighbors, run_searchlight
from rsa.shared.rsa_utils import (
    align_and_assert_bins, preprocess_fmri,
    process_model_embeddings,
)
from cifti_io import get_bm_axis, get_cortex_vertex_indices

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# NOTE: subject exclusions are handled upstream — data/subjects.txt already
# contains only the 175 valid subjects (126931 and 745555 removed; see
# data/excluded.txt).  Pass --subjects-list pointing to that file; no further
# filtering is needed here.  The variable below is kept only as a last-resort
# safeguard in case a stale subjects file is passed accidentally.
_EXCLUDED_SUBJECTS_SAFEGUARD: set[str] = {"126931", "745555"}

# Default paths
_DEFAULT_PREPROCESSED_DIR = (
    "/home/amin/Research/Representation/Movie/data/preprocessed"
)
_DEFAULT_OUTDIR = "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/outputs/k_splithalf"
_DEFAULT_SURF_TEMPLATE = (
    "/home/amin/Research/Representation/Movie/data/midthickness_1.6"
    "/{sub}.{hem}.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
)
_DEFAULT_WORKBENCH = "/opt/workbench/bin_linux64/wb_command"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Split-half reliability for RSA maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    p.add_argument(
        "--preprocessed-dir",
        default=_DEFAULT_PREPROCESSED_DIR,
        help="Directory containing {sub}_{fmri-suffix}_cortex_59k.dtseries.nii "
             "and matching _run_trs.npy files.",
    )
    p.add_argument(
        "--fmri-suffix", default="raw",
        help="Filename infix that encodes the preprocessing pipeline.",
    )
    p.add_argument("--raw-dir", default=None,
                   help="[streaming mode] Root of raw 7T CIFTI files. "
                        "Mutually exclusive with --preprocessed-dir.")

    # --- Subjects ---
    p.add_argument("--subjects-list", required=True,
                   help="Plain-text file, one subject ID per line. "
                        "Commented lines (# …) and blank lines are skipped.")

    # --- Model ---
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--modality", required=True, choices=["v", "a", "av"])

    # --- CIFTI template ---
    p.add_argument("--template-cifti", required=True,
                   help="Reference CIFTI (dtseries or dscalar) for BrainModelAxis.")

    # --- Surfaces ---
    p.add_argument(
        "--indiv-surf-template",
        default=_DEFAULT_SURF_TEMPLATE,
        help="Path template for individual midthickness surfaces. "
             "Use {sub} and {hem} (L/R) placeholders.",
    )
    p.add_argument("--workbench", default=_DEFAULT_WORKBENCH)

    # --- Output ---
    p.add_argument("--outdir", default=_DEFAULT_OUTDIR)

    # --- Reliability parameters ---
    p.add_argument("--k", type=int, nargs="+", default=[100, 150, 200],
                   help="Neighbourhood sizes to compare.")
    p.add_argument("--n-splits", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # --- RSA parameters ---
    p.add_argument("--bin-sec", type=float, default=2.0,
                   help="Temporal bin width in seconds (2 s → 1 TR at 0.5 Hz).")
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--hrf", action="store_true",
                   help="Convolve embeddings with SPM canonical HRF.")
    p.add_argument("--method", default="spearman",
                   choices=["spearman", "pearson"])
    p.add_argument("--n-jobs", type=int, default=-1)

    return p.parse_args()


# =============================================================================
# Surface averaging
# =============================================================================

def generate_half_surface(
    subjects: list[str], hem_short: str, args, out_path: Path
) -> None:
    """Call wb_command to compute the average midthickness surface for a split half.

    Skips if *out_path* already exists.
    Raises FileNotFoundError if an individual surface is missing.
    """
    if out_path.exists():
        return

    log.info(
        f"    Averaging {hem_short} surface for {len(subjects)} subjects → "
        f"{out_path.name}"
    )
    cmd = [args.workbench, "-surface-average", str(out_path)]
    for sub in subjects:
        surf_file = Path(args.indiv_surf_template.format(sub=sub, hem=hem_short))
        if not surf_file.exists():
            raise FileNotFoundError(
                f"Missing individual surface: {surf_file}"
            )
        cmd.extend(["-surf", str(surf_file)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"wb_command -surface-average failed:\n{result.stderr}"
        )


# =============================================================================
# fMRI loading & averaging
# =============================================================================

def load_half_average(
    subjects: list[str], args
) -> tuple[np.ndarray, np.ndarray, nib.Cifti2Image]:
    """Load preprocessed CIFTIs for *subjects* and return their running mean.

    Returns
    -------
    running_mean : (n_vertices, T) float32
    shared_run_trs : (n_runs,) int — from the first successfully loaded subject
    template_img   : nibabel image whose header is used to save the average CIFTI
    """
    running_mean: np.ndarray | None = None
    shared_run_trs: np.ndarray | None = None
    template_img: nib.Cifti2Image | None = None
    n_loaded = 0

    prep_dir = Path(args.preprocessed_dir)
    suffix = args.fmri_suffix

    for sub in tqdm(subjects, desc="  Averaging half", leave=False):
        cifti_path = prep_dir / f"{sub}_{suffix}_cortex_59k.dtseries.nii"
        trs_path = prep_dir / f"{sub}_{suffix}_run_trs.npy"

        try:
            img = nib.load(str(cifti_path))
            data = img.get_fdata(dtype=np.float32).T  # (n_vertices, T)
            run_trs = np.load(str(trs_path))
        except Exception as exc:
            log.warning(f"    [{sub}] skipped — {exc}")
            continue

        if running_mean is None:
            running_mean = data.astype(np.float64)
            shared_run_trs = run_trs
            template_img = img
            n_loaded = 1
        else:
            # Welford-style online mean to avoid a 3+ GB temporary array
            n_loaded += 1
            delta = data - running_mean
            delta /= n_loaded
            running_mean += delta
            del delta

        del data
        gc.collect()

    if running_mean is None:
        raise RuntimeError("No subjects successfully loaded for this half.")

    log.info(f"    Averaged {n_loaded} subjects  shape={running_mean.shape}")
    return running_mean.astype(np.float32), shared_run_trs, template_img


def save_half_average_cifti(
    running_mean: np.ndarray,
    template_img: nib.Cifti2Image,
    out_path: Path,
) -> None:
    """Save (n_vertices, T) running mean as a CIFTI dtseries.nii.

    Uses the SeriesAxis + BrainModelAxis from *template_img* (first-subject
    header), so the TR and grayordinate layout are preserved exactly.
    """
    if out_path.exists():
        log.info(f"    Average CIFTI exists — skipping: {out_path.name}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_disk = running_mean.T.astype(np.float32)  # (T, n_vertices) on disk
    img_out = nib.Cifti2Image(data_disk, header=template_img.header)
    nib.save(img_out, str(out_path))
    log.info(f"    Saved average CIFTI: {out_path.name}")


# =============================================================================
# Searchlight across multiple k values
# =============================================================================

def compute_rsa_maps_for_ks(
    fmri_binned: np.ndarray,
    model_binned: np.ndarray,
    subjects: list[str],
    split_id: str,
    half_label: str,
    args,
) -> dict[int, np.ndarray]:
    """Compute searchlight RSA maps for all requested k values.

    For each hemisphere:
      1. Build the half-specific average midthickness surface (wb_command).
      2. Extract k-NN neighbours at max_k (two-level cache in cache_dir).
      3. Slice to each requested k — no recomputation needed.
      4. Call run_searchlight once per k.

    Parameters
    ----------
    fmri_binned  : (n_bins, n_vertices) float32
    model_binned : (n_bins, n_features) float32 — embeddings (RDM built inside)
    subjects     : subjects in this half (used to name the surface cache)
    split_id     : e.g. "split_000"
    half_label   : "half1" or "half2"
    args         : parsed arguments namespace

    Returns
    -------
    maps_by_k : dict {k → (n_vertices,) float32}
    """
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    cache_dir = Path(args.outdir) / "_geodesic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    maps_by_k = {k: np.zeros(fmri_binned.shape[1], dtype=np.float32) for k in args.k}
    max_k = max(args.k)

    for hem, surf_indices in [("left", left_indices), ("right", right_indices)]:
        hem_short = "L" if hem == "left" else "R"
        surf_indices = surf_indices.astype(np.int32)

        # Map surface vertex index → fmri column (-1 for medial-wall vertices)
        vertex_to_col = np.full(surf_indices.max() + 1, -1, dtype=np.int32)
        vertex_to_col[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)

        fmri_hem = fmri_binned[:, :n_left] if hem == "left" else fmri_binned[:, n_left:]
        offset = 0 if hem == "left" else n_left

        # 1. Build the half-specific average midthickness surface
        half_surf_path = cache_dir / f"{split_id}_{half_label}_{hem_short}_avg.surf.gii"
        generate_half_surface(subjects, hem_short, args, half_surf_path)

        # 2. Extract k-NN at max_k (three-level cache: exact .npy → k_max derivation → dconn)
        sub_id = f"{split_id}_{half_label}"
        neighbors_max = get_neighbors(
            str(half_surf_path), args.workbench, sub_id, hem, max_k, cache_dir
        )

        # 3. Clean up the large transient surface file
        if half_surf_path.exists():
            half_surf_path.unlink()

        # 4. Run searchlight for each k by slicing the max-k neighbour array
        for k in sorted(args.k, reverse=True):
            neighbors_k = neighbors_max[:, :k]

            corr_hem = run_searchlight(
                fmri_hem,
                model_binned,
                neighbors_k,
                surface_indices=surf_indices,
                vertex_to_col=vertex_to_col,
                method=args.method,
                n_jobs=args.n_jobs,
            )
            maps_by_k[k][offset : offset + len(corr_hem)] = corr_hem

            del corr_hem
            gc.collect()

        del neighbors_max
        gc.collect()

    return maps_by_k


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Validate mutual exclusion
    if args.preprocessed_dir and args.raw_dir:
        log.error("--preprocessed-dir and --raw-dir are mutually exclusive.")
        sys.exit(1)
    if not args.preprocessed_dir and not args.raw_dir:
        log.error("Provide either --preprocessed-dir or --raw-dir.")
        sys.exit(1)
    if args.raw_dir:
        log.error("Streaming mode (--raw-dir) is not yet implemented in kreliability.py. "
                  "Use --preprocessed-dir instead.")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load + filter subjects ───────────────────────────────────────────────
    raw_subjects = [
        ln.strip()
        for ln in Path(args.subjects_list).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # subjects.txt is the authoritative list (n=175, excludes no-midthickness subs).
    # The safeguard below catches accidental use of an older/larger subjects file.
    subjects = [s for s in raw_subjects if s not in _EXCLUDED_SUBJECTS_SAFEGUARD]
    n_excluded = len(raw_subjects) - len(subjects)
    if n_excluded:
        log.warning(
            f"Safeguard: removed {n_excluded} subject(s) lacking midthickness "
            f"({_EXCLUDED_SUBJECTS_SAFEGUARD & set(raw_subjects)}) from {args.subjects_list}. "
            f"Update subjects.txt to avoid this warning."
        )
    log.info(
        f"Subjects: {len(raw_subjects)} in {args.subjects_list}, "
        f"{n_excluded} safeguard-excluded, "
        f"{len(subjects)} active."
    )

    # ── Timing + embeddings setup ────────────────────────────────────────────
    timing_df = pd.read_csv(args.timing_csv)
    bin_sec_int = int(args.bin_sec)
    skip_int = int(args.bin_sec)  # skip_sec defaults to bin_sec in this script
    emb_file = (
        Path(args.embeddings_dir) / args.model
        / f"bin{bin_sec_int}s_skip{skip_int}s" / f"{args.model}_{args.modality}.npy"
    )
    log.info(f"Embeddings: {emb_file}")

    # ── Resume support ───────────────────────────────────────────────────────
    metadata_file = outdir / "split_half_reliability_results.json"
    results: dict = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            results = json.load(f)

    rng = np.random.default_rng(args.seed)

    # model_binned is the same for every split; initialise lazily after the
    # first subject is loaded (so we know run_trs)
    model_binned: np.ndarray | None = None

    # ── Main split-half loop ─────────────────────────────────────────────────
    for split_idx in range(args.n_splits):
        split_id = f"split_{split_idx:03d}"

        if split_id in results and all(
            str(k) in results[split_id] for k in args.k
        ):
            log.info(f"Skipping {split_id} (already complete).")
            # Advance the RNG the same number of steps even when skipping,
            # so subsequent splits always get the same shuffle.
            rng.shuffle(subjects)
            continue

        log.info(f"── {split_id} ──────────────────────────────────────────")
        subjects_copy = list(subjects)
        rng.shuffle(subjects_copy)
        half_n = len(subjects_copy) // 2
        half1_subs = subjects_copy[:half_n]
        half2_subs = subjects_copy[half_n : 2 * half_n]
        log.info(f"  Half 1: {len(half1_subs)} subs  |  Half 2: {len(half2_subs)} subs")

        split_dir = outdir / split_id
        split_dir.mkdir(parents=True, exist_ok=True)

        # ── Half 1 ──────────────────────────────────────────────────────────
        log.info("  [Half 1] Loading & averaging fMRI ...")
        fmri_h1, run_trs_h1, tmpl_img_h1 = load_half_average(half1_subs, args)

        # Lazy model initialisation (needs run_trs for boundary handling)
        if model_binned is None:
            log.info("  Processing model embeddings (once) ...")
            model_binned = process_model_embeddings(
                str(emb_file), timing_df,
                bin_sec=args.bin_sec, hrf=args.hrf,
                tr=args.tr, run_trs=run_trs_h1,
                delay_sec=args.delay_sec,
            )
            log.info(f"  Model binned: {model_binned.shape}")

        # Save the half-average fMRI as a CIFTI dtseries
        avg_h1_path = split_dir / f"half1_avg.dtseries.nii"
        save_half_average_cifti(fmri_h1, tmpl_img_h1, avg_h1_path)

        log.info("  [Half 1] Binning & searchlight ...")
        fmri_binned_h1 = preprocess_fmri(
            fmri_h1, timing_df, run_trs_h1, args.bin_sec, args.tr, args.delay_sec
        )
        fmri_binned_h1, _ = align_and_assert_bins(fmri_binned_h1, model_binned)
        maps_h1 = compute_rsa_maps_for_ks(
            fmri_binned_h1, model_binned, half1_subs, split_id, "half1", args
        )

        del fmri_h1, fmri_binned_h1, tmpl_img_h1
        gc.collect()

        # ── Half 2 ──────────────────────────────────────────────────────────
        log.info("  [Half 2] Loading & averaging fMRI ...")
        fmri_h2, run_trs_h2, tmpl_img_h2 = load_half_average(half2_subs, args)

        avg_h2_path = split_dir / f"half2_avg.dtseries.nii"
        save_half_average_cifti(fmri_h2, tmpl_img_h2, avg_h2_path)

        log.info("  [Half 2] Binning & searchlight ...")
        fmri_binned_h2 = preprocess_fmri(
            fmri_h2, timing_df, run_trs_h2, args.bin_sec, args.tr, args.delay_sec
        )
        fmri_binned_h2, _ = align_and_assert_bins(fmri_binned_h2, model_binned)
        maps_h2 = compute_rsa_maps_for_ks(
            fmri_binned_h2, model_binned, half2_subs, split_id, "half2", args
        )

        del fmri_h2, fmri_binned_h2, tmpl_img_h2
        gc.collect()

        # ── Reliability metrics ──────────────────────────────────────────────
        results[split_id] = {}
        for k in args.k:
            map1, map2 = maps_h1[k], maps_h2[k]
            r_global, _ = pearsonr(map1, map2)

            thresh1 = np.percentile(map1, 90)
            thresh2 = np.percentile(map2, 90)
            mask1, mask2 = map1 >= thresh1, map2 >= thresh2
            dice = (
                2 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))
            )

            results[split_id][str(k)] = {
                "r_global":   float(r_global),
                "dice_top10": float(dice),
            }
            log.info(
                f"  k={k:3d} | Global r: {r_global:.4f} | "
                f"Top-10% Dice: {dice:.4f}"
            )

        # Persist after each split so partial runs are resumable
        with open(metadata_file, "w") as f:
            json.dump(results, f, indent=4)
        log.info(f"  Results saved → {metadata_file.name}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("═══ All splits complete ═══")
    for k in args.k:
        r_vals = [results[s][str(k)]["r_global"]   for s in results if str(k) in results[s]]
        d_vals = [results[s][str(k)]["dice_top10"] for s in results if str(k) in results[s]]
        if r_vals:
            log.info(
                f"k={k:3d} | r = {np.mean(r_vals):.4f} ± {np.std(r_vals):.4f} | "
                f"Dice = {np.mean(d_vals):.4f} ± {np.std(d_vals):.4f}  "
                f"(n={len(r_vals)} splits)"
            )


if __name__ == "__main__":
    main()
