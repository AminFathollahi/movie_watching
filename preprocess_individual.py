"""
Preprocess HCP 7T movie-watching CIFTI dtseries data per subject.

Pipeline per subject (4 runs):
  1. Load CIFTI dtseries
  2. Extract left + right cortical vertices (drop subcortex)
  3. Optional preprocessing (flags control which steps run):
       --gsr      — global signal regression  (default: on)
       --z-score  — per-vertex z-score per run (default: on)
       --sg-filter — Savitzky-Golay high-pass  (default: off, used by cf_modeling)
       --psc      — percent signal change      (default: off, used by cf_modeling)
  4. Concatenate all 4 runs (ALL timepoints — no timing filtering here)
  5. Save as a single CIFTI dtseries  (n_cortex_vertices × T_total)
     and a run-length array           (n_runs,)  run_trs.npy

Timing-based extraction and hemodynamic delay are applied downstream in the
RSA or encoding analysis scripts, which receive --timing-csv and --delay-sec.

Output filenames encode preprocessing applied:
  {sub}_gsr_zscore_cortex_59k.dtseries.nii
  {sub}_gsr_zscore_run_trs.npy
  group_average_gsr_zscore_cortex_59k.dtseries.nii
  group_average_gsr_zscore_run_trs.npy   (identical to any subject's)

Usage:
  python preprocess_individual.py \\
      --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \\
      --out-dir /path/to/preprocessed \\
      --subjects-list /path/to/subjects.txt \\
      [--tr 1.0] [--dry-run] \\
      [--no-gsr] [--no-z-score] [--sg-filter] [--psc] \\
      [--no-save-individual] [--no-save-average]
"""

import argparse
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Run 1→AP, 2→PA, 3→PA, 4→AP (HCP 7T phase-encoding protocol)
PHASE_MAP = {1: "AP", 2: "PA", 3: "PA", 4: "AP"}
RUN_IDS   = [1, 2, 3, 4]

SG_WINDOW = 201   # Savitzky-Golay window length (cf_modeling pipeline)
SG_ORDER  = 3     # Savitzky-Golay polynomial order


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Preprocess HCP 7T movie fMRI CIFTI data per subject.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", required=True,
                   help="Root directory containing flat CIFTI dtseries files.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for preprocessed dtseries files.")
    p.add_argument("--subjects-list", required=True,
                   help="Text file with one 6-digit subject ID per line.")
    p.add_argument("--tr", type=float, default=1.0,
                   help="TR in seconds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Process first subject only; print shapes and stats, do not save.")

    prep = p.add_argument_group("preprocessing steps")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction,
                      help="Global signal regression.")
    prep.add_argument("--z-score", default=True, action=argparse.BooleanOptionalAction,
                      dest="z_score", help="Z-score per vertex per run.")
    prep.add_argument("--sg-filter", default=False, action=argparse.BooleanOptionalAction,
                      dest="sg_filter",
                      help=f"Savitzky-Golay high-pass (window={SG_WINDOW}, order={SG_ORDER}).")
    prep.add_argument("--psc", default=False, action=argparse.BooleanOptionalAction,
                      help="Percent signal change normalization.")

    out = p.add_argument_group("output control")
    out.add_argument("--save-individual", default=True,
                     action=argparse.BooleanOptionalAction, dest="save_individual",
                     help="Save per-subject preprocessed dtseries files.")
    out.add_argument("--save-average", default=True,
                     action=argparse.BooleanOptionalAction, dest="save_average",
                     help="Compute and save group-average dtseries.")

    return p.parse_args()


# =============================================================================
# Preprocessing helpers
# =============================================================================

def preprocessing_suffix(args) -> str:
    """Build a filename suffix reflecting active preprocessing steps.

    Args:
        args: parsed argparse namespace

    Returns:
        str — e.g. "gsr_zscore" or "sg_psc"
    """
    parts = []
    if args.gsr:       parts.append("gsr")
    if args.z_score:   parts.append("zscore")
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    return "_".join(parts) if parts else "raw"


def apply_gsr(data: np.ndarray) -> None:
    """Subtract the global signal (mean across vertices) per timepoint, in-place.

    Args:
        data: (n_cortex, T)
    """
    data -= data.mean(axis=0, keepdims=True)


def apply_zscore(data: np.ndarray) -> None:
    """Z-score per vertex across time, in-place.

    Args:
        data: (n_cortex, T)
    """
    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    data -= mu
    data /= sd
    np.nan_to_num(data, copy=False)


def apply_sg_filter(data: np.ndarray) -> None:
    """Subtract Savitzky-Golay low-frequency trend, in-place.

    Args:
        data: (n_cortex, T)
    """
    from scipy.signal import savgol_filter
    data -= savgol_filter(data, window_length=SG_WINDOW, polyorder=SG_ORDER, axis=1)


def apply_psc(data: np.ndarray) -> None:
    """Percent signal change normalization, in-place.

    Args:
        data: (n_cortex, T)
    """
    mean = data.mean(axis=1, keepdims=True)
    mean[mean == 0] = 1.0
    data -= mean
    data /= (np.abs(mean) / 100.0)
    np.nan_to_num(data, copy=False)


def preprocess_run(data: np.ndarray, args) -> None:
    """Apply the active preprocessing steps to one run's data, in-place.

    Args:
        data: (n_cortex, T)
        args: parsed argparse namespace
    """
    if args.sg_filter: apply_sg_filter(data)
    if args.psc:       apply_psc(data)
    if args.gsr:       apply_gsr(data)
    if args.z_score:   apply_zscore(data)


# =============================================================================
# I/O helpers
# =============================================================================

def load_subjects(subjects_list: str) -> list:
    """Read subject IDs from a plain-text file (one per line)."""
    return [ln.strip() for ln in Path(subjects_list).read_text().strip().splitlines()
            if ln.strip()]


def get_run_path(raw_dir: Path, sub: str, run_id: int) -> Path:
    """Return path to one subject's CIFTI dtseries for the given run."""
    phase = PHASE_MAP[run_id]
    return raw_dir / f"{sub}_tfMRI_MOVIE{run_id}_7T_{phase}_Atlas_1.6mm_hp2000_clean.dtseries.nii"


def extract_cortex(img) -> tuple:
    """Extract left + right cortical vertices from a CIFTI dtseries image.

    Args:
        img: nibabel Cifti2Image

    Returns:
        data: (n_cortex, T) float32
        cortex_bm_axis: BrainModelAxis for L+R cortex
    """
    bm_axis = img.header.get_axis(1)
    full_data = img.get_fdata(dtype=np.float32)  # (T, n_all)

    cortex_slices = []
    for name, sl, _ in bm_axis.iter_structures():
        if "CORTEX_LEFT" in name or "CORTEX_RIGHT" in name:
            cortex_slices.append(sl)

    if not cortex_slices:
        raise RuntimeError("No cortical structures found in CIFTI brain model axis.")

    cortex_cols = np.concatenate([np.arange(sl.start, sl.stop) for sl in cortex_slices])
    return full_data[:, cortex_cols].T, bm_axis[cortex_cols]


def save_dtseries(data: np.ndarray, cortex_bm_axis, out_path: Path, tr: float) -> None:
    """Save (n_cortex, T) array as a CIFTI dtseries.nii file.

    Args:
        data: (n_cortex, T) float32
        cortex_bm_axis: BrainModelAxis
        out_path: Path
        tr: float
    """
    n_vertices, n_trs = data.shape
    series_axis = nib.cifti2.SeriesAxis(start=0.0, step=tr, size=n_trs, unit="SECOND")
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, cortex_bm_axis))
    img = nib.Cifti2Image(data.T.astype(np.float32), header=header)
    nib.save(img, str(out_path))


# =============================================================================
# Subject pipeline
# =============================================================================

def preprocess_subject(sub: str, raw_dir: Path, out_dir: Path,
                        tr: float, dry_run: bool, suffix: str, args) -> bool:
    """Run the full preprocessing pipeline for one subject.

    Concatenates all 4 run TRs (no timing filtering — downstream analysis handles
    video-on extraction and hemodynamic delay).

    Args:
        sub: str
        raw_dir: Path
        out_dir: Path
        tr: float
        dry_run: bool
        suffix: str — preprocessing suffix for filenames
        args: argparse namespace

    Returns:
        bool — True on success
    """
    cifti_out = out_dir / f"{sub}_{suffix}_cortex_59k.dtseries.nii"
    trs_out   = out_dir / f"{sub}_{suffix}_run_trs.npy"

    if cifti_out.exists() and trs_out.exists() and not dry_run and args.save_individual:
        print(f"    skipping — outputs already exist: {cifti_out.name}")
        return True

    cortex_bm_axis = None
    run_chunks = []
    run_trs    = []

    for run_id in RUN_IDS:
        path = get_run_path(raw_dir, sub, run_id)
        img = nib.load(str(path))
        n_trs_run = img.shape[0]

        cortex_data, bm_ax = extract_cortex(img)   # (n_cortex, T)
        if cortex_bm_axis is None:
            cortex_bm_axis = bm_ax

        if dry_run:
            print(f"    run {run_id}: raw {cortex_data.shape}  "
                  f"GS=[{cortex_data.mean(axis=0).min():.2f}, "
                  f"{cortex_data.mean(axis=0).max():.2f}]", flush=True)

        preprocess_run(cortex_data, args)

        if dry_run:
            print(f"    run {run_id}: post-prep  mean={cortex_data.mean():.4f}  "
                  f"std={cortex_data.std():.4f}", flush=True)

        run_chunks.append(cortex_data)       # (n_cortex, T_run)
        run_trs.append(n_trs_run)

    combined = np.concatenate(run_chunks, axis=1)   # (n_cortex, T_total)

    if dry_run:
        print(f"\n  [DRY RUN] combined shape: {combined.shape}  "
              f"run_trs={run_trs}")
        print(f"  [DRY RUN] mean={combined.mean():.4f}  std={combined.std():.4f}")
        print("  [DRY RUN] no files saved.")
        return True

    if args.save_individual:
        save_dtseries(combined, cortex_bm_axis, cifti_out, tr)
        np.save(str(trs_out), np.array(run_trs, dtype=np.int32))

    return True


# =============================================================================
# Group average
# =============================================================================

def compute_group_average(out_dir: Path, subjects: list, tr: float, suffix: str) -> None:
    """Compute and save the group-average dtseries using Welford's online algorithm.

    Args:
        out_dir: Path
        subjects: list[str]
        tr: float
        suffix: str
    """
    print("\n" + "=" * 60)
    print("COMPUTING GROUP AVERAGE")
    print("=" * 60)

    avg_cifti = out_dir / f"group_average_{suffix}_cortex_59k.dtseries.nii"
    avg_trs   = out_dir / f"group_average_{suffix}_run_trs.npy"

    running_mean   = None
    cortex_bm_axis = None
    ref_run_trs    = None
    n_loaded = 0
    skipped  = []

    for idx, sub in enumerate(subjects, 1):
        fpath = out_dir / f"{sub}_{suffix}_cortex_59k.dtseries.nii"
        tpath = out_dir / f"{sub}_{suffix}_run_trs.npy"
        if not fpath.exists():
            skipped.append(sub)
            continue

        print(f"  [{idx:03d}/{len(subjects):03d}] {sub} ...", end=" ", flush=True)
        t0 = time.time()
        img  = nib.load(str(fpath))
        data = img.get_fdata(dtype=np.float32).T   # (n_cortex, T)

        if running_mean is None:
            running_mean   = np.zeros_like(data, dtype=np.float64)
            cortex_bm_axis = img.header.get_axis(1)
            if tpath.exists():
                ref_run_trs = np.load(str(tpath))

        if data.shape != running_mean.shape:
            print(f"SKIPPED — shape mismatch {data.shape}")
            skipped.append(sub)
            continue

        n_loaded += 1
        running_mean += (data - running_mean) / n_loaded   # Welford
        print(f"ok ({time.time() - t0:.1f} s)")

    if n_loaded == 0:
        print("No subjects loaded — group average not saved.")
        return

    print(f"\n  Averaged {n_loaded} subjects.")
    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")

    group_mean = running_mean.astype(np.float32)
    save_dtseries(group_mean, cortex_bm_axis, avg_cifti, tr)
    print(f"  Saved: {avg_cifti.name}  shape={group_mean.shape}")

    if ref_run_trs is not None:
        np.save(str(avg_trs), ref_run_trs)
        print(f"  Saved: {avg_trs.name}  run_trs={ref_run_trs.tolist()}")


# =============================================================================
# Main
# =============================================================================

def main():
    args     = parse_args()
    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(args.subjects_list)
    suffix   = preprocessing_suffix(args)

    if args.dry_run:
        subjects = subjects[:1]
        print(f"DRY RUN — {subjects[0]} only.  preprocessing={suffix}\n")

    fail_log = out_dir / "preprocessing_failures.txt"
    n_ok = n_fail = 0
    t_wall = time.time()

    for idx, sub in enumerate(subjects, 1):
        t_sub = time.time()
        print(f"[{idx:03d}/{len(subjects):03d}] {sub} ...", end=" ", flush=True)
        try:
            preprocess_subject(sub=sub, raw_dir=raw_dir, out_dir=out_dir,
                               tr=args.tr, dry_run=args.dry_run,
                               suffix=suffix, args=args)
            print(f"done in {time.time() - t_sub:.1f} s")
            n_ok += 1
        except Exception:
            print(f"FAILED ({time.time() - t_sub:.1f} s)  — see {fail_log.name}")
            n_fail += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{sub}\t{traceback.format_exc()}\n{'─'*60}\n")

    print(f"\nDone. {n_ok} succeeded, {n_fail} failed. "
          f"Total: {time.time() - t_wall:.1f} s")
    if n_fail:
        print(f"Failures: {fail_log}")

    if not args.dry_run and args.save_average:
        compute_group_average(out_dir, subjects, args.tr, suffix)


if __name__ == "__main__":
    main()
