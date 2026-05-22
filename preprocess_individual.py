"""
Preprocess HCP 7T movie-watching CIFTI dtseries data per subject.

This script isolates biological variance by applying continuous signal cleaning 
before any temporal alignment or analysis. 

Per run: load → extract cortex → SG (optional) → PSC (optional) → GSR (optional) → concatenate 4 runs
Output: {sub}_<suffix>_cortex_59k.dtseries.nii
        {sub}_<suffix>_run_trs.npy

Streaming / batch use
---------------------
The core preprocessing function (preprocess_subject) returns numpy arrays and 
does NOT save to disk. Analysis scripts can import and call it directly for 
in-memory pipelines:

    from preprocess_individual import preprocess_subject
    import types
    args = types.SimpleNamespace(sg_filter=True, psc=True, gsr=True)
    data, bm_axis, run_trs = preprocess_subject(sub, raw_dir, tr=1.0, args=args)
    # pass to rsa_utils for delay slicing, binning, and z-scoring

Use --save-individual to persist per-subject continuous CIFTIs to disk.
Use --save-average to compute and save a continuous group-average CIFTI 
(accumulates subjects in-memory; does not require --save-individual).

Usage (saving individual continuous maps to disk):
  python preprocess_individual.py \
      --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
      --out-dir /home/amin/Research/Representation/Movie/outputs/preprocessed \
      --subjects-list /home/amin/Research/Representation/Movie/data/subjects.txt \
      --sg-filter --psc --gsr \
      --save-individual
"""

import argparse
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

PHASE_MAP = {1: "AP", 2: "PA", 3: "PA", 4: "AP"}
RUN_IDS   = [1, 2, 3, 4]

SG_WINDOW = 201
SG_ORDER  = 3


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess HCP 7T movie fMRI CIFTI data per subject (continuous signal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", required=True,
                   help="Root directory containing flat raw CIFTI dtseries files.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for preprocessed dtseries files.")
    p.add_argument("--subjects-list", required=True,
                   help="Text file with one 6-digit subject ID per line.")
    p.add_argument("--tr", type=float, default=1.0,
                   help="TR in seconds.")
    p.add_argument("--dry-run", action="store_true",
                   help="Process first subject only; print shapes, do not save.")

    prep = p.add_argument_group("preprocessing steps")
    prep.add_argument("--sg-filter", default=False, action=argparse.BooleanOptionalAction,
                      dest="sg_filter",
                      help=f"Savitzky-Golay high-pass (window={SG_WINDOW}, order={SG_ORDER}).")
    prep.add_argument("--psc", default=False, action=argparse.BooleanOptionalAction,
                      help="Percent signal change normalization (per run).")
    prep.add_argument("--gsr", default=True, action=argparse.BooleanOptionalAction,
                      help="Global signal regression.")

    out = p.add_argument_group("output control (default: no saving — streaming mode)")
    out.add_argument("--save-individual", default=False,
                     action=argparse.BooleanOptionalAction, dest="save_individual",
                     help="Save per-subject preprocessed dtseries + run_trs.npy to --out-dir.")
    out.add_argument("--save-average", default=False,
                     action=argparse.BooleanOptionalAction, dest="save_average",
                     help="Compute and save group-average dtseries to --out-dir. "
                          "Accumulates subjects in-memory; does not require --save-individual.")

    return p.parse_args()


# =============================================================================
# Preprocessing helpers
# =============================================================================

def preprocessing_suffix(args) -> str:
    """Build filename suffix from active preprocessing steps."""
    parts = []
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    if args.gsr:       parts.append("gsr")
    return "_".join(parts) if parts else "raw"


def apply_gsr(data: np.ndarray) -> None:
    """Subtract global signal (mean across vertices) per timepoint, in-place."""
    data -= data.mean(axis=0, keepdims=True)


def apply_sg_filter(data: np.ndarray) -> None:
    """Subtract Savitzky-Golay low-frequency trend, in-place."""
    from scipy.signal import savgol_filter
    data -= savgol_filter(data, window_length=SG_WINDOW, polyorder=SG_ORDER, axis=1)


def apply_psc(data: np.ndarray, baseline_mean: np.ndarray = None) -> None:
    """Percent signal change normalization, in-place.

    PSC = (signal − mean) / |mean| × 100

    Args:
        data:          (n_vertices, T) float32 — preprocessed data (may already
                       have had a trend subtracted if SG was applied first)
        baseline_mean: (n_vertices, 1) optional pre-SG temporal mean to use as
                       the normalization baseline.  Pass this when SG has already
                       been applied; otherwise the post-SG mean is near-zero,
                       causing numerical explosion.  If None (PSC without prior
                       SG), the current temporal mean is computed and subtracted
                       as part of normalization.
    """
    if baseline_mean is None:
        # No prior SG — compute mean from current data and subtract it
        mean = data.mean(axis=1, keepdims=True).copy()
        mean[np.abs(mean) < 1e-6] = 1.0
        data -= mean
        data /= (np.abs(mean) / 100.0)
    else:
        # SG was applied before PSC: SG already subtracted the trend (≈ mean),
        # so just scale by the pre-SG absolute baseline.
        mean = baseline_mean.copy()
        mean[np.abs(mean) < 1e-6] = 1.0
        data /= (np.abs(mean) / 100.0)
    np.nan_to_num(data, copy=False)


def preprocess_run(data: np.ndarray, args) -> None:
    """Apply active preprocessing steps to one run independently, in-place.

    Order: SG → PSC → GSR.  When both SG and PSC are active, the pre-SG
    temporal mean is saved and passed to PSC so that division uses the absolute
    BOLD baseline rather than the near-zero post-SG residual.
    """
    # Capture baseline BEFORE SG removes the DC offset (needed for PSC denominator)
    pre_sg_mean = None
    if getattr(args, 'sg_filter', False) and getattr(args, 'psc', False):
        pre_sg_mean = data.mean(axis=1, keepdims=True).copy()

    if getattr(args, 'sg_filter', False): apply_sg_filter(data)
    if getattr(args, 'psc', False):       apply_psc(data, pre_sg_mean)
    if getattr(args, 'gsr', False):       apply_gsr(data)


# =============================================================================
# I/O helpers
# =============================================================================

def load_subjects(subjects_list: str) -> list:
    """Load subject IDs from a text file, skipping blank lines and # comments."""
    subs = []
    for ln in Path(subjects_list).read_text().strip().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            subs.append(ln)
    return subs


def get_run_path(raw_dir: Path, sub: str, run_id: int) -> Path:
    phase = PHASE_MAP[run_id]
    return raw_dir / f"{sub}_tfMRI_MOVIE{run_id}_7T_{phase}_Atlas_1.6mm_hp2000_clean.dtseries.nii"


def extract_cortex(img) -> tuple:
    """Extract left + right cortical vertices from a CIFTI dtseries image."""
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
    """Save (n_cortex, T) array as a CIFTI dtseries.nii."""
    n_vertices, n_trs = data.shape
    series_axis = nib.cifti2.SeriesAxis(start=0.0, step=tr, size=n_trs, unit="SECOND")
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, cortex_bm_axis))
    img = nib.Cifti2Image(data.T.astype(np.float32), header=header)
    nib.save(img, str(out_path))


# =============================================================================
# Core preprocessing function
# =============================================================================

def preprocess_subject(sub: str, raw_dir: Path, tr: float, args) -> tuple:
    """Load and preprocess all 4 runs (continuous mode).

    Per-run preprocessing: SG→PSC→GSR (steps active in args).

    Args:
        sub:     subject ID string
        raw_dir: root directory of raw CIFTI files
        tr:      TR in seconds
        args:    namespace with sg_filter, psc, gsr boolean flags

    Returns:
        data:           (n_cortex, T_total) float32 — all 4 runs concatenated
        cortex_bm_axis: nibabel BrainModelAxis for the cortex columns
        run_trs:        (4,) int32 — number of TRs per run
    """
    cortex_bm_axis = None
    run_chunks = []
    run_trs    = []

    for run_id in RUN_IDS:
        path = get_run_path(raw_dir, sub, run_id)
        img  = nib.load(str(path))
        cortex_data, bm_ax = extract_cortex(img)
        if cortex_bm_axis is None:
            cortex_bm_axis = bm_ax
        run_trs.append(img.shape[0])
        
        preprocess_run(cortex_data, args)
        run_chunks.append(cortex_data)

    combined = np.concatenate(run_chunks, axis=1)
    return combined, cortex_bm_axis, np.array(run_trs, dtype=np.int32)


# =============================================================================
# Disk-based group average
# =============================================================================

def compute_group_average_from_disk(out_dir: Path, subjects: list,
                                    tr: float, suffix: str) -> None:
    """Load existing per-subject continuous CIFTIs from disk and save a group average."""
    print("\n" + "=" * 60)
    print("COMPUTING GROUP AVERAGE FROM DISK")
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
        running_mean += (data - running_mean) / n_loaded
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

    subjects  = load_subjects(args.subjects_list)
    suffix    = preprocessing_suffix(args)

    print(f"Preprocessing continuous signals. Active flags: {suffix}")

    if not args.save_individual and not args.save_average:
        print("Note: running without --save-individual or --save-average — "
              "data will be preprocessed in memory only and not written to disk.")

    if args.dry_run:
        subjects = subjects[:1]
        print(f"DRY RUN — {subjects[0]} only. preprocessing={suffix}\n")

    running_mean   = None
    cortex_bm_ref  = None
    run_trs_ref    = None
    n_avg          = 0

    fail_log = out_dir / "preprocessing_failures.txt"
    n_ok = n_fail = 0
    t_wall = time.time()

    for idx, sub in enumerate(subjects, 1):
        if args.save_individual and not args.dry_run:
            cifti_out = out_dir / f"{sub}_{suffix}_cortex_59k.dtseries.nii"
            trs_out   = out_dir / f"{sub}_{suffix}_run_trs.npy"
            if cifti_out.exists() and trs_out.exists():
                print(f"[{idx:03d}/{len(subjects):03d}] {sub} — skipping (exists)")
                n_ok += 1
                continue

        t_sub = time.time()
        print(f"[{idx:03d}/{len(subjects):03d}] {sub} ...", end=" ", flush=True)

        try:
            data, bm_axis, run_trs = preprocess_subject(
                sub=sub, raw_dir=raw_dir, tr=args.tr, args=args)

            if args.dry_run:
                print(f"\n  shape={data.shape}  mean={data.mean():.4f}  "
                      f"std={data.std():.4f}")
                print("  [DRY RUN] no files saved.")
                n_ok += 1
                break

            if args.save_individual:
                cifti_out = out_dir / f"{sub}_{suffix}_cortex_59k.dtseries.nii"
                trs_out   = out_dir / f"{sub}_{suffix}_run_trs.npy"
                save_dtseries(data, bm_axis, cifti_out, args.tr)
                np.save(str(trs_out), run_trs)

            if args.save_average:
                n_avg += 1
                if running_mean is None:
                    running_mean  = np.zeros_like(data, dtype=np.float64)
                    cortex_bm_ref = bm_axis
                    run_trs_ref   = run_trs
                if data.shape == running_mean.shape:
                    running_mean += (data - running_mean) / n_avg
                else:
                    print(f"  WARNING: {sub} shape {data.shape} != "
                          f"{running_mean.shape} — excluded from average")
                    n_avg -= 1

            print(f"done ({time.time() - t_sub:.1f} s)")
            n_ok += 1

        except Exception:
            print(f"FAILED ({time.time() - t_sub:.1f} s)")
            n_fail += 1
            with open(fail_log, "a") as fh:
                fh.write(f"{sub}\t{traceback.format_exc()}\n{'─'*60}\n")

    print(f"\nDone. {n_ok} succeeded, {n_fail} failed. "
          f"Total: {time.time() - t_wall:.1f} s")
    if n_fail:
        print(f"Failures logged to: {fail_log}")

    if args.save_average and running_mean is not None and not args.dry_run:
        print(f"\nSaving group average (n={n_avg}) ...")
        avg_cifti = out_dir / f"group_average_{suffix}_cortex_59k.dtseries.nii"
        avg_trs   = out_dir / f"group_average_{suffix}_run_trs.npy"
        save_dtseries(running_mean.astype(np.float32), cortex_bm_ref, avg_cifti, args.tr)
        print(f"  Saved: {avg_cifti.name}  shape={running_mean.shape}")
        if run_trs_ref is not None:
            np.save(str(avg_trs), run_trs_ref)
            print(f"  Saved: {avg_trs.name}  run_trs={run_trs_ref.tolist()}")


if __name__ == "__main__":
    main()