"""
cf_modeling/overlap.py
======================
Per-subject group statistics.

One-sample t-test (H0: mean=0), Cohen's d, and FDR-corrected significance
masks (Benjamini-Hochberg, q=0.05) across subjects for R2_a_nc, R2_b_nc,
and per-subject integration score.  Outputs .npy + CIFTI dscalar.nii.

Run
---
    python overlap.py --mode per_subject --roi-a A5 --roi-b FFC --min-subjects 5
"""

import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUTPUTS  # noqa: E402

try:
    from cf_modeling.cf_naming import (
        cf_model_map_stems, legacy_cf_model_map_stems, resolve_cf_model_map_path,
    )
except ModuleNotFoundError:
    from cf_naming import (
        cf_model_map_stems, legacy_cf_model_map_stems, resolve_cf_model_map_path,
    )

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_OUT_BASE = str(OUTPUTS / "cf_modeling")

FDR_Q = 0.05
ALPHA = 0.05


# =============================================================================
# Shared CIFTI helpers
# =============================================================================

def _bm_axis_from_template(template_cifti):
    return nib.load(template_cifti).header.get_axis(1)


def _save_dscalar(arr, name, bm_axis, out_dir):
    arr_f32   = arr.astype(np.float32)
    scalar_ax = nib.cifti2.ScalarAxis([name])
    header    = nib.cifti2.Cifti2Header.from_axes((scalar_ax, bm_axis))
    img       = nib.Cifti2Image(arr_f32.reshape(1, -1), header=header)
    path      = os.path.join(out_dir, f"{name}.dscalar.nii")
    nib.save(img, path)
    log.info(f"  Saved {name}.dscalar.nii")
    return path


# =============================================================================
# Per-subject map collection
# =============================================================================

def collect_maps(subjects_dir, map_name, min_subjects, legacy_name=None):
    """Load canonical subject CF maps, with a temporary legacy fallback."""
    sub_dirs = sorted(glob(os.path.join(subjects_dir, "*")))
    arrays, missing = [], []
    for sd in sub_dirs:
        path = resolve_cf_model_map_path(sd, map_name, legacy_name)
        if os.path.exists(path):
            arrays.append(np.load(path))
        else:
            missing.append(os.path.basename(sd))
    if missing:
        log.warning(f"  {map_name}: missing for {len(missing)} subjects "
                    f"(e.g. {missing[:3]})")
    if len(arrays) < min_subjects:
        raise RuntimeError(
            f"Only {len(arrays)} subjects have {map_name}.npy — "
            f"need at least {min_subjects}.")
    log.info(f"  {map_name}: loaded {len(arrays)} subjects")
    return np.stack(arrays, axis=0)


# =============================================================================
# PER_SUBJECT: Group statistics
# =============================================================================

def _one_sample_stats(maps):
    N = maps.shape[0]
    t_stat, p_val = scipy_stats.ttest_1samp(maps, popmean=0, axis=0, nan_policy="omit")
    d_stat = t_stat / np.sqrt(N)
    return t_stat.astype(np.float32), d_stat.astype(np.float32), p_val.astype(np.float32)


def _fdr_mask(p_values, q=FDR_Q):
    finite = np.isfinite(p_values)
    mask   = np.zeros(p_values.shape, dtype=bool)
    if not finite.any():
        return mask
    try:
        rejected = scipy_stats.false_discovery_control(
            p_values[finite], axis=0, method="bh") < q
    except AttributeError:
        p_fin  = p_values[finite]
        n      = len(p_fin)
        order  = np.argsort(p_fin)
        ranked = np.empty(n, dtype=int)
        ranked[order] = np.arange(1, n + 1)
        rejected = p_fin <= (ranked / n) * q
    mask[finite] = rejected
    return mask


def run_per_subject(args):
    roi_root     = f"{args.output_base}/per_subject/{args.roi_a}_{args.roi_b}"
    subjects_dir = f"{roi_root}/subjects"
    group_dir    = f"{roi_root}/group"
    stats_dir    = f"{group_dir}/stats"
    cifti_dir    = f"{group_dir}/cifti_maps"
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(cifti_dir, exist_ok=True)

    log.info("\nLoading per-subject null-corrected R² maps …")
    stems = cf_model_map_stems(args.roi_a, args.roi_b)
    legacy_stems = legacy_cf_model_map_stems(args.roi_a, args.roi_b)
    maps_a = collect_maps(subjects_dir, stems["null_corrected_split_r2_a"],
                          args.min_subjects, legacy_stems["null_corrected_split_r2_a"])
    maps_b = collect_maps(subjects_dir, stems["null_corrected_split_r2_b"],
                          args.min_subjects, legacy_stems["null_corrected_split_r2_b"])
    maps_product = collect_maps(subjects_dir, stems["joint_split_r2_geomean"],
                                args.min_subjects, legacy_stems["joint_split_r2_geomean"])
    N = maps_a.shape[0]
    log.info(f"  N subjects: {N}")

    critical_t = float(scipy_stats.t.ppf(1 - ALPHA / 2, df=N - 1))
    log.info(f"  Critical t (two-tailed α={ALPHA}, df={N-1}): {critical_t:.3f}")

    log.info("\nRunning one-sample t-tests (H0: mean=0) …")
    t_a, d_a, p_a = _one_sample_stats(maps_a)
    t_b, d_b, p_b = _one_sample_stats(maps_b)
    t_i, d_i, p_i = _one_sample_stats(maps_product) 

    log.info(f"  {args.roi_a}: mean_d={d_a.mean():.4f}  "
             f"frac_sig: {np.mean(np.abs(t_a) > critical_t):.1%}")
    log.info(f"  {args.roi_b}: mean_d={d_b.mean():.4f}  "
             f"frac_sig: {np.mean(np.abs(t_b) > critical_t):.1%}")
    log.info(f"  Integration: mean_d={d_i.mean():.4f}  "
             f"frac_sig: {np.mean(np.abs(t_i) > critical_t):.1%}")

    log.info("\nApplying FDR correction (Benjamini-Hochberg) …")
    fdr_a = _fdr_mask(p_a)
    fdr_b = _fdr_mask(p_b)
    log.info(f"  FDR-sig {args.roi_a}: {fdr_a.sum()} / {len(fdr_a)} "
             f"({fdr_a.mean():.1%})")
    log.info(f"  FDR-sig {args.roi_b}: {fdr_b.sum()} / {len(fdr_b)} "
             f"({fdr_b.mean():.1%})")

    log.info(f"\nSaving stats .npy to {stats_dir} …")
    stat_maps = {
        f"t_{args.roi_a}_nc":         t_a,
        f"d_{args.roi_a}_nc":         d_a,
        f"p_{args.roi_a}_nc":         p_a,
        f"t_{args.roi_b}_nc":         t_b,
        f"d_{args.roi_b}_nc":         d_b,
        f"p_{args.roi_b}_nc":         p_b,
        "t_integration":               t_i,
        "d_integration":               d_i,
        "p_integration":               p_i,
        f"fdr_mask_{args.roi_a}_nc":  fdr_a.astype(np.float32),
        f"fdr_mask_{args.roi_b}_nc":  fdr_b.astype(np.float32),
    }
    for name, arr in stat_maps.items():
        np.save(os.path.join(stats_dir, f"{name}.npy"), arr)

    summary = {
        "N_subjects":                   N,
        "ROI_A":                        args.roi_a,
        "ROI_B":                        args.roi_b,
        "critical_t":                   critical_t,
        "alpha":                        ALPHA,
        "fdr_q":                        FDR_Q,
        f"mean_d_{args.roi_a}_nc":      float(d_a[np.isfinite(d_a)].mean()),
        f"mean_d_{args.roi_b}_nc":      float(d_b[np.isfinite(d_b)].mean()),
        "mean_d_integration":           float(d_i[np.isfinite(d_i)].mean()),
        f"fdr_sig_{args.roi_a}":        int(fdr_a.sum()),
        f"fdr_sig_{args.roi_b}":        int(fdr_b.sum()),
        f"uncorr_sig_{args.roi_a}":     int((np.abs(t_a) > critical_t).sum()),
        f"uncorr_sig_{args.roi_b}":     int((np.abs(t_b) > critical_t).sum()),
    }
    with open(os.path.join(stats_dir, "stats_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("  stats_summary.json saved")

    log.info("\nSaving stat maps as CIFTI dscalar.nii …")
    bm_axis = _bm_axis_from_template(args.template_cifti)
    cifti_maps = [
        (f"t_{args.roi_a}_nc",         t_a),
        (f"d_{args.roi_a}_nc",         d_a),
        (f"t_{args.roi_b}_nc",         t_b),
        (f"d_{args.roi_b}_nc",         d_b),
        ("t_integration",               t_i),
        ("d_integration",               d_i),
        (f"fdr_mask_{args.roi_a}_nc",  fdr_a.astype(np.float32)),
        (f"fdr_mask_{args.roi_b}_nc",  fdr_b.astype(np.float32)),
    ]
    for name, arr in cifti_maps:
        _save_dscalar(arr, name, bm_axis, cifti_dir)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-subject group statistics of CF model maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",          required=True, choices=["per_subject"])
    p.add_argument("--roi-a",         dest="roi_a",          default="A1")
    p.add_argument("--roi-b",         dest="roi_b",          default="V1")
    p.add_argument("--output-base",   dest="output_base",    default=_OUT_BASE)
    p.add_argument("--template-cifti", dest="template_cifti", default=None,
                   help="59k cortex-only CIFTI template (required).")
    p.add_argument("--min-subjects",  dest="min_subjects",   type=int, default=1,
                   help="Minimum subjects required.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.template_cifti is None:
        raise ValueError(
            "--template-cifti is required. Pass the preprocessed group-average 59k "
            "CIFTI (group_average_{suffix}_cortex_59k.dtseries.nii).")

    log.info("=" * 60)
    log.info(f"Group statistics ({args.mode})")
    log.info(f"  ROIs: {args.roi_a} x {args.roi_b}")
    log.info("=" * 60)

    run_per_subject(args)

    log.info("\nGroup statistics complete.")


if __name__ == "__main__":
    main()
