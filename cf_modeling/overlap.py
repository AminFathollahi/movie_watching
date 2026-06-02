"""
cf_modeling/overlap.py
======================
Mode-specific summary analysis.

group_average mode — RSA spatial overlap:
    Spatial correlation (Spearman + 95% CI bootstrap) between the CF-model
    integration_score and PE-AV searchlight RSA maps (audio/video/joint).
    Overlap map: min(integration_score_norm, rsa_joint_norm) per vertex.
    Results JSON saved for each RSA config.

per_subject mode — Group statistics:
    One-sample t-test (H0: mean=0), Cohen's d, and FDR-corrected significance
    masks (Benjamini-Hochberg, q=0.05) across subjects for R2_a_nc, R2_b_nc,
    and per-subject integration score.  Outputs .npy + CIFTI dscalar.nii.

Run
---
    python summary.py --mode group_average --roi_a A1 --roi_b V1 \\
        --rsa_base /path/to/searchlight_rsa_output
    python summary.py --mode per_subject   --roi_a A5 --roi_b FFC \\
        --min_subjects 5
"""

import argparse
import json
import logging
import os
from glob import glob

import nibabel as nib
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_DATA_BASE = "/home/amin/Research/Representation/Movie/data/Setareh"
_HCP_DIR   = f"{_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_OUT_BASE  = "/home/amin/Research/Representation/Movie/outputs/cf_modeling"
_RSA_BASE  = "/home/amin/Research/Representation/Movie/outputs/searchlight_rsa_output"

RSA_MODEL = "pe-av-small-16-frame"

RSA_CONFIGS = {
    "notnormalized_nohrf_global_2s"  : dict(norm="notnormalized", hrf_tag="nohrf", method="global",    bin="2s"),
    "normalized_hrf_blockdiag_2s"    : dict(norm="normalized",    hrf_tag="hrf",   method="blockdiag", bin="2s"),
    "normalized_hrf_global_2s"       : dict(norm="normalized",    hrf_tag="hrf",   method="global",    bin="2s"),
    "notnormalized_hrf_blockdiag_2s" : dict(norm="notnormalized", hrf_tag="hrf",   method="blockdiag", bin="2s"),
    "normalized_nohrf_blockdiag_2s"  : dict(norm="normalized",    hrf_tag="nohrf", method="blockdiag", bin="2s"),
    "normalized_hrf_blockdiag_5s"    : dict(norm="normalized",    hrf_tag="hrf",   method="blockdiag", bin="5s"),
    "notnormalized_hrf_blockdiag_5s" : dict(norm="notnormalized", hrf_tag="hrf",   method="blockdiag", bin="5s"),
}

N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 42
FDR_Q          = 0.05
ALPHA          = 0.05


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

def collect_maps(subjects_dir, map_name, min_subjects):
    """Load map_name.npy from all completed subject directories → (N, n_verts)."""
    sub_dirs = sorted(glob(os.path.join(subjects_dir, "*")))
    arrays, missing = [], []
    for sd in sub_dirs:
        path = os.path.join(sd, f"{map_name}.npy")
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
# GROUP_AVERAGE: RSA spatial overlap
# =============================================================================

def _config_dir(norm, hrf_tag, method):
    if hrf_tag == "hrf":
        return f"k100_{norm}_hrf_{method}"
    return f"k100_{norm}_{method}_delay5s"


def _rsa_path(rsa_base, rsa_model, norm, hrf_tag, method, bin_size, modality, hem):
    config_dir = _config_dir(norm, hrf_tag, method)
    fname = (f"rsa_{rsa_model}_{modality}_k100_spearman_"
             f"{norm}_{hrf_tag}_bin{bin_size}_{hem}_{method}.npy")
    return os.path.join(rsa_base, rsa_model, config_dir, bin_size,
                        modality, f"{hem}_hemisphere", fname)


def _get_grayordinate_indices(template_cifti):
    bm = nib.load(template_cifti).header.get_axis(1)
    lv = rv = None
    for name, _, model in bm.iter_structures():
        if "LEFT"  in name: lv = model.vertex
        if "RIGHT" in name: rv = model.vertex
    return lv, rv


def _load_rsa_fullbrain(rsa_base, rsa_model, norm, hrf_tag, method,
                        bin_size, modality, gray_L, gray_R):
    lf = np.load(_rsa_path(rsa_base, rsa_model, norm, hrf_tag, method, bin_size, modality, "left"))
    rf = np.load(_rsa_path(rsa_base, rsa_model, norm, hrf_tag, method, bin_size, modality, "right"))
    out = np.concatenate([lf[gray_L], rf[gray_R]]).astype(np.float32)
    if out.shape[0] != 59412:
        raise RuntimeError(f"Expected 59412 grayordinates, got {out.shape[0]}")
    return out


def _bootstrap_spearman(x, y, n_boot, seed):
    rng     = np.random.default_rng(seed)
    n       = len(x)
    rho_obs = float(spearmanr(x, y).statistic)
    boots   = np.array([spearmanr(x[rng.integers(0, n, n)],
                                   y[rng.integers(0, n, n)]).statistic
                        for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return rho_obs, float(ci_lo), float(ci_hi)


def _normalise_positive(arr):
    pos = np.clip(arr, 0, None)
    p95 = np.percentile(pos[pos > 0], 95) if (pos > 0).any() else 1.0
    return np.clip(pos / p95, 0, 1).astype(np.float32)


def _run_one_rsa_config(config_name, integration_score, R2_a_nc, R2_b_nc,
                        roi_a, roi_b, rsa_base, rsa_model, gray_L, gray_R,
                        bm_axis, cifti_dir, results_dir):
    cfg = RSA_CONFIGS[config_name]
    norm, hrf_tag, method, bin_size = cfg["norm"], cfg["hrf_tag"], cfg["method"], cfg["bin"]

    log.info(f"\n{'='*60}")
    log.info(f"Config: {config_name}")
    log.info(f"  norm={norm}  hrf={hrf_tag}  method={method}  bin={bin_size}")

    rsa_maps = {}
    for mod_name in ("audio", "video", "joint"):
        try:
            rsa_maps[mod_name] = _load_rsa_fullbrain(
                rsa_base, rsa_model, norm, hrf_tag, method,
                bin_size, mod_name, gray_L, gray_R)
        except FileNotFoundError:
            log.warning(f"  {mod_name} RSA map not found — skipping modality")

    if not rsa_maps:
        log.error(f"  SKIP — no RSA modality maps found for {rsa_model}")
        return None

    for mod_name, rsa_map in rsa_maps.items():
        log.info(f"  RSA {mod_name}={rsa_map.mean():.4f}")

    results  = {}
    rng_seed = BOOTSTRAP_SEED

    for rsa_name, rsa_map in rsa_maps.items():
        rho, ci_lo, ci_hi = _bootstrap_spearman(
            integration_score, rsa_map, N_BOOTSTRAP, rng_seed)
        results[f"rho_integration_vs_rsa_{rsa_name}"] = rho
        results[f"ci_lo_integration_vs_rsa_{rsa_name}"] = ci_lo
        results[f"ci_hi_integration_vs_rsa_{rsa_name}"] = ci_hi
        log.info(f"  Spearman(integration_score, rsa_{rsa_name}): "
                 f"rho={rho:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rng_seed += 1

    # R2_a_nc vs audio RSA, R2_b_nc vs video RSA — only when that modality loaded
    for r2_name, r2_map, rsa_modality in [
        (roi_a.lower(), R2_a_nc, "audio"),
        (roi_b.lower(), R2_b_nc, "video"),
    ]:
        if rsa_modality not in rsa_maps:
            continue
        rho, ci_lo, ci_hi = _bootstrap_spearman(
            r2_map, rsa_maps[rsa_modality], N_BOOTSTRAP, rng_seed)
        results[f"rho_r2{r2_name}_vs_rsa_{rsa_modality}"] = rho
        results[f"ci_lo_r2{r2_name}_vs_rsa_{rsa_modality}"] = ci_lo
        results[f"ci_hi_r2{r2_name}_vs_rsa_{rsa_modality}"] = ci_hi
        log.info(f"  Spearman(R2_{r2_name}_nc, rsa_{rsa_modality}): "
                 f"rho={rho:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rng_seed += 1

    int_norm = _normalise_positive(integration_score)
    overlap_stats = {}
    for mod_name, rsa_map in rsa_maps.items():
        rsa_norm   = _normalise_positive(rsa_map)
        overlap    = np.minimum(int_norm, rsa_norm)
        map_name   = f"overlap_score_{rsa_model}_{mod_name}_{config_name}"
        _save_dscalar(overlap, map_name, bm_axis, cifti_dir)
        log.info(f"  {map_name}: mean={overlap.mean():.4f}  "
                 f"frac>0.1={np.mean(overlap>0.1):.1%}")
        overlap_stats[mod_name] = {
            "mean": float(overlap.mean()),
            "frac_gt_0.1": float(np.mean(overlap > 0.1)),
        }

    results.update({
        "rsa_model": rsa_model,
        "config": config_name, "n_verts": 59412, "n_boot": N_BOOTSTRAP,
        "overlap": overlap_stats,
    })
    json_path = os.path.join(results_dir, f"rsa_overlap_{rsa_model}_{config_name}.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"  Results: {json_path}")
    return results


def run_group_average(args):
    roi_root    = f"{args.output_base}/group_average/{args.roi_a}_{args.roi_b}"
    prep_dir    = f"{roi_root}/prep"
    cifti_dir   = f"{roi_root}/cifti_maps"
    results_dir = f"{roi_root}/results"
    os.makedirs(cifti_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log.info("\nLoading CF-model integration maps …")
    product_map = np.load(os.path.join(prep_dir, "product_map.npy"))
    R2_a_nc           = np.load(os.path.join(prep_dir, f"R2_{args.roi_a}_nc.npy"))
    R2_b_nc           = np.load(os.path.join(prep_dir, f"R2_{args.roi_b}_nc.npy"))
    log.info(f"  product_map: frac>0={np.mean(product_map>0):.1%}")

    bm_axis        = _bm_axis_from_template(args.template_cifti)
    gray_L, gray_R = _get_grayordinate_indices(args.template_cifti)

    configs = list(RSA_CONFIGS.keys()) if args.all_configs else [args.rsa_config]
    log.info(f"\nRunning {len(configs)} RSA config(s): {configs}")

    all_results = {}
    for cfg_name in configs:
        res = _run_one_rsa_config(
            cfg_name, product_map, R2_a_nc, R2_b_nc,
            args.roi_a, args.roi_b,
            args.rsa_base, args.rsa_model,
            gray_L, gray_R, bm_axis, cifti_dir, results_dir,
        )
        if res is not None:
            all_results[cfg_name] = res

    if len(all_results) > 1:
        log.info("\n" + "=" * 60)
        log.info("Summary — Spearman rho(integration_score, rsa_joint):")
        for cname, res in all_results.items():
            rho   = res.get("rho_integration_vs_rsa_joint", float("nan"))
            ci_lo = res.get("ci_lo_integration_vs_rsa_joint", float("nan"))
            ci_hi = res.get("ci_hi_integration_vs_rsa_joint", float("nan"))
            log.info(f"  {cname:45s}  rho={rho:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")


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
    maps_a = collect_maps(subjects_dir, f"R2_{args.roi_a}_nc", args.min_subjects)
    maps_b = collect_maps(subjects_dir, f"R2_{args.roi_b}_nc", args.min_subjects)
    maps_product = collect_maps(subjects_dir, "product_map", args.min_subjects) 
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
        description="Phase 6: RSA overlap (group_average) or group stats (per_subject).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",          required=True, choices=["group_average", "per_subject"])
    p.add_argument("--roi-a",         dest="roi_a",          default="A1")
    p.add_argument("--roi-b",         dest="roi_b",          default="V1")
    p.add_argument("--output-base",   dest="output_base",    default=_OUT_BASE)
    p.add_argument("--template-cifti", dest="template_cifti", default=None,
                   help="59k cortex-only CIFTI template (required).")
    # group_average
    p.add_argument("--rsa-base",      dest="rsa_base",       default=_RSA_BASE,
                   help="Root directory of searchlight RSA outputs (group_average mode).")
    p.add_argument("--rsa-model",     dest="rsa_model",      default=RSA_MODEL,
                   help="RSA model directory name (group_average mode).")
    p.add_argument("--rsa-config",    dest="rsa_config",
                   default="notnormalized_nohrf_global_2s",
                   choices=list(RSA_CONFIGS.keys()),
                   help="RSA config to run (group_average mode).")
    p.add_argument("--all-configs",   dest="all_configs",    action="store_true",
                   help="Run all RSA configs instead of --rsa-config (group_average mode).")
    # per_subject
    p.add_argument("--min-subjects",  dest="min_subjects",   type=int, default=1,
                   help="Minimum subjects required (per_subject mode).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.template_cifti is None:
        raise ValueError(
            "--template_cifti is required. Pass the preprocessed group-average 59k "
            "CIFTI (group_average_{suffix}_cortex_59k.dtseries.nii).")

    log.info("=" * 60)
    log.info(f"Script 06 — Summary ({args.mode})")
    log.info(f"  ROIs: {args.roi_a} × {args.roi_b}")
    log.info("=" * 60)

    if args.mode == "group_average":
        run_group_average(args)
    else:
        run_per_subject(args)

    log.info("\nScript 06 complete.")


if __name__ == "__main__":
    main()
