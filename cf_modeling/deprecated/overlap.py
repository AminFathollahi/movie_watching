"""
cf_modeling/deprecated/overlap.py
=================================
RSA spatial overlap (group_average):
    Spatial correlation (Spearman + 95% CI bootstrap) between the CF-model
    integration_score and PE-AV searchlight RSA maps (audio/video/joint).
    Overlap map: min(integration_score_norm, rsa_joint_norm) per vertex.
    Results JSON saved for each RSA config.

Run
---
    python overlap.py --mode group_average --roi_a A1 --roi_b V1 \\
        --rsa_base /path/to/searchlight_rsa_output
"""

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

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
    stems = cf_model_map_stems(args.roi_a, args.roi_b)
    legacy_stems = legacy_cf_model_map_stems(args.roi_a, args.roi_b)
    product_map = np.load(resolve_cf_model_map_path(
        prep_dir, stems["joint_split_r2_geomean"],
        legacy_stems["joint_split_r2_geomean"]))
    R2_a_nc = np.load(resolve_cf_model_map_path(
        prep_dir, stems["null_corrected_split_r2_a"],
        legacy_stems["null_corrected_split_r2_a"]))
    R2_b_nc = np.load(resolve_cf_model_map_path(
        prep_dir, stems["null_corrected_split_r2_b"],
        legacy_stems["null_corrected_split_r2_b"]))
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
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RSA overlap of CF integration maps (group_average).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",          required=True, choices=["group_average"])
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
    return p.parse_args()


def main():
    args = parse_args()

    if args.template_cifti is None:
        raise ValueError(
            "--template_cifti is required. Pass the preprocessed group-average 59k "
            "CIFTI (group_average_{suffix}_cortex_59k.dtseries.nii).")

    log.info("=" * 60)
    log.info(f"RSA overlap ({args.mode})")
    log.info(f"  ROIs: {args.roi_a} × {args.roi_b}")
    log.info("=" * 60)

    run_group_average(args)

    log.info("\nRSA overlap complete.")


if __name__ == "__main__":
    main()
