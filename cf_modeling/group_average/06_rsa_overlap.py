"""
06_rsa_overlap.py
=================
Spatial overlap between the CF-model audiovisual integration zone and the
PE-AV searchlight RSA maps.

Scientific rationale:
    Script 05 identifies the bimodal integration zone from the CF model:
    vertices where both ROI_A AND ROI_B topographic patterns predict activity.
    Here we ask: does the PE-AV joint embedding independently identify the same
    regions? Convergence = convergent validity for a genuine integration zone.

Three complementary analyses:
    1. Spatial correlation (Spearman) between the CF integration_score and the
       PE-AV RSA maps (audio, video, joint). Reports rho + 95% CI (bootstrap).

    2. Overlap map: min(integration_score_norm, RSA_joint_norm) per vertex.
       High only where BOTH methods agree. Saved as a CIFTI for Workbench.

    3. Results JSON: all correlation values for every active config, to compare
       across RSA variants (blockdiag vs global, hrf vs nohrf, bin sizes).

Run
---
    conda activate vicsompy_av
    python 06_rsa_overlap.py \\
        --roi_a A1 --roi_b V1 \\
        --output_base /path/to/outputs \\
        --rsa_base /path/to/searchlight_rsa_output \\
        --template_cifti /path/to/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os

_DEFAULT_DATA_BASE   = "/home/amin/Research/Representation/Movie/data/Setareh"
_DEFAULT_ROI_A       = "A1"
_DEFAULT_ROI_B       = "V1"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/group_average"
_DEFAULT_TEMPLATE    = (
    f"{_DEFAULT_DATA_BASE}/HCP_S1200_GroupAvg_v1/"
    "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
)
_DEFAULT_RSA_BASE = "/home/amin/Research/Representation/Movie/outputs/searchlight_rsa_output"

ROI_A          = _DEFAULT_ROI_A
ROI_B          = _DEFAULT_ROI_B
OUTPUT_BASE    = _DEFAULT_OUTPUT_BASE
TEMPLATE_CIFTI = _DEFAULT_TEMPLATE
RSA_BASE       = _DEFAULT_RSA_BASE

OUTPUT_DIR  = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
CIFTI_DIR   = f"{OUTPUT_DIR}/cifti_maps"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

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

ACTIVE_CONFIG   = "notnormalized_nohrf_global_2s"
RUN_ALL_CONFIGS = False

N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 42

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import json
import logging

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# RSA FILE LOADING
# =============================================================================

def _config_dir(norm, hrf_tag, method):
    if hrf_tag == "hrf":
        return f"k100_{norm}_hrf_{method}"
    return f"k100_{norm}_{method}_delay5s"


def get_rsa_path(norm, hrf_tag, method, bin_size, modality, hem):
    config_dir = _config_dir(norm, hrf_tag, method)
    hem_dir    = f"{hem}_hemisphere"
    fname      = (f"rsa_{RSA_MODEL}_{modality}_k100_spearman_"
                  f"{norm}_{hrf_tag}_bin{bin_size}_{hem}_{method}.npy")
    return os.path.join(RSA_BASE, RSA_MODEL, config_dir, bin_size, modality, hem_dir, fname)


_GRAY_LEFT, _GRAY_RIGHT = None, None


def get_grayordinate_indices():
    template = nib.load(TEMPLATE_CIFTI)
    bm = template.header.get_axis(1)
    lv = rv = None
    for name, _, model in bm.iter_structures():
        if "LEFT"  in name: lv = model.vertex
        if "RIGHT" in name: rv = model.vertex
    return lv, rv


def load_rsa_fullbrain(norm, hrf_tag, method, bin_size, modality):
    global _GRAY_LEFT, _GRAY_RIGHT
    if _GRAY_LEFT is None:
        _GRAY_LEFT, _GRAY_RIGHT = get_grayordinate_indices()

    left_full  = np.load(get_rsa_path(norm, hrf_tag, method, bin_size, modality, "left"))
    right_full = np.load(get_rsa_path(norm, hrf_tag, method, bin_size, modality, "right"))

    full = np.concatenate([left_full[_GRAY_LEFT], right_full[_GRAY_RIGHT]]).astype(np.float32)
    assert full.shape[0] == 59412, f"Expected 59412 vertices, got {full.shape[0]}"
    return full


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def bootstrap_spearman(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n   = len(x)
    rho_obs   = spearmanr(x, y).statistic
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i] = spearmanr(x[idx], y[idx]).statistic
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    return float(rho_obs), float(ci_lo), float(ci_hi)


def normalise_positive(arr):
    pos = np.clip(arr, 0, None)
    p95 = np.percentile(pos[pos > 0], 95) if (pos > 0).any() else 1.0
    return np.clip(pos / p95, 0, 1).astype(np.float32)


def save_map(arr, name, template):
    arr_f32 = arr.astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr_f32)
    cifti = nib.Cifti2Image(arr_f32.reshape(1, -1),
                            header=template.header,
                            nifti_header=template.nifti_header)
    nib.save(cifti, os.path.join(CIFTI_DIR, f"{name}.dscalar.nii"))


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def run_overlap_analysis(config_name, integration_score, R2_a_nc, R2_b_nc,
                         integration_mask, template):
    cfg = RSA_CONFIGS[config_name]
    norm, hrf_tag, method, bin_size = cfg["norm"], cfg["hrf_tag"], cfg["method"], cfg["bin"]

    log.info(f"\n{'='*60}")
    log.info(f"Config: {config_name}")
    log.info(f"  norm={norm}  hrf={hrf_tag}  method={method}  bin={bin_size}")

    log.info("  Loading RSA maps …")
    try:
        rsa_audio = load_rsa_fullbrain(norm, hrf_tag, method, bin_size, "audio")
        rsa_video = load_rsa_fullbrain(norm, hrf_tag, method, bin_size, "video")
        rsa_joint = load_rsa_fullbrain(norm, hrf_tag, method, bin_size, "joint")
    except FileNotFoundError as e:
        log.error(f"  SKIP — file not found: {e}")
        return None

    log.info(f"  RSA audio={rsa_audio.mean():.4f}  video={rsa_video.mean():.4f}  "
             f"joint={rsa_joint.mean():.4f}")

    rng_seed = BOOTSTRAP_SEED
    results  = {}

    for rsa_name, rsa_map in [("audio", rsa_audio), ("video", rsa_video), ("joint", rsa_joint)]:
        rho, ci_lo, ci_hi = bootstrap_spearman(
            integration_score, rsa_map, N_BOOTSTRAP, rng_seed)
        results[f"rho_integration_vs_rsa_{rsa_name}"] = rho
        results[f"ci_lo_integration_vs_rsa_{rsa_name}"] = ci_lo
        results[f"ci_hi_integration_vs_rsa_{rsa_name}"] = ci_hi
        log.info(f"  Spearman(integration_score, rsa_{rsa_name}): "
                 f"rho={rho:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rng_seed += 1

    for r2_name, r2_map, rsa_map in [
        (ROI_A.lower(), R2_a_nc, rsa_audio),
        (ROI_B.lower(), R2_b_nc, rsa_video),
    ]:
        rho, ci_lo, ci_hi = bootstrap_spearman(r2_map, rsa_map, N_BOOTSTRAP, rng_seed)
        results[f"rho_r2{r2_name}_vs_rsa_{r2_name}"] = rho
        results[f"ci_lo_r2{r2_name}_vs_rsa_{r2_name}"] = ci_lo
        results[f"ci_hi_r2{r2_name}_vs_rsa_{r2_name}"] = ci_hi
        log.info(f"  Spearman(R2_{r2_name}_nc, rsa_{r2_name}): "
                 f"rho={rho:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rng_seed += 1

    results.update({"config": config_name, "n_verts": 59412, "n_boot": N_BOOTSTRAP})

    int_norm = normalise_positive(integration_score)
    rsa_norm = normalise_positive(rsa_joint)
    overlap  = np.minimum(int_norm, rsa_norm)

    map_name = f"overlap_score_{config_name}"
    save_map(overlap, map_name, template)
    log.info(f"  Saved {map_name}: mean={overlap.mean():.4f}  frac>0.1={np.mean(overlap>0.1):.1%}")

    results["overlap_mean"]        = float(overlap.mean())
    results["overlap_frac_gt_0.1"] = float(np.mean(overlap > 0.1))

    json_path = os.path.join(RESULTS_DIR, f"rsa_overlap_{config_name}.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"  Saved results: {json_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(CIFTI_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Script 06 — RSA spatial overlap with PE-AV searchlight maps")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info("=" * 60)

    log.info("\nLoading CF-model integration maps …")
    integration_score = np.load(os.path.join(OUTPUT_DIR, "integration_score.npy"))
    integration_mask  = np.load(os.path.join(OUTPUT_DIR,
                                             f"top_{90}_percentile_integration.npy"))
    R2_a_nc           = np.load(os.path.join(OUTPUT_DIR, f"R2_{ROI_A}_nc.npy"))
    R2_b_nc           = np.load(os.path.join(OUTPUT_DIR, f"R2_{ROI_B}_nc.npy"))
    log.info(f"  integration_score: frac>0={np.mean(integration_score>0):.1%}  "
             f"mask: n={int(integration_mask.sum())} verts")

    template = nib.load(TEMPLATE_CIFTI)

    configs_to_run = list(RSA_CONFIGS.keys()) if RUN_ALL_CONFIGS else [ACTIVE_CONFIG]
    log.info(f"\nRunning {len(configs_to_run)} config(s): {configs_to_run}")

    all_results = {}
    for config_name in configs_to_run:
        res = run_overlap_analysis(config_name, integration_score,
                                   R2_a_nc, R2_b_nc, integration_mask, template)
        if res is not None:
            all_results[config_name] = res

    if len(all_results) > 1:
        log.info("\n" + "="*60)
        log.info("Summary — Spearman rho(integration_score, rsa_joint):")
        for cname, res in all_results.items():
            rho   = res.get("rho_integration_vs_rsa_joint", float("nan"))
            ci_lo = res.get("ci_lo_integration_vs_rsa_joint", float("nan"))
            ci_hi = res.get("ci_hi_integration_vs_rsa_joint", float("nan"))
            log.info(f"  {cname:45s}  rho={rho:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    log.info("\nScript 06 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RSA spatial overlap with PE-AV searchlight maps."
    )
    parser.add_argument("--roi_a",          default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",          default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base",    default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--template_cifti", default=_DEFAULT_TEMPLATE,
                        help="32k template dscalar.nii (default: %(default)s)")
    parser.add_argument("--rsa_base",       default=_DEFAULT_RSA_BASE,
                        help="Root directory of searchlight RSA outputs (default: %(default)s)")
    args = parser.parse_args()

    ROI_A          = args.roi_a
    ROI_B          = args.roi_b
    OUTPUT_BASE    = args.output_base
    TEMPLATE_CIFTI = args.template_cifti
    RSA_BASE       = args.rsa_base
    OUTPUT_DIR     = f"{OUTPUT_BASE}/{ROI_A}_{ROI_B}"
    CIFTI_DIR      = f"{OUTPUT_DIR}/cifti_maps"
    RESULTS_DIR    = f"{OUTPUT_DIR}/results"

    main()
