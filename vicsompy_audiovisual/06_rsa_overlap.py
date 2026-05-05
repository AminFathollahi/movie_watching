"""
06_rsa_overlap.py
=================
Spatial overlap between the CF-model audiovisual integration zone and the
PE-AV searchlight RSA maps.

Scientific rationale:
    Script 05 identifies the audiovisual integration zone from the CF model:
    vertices where both A1 AND V1 topographic patterns predict brain activity.
    Here we ask: does the PE-AV joint embedding independently identify the same
    regions? These are two completely different methods applied to the same data.
    Convergence = convergent validity for a genuine audiovisual integration zone.

Three complementary analyses:
    1. Spatial correlation (Spearman) between the CF integration_score and the
       PE-AV RSA maps (audio, video, joint). Reports rho + 95% CI (bootstrap).

    2. Overlap map: min(integration_score_norm, RSA_joint_norm) per vertex.
       High only where BOTH methods agree. Saved as a CIFTI for Workbench.

    3. Results JSON: all correlation values for every active config, to compare
       across RSA variants (blockdiag vs global, hrf vs nohrf, bin sizes).

RSA Config (modular — swap ACTIVE_CONFIG to compare variants):
    Each config specifies (norm, hrf_tag, method, bin) which fully determines
    the .npy file paths under the searchlight RSA output directory.
    Add new entries to RSA_CONFIGS freely; only ACTIVE_CONFIG is processed.

    To run all configs in a loop:
        for name in RSA_CONFIGS:
            run_overlap_analysis(name, ...)

Inputs:
    OUTPUT_DIR/integration_score.npy    (59412,)  from Script 05
    OUTPUT_DIR/integration_mask.npy     (59412,)  from Script 05
    OUTPUT_DIR/R2_audio_nc.npy          (59412,)  from Script 04
    OUTPUT_DIR/R2_video_nc.npy          (59412,)  from Script 04
    RSA_BASE/.../rsa_*_{hem}_*.npy      per-hemisphere RSA maps

Outputs (per config):
    OUTPUT_DIR/overlap_score_{config}.{npy,dscalar.nii}
    OUTPUT_DIR/results/rsa_overlap_{config}.json

Run
---
    conda activate vicsompy_av
    python 06_rsa_overlap.py
    # To run all configs: set RUN_ALL_CONFIGS = True below
"""

# =============================================================================
# CONFIG
# =============================================================================

DATA_BASE  = "/home/amin/Research/Representation/Movie/data/Setareh"
TEMPLATE_CIFTI = (f"{DATA_BASE}/HCP_S1200_GroupAvg_v1/"
                  "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii")

OUTPUT_DIR = "/home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual"
CIFTI_DIR  = f"{OUTPUT_DIR}/cifti_maps"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

RSA_BASE   = "/home/amin/Research/Representation/Movie/outputs/searchlight_rsa_output"
RSA_MODEL  = "pe-av-small-16-frame"

# ── RSA variant selection ────────────────────────────────────────────────────
# Each entry: (norm, hrf_tag, method, bin)
#   norm     : 'normalized' | 'notnormalized'
#   hrf_tag  : 'hrf' | 'nohrf'   (nohrf = delay5s variant in directory name)
#   method   : 'blockdiag' | 'global'
#   bin      : '1s' | '2s' | '5s' | '10s'
#
# Directory path built as:
#   k100_{norm}_hrf_{method}   if hrf_tag='hrf'
#   k100_{norm}_{method}_delay5s  if hrf_tag='nohrf'
# Filename: rsa_{model}_{modality}_k100_spearman_{norm}_{hrf_tag}_bin{bin}_{hem}_{method}.npy
RSA_CONFIGS = {
    "normalized_hrf_blockdiag_2s"    : dict(norm="normalized",    hrf_tag="hrf",   method="blockdiag", bin="2s"),
    "normalized_hrf_global_2s"       : dict(norm="normalized",    hrf_tag="hrf",   method="global",    bin="2s"),
    "notnormalized_hrf_blockdiag_2s" : dict(norm="notnormalized", hrf_tag="hrf",   method="blockdiag", bin="2s"),
    "normalized_nohrf_blockdiag_2s"  : dict(norm="normalized",    hrf_tag="nohrf", method="blockdiag", bin="2s"),
    "normalized_hrf_blockdiag_5s"    : dict(norm="normalized",    hrf_tag="hrf",   method="blockdiag", bin="5s"),
    "notnormalized_hrf_blockdiag_5s" : dict(norm="notnormalized", hrf_tag="hrf",   method="blockdiag", bin="5s"),
}

ACTIVE_CONFIG   = "normalized_nohrf_blockdiag_2s"   # ← swap to compare variants
RUN_ALL_CONFIGS = False                            # ← set True to process all

N_BOOTSTRAP = 1000    # bootstrap iterations for 95% CI on Spearman rho
BOOTSTRAP_SEED = 42

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import logging

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
os.makedirs(CIFTI_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# RSA FILE LOADING
# =============================================================================

def _config_dir(norm, hrf_tag, method):
    """Build the RSA config directory name from parameters."""
    if hrf_tag == "hrf":
        return f"k100_{norm}_hrf_{method}"
    else:
        return f"k100_{norm}_{method}_delay5s"


def get_rsa_path(norm, hrf_tag, method, bin_size, modality, hem):
    """Build the full path to a per-hemisphere RSA .npy file."""
    config_dir = _config_dir(norm, hrf_tag, method)
    hem_dir    = f"{hem}_hemisphere"
    fname      = (f"rsa_{RSA_MODEL}_{modality}_k100_spearman_"
                  f"{norm}_{hrf_tag}_bin{bin_size}_{hem}_{method}.npy")
    return os.path.join(RSA_BASE, RSA_MODEL, config_dir, bin_size, modality, hem_dir, fname)


def get_grayordinate_indices():
    """Return (left_verts, right_verts) grayordinate index arrays from the template CIFTI.

    RSA .npy files cover all 32492 surface vertices per hemisphere.
    The CIFTI grayordinate space excludes medial wall vertices (29696 L + 29716 R = 59412).
    We use the template BrainModelAxis to select only the grayordinate vertices.
    """
    template = nib.load(TEMPLATE_CIFTI)
    bm = template.header.get_axis(1)
    left_verts = right_verts = None
    for name, _, model in bm.iter_structures():
        if "LEFT"  in name: left_verts  = model.vertex
        if "RIGHT" in name: right_verts = model.vertex
    return left_verts, right_verts   # (29696,), (29716,)


# Cache grayordinate indices (loaded once at module level to avoid repeated CIFTI reads)
_GRAY_LEFT, _GRAY_RIGHT = None, None

def load_rsa_fullbrain(norm, hrf_tag, method, bin_size, modality):
    """Load L+R hemisphere RSA maps, index to grayordinates, concatenate to (59412,).

    RSA files are (32492,) covering the full surface including medial wall.
    We index to the 29696 L + 29716 R grayordinate vertices from the template CIFTI,
    matching the ordering used when building Y_test in Script 02.
    """
    global _GRAY_LEFT, _GRAY_RIGHT
    if _GRAY_LEFT is None:
        _GRAY_LEFT, _GRAY_RIGHT = get_grayordinate_indices()

    left_full  = np.load(get_rsa_path(norm, hrf_tag, method, bin_size, modality, "left"))   # (32492,)
    right_full = np.load(get_rsa_path(norm, hrf_tag, method, bin_size, modality, "right"))  # (32492,)

    full = np.concatenate([left_full[_GRAY_LEFT], right_full[_GRAY_RIGHT]]).astype(np.float32)
    assert full.shape[0] == 59412, f"Expected 59412 vertices, got {full.shape[0]}"
    return full

# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def bootstrap_spearman(x, y, n_boot, seed):
    """Spearman rho with 95% bootstrap CI (resample vertices with replacement)."""
    rng = np.random.default_rng(seed)
    n   = len(x)
    rho_obs = spearmanr(x, y).statistic
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i] = spearmanr(x[idx], y[idx]).statistic
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    return float(rho_obs), float(ci_lo), float(ci_hi)


def normalise_positive(arr):
    """Scale positive part of arr to [0, 1] using the 95th percentile."""
    pos = np.clip(arr, 0, None)
    p95 = np.percentile(pos[pos > 0], 95) if (pos > 0).any() else 1.0
    return np.clip(pos / p95, 0, 1).astype(np.float32)


def save_map(arr, name, template):
    """Save (59412,) array as .npy and .dscalar.nii."""
    arr_f32 = arr.astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr_f32)
    cifti = nib.Cifti2Image(arr_f32.reshape(1, -1),
                            header=template.header,
                            nifti_header=template.nifti_header)
    nib.save(cifti, os.path.join(CIFTI_DIR, f"{name}.dscalar.nii"))

# =============================================================================
# CORE ANALYSIS — one config at a time
# =============================================================================

def run_overlap_analysis(config_name, integration_score, R2_audio_nc, R2_video_nc,
                         integration_mask, template):
    cfg = RSA_CONFIGS[config_name]
    norm, hrf_tag, method, bin_size = cfg["norm"], cfg["hrf_tag"], cfg["method"], cfg["bin"]

    log.info(f"\n{'='*60}")
    log.info(f"Config: {config_name}")
    log.info(f"  norm={norm}  hrf={hrf_tag}  method={method}  bin={bin_size}")

    # --- Load RSA maps ---
    log.info("  Loading RSA maps …")
    try:
        rsa_audio = load_rsa_fullbrain(norm, hrf_tag, method, bin_size, "audio")
        rsa_video = load_rsa_fullbrain(norm, hrf_tag, method, bin_size, "video")
        rsa_joint = load_rsa_fullbrain(norm, hrf_tag, method, bin_size, "joint")
    except FileNotFoundError as e:
        log.error(f"  SKIP — file not found: {e}")
        return None

    log.info(f"  RSA audio: mean={rsa_audio.mean():.4f}  "
             f"video: mean={rsa_video.mean():.4f}  "
             f"joint: mean={rsa_joint.mean():.4f}")

    # --- Spatial correlations (Spearman) over all cortical vertices ---
    # Compares the CF-model integration score to each RSA modality map.
    # We expect the joint RSA to correlate most with the integration score
    # because both methods are trying to find the same audiovisual zone.
    rng_seed = BOOTSTRAP_SEED
    results = {}

    for rsa_name, rsa_map in [("audio", rsa_audio), ("video", rsa_video), ("joint", rsa_joint)]:
        rho, ci_lo, ci_hi = bootstrap_spearman(integration_score, rsa_map,
                                               N_BOOTSTRAP, rng_seed)
        results[f"rho_integration_vs_rsa_{rsa_name}"] = rho
        results[f"ci_lo_integration_vs_rsa_{rsa_name}"] = ci_lo
        results[f"ci_hi_integration_vs_rsa_{rsa_name}"] = ci_hi
        log.info(f"  Spearman(integration_score, rsa_{rsa_name}): "
                 f"rho={rho:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rng_seed += 1

    # Modality-specific correlations: audio R² vs RSA_audio, video R² vs RSA_video
    for r2_name, r2_map, rsa_map in [
        ("audio", R2_audio_nc, rsa_audio),
        ("video", R2_video_nc, rsa_video),
    ]:
        rho, ci_lo, ci_hi = bootstrap_spearman(r2_map, rsa_map, N_BOOTSTRAP, rng_seed)
        results[f"rho_r2{r2_name}_vs_rsa_{r2_name}"] = rho
        results[f"ci_lo_r2{r2_name}_vs_rsa_{r2_name}"] = ci_lo
        results[f"ci_hi_r2{r2_name}_vs_rsa_{r2_name}"] = ci_hi
        log.info(f"  Spearman(R2_{r2_name}_nc, rsa_{r2_name}): "
                 f"rho={rho:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        rng_seed += 1

    results["config"]   = config_name
    results["n_verts"]  = 59412
    results["n_boot"]   = N_BOOTSTRAP

    # --- Overlap map: min(norm_integration, norm_rsa_joint) ---
    # High only where BOTH the CF integration score AND the PE-AV joint RSA are high.
    # This is the convergent-validity map: two independent methods pointing at the same vertex.
    int_norm = normalise_positive(integration_score)
    rsa_norm = normalise_positive(rsa_joint)
    overlap  = np.minimum(int_norm, rsa_norm)

    map_name = f"overlap_score_{config_name}"
    save_map(overlap, map_name, template)
    log.info(f"  Saved overlap map: {map_name}  mean={overlap.mean():.4f}  "
             f"frac>0.1={np.mean(overlap>0.1):.1%}")

    results["overlap_mean"]        = float(overlap.mean())
    results["overlap_frac_gt_0.1"] = float(np.mean(overlap > 0.1))

    # --- Save JSON results ---
    json_path = os.path.join(RESULTS_DIR, f"rsa_overlap_{config_name}.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"  Saved results: {json_path}")

    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Script 06 — RSA spatial overlap with PE-AV searchlight maps")
    log.info("=" * 60)

    # --- Load CF-model maps ---
    log.info("\nLoading CF-model integration maps …")
    integration_score = np.load(os.path.join(OUTPUT_DIR, "integration_score.npy"))
    integration_mask  = np.load(os.path.join(OUTPUT_DIR, "integration_mask.npy"))
    R2_audio_nc       = np.load(os.path.join(OUTPUT_DIR, "R2_audio_nc.npy"))
    R2_video_nc       = np.load(os.path.join(OUTPUT_DIR, "R2_video_nc.npy"))
    log.info(f"  integration_score: frac>0={np.mean(integration_score>0):.1%}  "
             f"integration_mask: n={int(integration_mask.sum())} verts")

    template = nib.load(TEMPLATE_CIFTI)

    configs_to_run = list(RSA_CONFIGS.keys()) if RUN_ALL_CONFIGS else [ACTIVE_CONFIG]
    log.info(f"\nRunning {len(configs_to_run)} config(s): {configs_to_run}")

    all_results = {}
    for config_name in configs_to_run:
        res = run_overlap_analysis(config_name, integration_score,
                                   R2_audio_nc, R2_video_nc,
                                   integration_mask, template)
        if res is not None:
            all_results[config_name] = res

    if len(all_results) > 1:
        # Summary table across configs
        log.info("\n" + "="*60)
        log.info("Summary — Spearman rho(integration_score, rsa_joint):")
        for cname, res in all_results.items():
            rho = res.get("rho_integration_vs_rsa_joint", float("nan"))
            ci_lo = res.get("ci_lo_integration_vs_rsa_joint", float("nan"))
            ci_hi = res.get("ci_hi_integration_vs_rsa_joint", float("nan"))
            log.info(f"  {cname:45s}  rho={rho:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    log.info("\nScript 06 complete.")


if __name__ == "__main__":
    main()
