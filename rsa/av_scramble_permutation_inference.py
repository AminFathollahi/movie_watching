"""Aggregate completed AV-scramble maps and save permutation inference maps."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import save_cifti_map, save_cifti_multimap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Empirical null for group-average AV searchlight RSA from real "
                     "AV-pairing re-inference permutations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, help="Root RSA output directory.")
    p.add_argument("--intact-model", required=True, dest="intact_model")
    p.add_argument("--scramble-base", required=True, dest="scramble_base",
                    help="Base scrambled model name, e.g. pe-av-small-16-frame_avscramble "
                         "(seed K != 42 reads {scramble_base}_seed{K}; seed 42 reads "
                         "{scramble_base} itself, matching pe_av_extract_scramble.py's naming).")
    p.add_argument("--total-permutations", type=int, default=500,
                   help="Target seed range is 1..N; all completed maps are used.")
    p.add_argument("--require-complete", action="store_true",
                   help="Fail unless every target map exists.")
    p.add_argument("--modality", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--method", required=True)
    p.add_argument("--fmri-tag", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--alpha", type=float, default=0.01,
                   help="Threshold for uncorrected p and BH-FDR q maps.")
    return p.parse_args()


def _rho_path(output_dir: Path, model: str, modality: str, config: str, fmri_tag: str,
              k: int, delay_sec: float, bin_sec_int: int, skip_int: int, method: str) -> Path:
    return (output_dir / "group_average" / f"{model}_{modality}" / config /
            f"rsa_59k_{fmri_tag}_k{k}_delay{int(delay_sec)}s_bin{bin_sec_int}s_"
            f"skip{skip_int}s_{method}_searchlight.npy")


def _signed_sigmap(p_values: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Project-standard signed -log10(p) significance map."""
    eps = np.finfo(np.float32).tiny
    return (np.sign(effect) *
            -np.log10(np.maximum(p_values, eps))).astype(np.float32)


def _fdr_maps(p_perm: np.ndarray, effect: np.ndarray, alpha: float):
    """BH-FDR adjusted p-values, signed sigmap, and binary threshold mask."""
    p_fdr = stats.false_discovery_control(p_perm, method="bh").astype(np.float32)
    sigmap_fdr = _signed_sigmap(p_fdr, effect)
    fdr_mask = (p_fdr < alpha).astype(np.float32)
    return p_fdr, sigmap_fdr, fdr_mask


def _max_stat_fwe_maps(
    null_stack: np.ndarray,
    intact_rho: np.ndarray,
    effect: np.ndarray,
    alpha: float,
):
    """Single-step max-statistic FWE correction across all grayordinates.

    The statistic is rho itself, matching the vertexwise permutation test.
    For each permutation, its maximum rho over the complete map forms the
    cortex-wide null distribution. This is intentionally more conservative
    than BH-FDR and controls the probability of any false-positive vertex.
    """
    null_max_rho = null_stack.max(axis=1)
    exceed_count = (null_max_rho[:, None] >= intact_rho[None, :]).sum(axis=0)
    p_maxT_fwe = ((exceed_count + 1) / (len(null_max_rho) + 1)).astype(np.float32)
    sigmap_maxT_fwe = _signed_sigmap(p_maxT_fwe, effect)
    maxT_fwe_mask = (p_maxT_fwe < alpha).astype(np.float32)
    critical_rho = float(np.quantile(null_max_rho, 1 - alpha, method="higher"))
    return p_maxT_fwe, sigmap_maxT_fwe, maxT_fwe_mask, null_max_rho, critical_rho


def main():
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError(f"--alpha must be between 0 and 1, got {args.alpha}")
    if args.total_permutations < 1:
        raise ValueError("--total-permutations must be positive")
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    bin_sec_int, skip_int = int(args.bin_sec), int(args.skip_sec)
    config = f"k{args.k}_delay{int(args.delay_sec)}s_bin{bin_sec_int}s_skip{skip_int}s_{args.method}"
    output_dir = Path(args.output_dir)

    def load(model):
        path = _rho_path(output_dir, model, args.modality, config, args.fmri_tag,
                          args.k, args.delay_sec, bin_sec_int, skip_int, args.method)
        return np.load(str(path)).astype(np.float32) if path.exists() else None

    intact_rho = load(args.intact_model)
    if intact_rho is None:
        log.error(f"Missing intact rho for {args.intact_model}; run rsa/searchlight.py first.")
        sys.exit(1)

    null_rhos, found_seeds, missing_seeds = [], [], []
    for seed in range(1, args.total_permutations + 1):
        model = args.scramble_base if seed == 42 else f"{args.scramble_base}_seed{seed}"
        rho = load(model)
        if rho is None:
            missing_seeds.append(seed)
            continue
        null_rhos.append(rho)
        found_seeds.append(seed)

    n_perm = len(null_rhos)
    if n_perm == 0:
        log.error("No scrambled-seed rho maps found. Nothing to build a null from.")
        sys.exit(1)
    if args.require_complete and missing_seeds:
        log.error("Missing %d of %d permutations: %s", len(missing_seeds),
                  args.total_permutations, missing_seeds)
        sys.exit(1)
    log.info("Using %d of %d target permutations; %d remain.", n_perm,
             args.total_permutations, len(missing_seeds))

    null_stack = np.stack(null_rhos, axis=0)  # (n_perm, n_verts)
    null_mean = null_stack.mean(axis=0).astype(np.float32)
    null_std = null_stack.std(axis=0).astype(np.float32)
    delta_rho = (intact_rho - null_mean).astype(np.float32)

    ge_count = (null_stack >= intact_rho[None, :]).sum(axis=0)
    p_perm = ((ge_count + 1) / (n_perm + 1)).astype(np.float32)
    sigmap_perm = _signed_sigmap(p_perm, delta_rho)
    p_perm_fdr, sigmap_perm_fdr, fdr_mask = _fdr_maps(
        p_perm, delta_rho, args.alpha)
    (p_perm_maxT_fwe, sigmap_perm_maxT_fwe, maxT_fwe_mask,
     null_max_rho, maxT_critical_rho) = _max_stat_fwe_maps(
        null_stack, intact_rho, delta_rho, args.alpha)
    p_uncorr_mask = (p_perm < args.alpha).astype(np.float32)
    delta_rho_uncorr = (delta_rho * p_uncorr_mask).astype(np.float32)
    delta_rho_fdr = (delta_rho * fdr_mask).astype(np.float32)
    delta_rho_maxT_fwe = (delta_rho * maxT_fwe_mask).astype(np.float32)

    n_sig_uncorr = int(p_uncorr_mask.sum())
    n_sig_fdr = int(fdr_mask.sum())
    n_sig_maxT_fwe = int(maxT_fwe_mask.sum())

    log.info("completed seeds: %s", found_seeds)
    log.info(f"intact rho: mean={intact_rho.mean():.4f} max={intact_rho.max():.4f}")
    log.info(f"null rho:   mean={null_mean.mean():.4f} max={null_mean.max():.4f}")
    log.info(f"delta rho:  mean={delta_rho.mean():.4f} max={delta_rho.max():.4f}")
    log.info(f"uncorrected p<{args.alpha}: {n_sig_uncorr:,} / {len(p_perm):,} verts")
    log.info(f"BH-FDR q<{args.alpha} (secondary): {n_sig_fdr:,} / {len(p_perm):,} verts")
    log.info(f"maxT-FWE p<{args.alpha} (primary): {n_sig_maxT_fwe:,} / {len(p_perm):,} verts "
             f"(critical rho > {maxT_critical_rho:.6f})")

    out_dir = output_dir / "group_average" / "_av_scramble_permutation" / \
        f"{args.intact_model}_{args.modality}" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = f"{args.alpha:g}".replace(".", "p")
    run_tag = f"n{n_perm}_of{args.total_permutations}_alpha{alpha_tag}"
    out_path = out_dir / f"av_scramble_{run_tag}_all_maps.dscalar.nii"
    save_cifti_multimap(
        np.stack([
            intact_rho, null_mean, null_std, delta_rho,
            p_perm, sigmap_perm, p_perm_fdr, sigmap_perm_fdr,
            p_perm_maxT_fwe, sigmap_perm_maxT_fwe,
            p_uncorr_mask, fdr_mask, maxT_fwe_mask,
            delta_rho_uncorr, delta_rho_fdr, delta_rho_maxT_fwe,
        ], axis=0),
        [
            "intact_rho", "null_mean_rho", "null_std_rho", "delta_rho",
            "p_perm", "sigmap_perm", "p_perm_fdr", "sigmap_perm_fdr",
            "p_perm_maxT_fwe", "sigmap_perm_maxT_fwe",
            "p_uncorr_mask", "fdr_mask", "maxT_fwe_mask",
            "delta_rho_p_uncorr", "delta_rho_fdr", "delta_rho_maxT_fwe",
        ],
        args.template_cifti, str(out_path),
    )
    log.info(f"Saved: {out_path}")

    sigmaps_path = out_dir / f"av_scramble_{run_tag}_significance_maps.dscalar.nii"
    save_cifti_multimap(
        np.stack([
            sigmap_perm_maxT_fwe, delta_rho_maxT_fwe,
            sigmap_perm, sigmap_perm_fdr, delta_rho_uncorr, delta_rho_fdr,
        ], axis=0),
        [
            "sigmap_perm_maxT_fwe", "delta_rho_maxT_fwe",
            "sigmap_perm", "sigmap_perm_fdr", "delta_rho_p_uncorr",
            "delta_rho_fdr",
        ],
        args.template_cifti, str(sigmaps_path),
    )
    log.info(f"Saved significance maps: {sigmaps_path}")

    p_uncorr_mask_path = out_dir / f"av_scramble_{run_tag}_uncorrected_mask.dscalar.nii"
    save_cifti_map(p_uncorr_mask, args.template_cifti, str(p_uncorr_mask_path),
                   "p_uncorr_mask")
    log.info(f"Saved uncorrected mask: {p_uncorr_mask_path}")

    fdr_mask_path = out_dir / f"av_scramble_{run_tag}_fdr_mask.dscalar.nii"
    save_cifti_map(fdr_mask, args.template_cifti, str(fdr_mask_path), "fdr_mask")
    log.info(f"Saved FDR mask: {fdr_mask_path}")

    maxT_fwe_mask_path = out_dir / f"av_scramble_{run_tag}_maxT_fwe_mask.dscalar.nii"
    save_cifti_map(maxT_fwe_mask, args.template_cifti, str(maxT_fwe_mask_path),
                   "maxT_fwe_mask")
    log.info(f"Saved primary maxT-FWE mask: {maxT_fwe_mask_path}")

    np.save(out_dir / f"av_scramble_{run_tag}_null_stack.npy", null_stack)
    np.save(out_dir / f"av_scramble_{run_tag}_p_uncorrected.npy", p_perm)
    np.save(out_dir / f"av_scramble_{run_tag}_p_fdr.npy", p_perm_fdr)
    np.save(out_dir / f"av_scramble_{run_tag}_p_maxT_fwe.npy", p_perm_maxT_fwe)
    np.save(out_dir / f"av_scramble_{run_tag}_null_max_rho.npy", null_max_rho)

    summary = {
        "intact_model": args.intact_model, "scramble_base": args.scramble_base,
        "total_permutations": args.total_permutations,
        "completed_seeds": found_seeds, "missing_seeds": missing_seeds,
        "n_completed": n_perm, "n_missing": len(missing_seeds),
        "modality": args.modality, "config": config,
        "alpha": args.alpha,
        "min_reachable_p": round(1 / (n_perm + 1), 4),
        "complete": not missing_seeds,
        "intact_rho_mean": float(intact_rho.mean()), "intact_rho_max": float(intact_rho.max()),
        "null_rho_mean": float(null_mean.mean()), "null_rho_max": float(null_mean.max()),
        "delta_rho_mean": float(delta_rho.mean()), "delta_rho_max": float(delta_rho.max()),
        "n_verts_p_uncorr": n_sig_uncorr, "n_verts_fdr": n_sig_fdr,
        "n_verts_maxT_fwe": n_sig_maxT_fwe,
        "min_p_fdr": float(p_perm_fdr.min()),
        "maxT_critical_rho": maxT_critical_rho,
        "primary_inference": "maxT-FWE",
        "null_cifti": str(out_path),
        "significance_cifti": str(sigmaps_path),
        "p_uncorr_mask_cifti": str(p_uncorr_mask_path),
        "fdr_mask_cifti": str(fdr_mask_path),
        "maxT_fwe_mask_cifti": str(maxT_fwe_mask_path),
    }
    summary_path = out_dir / f"av_scramble_{run_tag}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
