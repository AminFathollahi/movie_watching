"""
rsa/scramble_paired_stats.py
================================
Per-subject scramble-vs-intact PAIRED significance test.

Upgrades the existing scramble-vs-intact comparison
(rsa/scramble_diff_maps.py, group-average only, single number per vertex, no
significance test) to a proper subject-level inferential test, by extending
rsa/group_stats.py's one-sample-t-test/FDR framework to a paired design: for
each subject and vertex, diff = rho_intact - rho_scrambled, then a one-sample
t-test on diff (H0: mean diff = 0) -- equivalent to a paired t-test between
the two per-subject rho maps, since same-subject/same-vertex pairing cancels
subject-level baseline differences.

PREREQUISITE: per-subject scrambled RSA must exist under
{output-dir}/subject_data/{sub}/{scrambled-model}_{modality}/..., i.e.
rsa/searchlight.py run per-subject with the scrambled embedding, the same way
the existing per-subject intact runs were produced. (Historically this had
not been run; it has since been run per-subject for
pe-av-small-16-frame_avscramble/av.)

FDR convention and the optional corrected 2-factor bootstrap (Schutt et al.
2023 Eq. 5, requires --n-blocks block-level .npy files for BOTH intact and
scrambled) exactly mirror rsa/group_stats.py, just computed on the per-subject
DIFF stack instead of the raw rho stack.

Optional sign-flip permutation test (--n-perm > 0): a nonparametric
alternative/complement to the parametric paired t-test above. Under H0 (no
true AV-binding effect), each subject's diff = rho_intact - rho_scrambled is
exchangeable in sign. For n_perm draws, flip each subject's diff sign with
p=0.5, recompute the group mean, and use the resulting null distribution to
derive an empirical per-vertex p-value -- no new embeddings or RSA runs
needed, just a reshuffle of data already on disk.

Output: ONE combined CIFTI with mean_rho_intact / mean_rho_scrambled /
mean_diff / t_stat / sigmap_uncorr / sigmap_fdr (+ mean_diff_c2f / t_c2f /
sigmap_c2f if block files are found; + sigmap_perm / sigmap_perm_fdr if
--n-perm > 0), plus standalone fdr_mask / fdr_c2f_mask / fdr_perm_mask
dscalar files -- matching group_stats.py's existing map-naming and
file-layout conventions.

Run with:
    conda run -n movie python rsa/scramble_paired_stats.py \\
        --output-dir /home/amin/Research/Representation/Movie/outputs/rsa/raw \\
        --intact-model pe-av-small-16-frame \\
        --scrambled-model pe-av-small-16-frame_avscramble \\
        --modality av --k 100 --bin-sec 5.0 --delay-sec 5.0 --method spearman \\
        --fmri-tag raw \\
        --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \\
        --n-perm 100
"""

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
from rsa.shared.naming import add_model_norm_arg, method_label  # noqa: E402
from rsa.shared.rsa_utils import aggregate_blocks, corrected_2factor_bootstrap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-subject scramble-vs-intact paired RSA significance test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, help="Root RSA output directory.")
    p.add_argument("--intact-model", required=True, dest="intact_model")
    p.add_argument("--scrambled-model", required=True, dest="scrambled_model")
    p.add_argument("--modality", required=True, choices=["v", "a", "av", "at", "vt", "avt", "t"])
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--method", required=True, choices=["spearman", "pearson", "rho_a"])
    add_model_norm_arg(p)
    p.add_argument("--fmri-tag", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-blocks", type=int, default=4, dest="n_blocks")
    p.add_argument("--n-bootstrap", type=int, default=2000, dest="n_bootstrap")
    p.add_argument("--n-perm", type=int, default=0, dest="n_perm",
                   help="Sign-flip permutations for a nonparametric null on the paired "
                        "diff (0 = skip; e.g. 100).")
    p.add_argument("--perm-seed", type=int, default=42, dest="perm_seed")
    return p.parse_args()


def _compute_fdr_maps(p_uncorr: np.ndarray, mean_val: np.ndarray, alpha: float):
    """Matches rsa/group_stats.py's _compute_fdr_maps exactly (BH-FDR, signed -log10(q))."""
    p_fdr = stats.false_discovery_control(p_uncorr, method="bh")
    eps = np.finfo(np.float32).tiny
    sigmap_fdr = (np.sign(mean_val) * (-np.log10(np.maximum(p_fdr, eps)))).astype(np.float32)
    fdr_mask = (p_fdr < alpha).astype(np.float32)
    return sigmap_fdr, fdr_mask, int(fdr_mask.sum())


def _one_sample_test(x: np.ndarray, alpha: float):
    """x: (n_subs, n_verts). Returns mean, t, sigmap_uncorr, sigmap_fdr, fdr_mask, n_sig_fdr."""
    mean_val = x.mean(axis=0).astype(np.float32)
    t_vals, p_two = stats.ttest_1samp(x.astype(np.float64), popmean=0.0, axis=0)
    t_vals = t_vals.astype(np.float32)
    p_uncorr = np.where(t_vals > 0, p_two / 2.0, 1.0 - p_two / 2.0).astype(np.float32)
    eps = np.finfo(np.float32).tiny
    sigmap_uncorr = (np.sign(mean_val) * (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)
    sigmap_fdr, fdr_mask, n_sig_fdr = _compute_fdr_maps(p_uncorr, mean_val, alpha)
    return mean_val, t_vals, p_uncorr, sigmap_uncorr, sigmap_fdr, fdr_mask, n_sig_fdr


def _sign_flip_perm_test(diff_stack: np.ndarray, mean_diff: np.ndarray,
                          n_perm: int, seed: int, alpha: float):
    """Sign-flip permutation null for the paired diff (n_subs, n_verts).

    H0: each subject's diff sign is exchangeable (no true AV-binding effect).
    null_mean[p, v] = mean_s( sign[p, s] * diff_stack[s, v] ), sign in {-1, +1}.
    One-tailed empirical p in the direction of the observed mean_diff's sign,
    matching _one_sample_test's p_uncorr convention.
    """
    rng = np.random.default_rng(seed)
    n_subs = diff_stack.shape[0]
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_perm, n_subs))
    null_mean = (signs @ diff_stack.astype(np.float32)) / n_subs  # (n_perm, n_verts)

    ge_pos = (null_mean >= mean_diff[None, :]).sum(axis=0)
    ge_neg = (null_mean <= mean_diff[None, :]).sum(axis=0)
    p_perm = np.where(mean_diff > 0, ge_pos, ge_neg).astype(np.float64)
    p_perm = (p_perm + 1) / (n_perm + 1)

    eps = np.finfo(np.float32).tiny
    sigmap_perm = (np.sign(mean_diff) * (-np.log10(np.maximum(p_perm, eps)))).astype(np.float32)
    sigmap_perm_fdr, fdr_perm_mask, n_sig_perm_fdr = _compute_fdr_maps(
        p_perm.astype(np.float32), mean_diff, alpha)
    return p_perm.astype(np.float32), sigmap_perm, sigmap_perm_fdr, fdr_perm_mask, n_sig_perm_fdr


def _load_subject_stack(output_dir: Path, model: str, modality: str, config: str,
                         fname_pattern: str) -> tuple[np.ndarray, list[Path]]:
    model_mod_dir = f"{model}_{modality}"
    files = sorted(output_dir.glob(f"subject_data/*/{model_mod_dir}/{config}/{fname_pattern}"))
    if not files:
        return None, files
    stack = np.stack([np.load(str(f)).astype(np.float32) for f in files], axis=0)
    return stack, files


def _load_block_stack(output_dir: Path, model: str, modality: str, config: str,
                       fname_pattern: str, n_blocks: int) -> np.ndarray | None:
    model_mod_dir = f"{model}_{modality}"
    pat = fname_pattern.replace("_searchlight.npy", f"_searchlight_nblocks{n_blocks}.npy")
    files = sorted(output_dir.glob(f"subject_data/*/{model_mod_dir}/{config}/{pat}"))
    if not files:
        return None
    return np.stack([np.load(str(f)).astype(np.float32) for f in files], axis=0)


def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    bin_sec_int, skip_int = int(args.bin_sec), int(args.skip_sec)
    delay_tag = f"delay{int(args.delay_sec)}s"
    method = method_label(args.method, args.model_norm)
    config = f"k{args.k}_{delay_tag}_bin{bin_sec_int}s_skip{skip_int}s_{method}"

    output_dir = Path(args.output_dir)
    if args.method == "rho_a":
        fname_pattern = f"crossnobis_rho_a_k{args.k}_{delay_tag}_bin{bin_sec_int}s_skip{skip_int}s.npy"
    else:
        fname_pattern = (f"rsa_59k_{args.fmri_tag}_k{args.k}_{delay_tag}"
                          f"_bin{bin_sec_int}s_skip{skip_int}s_{method}_searchlight.npy")

    intact_stack, intact_files = _load_subject_stack(
        output_dir, args.intact_model, args.modality, config, fname_pattern)
    scrambled_stack, scrambled_files = _load_subject_stack(
        output_dir, args.scrambled_model, args.modality, config, fname_pattern)

    if intact_stack is None or scrambled_stack is None:
        log.error(
            f"Missing per-subject maps: intact={len(intact_files)} found, "
            f"scrambled={len(scrambled_files)} found. Run rsa/searchlight.py per-subject "
            f"with --model {args.scrambled_model} first (see module docstring)."
        )
        sys.exit(1)

    intact_subs = [f.parent.parent.parent.name for f in intact_files]
    scrambled_subs = [f.parent.parent.parent.name for f in scrambled_files]
    if intact_subs != scrambled_subs:
        common = sorted(set(intact_subs) & set(scrambled_subs))
        log.warning(f"Subject sets differ (intact={len(intact_subs)}, scrambled={len(scrambled_subs)}); "
                    f"restricting to {len(common)} common subjects.")
        intact_idx = [intact_subs.index(s) for s in common]
        scrambled_idx = [scrambled_subs.index(s) for s in common]
        intact_stack = intact_stack[intact_idx]
        scrambled_stack = scrambled_stack[scrambled_idx]
        subs = common
    else:
        subs = intact_subs

    n_subs, n_grays = intact_stack.shape
    log.info(f"Paired subjects: {n_subs}  grayordinates: {n_grays}")

    diff_stack = intact_stack - scrambled_stack

    mean_rho_intact = intact_stack.mean(axis=0).astype(np.float32)
    mean_rho_scrambled = scrambled_stack.mean(axis=0).astype(np.float32)

    log.warning("Paired one-sample test on diff=rho_intact-rho_scrambled (H0: mean diff = 0). "
                "Tests whether real temporal AV pairing explains fMRI better than scrambled "
                "pairing, per subject -- not merely a group-average descriptive difference.")
    (mean_diff, t_vals, p_uncorr, sigmap_uncorr,
     sigmap_fdr, fdr_mask, n_sig_fdr) = _one_sample_test(diff_stack, args.alpha)

    log.info(f"mean_diff range: [{mean_diff.min():.4f}, {mean_diff.max():.4f}]")
    log.info(f"Uncorrected p<{args.alpha}: {(p_uncorr < args.alpha).sum():,} / {n_grays:,}")
    log.info(f"BH-FDR p<{args.alpha}: {n_sig_fdr:,} / {n_grays:,}")

    # ── Optional corrected 2-factor bootstrap on the DIFF, if block files exist for both ──
    have_blocks = False
    n_sig_c2f = 0
    fdr_c2f_mask = None
    if args.method == "rho_a":
        # fname_pattern for rho_a (crossnobis_rho_a_..._bin{..}s_skip{..}s.npy) contains no
        # "_searchlight.npy", so _load_block_stack's replace() would be a silent no-op and
        # never match a file. No crossnobis_rho_a_*_nblocks*.npy files are produced by any
        # script today, so there is nothing to load -- skip explicitly instead of relying
        # on that broken match.
        log.info("2-factor bootstrap not supported for method=rho_a; skipping")
        intact_block = scrambled_block = None
    else:
        intact_block = _load_block_stack(output_dir, args.intact_model, args.modality, config,
                                          fname_pattern, args.n_blocks)
        scrambled_block = _load_block_stack(output_dir, args.scrambled_model, args.modality, config,
                                             fname_pattern, args.n_blocks)
    if intact_block is not None and scrambled_block is not None \
            and intact_block.shape[0] == n_subs and scrambled_block.shape[0] == n_subs:
        # Difference BEFORE bootstrapping, deliberately: intact/scrambled share the
        # same subject's same-block brain data (only the model-side AV pairing
        # differs), so subject- and block-level noise must cancel in the diff.
        # Passing this single pre-differenced array to corrected_2factor_bootstrap
        # makes every subject/block draw apply identically to both conditions
        # (joint resampling). Bootstrapping intact and scrambled separately and
        # summing their variances would double-count that shared noise instead of
        # cancelling it, inflating the null and silently destroying power -- see
        # tests/test_paired_bootstrap_resampling.py for a synthetic demonstration.
        diff_block = intact_block - scrambled_block
        have_blocks = True
        log.info(f"Block files found ({args.n_blocks} blocks) -- running corrected "
                 f"2-factor bootstrap on the diff (Schutt et al. 2023, Eq. 5) ...")
        var_c2f, _, _ = corrected_2factor_bootstrap(diff_stack, diff_block, n_boot=args.n_bootstrap)
        mean_diff_c2f = diff_stack.mean(axis=0).astype(np.float32)
        se_c2f = np.sqrt(np.maximum(var_c2f, 0.0)).astype(np.float64)
        df_c2f = max(min(n_subs - 1, args.n_blocks - 1), 1)
        safe_se = np.where(se_c2f > 0, se_c2f, 1.0)
        t_c2f = np.where(se_c2f > 0, mean_diff_c2f.astype(np.float64) / safe_se, 0.0).astype(np.float32)
        p_two_c2f = stats.t.sf(np.abs(t_c2f.astype(np.float64)), df=df_c2f) * 2.0
        p_c2f = np.where(t_c2f > 0, p_two_c2f / 2.0, 1.0 - p_two_c2f / 2.0).astype(np.float32)
        sigmap_c2f, fdr_c2f_mask, n_sig_c2f = _compute_fdr_maps(p_c2f, mean_diff_c2f, args.alpha)
        log.info(f"Corrected 2-factor bootstrap: df={df_c2f}, FDR sig verts = {n_sig_c2f:,} / {n_grays:,}")
    else:
        log.info(f"No usable block files for n_blocks={args.n_blocks} for both intact and "
                 f"scrambled -- skipping corrected 2-factor bootstrap.")

    # ── Optional sign-flip permutation null on the diff ──────────────────────
    have_perm = args.n_perm > 0
    n_sig_perm_fdr = 0
    fdr_perm_mask = None
    if have_perm:
        log.info(f"Running {args.n_perm} sign-flip permutations (seed={args.perm_seed}) ...")
        p_perm, sigmap_perm, sigmap_perm_fdr, fdr_perm_mask, n_sig_perm_fdr = _sign_flip_perm_test(
            diff_stack, mean_diff, args.n_perm, args.perm_seed, args.alpha)
        log.info(f"Sign-flip perm: uncorrected p<{args.alpha}: {(p_perm < args.alpha).sum():,} / {n_grays:,}  "
                  f"BH-FDR sig: {n_sig_perm_fdr:,} / {n_grays:,}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = output_dir / "groupstats" / f"{args.intact_model}_vs_{args.scrambled_model}_{args.modality}" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_nblocks{args.n_blocks}" if have_blocks else ""
    suffix += f"_perm{args.n_perm}" if have_perm else ""
    out_path = out_dir / f"scramble_paired_stats_{n_subs}subs{suffix}.dscalar.nii"

    maps = [mean_rho_intact, mean_rho_scrambled, mean_diff, t_vals, sigmap_uncorr, sigmap_fdr]
    map_names = ["mean_rho_intact", "mean_rho_scrambled", "mean_diff", "t_stat", "sigmap_uncorr", "sigmap_fdr"]
    if have_blocks:
        maps += [mean_diff_c2f, t_c2f, sigmap_c2f]
        map_names += ["mean_diff_c2f", "t_c2f", "sigmap_c2f"]
    if have_perm:
        maps += [sigmap_perm, sigmap_perm_fdr]
        map_names += ["sigmap_perm", "sigmap_perm_fdr"]

    save_cifti_multimap(np.stack(maps, axis=0), map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")

    fdr_mask_path = out_dir / f"scramble_paired_stats_{n_subs}subs{suffix}_fdr_mask.dscalar.nii"
    save_cifti_map(fdr_mask, args.template_cifti, str(fdr_mask_path), "fdr_mask")
    log.info(f"Saved FDR mask: {fdr_mask_path.name}")

    if have_blocks and fdr_c2f_mask is not None:
        fdr_c2f_path = out_dir / f"scramble_paired_stats_{n_subs}subs{suffix}_fdr_c2f_mask.dscalar.nii"
        save_cifti_map(fdr_c2f_mask, args.template_cifti, str(fdr_c2f_path), "fdr_c2f_mask")
        log.info(f"Saved 2-factor FDR mask: {fdr_c2f_path.name}")

    if have_perm and fdr_perm_mask is not None:
        fdr_perm_path = out_dir / f"scramble_paired_stats_{n_subs}subs{suffix}_fdr_perm_mask.dscalar.nii"
        save_cifti_map(fdr_perm_mask, args.template_cifti, str(fdr_perm_path), "fdr_perm_mask")
        log.info(f"Saved sign-flip-perm FDR mask: {fdr_perm_path.name}")

    summary = {
        "intact_model": args.intact_model, "scrambled_model": args.scrambled_model,
        "modality": args.modality, "config": config, "fmri_tag": args.fmri_tag,
        "n_subjects": n_subs, "n_grayordinates": n_grays, "alpha": args.alpha,
        "n_sig_uncorr": int((p_uncorr < args.alpha).sum()), "n_sig_fdr": n_sig_fdr,
        "max_t_stat": float(t_vals.max()), "mean_diff_range": [float(mean_diff.min()), float(mean_diff.max())],
        "have_blocks": have_blocks, "n_blocks": args.n_blocks, "n_sig_c2f": n_sig_c2f,
        "have_perm": have_perm, "n_perm": args.n_perm, "perm_seed": args.perm_seed,
        "n_sig_perm_fdr": n_sig_perm_fdr,
        "subjects": subs,
    }
    summary_path = out_dir / f"scramble_paired_stats_{n_subs}subs{suffix}_summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    log.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
