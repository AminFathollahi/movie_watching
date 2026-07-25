"""
rsa/delta_rho.py
================
Post-hoc Δρ analysis: measures how much a target RSA model exceeds its
best-performing baseline at every cortical vertex.

  Δρ(v) = ρ_target(v) − max(ρ_baseline_1(v), ρ_baseline_2(v), ...)

Per-subject rho maps are loaded for every model/modality, aligned to the
common subject set, and Δρ is computed per subject before group inference
(one-sample t-test + BH-FDR). This gives proper statistics on the advantage
rather than just subtracting group means.

Typical comparisons
-------------------
Within-architecture unimodal:
  --target pe-av-small-16-frame av
  --baselines pe-av-small-16-frame a  pe-av-small-16-frame v

Cross-architecture unimodal:
  --target pe-av-small-16-frame av
  --baselines audiomae a  videomaev2-large v

Bimodal-with-text baseline:
  --target pe-av-small-16-frame av
  --baselines pe-av-small-16-frame avt

Usage
-----
python rsa/delta_rho.py \\
    --output-dir /path/to/searchlight_rsa_output \\
    --target pe-av-small-16-frame av \\
    --baselines pe-av-small-16-frame a pe-av-small-16-frame v \\
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 \\
    --method spearman --fmri-tag raw \\
    --template-cifti /path/to/template.dscalar.nii \\
    --out-dir /path/to/output

Output CIFTI maps
-----------------
  delta_rho     — group mean Δρ
  rho_target    — group mean ρ for the target model
  rho_max_base  — vertex-wise maximum over baseline group mean ρ maps
  t_stat        — one-sample t on per-subject Δρ
  sigmap_uncorr — sign(Δρ) × −log10(p_uncorr)
  sigmap_fdr    — sign(Δρ) × −log10(p_fdr)   [BH-FDR]

A binary fdr_mask is saved as a companion dscalar.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import (
    get_combined_map_names,
    save_cifti_map,
    save_cifti_multimap,
)
from rsa.shared.rsa_utils import aggregate_blocks, corrected_2factor_bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# Predefined runs
# =============================================================================

# Each entry: target (model, modality) and baselines [(model, modality), ...]
# Mirrors the philosophy of PARTIAL_RSA_RUNS in model_registry.py but operates
# on already-computed per-subject rho maps (no searchlight re-run needed).

DELTA_RHO_RUNS: dict[str, dict] = {
    # Within-architecture: does the joint embedding exceed the better of its
    # own audio-only or video-only decoder?
    "within_architecture_unimodal": {
        "target":      ("pe-av-small-16-frame", "av"),
        "baselines":   [("pe-av-small-16-frame", "a"), ("pe-av-small-16-frame", "v")],
        "description": (
            "PE-AV/av vs max(PE-AV/a, PE-AV/v) — "
            "within-architecture AV advantage over own unimodal decoders."
        ),
    },
    # Cross-architecture: does PE-AV/av beat the best independently-trained
    # specialist available (AudioMAE for audio, VideoMAEv2 for video)?
    "cross_architecture_specialist": {
        "target":      ("pe-av-small-16-frame", "av"),
        "baselines":   [("audiomae", "a"), ("videomaev2-large", "v")],
        "description": (
            "PE-AV/av vs max(AudioMAE/a, VideoMAEv2/v) — "
            "cross-architecture AV advantage over independently-trained specialists."
        ),
    },
    # Strictest specialist baseline: WavLM is the strongest speech/audio
    # model; PE-Core is a vision-only Perception Encoder.
    "cross_family_specialist_strict": {
        "target":      ("pe-av-small-16-frame", "av"),
        "baselines":   [("wavlm-large", "a"), ("pe-core-l14", "v")],
        "description": (
            "PE-AV/av vs max(WavLM-Large/a, PE-Core-L14/v) — "
            "strictest cross-family specialist baseline."
        ),
    },
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Δρ = ρ_target − max(ρ_baselines): post-hoc advantage map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Predefined runs (--run): "
            + "  ".join(f"{k}: {v['description']}" for k, v in DELTA_RHO_RUNS.items())
        ),
    )
    p.add_argument("--output-dir",    required=True,
                   help="Root RSA output directory containing per-subject subdirs.")
    # Either use a predefined run OR specify target/baselines manually
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run",         choices=list(DELTA_RHO_RUNS.keys()),
                     help="Predefined run key (see DELTA_RHO_RUNS above).")
    grp.add_argument("--target",      nargs=2, metavar=("MODEL", "MODALITY"),
                     help="Target model and modality.")
    p.add_argument("--baselines",     nargs="+", metavar="TOKEN",
                   help="Flat list of baseline MODEL MODALITY pairs (required if --target used).")
    p.add_argument("--k",             type=int,   required=True)
    p.add_argument("--bin-sec",       type=float, required=True)
    p.add_argument("--skip-sec",      type=float, default=None, dest="skip_sec")
    p.add_argument("--delay-sec",     type=float, default=5.0)
    p.add_argument("--method",        default="spearman",
                   choices=["spearman", "pearson", "rho_a"])
    p.add_argument("--fmri-tag",      required=True, dest="fmri_tag")
    p.add_argument("--template-cifti", required=True, dest="template_cifti")
    p.add_argument("--out-dir",       required=True, dest="out_dir")
    p.add_argument("--alpha",         type=float, default=0.05)
    p.add_argument("--n-blocks",      type=int, default=4, dest="n_blocks",
                   help="Number of temporal blocks expected in per-subject block .npy files "
                        "(from searchlight.py --n-blocks). Set to 1 to skip 2-factor bootstrap.")
    p.add_argument("--n-bootstrap",   type=int, default=2000, dest="n_bootstrap",
                   help="Bootstrap iterations for the corrected 2-factor variance estimate "
                        "(Schütt et al. 2023, Eq. 5).")
    args = p.parse_args()

    # Resolve predefined run into target/baselines
    if args.run is not None:
        cfg = DELTA_RHO_RUNS[args.run]
        args.target    = list(cfg["target"])
        args.baselines = [tok for pair in cfg["baselines"] for tok in pair]
        log.info(f"Run {args.run}: {cfg['description']}")
    elif args.baselines is None:
        p.error("--baselines is required when --target is used.")

    return args


def _parse_baseline_pairs(tokens: list[str]) -> list[tuple[str, str]]:
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"--baselines requires an even number of tokens (MODEL MODALITY pairs); "
            f"got {len(tokens)}: {tokens}"
        )
    return [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]


# =============================================================================
# Per-subject rho loading
# =============================================================================

def _config_path_parts(k, bin_sec, skip_sec, delay_sec, method):
    bin_int  = int(bin_sec)
    skip_int = int(skip_sec)
    delay_tag = f"delay{int(delay_sec)}s"
    config    = f"k{k}_{delay_tag}_bin{bin_int}s_skip{skip_int}s_{method}"
    fname     = (f"rsa_59k_{{fmri_tag}}_k{k}_{delay_tag}"
                 f"_bin{bin_int}s_skip{skip_int}s_{method}_searchlight.npy")
    return config, fname


def load_rho_stack(
    output_dir: str,
    model: str,
    modality: str,
    k: int,
    bin_sec: float,
    skip_sec: float,
    delay_sec: float,
    method: str,
    fmri_tag: str,
) -> tuple[np.ndarray, list[str]]:
    """Load per-subject rho .npy files for one model/modality.

    Returns
    -------
    rho_stack  : (n_subs, n_verts) float32
    subject_ids: list[str]  — parent dir names, one per subject
    """
    bin_int   = int(bin_sec)
    skip_int  = int(skip_sec)
    delay_tag = f"delay{int(delay_sec)}s"
    config    = f"k{k}_{delay_tag}_bin{bin_int}s_skip{skip_int}s_{method}"
    fname     = (f"rsa_59k_{fmri_tag}_k{k}_{delay_tag}"
                 f"_bin{bin_int}s_skip{skip_int}s_{method}_searchlight.npy")
    model_mod = f"{model}_{modality}"

    files = sorted(
        Path(output_dir).glob(f"subject_data/*/{model_mod}/{config}/{fname}")
    )
    if not files:
        raise FileNotFoundError(
            f"No per-subject rho files found for {model}/{modality}.\n"
            f"Pattern: {output_dir}/subject_data/*/{model_mod}/{config}/{fname}"
        )

    rho_maps = [np.load(str(f)).astype(np.float32) for f in files]
    subject_ids = [f.parent.parent.parent.name for f in files]
    log.info(f"  Loaded {len(files)} subjects for {model}/{modality}")
    return np.stack(rho_maps, axis=0), subject_ids


def load_block_stack(
    output_dir: str,
    model: str,
    modality: str,
    k: int,
    bin_sec: float,
    skip_sec: float,
    delay_sec: float,
    method: str,
    fmri_tag: str,
    n_blocks: int,
) -> tuple[np.ndarray, list[str]]:
    """Load per-subject block-wise rho .npy files, with aggregation fallback.

    If exact n_blocks files are not found, searches for the smallest available
    N_file that (a) is larger than n_blocks and (b) divides evenly, then
    aggregates via aggregate_blocks().  This lets a single searchlight run at
    a large N serve multiple group_stats sweep values without re-running.

    Returns
    -------
    block_stack : (n_subs, n_blocks, n_verts) float32
    subject_ids : list[str]
    """
    bin_int   = int(bin_sec)
    skip_int  = int(skip_sec)
    delay_tag = f"delay{int(delay_sec)}s"
    config    = f"k{k}_{delay_tag}_bin{bin_int}s_skip{skip_int}s_{method}"
    model_mod = f"{model}_{modality}"

    def _collect(n_b):
        fname = (f"rsa_59k_{fmri_tag}_k{k}_{delay_tag}"
                 f"_bin{bin_int}s_skip{skip_int}s_{method}_searchlight_nblocks{n_b}.npy")
        return sorted(Path(output_dir).glob(f"subject_data/*/{model_mod}/{config}/{fname}"))

    files = _collect(n_blocks)
    n_file = n_blocks

    if not files:
        available_ns = sorted(set(
            int(m.group(1))
            for f in Path(output_dir).glob(
                f"subject_data/*/{model_mod}/{config}/*_nblocks*.npy"
            )
            for m in [re.search(r'_nblocks(\d+)\.npy$', f.name)]
            if m
        ))
        candidates = [n for n in available_ns if n > n_blocks and n % n_blocks == 0]
        if candidates:
            n_file = min(candidates)
            files = _collect(n_file)

    if not files:
        raise FileNotFoundError(
            f"No block rho files found for {model}/{modality} "
            f"(n_blocks={n_blocks} or any divisible multiple).\n"
            f"Pattern: {output_dir}/subject_data/*/{model_mod}/{config}/*_nblocks*.npy"
        )

    block_maps  = [np.load(str(f)).astype(np.float32) for f in files]
    subject_ids = [f.parent.parent.parent.name for f in files]
    raw_stack   = np.stack(block_maps, axis=0)

    if n_file != n_blocks:
        raw_stack = aggregate_blocks(raw_stack, n_blocks)
        log.info(f"  Aggregated {n_file}→{n_blocks} blocks for {model}/{modality}")

    log.info(f"  Loaded {len(files)} block stacks for {model}/{modality} ({n_blocks} blocks)")
    return raw_stack, subject_ids


def _align_stacks(
    target_stack: np.ndarray,
    target_subs: list[str],
    baseline_stacks: list[np.ndarray],
    baseline_subs: list[list[str]],
    baseline_labels: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Intersect subject sets and reindex all stacks to the common subset."""
    common = set(target_subs)
    for subs in baseline_subs:
        common &= set(subs)
    common = sorted(common)

    if not common:
        raise RuntimeError("No subjects in common across target and all baselines.")

    n_before = len(target_subs)
    if len(common) < n_before:
        dropped = set(target_subs) - set(common)
        log.warning(f"Subject alignment: keeping {len(common)}/{n_before}; "
                    f"dropped {dropped}")

    def _reindex(stack, subs):
        idx = [subs.index(s) for s in common]
        return stack[idx]

    t_aligned = _reindex(target_stack, target_subs)
    b_aligned = [_reindex(bs, bl) for bs, bl in zip(baseline_stacks, baseline_subs)]

    for label, ba in zip(baseline_labels, b_aligned):
        log.info(f"  {label}: aligned to {len(common)} subjects, "
                 f"mean_rho={ba.mean():.4f}")

    return t_aligned, b_aligned, common


# =============================================================================
# Group inference helpers (mirror group_stats.py)
# =============================================================================

def _fdr_sigmap(p_uncorr, sign_vec, alpha):
    p_fdr = stats.false_discovery_control(p_uncorr, method="bh")
    eps = np.finfo(np.float32).tiny
    sigmap_fdr = (sign_vec * (-np.log10(np.maximum(p_fdr, eps)))).astype(np.float32)
    fdr_mask   = (p_fdr < alpha).astype(np.float32)
    return sigmap_fdr, fdr_mask, int(fdr_mask.sum())


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    baseline_pairs  = _parse_baseline_pairs(args.baselines)
    baseline_labels = [f"{m}/{mod}" for m, mod in baseline_pairs]
    target_label    = f"{args.target[0]}/{args.target[1]}"

    log.info("=" * 70)
    log.info(f"Δρ analysis")
    log.info(f"  target   : {target_label}")
    log.info(f"  baselines: {baseline_labels}")
    log.info(f"  k={args.k}  bin={args.bin_sec}s  skip={args.skip_sec}s  "
             f"delay={args.delay_sec}s  method={args.method}")
    log.info("=" * 70)

    # ── Build output path and skip-check ──────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_slug = "+".join(f"{m}_{mod}" for m, mod in baseline_pairs)
    tgt_slug  = f"{args.target[0]}_{args.target[1]}"
    bin_int   = int(args.bin_sec)
    skip_int  = int(args.skip_sec)
    delay_tag = f"delay{int(args.delay_sec)}s"
    config    = f"k{args.k}_{delay_tag}_bin{bin_int}s_skip{skip_int}s_{args.method}"

    # Placeholder n_subs name — resolved once stacks loaded
    out_stem  = f"delta_rho_{tgt_slug}_vs_{base_slug}_{config}"

    # ── Load rho stacks ───────────────────────────────────────────────────────
    log.info("Loading target rho maps ...")
    target_stack, target_subs = load_rho_stack(
        args.output_dir, args.target[0], args.target[1],
        args.k, args.bin_sec, args.skip_sec, args.delay_sec,
        args.method, args.fmri_tag,
    )

    log.info("Loading baseline rho maps ...")
    baseline_stacks = []
    baseline_sub_lists = []
    for model, modality in baseline_pairs:
        bs, subs = load_rho_stack(
            args.output_dir, model, modality,
            args.k, args.bin_sec, args.skip_sec, args.delay_sec,
            args.method, args.fmri_tag,
        )
        baseline_stacks.append(bs)
        baseline_sub_lists.append(subs)

    # ── Align subjects ────────────────────────────────────────────────────────
    target_stack, baseline_stacks, common_subs = _align_stacks(
        target_stack, target_subs,
        baseline_stacks, baseline_sub_lists,
        baseline_labels,
    )

    n_subs, n_verts = target_stack.shape

    # ── Try to load block stacks for 2-factor bootstrap ───────────────────────
    have_blocks = False
    if args.n_blocks > 1:
        try:
            target_block_stack, tgt_blk_subs = load_block_stack(
                args.output_dir, args.target[0], args.target[1],
                args.k, args.bin_sec, args.skip_sec, args.delay_sec,
                args.method, args.fmri_tag, args.n_blocks,
            )
            baseline_block_stacks = []
            for model, modality in baseline_pairs:
                bs_blk, _ = load_block_stack(
                    args.output_dir, model, modality,
                    args.k, args.bin_sec, args.skip_sec, args.delay_sec,
                    args.method, args.fmri_tag, args.n_blocks,
                )
                baseline_block_stacks.append(bs_blk)
            # Align block stacks to the same subject set as the full-series stacks
            blk_idx_map = {s: i for i, s in enumerate(tgt_blk_subs)}
            common_idx  = [blk_idx_map[s] for s in common_subs if s in blk_idx_map]
            if len(common_idx) == n_subs:
                target_block_stack  = target_block_stack[common_idx]
                baseline_block_stacks = [b[common_idx] for b in baseline_block_stacks]
                have_blocks = True
                log.info(
                    f"Block files found ({n_subs} subjects, {args.n_blocks} blocks) — "
                    f"will run corrected 2-factor bootstrap (Schütt et al. 2023, Eq. 5)"
                )
            else:
                log.warning(
                    f"Block subject set doesn't fully overlap ({len(common_idx)}/{n_subs}) — "
                    f"skipping 2-factor bootstrap."
                )
        except FileNotFoundError as e:
            log.info(f"Block files not available — skipping 2-factor bootstrap.\n  {e}")

    out_path      = out_dir / f"{out_stem}_{n_subs}subs.dscalar.nii"
    fdr_mask_path = out_dir / f"{out_stem}_{n_subs}subs_fdr_mask.dscalar.nii"

    existing = get_combined_map_names(str(out_path)) if out_path.exists() else []
    if "sigmap_fdr" in existing:
        log.info(f"Already computed: {out_path.name} — skipping.")
        return

    # ── Per-subject Δρ ────────────────────────────────────────────────────────
    # max_baseline shape: (n_subs, n_verts)
    max_baseline = baseline_stacks[0].copy()
    for bs in baseline_stacks[1:]:
        np.maximum(max_baseline, bs, out=max_baseline)

    delta_stack = target_stack - max_baseline      # (n_subs, n_verts), float32

    # ── Per-subject per-block Δρ (for 2-factor bootstrap) ────────────────────
    delta_block_stack = None
    if have_blocks:
        max_base_block = baseline_block_stacks[0].copy()
        for bs_blk in baseline_block_stacks[1:]:
            np.maximum(max_base_block, bs_blk, out=max_base_block)
        delta_block_stack = target_block_stack - max_base_block  # (n_subs, n_blocks, n_verts)

    log.info(f"n_subjects = {n_subs}  n_verts = {n_verts:,}")
    log.info(f"mean Δρ across subjects & vertices: {delta_stack.mean():.4f}")

    # ── Group means ───────────────────────────────────────────────────────────
    mean_delta     = delta_stack.mean(axis=0).astype(np.float32)
    mean_target    = target_stack.mean(axis=0).astype(np.float32)
    mean_max_base  = max_baseline.mean(axis=0).astype(np.float32)

    # ── One-sample t-test on Δρ (H₀: mean Δρ = 0) ────────────────────────────
    D = delta_stack.astype(np.float64)
    t_vals, p_two = stats.ttest_1samp(D, popmean=0.0, axis=0)
    t_vals = t_vals.astype(np.float32)

    p_uncorr = np.where(t_vals > 0,
                        p_two / 2.0,
                        1.0 - p_two / 2.0).astype(np.float32)

    eps = np.finfo(np.float32).tiny
    sign_delta    = np.sign(mean_delta).astype(np.float32)
    sigmap_uncorr = (sign_delta *
                     (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)

    sigmap_fdr, fdr_mask, n_sig_fdr = _fdr_sigmap(p_uncorr, sign_delta, args.alpha)

    log.info(f"Δρ range: [{mean_delta.min():.4f}, {mean_delta.max():.4f}]")
    log.info(f"Verts with Δρ > 0: {(mean_delta > 0).sum():,} / {n_verts:,}")
    log.info(f"Uncorrected p<{args.alpha}: {(p_uncorr < args.alpha).sum():,} / {n_verts:,}")
    log.info(f"BH-FDR p<{args.alpha}: {n_sig_fdr:,} / {n_verts:,}")

    # ── Corrected 2-factor bootstrap on Δρ (Schütt et al. 2023, Eq. 5) ──────
    n_sig_c2f_delta = 0
    fdr_c2f_delta_mask = None
    if have_blocks and delta_block_stack is not None:
        log.info(
            f"Running corrected 2-factor bootstrap on Δρ "
            f"(n_boot={args.n_bootstrap}, n_blocks={args.n_blocks}) ..."
        )
        var_c2f, var_subj_boot, var_block_boot = corrected_2factor_bootstrap(
            delta_stack, delta_block_stack, n_boot=args.n_bootstrap
        )
        se_c2f = np.sqrt(np.maximum(var_c2f, 0.0)).astype(np.float64)
        # df = min(N_s-1, N_c-1) per Schütt et al. 2023 §5.1.4 (conservative choice)
        df_c2f = max(min(n_subs - 1, args.n_blocks - 1), 1)
        mean_delta_f64 = mean_delta.astype(np.float64)
        safe_se = np.where(se_c2f > 0, se_c2f, 1.0)  # avoid divide-by-zero; masked below
        t_c2f_delta = np.where(
            se_c2f > 0, mean_delta_f64 / safe_se, 0.0
        ).astype(np.float32)
        p_two_c2f = stats.t.sf(np.abs(t_c2f_delta.astype(np.float64)), df=df_c2f) * 2.0
        p_c2f = np.where(
            t_c2f_delta > 0, p_two_c2f / 2.0, 1.0 - p_two_c2f / 2.0
        ).astype(np.float32)
        sigmap_c2f_delta, fdr_c2f_delta_mask, n_sig_c2f_delta = _fdr_sigmap(
            p_c2f, np.sign(mean_delta).astype(np.float32), args.alpha
        )
        log.info(
            f"2-factor Δρ bootstrap: df={df_c2f}, "
            f"FDR sig verts = {n_sig_c2f_delta:,} / {n_verts:,}"
        )

    # ── Save CIFTI ────────────────────────────────────────────────────────────
    maps_list = [mean_delta, mean_target, mean_max_base, t_vals, sigmap_uncorr, sigmap_fdr]
    map_names = ["delta_rho", "rho_target", "rho_max_base", "t_stat", "sigmap_uncorr", "sigmap_fdr"]

    if have_blocks and delta_block_stack is not None:
        maps_list += [t_c2f_delta, sigmap_c2f_delta]
        map_names += ["t_c2f_delta", "sigmap_c2f_delta"]

    save_cifti_multimap(np.stack(maps_list, axis=0), map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path.name}")

    save_cifti_map(fdr_mask, args.template_cifti, str(fdr_mask_path), "fdr_mask")
    log.info(f"Saved FDR mask: {fdr_mask_path.name}")

    if have_blocks and fdr_c2f_delta_mask is not None:
        fdr_c2f_delta_path = out_dir / f"{out_stem}_{n_subs}subs_fdr_c2f_mask.dscalar.nii"
        save_cifti_map(
            fdr_c2f_delta_mask, args.template_cifti, str(fdr_c2f_delta_path), "fdr_c2f_delta_mask"
        )
        log.info(f"Saved 2-factor FDR mask: {fdr_c2f_delta_path.name}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "target":              target_label,
        "baselines":           baseline_labels,
        "config":              config,
        "fmri_tag":            args.fmri_tag,
        "n_subjects":          n_subs,
        "n_grayordinates":     n_verts,
        "alpha":               args.alpha,
        "delta_rho_range":     [float(mean_delta.min()), float(mean_delta.max())],
        "frac_positive":       float((mean_delta > 0).mean()),
        "max_t_stat":          float(t_vals.max()),
        "max_sigmap_uncorr":   float(sigmap_uncorr.max()),
        "max_sigmap_fdr":      float(sigmap_fdr.max()),
        "n_sig_uncorr":        int((p_uncorr < args.alpha).sum()),
        "n_sig_fdr":           n_sig_fdr,
        "have_blocks":         have_blocks,
        "n_blocks":            args.n_blocks,
        "n_bootstrap":         args.n_bootstrap,
        "n_sig_c2f_delta":     n_sig_c2f_delta,
        "rho_target_range":    [float(mean_target.min()), float(mean_target.max())],
        "rho_max_base_range":  [float(mean_max_base.min()), float(mean_max_base.max())],
        "subjects":            common_subs,
    }

    summary_path = out_dir / f"{out_stem}_{n_subs}subs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: {summary_path.name}")
    log.info(
        f"Done. Δρ_mean={mean_delta.mean():.4f}  "
        f"FDR sig={n_sig_fdr:,}/{n_verts:,}"
    )


if __name__ == "__main__":
    main()
