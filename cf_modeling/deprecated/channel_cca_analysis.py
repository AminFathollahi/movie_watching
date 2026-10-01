#!/usr/bin/env python3
"""Per-channel modality axis vs CCA axis: one continuous scatter, no classes.

For every channel of a model's full audiovisual embedding, build two signed,
continuous, scale-free axes and correlate them across channels:

  MODALITY axis_i = f(intact_i, from_a_i) - f(intact_i, from_v_i)
      positive = audio-preferring
  CCA axis_i      = f(intact_i, cca_a)    - f(intact_i, cca_p)
      positive = anterior-preferring

``from_a``/``from_v`` are this model's audio-only / video-only ablation
passes (``_clsav_from_a`` / ``_clsav_from_v``); ``cca_a``/``cca_p`` are the
group-average anterior/posterior CCA-ROI-mean fMRI timecourses. ``f`` is
either the zero-order Pearson correlation (VARIANT 1) or the partial
correlation controlling for the other series on the same axis (VARIANT 2,
formula ``r_xy.z = (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))``).

Preprocessing (identical for every series entering either axis): z-score
PER RUN over the 626 movie bins (runs = [151, 161, 156, 158]), done once by
``rsa.shared.rsa_utils.process_model_embeddings``/``preprocess_fmri`` with
``normalize=True`` before any correlation is taken.

No channel is ever assigned to a class, region, region-pair, or
"selective"/"redundant"/"unimodal" label anywhere in this module: both axes
are reported only as per-channel continuous values and as ONE Spearman and
ONE Pearson correlation between them (each with a single block-bootstrap 95%
CI, block_length=5 bins ~25s, resampled within run). That is the entire
analysis: 3 models x 2 variants = 6 scatterplots, 6 correlations. Nothing
else is computed here -- no permutation testing, no null distributions, no
significance counts, no FDR/BH, no winner-take-all group test, no 4-class
sensitivity scheme, no reliability/disattenuation/split-half machinery. All
of that previously lived in this module and has been deleted; see
``cf_modeling/deprecated/persubject_cca_channel_1pct.py`` and
``rsa/channel_class_rsa.py`` for the two downstream consumers that had to be
updated (the latter is deprecated, not deleted, and now defines its own
local ``CLASS_ORDER`` rather than importing it from here).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from textwrap import fill

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_modeling.roi_mean_partial_connectivity import (  # noqa: E402
    _load_mask, _stream_mask_means, paired_partial_correlations,
)
from rsa.shared.rsa_utils import (  # noqa: E402
    get_run_bin_counts, preprocess_fmri, process_model_embeddings,
)


log = logging.getLogger("channel_cca_analysis")

FAMILY_LABELS = {
    "peav": "PE-AV",
    "nemotron_layer18_mp": "Nemotron-18",
    "topoomni_layer18_sheet_mp": "Topo-Omni-18 sheet",
}
MODEL_CONFIGS = {
    "peav": "pe-av-small-16-frame",
    "nemotron_layer18_mp": "nemotron_layer18_mp",
    "topoomni_layer18_sheet_mp": "topoomni_layer18_sheet_mp",
}
VARIANTS = ("zero_order", "partial")

MODALITY_AXIS_LABEL = {
    "zero_order": "Modality axis: corr(intact, audio-only) − corr(intact, video-only)"
                  " (audio − video; positive = audio-preferring)",
    "partial": "Modality axis (partial): pcorr(intact, audio-only|video-only) − "
               "pcorr(intact, video-only|audio-only) (positive = audio-preferring)",
}
CCA_AXIS_LABEL = {
    "zero_order": "CCA axis: corr(intact, CCA-A) − corr(intact, CCA-P)"
                  " (anterior − posterior; positive = anterior-preferring)",
    "partial": "CCA axis (partial): pcorr(intact, CCA-A|CCA-P) − "
               "pcorr(intact, CCA-P|CCA-A) (positive = anterior-preferring)",
}


def _profile_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation between (T, C) arrays ``x`` and ``y``
    (``y`` may be (T, 1) to correlate every column of ``x`` against one
    shared series, e.g. a CCA-ROI timecourse). Demeans internally, so it does
    not depend on the per-run z-scoring already applied upstream being exact
    for the whole pooled series."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = (x * y).sum(axis=0)
    denominator = np.sqrt((x ** 2).sum(axis=0) * (y ** 2).sum(axis=0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator > np.finfo(float).eps)


def _elementwise_partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-column partial correlation r_xy.z for (T, C) arrays x, y, z that
    ALL vary per channel (unlike ``paired_partial_correlations``, whose
    second and third arguments are one shared fixed series for every
    channel). Same three-correlation identity, computed from three
    ``_profile_correlation`` calls."""
    r_xy = _profile_correlation(x, y)
    r_xz = _profile_correlation(x, z)
    r_yz = _profile_correlation(y, z)
    denominator = np.sqrt(np.clip((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2), 1e-12, None))
    return (r_xy - r_xz * r_yz) / denominator


def _standardize_runs(values: np.ndarray, run_bins: np.ndarray) -> np.ndarray:
    """Exact per-run z-score (mean 0, var 1 within each run individually),
    required by ``paired_partial_correlations``'s dot-product shortcut,
    which assumes its inputs are already standardized over the observations
    it sums across."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    output = np.empty_like(values)
    start = 0
    for count in run_bins:
        stop = start + int(count)
        block = values[start:stop]
        std = block.std(axis=0, keepdims=True)
        output[start:stop] = np.divide(
            block - block.mean(axis=0, keepdims=True), std,
            out=np.zeros_like(block), where=std > np.finfo(float).eps)
        start = stop
    return output


def _block_indices(run_bins: np.ndarray, block_length: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap resample, within each run independently."""
    pieces, start = [], 0
    for count in run_bins:
        count = int(count)
        selected = []
        while len(selected) < count:
            block_start = int(rng.integers(0, count))
            selected.extend((block_start + np.arange(block_length)) % count)
        pieces.append(start + np.asarray(selected[:count]))
        start += count
    return np.concatenate(pieces)


def modality_axis(variant: str, intact: np.ndarray, from_a: np.ndarray,
                  from_v: np.ndarray) -> np.ndarray:
    """audio-minus-video axis, shape (n_channels,). Depends only on this
    model's own embeddings (never on fMRI/CCA), so it is identical for every
    subject/ROI-set analysis that shares the same model and variant --
    callers needing just this axis (e.g. persubject_cca_channel_1pct.py's
    fixed group-level axis) should call this directly rather than the CCA
    half of ``axis_pair``, which requires fMRI ROI timecourses this
    computation does not use."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    if variant == "zero_order":
        return _profile_correlation(intact, from_a) - _profile_correlation(intact, from_v)
    return (_elementwise_partial_corr(intact, from_a, from_v)
            - _elementwise_partial_corr(intact, from_v, from_a))


def cca_axis(variant: str, intact: np.ndarray, cca_a: np.ndarray, cca_p: np.ndarray,
            run_bins: np.ndarray) -> np.ndarray:
    """anterior-minus-posterior axis, shape (n_channels,)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    if variant == "zero_order":
        return (_profile_correlation(intact, cca_a[:, None])
                - _profile_correlation(intact, cca_p[:, None]))
    intact_z = _standardize_runs(intact, run_bins)
    cca_a_z = _standardize_runs(cca_a, run_bins)[:, 0]
    cca_p_z = _standardize_runs(cca_p, run_bins)[:, 0]
    partial_a, partial_p = paired_partial_correlations(intact_z, cca_a_z, cca_p_z)
    return partial_a.astype(np.float64) - partial_p.astype(np.float64)


def axis_pair(
    variant: str, intact: np.ndarray, from_a: np.ndarray, from_v: np.ndarray,
    cca_a: np.ndarray, cca_p: np.ndarray, run_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (modality_axis, cca_axis), each shape (n_channels,)."""
    return (modality_axis(variant, intact, from_a, from_v),
            cca_axis(variant, intact, cca_a, cca_p, run_bins))


def block_bootstrap_axis_correlation(
    variant: str, intact: np.ndarray, from_a: np.ndarray, from_v: np.ndarray,
    cca_a: np.ndarray, cca_p: np.ndarray, run_bins: np.ndarray, *,
    n_boot: int, block_length: int, seed: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Observed Spearman+Pearson correlation between the two axes, each with
    ONE block-bootstrap 95% CI (jointly resampling all five input series
    from the same drawn bins each iteration, then recomputing both axes and
    both correlations). Returns (stats, modality_axis, cca_axis)."""
    modality_obs, cca_obs = axis_pair(variant, intact, from_a, from_v, cca_a, cca_p, run_bins)
    spearman_obs = float(spearmanr(modality_obs, cca_obs).statistic)
    pearson_obs = float(np.corrcoef(modality_obs, cca_obs)[0, 1])

    rng = np.random.default_rng(seed)
    boot_spearman = np.empty(n_boot)
    boot_pearson = np.empty(n_boot)
    for i in range(n_boot):
        selected = _block_indices(run_bins, block_length, rng)
        modality_i, cca_i = axis_pair(
            variant, intact[selected], from_a[selected], from_v[selected],
            cca_a[selected], cca_p[selected], run_bins)
        boot_spearman[i] = spearmanr(modality_i, cca_i).statistic
        boot_pearson[i] = np.corrcoef(modality_i, cca_i)[0, 1]

    stats = {
        "spearman_rho": spearman_obs,
        "spearman_ci_low": float(np.quantile(boot_spearman, 0.025)),
        "spearman_ci_high": float(np.quantile(boot_spearman, 0.975)),
        "pearson_r": pearson_obs,
        "pearson_ci_low": float(np.quantile(boot_pearson, 0.025)),
        "pearson_ci_high": float(np.quantile(boot_pearson, 0.975)),
        "n_channels": int(modality_obs.size),
        "n_boot": int(n_boot),
        "block_length_bins": int(block_length),
    }
    return stats, modality_obs, cca_obs


def save_scatter(
    modality: np.ndarray, cca: np.ndarray, stats: dict, variant: str,
    output_dir: Path, tag: str,
) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.8, 5.6), dpi=180)
    axis.scatter(modality, cca, s=12, alpha=0.35, color="#4c72b0", rasterized=True)
    axis.axhline(0, color="#aaaaaa", linewidth=0.6)
    axis.axvline(0, color="#aaaaaa", linewidth=0.6)
    axis.set_xlabel(fill(MODALITY_AXIS_LABEL[variant], 60), fontsize=8.5)
    axis.set_ylabel(fill(CCA_AXIS_LABEL[variant], 60), fontsize=8.5)
    text = (f"Spearman ρ={stats['spearman_rho']:+.3f} "
           f"[{stats['spearman_ci_low']:+.3f}, {stats['spearman_ci_high']:+.3f}]\n"
           f"Pearson r={stats['pearson_r']:+.3f} "
           f"[{stats['pearson_ci_low']:+.3f}, {stats['pearson_ci_high']:+.3f}]\n"
           f"n={stats['n_channels']} channels, {stats['n_boot']} bootstraps")
    axis.text(0.02, 0.98, text, transform=axis.transAxes, va="top", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92})
    axis.set_title(fill(f"{FAMILY_LABELS.get(tag, tag)} — {variant.replace('_', ' ')}", 56),
                  fontsize=10)
    figure.tight_layout()
    path = output_dir / f"{tag}_{variant}_scatter.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _model_files(root: Path, model: str) -> tuple[Path, Path, Path]:
    """(intact, from_a, from_v). ``from_a`` is the audio-only-input pass
    (video ablated, file suffix ``_clsav_from_a``); ``from_v`` is the
    video-only-input pass (audio ablated, file suffix ``_clsav_from_v``)."""
    def _path(tag: str) -> Path:
        return root / tag / "bin5s_skip5s" / f"{tag}_av.npy"
    return _path(model), _path(f"{model}_clsav_from_a"), _path(f"{model}_clsav_from_v")


def _processed_embedding(path: Path, args: argparse.Namespace, run_bins: np.ndarray) -> np.ndarray:
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    values = process_model_embeddings(
        str(path), timing, args.bin_sec, args.tr, run_trs,
        delay_sec=args.delay_sec, hrf=False, skip_sec=args.skip_sec, normalize=True)
    if values.shape[0] != int(run_bins.sum()):
        raise ValueError(f"Embedding bins {values.shape[0]} != expected {run_bins.sum()}: {path}")
    return values.astype(np.float64)


def _load_binned_rois(args: argparse.Namespace, run_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(args.dtseries)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    n_trs, n_gray = image.shape
    masks = {"a": _load_mask(args.mask_a, n_gray), "p": _load_mask(args.mask_p, n_gray)}
    means = _stream_mask_means(image.dataobj, masks, n_trs, n_gray, args.io_batch_size)
    timing = pd.read_csv(args.timing_csv)
    continuous = np.vstack([means["a"], means["p"]])
    binned = preprocess_fmri(
        continuous, timing, run_trs, args.bin_sec, args.tr, args.delay_sec,
        skip_sec=args.skip_sec, normalize=True)
    if binned.shape[0] != int(run_bins.sum()):
        raise ValueError(f"ROI bins {binned.shape[0]} != expected {run_bins.sum()}")
    return binned[:, 0].astype(np.float64), binned[:, 1].astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--dtseries", type=Path, required=True)
    parser.add_argument("--run-trs", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--mask-a", type=Path, required=True)
    parser.add_argument("--mask-p", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIGS),
        default=tuple(MODEL_CONFIGS), help="Model families to analyze.")
    parser.add_argument("--roi-set", help="Output-filename tag for the mask pair used.")
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--io-batch-size", type=int, default=4096)
    return parser.parse_args()


def _selfcheck() -> None:
    """Fast assert-based check of the partial-correlation math and sign
    conventions on synthetic data.

      1. ``_elementwise_partial_corr`` agrees with the fixed-vector
         ``paired_partial_correlations`` when the "varying per channel"
         series are in fact constant across channels (same math, two call
         shapes).
      2. An audio-driven synthetic channel (from_a == intact, from_v ==
         noise) scores positive on the modality axis in both variants; a
         video-driven channel scores negative.
      3. ``axis_pair`` for the CCA side reduces to the modality-axis case
         under the same construction (anterior-driven -> positive).
    """
    rng = np.random.default_rng(0)
    n = 200
    zscore = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    a = zscore(rng.normal(size=n))
    p = zscore(rng.normal(size=n))
    x = zscore(0.7 * a + 0.2 * p + rng.normal(scale=0.1, size=n))

    manual = _elementwise_partial_corr(x[:, None], a[:, None], p[:, None])[0]
    reference, _ = paired_partial_correlations(x[:, None], a, p)
    assert abs(manual - float(reference[0])) < 1e-5

    run_bins = np.array([n])
    base_a = np.sin(np.linspace(0, 6 * np.pi, n))
    base_v = np.cos(np.linspace(0, 10 * np.pi, n))
    noise = lambda: rng.normal(size=n)  # noqa: E731
    # channel 0: audio- and anterior-driven; channel 1: video- and posterior-driven.
    intact = np.column_stack([base_a, base_v])
    from_a = np.column_stack([base_a + 0.01 * noise(), noise()])
    from_v = np.column_stack([noise(), base_v + 0.01 * noise()])
    cca_a = base_a + 0.01 * noise()
    cca_p = base_v + 0.01 * noise()

    for variant in VARIANTS:
        modality, cca = axis_pair(variant, intact, from_a, from_v, cca_a, cca_p, run_bins)
        assert modality[0] > 0.5 and modality[1] < -0.5, (variant, modality)
        assert cca[0] > 0.5 and cca[1] < -0.5, (variant, cca)


def main() -> None:
    _selfcheck()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    run_bins = get_run_bin_counts(
        timing, run_trs, args.bin_sec, args.tr, args.delay_sec, args.skip_sec)
    cca_a, cca_p = _load_binned_rois(args, run_bins)

    results: dict[str, dict] = {}
    for tag in args.models:
        model = MODEL_CONFIGS[tag]
        intact_path, from_a_path, from_v_path = _model_files(args.embeddings_dir, model)
        for path in (intact_path, from_a_path, from_v_path):
            if not path.exists():
                raise FileNotFoundError(path)
        intact = _processed_embedding(intact_path, args, run_bins)
        from_a = _processed_embedding(from_a_path, args, run_bins)
        from_v = _processed_embedding(from_v_path, args, run_bins)

        results[tag] = {}
        for variant in VARIANTS:
            stats, modality, cca = block_bootstrap_axis_correlation(
                variant, intact, from_a, from_v, cca_a, cca_p, run_bins,
                n_boot=args.n_bootstraps, block_length=args.block_length,
                seed=args.seed)
            save_scatter(modality, cca, stats, variant, args.output_dir, tag)
            results[tag][variant] = stats
            log.info(
                "%s %s: Spearman %+.3f [%+.3f, %+.3f], Pearson %+.3f [%+.3f, %+.3f]",
                tag, variant, stats["spearman_rho"], stats["spearman_ci_low"],
                stats["spearman_ci_high"], stats["pearson_r"], stats["pearson_ci_low"],
                stats["pearson_ci_high"])

    metadata = {
        "analysis": "Per-channel modality axis vs CCA axis correlation (zero-order + partial)",
        "models": list(args.models), "roi_set": args.roi_set,
        "mask_a": str(args.mask_a), "mask_p": str(args.mask_p),
        "run_bins": run_bins.tolist(), "total_bins": int(run_bins.sum()),
        "n_bootstraps": args.n_bootstraps, "block_length_bins": args.block_length,
        "seed": args.seed,
        "results": results,
    }
    suffix = f"_{args.roi_set}" if args.roi_set else ""
    model_slug = "_and_".join(args.models)
    metadata_path = args.output_dir / f"channel_modality_cca_correlation_{model_slug}{suffix}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Wrote %s", metadata_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
