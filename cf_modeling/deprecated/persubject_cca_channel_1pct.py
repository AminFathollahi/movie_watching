#!/usr/bin/env python3
"""Per-subject top-1% PE-AV CCA-A/CCA-P masks vs the group-level modality axis.

Each subject's CCA-A/CCA-P mask is the top 1% of that subject's OWN PE-AV RSA
searchlight map (outputs/rsa/raw/subject_data/{sub}/...), not the group-average
mask applied to every subject. No CF banded-ridge refit is needed: mask
definition reuses run_cca_islands.py verbatim (subprocess, group-average
defaults), and ROI-mean fMRI timecourses come from in-memory SG->PSC streaming
preprocessing (preprocess_individual.preprocess_subject) so no per-subject
CIFTI is written to disk.

Rebuilt 2026-09-02 around ``cf_modeling.deprecated.channel_cca_analysis``'s two-axis
framework (see that module's docstring): the MODALITY axis (audio-minus-video
sufficiency, per channel) depends only on PE-AV's model embeddings, never on
fMRI, so it is identical for every subject and computed exactly ONCE here from
the group embeddings. Only the CCA axis varies per subject, because it is
built from that subject's own CCA-A/CCA-P mask and timecourses. For each of
the two variants (zero-order, partial), each subject contributes one point
estimate: Spearman and Pearson correlation (across channels) between the
fixed modality axis and that subject's own CCA axis. No per-subject
block-bootstrap CI is computed -- with ~175 subjects this would multiply the
group-level analysis's runtime roughly 175x for a quantity (a single
channel-axis-vs-axis correlation) that is not itself the target of inference
here; only the cross-subject distribution is. That distribution is
summarized with its mean and ONE ordinary (i.i.d., subjects are independent
draws) bootstrap 95% CI over subjects, per variant per correlation type (4
CIs total) -- no one-sample t-test, no Wilcoxon, no other significance
machinery. This mirrors the group-level module's "exactly one CI, nothing
else inferential" rule at the across-subjects level.

The previous version of this script depended on the retired 4-class
``channel_sensitivity`` labelling (``group_a``/``group_b`` == "audio_only"/
"video_only" row selection) and hardcoded a stale headline reference
(+0.0773, 95% CI [0.0291, 0.1191], BH q=0.0024) from the deleted
median-difference group test. Both are gone; compare this script's
``group_summary.json`` against the CURRENT group-level output written by
``channel_cca_analysis.py`` (``outputs/cf_modeling/channel_cca_preference/
results/channel_modality_cca_correlation_*.json``) rather than a number
frozen in source.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_modeling.cf_naming import persubject_output_root  # noqa: E402
from cf_modeling.roi_mean_partial_connectivity import _load_mask  # noqa: E402
from cf_modeling.deprecated.channel_cca_analysis import (  # noqa: E402
    MODEL_CONFIGS, VARIANTS, _model_files, _processed_embedding, cca_axis,
    modality_axis,
)
from rsa.shared.rsa_utils import get_run_bin_counts, preprocess_fmri  # noqa: E402
from preprocess_individual import preprocess_subject  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("persubject_cca")

MOVIE_ROOT = Path("/home/amin/Research/Representation/Movie")
DATA = MOVIE_ROOT / "data"
OUTPUTS = MOVIE_ROOT / "outputs"
RAW_DIR = Path(os.environ.get(
    "MOVIE_RAW_CIFTI_DIR",
    "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/individual-59k",
))
SUBJECTS_LIST = DATA / "subjects.txt"
TIMING_CSV = DATA / "movie_timing.csv"
ROI_CONFIG = ROOT / "cf_modeling" / "deprecated" / "roi_definitions.json"
GLASSER_DLABEL = (
    Path(os.environ.get(
        "MOVIE_HCP_DIR",
        "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1",
    ))
    / "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_"
    "Group_Colors.59k_fs_LR.dlabel.nii")
SEARCHLIGHT_TMPL = (
    str(OUTPUTS / "rsa/raw/subject_data/{sub}/pe-av-small-16-frame_av"
        "/k100_delay5s_bin5s_skip5s_spearman"
        "/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"))
EMBEDDINGS_DIR = OUTPUTS / "model_embeddings"
OUT_DIR = persubject_output_root() / "persubject_cca_1pct"
N_GRAY = 108441
BIN_SEC = SKIP_SEC = DELAY_SEC = 5.0
TR = 1.0
SEED = 20260826
N_BOOT_SUBJECTS = 2000  # ordinary bootstrap over subjects, for the group CI only


class _Args(types.SimpleNamespace):
    """Minimal stand-in for channel_cca_analysis.py's argparse Namespace,
    just the fields _processed_embedding reads."""


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * np.sum(a & b) / (np.sum(a) + np.sum(b)))


def _selfcheck() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    c = np.array([False, False, True, True])
    assert _dice(a, a) == 1.0
    assert abs(_dice(a, b) - 0.5) < 1e-9
    assert _dice(a, c) == 0.0


def load_subjects() -> list[str]:
    subs = []
    for line in SUBJECTS_LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        subs.append(line.split()[0])
    return subs


def subject_mask(sub: str, tmp_base: Path) -> dict | None:
    map_path = Path(SEARCHLIGHT_TMPL.format(sub=sub))
    if not map_path.exists():
        log.warning("%s: no per-subject PE-AV searchlight map", sub)
        return None
    tag = f"peav_1pct_sub{sub}"
    out_base = tmp_base / sub
    out_base.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "cf_modeling/deprecated/run_cca_islands.py"),
        "--map", str(map_path), "--top-percent", "1",
        "--roi-suffix", tag, "--roi-config", str(ROI_CONFIG),
        "--glasser-dlabel", str(GLASSER_DLABEL),
        "--output-base", str(out_base),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()
        log.warning("%s: mask extraction failed: %s", sub, tail[-1] if tail else "")
        shutil.rmtree(out_base, ignore_errors=True)
        return None
    masks_dir = out_base / "masks"
    mask_a = _load_mask(masks_dir / f"cca_a_{tag}_mask.dscalar.nii", N_GRAY)
    mask_p = _load_mask(masks_dir / f"cca_p_{tag}_mask.dscalar.nii", N_GRAY)
    shutil.rmtree(out_base, ignore_errors=True)
    return {"mask_a": mask_a, "mask_p": mask_p}


def pairwise_dice(mask_by_sub: dict[str, np.ndarray]) -> np.ndarray:
    subs = list(mask_by_sub)
    stacked = np.stack([mask_by_sub[s] for s in subs]).astype(np.float32)
    inter = stacked @ stacked.T
    sizes = stacked.sum(axis=1)
    dice = 2 * inter / (sizes[:, None] + sizes[None, :])
    iu = np.triu_indices(len(subs), k=1)
    return dice[iu]


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict:
    """ONE ordinary (subjects are i.i.d.) bootstrap 95% CI on the mean."""
    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = values[rng.integers(0, n, size=n)].mean()
    return {
        "n_subjects": int(n), "mean": float(values.mean()),
        "ci_95_low": float(np.quantile(boot_means, 0.025)),
        "ci_95_high": float(np.quantile(boot_means, 0.975)),
    }


def main() -> None:
    _selfcheck()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects()
    timing = pd.read_csv(TIMING_CSV)
    run_trs = np.asarray(np.load(DATA / "preprocessed/average_sub/hedger_sg_psc"
                                 "/group_average_hedger_sg_psc_run_trs.npy"), dtype=int)
    run_bins = get_run_bin_counts(timing, run_trs, BIN_SEC, TR, DELAY_SEC, SKIP_SEC)

    args = _Args(timing_csv=TIMING_CSV, run_trs=DATA / "preprocessed/average_sub/hedger_sg_psc"
                 "/group_average_hedger_sg_psc_run_trs.npy",
                 bin_sec=BIN_SEC, skip_sec=SKIP_SEC, delay_sec=DELAY_SEC, tr=TR)
    intact_path, from_a_path, from_v_path = _model_files(EMBEDDINGS_DIR, MODEL_CONFIGS["peav"])
    intact = _processed_embedding(intact_path, args, run_bins)
    from_a = _processed_embedding(from_a_path, args, run_bins)
    from_v = _processed_embedding(from_v_path, args, run_bins)
    # Fixed, fMRI-independent modality axis, once, both variants.
    fixed_modality_axis = {
        variant: modality_axis(variant, intact, from_a, from_v) for variant in VARIANTS
    }

    tmp_base = Path(tempfile.mkdtemp(prefix="persubject_cca_"))
    prep_args = types.SimpleNamespace(sg_filter=True, psc=True, gsr=False)

    rows = []
    mask_a_by_sub: dict[str, np.ndarray] = {}
    mask_p_by_sub: dict[str, np.ndarray] = {}
    try:
        for i, sub in enumerate(subjects, 1):
            log.info("[%d/%d] %s", i, len(subjects), sub)
            masks = subject_mask(sub, tmp_base)
            if masks is None:
                continue
            mask_a, mask_p = masks["mask_a"], masks["mask_p"]
            try:
                data, _bm, sub_run_trs = preprocess_subject(sub, RAW_DIR, TR, prep_args)
            except Exception as exc:
                log.warning("%s: preprocessing failed: %s", sub, exc)
                continue
            sub_run_bins = get_run_bin_counts(timing, sub_run_trs, BIN_SEC, TR, DELAY_SEC, SKIP_SEC)
            if int(sub_run_bins.sum()) != intact.shape[0]:
                log.warning("%s: bin mismatch (%d vs %d), skipping",
                            sub, int(sub_run_bins.sum()), intact.shape[0])
                continue
            mean_a = data[mask_a].mean(axis=0)
            mean_p = data[mask_p].mean(axis=0)
            del data
            continuous = np.vstack([mean_a, mean_p])
            binned = preprocess_fmri(continuous, timing, sub_run_trs, BIN_SEC, TR,
                                     DELAY_SEC, skip_sec=SKIP_SEC, normalize=True)
            cca_a, cca_p = binned[:, 0].astype(np.float64), binned[:, 1].astype(np.float64)

            row = {"subject": sub, "n_vertices_a": int(mask_a.sum()), "n_vertices_p": int(mask_p.sum())}
            for variant in VARIANTS:
                # sub_run_bins (not the group's run_bins): the partial variant's
                # internal per-run z-scoring must segment on THIS subject's own
                # run-TR boundaries, not the group-average ones, even though in
                # practice HCP movie-watching runs are the same length for
                # every subject and the two arrays are expected to agree.
                subject_cca_axis = cca_axis(variant, intact, cca_a, cca_p, sub_run_bins)
                row[f"{variant}_spearman_rho"] = float(
                    spearmanr(fixed_modality_axis[variant], subject_cca_axis).statistic)
                row[f"{variant}_pearson_r"] = float(
                    np.corrcoef(fixed_modality_axis[variant], subject_cca_axis)[0, 1])
            rows.append(row)
            mask_a_by_sub[sub] = mask_a
            mask_p_by_sub[sub] = mask_p
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "per_subject_correlation.csv", index=False)

    dice_a = pairwise_dice(mask_a_by_sub)
    dice_p = pairwise_dice(mask_p_by_sub)
    dice_summary = {
        "n_subjects_with_masks": len(mask_a_by_sub),
        "cca_a_mean_pairwise_dice": float(dice_a.mean()),
        "cca_a_std_pairwise_dice": float(dice_a.std()),
        "cca_p_mean_pairwise_dice": float(dice_p.mean()),
        "cca_p_std_pairwise_dice": float(dice_p.std()),
    }
    (OUT_DIR / "mask_dice_summary.json").write_text(json.dumps(dice_summary, indent=2) + "\n")

    group_summary = {"n_subjects": int(len(table))}
    for variant in VARIANTS:
        group_summary[variant] = {
            "spearman_rho": _bootstrap_mean_ci(
                table[f"{variant}_spearman_rho"].to_numpy(), N_BOOT_SUBJECTS, SEED),
            "pearson_r": _bootstrap_mean_ci(
                table[f"{variant}_pearson_r"].to_numpy(), N_BOOT_SUBJECTS, SEED + 1),
        }
    (OUT_DIR / "group_summary.json").write_text(json.dumps(group_summary, indent=2) + "\n")
    log.info("Done. n=%d subjects. %s", len(table), json.dumps(group_summary))


if __name__ == "__main__":
    main()
