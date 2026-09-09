"""Tests for the mediation-asymmetry directionality test in roi_hierarchy.py."""

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "cf_modeling" / "roi_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("roi_hierarchy", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

N_RUNS = 4
N_BLOCKS_PER_RUN = 4
N_PER_BLOCK = 20


def _synthetic_chain(seed: int, downstream_predictable: bool = True):
    """S and T are both linear readouts of a shared stimulus, S with much
    less noise than T. Regressing S out of T removes nearly all of T's
    stimulus-driven signal (S is a near-clean stimulus proxy); regressing T
    out of S removes only part of S's signal (T is a noisy proxy) -- so T is
    "downstream of" (attenuated relative to) S, the classic asymmetric
    mediation setup, without relying on autocorrelation the synthetic
    regressors don't have.

    Samples are laid out as N_RUNS runs x N_BLOCKS_PER_RUN movie blocks each,
    with a distinct video_id per block, mirroring the real pipeline's
    run_id/video_id structure closely enough to exercise block-level
    resampling (as opposed to the old fold-level resampling).
    """
    rng = np.random.default_rng(seed)
    n_per_run = N_BLOCKS_PER_RUN * N_PER_BLOCK
    n_bins = N_RUNS * n_per_run
    run_ids = np.repeat(np.arange(N_RUNS), n_per_run)
    video_ids = np.concatenate([
        np.repeat(np.arange(run * N_BLOCKS_PER_RUN, (run + 1) * N_BLOCKS_PER_RUN), N_PER_BLOCK)
        for run in range(N_RUNS)
    ])

    audio = rng.normal(size=(n_bins, 3)).astype(np.float32)
    video = rng.normal(size=(n_bins, 3)).astype(np.float32)
    joint = rng.normal(size=(n_bins, 3)).astype(np.float32)
    stimulus = audio[:, 0] + video[:, 0] + joint[:, 0]

    source = stimulus + 0.2 * rng.normal(size=n_bins)
    if downstream_predictable:
        downstream = stimulus + 1.2 * rng.normal(size=n_bins)
    else:
        # Pure noise, unrelated to audio/video/joint: its own encoding R2
        # should sit at or below the redundancy floor.
        downstream = rng.normal(size=n_bins)

    roi_matrix = np.column_stack([source, downstream])
    return audio, video, joint, roi_matrix, run_ids, video_ids


def _fit(sources, downstream_predictable=True, seed=0):
    audio, video, joint, roi_matrix, run_ids, video_ids = _synthetic_chain(
        seed=seed, downstream_predictable=downstream_predictable,
    )
    alphas = np.logspace(-1, 3, 5)
    return MODULE.mediation_asymmetry(
        audio, video, joint, roi_matrix, ["source", "downstream"], sources, run_ids, video_ids, alphas,
        n_iter=5, backend="torch", model_random_state=0,
        n_bootstrap=200, n_permutations=200, random_state=0,
    )


def test_mediation_asymmetry_positive_for_true_downstream_direction():
    table = _fit(["source"])
    row = table.iloc[0]
    assert row["source"] == "source" and row["target"] == "downstream"
    assert row["asymmetry"] > 0


def test_mediation_asymmetry_negative_when_chain_is_reversed():
    table = _fit(["downstream"])
    row = table.iloc[0]
    assert row["source"] == "downstream" and row["target"] == "source"
    assert row["asymmetry"] < 0


def test_inference_resamples_over_blocks_not_folds():
    # Regression: the old implementation bootstrapped/sign-flipped over the
    # 4 outer LORO folds, flooring p at 1/2**4 = 0.0625. It must now resample
    # over movie blocks (N_RUNS * N_BLOCKS_PER_RUN = 16 here), which both
    # reports more blocks than folds and allows finer p-values.
    table = _fit(["source"])
    row = table.iloc[0]
    assert row["n_blocks"] == N_RUNS * N_BLOCKS_PER_RUN
    assert row["n_blocks"] > N_RUNS
    assert row["sign_flip_p_two_sided"] < 1.0 / (2 ** N_RUNS)


def test_attenuation_ratio_gated_when_target_r2_is_near_zero():
    # Regression: a ratio A(X->Y) = 1 - R2_given_X / R2_Y used to divide by a
    # clipped-to-floor R2_Y even when R2_Y was ~0 or negative, producing a
    # meaningless huge/wild ratio instead of an undefined one. "downstream"
    # here is pure noise unrelated to audio/video/joint, so its own held-out
    # R2 should sit at or below MIN_R2_FOR_REDUNDANCY on most blocks, and the
    # source->downstream attenuation should be gated to NaN there rather than
    # silently computed against a near-zero denominator.
    table = _fit(["source"], downstream_predictable=False)
    row = table.iloc[0]
    assert row["target"] == "downstream"
    assert row["n_blocks_defined"] < row["n_blocks"]


if __name__ == "__main__":
    test_mediation_asymmetry_positive_for_true_downstream_direction()
    test_mediation_asymmetry_negative_when_chain_is_reversed()
    test_inference_resamples_over_blocks_not_folds()
    test_attenuation_ratio_gated_when_target_r2_is_near_zero()
    print("ok")
