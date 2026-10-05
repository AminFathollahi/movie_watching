#!/usr/bin/env python3
"""
rsa/full_sheet_rsa.py
======================
RSA between the two CCA seed ROIs (audio-preferring / video-preferring) and
Topo-Omni's COMPLETE 304x512 (155,648-unit) cortical sheet, under TRUE
(trained, permute_coordinates seed=42) coordinates: the full vision + audio
encoder block (rows 0-159) plus the full 36-layer thinker (language-model)
stack (rows 160-303). Consolidates the three earlier sheet-RSA scripts
(layer-18-only raster lattice, hand-picked-layer true coordinates, and this
full-sheet driver) into one sweep over the whole sheet; geometry, plotting,
and characterization live in rsa/shared/sheet_rsa.py.

Per seed this computes: per-unit rho over all 155,648 units, per-tower
(vision/audio/thinker) mean rho, the cca_a-cca_p contrast per tower, hotspot
composition by tower, cross-seed hotspot overlap (Jaccard), and a topography
control -- true k-NN neighbourhoods vs. random same-tower unit samples,
reported honestly even when it does not favour the true neighbourhood.

Each seed's RDM is built from ALL vertices in its mask (cca_a 588, cca_p
370) across all 626 time bins, correlation distance -- NOT the ROI's mean
timecourse -- then Spearman-correlated against each sheet unit's own k=100
correlation-distance searchlight RDM. `sanity_corr` in the saved results is
a separate, cheaper side check on the seed's mean time series only; it is
not the RSA result. See rsa/shared/sheet_rsa.py for both.

Cross-seed spatial correlation (0.86) and hotspot Jaccard (0.79) are both
high, meaning the two seed maps largely coincide -- AND the cca_a-cca_p
difference map has real structure concentrated in the audio tower (per-
tower mean rho, cca_a/cca_p: vision 0.040/0.051, audio 0.273/0.144, thinker
0.071/0.093). Both facts hold at once; neither should be reported alone.

Modality: intact AV only -- topo_omni_extract_full_sheet.py extracts no
unimodal a/v passes for the full sheet, so no per-unit modality-preference
analysis is possible here.

Usage
-----
  python rsa/full_sheet_rsa.py --output-dir <out> --k 100 \\
      --gpu-batch-size 64 --perm-batch-size 20

  python rsa/full_sheet_rsa.py --demo
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DATA, OUTPUTS  # noqa: E402
from rsa.perm_searchlight import (  # noqa: E402
    _run_hemisphere, _run_hemisphere_paired, within_run_shift_pair_indices,
)
from rsa.shared.rsa_utils import (  # noqa: E402
    align_and_assert_bins, get_run_bin_counts, load_fmri_cifti,
    preprocess_fmri, process_model_embeddings,
)
from cf_modeling.roi_mean_partial_connectivity import _load_mask  # noqa: E402
from rsa.shared.sheet_rsa import (  # noqa: E402
    N_UNITS, SHEET_COLS, SHEET_ROWS, TOWER_NAMES, characterize, knn_on_sheet,
    load_true_coords, plot_sheet_map, random_neighbors_within_tower,
    robust_vlim, sanity_corr, tower_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "topoomni_fullsheet"


def demo() -> None:
    """Self-check: tower_id() geometry matches the spec (vision rows 0-159/
    cols 0-255, audio rows 0-159/cols 256-511, thinker rows 160-303)."""
    tid = tower_id()
    assert tid.shape == (N_UNITS,)
    assert (tid == 0).sum() == 160 * 256 == 40960  # vision
    assert (tid == 1).sum() == 160 * 256 == 40960  # audio
    assert (tid == 2).sum() == 144 * 512 == 73728  # thinker
    # spot-checks
    assert tid[0 * SHEET_COLS + 0] == 0       # row 0, col 0 -> vision
    assert tid[0 * SHEET_COLS + 511] == 1     # row 0, col 511 -> audio
    assert tid[159 * SHEET_COLS + 255] == 0   # row 159, col 255 -> vision
    assert tid[159 * SHEET_COLS + 256] == 1   # row 159, col 256 -> audio
    assert tid[160 * SHEET_COLS + 0] == 2     # row 160, col 0 -> thinker
    assert tid[303 * SHEET_COLS + 511] == 2   # row 303, col 511 -> thinker
    print("demo OK")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RSA between CCA seed ROIs and Topo-Omni's FULL cortical sheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--demo", action="store_true")

    data_base = str(DATA)
    outputs_base = str(OUTPUTS)

    p.add_argument("--preprocessed-dir", default=f"{data_base}/preprocessed/average_sub/raw")
    p.add_argument("--fmri-suffix", default="raw")
    p.add_argument("--timing-csv", default=f"{data_base}/movie_timing.csv")
    p.add_argument("--embeddings-dir", default=f"{outputs_base}/model_embeddings")
    p.add_argument("--masks-dir", default=f"{outputs_base}/cf_modeling/masks")
    p.add_argument("--seed-a-name", default="cca_a")
    p.add_argument("--seed-p-name", default="cca_p")
    p.add_argument("--seed-a-mask", default="cca_a_peav_1pct_mask.dscalar.nii")
    p.add_argument("--seed-p-mask", default="cca_p_peav_1pct_mask.dscalar.nii")
    p.add_argument("--true-coords-cache",
                   default=f"{outputs_base}/rsa/cca_seed_sheet_rsa/"
                            "topoomni_true_coords_seed42.npy")
    p.add_argument("--output-dir", default=None, help="Required unless --demo.")

    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--pref-top-decile", type=float, default=0.1,
                   help="Fraction of units (by rho) treated as each seed's hotspot.")
    p.add_argument("--topo-n-draws", type=int, default=3,
                   help="Random same-tower neighbourhood draws for the topography control.")
    p.add_argument("--topo-n-perm", type=int, default=50,
                   help="Small: the topography control compares rho means, not "
                        "p-values (actual_rho does not depend on n_perm).")
    p.add_argument("--gpu-batch-size", type=int, default=64)
    p.add_argument("--perm-batch-size", type=int, default=20)
    return p.parse_args()


# =============================================================================
# Core
# =============================================================================

def _load_seed_embedding(mask_path, fmri_continuous, timing_df, run_trs, args):
    n_vertices = fmri_continuous.shape[0]
    mask = _load_mask(mask_path, n_vertices)
    roi_binned = preprocess_fmri(
        fmri_continuous[mask, :], timing_df, run_trs, args.bin_sec, args.tr,
        args.delay_sec, skip_sec=args.skip_sec, normalize=True,
    )
    return roi_binned, int(mask.sum())


def _topo_summary(true_rho: np.ndarray, rand_rho_draws: np.ndarray) -> dict:
    pooled = rand_rho_draws.reshape(-1)
    return dict(
        true_rho_mean=float(true_rho.mean()), true_rho_std=float(true_rho.std()),
        random_rho_mean=float(pooled.mean()), random_rho_std=float(pooled.std()),
        mean_diff_true_minus_random=float(true_rho.mean() - pooled.mean()),
        true_beats_random=bool(true_rho.mean() > pooled.mean()),
    )


def run_topography_control(sheet_emb, seed_embeddings, results, tid, args,
                           run_bins, surf_idx, vertex_to_col) -> dict:
    """True k=100 neighbourhood vs. random same-tower k-unit samples, per
    seed, overall and broken down by tower. Reported honestly regardless of
    outcome -- see module docstring."""
    perm_idx_topo = within_run_shift_pair_indices(run_bins, args.topo_n_perm, args.seed)
    topo: dict = {}
    for seed_name, seed_emb in seed_embeddings.items():
        sheet_aligned, seed_aligned = align_and_assert_bins(sheet_emb, seed_emb)
        true_rho = results[f"{seed_name}_av"]["rho"]
        draws = np.empty((args.topo_n_draws, N_UNITS), dtype=np.float32)
        for d in range(args.topo_n_draws):
            draw_rng = np.random.default_rng(args.seed + 1000 * (d + 1))
            rand_neighbors = random_neighbors_within_tower(tid, args.k, draw_rng)
            rand_rho, _p, _n = _run_hemisphere(
                sheet_aligned, seed_aligned, rand_neighbors, surf_idx, vertex_to_col,
                args.method, perm_idx_topo, args.gpu_batch_size, args.perm_batch_size,
            )
            draws[d] = rand_rho
            log.info(f"  topography control {seed_name} draw {d}: "
                     f"random mean={rand_rho.mean():.4f}")
        overall = _topo_summary(true_rho, draws)
        by_tower = {TOWER_NAMES[t]: _topo_summary(true_rho[tid == t], draws[:, tid == t])
                   for t in sorted(TOWER_NAMES)}
        topo[seed_name] = dict(k=args.k, n_draws=args.topo_n_draws,
                               n_perm=args.topo_n_perm, overall=overall, by_tower=by_tower)
        log.info(f"  topography control {seed_name}: true={overall['true_rho_mean']:.4f} "
                 f"random={overall['random_rho_mean']:.4f} "
                 f"true_beats_random={overall['true_beats_random']}")
    return topo


def main() -> None:
    args = parse_args()
    if args.demo:
        demo()
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required (unless --demo)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = load_true_coords(Path(args.true_coords_cache))
    assert coords.shape == (N_UNITS, 2), coords.shape
    tid = tower_id()

    timing_df = pd.read_csv(args.timing_csv)
    run_trs = np.load(str(Path(args.preprocessed_dir) /
                          f"group_average_{args.fmri_suffix}_run_trs.npy"))
    cifti_path = (Path(args.preprocessed_dir) /
                 f"group_average_{args.fmri_suffix}_cortex_59k.dtseries.nii")

    log.info(f"Loading group-average fMRI: {cifti_path}")
    fmri_continuous = load_fmri_cifti(str(cifti_path))

    run_bins = get_run_bin_counts(
        timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec, args.skip_sec)
    perm_idx_all = within_run_shift_pair_indices(run_bins, args.n_perm, args.seed)

    seeds = {
        args.seed_a_name: Path(args.masks_dir) / args.seed_a_mask,
        args.seed_p_name: Path(args.masks_dir) / args.seed_p_mask,
    }
    seed_embeddings, seed_n_vertices = {}, {}
    for name, mask_path in seeds.items():
        emb, n_verts = _load_seed_embedding(mask_path, fmri_continuous, timing_df, run_trs, args)
        seed_embeddings[name] = emb
        seed_n_vertices[name] = n_verts
        log.info(f"  seed {name}: {n_verts} vertices -> {emb.shape}")
    del fmri_continuous

    bin_tag = f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"
    emb_path = Path(args.embeddings_dir) / MODEL_NAME / bin_tag / f"{MODEL_NAME}_av.npy"
    log.info(f"Loading full-sheet embeddings: {emb_path}")
    sheet_emb = process_model_embeddings(
        str(emb_path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
        run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
        model_norm="zscore",
    )
    assert sheet_emb.shape[1] == N_UNITS, f"expected {N_UNITS} units, got {sheet_emb.shape[1]}"
    log.info(f"  sheet embeddings: {sheet_emb.shape}")

    log.info(f"Building k={args.k} nearest-neighbour searchlights (cKDTree, {N_UNITS} units) ...")
    t0 = time.time()
    neighbors = knn_on_sheet(coords, args.k)
    log.info(f"  done in {time.time() - t0:.1f}s")
    surf_idx = np.arange(N_UNITS, dtype=np.int32)
    vertex_to_col = np.arange(N_UNITS, dtype=np.int32)

    # Both seeds share one brain RDM per unit (built from sheet_emb alone, not
    # the seed) and perm_idx_all is generated once above and reused for both,
    # so permutation j is the identical within-run circular shift for both --
    # the paired null the difference test needs. _run_hemisphere_paired
    # computes the shared brain RDM once per unit and derives both seeds' rho
    # plus the paired two-sided diff null in the same permutation batch,
    # rather than three independent passes.
    seed_a_name, seed_p_name = args.seed_a_name, args.seed_p_name
    sheet_al_a, seed_al_a = align_and_assert_bins(sheet_emb, seed_embeddings[seed_a_name])
    sheet_al_p, seed_al_p = align_and_assert_bins(sheet_emb, seed_embeddings[seed_p_name])
    assert np.array_equal(sheet_al_a, sheet_al_p), (
        "Shared-permutation paired null requires an identical sheet-side "
        "array for both seeds (same n_bins and values) -- got a mismatch."
    )
    sheet_aligned = sheet_al_a

    log.info(f"Running paired searchlight RSA: seeds={seed_a_name},{seed_p_name} "
             f"n_units={N_UNITS} gpu_batch_size={args.gpu_batch_size} "
             f"perm_batch_size={args.perm_batch_size}")
    t0 = time.time()
    actual_rho_a, p_perm_a, actual_rho_p, p_perm_p, p_perm_diff = _run_hemisphere_paired(
        sheet_aligned, seed_al_a, seed_al_p, neighbors, surf_idx, vertex_to_col,
        args.method, perm_idx_all, args.gpu_batch_size, args.perm_batch_size,
    )
    log.info(f"  done in {time.time() - t0:.1f}s")

    # Verify the observed rho reproduces the arrays already on disk before
    # overwriting them -- the null is recomputed here, but actual_rho should
    # not change.
    reproduction_check: dict = {}
    for name, new_rho in ((seed_a_name, actual_rho_a), (seed_p_name, actual_rho_p)):
        old_path = out_dir / f"{name}_av_rho.npy"
        if not old_path.exists():
            continue
        old_rho = np.load(old_path)
        max_abs_diff = float(np.max(np.abs(old_rho - new_rho)))
        close = bool(np.allclose(old_rho, new_rho, rtol=1e-4, atol=1e-6))
        reproduction_check[name] = dict(close=close, max_abs_diff=max_abs_diff)
        log.info(f"  reproduction check {name}: close={close} max_abs_diff={max_abs_diff:.3e}")
        if not close:
            raise SystemExit(
                f"Observed rho for {name} did NOT reproduce the on-disk array "
                f"{old_path} (max abs diff {max_abs_diff:.3e}). Stopping without "
                "overwriting -- investigate before rerunning."
            )

    results: dict[str, dict] = {}
    for seed_name, actual_rho, p_perm, seed_aligned in (
        (seed_a_name, actual_rho_a, p_perm_a, seed_al_a),
        (seed_p_name, actual_rho_p, p_perm_p, seed_al_p),
    ):
        p_fdr = stats.false_discovery_control(
            p_perm.astype(np.float64), method="bh").astype(np.float32)
        n_sig = int((p_fdr < args.fdr_alpha).sum())
        log.info(
            f"  {seed_name}: rho [{actual_rho.min():.3f}, {actual_rho.max():.3f}] "
            f"mean={actual_rho.mean():.3f}  n_sig_fdr={n_sig}/{N_UNITS}"
        )
        sanity = sanity_corr(sheet_aligned, seed_aligned)
        key = f"{seed_name}_av"
        results[key] = dict(rho=actual_rho, p_perm=p_perm, p_fdr=p_fdr,
                            n_sig_fdr=n_sig, sanity_corr=sanity)
        np.save(out_dir / f"{key}_rho.npy", actual_rho)
        pd.DataFrame({
            "unit_index": np.arange(N_UNITS),
            "tower": [TOWER_NAMES[t] for t in tid],
            "true_row": coords[:, 0], "true_col": coords[:, 1],
            "raster_row": np.arange(N_UNITS) // SHEET_COLS,
            "raster_col": np.arange(N_UNITS) % SHEET_COLS,
            "rho": actual_rho, "p_perm": p_perm, "p_fdr": p_fdr,
        }).to_csv(out_dir / f"{key}.csv", index=False)

    all_sig = all(v["n_sig_fdr"] / N_UNITS >= 0.98 for v in results.values())
    min_sig_frac = min(v["n_sig_fdr"] / N_UNITS for v in results.values())

    # Paired difference inference: per-unit two-sided p on rho(cca_a) -
    # rho(cca_p), from the paired null computed above; BH-FDR across all
    # N_UNITS. Kept separate from `results` (which characterize() and the
    # topography control key off of by seed name) to avoid touching their
    # schema.
    diff_rho = actual_rho_a - actual_rho_p
    p_fdr_diff = stats.false_discovery_control(
        p_perm_diff.astype(np.float64), method="bh").astype(np.float32)
    diff_sig_mask = p_fdr_diff < args.fdr_alpha
    n_sig_diff = int(diff_sig_mask.sum())
    frac_sig_diff = n_sig_diff / N_UNITS
    log.info(f"  diff (cca_a - cca_p): n_sig_fdr={n_sig_diff}/{N_UNITS} ({frac_sig_diff:.2%})")

    diff_by_tower = {}
    for t in sorted(TOWER_NAMES):
        m = tid == t
        diff_by_tower[TOWER_NAMES[t]] = dict(
            n_units=int(m.sum()), n_sig_fdr=int((diff_sig_mask & m).sum()),
            frac_sig_fdr=float((diff_sig_mask & m).sum() / m.sum()),
        )
        log.info(f"    {TOWER_NAMES[t]}: {diff_by_tower[TOWER_NAMES[t]]['n_sig_fdr']}/"
                 f"{diff_by_tower[TOWER_NAMES[t]]['n_units']} "
                 f"({diff_by_tower[TOWER_NAMES[t]]['frac_sig_fdr']:.2%})")

    diff_result = dict(rho=diff_rho, p_perm=p_perm_diff, p_fdr=p_fdr_diff,
                       n_sig_fdr=n_sig_diff, frac_sig_fdr=frac_sig_diff,
                       by_tower=diff_by_tower)
    np.save(out_dir / "diff_p_perm.npy", p_perm_diff)
    np.save(out_dir / "diff_p_fdr.npy", p_fdr_diff)
    pd.DataFrame({
        "unit_index": np.arange(N_UNITS),
        "tower": [TOWER_NAMES[t] for t in tid],
        "true_row": coords[:, 0], "true_col": coords[:, 1],
        "raster_row": np.arange(N_UNITS) // SHEET_COLS,
        "raster_col": np.arange(N_UNITS) % SHEET_COLS,
        "diff_rho": diff_rho, "p_perm": p_perm_diff, "p_fdr": p_fdr_diff,
    }).to_csv(out_dir / "diff.csv", index=False)

    if frac_sig_diff >= 0.98:
        diff_caveat = (
            f"The difference test is ALSO at ceiling ({frac_sig_diff:.1%} FDR-"
            "significant, p at the permutation floor for nearly every unit): for "
            "the same reason as the per-seed against-zero tests, near-ceiling "
            "significance here is a manipulation check, not a localization "
            "finding. Magnitude (per-tower diff_mean, hotspot composition) "
            "remains the informative signal."
        )
    elif frac_sig_diff < 0.5:
        diff_caveat = (
            f"The difference test is NOT at ceiling: {frac_sig_diff:.1%} of units "
            "are FDR-significant overall, and it is uneven by tower (audio "
            f"{diff_by_tower['audio']['frac_sig_fdr']:.1%}, vision "
            f"{diff_by_tower['vision']['frac_sig_fdr']:.1%}, thinker "
            f"{diff_by_tower['thinker']['frac_sig_fdr']:.1%}) -- this is "
            "informative: the a-p contrast is not uniformly significant "
            "across the sheet."
        )
    else:
        diff_caveat = (
            f"{frac_sig_diff:.1%} of units are FDR-significant on the a-p "
            "difference -- neither at ceiling nor sparse; see diff_inference."
            "by_tower for the distribution."
        )

    characterization = characterize(results, coords, tid, args)

    log.info("Running topography control (true k-NN vs. random same-tower draws) ...")
    topography_control = run_topography_control(
        sheet_emb, seed_embeddings, results, tid, args, run_bins, surf_idx, vertex_to_col)

    metadata = {
        "analysis": "full_sheet_rsa",
        "n_units": N_UNITS,
        "sheet_shape": [SHEET_ROWS, SHEET_COLS],
        "coordinate_system": "true (permute_coordinates, seed=42) -- a deterministic "
                             "seeded permutation of the raster lattice within each "
                             "architectural block, not a rotation. See "
                             "rsa/shared/sheet_rsa.py module docstring and "
                             "topoomni_true_coords_seed42_provenance.json",
        "true_coords_cache": args.true_coords_cache,
        "tower_scheme": (
            "vision encoder: rows 0-159, cols 0-255 (40,960 units). audio encoder: "
            "rows 0-159, cols 256-511 (40,960 units). thinker (language-model) stack: "
            "rows 160-303, all cols (73,728 units). Towers are from each unit's RASTER "
            "position, unaffected by the true-coordinate permutation (which only moves "
            "where a unit is plotted, not which architectural block produced it)."
        ),
        "modality": "av only -- no unimodal a/v ablations extracted for the full "
                   "sheet (see topo_omni_extract_full_sheet.py module docstring).",
        "k_neighbors": args.k, "n_perm": args.n_perm, "perm_seed": args.seed,
        "fdr_alpha": args.fdr_alpha,
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec,
        "delay_sec": args.delay_sec, "tr": args.tr, "method": args.method,
        "gpu_batch_size": args.gpu_batch_size, "perm_batch_size": args.perm_batch_size,
        "seeds": {name: {"mask_path": str(seeds[name]), "n_vertices": seed_n_vertices[name]}
                 for name in seeds},
        "n_bins": int(run_bins.sum()),
        "significance_caveat": (
            (f"Essentially every unit (>={min_sig_frac:.1%}, p at the permutation "
             "floor) was FDR-significant for every seed -- at ceiling. " if all_sig else
             f"Significance was meaningfully below ceiling for at least one seed "
             f"(minimum {min_sig_frac:.1%} FDR-significant). ") +
            "This is a manipulation check, not a finding: it shows a sheet unit's "
            "k-NN patch tracks the movie at all, not that any unit is special. "
            "Magnitude (rho_mean by tower, the A-P contrast, hotspot composition) "
            "is the informative signal."
        ),
        "results_summary": {
            key: {"rho_min": float(v["rho"].min()), "rho_max": float(v["rho"].max()),
                 "rho_mean": float(v["rho"].mean()), "n_sig_fdr": v["n_sig_fdr"],
                 "n_units": N_UNITS}
            for key, v in results.items()
        },
        "characterization": characterization,
        "topography_control": topography_control,
        "topography_control_interpretation": (
            "For each seed, true_rho_mean (true k-NN neighbourhood) vs. "
            "random_rho_mean (k=same, drawn uniformly from the same tower, "
            "coordinates ignored, topo_n_draws independent draws pooled). "
            "true_beats_random=false throughout is what spatial smoothness "
            "predicts, not evidence against topography: the sheet maps show "
            "clear spatial structure (smooth, labyrinthine, not scattered), and "
            "smoothness is exactly what makes a compact k=100 patch of mutually "
            "redundant neighbours span fewer independent dimensions than a "
            "scattered same-size sample -- so it scores lower by construction "
            "regardless of whether the true coordinates are meaningful. This "
            "control does not test topographic significance either way; a claim "
            "that the topography IS meaningful would need a different control "
            "(e.g. against sheets with the spatial loss ablated, or a spatially "
            "shuffled sheet preserving the marginal rho distribution)."
        ),
        "diff_inference": {
            "description": (
                "Per-unit paired two-sided permutation test on "
                "rho(cca_a) - rho(cca_p). perm_idx_all (within_run_shift_pair_"
                "indices, seed=42) is generated once and reused for both seeds "
                "(see main()), so permutation j applies the identical within-run "
                "circular shift to both -- null_diff_j(unit) = null_rho_a_j(unit) "
                "- null_rho_p_j(unit) is a genuinely paired null. p(unit) = "
                "(1 + #{j : |null_diff_j(unit)| >= |observed_diff(unit)|}) / "
                "(1 + n_perm); BH-FDR across all units."
            ),
            "n_perm": args.n_perm, "perm_seed": args.seed, "fdr_alpha": args.fdr_alpha,
            "n_units": N_UNITS, "n_sig_fdr": n_sig_diff, "frac_sig_fdr": frac_sig_diff,
            "by_tower": diff_by_tower,
            "reproduction_check": reproduction_check,
            "caveat": diff_caveat,
            "files": ["diff_p_perm.npy", "diff_p_fdr.npy", "diff.csv",
                     "diff_p_sheet_map.png", "diff_fdr_sig_sheet_map.png"],
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log.info(f"Saved metadata: {out_dir / 'metadata.json'}")

    make_figures(results, diff_result, coords, out_dir, args)
    log.info("Done.")


# =============================================================================
# Figures: one 304x512 sheet heatmap PNG per seed
# =============================================================================

def make_figures(results: dict, diff_result: dict, coords: np.ndarray, out_dir: Path,
                 args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_a, seed_p = args.seed_a_name, args.seed_p_name
    ra, rp = results[f"{seed_a}_av"], results[f"{seed_p}_av"]

    # Per-seed maps: sequential perceptually-uniform cmap, SHARED colour scale
    # -- robust (1st/99th pct) limits over the pooled rho of both seeds, so a
    # colour means the same rho in both panels (cca_p's weaker values are not
    # stretched to fill the same range as cca_a's stronger ones). rho is at
    # FDR ceiling here (>=99% of units significant for both seeds -- see
    # significance_caveat in metadata.json), so marking significant units
    # would paint over nearly the whole map; mark the rare NON-significant
    # units instead.
    pooled_rho = np.concatenate([ra["rho"], rp["rho"]])
    vmin, vmax = robust_vlim(pooled_rho, pct=1.0)
    for name, r in ((seed_a, ra), (seed_p, rp)):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = plot_sheet_map(ax, r["rho"], coords, f"{name} (av) full-sheet RSA rho",
                            vmin, vmax, r["p_fdr"] >= args.fdr_alpha, "viridis")
        fig.colorbar(im, ax=ax, shrink=0.8,
                    label=f"Spearman rho (scale shared across seeds, clipped to "
                          f"pooled 1st-99th pct: [{vmin:.3f}, {vmax:.3f}])")
        fig.savefig(out_dir / f"{name}_av_rho_sheet_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    diff = ra["rho"] - rp["rho"]
    _, vlim_diff = robust_vlim(np.abs(diff), pct=1.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = plot_sheet_map(ax, diff, coords, f"rho({seed_a}) - rho({seed_p})  [av]",
                        -vlim_diff, vlim_diff, None, "RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8,
                label=f"Delta rho (clipped to 99th pct |diff|: +/-{vlim_diff:.3f})")
    fig.savefig(out_dir / "diff_rho_sheet_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Paired-difference inference maps: uncorrected -log10(p) for readability
    # (transform stated in the colorbar label), and the FDR-significance map
    # signed by which seed wins (+1 cca_a, -1 cca_p, 0 not significant).
    eps = np.finfo(np.float32).tiny
    neglog_p = -np.log10(np.maximum(diff_result["p_perm"], eps))
    vmin_p, vmax_p = robust_vlim(neglog_p, pct=1.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = plot_sheet_map(ax, neglog_p, coords,
                        f"rho({seed_a}) - rho({seed_p}): paired two-sided permutation p, uncorrected",
                        vmin_p, vmax_p, None, "viridis")
    fig.colorbar(im, ax=ax, shrink=0.8,
                label=f"-log10(p) (clipped to 1st-99th pct: [{vmin_p:.2f}, {vmax_p:.2f}])")
    fig.savefig(out_dir / "diff_p_sheet_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sig_mask = diff_result["p_fdr"] < args.fdr_alpha
    sig_dir = np.where(sig_mask, np.sign(diff), 0.0).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = plot_sheet_map(
        ax, sig_dir, coords,
        f"rho({seed_a}) - rho({seed_p}): FDR-significant at alpha={args.fdr_alpha} "
        f"(n={diff_result['n_sig_fdr']}/{diff_result['rho'].size}, "
        f"{diff_result['frac_sig_fdr']:.1%})",
        -1, 1, None, "RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8,
                label=f"+1 = {seed_a} > {seed_p} sig., -1 = {seed_p} > {seed_a} sig., 0 = not FDR-significant")
    fig.savefig(out_dir / "diff_fdr_sig_sheet_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved figures under {out_dir}")


if __name__ == "__main__":
    main()
