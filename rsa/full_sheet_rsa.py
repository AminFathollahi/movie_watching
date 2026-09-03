#!/usr/bin/env python3
"""
rsa/full_sheet_rsa.py
======================
RSA between the two CCA seed ROIs and Topo-Omni's COMPLETE 304x512
(155,648-unit) cortical sheet -- the full encoder block (rows 0-159: vision
cols 0-255, audio cols 256-511, all 32+32 layers) plus the full 36-layer
decoder block (rows 160-303) -- under TRUE (trained, permute_coordinates
seed=42) coordinates.

Modeled on rsa/cca_seed_sheet_rsa_truecoords.py (READ FULLY before touching
that file -- it is untracked and irreplaceable; this script only imports
reusable pieces from it and from rsa/cca_seed_sheet_rsa.py, never edits
either).

Differences from cca_seed_sheet_rsa_truecoords.py
----------------------------------------------------
* Single embedding source: notebooks/feature_extraction/topo_omni_extract_
  full_sheet.py's topoomni_fullsheet_av.npy (n_bins, 155648). ALL 304 rows
  are already assembled there (encoder + all 36 decoder layers), not a
  per-layer stack of a handful of decoder layers picked out of TARGET_LAYERS.
  So `coords` here is simply the full true-coords LUT itself
  (topoomni_true_coords_seed42.npy), used index-for-index: the LUT was built
  as flat_k = i*512+j for i in range(304), j in range(512) (see
  cca_seed_sheet_rsa_truecoords.load_true_coords), which is EXACTLY this
  sheet's own flat unit ordering (see the extraction script's module
  docstring -- index = abs_row*512+col, standard row-major flatten). No
  `true_sheet_coords()` per-layer indexing indirection is needed or correct
  here.
* KNN: cca_seed_sheet_rsa.knn_on_sheet() is an O(n^2) brute-force cdist --
  fine at 2048-12,288 points, but a 155,648x155,648 distance matrix would
  need ~180 GB. Replaced here with scipy.spatial.cKDTree (O(n log n)), same
  semantics: k nearest neighbours by Euclidean distance in (row, col) space,
  self excluded, nearest-first.
* Modality: intact AV only. topo_omni_extract_full_sheet.py extracts no
  unimodal a/v passes (tripling cost was out of scope for the full-sheet
  deliverable), so there is no modality-preference / hotspot-by-modality
  section here -- only the A-vs-P contrast + hotspot-contiguity/overlap
  characterization (cca_seed_sheet_rsa_truecoords.characterize(), reused
  as-is) is computed.
* layer_id (for characterize()'s per-layer/per-region breakdown) is a
  synthetic region id computed from each unit's RASTER position (its fixed
  index in the saved array -- unaffected by the true-coordinate permutation,
  since that permutation only changes where a unit is PLOTTED, not which
  architectural layer produced it): vision layer l -> l (0-31), audio layer
  l -> 100+l (100-131), decoder layer l -> 200+l (200-235). Unlike the
  truecoords script's 6-layer subset, every decoder layer is present and
  contiguous here (200-235), so `characterize()`'s per_layer breakdown is a
  genuine full decoder-depth profile, not an isolated-band caveat.

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
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.perm_searchlight import _run_hemisphere, within_run_shift_pair_indices  # noqa: E402
from rsa.shared.rsa_utils import (  # noqa: E402
    align_and_assert_bins, get_run_bin_counts, load_fmri_cifti,
    preprocess_fmri, process_model_embeddings,
)
from cf_modeling.roi_mean_partial_connectivity import _load_mask  # noqa: E402
from rsa.cca_seed_sheet_rsa import _plot_sheet_map, _sanity_corr  # noqa: E402
from rsa.cca_seed_sheet_rsa_truecoords import load_true_coords, characterize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SHEET_ROWS = 304
SHEET_COLS = 512
N_UNITS = SHEET_ROWS * SHEET_COLS  # 155648
ENCODER_ROWS = 160
ENCODER_ROWS_PER_LAYER = 5
DECODER_ROWS_PER_LAYER = 4
MODEL_NAME = "topoomni_fullsheet"


# =============================================================================
# KNN (cKDTree -- see module docstring for why brute-force cdist is unusable here)
# =============================================================================

def knn_on_sheet_fast(coords: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(coords.astype(np.float64))
    _, idx = tree.query(coords.astype(np.float64), k=k + 1, workers=-1)
    return idx[:, 1:].astype(np.int32)  # drop self (column 0), keep k nearest


def region_layer_id() -> np.ndarray:
    """Synthetic per-unit region id from RASTER position (see module docstring):
    vision layer 0-31, audio layer 100-131, decoder layer 200-235."""
    k = np.arange(N_UNITS)
    row = k // SHEET_COLS
    col = k % SHEET_COLS
    layer_id = np.full(N_UNITS, -1, dtype=np.int32)
    enc = row < ENCODER_ROWS
    layer_in_block_enc = row[enc] // ENCODER_ROWS_PER_LAYER
    vision_mask = enc & (col < 256)
    audio_mask = enc & (col >= 256)
    layer_id[vision_mask] = (row[vision_mask] // ENCODER_ROWS_PER_LAYER)
    layer_id[audio_mask] = 100 + (row[audio_mask] // ENCODER_ROWS_PER_LAYER)
    dec = ~enc
    layer_id[dec] = 200 + ((row[dec] - ENCODER_ROWS) // DECODER_ROWS_PER_LAYER)
    assert (layer_id >= 0).all()
    return layer_id


def demo() -> None:
    """Self-check: KDTree-KNN against brute force on a tiny lattice, and the
    region_layer_id() geometry (vision/audio/decoder block boundaries)."""
    coords2 = np.array([(r, c) for r in range(2) for c in range(4)])
    nn = knn_on_sheet_fast(coords2, k=2)
    assert set(nn[0].tolist()) == {1, 4}, nn[0]

    lut_cache = Path("/home/amin/Research/Representation/Movie/outputs/rsa/"
                     "cca_seed_sheet_rsa/topoomni_true_coords_seed42.npy")
    lut = load_true_coords(lut_cache)
    assert lut.shape == (N_UNITS, 2)

    lid = region_layer_id()
    assert lid.shape == (N_UNITS,)
    assert set(np.unique(lid[:100 * 512]).tolist()) <= set(range(32)) | set(range(100, 132))
    assert set(lid[ENCODER_ROWS * SHEET_COLS:].tolist()) == set(range(200, 236))
    # spot-check: unit at raster (row=232, col=0) is decoder layer 18, row-in-layer 0
    k0 = 232 * SHEET_COLS + 0
    assert lid[k0] == 218, lid[k0]
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

    data_base = "/home/amin/Research/Representation/Movie/data"
    outputs_base = "/home/amin/Research/Representation/Movie/outputs"

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


def main() -> None:
    args = parse_args()
    if args.demo:
        demo()
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required (unless --demo)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    true_coords_lut = load_true_coords(Path(args.true_coords_cache))
    assert true_coords_lut.shape == (N_UNITS, 2), true_coords_lut.shape
    coords = true_coords_lut  # index-for-index match with this sheet's flat ordering
    layer_id = region_layer_id()

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
        normalize=True,
    )
    assert sheet_emb.shape[1] == N_UNITS, f"expected {N_UNITS} units, got {sheet_emb.shape[1]}"
    log.info(f"  sheet embeddings: {sheet_emb.shape}")

    log.info(f"Building k={args.k} nearest-neighbour searchlights (cKDTree, {N_UNITS} units) ...")
    t0 = time.time()
    neighbors = knn_on_sheet_fast(coords, args.k)
    log.info(f"  done in {time.time() - t0:.1f}s")
    surf_idx = np.arange(N_UNITS, dtype=np.int32)
    vertex_to_col = np.arange(N_UNITS, dtype=np.int32)

    results: dict[str, dict] = {}
    for seed_name, seed_emb in seed_embeddings.items():
        sheet_aligned, seed_aligned = align_and_assert_bins(sheet_emb, seed_emb)
        log.info(f"Running searchlight RSA: seed={seed_name} n_units={N_UNITS} "
                 f"gpu_batch_size={args.gpu_batch_size} perm_batch_size={args.perm_batch_size}")
        t0 = time.time()
        actual_rho, p_perm, _null_max = _run_hemisphere(
            sheet_aligned, seed_aligned, neighbors, surf_idx, vertex_to_col,
            args.method, perm_idx_all, args.gpu_batch_size, args.perm_batch_size,
        )
        log.info(f"  done in {time.time() - t0:.1f}s")
        p_fdr = stats.false_discovery_control(
            p_perm.astype(np.float64), method="bh").astype(np.float32)
        n_sig = int((p_fdr < args.fdr_alpha).sum())
        log.info(
            f"  {seed_name}: rho [{actual_rho.min():.3f}, {actual_rho.max():.3f}] "
            f"mean={actual_rho.mean():.3f}  n_sig_fdr={n_sig}/{N_UNITS}"
        )
        sanity_corr = _sanity_corr(sheet_aligned, seed_aligned)
        key = f"{seed_name}_av"
        results[key] = dict(rho=actual_rho, p_perm=p_perm, p_fdr=p_fdr,
                            n_sig_fdr=n_sig, sanity_corr=sanity_corr)
        np.save(out_dir / f"{key}_rho.npy", actual_rho)
        pd.DataFrame({
            "unit_index": np.arange(N_UNITS),
            "layer_id": layer_id,
            "true_row": coords[:, 0], "true_col": coords[:, 1],
            "raster_row": np.arange(N_UNITS) // SHEET_COLS,
            "raster_col": np.arange(N_UNITS) % SHEET_COLS,
            "rho": actual_rho, "p_perm": p_perm, "p_fdr": p_fdr,
        }).to_csv(out_dir / f"{key}.csv", index=False)

    all_sig = all(v["n_sig_fdr"] / N_UNITS >= 0.98 for v in results.values())
    min_sig_frac = min(v["n_sig_fdr"] / N_UNITS for v in results.values())

    class _Args:
        seed_a_name = args.seed_a_name
        seed_p_name = args.seed_p_name
        seed = args.seed
        pref_top_decile = 0.1

    characterization = characterize(results, coords, layer_id, _Args())

    metadata = {
        "analysis": "full_sheet_rsa",
        "n_units": N_UNITS,
        "sheet_shape": [SHEET_ROWS, SHEET_COLS],
        "coordinate_system": "true (permute_coordinates, seed=42) -- see "
                             "cca_seed_sheet_rsa_truecoords.py module docstring "
                             "and topoomni_true_coords_seed42_provenance.json",
        "true_coords_cache": args.true_coords_cache,
        "region_layer_id_scheme": (
            "vision layer l -> l (0-31), audio layer l -> 100+l (100-131), "
            "decoder layer l -> 200+l (200-235). All 36 decoder layers and both "
            "full 32-layer encoder towers are present and each block is "
            "internally contiguous -- unlike cca_seed_sheet_rsa_truecoords.py's "
            "6-layer decoder subset, there are no isolated un-extracted-neighbour "
            "bands within any single block. Cross-block adjacency (row 159/160, "
            "vision-audio column boundary at col 255/256) is architectural, not "
            "a claim about cortex."
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
            "This is a manipulation check, not a finding -- see rsa/cca_seed_sheet_rsa.py "
            "module docstring. Magnitude (rho_mean, the A-P contrast, hotspot "
            "composition by region_layer_id) is the informative signal."
        ),
        "results_summary": {
            key: {"rho_min": float(v["rho"].min()), "rho_max": float(v["rho"].max()),
                 "rho_mean": float(v["rho"].mean()), "n_sig_fdr": v["n_sig_fdr"],
                 "n_units": N_UNITS}
            for key, v in results.items()
        },
        "characterization": characterization,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log.info(f"Saved metadata: {out_dir / 'metadata.json'}")

    make_figures(results, coords, out_dir, args)
    log.info("Done.")


# =============================================================================
# Figures: one 304x512 sheet heatmap PNG per seed
# =============================================================================

def make_figures(results: dict, coords: np.ndarray, out_dir: Path,
                 args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caveat = (
        "Full 304x512 sheet, TRUE (trained) coordinates. Rows 0-159: encoder "
        "(vision cols 0-255, audio cols 256-511, 32 layers each x 5 rows). "
        "Rows 160-303: all 36 decoder layers x 4 rows."
    )
    seed_a, seed_p = args.seed_a_name, args.seed_p_name
    ra, rp = results[f"{seed_a}_av"], results[f"{seed_p}_av"]
    vlim = float(max(np.abs(ra["rho"]).max(), np.abs(rp["rho"]).max()))

    for name, r in ((seed_a, ra), (seed_p, rp)):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = _plot_sheet_map(ax, r["rho"], coords, f"{name} (av) full-sheet RSA rho",
                             vlim, r["p_fdr"] < args.fdr_alpha, "RdBu_r")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
        fig.suptitle(caveat, fontsize=7.5, y=1.02)
        fig.savefig(out_dir / f"{name}_av_rho_sheet_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    diff = ra["rho"] - rp["rho"]
    vlim_diff = float(np.abs(diff).max())
    fig, ax = plt.subplots(figsize=(10, 6))
    im = _plot_sheet_map(ax, diff, coords, f"rho({seed_a}) - rho({seed_p})  [av]",
                         vlim_diff, None, "RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta rho")
    fig.suptitle(caveat, fontsize=7.5)
    fig.savefig(out_dir / "diff_rho_sheet_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved figures under {out_dir}")


if __name__ == "__main__":
    main()
