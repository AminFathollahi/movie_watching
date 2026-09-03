#!/usr/bin/env python3
"""
rsa/cca_seed_sheet_rsa_truecoords.py
=====================================
TRUE-coordinate correction + multi-layer extension of rsa/cca_seed_sheet_rsa.py.

Why this script exists
-----------------------
rsa/cca_seed_sheet_rsa.py plotted Topo-Omni sheet units at their RASTER
identity position: `qwen2_5_omni.py`'s `load_positions()` falls back to
`[(i,j) for i in range(304) for j in range(512)]` when no `coords.npy` is
present, and that raster lattice is what got used. But that fallback is NOT
the coordinate system the released checkpoint was actually trained under.
`Model Repos/topo-omni/src/init_coords.py` builds the real `coords.npy` once
at init time by taking that same raster lattice and randomly permuting it
(`permute_coordinates()`, `torch.Generator` seeded 42) within fixed
(row, col) blocks -- vision encoder rows 0-159/cols 0-255, audio encoder
rows 0-159/cols 256-511 (5 rows per "layer" each), thinker/decoder rows
160-303/cols 0-511 (3 rows per "layer") -- and `train.yml` (`apply-spatial-
loss: true`, `position-dir: neighborhoods/model=qwen2-5omni-3b-unified-20-
v3_radius=20_neighborhoods=20_coords=42`) confirms this seed=42 file is what
the spatial-smoothness loss (Chebyshev radius=20 neighbourhoods) that gives
the sheet its topography was actually computed against.

No authoritative coords.npy could be recovered: `huggingface_hub.
list_repo_files('epfl-neuroai/topo-omni')` lists only weights/tokenizer/
config, no `neighborhoods/` or `coords.npy`; the local HF cache is an empty
stub; no external drive is attached to this machine (checked `lsblk`,
`df -h`, `ls /media/amin` -- the `/media/amin/EXTERNAL_USB` path referenced
by `topo_omni_extract.py` is not mounted); and no copy exists anywhere else
on this machine. So `outputs/rsa/cca_seed_sheet_rsa/
topoomni_true_coords_seed42.npy` is a DETERMINISTIC REGENERATION of it,
reproducing `permute_coordinates()` bug-for-bug (including its thinker-block
3-rows-per-layer grouping, which does NOT line up with the true 4-rows/layer
decoder packing verified in the sibling script's `demo()` -- this is
reproduced as shipped, not "fixed", because whatever the repo's own script
produced is what the released checkpoint trained under). Validated by
generating it independently in two conda envs with different torch builds
(2.9.1+cu128 and 2.11.0+cu128, both CPU-generator, both deterministic within-
process) and confirming the two (155648, 2) arrays are bit-identical -- see
`topoomni_true_coords_seed42_provenance.json` next to the .npy for the full
methodology and the empirical raster-vs-true displacement numbers for layer
18 (mean move ~170 units, true rows 232-237 vs raster's 232-235, 1/2048
units unmoved).

Multi-layer, and why it's only partial
----------------------------------------
Because `permute_coordinates()` only shuffles WITHIN each local block, a
unit's true position never leaves the neighbourhood of its own raster
identity -- it does not teleport across the sheet. Of Topo-Omni's 36
decoder layers, only 6 were ever extracted with genuinely-separate audio-
only/video-only forward passes (`topo_omni_extract_intact.py`'s
TARGET_LAYERS = [1, 9, 18, 27, 34, 35] -- verified against that script; the
bare-named `topoomni_layer2_sheet`/`topoomni_layer4_sheet` files on disk are
NOT in that list and were never touched by the intact extraction, so their
"_a"/"_v" are the OLD masked-mean-pool-of-a-single-joint-pass ("_av" =
mean(_a,_v) exactly), not genuine ablations -- excluded here for that
reason). Of those 6, only 34 and 35 are adjacent in the true 36-layer stack
(raster rows 296-299 / 300-303); 1, 9, 18, 27 remain isolated bands even
under true coordinates because their real architectural neighbours (layers
0/2, 8/10, 17/19, 26/28) were never extracted. So a k=100 neighbourhood here
gets genuinely richer LOCAL 2D structure within each extracted layer's own
window (from the scrambling) plus ONE genuine cross-layer neighbourhood
(34/35) -- not full continuous 2D coverage of the decoder stack. Report this
plainly; don't oversell "multi-layer" as solving the topology problem
everywhere.

Encoder block (rows 0-159, split by column into vision/audio halves by
`permute_coordinates()`'s own block definition -- directly relevant to "what
does this region care about") is NOT included. The only encoder-side
embedding on disk, `topoomni_encoder_penultimate` (626, 1280), is a flat
pooled vector, not per-sheet-unit CorticalAdaptor activations -- it cannot
be placed on the sheet lattice without a new extraction run (hooking the
vision/audio encoder's own CorticalAdaptor Z output, analogous to what
`topo_omni_extract_intact.py` already does for decoder layers). Not
attempted here; flagged as future work requiring a new extraction pass.

Everything else (seeds, RDM/searchlight machinery, modality-preference
methodology, FDR-is-a-manipulation-check caveat) is identical to
rsa/cca_seed_sheet_rsa.py -- this script imports its reusable pieces rather
than re-deriving them.

Usage
-----
  # Single layer under TRUE coordinates -- directly comparable to the raster
  # k=100 run in outputs/rsa/cca_seed_sheet_rsa/k100/ (same 2048 units, same
  # k; only the coordinate system differs):
  python rsa/cca_seed_sheet_rsa_truecoords.py --layers 18 --k 100 \\
      --output-dir <out>/k100_layer18_truecoords

  # Multi-layer (primary deliverable): stack all 6 genuinely-unimodal
  # decoder layers into their true coordinates, search neighbourhoods across
  # the combined set:
  python rsa/cca_seed_sheet_rsa_truecoords.py --layers 1,9,18,27,34,35 \\
      --k 100 --output-dir <out>/k100_multilayer_truecoords

  python rsa/cca_seed_sheet_rsa_truecoords.py --demo
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.perm_searchlight import _run_hemisphere, within_run_shift_pair_indices  # noqa: E402
from rsa.shared.rsa_utils import (  # noqa: E402
    align_and_assert_bins, get_run_bin_counts, load_fmri_cifti,
    preprocess_fmri, process_model_embeddings,
)
from cf_modeling.roi_mean_partial_connectivity import _load_mask  # noqa: E402
from rsa.cca_seed_sheet_rsa import (  # noqa: E402
    SHEET_COLS, VISUAL_AUDIO_ROWS, sheet_coords, knn_on_sheet,
    _plot_sheet_map, _sanity_corr, _unit_permutation_test,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOPO_REPO = "/home/amin/Research/Representation/Movie/Model Repos/topo-omni"
MP_TAGGED_LAYERS = {9, 18, 27, 34}  # matches topo_omni_extract_intact.py exactly
GENUINE_UNIMODAL_LAYERS = {1, 9, 18, 27, 34, 35}  # TARGET_LAYERS of that script


def _sheet_model_name(layer_idx: int) -> str:
    suffix = "_mp" if layer_idx in MP_TAGGED_LAYERS else ""
    return f"topoomni_layer{layer_idx}_sheet{suffix}"


# =============================================================================
# True coordinates
# =============================================================================

def load_true_coords(cache_path: Path) -> np.ndarray:
    """(155648, 2) int64 true (row, col) per raster flat index; see the module
    docstring and the provenance JSON saved next to `cache_path` for how this
    was obtained (repo ships no coords.npy; this is a validated deterministic
    regeneration of Model Repos/topo-omni/src/init_coords.py's algorithm)."""
    if cache_path.exists():
        return np.load(cache_path)
    log.warning(f"{cache_path} not found -- regenerating (seed=42, CPU, "
                "deterministic; see module docstring)")
    import torch
    sys.path.insert(0, TOPO_REPO)
    sys.path.insert(0, f"{TOPO_REPO}/src")
    import init_coords as ic  # type: ignore
    ic.N_col = SHEET_COLS
    rng = torch.Generator()
    rng.manual_seed(42)
    coordinates = torch.Tensor([(i, j) for i in range(304) for j in range(SHEET_COLS)])
    coordinates = ic.permute_coordinates(coordinates, rng)
    arr = coordinates.numpy().astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def true_sheet_coords(layer_idx: int, n_units: int, true_coords_lut: np.ndarray) -> np.ndarray:
    """True (permuted) (row, col) for one decoder layer's units."""
    raster = sheet_coords(layer_idx, n_units)
    flat_k = raster[:, 0] * SHEET_COLS + raster[:, 1]
    return true_coords_lut[flat_k]


def demo() -> None:
    """Self-check: true-coords lookup shape/range sanity + the layer18
    raster-vs-true displacement numbers quoted in the module docstring."""
    cache = Path("/home/amin/Research/Representation/Movie/outputs/rsa/"
                 "cca_seed_sheet_rsa/topoomni_true_coords_seed42.npy")
    lut = load_true_coords(cache)
    assert lut.shape == (304 * SHEET_COLS, 2)

    raster = sheet_coords(18, 2048)
    true = true_sheet_coords(18, 2048, lut)
    assert set(raster[:, 0].tolist()) == {232, 233, 234, 235}
    dist = np.linalg.norm((true - raster).astype(np.float64), axis=1)
    unmoved = int((dist == 0).sum())
    assert unmoved <= 5, f"expected near-total scrambling, got {unmoved} unmoved"
    assert true[:, 0].min() >= 228 and true[:, 0].max() <= 240, "true rows should stay LOCAL to raster block"

    # Layers 34/35 adjacency in raster space (the one genuine cross-layer pair)
    r34 = sheet_coords(34, 2048)[:, 0]
    r35 = sheet_coords(35, 2048)[:, 0]
    assert r34.max() + 1 == r35.min()
    print(f"demo OK (layer18: {unmoved}/2048 unmoved, true rows "
          f"{true[:,0].min()}-{true[:,0].max()})")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRUE-coordinate / multi-layer RSA between CCA seed ROIs "
                    "and Topo-Omni's cortical sheet.",
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
    p.add_argument("--layers", default="1,9,18,27,34,35",
                   help="Comma-separated decoder layer indices to stack (must "
                        "be a subset of the genuinely-unimodal set "
                        f"{sorted(GENUINE_UNIMODAL_LAYERS)}).")
    p.add_argument("--true-coords-cache",
                   default=f"{outputs_base}/rsa/cca_seed_sheet_rsa/"
                            "topoomni_true_coords_seed42.npy")
    p.add_argument("--modalities", default="av,a,v")
    p.add_argument("--output-dir", default=None,
                   help="Required unless --demo.")

    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--pref-top-decile", type=float, default=0.1)
    p.add_argument("--pref-n-perm", type=int, default=10000)
    p.add_argument("--gpu-batch-size", type=int, default=512)
    p.add_argument("--perm-batch-size", type=int, default=100)
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

    layers = sorted(int(x) for x in args.layers.split(",") if x.strip())
    bad = set(layers) - GENUINE_UNIMODAL_LAYERS
    if bad:
        raise ValueError(f"Layers {sorted(bad)} are not in the genuinely-unimodal "
                         f"set {sorted(GENUINE_UNIMODAL_LAYERS)} (topo_omni_extract_"
                         "intact.py's TARGET_LAYERS) -- their a/v are not real "
                         "ablations, refusing to use them for modality preference.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true_coords_lut = load_true_coords(Path(args.true_coords_cache))

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

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    bin_tag = f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"

    # ── Stack layers: embeddings (626, sum_units), true coords, layer_id ──────
    emb_by_modality: dict[str, list[np.ndarray]] = {m: [] for m in modalities}
    coords_parts, layer_id_parts, unit_in_layer_parts = [], [], []
    for layer_idx in layers:
        model_name = _sheet_model_name(layer_idx)
        n_units = None
        for modality in modalities:
            emb_path = (Path(args.embeddings_dir) / model_name / bin_tag /
                       f"{model_name}_{modality}.npy")
            sheet_emb = process_model_embeddings(
                str(emb_path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
                run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
                normalize=True,
            )
            emb_by_modality[modality].append(sheet_emb)
            n_units = sheet_emb.shape[1]
        coords_parts.append(true_sheet_coords(layer_idx, n_units, true_coords_lut))
        layer_id_parts.append(np.full(n_units, layer_idx, dtype=np.int32))
        unit_in_layer_parts.append(np.arange(n_units, dtype=np.int32))
        log.info(f"  layer {layer_idx} ({model_name}): {n_units} units")

    coords = np.concatenate(coords_parts, axis=0)
    layer_id = np.concatenate(layer_id_parts, axis=0)
    unit_in_layer = np.concatenate(unit_in_layer_parts, axis=0)
    n_total_units = coords.shape[0]
    log.info(f"Combined sheet: {n_total_units} units from layers {layers}, "
            f"true row range [{coords[:,0].min()}, {coords[:,0].max()}]")

    neighbors = knn_on_sheet(coords, args.k)
    surf_idx = np.arange(n_total_units, dtype=np.int32)
    vertex_to_col = np.arange(n_total_units, dtype=np.int32)

    results: dict[str, dict] = {}
    for modality in modalities:
        sheet_emb = np.concatenate(emb_by_modality[modality], axis=1)  # (626, n_total_units)
        for seed_name, seed_emb in seed_embeddings.items():
            sheet_aligned, seed_aligned = align_and_assert_bins(sheet_emb, seed_emb)
            actual_rho, p_perm, _null_max = _run_hemisphere(
                sheet_aligned, seed_aligned, neighbors, surf_idx, vertex_to_col,
                args.method, perm_idx_all, args.gpu_batch_size, args.perm_batch_size,
            )
            p_fdr = stats.false_discovery_control(
                p_perm.astype(np.float64), method="bh").astype(np.float32)
            n_sig = int((p_fdr < args.fdr_alpha).sum())
            log.info(
                f"  {seed_name}/{modality}: rho [{actual_rho.min():.3f}, {actual_rho.max():.3f}] "
                f"mean={actual_rho.mean():.3f}  n_sig_fdr={n_sig}/{n_total_units}"
            )
            sanity_corr = _sanity_corr(sheet_aligned, seed_aligned)
            key = f"{seed_name}_{modality}"
            results[key] = dict(rho=actual_rho, p_perm=p_perm, p_fdr=p_fdr,
                                n_sig_fdr=n_sig, sanity_corr=sanity_corr)
            np.save(out_dir / f"{key}_rho.npy", actual_rho)
            pd.DataFrame({
                "unit_index": np.arange(n_total_units),
                "layer": layer_id, "unit_in_layer": unit_in_layer,
                "true_row": coords[:, 0], "true_col": coords[:, 1],
                "rho": actual_rho, "p_perm": p_perm, "p_fdr": p_fdr,
            }).to_csv(out_dir / f"{key}.csv", index=False)

    # >=99.9% FDR-significant is "at ceiling" for this caveat's purpose -- a
    # handful of units missing (e.g. 12286/12288) is not evidence significance
    # was NOT at ceiling; only report the softer wording below that.
    all_sig = all(v["n_sig_fdr"] / n_total_units >= 0.999 for v in results.values())

    # ── Cross-seed / hotspot / layer-composition characterization ───────────
    characterization = characterize(results, coords, layer_id, args)

    # ── Modality preference of hotspots (per seed) ───────────────────────────
    modality_pref = None
    if {"av", "a", "v"}.issubset(set(modalities)):
        modality_pref = modality_preference_analysis(
            results, coords, layer_id, out_dir, args)

    metadata = {
        "analysis": "cca_seed_sheet_rsa_truecoords",
        "layers": layers,
        "n_total_units": n_total_units,
        "coordinate_system": "true (permute_coordinates, seed=42) -- see module "
                             "docstring and topoomni_true_coords_seed42_provenance.json",
        "true_coords_cache": args.true_coords_cache,
        "layer18_included_alone": layers == [18],
        "adjacent_layer_pairs_in_this_set": [
            [a, b] for a, b in zip(layers, layers[1:]) if b - a == 1
        ],
        "sheet_neighbourhood_caveat": (
            "permute_coordinates() only shuffles WITHIN local blocks, so a unit's "
            "true position stays near its own raster identity -- it does not "
            "teleport across the sheet. Only adjacent extracted layers (listed "
            "in adjacent_layer_pairs_in_this_set) can share a genuine k-NN "
            "neighbourhood; isolated layers remain their own local band even "
            "under true coordinates. Coordinates are still not a claim about "
            "cortex -- they are the model's own trained UNIT-TO-UNIT layout, "
            "recovered via a validated deterministic regeneration (no "
            "authoritative coords.npy exists on this machine or on HF)."
        ),
        "encoder_block_caveat": (
            "Vision/audio encoder rows (0-159) are not included. The only "
            "encoder-side embedding on disk (topoomni_encoder_penultimate, "
            "626x1280) is a flat pooled vector, not per-sheet-unit activations, "
            "so it cannot be placed on this lattice without a new extraction "
            "run (hooking the encoder's CorticalAdaptor Z output). Not attempted."
        ),
        "excluded_layers_2_4_caveat": (
            "topoomni_layer2_sheet / topoomni_layer4_sheet exist on disk but are "
            "excluded: they were never touched by topo_omni_extract_intact.py "
            "(not in its TARGET_LAYERS), so their _a/_v are the OLD masked-mean-"
            "pool-of-a-single-joint-pass (not genuine unimodal ablations)."
        ),
        "k_neighbors": args.k, "n_perm": args.n_perm, "perm_seed": args.seed,
        "fdr_alpha": args.fdr_alpha,
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec,
        "delay_sec": args.delay_sec, "tr": args.tr, "method": args.method,
        "modalities": modalities,
        "seeds": {name: {"mask_path": str(seeds[name]), "n_vertices": seed_n_vertices[name]}
                 for name in seeds},
        "n_bins": int(run_bins.sum()),
        "significance_caveat": (
            ("Essentially every unit (>=99.9%, p at the permutation floor) was "
             "FDR-significant for every seed x modality condition -- at ceiling "
             "(see results_summary for the handful of exceptions, if any). " if all_sig else
             "Significance was meaningfully below ceiling for at least one "
             "condition (see results_summary). ") +
            "This is a manipulation check, not a finding -- see rsa/cca_seed_sheet_rsa.py "
            "module docstring. Magnitude (rho_mean, the A-P contrast, hotspot "
            "composition) is the informative signal."
        ),
        "results_summary": {
            key: {"rho_min": float(v["rho"].min()), "rho_max": float(v["rho"].max()),
                 "rho_mean": float(v["rho"].mean()), "n_sig_fdr": v["n_sig_fdr"],
                 "n_units": n_total_units}
            for key, v in results.items()
        },
        "characterization": characterization,
    }
    if modality_pref is not None:
        metadata["modality_preference"] = modality_pref
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log.info(f"Saved metadata: {out_dir / 'metadata.json'}")

    if "av" in modalities:
        make_figures(results, coords, layer_id, out_dir, args)
    if modality_pref is not None:
        make_modality_preference_figures(results, coords, out_dir, args, layer_id=layer_id)

    log.info("Done.")


# =============================================================================
# Characterization: A/P difference, hotspot contiguity, hotspot overlap
# =============================================================================

def _mean_nn_dist(xy: np.ndarray) -> float:
    """Mean nearest-OTHER-point Euclidean distance, via cKDTree (O(n log n)).
    Equivalent to cdist(xy, xy) + fill_diagonal(inf) + min(axis=1) + mean(),
    which is O(n^2) memory and did not scale to the 155,648-unit full sheet.
    """
    from scipy.spatial import cKDTree
    return float(cKDTree(xy).query(xy, k=2)[0][:, 1].mean())


def characterize(results: dict, coords: np.ndarray, layer_id: np.ndarray,
                 args: argparse.Namespace) -> dict:
    seed_a, seed_p = args.seed_a_name, args.seed_p_name
    a, p = results[f"{seed_a}_av"]["rho"], results[f"{seed_p}_av"]["rho"]
    n = a.size
    diff = a - p
    cross_r = float(np.corrcoef(a, p)[0, 1])
    max_i, min_i = int(np.argmax(diff)), int(np.argmin(diff))

    def _loc(i):
        return {"unit_index": i, "layer": int(layer_id[i]),
               "true_row": int(coords[i, 0]), "true_col": int(coords[i, 1])}

    per_layer = {}
    for L in sorted(set(layer_id.tolist())):
        m = layer_id == L
        per_layer[str(L)] = {
            f"{seed_a}_av_rho_mean": float(a[m].mean()),
            f"{seed_p}_av_rho_mean": float(p[m].mean()),
            "n_units": int(m.sum()),
        }

    rng = np.random.default_rng(args.seed)
    hotspots, top_masks = {}, {}
    for name, rho in ((seed_a, a), (seed_p, p)):
        n_top = int(np.ceil(args.pref_top_decile * n))
        mask = np.zeros(n, dtype=bool)
        mask[np.argsort(rho)[::-1][:n_top]] = True
        top_masks[name] = mask

        # Layer composition of the hotspot vs. each layer's overall share.
        layer_counts = {str(L): int(((layer_id == L) & mask).sum())
                        for L in sorted(set(layer_id.tolist()))}

        # Spatial contiguity: mean nearest-OTHER-hotspot-unit distance in true
        # coord space, vs. a null of random same-size unit subsets (same
        # unit-identity permutation logic as _unit_permutation_test).
        top_xy = coords[mask].astype(np.float64)
        observed_nn = _mean_nn_dist(top_xy)
        n_top_n = int(mask.sum())
        null_nn = np.empty(2000, dtype=np.float64)
        all_xy = coords.astype(np.float64)
        for i in range(2000):
            sel = rng.permutation(n)[:n_top_n]
            null_nn[i] = _mean_nn_dist(all_xy[sel])
        p_contig = float((np.sum(null_nn <= observed_nn) + 1) / (2000 + 1))

        hotspots[name] = dict(
            n_top=n_top_n, layer_composition=layer_counts,
            observed_mean_nn_dist=observed_nn,
            null_mean_nn_dist_mean=float(null_nn.mean()),
            null_mean_nn_dist_p5=float(np.quantile(null_nn, 0.05)),
            p_more_contiguous_than_random=p_contig,
            interpretation=(
                "p < 0.05 means hotspot units sit closer together (in true "
                "coord space) than a random same-size subset of all units -- "
                "i.e. genuinely spatially clustered, not scattered."
            ),
        )

    overlap_n = int((top_masks[seed_a] & top_masks[seed_p]).sum())
    union_n = int((top_masks[seed_a] | top_masks[seed_p]).sum())
    jaccard = overlap_n / union_n if union_n else 0.0
    # Null for overlap: two independent random same-size subsets.
    n_top_n = int(top_masks[seed_a].sum())
    null_overlap = np.empty(2000, dtype=np.int64)
    for i in range(2000):
        s1 = rng.permutation(n)[:n_top_n]
        s2 = rng.permutation(n)[:n_top_n]
        null_overlap[i] = np.intersect1d(s1, s2, assume_unique=True).size
    p_overlap = float((np.sum(null_overlap >= overlap_n) + 1) / (2000 + 1))

    return dict(
        cross_seed_spatial_pearson_r=cross_r,
        diff_mean=float(diff.mean()), diff_std=float(diff.std()),
        diff_positive_fraction=float((diff > 0).mean()),
        diff_max_loc=_loc(max_i), diff_max_value=float(diff[max_i]),
        diff_min_loc=_loc(min_i), diff_min_value=float(diff[min_i]),
        rho_av_mean_by_layer=per_layer,
        hotspots=hotspots,
        hotspot_overlap={
            "n_top_decile": n_top_n, "overlap_n": overlap_n, "union_n": union_n,
            "jaccard": jaccard, "p_more_overlap_than_random": p_overlap,
            "interpretation": (
                "p < 0.05 means cca_a's and cca_p's top-decile units share "
                "more sheet units than expected by chance -- the two seeds' "
                "hotspots are not independent."
            ),
        },
    )


# =============================================================================
# Modality preference (identical methodology to cca_seed_sheet_rsa.py, with a
# per-layer composition breakdown added since units now span multiple layers)
# =============================================================================

def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z CI for a correlation coefficient (valid for Pearson r; used as
    the standard approximation for Spearman rho too, se scaled by 1.06)."""
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def continuous_preference_correlation(av: np.ndarray, pref: np.ndarray,
                                      layer_id: np.ndarray) -> dict:
    """The continuous replacement for the top-decile-vs-all framing: does a
    unit's overall fit to the seed (av_rho) predict its audio/video
    preference (a_rho - v_rho), across ALL units, not just a thresholded
    top group? Pooling across layers is itself confounded when >1 layer is
    present -- av_rho and pref both vary systematically by layer (depth), so
    a naive pooled correlation can even flip sign relative to the true
    within-layer relationship (Simpson's paradox). Reports: naive pooled
    (labelled as confounded when >1 layer), per-layer, and a layer-partialled
    version (regress both variables on layer dummies, correlate residuals) --
    the layer-partialled number is the one that isolates "does fit quality
    predict modality preference" from "later layers differ in both average
    fit and average preference"."""
    n = av.size
    r_pool, p_pool = stats.pearsonr(av, pref)
    rho_pool, p_rho_pool = stats.spearmanr(av, pref)
    r_lo, r_hi = _fisher_ci(r_pool, n)
    rho_lo, rho_hi = _fisher_ci(rho_pool, n)
    out = dict(
        n_units=int(n),
        pooled_pearson_r=float(r_pool), pooled_pearson_ci95=[r_lo, r_hi],
        pooled_pearson_p=float(p_pool),
        pooled_spearman_rho=float(rho_pool), pooled_spearman_ci95=[rho_lo, rho_hi],
        pooled_spearman_p=float(p_rho_pool),
        pooled_is_layer_confounded=bool(len(set(layer_id.tolist())) > 1),
    )
    layers = sorted(set(layer_id.tolist()))
    if len(layers) > 1:
        per_layer = {}
        for L in layers:
            m = layer_id == L
            r, p = stats.pearsonr(av[m], pref[m])
            rho, ps = stats.spearmanr(av[m], pref[m])
            per_layer[str(L)] = dict(n=int(m.sum()), pearson_r=float(r), pearson_p=float(p),
                                     spearman_rho=float(rho), spearman_p=float(ps))
        out["per_layer"] = per_layer

        dummies = pd.get_dummies(pd.Series(layer_id), drop_first=True).astype(float).values
        X = np.column_stack([np.ones(n), dummies])
        def _resid(y):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        av_resid, pref_resid = _resid(av.astype(np.float64)), _resid(pref.astype(np.float64))
        r_part, p_part = stats.pearsonr(av_resid, pref_resid)
        out["layer_partialled_pearson_r"] = float(r_part)
        out["layer_partialled_pearson_p"] = float(p_part)
        out["layer_partialled_note"] = (
            "Correlation of av_rho and pref after regressing out layer-mean "
            "differences from both (dummy-coded OLS residuals) -- isolates the "
            "within-layer unit-level relationship from the confounding "
            "layer-level trend visible in per_layer and pooled_*."
        )
    return out


def modality_preference_analysis(results: dict, coords: np.ndarray, layer_id: np.ndarray,
                                 out_dir: Path, args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    summary: dict = {}
    for seed_name in (args.seed_a_name, args.seed_p_name):
        av = results[f"{seed_name}_av"]["rho"]
        a_rho = results[f"{seed_name}_a"]["rho"]
        v_rho = results[f"{seed_name}_v"]["rho"]
        pref = a_rho - v_rho

        n_top = int(np.ceil(args.pref_top_decile * av.size))
        top_idx = np.argsort(av)[::-1][:n_top]
        top_mask = np.zeros(av.size, dtype=bool)
        top_mask[top_idx] = True

        observed, p_val = _unit_permutation_test(pref, top_mask, args.pref_n_perm, rng)
        continuous = continuous_preference_correlation(av, pref, layer_id)

        pd.DataFrame({
            "unit_index": np.arange(av.size), "layer": layer_id,
            "true_row": coords[:, 0], "true_col": coords[:, 1],
            "av_rho": av, "a_rho": a_rho, "v_rho": v_rho,
            "modality_preference_a_minus_v": pref,
            "is_top_decile_by_av_rho": top_mask,
        }).to_csv(out_dir / f"{seed_name}_modality_preference.csv", index=False)

        layer_pref = {str(L): float(pref[layer_id == L].mean())
                     for L in sorted(set(layer_id.tolist()))}

        summary[seed_name] = dict(
            continuous_correlation=continuous,
            continuous_correlation_note=(
                "PRIMARY: does av_rho (a unit's overall fit to this seed) predict "
                "its audio/video preference index (a_rho - v_rho), across ALL "
                "units? See continuous.layer_partialled_pearson_r when >1 layer "
                "(the confound-corrected number); continuous.pooled_pearson_r is "
                "confounded by cross-layer trends in that case."
            ),
            # Top-decile-vs-all is kept only as a secondary/legacy summary -- an
            # arbitrary threshold on a continuous quantity; continuous_correlation
            # above is the primary characterization.
            n_units=int(av.size), n_top_decile=n_top,
            a_rho_mean_all=float(a_rho.mean()), a_rho_mean_top_decile=float(a_rho[top_mask].mean()),
            v_rho_mean_all=float(v_rho.mean()), v_rho_mean_top_decile=float(v_rho[top_mask].mean()),
            pref_mean_all=float(pref.mean()), pref_mean_top_decile=float(pref[top_mask].mean()),
            observed_diff_top_minus_all=observed,
            null_hypothesis="unit-identity permutation",
            n_perm=args.pref_n_perm, p_value_two_sided=p_val,
            pref_mean_by_layer=layer_pref,
        )
        log.info(f"  modality preference {seed_name}: continuous pooled pearson r="
                f"{continuous['pooled_pearson_r']:+.3f} "
                + (f"layer-partialled r={continuous.get('layer_partialled_pearson_r', float('nan')):+.3f} "
                   if 'layer_partialled_pearson_r' in continuous else "")
                + f"| [legacy] pref_top={pref[top_mask].mean():.3f} vs pref_all={pref.mean():.3f} "
                f"diff={observed:.4f} p={p_val:.4g}")

    (out_dir / "modality_preference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


# =============================================================================
# Figures
# =============================================================================

def make_figures(results: dict, coords: np.ndarray, layer_id: np.ndarray,
                 out_dir: Path, args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caveat = (
        "TRUE (trained) coordinates, not raster identity -- see module docstring. "
        "Blank rows = un-extracted decoder layers / encoder block."
    )
    seed_a, seed_p = args.seed_a_name, args.seed_p_name
    ra, rp = results[f"{seed_a}_av"], results[f"{seed_p}_av"]
    vlim = float(max(np.abs(ra["rho"]).max(), np.abs(rp["rho"]).max()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im0 = _plot_sheet_map(axes[0], ra["rho"], coords, f"{seed_a} (av) RSA rho",
                          vlim, ra["p_fdr"] < args.fdr_alpha, "RdBu_r")
    _plot_sheet_map(axes[1], rp["rho"], coords, f"{seed_p} (av) RSA rho",
                    vlim, rp["p_fdr"] < args.fdr_alpha, "RdBu_r")
    fig.colorbar(im0, ax=axes, shrink=0.8, label="Spearman rho")
    fig.suptitle(caveat, fontsize=7.5, y=1.02)
    fig.savefig(out_dir / "seed_rho_sheet_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    diff = ra["rho"] - rp["rho"]
    vlim_diff = float(np.abs(diff).max())
    fig, ax = plt.subplots(figsize=(10, 4))
    im = _plot_sheet_map(ax, diff, coords, f"rho({seed_a}) - rho({seed_p})  [av]",
                         vlim_diff, None, "RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta rho")
    fig.suptitle(caveat, fontsize=7.5)
    fig.savefig(out_dir / "diff_rho_sheet_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved figures under {out_dir}")


def make_modality_preference_figures(results: dict, coords: np.ndarray, out_dir: Path,
                                     args: argparse.Namespace, layer_id: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caption = ("TRUE coordinates. Circled = top decile by intact-av rho (legacy "
              "reference only -- see continuous trend line/stats, the primary "
              "characterization). See README for the isolated-vs-adjacent-layer caveat.")
    seeds = (args.seed_a_name, args.seed_p_name)
    prefs, avs, tops = {}, {}, {}
    for name in seeds:
        avs[name] = results[f"{name}_av"]["rho"]
        prefs[name] = results[f"{name}_a"]["rho"] - results[f"{name}_v"]["rho"]
        n_top = int(np.ceil(args.pref_top_decile * avs[name].size))
        mask = np.zeros(avs[name].size, dtype=bool)
        mask[np.argsort(avs[name])[::-1][:n_top]] = True
        tops[name] = mask

    vlim = float(max(np.abs(prefs[seeds[0]]).max(), np.abs(prefs[seeds[1]]).max()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im0 = _plot_sheet_map(axes[0], prefs[seeds[0]], coords,
                          f"{seeds[0]}: rho_a - rho_v", vlim, tops[seeds[0]], "RdBu_r")
    _plot_sheet_map(axes[1], prefs[seeds[1]], coords,
                    f"{seeds[1]}: rho_a - rho_v", vlim, tops[seeds[1]], "RdBu_r")
    fig.colorbar(im0, ax=axes, shrink=0.8, label="rho_a - rho_v")
    fig.suptitle(caption, fontsize=7.5, y=1.02)
    fig.savefig(out_dir / "modality_preference_sheet_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, seeds):
        m = tops[name]
        av_, pref_ = avs[name], prefs[name]
        multi_layer = layer_id is not None and len(set(layer_id.tolist())) > 1
        if multi_layer:
            cmap = plt.get_cmap("viridis")
            layers_sorted = sorted(set(layer_id.tolist()))
            colors = {L: cmap(i / max(len(layers_sorted) - 1, 1)) for i, L in enumerate(layers_sorted)}
            for L in layers_sorted:
                lm = layer_id == L
                ax.scatter(av_[lm], pref_[lm], s=5, alpha=0.35, color=colors[L], label=f"layer {L}")
        else:
            ax.scatter(av_[~m], pref_[~m], s=6, alpha=0.3, color="gray", label="other")
            ax.scatter(av_[m], pref_[m], s=12, alpha=0.85, color="crimson",
                      label=f"top decile ({int(m.sum())}, legacy)")
        b, a0 = np.polyfit(av_, pref_, 1)
        xs = np.linspace(av_.min(), av_.max(), 50)
        ax.plot(xs, a0 + b * xs, color="black", lw=1.5, ls="--", label="OLS trend (pooled)")
        r_pool, _ = stats.pearsonr(av_, pref_)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("intact av rho"); ax.set_ylabel("rho_a - rho_v")
        title = f"{name}  (pooled pearson r={r_pool:+.2f}"
        if multi_layer:
            title += ", CONFOUNDED by layer -- see layer-partialled r in JSON"
        title += ")"
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(caption, fontsize=7.5)
    fig.savefig(out_dir / "modality_preference_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved modality-preference figures under {out_dir}")


if __name__ == "__main__":
    main()
