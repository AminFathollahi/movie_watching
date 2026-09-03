#!/usr/bin/env python3
"""
rsa/cca_seed_sheet_rsa.py
==========================
RSA between two CCA cortical seed ROIs (audio-preferring / video-preferring)
and Topo-Omni's layer-18 cortical-sheet units, plotted in the sheet's true
2D layout.

Sheet geometry (verified against Model Repos/topo-omni/src/models/qwen2_5_omni.py)
------------------------------------------------------------------------------
`unified_sheet = cat([visual+audio sheet (160 rows x 512 cols),
multimodal_cortical_sheet (144 rows x 512 cols)], dim=1)` -> [T, 304, 512].
The text-decoder block is built as (line ~1245):
    multimodal_cortical_sheet.permute(1, 0, 2).reshape(-1, 144, 512)
from a stack of shape [L=36 layers, T, D=2048]. permute(1,0,2) on [L,T,D]
gives [T,L,D] (not [T,D,L] despite the source comment -- verified directly,
see demo() below); reshape merges the trailing (L,D) into (144,512) in
C-order, so for a fixed layer l and unit d in [0,2048):
    flat        = l * D + d
    row_in_block = flat // 512      (0..143)
    col          = flat % 512       (0..511)
    abs_row      = 160 + row_in_block
Since D=2048=4*512, one layer occupies exactly 4 contiguous absolute rows.
Layer 18 (0-indexed, matching the `cortical_adaptors[18]` forward-hook used
by topo_omni_extract_intact.py -- extraction hooks each CorticalAdaptor's
raw Z output directly, bypassing load_positions()/unified_sheet entirely)
occupies abs_row in {232, 233, 234, 235}, col = d % 512.

Coordinate provenance: NO trained coords.npy exists anywhere reachable --
~/.cache/huggingface/hub/models--epfl-neuroai--topo-omni is an empty 4KB
stub, the external USB cache (/media/amin/EXTERNAL_USB, referenced in
topo_omni_extract.py) is not attached to this machine, and the topo-omni
repo itself ships no coords.npy. load_positions() in qwen2_5_omni.py falls
back to this exact same raster lattice `[(i,j) for i in range(304) for j in
range(512)]` when coords.npy is absent -- but it doesn't even matter here,
because our extraction pipeline never calls load_positions() at all. The
(row, col) used below is therefore the model's fixed ARCHITECTURAL raster
placement, not a trained/optimized topographic coordinate.

Model side: a sheet unit alone gives a degenerate one-feature-per-bin RDM
(see rsa/topoomni_sheet_localizer.py docstring), so each of the 2048 layer-18
units is scored via its k nearest neighbours in (row, col) sheet space
(626 x k patch -> 626x626 RDM), Spearman-correlated against the seed ROI's
own 626x626 brain RDM. This maps exactly onto the existing GPU searchlight
machinery in rsa/perm_searchlight.py: sheet units play the role of surface
"vertices" (searchlight neighbourhoods), the seed ROI's per-bin vertex
pattern plays the role of the fixed "model" RDM.

CAVEAT -- what the sheet neighbourhood is NOT (read before calling any of
this "topographic"): layer 18's 2048 units occupy exactly 4 contiguous
absolute rows (232-235) with col = d % 512. A k-NN neighbourhood confined to
this 4-row x 512-col slab runs out of row space almost immediately, so for
any k used here (k=50 or k=100) the neighbourhood is effectively a 1-D band
running along the column axis, not a 2-D patch. Restricting neighbourhoods to
one layer's rows is also not how the model's own lattice is organised: in the
real 304x512 unified_sheet, adjacent decoder layers occupy neighbouring rows,
so a unit's true architectural neighbours include units from layers 17 and 19
that this analysis never looks at. Combined with the raster-lattice coordinate
caveat above (not a trained coords.npy), nothing here licenses a claim that
the sheet has a learned 2-D topographic organisation -- only that nearby
column positions within layer 18's row band behave similarly.

CAVEAT -- FDR significance here is a manipulation check, not a finding: at
k=50, all 2048/2048 units were FDR-significant (p at the permutation floor,
1/(n_perm+1)) for every seed x modality combination. That only shows every
sheet unit's k-NN patch tracks the movie at all (unsurprising for a model
that watched/heard the whole thing), not that any unit is special. The
informative comparisons are magnitude ones: rho_mean by seed, rho_mean by
modality condition (av/a/v), and the spatial correlation of the rho map
across k. If k=100 reproduces the same all-significant pattern, that is
reported as uninformative, not re-litigated as a positive result.

Modality characterization (added on top of the k=50 RSA)
----------------------------------------------------------
Each modality condition against which the sheet is scored comes from
notebooks/feature_extraction/topo_omni_extract_intact.py's `topoomni_layer18
_sheet_mp_{av,a,v}.npy`: "av" is the intact joint audiovisual forward pass
(masked mean-pool over the joint sequence's audio+video token positions);
"a" and "v" are GENUINELY SEPARATE unimodal forward passes (audio-only input
with no video frames at all / video-only input with no audio at all), not a
post-hoc split of the joint "av" pass. So rho_a and rho_v measure how well a
sheet unit's activity pattern (elicited with the OTHER modality entirely
absent from the input) tracks the seed ROI, and `rho_a - rho_v` is a
defensible per-unit audio-vs-video preference index for characterizing what
the seed's best-corresponding sheet units respond to. `modality_preference_
analysis()` below computes this index, takes each seed's top decile of
sheet units by intact "av" rho, and compares their preference-index
distribution against all 2048 units using a permutation test OVER UNIT
IDENTITY (not the usual 626-bin block bootstrap used elsewhere in this
repo, e.g. cf_modeling/channel_cca_analysis.py's `_block_indices`): rho
values here are already scalars per sheet unit, collapsed over all 626 bins
inside the RSA searchlight itself, so the quantity being tested -- does
av-rho-based top-decile membership predict a unit's audio/video mix? -- is a
statement about units, not about time. Resampling bins again would ask
whether the rho values are numerically stable, a different question already
answered by the FDR permutation p-values. Resampling units (draw random
same-size subsets of the 2048 units, ask how often their mean preference is
as extreme as the observed top-decile's) directly targets the unit-identity
question and needs no re-run of the GPU searchlight.

Usage
-----
  python rsa/cca_seed_sheet_rsa.py \\
      --preprocessed-dir <path> --fmri-suffix raw --timing-csv <path> \\
      --embeddings-dir <path> --masks-dir <path> --output-dir <path> \\
      --sheet-model topoomni_layer18_sheet_mp --k 50 --n-perm 1000 \\
      --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0

  python rsa/cca_seed_sheet_rsa.py --demo
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.perm_searchlight import _run_hemisphere, within_run_shift_pair_indices  # noqa: E402
from rsa.shared.rsa_utils import (  # noqa: E402
    align_and_assert_bins, get_run_bin_counts, load_fmri_cifti,
    preprocess_fmri, process_model_embeddings,
)
from cf_modeling.roi_mean_partial_connectivity import _load_mask  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Architectural constants of the Topo-Omni unified sheet (see module docstring).
SHEET_COLS = 512
VISUAL_AUDIO_ROWS = 160
TEXT_ROWS = 144
N_DECODER_LAYERS = 36


# =============================================================================
# Sheet geometry
# =============================================================================

def sheet_coords(layer_idx: int, n_units: int) -> np.ndarray:
    """(n_units, 2) int array of (abs_row, col) for one decoder layer's units."""
    if n_units % SHEET_COLS != 0:
        raise ValueError(f"n_units={n_units} is not a multiple of {SHEET_COLS}")
    flat_local = layer_idx * n_units + np.arange(n_units)
    row_in_block = flat_local // SHEET_COLS
    col = flat_local % SHEET_COLS
    abs_row = VISUAL_AUDIO_ROWS + row_in_block
    return np.stack([abs_row, col], axis=1).astype(np.int64)


def knn_on_sheet(coords: np.ndarray, k: int) -> np.ndarray:
    """Brute-force Euclidean k-NN among sheet units (2048 points -> trivial)."""
    d = cdist(coords.astype(np.float64), coords.astype(np.float64))
    np.fill_diagonal(d, np.inf)
    part = np.argpartition(d, k, axis=1)[:, :k]
    part_d = np.take_along_axis(d, part, axis=1)
    order = np.argsort(part_d, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def demo() -> None:
    """Self-check: sheet-coordinate arithmetic against the real permute/reshape,
    and the k-NN lookup against a tiny hand-verifiable lattice."""
    import torch

    L, D = N_DECODER_LAYERS, 2048
    src = torch.arange(L * D, dtype=torch.float64).reshape(L, 1, D)  # [L, T=1, D]
    reshaped = src.permute(1, 0, 2).reshape(-1, TEXT_ROWS, SHEET_COLS)[0]  # (144, 512)
    assert reshaped.shape == (TEXT_ROWS, SHEET_COLS)

    for l, d in [(0, 0), (18, 0), (18, 511), (18, 512), (18, 2047), (35, 2047), (17, 2047)]:
        marker = l * D + d
        coords = sheet_coords(l, D)
        abs_row, col = coords[d]
        row_in_block = abs_row - VISUAL_AUDIO_ROWS
        assert reshaped[row_in_block, col].item() == marker, (l, d, row_in_block, col)

    # Layer 17's last row must sit immediately above layer 18's first row.
    assert sheet_coords(17, D)[-1, 0] == sheet_coords(18, D)[0, 0] - 1
    # Layer 18 occupies exactly abs rows 232-235.
    assert set(sheet_coords(18, D)[:, 0].tolist()) == {232, 233, 234, 235}

    # k-NN sanity on a tiny 2x4 lattice: point 0 = (0,0)'s 2 nearest are its
    # row/col neighbours (0,1)=idx1 and (1,0)=idx4, both at distance 1.
    coords2 = np.array([(r, c) for r in range(2) for c in range(4)])
    nn = knn_on_sheet(coords2, k=2)
    assert set(nn[0].tolist()) == {1, 4}

    print("demo OK")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RSA between CCA seed ROIs and Topo-Omni's cortical sheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--demo", action="store_true", help="Run self-check and exit.")

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
    p.add_argument("--sheet-model", default="topoomni_layer18_sheet_mp")
    p.add_argument("--modalities", default="av,a,v",
                   help="Comma-separated modality suffixes to run (av is primary).")
    p.add_argument("--output-dir", default=f"{outputs_base}/rsa/cca_seed_sheet_rsa")

    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--k", type=int, default=50,
                   help="Sheet-lattice neighbourhood size (20-100 range, "
                        "matching the spirit of the k=100 brain searchlight).")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--pref-top-decile", type=float, default=0.1,
                   help="Fraction of units (by intact-av rho) treated as the "
                        "'top-corresponding' group in the modality-preference "
                        "characterization.")
    p.add_argument("--pref-n-perm", type=int, default=10000,
                   help="Unit-identity permutations for the modality-preference test.")
    p.add_argument("--gpu-batch-size", type=int, default=512)
    p.add_argument("--perm-batch-size", type=int, default=100)
    return p.parse_args()


# =============================================================================
# Core analysis
# =============================================================================

def _load_seed_embedding(mask_path: Path, fmri_continuous: np.ndarray,
                         timing_df: pd.DataFrame, run_trs: np.ndarray,
                         args: argparse.Namespace) -> tuple[np.ndarray, int]:
    n_vertices = fmri_continuous.shape[0]
    mask = _load_mask(mask_path, n_vertices)
    roi_binned = preprocess_fmri(
        fmri_continuous[mask, :], timing_df, run_trs, args.bin_sec, args.tr,
        args.delay_sec, skip_sec=args.skip_sec, normalize=True,
    )
    return roi_binned, int(mask.sum())


def _sanity_corr(sheet_emb: np.ndarray, seed_emb: np.ndarray) -> np.ndarray:
    """Plain Pearson r between the seed ROI's mean time series and each unit."""
    seed_mean = seed_emb.mean(axis=1)
    seed_z = (seed_mean - seed_mean.mean()) / (seed_mean.std() + 1e-12)
    unit_z = (sheet_emb - sheet_emb.mean(axis=0, keepdims=True)) / (
        sheet_emb.std(axis=0, keepdims=True) + 1e-12)
    return (unit_z * seed_z[:, None]).mean(axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.demo:
        demo()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_match = re.search(r"layer(\d+)", args.sheet_model)
    if layer_match is None:
        raise ValueError(f"Could not parse decoder layer index from {args.sheet_model!r}")
    layer_idx = int(layer_match.group(1))

    timing_df = pd.read_csv(args.timing_csv)
    run_trs = np.load(str(Path(args.preprocessed_dir) /
                          f"group_average_{args.fmri_suffix}_run_trs.npy"))
    cifti_path = (Path(args.preprocessed_dir) /
                 f"group_average_{args.fmri_suffix}_cortex_59k.dtseries.nii")

    log.info(f"Loading group-average fMRI: {cifti_path}")
    fmri_continuous = load_fmri_cifti(str(cifti_path))
    log.info(f"  fMRI continuous: {fmri_continuous.shape}")

    run_bins = get_run_bin_counts(
        timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec, args.skip_sec)
    log.info(f"  run bins: {run_bins.tolist()} (total {int(run_bins.sum())})")
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

    results: dict[str, dict] = {}
    coords_by_modality: dict[str, np.ndarray] = {}

    for modality in modalities:
        emb_path = (Path(args.embeddings_dir) / args.sheet_model / bin_tag /
                    f"{args.sheet_model}_{modality}.npy")
        log.info(f"Modality {modality}: {emb_path}")
        sheet_emb = process_model_embeddings(
            str(emb_path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
            run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
            normalize=True,
        )
        n_units = sheet_emb.shape[1]
        coords = sheet_coords(layer_idx, n_units)
        coords_by_modality[modality] = coords
        neighbors = knn_on_sheet(coords, args.k)
        surf_idx = np.arange(n_units, dtype=np.int32)
        vertex_to_col = np.arange(n_units, dtype=np.int32)

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
                f"mean={actual_rho.mean():.3f}  n_sig_fdr={n_sig}/{n_units}"
            )

            sanity_corr = _sanity_corr(sheet_aligned, seed_aligned)

            key = f"{seed_name}_{modality}"
            results[key] = dict(
                rho=actual_rho, p_perm=p_perm, p_fdr=p_fdr, n_sig_fdr=n_sig,
                coords=coords, sanity_corr=sanity_corr, n_units=n_units,
            )

            np.save(out_dir / f"{key}_rho.npy", actual_rho)
            pd.DataFrame({
                "unit_index": np.arange(n_units),
                "sheet_row": coords[:, 0],
                "sheet_col": coords[:, 1],
                "rho": actual_rho,
                "p_perm": p_perm,
                "p_fdr": p_fdr,
            }).to_csv(out_dir / f"{key}.csv", index=False)

    # ── Metadata ────────────────────────────────────────────────────────────
    metadata = {
        "analysis": "cca_seed_sheet_rsa",
        "sheet_geometry": {
            "sheet_cols": SHEET_COLS,
            "visual_audio_rows": VISUAL_AUDIO_ROWS,
            "text_rows": TEXT_ROWS,
            "n_decoder_layers": N_DECODER_LAYERS,
            "layer_idx": layer_idx,
            "abs_row_range": [
                int(coords_by_modality[modalities[0]][:, 0].min()),
                int(coords_by_modality[modalities[0]][:, 0].max()),
            ],
            "coordinate_provenance": (
                "Default architectural raster lattice, NOT a trained coords.npy. "
                "No coords.npy found: ~/.cache/huggingface/hub/models--epfl-neuroai--topo-omni "
                "is an empty stub, the external USB cache referenced by "
                "notebooks/feature_extraction/topo_omni_extract.py "
                "(/media/amin/EXTERNAL_USB/SMAF/hf_models/hub) is not attached to this machine, "
                "and the topo-omni repo ships no coords.npy. qwen2_5_omni.py's load_positions() "
                "falls back to this exact raster lattice when coords.npy is absent -- but it is "
                "moot here: our extraction (topo_omni_extract_intact.py) hooks each decoder "
                "layer's CorticalAdaptor Z output directly and never calls load_positions() or "
                "assembles unified_sheet at all. (row, col) below are derived purely from the "
                "permute(1,0,2).reshape(-1,144,512) arithmetic in qwen2_5_omni.py, verified "
                "against the real op in demo()."
            ),
        },
        "k_neighbors": args.k,
        "k_justification": (
            f"k={args.k}, within the requested 20-100 range: comparable in order of magnitude "
            "to sqrt(2048)~45 (a common searchlight-size heuristic relative to total unit "
            "count), large enough for a stable 626-bin RDM, small enough to keep spatial "
            "localization on the sheet meaningful given only 4 rows are populated per layer."
        ),
        "sheet_neighbourhood_caveat": (
            "Layer 18's 2048 units occupy exactly 4 contiguous absolute rows (232-235), "
            f"col = d % 512. A k={args.k}-NN neighbourhood in this 4-row x 512-col slab runs "
            "out of row space almost immediately, so it is effectively a 1-D band along the "
            "column axis, not a 2-D cortical patch. Restricting neighbourhoods to one layer's "
            "rows is also not the model's own neighbourhood structure: in the real 304x512 "
            "unified_sheet lattice, adjacent decoder layers (17, 19) sit in neighbouring rows "
            "and would be true architectural neighbours of layer-18 units, but this analysis "
            "never looks at them. Combined with coordinate_provenance above (a fixed raster "
            "lattice, not a trained coords.npy), nothing here licenses a 'learned 2-D "
            "topographic organization' claim -- only that nearby column positions within "
            "layer 18's row band behave similarly."
        ),
        "n_perm": args.n_perm,
        "perm_seed": args.seed,
        "fdr_alpha": args.fdr_alpha,
        "bin_sec": args.bin_sec, "skip_sec": args.skip_sec,
        "delay_sec": args.delay_sec, "tr": args.tr, "method": args.method,
        "sheet_model": args.sheet_model,
        "modalities": modalities,
        "seeds": {
            name: {"mask_path": str(seeds[name]), "n_vertices": seed_n_vertices[name]}
            for name in seeds
        },
        "n_bins": int(run_bins.sum()),
        "input_paths": {
            "preprocessed_dir": args.preprocessed_dir,
            "fmri_cifti": str(cifti_path),
            "timing_csv": args.timing_csv,
            "embeddings_dir": args.embeddings_dir,
            "masks_dir": args.masks_dir,
        },
        "results_summary": {
            key: {
                "rho_min": float(v["rho"].min()), "rho_max": float(v["rho"].max()),
                "rho_mean": float(v["rho"].mean()), "n_sig_fdr": v["n_sig_fdr"],
                "n_units": v["n_units"],
            }
            for key, v in results.items()
        },
    }

    all_sig = all(v["n_sig_fdr"] == v["n_units"] for v in results.values())
    metadata["significance_caveat"] = (
        ("Every unit was FDR-significant (p at the permutation floor) for every seed x "
         "modality condition at this k. " if all_sig else
         "Significance was NOT uniformly at ceiling at this k (see results_summary). ") +
        "FDR significance here is a manipulation check -- it shows a sheet unit's k-NN "
        "patch tracks the movie at all, not that any unit is special. The informative "
        "comparisons are magnitude ones: rho_mean by seed, rho_mean by modality condition, "
        "and cross-k spatial correlation of the rho map (see README)."
    )

    # ── Modality characterization: what do the top-corresponding units care about? ──
    if {"av", "a", "v"}.issubset(set(modalities)):
        coords_ref = coords_by_modality["av"]
        metadata["modality_preference"] = modality_preference_analysis(
            results, coords_ref, out_dir, args)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log.info(f"Saved metadata: {out_dir / 'metadata.json'}")

    # ── Figures (av modality, the primary deliverable) ────────────────────
    if "av" in modalities:
        make_figures(results, out_dir, args)
    if {"av", "a", "v"}.issubset(set(modalities)):
        make_modality_preference_figures(results, coords_by_modality["av"], out_dir, args)

    log.info("Done.")


# =============================================================================
# Figures
# =============================================================================

def _canvas(values: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    rmin, rmax = int(coords[:, 0].min()), int(coords[:, 0].max())
    canvas = np.full((rmax - rmin + 1, SHEET_COLS), np.nan, dtype=np.float32)
    canvas[coords[:, 0] - rmin, coords[:, 1]] = values
    return canvas, rmin, rmax


def _plot_sheet_map(ax, values: np.ndarray, coords: np.ndarray, title: str,
                    vlim: float, fdr_mask: np.ndarray | None, cmap: str) -> None:
    canvas, rmin, rmax = _canvas(values, coords)
    im = ax.imshow(canvas, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim,
                    extent=[0, SHEET_COLS, rmax + 0.5, rmin - 0.5])
    if fdr_mask is not None and fdr_mask.any():
        rows = coords[fdr_mask, 0]
        cols = coords[fdr_mask, 1]
        ax.scatter(cols + 0.5, rows + 0.5, s=4, facecolors="none",
                   edgecolors="black", linewidths=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sheet col")
    ax.set_ylabel("sheet row (abs)")
    return im


def make_figures(results: dict, out_dir: Path, args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_a, seed_p = args.seed_a_name, args.seed_p_name
    key_a, key_p = f"{seed_a}_av", f"{seed_p}_av"
    ra, rp = results[key_a], results[key_p]
    coords = ra["coords"]

    vlim_rho = float(max(np.abs(ra["rho"]).max(), np.abs(rp["rho"]).max()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    im0 = _plot_sheet_map(axes[0], ra["rho"], coords, f"{seed_a} (av) RSA rho",
                          vlim_rho, ra["p_fdr"] < args.fdr_alpha, "RdBu_r")
    im1 = _plot_sheet_map(axes[1], rp["rho"], coords, f"{seed_p} (av) RSA rho",
                          vlim_rho, rp["p_fdr"] < args.fdr_alpha, "RdBu_r")
    fig.colorbar(im0, ax=axes, shrink=0.8, label="Spearman rho")
    fig.savefig(out_dir / "seed_rho_sheet_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    diff = ra["rho"] - rp["rho"]
    vlim_diff = float(np.abs(diff).max())
    fig, ax = plt.subplots(figsize=(10, 3))
    im = _plot_sheet_map(ax, diff, coords, f"rho({seed_a}) - rho({seed_p})  [av]",
                         vlim_diff, None, "RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta rho")
    fig.savefig(out_dir / "diff_rho_sheet_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    vlim_sanity = float(max(np.abs(ra["sanity_corr"]).max(), np.abs(rp["sanity_corr"]).max()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    im0 = _plot_sheet_map(axes[0], ra["sanity_corr"], coords,
                          f"{seed_a} (av) mean-TS Pearson r (sanity)", vlim_sanity, None, "PuOr_r")
    im1 = _plot_sheet_map(axes[1], rp["sanity_corr"], coords,
                          f"{seed_p} (av) mean-TS Pearson r (sanity)", vlim_sanity, None, "PuOr_r")
    fig.colorbar(im0, ax=axes, shrink=0.8, label="Pearson r")
    fig.savefig(out_dir / "seed_sanity_corr_sheet_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved figures under {out_dir}")


# =============================================================================
# Modality characterization: what do the top-corresponding sheet units care about?
# =============================================================================

def _unit_permutation_test(pref: np.ndarray, top_mask: np.ndarray, n_perm: int,
                           rng: np.random.Generator) -> tuple[float, float]:
    """Two-sided permutation test over UNIT IDENTITY (see module docstring for
    why this, not the 626-bin block bootstrap, is the right null here):
    is mean(pref[top_mask]) - mean(pref) more extreme than a random
    same-size subset of the 2048 units would give by chance?
    """
    n, n_top = pref.size, int(top_mask.sum())
    observed = float(pref[top_mask].mean() - pref.mean())
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        sel = rng.permutation(n)[:n_top]
        null[i] = pref[sel].mean() - pref.mean()
    p_two_sided = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return observed, p_two_sided


def modality_preference_analysis(results: dict, coords: np.ndarray, out_dir: Path,
                                 args: argparse.Namespace) -> dict:
    """Per seed: rho_a - rho_v preference index for every sheet unit, and
    whether the seed's top-decile units (by intact "av" rho) differ from the
    full 2048-unit population. Writes one CSV per seed; returns the JSON-
    ready summary dict (also folded into metadata.json by the caller).
    """
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

        pd.DataFrame({
            "unit_index": np.arange(av.size),
            "sheet_row": coords[:, 0], "sheet_col": coords[:, 1],
            "av_rho": av, "a_rho": a_rho, "v_rho": v_rho,
            "modality_preference_a_minus_v": pref,
            "is_top_decile_by_av_rho": top_mask,
        }).to_csv(out_dir / f"{seed_name}_modality_preference.csv", index=False)

        summary[seed_name] = dict(
            n_units=int(av.size), n_top_decile=n_top,
            top_decile_fraction=args.pref_top_decile,
            a_rho_mean_all=float(a_rho.mean()), a_rho_mean_top_decile=float(a_rho[top_mask].mean()),
            v_rho_mean_all=float(v_rho.mean()), v_rho_mean_top_decile=float(v_rho[top_mask].mean()),
            pref_mean_all=float(pref.mean()), pref_mean_top_decile=float(pref[top_mask].mean()),
            pref_median_all=float(np.median(pref)),
            pref_median_top_decile=float(np.median(pref[top_mask])),
            observed_diff_top_minus_all=observed,
            null_hypothesis="unit-identity permutation (see module docstring)",
            n_perm=args.pref_n_perm, p_value_two_sided=p_val,
        )
        log.info(
            f"  modality preference {seed_name}: a_rho_top={a_rho[top_mask].mean():.3f} "
            f"v_rho_top={v_rho[top_mask].mean():.3f} pref_top={pref[top_mask].mean():.3f} "
            f"vs pref_all={pref.mean():.3f}  diff={observed:.3f}  p={p_val:.4g}"
        )

    (out_dir / "modality_preference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def make_modality_preference_figures(results: dict, coords: np.ndarray, out_dir: Path,
                                     args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caption = (
        "Circled = top decile by intact-av rho. Sheet neighbourhoods span only 4 rows "
        "(layer 18) -> effectively 1-D along columns, not a 2-D cortical patch; coordinates "
        "are the model's default raster lattice, not a trained topography (see README)."
    )

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
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    im0 = _plot_sheet_map(axes[0], prefs[seeds[0]], coords,
                          f"{seeds[0]}: rho_a - rho_v (audio<->video preference)",
                          vlim, tops[seeds[0]], "RdBu_r")
    _plot_sheet_map(axes[1], prefs[seeds[1]], coords,
                    f"{seeds[1]}: rho_a - rho_v (audio<->video preference)",
                    vlim, tops[seeds[1]], "RdBu_r")
    fig.colorbar(im0, ax=axes, shrink=0.8, label="rho_a - rho_v")
    fig.suptitle(caption, fontsize=7.5, y=1.03)
    fig.savefig(out_dir / "modality_preference_sheet_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, seeds):
        m = tops[name]
        ax.scatter(avs[name][~m], prefs[name][~m], s=6, alpha=0.3, color="gray",
                  label="other units")
        ax.scatter(avs[name][m], prefs[name][m], s=12, alpha=0.85, color="crimson",
                  label=f"top decile ({int(m.sum())} units)")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("intact av rho")
        ax.set_ylabel("rho_a - rho_v")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Modality preference vs intact-model fit, per sheet unit. " + caption,
                fontsize=7.5)
    fig.savefig(out_dir / "modality_preference_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved modality-preference figures under {out_dir}")


if __name__ == "__main__":
    main()
