#!/usr/bin/env python3
"""
rsa/topography_control.py
==========================
Control for rsa/cca_seed_sheet_rsa_truecoords.py: does the sheet's true k-NN
NEIGHBOURHOOD STRUCTURE do any work, or would any random 100-unit sample of
the same layer pool score about as well?

Motivation: the layer-18 raster-vs-truecoord comparison found the rho
DISTRIBUTION (mean/spread) essentially unchanged between two totally
different coordinate systems, while per-unit rho was uncorrelated across
them (spatial Spearman ~0 for both seeds). That is consistent with a
searchlight that isn't using topography at all -- if a k=100 patch behaves
like an arbitrary 100-unit sample, its rho reflects sampling variance of
"100 units out of the layer", not spatial structure. This script tests that
directly: for the SAME units, seeds and a/v/av conditions used by a true-
coordinate run, replace the k-NN neighbour set with `--n-draws` independent
random k-unit draws (uniform over the run's own unit pool, ignoring
coordinates entirely) and compare the resulting rho distribution (mean, std,
full sample) against the true-coordinate run's saved rho.

This intentionally reuses the true run's loaded embeddings/seeds rather than
re-deriving them, and skips re-deriving a null p-value (n-perm is small here
by default -- this is a distribution-shape diagnostic, not a significance
test; actual_rho does not depend on n_perm, only the discarded null does).

Usage
-----
  python rsa/topography_control.py --layers 18 \\
      --true-run-dir <out>/k100_layer18_truecoords --k 100 --n-draws 3 \\
      --output-dir <out>/k100_layer18_truecoords/topography_control

  python rsa/topography_control.py --layers 1,9,18,27,34,35 \\
      --true-run-dir <out>/k100_multilayer_truecoords --k 100 --n-draws 3 \\
      --output-dir <out>/k100_multilayer_truecoords/topography_control
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.perm_searchlight import _run_hemisphere, within_run_shift_pair_indices  # noqa: E402
from rsa.shared.rsa_utils import (  # noqa: E402
    align_and_assert_bins, get_run_bin_counts, load_fmri_cifti,
    preprocess_fmri, process_model_embeddings,
)
from cf_modeling.roi_mean_partial_connectivity import _load_mask  # noqa: E402
from rsa.cca_seed_sheet_rsa_truecoords import _sheet_model_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def random_neighbors(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """(n_units, k) int32 neighbour indices drawn uniformly at random from the
    whole unit pool (coordinates ignored entirely), excluding self.

    CONFOUNDED for a multi-layer/multi-block pool: rho varies strongly by
    layer (e.g. 0.219 at layer 1 vs. 0.386 at layer 18 in the multi-layer
    run), so a cross-layer random draw differs from a true within-layer k-NN
    neighbourhood for a reason that has nothing to do with spatial locality
    -- it is pooling across a mean-rho gradient. Kept only for reference /
    backward comparison; `random_neighbors_within_group` is the corrected
    version and is what `main()` uses."""
    out = np.empty((n_units, k), dtype=np.int32)
    for i in range(n_units):
        c = rng.choice(n_units - 1, size=k, replace=False)
        c[c >= i] += 1
        out[i] = c
    return out


def random_neighbors_within_group(group_id: np.ndarray, k: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """(n_units, k) int32 neighbour indices drawn uniformly at random, but
    restricted to units sharing the same `group_id` (e.g. the same decoder
    layer, or the same encoder block) -- coordinates within the group are
    still ignored, but the cross-group mean-rho confound in
    `random_neighbors` is removed: the only thing varying between a true
    k-NN neighbourhood and this random one is spatial locality *within* the
    group. Groups smaller than k+1 raise -- k should be chosen <= the
    smallest group's size (true k-NN neighbourhoods have the same limit)."""
    n = group_id.size
    out = np.empty((n, k), dtype=np.int32)
    groups = {}
    for g in np.unique(group_id):
        groups[g] = np.flatnonzero(group_id == g)
    for i in range(n):
        pool = groups[group_id[i]]
        if pool.size < k + 1:
            raise ValueError(f"group {group_id[i]} has only {pool.size} units, need > {k}")
        pool_wo_self = pool[pool != i]
        c = rng.choice(pool_wo_self, size=k, replace=False)
        out[i] = c
    return out



def demo() -> None:
    """Self-check: random_neighbors returns k unique, self-excluding indices
    per row, and (with a huge population) looks roughly uniform."""
    rng = np.random.default_rng(0)
    n, k = 500, 50
    nb = random_neighbors(n, k, rng)
    assert nb.shape == (n, k)
    for i in range(n):
        row = nb[i]
        assert len(set(row.tolist())) == k, "duplicate neighbours"
        assert i not in row, "self included"
    counts = np.bincount(nb.reshape(-1), minlength=n)
    assert counts.min() > 0, "some unit never drawn -- suspiciously non-uniform"

    # within-group: two groups of 100, k=20 -- every draw must stay in-group.
    group_id = np.array([0] * 100 + [1] * 100)
    nb2 = random_neighbors_within_group(group_id, 20, rng)
    for i in range(200):
        assert i not in nb2[i]
        assert (group_id[nb2[i]] == group_id[i]).all(), "cross-group leak"
        assert len(set(nb2[i].tolist())) == 20
    print("demo OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    p.add_argument("--layers", required=False)
    p.add_argument("--true-run-dir", required=False,
                   help="Directory holding the matching true-coordinate run's "
                        "{seed}_{modality}_rho.npy files (for comparison).")
    p.add_argument("--modalities", default="av,a,v")
    p.add_argument("--output-dir", required=False)
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--n-draws", type=int, default=3)
    p.add_argument("--within-layer", action="store_true", default=True,
                   help="Draw random neighbours from the SAME layer only "
                        "(corrected control -- see random_neighbors_within_group "
                        "docstring). Default on.")
    p.add_argument("--cross-layer", dest="within_layer", action="store_false",
                   help="Use the confounded whole-pool draw instead (kept for "
                        "backward comparison only).")
    p.add_argument("--n-perm", type=int, default=200,
                   help="Small: this control compares rho distributions, not "
                        "p-values (actual_rho is independent of n_perm).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-batch-size", type=int, default=64)
    p.add_argument("--perm-batch-size", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        demo()
        return
    if not (args.layers and args.true_run_dir and args.output_dir):
        raise SystemExit("--layers, --true-run-dir, --output-dir are required (unless --demo)")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true_run_dir = Path(args.true_run_dir)
    layers = sorted(int(x) for x in args.layers.split(",") if x.strip())

    timing_df = pd.read_csv(args.timing_csv)
    run_trs = np.load(str(Path(args.preprocessed_dir) /
                          f"group_average_{args.fmri_suffix}_run_trs.npy"))
    cifti_path = (Path(args.preprocessed_dir) /
                 f"group_average_{args.fmri_suffix}_cortex_59k.dtseries.nii")
    fmri_continuous = load_fmri_cifti(str(cifti_path))
    run_bins = get_run_bin_counts(
        timing_df, run_trs, args.bin_sec, args.tr, args.delay_sec, args.skip_sec)
    perm_idx_all = within_run_shift_pair_indices(run_bins, args.n_perm, args.seed)

    seeds = {
        args.seed_a_name: Path(args.masks_dir) / args.seed_a_mask,
        args.seed_p_name: Path(args.masks_dir) / args.seed_p_mask,
    }
    seed_embeddings = {}
    for name, mask_path in seeds.items():
        mask = _load_mask(mask_path, fmri_continuous.shape[0])
        seed_embeddings[name] = preprocess_fmri(
            fmri_continuous[mask, :], timing_df, run_trs, args.bin_sec, args.tr,
            args.delay_sec, skip_sec=args.skip_sec, normalize=True,
        )
    del fmri_continuous

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    bin_tag = f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"

    emb_by_modality = {}
    n_total_units = None
    layer_id_parts = []
    for modality in modalities:
        parts = []
        for layer_idx in layers:
            model_name = _sheet_model_name(layer_idx)
            emb_path = (Path(args.embeddings_dir) / model_name / bin_tag /
                       f"{model_name}_{modality}.npy")
            emb = process_model_embeddings(
                str(emb_path), timing_df, bin_sec=args.bin_sec, tr=args.tr,
                run_trs=run_trs, delay_sec=args.delay_sec, skip_sec=args.skip_sec,
                normalize=True,
            )
            parts.append(emb)
            if not layer_id_parts or len(layer_id_parts) < len(layers):
                layer_id_parts.append(np.full(emb.shape[1], layer_idx, dtype=np.int32))
        emb_by_modality[modality] = np.concatenate(parts, axis=1)
        n_total_units = emb_by_modality[modality].shape[1]
    layer_id = np.concatenate(layer_id_parts, axis=0)
    assert layer_id.size == n_total_units
    log.info(f"Layers {layers}: {n_total_units} total units, {args.n_draws} random draws, "
            f"k={args.k}, within_layer={args.within_layer}")

    rng = np.random.default_rng(args.seed)
    surf_idx = np.arange(n_total_units, dtype=np.int32)
    vertex_to_col = np.arange(n_total_units, dtype=np.int32)

    summary = {}
    draw_rho = {}  # key -> (n_draws, n_units)
    for modality in modalities:
        sheet_emb = emb_by_modality[modality]
        for seed_name, seed_emb in seed_embeddings.items():
            sheet_aligned, seed_aligned = align_and_assert_bins(sheet_emb, seed_emb)
            key = f"{seed_name}_{modality}"
            rho_draws = np.empty((args.n_draws, n_total_units), dtype=np.float32)
            for d in range(args.n_draws):
                draw_rng = np.random.default_rng(args.seed + 1000 * (d + 1))
                if args.within_layer:
                    neighbors = random_neighbors_within_group(layer_id, args.k, draw_rng)
                else:
                    neighbors = random_neighbors(n_total_units, args.k, draw_rng)
                actual_rho, _p, _null = _run_hemisphere(
                    sheet_aligned, seed_aligned, neighbors, surf_idx, vertex_to_col,
                    args.method, perm_idx_all, args.gpu_batch_size, args.perm_batch_size,
                )
                rho_draws[d] = actual_rho
                log.info(f"  {key} draw {d}: mean={actual_rho.mean():.4f} std={actual_rho.std():.4f}")
            draw_rho[key] = rho_draws

            true_rho = np.load(true_run_dir / f"{key}_rho.npy")
            random_pooled = rho_draws.reshape(-1)
            summary[key] = dict(
                n_units=n_total_units, n_draws=args.n_draws, k=args.k,
                true_rho_mean=float(true_rho.mean()), true_rho_std=float(true_rho.std()),
                random_rho_mean=float(random_pooled.mean()), random_rho_std=float(random_pooled.std()),
                random_rho_mean_per_draw=[float(rho_draws[d].mean()) for d in range(args.n_draws)],
                random_rho_std_per_draw=[float(rho_draws[d].std()) for d in range(args.n_draws)],
                mean_diff_true_minus_random=float(true_rho.mean() - random_pooled.mean()),
                # Welch t-test on the mean difference (true vs. pooled-random units)
            )
            log.info(f"  {key}: true mean={true_rho.mean():.4f} std={true_rho.std():.4f} | "
                     f"random mean={random_pooled.mean():.4f} std={random_pooled.std():.4f} | "
                     f"diff={summary[key]['mean_diff_true_minus_random']:+.4f}")

    group_desc = ("drawn uniformly WITHIN the same layer only (coordinates within "
                 "the layer still ignored -- the only thing varying vs. a true "
                 "k-NN neighbourhood is spatial locality within that layer)"
                 if args.within_layer else
                 "drawn uniformly from the WHOLE unit pool across all layers "
                 "(CONFOUNDED by cross-layer mean-rho differences -- see "
                 "random_neighbors docstring; kept only for backward comparison)")
    interpretation = (
        "For every seed x modality, compare true_rho_mean/std (from the matching "
        f"true-coordinate k-NN run) against random_rho_mean/std (k=same, neighbours "
        f"{group_desc}, {args.n_draws} independent draws pooled). If these are close "
        "(within random-draw spread) for all combinations, the k-NN searchlight is "
        "not detecting spatial/topographic structure -- any k-unit sample of this "
        "layer tracks the seed about as well, and no hotspot-contiguity or "
        "spatial-map claim from the true-coordinate run is supported by this "
        "check alone. A true_rho_mean that consistently and substantially exceeds "
        "random_rho_mean, especially by more than the spread across the "
        f"{args.n_draws} random draws (random_rho_std_per_draw), is evidence that "
        "true neighbourhoods carry information beyond an arbitrary same-size "
        "sample of the same layer."
    )
    (out_dir / "topography_control_summary.json").write_text(
        json.dumps({"layers": layers, "within_layer": args.within_layer,
                   "interpretation": interpretation,
                   "results": summary}, indent=2) + "\n")
    log.info(f"Saved {out_dir / 'topography_control_summary.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    for ax, (key, rho_draws) in zip(axes.flat, draw_rho.items()):
        true_rho = np.load(true_run_dir / f"{key}_rho.npy")
        ax.hist(true_rho, bins=40, alpha=0.5, density=True, label="true coords", color="crimson")
        ax.hist(rho_draws.reshape(-1), bins=40, alpha=0.5, density=True,
               label=f"random ({args.n_draws} draws)", color="gray")
        ax.set_title(key, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("rho")
    fig.suptitle(f"Topography control: true-neighbourhood vs random-neighbourhood rho "
                f"(layers {layers}, k={args.k})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "topography_control_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Done.")


if __name__ == "__main__":
    main()
