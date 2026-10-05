"""
rsa/topoomni_av_separability_localizer.py
=============================================
In-silico AV-integration localizer. Supports two condition designs
(--design), both using the same Ward's-linkage clustering + per-cluster
Fisher's-exact enrichment machinery, just with a different set of
artificially-constructed "conditions" per stimulus bin:

  --design dummy (default, 3 conditions, 626*3 = 1878 stimuli):
    (1) av            -- real audio + real video (genuine pairing)
    (2) clsav_from_a  -- real audio + dummy (blank) video
    (3) clsav_from_v  -- dummy (silent) audio + real video
    Tests modality-presence selectivity: does a cluster represent genuinely
    paired AV input differently from a unimodal-plus-blank substitute?

  --design scramble (2 conditions, 626*2 = 1252 stimuli):
    (1) av         -- real audio + real video, correct temporal pairing
    (2) avscramble -- video[i] + a randomly permuted audio[perm(i)] (seed=42,
                       same permutation as every other scramble script here)
    Tests binding/synchrony selectivity: does a cluster represent correctly-
    paired AV input differently from ANY audio+video pairing regardless of
    whether the two streams actually co-occurred?

Both designs cluster the stimuli with Algorithm 1 (Ward's linkage + top-down
selectivity-scored early stopping, identical machinery to
topoomni_sheet_localizer.py), driven by an INDEPENDENT embedding
(--cluster-embedding-model's own per-condition vectors, concatenated across
conditions) and scored against the sheet model's own per-condition response
(--sheet-model's per-condition .npy arrays).

Then the actual localizer question: does any terminal cluster come out
enriched for the TRUE "av" condition relative to the other condition(s)
pooled? (one-vs-rest Fisher's exact test per cluster). A cluster
significantly enriched for real/matched pairing indicates the sheet model
represents that distinction at those units, not just "the model responds to
whichever modality is present" (dummy design) or "the model responds to
audio+video co-occurrence regardless of match" (scramble design).

The winning (and every --p-threshold-significant) cluster's top-|t| units are
read out over the 626 REAL av bins only (sheet_av[:, units]) and saved as an
RSA-ready embedding, to be run through rsa/searchlight.py exactly like the
auditory localizer -- same "mask, not model" mapping-to-real-cortex step.

When --sheet-model is a TopoOmni text-decoder sheet (topoomni_layer{N}_sheet_
{mp,lt}), each significant cluster's selected units are also scored for
2D spatial compactness on TopoOmni's own cortical-sheet grid (rsa/
spatial_stats.py's island_morans_i / compactness_index) -- the paper's own
internal validation that a functionally-defined cluster also forms a
contiguous patch on the sheet, not a scattered set of units.

Usage
-----
python rsa/topoomni_av_separability_localizer.py \\
    --sheet-model topoomni_layer18_sheet_mp \\
    --cluster-embedding-model pe-av-small-16-frame \\
    --design scramble \\
    --embeddings-dir $ROOT/outputs/model_embeddings \\
    --bin-sec 5.0 --skip-sec 5.0 --n-min 10 --n-max 450
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.topoomni_sheet_localizer import ward_cluster_and_score, top_units_for_cluster
from rsa.localizer_naming import base_name as loc_base_name, model_tag, summary_json_name
from rsa.spatial_stats import island_morans_i, compactness_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DESIGN_CONDITIONS = {
    "dummy": ["av", "clsav_from_a", "clsav_from_v"],
    "scramble": ["av", "avscramble"],
}

# Matches sheet models with a real 2D grid position (rsa/spatial_stats.py):
# TopoOmni's text-decoder cortical sheet at a given layer, mean-pool ("_mp") or
# last-token ("_lt").
_TOPOOMNI_SHEET_RE = re.compile(r"^topoomni_layer(\d+)_sheet_(?:mp|lt)$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet-model", required=True, dest="sheet_model", help="e.g. topoomni_layer18_sheet_mp")
    p.add_argument("--cluster-embedding-model", default="pe-av-small-16-frame", dest="cluster_embedding_model")
    p.add_argument("--design", default="dummy", choices=list(DESIGN_CONDITIONS),
                   help="Which condition set to cluster: 'dummy' (3-condition modality-presence "
                        "test, default) or 'scramble' (2-condition binding/synchrony test).")
    p.add_argument("--embeddings-dir", required=True, dest="embeddings_dir")
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--n-min", type=int, default=10, dest="n_min")
    p.add_argument("--n-max", type=int, default=450, dest="n_max")
    p.add_argument("--top-pct-units", type=float, default=1.0, dest="top_pct_units")
    p.add_argument("--unit-selection-mode", default="topk", choices=["topk", "fdr"],
                   dest="unit_selection_mode",
                   help="How to select readout cortical-sheet units per cluster: 'topk' "
                        "(top --top-pct-units%% by |t| magnitude, default) or 'fdr' "
                        "(TopoOmni-paper-exact Benjamini-Hochberg FDR gate at --fdr-q).")
    p.add_argument("--fdr-q", type=float, default=0.001, dest="fdr_q",
                   help="BH-FDR q-value for --unit-selection-mode=fdr.")
    p.add_argument("--p-threshold", type=float, default=1e-4, dest="p_threshold",
                    help="Save an individual readout for every cluster with p_value below this "
                         "(in addition to the single 'winning' cluster), plus one combined "
                         "readout unioning all of their top units.")
    p.add_argument("--n-spatial-perm", type=int, default=2000, dest="n_spatial_perm",
                   help="Permutations for the Moran's I / compactness null (topoomni sheets only).")
    return p.parse_args()


def _emb_dir(embeddings_dir, model, bin_sec, skip_sec):
    return Path(embeddings_dir) / model / f"bin{int(bin_sec)}s_skip{int(skip_sec)}s"


def _load_condition(embeddings_dir, base_model, condition, suffix, bin_sec, skip_sec):
    """suffix='av' for cluster-driving embeddings and 'sheet av' for sheet arrays.
    condition='av' -> base_model itself; condition='clsav_from_a'/'clsav_from_v' ->
    base_model_clsav_from_{a,v} (PE-AV) or base_model_clsav_from_{a,v} (TopoOmni sheet)."""
    if condition == "av":
        model_name = base_model
    else:
        model_name = f"{base_model}_{condition}"
    d = _emb_dir(embeddings_dir, model_name, bin_sec, skip_sec)
    path = d / f"{model_name}_{suffix}.npy"
    return np.load(path).astype(np.float64), path


def main():
    args = parse_args()
    CONDITIONS = DESIGN_CONDITIONS[args.design]
    rest_conditions = [c for c in CONDITIONS if c != "av"]

    # ── Load the N-condition x 626-bin clustering-driving embedding ──────────
    stim_parts, stim_paths = [], []
    for cond in CONDITIONS:
        arr, path = _load_condition(args.embeddings_dir, args.cluster_embedding_model, cond, "av",
                                     args.bin_sec, args.skip_sec)
        stim_parts.append(arr)
        stim_paths.append(path)
        log.info(f"  clustering-driver [{cond}]: {path}  shape={arr.shape}")
    n_bins_per_cond = stim_parts[0].shape[0]
    assert all(a.shape[0] == n_bins_per_cond for a in stim_parts), "condition bin counts must match"
    stim_emb = np.concatenate(stim_parts, axis=0)  # (3*n_bins, D)

    # ── Load the N-condition x 626-bin sheet response (scored model) ────────
    sheet_parts = []
    for cond in CONDITIONS:
        arr, path = _load_condition(args.embeddings_dir, args.sheet_model, cond, "av",
                                     args.bin_sec, args.skip_sec)
        sheet_parts.append(arr)
        log.info(f"  sheet [{cond}]: {path}  shape={arr.shape}")
    sheet = np.concatenate(sheet_parts, axis=0)  # (N*n_bins, n_units)
    assert sheet.shape[0] == stim_emb.shape[0]

    condition_labels = np.array(
        [c for c in CONDITIONS for _ in range(n_bins_per_cond)]
    )
    n_total = len(condition_labels)
    log.info(f"Total stimuli: {n_total} ({n_bins_per_cond} bins x {len(CONDITIONS)} conditions, "
             f"design={args.design})")

    log.info(f"Running Algorithm 1 on {n_total} stimuli (n_min={args.n_min} n_max={args.n_max}) ...")
    clusters = ward_cluster_and_score(stim_emb, sheet, args.n_min, args.n_max)
    sizes = sorted([len(c) for c in clusters], reverse=True)
    log.info(f"  {len(clusters)} terminal clusters. Sizes (top 10): {sizes[:10]}  "
             f"(total covered: {sum(sizes)}/{n_total})")

    # ── Test: is any cluster enriched for the TRUE av condition? ────────────
    # NOTE: this per-cluster Fisher's-exact loop is a selection/enrichment step,
    # not a confirmatory test battery -- there is no cross-cluster multiple-
    # comparison correction here. The hard --p-threshold is the guard against
    # false positives, and the "winner" reported below is simply the min-p cluster.
    log.info("=" * 70)
    log.info(f"Testing each terminal cluster for av-condition enrichment "
             f"(av vs. {rest_conditions} pooled), Fisher's exact, one-sided")
    is_av = condition_labels == "av"
    n_av_total = int(is_av.sum())
    n_rest_total = n_total - n_av_total

    cluster_reports = []
    for ci, c in enumerate(clusters):
        mask = np.zeros(n_total, dtype=bool)
        mask[c] = True
        n_av_in = int(is_av[mask].sum())
        n_rest_in = int(mask.sum()) - n_av_in
        n_av_out = n_av_total - n_av_in
        n_rest_out = n_rest_total - n_rest_in
        table = [[n_av_in, n_rest_in], [n_av_out, n_rest_out]]
        odds_ratio, p_val = fisher_exact(table, alternative="greater")
        purity = n_av_in / max(1, len(c))
        per_condition_counts = {f"n_{cond}": int(np.sum(condition_labels[mask] == cond))
                                 for cond in rest_conditions}
        cluster_reports.append(dict(
            cluster_idx=ci, size=int(len(c)), n_av=n_av_in, n_rest=n_rest_in,
            **per_condition_counts,
            av_purity=float(purity), odds_ratio=float(odds_ratio), p_value=float(p_val),
        ))

    cluster_reports.sort(key=lambda r: r["p_value"])
    log.info(f"{'cluster':>7} {'size':>6} {'n_av':>6} {'n_rest':>7} "
             f"{'av_purity':>10} {'odds_ratio':>11} {'p_value':>10}")
    for r in cluster_reports[:15]:
        log.info(f"{r['cluster_idx']:>7} {r['size']:>6} {r['n_av']:>6} {r['n_rest']:>7} "
                  f"{r['av_purity']:>10.4f} {r['odds_ratio']:>11.3f} {r['p_value']:>10.2e}")

    winner = cluster_reports[0]
    winner_idx = winner["cluster_idx"]
    winner_bins = clusters[winner_idx]
    log.info("=" * 70)
    log.info(f"WINNING (most av-enriched) cluster #{winner_idx}: "
             f"size={winner['size']}  av_purity={winner['av_purity']:.4f}  "
             f"p={winner['p_value']:.2e}  (baseline av fraction = {n_av_total/n_total:.4f})")

    # ── Read out winning cluster's top-|t| units over the REAL 626 av bins ──
    units = top_units_for_cluster(winner_bins, sheet, args.top_pct_units,
                                   mode=args.unit_selection_mode, fdr_q=args.fdr_q)
    real_av_sheet = sheet_parts[0]  # first CONDITIONS entry is "av", real 626 bins
    pattern = real_av_sheet[:, units]

    driver = model_tag(args.cluster_embedding_model)
    sheet_tag = model_tag(args.sheet_model)
    suffix = "_fdr" if args.unit_selection_mode == "fdr" else ""
    base_name = loc_base_name("av_separability", driver, sheet_tag, suffix, design=args.design)
    log.info(f"Naming: driver={driver}  sheet={sheet_tag}  design={args.design}  -> {base_name}")

    sheet_layer_match = _TOPOOMNI_SHEET_RE.match(args.sheet_model)
    sheet_layer_idx = int(sheet_layer_match.group(1)) if sheet_layer_match else None
    if sheet_layer_idx is not None:
        log.info(f"Sheet model has a real 2D grid (TopoOmni text-decoder layer {sheet_layer_idx}) "
                 f"-- will score each significant cluster's spatial compactness.")
    out_model_name = base_name
    out_dir = _emb_dir(args.embeddings_dir, out_model_name, args.bin_sec, args.skip_sec)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_model_name}_av.npy"
    np.save(out_path, pattern.astype(np.float32))
    log.info(f"Saved localizer embedding (real-av readout of winning cluster's top units): "
             f"{out_path}  shape={pattern.shape}")

    summary = {
        "sheet_model": args.sheet_model,
        "cluster_embedding_model": args.cluster_embedding_model,
        "n_bins_per_condition": n_bins_per_cond,
        "n_total_stimuli": n_total,
        "n_min": args.n_min, "n_max": args.n_max,
        "n_terminal_clusters": len(clusters),
        "cluster_sizes": sizes,
        "baseline_av_fraction": n_av_total / n_total,
        "top_clusters_by_av_enrichment": cluster_reports[:15],
        "winning_cluster": winner,
        "n_units_readout": int(len(units)),
        "embedding_path": str(out_path),
        "unit_selection_mode": args.unit_selection_mode,
        "fdr_q": args.fdr_q if args.unit_selection_mode == "fdr" else None,
    }
    # ── Per-cluster readouts for every cluster below --p-threshold, + combined ──
    sig_clusters = [r for r in cluster_reports if r["p_value"] < args.p_threshold]
    log.info("=" * 70)
    log.info(f"Saving individual readouts for {len(sig_clusters)} clusters with "
             f"p < {args.p_threshold:.1e} (plus one combined/union readout) ...")

    per_cluster_paths = []
    union_units = set()
    for r in sig_clusters:
        ci = r["cluster_idx"]
        c_units = top_units_for_cluster(clusters[ci], sheet, args.top_pct_units,
                                         mode=args.unit_selection_mode, fdr_q=args.fdr_q)
        union_units.update(int(u) for u in c_units)
        c_pattern = real_av_sheet[:, c_units]
        c_model_name = f"{base_name}_c{ci}"
        c_dir = _emb_dir(args.embeddings_dir, c_model_name, args.bin_sec, args.skip_sec)
        c_dir.mkdir(parents=True, exist_ok=True)
        c_path = c_dir / f"{c_model_name}_av.npy"
        np.save(c_path, c_pattern.astype(np.float32))
        per_cluster_paths.append(str(c_path))

        if sheet_layer_idx is not None and len(c_units) >= 2:
            # Moran's I / compactness are computed on the LOCAL (4, 512) subgrid a
            # single layer's 2048 channels occupy (rsa/spatial_stats.py) -- global
            # row offset is a pure translation and does not affect either statistic,
            # so layer_idx itself is not needed here (only for absolute positions).
            moran = island_morans_i(np.asarray(c_units), n_perm=args.n_spatial_perm)
            compact = compactness_index(np.asarray(c_units), n_perm=args.n_spatial_perm)
            r["spatial_morans_i"] = moran
            r["spatial_compactness"] = compact
            log.info(f"  cluster {ci:>3} (size={r['size']:>4} purity={r['av_purity']:.4f} "
                     f"p={r['p_value']:.2e}): {c_path.name}  shape={c_pattern.shape}  "
                     f"morans_i={moran['morans_i']:.3f} (p={moran['p_value']:.3f})  "
                     f"compactness_z={compact['z_score']:.2f} (p={compact['p_value']:.3f})")
        else:
            log.info(f"  cluster {ci:>3} (size={r['size']:>4} purity={r['av_purity']:.4f} "
                     f"p={r['p_value']:.2e}): {c_path.name}  shape={c_pattern.shape}")

    union_units_sorted = np.array(sorted(union_units))
    combined_spatial = None
    if len(union_units_sorted) == 0:
        log.warning("  No significant clusters -> union of readout units is empty. "
                    "Skipping combined '_all' embedding (would be a 0-feature .npy).")
        combined_path = None
    else:
        combined_pattern = real_av_sheet[:, union_units_sorted]
        combined_model_name = f"{base_name}_all"
        combined_dir = _emb_dir(args.embeddings_dir, combined_model_name, args.bin_sec, args.skip_sec)
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / f"{combined_model_name}_av.npy"
        np.save(combined_path, combined_pattern.astype(np.float32))
        log.info(f"  combined (union of {len(sig_clusters)} clusters, {len(union_units_sorted)} unique "
                 f"units): {combined_path.name}  shape={combined_pattern.shape}")
        if sheet_layer_idx is not None and len(union_units_sorted) >= 2:
            combined_spatial = dict(
                morans_i=island_morans_i(union_units_sorted, n_perm=args.n_spatial_perm),
                compactness=compactness_index(union_units_sorted, n_perm=args.n_spatial_perm),
            )
            log.info(f"  combined spatial: morans_i={combined_spatial['morans_i']['morans_i']:.3f} "
                     f"(p={combined_spatial['morans_i']['p_value']:.3f})  "
                     f"compactness_z={combined_spatial['compactness']['z_score']:.2f} "
                     f"(p={combined_spatial['compactness']['p_value']:.3f})")

    summary["design"] = args.design
    summary["conditions"] = CONDITIONS
    summary["p_threshold"] = args.p_threshold
    summary["significant_clusters"] = sig_clusters
    summary["per_cluster_embedding_paths"] = per_cluster_paths
    summary["combined_embedding_path"] = str(combined_path) if combined_path is not None else None
    summary["combined_n_units"] = int(len(union_units_sorted))
    summary["combined_spatial_stats"] = combined_spatial
    summary["sheet_grid_layer_idx"] = sheet_layer_idx

    out_json = Path(args.embeddings_dir) / summary_json_name("av_separability", driver, sheet_tag, suffix,
                                                               design=args.design)
    json.dump(summary, open(out_json, "w"), indent=2)
    log.info(f"Saved cluster summary: {out_json}")
    log.info(f"Done. Next: run rsa/searchlight.py --model {out_model_name} --modality av "
             f"(and likewise for each {base_name}_c{{N}} / {base_name}_all model).")


if __name__ == "__main__":
    main()
