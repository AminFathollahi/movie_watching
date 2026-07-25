"""
rsa/topoomni_sheet_localizer.py
=================================
Topo-Omni cortical-sheet localizer, implementing Algorithm 1 from
papers/Topo-omni.pdf Sec 2.6 & 4.7.2.

Algorithm 1 clusters STIMULI (video/movie segments) via Ward's linkage on an
INDEPENDENT semantic embedding (the paper uses omni-embed-nemotron-3b),
scores each candidate cluster by a Welch's t-test computed PER
CORTICAL-SHEET UNIT (in-cluster vs out-of-cluster stimuli), summarized as
the median t-value across units, then does a top-down dendrogram traversal
with selectivity-based early stopping (exact pseudocode in Sec 4.7.2,
reproduced in ward_cluster_and_score() / _split() below).

Cross-model independence (avoids using a model to validate itself):
  - Stimulus-clustering embedding : an INDEPENDENT model's embedding, not
    topoomni's own. Default: pe-av-small-16-frame's _event_t (ModernBERT
    text embedding of the per-bin caption/transcript "event" text) --
    independent of Topo-Omni's own audio/video pathways. General convention:
    use a DIFFERENT model's embedding to drive each model's own
    sheet-selectivity analysis (e.g. PE-AV drives omni3b/topoomni sheet
    analyses; omni3b drives PE-AV sheet analyses), with text (_event_t)
    embeddings as a shared baseline narrative comparison across all of them.
  - Cortical response : topoomni_layer{N}_sheet_av.npy (the model's OWN
    joint-condition cortical sheet -- this is what Alg. 1 actually scores
    against, matching the paper).

POSITIVE-CONTROL (speech) LOCALIZER. Among the terminal clusters from Alg. 1,
pick the one whose member bins have the highest mean whisper_speech_proxy
per-bin drive (an independent, non-topoomni audio-content proxy derived from
Whisper-large-v3, chosen over an AudioMAE-based proxy that does not reliably
separate in-cluster from overall drive). Validates the pipeline: should land
on auditory/STG cortex.

The AV-integration / condition-enrichment analysis lives in
rsa/topoomni_av_separability_localizer.py, which tests condition enrichment
directly on the terminal clusters without an external proxy.

For each winning cluster, the RSA "model embedding" fed to
rsa/searchlight.py is built from the TOP-|t| cortical-sheet units that drove
that cluster's score, read out over ALL 626 bins -- i.e. the model's own
predicted topographic territory for that cluster, in a form with enough
feature dimensions (>= 2 units) for a valid correlation-distance RDM (a
single-feature-per-bin embedding degenerates the RDM).

Usage
-----
python rsa/topoomni_sheet_localizer.py \\
    --sheet-model topoomni_layer18_sheet_mp \\
    --cluster-embedding-model pe-av-small-16-frame --cluster-embedding-modality event_t \\
    --auditory-regressor-model whisper_speech_proxy \\
    --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \\
    --bin-sec 5.0 --skip-sec 5.0 --n-min 10 --n-max 375
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.localizer_naming import base_name as loc_base_name, driver_tag, model_tag, summary_json_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet-model", required=True, help="e.g. topoomni_layer18_sheet_mp")
    p.add_argument("--cluster-embedding-model", default="pe-av-small-16-frame",
                   dest="cluster_embedding_model",
                   help="INDEPENDENT model whose embedding drives Ward's-linkage "
                        "stimulus clustering (must not be --sheet-model's own family).")
    p.add_argument("--cluster-embedding-modality", default="event_t",
                   dest="cluster_embedding_modality")
    p.add_argument("--auditory-regressor-model", default="whisper_speech_proxy",
                   dest="auditory_regressor_model")
    p.add_argument("--embeddings-dir", required=True, dest="embeddings_dir")
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--n-min", type=int, default=10, dest="n_min")
    p.add_argument("--n-max", type=int, default=375, dest="n_max")
    p.add_argument("--top-pct-units", type=float, default=1.0, dest="top_pct_units",
                   help="Fraction of highest-|t| sheet units to read out as the RSA embedding "
                        "(only used when --unit-selection-mode=topk).")
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
                         "readout unioning all of their top units -- same convention as "
                         "topoomni_av_separability_localizer.py.")
    return p.parse_args()


def _emb_dir(embeddings_dir, model, bin_sec, skip_sec):
    return Path(embeddings_dir) / model / f"bin{int(bin_sec)}s_skip{int(skip_sec)}s"


# =============================================================================
# Algorithm 1 -- top-down dendrogram traversal with selectivity-based
# early stopping (Sec 4.7.2, pseudocode reproduced exactly).
# =============================================================================

def _score(indices: np.ndarray, sheet: np.ndarray, n_min: int, n_max: int) -> float:
    """score(S) = median_u Welch-t(sheet[S, u], sheet[~S, u]) over cortical units u."""
    n = len(indices)
    if n < n_min or n > n_max:
        return -np.inf
    mask = np.zeros(sheet.shape[0], dtype=bool)
    mask[indices] = True
    t_vals, _ = ttest_ind(sheet[mask], sheet[~mask], axis=0, equal_var=False)
    t_vals = np.nan_to_num(t_vals)
    return float(np.median(t_vals))


def _split(node, sheet: np.ndarray, n_min: int, n_max: int, results: list) -> None:
    if node.is_leaf():
        results.append(np.array(node.pre_order()))
        return
    leaves = np.array(node.pre_order())
    sp = _score(leaves, sheet, n_min, n_max)
    L, R = node.get_left(), node.get_right()
    l_leaves, r_leaves = np.array(L.pre_order()), np.array(R.pre_order())
    sL = _score(l_leaves, sheet, n_min, n_max)
    sR = _score(r_leaves, sheet, n_min, n_max)
    if sL < sp and sR < sp:
        results.append(leaves)                          # stop: neither child improves
    elif sL < sp:
        # left does not improve -> emit it as terminal, descend into the
        # improving right subtree only
        results.append(l_leaves)
        _split(R, sheet, n_min, n_max, results)
    elif sR < sp:
        # right does not improve -> emit it as terminal, descend into the
        # improving left subtree only
        results.append(r_leaves)
        _split(L, sheet, n_min, n_max, results)
    else:
        _split(L, sheet, n_min, n_max, results)
        _split(R, sheet, n_min, n_max, results)


def ward_cluster_and_score(stimulus_emb: np.ndarray, sheet: np.ndarray,
                            n_min: int, n_max: int) -> list[np.ndarray]:
    """Full Algorithm 1: Ward's-linkage dendrogram over stimulus_emb, top-down
    traversal with selectivity-based early stopping scored against `sheet`.
    Returns a flat partition: list of arrays of bin indices (terminal clusters)."""
    from scipy.stats import zscore
    emb_z = np.nan_to_num(zscore(stimulus_emb, axis=0, nan_policy="omit"))
    Z = linkage(emb_z, method="ward")
    tree = to_tree(Z)
    results: list[np.ndarray] = []
    _split(tree, sheet, n_min, n_max, results)
    return results


# =============================================================================
# Cluster identification via independent proxies
# =============================================================================

def cluster_reports_for_proxy(clusters: list[np.ndarray], proxy: np.ndarray) -> list[dict]:
    """Score every terminal cluster against `proxy` (one-sided Welch's t,
    alternative='greater' -- cluster ELEVATED relative to the rest, matching
    what identify_cluster_by_proxy's argmax-t winner already implicitly
    selects for). Sorted by ascending p-value. Used to find not just the
    single top-1 winner but every cluster that survives a significance
    threshold, so each can get its own brain-map readout. Depleted clusters
    (lower proxy than baseline) are NOT candidates here -- they are not
    "the auditory cluster" in any sense, just confidently not it."""
    n_total = len(proxy)
    reports = []
    for ci, c in enumerate(clusters):
        mask = np.zeros(n_total, dtype=bool)
        mask[c] = True
        t, p = ttest_ind(proxy[mask], proxy[~mask], equal_var=False, alternative="greater")
        t = float(t) if np.isfinite(t) else -np.inf
        p = float(p) if np.isfinite(p) else 1.0
        reports.append(dict(cluster_idx=ci, size=int(len(c)), t_value=t, p_value=p,
                             mean_in=float(proxy[mask].mean()), mean_out=float(proxy[~mask].mean())))
    reports.sort(key=lambda r: r["p_value"])
    return reports


def identify_cluster_by_proxy(clusters: list[np.ndarray], proxy: np.ndarray) -> tuple[int, np.ndarray]:
    """Pick the cluster most significantly ELEVATED in `proxy` relative to the
    rest of the bins (one-sided Welch's t, cluster vs. everything else) --
    more robust than a raw mean, which is biased toward small clusters with
    high variance. Returns (winning_cluster_index_into `clusters`, member bin
    indices)."""
    n_total = len(proxy)
    t_scores = []
    for c in clusters:
        mask = np.zeros(n_total, dtype=bool)
        mask[c] = True
        t, _ = ttest_ind(proxy[mask], proxy[~mask], equal_var=False)
        t_scores.append(t if np.isfinite(t) else -np.inf)
    winner = int(np.argmax(t_scores))
    log.info(f"  Cluster proxy t-scores: {[f'{t:.2f}' for t in t_scores]}")
    return winner, clusters[winner]


def _bh_fdr_mask(p_vals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg FDR gate at level `q`. Returns a boolean mask over
    p_vals of the units that survive."""
    m = len(p_vals)
    order = np.argsort(p_vals)
    sorted_p = p_vals[order]
    thresh_line = q * (np.arange(1, m + 1) / m)
    passing = sorted_p <= thresh_line
    mask = np.zeros(m, dtype=bool)
    if passing.any():
        k = np.max(np.where(passing)[0])  # largest index satisfying BH condition
        mask[order[:k + 1]] = True
    return mask


def top_units_for_cluster(member_bins: np.ndarray, sheet: np.ndarray, top_pct: float,
                           mode: str = "topk", fdr_q: float = 0.001) -> np.ndarray:
    """Re-run the Welch's t-test that scored this cluster and return the
    readout cortical-sheet units, in one of two selectable modes:
      - "topk" (default): the top `top_pct`%% highest-|t| units (magnitude-based).
      - "fdr": every unit whose one-sided Welch's-t (cluster-elevated,
        alternative='greater') p-value survives a Benjamini-Hochberg FDR gate
        at `fdr_q` (TopoOmni-paper-exact selection criterion, Sec 4.7.2) --
        falls back to the top-2 |t| units (with a warning) if fewer than 2
        units survive, since a single-feature-per-bin embedding degenerates
        the RDM used downstream."""
    mask = np.zeros(sheet.shape[0], dtype=bool)
    mask[member_bins] = True
    # One-sided (greater): p_vals gates on units where the cluster is ELEVATED
    # vs. the rest, matching the paper's Alg. 1 criterion. The t statistic is
    # unchanged by `alternative`, so the |t|-based topk branch is unaffected.
    t_vals, p_vals = ttest_ind(sheet[mask], sheet[~mask], axis=0, equal_var=False,
                               alternative="greater")
    t_vals = np.nan_to_num(t_vals)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    if mode == "topk":
        n_units = max(1, int(np.ceil(sheet.shape[1] * top_pct / 100.0)))
        return np.argsort(-np.abs(t_vals))[:n_units]
    elif mode == "fdr":
        sig_mask = _bh_fdr_mask(p_vals, fdr_q)
        n_sig = int(sig_mask.sum())
        if n_sig < 2:
            log.warning(f"  FDR-gated selection (q={fdr_q:.1e}) found only {n_sig} unit(s) "
                        f"-- falling back to top-2 |t| units to keep the RDM non-degenerate.")
            return np.argsort(-np.abs(t_vals))[:2]
        return np.where(sig_mask)[0]
    else:
        raise ValueError(f"Unknown unit-selection mode: {mode!r}")


def save_localizer_embedding(pattern: np.ndarray, out_model_name: str,
                              embeddings_dir: str, bin_sec: float, skip_sec: float) -> Path:
    out_dir = _emb_dir(embeddings_dir, out_model_name, bin_sec, skip_sec)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_model_name}_av.npy"
    np.save(out_path, pattern.astype(np.float32))
    log.info(f"  Saved localizer embedding: {out_path}  shape={pattern.shape}")
    return out_path


def save_significant_cluster_readouts(reports: list[dict], clusters: list[np.ndarray],
                                       sheet_av: np.ndarray, base_name: str, p_threshold: float,
                                       embeddings_dir: str, bin_sec: float, skip_sec: float,
                                       top_pct_units: float, unit_selection_mode: str = "topk",
                                       fdr_q: float = 0.001) -> dict:
    """Save one readout per cluster with p_value < p_threshold (name:
    {base_name}_c{ci}), plus one combined readout unioning all of their
    top units (name: {base_name}_all). Mirrors
    topoomni_av_separability_localizer.py's per-cluster/combined convention,
    so every localizer produces a full multi-cluster brain-mappable set
    instead of only the single top-1 winner."""
    sig = [r for r in reports if r["p_value"] < p_threshold]
    log.info(f"  {len(sig)}/{len(reports)} clusters have p < {p_threshold:.1e} "
             f"-- saving individual + combined readouts")
    per_cluster_paths = []
    union_units = set()
    for r in sig:
        ci = r["cluster_idx"]
        c_units = top_units_for_cluster(clusters[ci], sheet_av, top_pct_units,
                                         mode=unit_selection_mode, fdr_q=fdr_q)
        union_units.update(int(u) for u in c_units)
        c_pattern = sheet_av[:, c_units]
        c_model_name = f"{base_name}_c{ci}"
        c_path = save_localizer_embedding(c_pattern, c_model_name, embeddings_dir, bin_sec, skip_sec)
        per_cluster_paths.append(str(c_path))
        log.info(f"    cluster {ci:>3} (size={r['size']:>4} t={r['t_value']:.3f} "
                 f"p={r['p_value']:.2e}): {c_path.name}  shape={c_pattern.shape}")

    union_sorted = np.array(sorted(union_units)) if union_units else np.array([], dtype=int)
    if len(union_sorted) == 0:
        log.warning("  No significant clusters -> union of readout units is empty. "
                    "Skipping combined '_all' embedding (would be a 0-feature .npy).")
        combined_path = None
    else:
        combined_pattern = sheet_av[:, union_sorted]
        combined_name = f"{base_name}_all"
        combined_path = save_localizer_embedding(combined_pattern, combined_name, embeddings_dir, bin_sec, skip_sec)

    return {
        "p_threshold": p_threshold,
        "significant_clusters": sig,
        "per_cluster_embedding_paths": per_cluster_paths,
        "combined_embedding_path": str(combined_path) if combined_path is not None else None,
        "combined_n_units": int(len(union_sorted)),
    }


def main():
    args = parse_args()

    sheet_dir = _emb_dir(args.embeddings_dir, args.sheet_model, args.bin_sec, args.skip_sec)
    sheet_av = np.load(sheet_dir / f"{args.sheet_model}_av.npy").astype(np.float64)
    n_bins, n_units = sheet_av.shape
    log.info(f"Sheet model: {args.sheet_model}  n_bins={n_bins}  n_units={n_units}")

    clu_dir = _emb_dir(args.embeddings_dir, args.cluster_embedding_model, args.bin_sec, args.skip_sec)
    stim_emb = np.load(clu_dir / f"{args.cluster_embedding_model}_{args.cluster_embedding_modality}.npy").astype(np.float64)
    log.info(f"Clustering embedding: {args.cluster_embedding_model}_{args.cluster_embedding_modality}  "
             f"shape={stim_emb.shape}  (independent of {args.sheet_model})")
    assert stim_emb.shape[0] == n_bins, "Stimulus embedding and sheet must share n_bins"

    log.info(f"Running Algorithm 1 (Ward's linkage + top-down early-stopping, "
             f"n_min={args.n_min} n_max={args.n_max}) ...")
    clusters = ward_cluster_and_score(stim_emb, sheet_av, args.n_min, args.n_max)
    sizes = sorted([len(c) for c in clusters], reverse=True)
    log.info(f"  {len(clusters)} terminal clusters. Sizes (top 10): {sizes[:10]}  "
             f"(total bins covered: {sum(sizes)}/{n_bins})")

    reg_dir = _emb_dir(args.embeddings_dir, args.auditory_regressor_model, args.bin_sec, args.skip_sec)
    audio_drive = np.linalg.norm(
        np.load(reg_dir / f"{args.auditory_regressor_model}_a.npy").astype(np.float64), axis=1)

    summary = {
        "sheet_model": args.sheet_model,
        "cluster_embedding": f"{args.cluster_embedding_model}_{args.cluster_embedding_modality}",
        "n_bins": n_bins, "n_units": n_units,
        "n_min": args.n_min, "n_max": args.n_max,
        "n_terminal_clusters": len(clusters),
        "cluster_sizes": sizes,
        "unit_selection_mode": args.unit_selection_mode,
        "fdr_q": args.fdr_q if args.unit_selection_mode == "fdr" else None,
    }

    driver = driver_tag(args.cluster_embedding_model, args.cluster_embedding_modality)
    sheet = model_tag(args.sheet_model)
    suffix = "_fdr" if args.unit_selection_mode == "fdr" else ""
    base_name_aud = loc_base_name("speech", driver, sheet, suffix)
    log.info(f"Naming: driver={driver}  sheet={sheet}  -> {base_name_aud}")

    # ── (a) Positive control: auditory-selective cluster ─────────────────────
    log.info("=" * 70)
    log.info("(a) Positive control: identify auditory-selective terminal cluster")
    idx_aud, bins_aud = identify_cluster_by_proxy(clusters, audio_drive)
    log.info(f"  Winning cluster #{idx_aud}: {len(bins_aud)} bins  "
             f"mean_audio_drive={audio_drive[bins_aud].mean():.4f} "
             f"(overall mean={audio_drive.mean():.4f})")
    units_aud = top_units_for_cluster(bins_aud, sheet_av, args.top_pct_units,
                                       mode=args.unit_selection_mode, fdr_q=args.fdr_q)
    pattern_aud = sheet_av[:, units_aud]
    aud_path = save_localizer_embedding(pattern_aud, base_name_aud,
                                        args.embeddings_dir, args.bin_sec, args.skip_sec)
    summary["speech_localizer"] = {
        "cluster_size": int(len(bins_aud)), "n_units_readout": int(len(units_aud)),
        "mean_audio_drive_in_cluster": float(audio_drive[bins_aud].mean()),
        "mean_audio_drive_overall": float(audio_drive.mean()),
        "embedding_path": str(aud_path),
    }
    reports_aud = cluster_reports_for_proxy(clusters, audio_drive)
    summary["speech_localizer"].update(save_significant_cluster_readouts(
        reports_aud, clusters, sheet_av, base_name_aud,
        args.p_threshold, args.embeddings_dir, args.bin_sec, args.skip_sec, args.top_pct_units,
        unit_selection_mode=args.unit_selection_mode, fdr_q=args.fdr_q))

    out_json = Path(args.embeddings_dir) / summary_json_name("speech", driver, sheet, suffix)
    json.dump(summary, open(out_json, "w"), indent=2)
    log.info(f"Saved cluster summary: {out_json}")
    log.info(f"Done. Next: run rsa/searchlight.py with --model {base_name_aud}, --modality av.")


def _selftest_unit_selection():
    """Sanity check for the two --unit-selection-mode branches: 'topk' returns
    exactly the requested count; 'fdr' returns only units whose p-value
    survives BH-FDR and falls back to 2 units when none/one do."""
    rng = np.random.default_rng(0)
    n_bins, n_units = 200, 50
    sheet = rng.normal(size=(n_bins, n_units))
    member_bins = np.arange(20)
    sheet[member_bins, :5] += 5.0  # 5 units strongly separate this cluster

    topk = top_units_for_cluster(member_bins, sheet, top_pct=10.0, mode="topk")
    assert len(topk) == 5, f"expected ceil(50*0.10)=5 units, got {len(topk)}"

    fdr_units = top_units_for_cluster(member_bins, sheet, top_pct=10.0, mode="fdr", fdr_q=0.001)
    assert set(range(5)).issubset(set(fdr_units.tolist())), "the 5 true signal units must survive FDR"
    assert len(fdr_units) < n_units, "FDR gate should not pass every unit on this synthetic case"

    flat_sheet = rng.normal(size=(n_bins, n_units))  # no signal at all
    fallback_units = top_units_for_cluster(member_bins, flat_sheet, top_pct=10.0, mode="fdr", fdr_q=0.001)
    assert len(fallback_units) == 2, "with no signal, FDR mode must fall back to top-2 |t| units"
    print("topoomni_sheet_localizer self-test OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest_unit_selection()
    else:
        main()
