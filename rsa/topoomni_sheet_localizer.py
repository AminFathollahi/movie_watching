"""
rsa/topoomni_sheet_localizer.py
=================================
Move 5 — Topo-Omni cortical-sheet localizer, faithfully replicating
Algorithm 1 / papers/Topo-omni.pdf Sec 2.6 & 4.7.2.

REWRITE NOTE: an earlier version of this script clustered cortical-SHEET
UNITS by their response profile. Re-reading the paper's actual Algorithm 1
showed that is backwards: Alg. 1 clusters STIMULI (video/movie segments) via
Ward's linkage on an INDEPENDENT semantic embedding (the paper uses
omni-embed-nemotron-3b), scores each candidate cluster by a Welch's t-test
computed PER CORTICAL-SHEET UNIT (in-cluster vs out-of-cluster stimuli),
summarized as the median t-value across units, then does a top-down
dendrogram traversal with selectivity-based early stopping (exact pseudocode
in Sec 4.7.2, reproduced in ward_cluster_and_score() / _split() below). This
version replaces the unit-clustering approach with the paper's actual
stimulus-clustering procedure.

Cross-model independence (avoids using a model to validate itself):
  - Stimulus-clustering embedding : an INDEPENDENT model's embedding, not
    topoomni's own. Default: pe-av-small-16-frame's _event_t (ModernBERT
    text embedding of the per-bin caption/transcript "event" text) --
    independent of Topo-Omni's own audio/video pathways, and (per user
    design) the general convention going forward: use a DIFFERENT model's
    embedding to drive each model's own sheet-selectivity analysis (e.g. use
    PE-AV to drive omni3b/topoomni sheet analyses; use omni3b to drive any
    future PE-AV sheet analysis), with text (_event_t) embeddings as a
    shared baseline narrative comparison across all of them.
  - Cortical response : topoomni_layer{N}_sheet_av.npy (the model's OWN
    joint-condition cortical sheet -- this is what Alg. 1 actually scores
    against, matching the paper).

Two localizers (both use the SAME clustering + scoring machinery; they
differ only in how the winning terminal cluster is IDENTIFIED post-hoc,
also via independent signals, matching how the paper post-hoc identifies
its "faces" cluster by inspecting cluster content):
  (a) POSITIVE CONTROL -- auditory/voice-selective cluster. Among the
      terminal clusters from Alg. 1, pick the one whose member bins have the
      highest mean AudioMAE per-bin L2-norm (an independent, non-topoomni
      audio-content proxy). Validates the pipeline: should land on
      auditory/STG cortex.
  (b) AV-INTEGRATION LOCALIZER. Among the SAME terminal clusters, pick the
      one whose member bins have the highest mean per-bin norm of the Move-1
      interaction residual (rsa.multimodal_decomposition.compute_interaction_residual_cv
      on PE-AV-small-16-frame's av/a/v -- the already-validated, non-degenerate
      fusion signal from Move 1) -- an independent "how much genuine AV
      fusion content is in this bin" proxy that does not depend on topoomni.

For each winning cluster, the RSA "model embedding" fed to
rsa/searchlight.py is built from the TOP-|t| cortical-sheet units that drove
that cluster's score (the same units used to compute the winning score),
read out over ALL 626 bins -- i.e. the model's own predicted topographic
territory for that cluster, in a form with enough feature dimensions
(>= 2 units) for a valid correlation-distance RDM (see the module-level
note in the pre-rewrite git history: a single-feature-per-bin embedding
degenerates the RDM).

Usage
-----
python rsa/topoomni_sheet_localizer.py \\
    --sheet-model topoomni_layer18_sheet \\
    --cluster-embedding-model pe-av-small-16-frame --cluster-embedding-modality event_t \\
    --auditory-regressor-model audiomae \\
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
from rsa.multimodal_decomposition import compute_interaction_residual_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet-model", required=True, help="e.g. topoomni_layer18_sheet")
    p.add_argument("--cluster-embedding-model", default="pe-av-small-16-frame",
                   dest="cluster_embedding_model",
                   help="INDEPENDENT model whose embedding drives Ward's-linkage "
                        "stimulus clustering (must not be --sheet-model's own family).")
    p.add_argument("--cluster-embedding-modality", default="event_t",
                   dest="cluster_embedding_modality")
    p.add_argument("--auditory-regressor-model", default="audiomae",
                   dest="auditory_regressor_model")
    p.add_argument("--integration-model", default="pe-av-small-16-frame",
                   dest="integration_model",
                   help="Native-AV model whose Move-1 interaction residual (av/a/v) "
                        "identifies the AV-integration terminal cluster.")
    p.add_argument("--embeddings-dir", required=True, dest="embeddings_dir")
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--n-min", type=int, default=10, dest="n_min")
    p.add_argument("--n-max", type=int, default=375, dest="n_max")
    p.add_argument("--top-pct-units", type=float, default=1.0, dest="top_pct_units",
                   help="Fraction of highest-|t| sheet units to read out as the RSA embedding.")
    p.add_argument("--output-tag", default="", dest="output_tag",
                   help="Suffix appended to output model names/summary json so multiple "
                        "driving-embedding runs (e.g. text vs av vs clsav_from_a/v) don't "
                        "overwrite each other. Empty (default) preserves the original "
                        "unsuffixed topoomni_auditory_localizer/topoomni_integration_localizer "
                        "naming for backward compatibility with the text-driven run already "
                        "in the report.")
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


def top_units_for_cluster(member_bins: np.ndarray, sheet: np.ndarray, top_pct: float) -> np.ndarray:
    """Re-run the Welch's t-test that scored this cluster and return the indices
    of the top `top_pct`%% |t|-value cortical-sheet units."""
    mask = np.zeros(sheet.shape[0], dtype=bool)
    mask[member_bins] = True
    t_vals, _ = ttest_ind(sheet[mask], sheet[~mask], axis=0, equal_var=False)
    t_vals = np.nan_to_num(t_vals)
    n_units = max(1, int(np.ceil(sheet.shape[1] * top_pct / 100.0)))
    return np.argsort(-np.abs(t_vals))[:n_units]


def save_localizer_embedding(pattern: np.ndarray, out_model_name: str,
                              embeddings_dir: str, bin_sec: float, skip_sec: float) -> Path:
    out_dir = _emb_dir(embeddings_dir, out_model_name, bin_sec, skip_sec)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_model_name}_av.npy"
    np.save(out_path, pattern.astype(np.float32))
    log.info(f"  Saved localizer embedding: {out_path}  shape={pattern.shape}")
    return out_path


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

    int_dir = _emb_dir(args.embeddings_dir, args.integration_model, args.bin_sec, args.skip_sec)
    a_emb  = np.load(int_dir / f"{args.integration_model}_a.npy").astype(np.float64)
    v_emb  = np.load(int_dir / f"{args.integration_model}_v.npy").astype(np.float64)
    av_emb = np.load(int_dir / f"{args.integration_model}_av.npy").astype(np.float64)
    R, ms_score, best_alpha = compute_interaction_residual_cv(av_emb, [a_emb, v_emb])
    integration_drive = np.linalg.norm(R, axis=1)
    log.info(f"Integration-drive proxy from {args.integration_model}: "
             f"alpha={best_alpha:.2e} ms_score={ms_score:.4f}")

    summary = {
        "sheet_model": args.sheet_model,
        "cluster_embedding": f"{args.cluster_embedding_model}_{args.cluster_embedding_modality}",
        "n_bins": n_bins, "n_units": n_units,
        "n_min": args.n_min, "n_max": args.n_max,
        "n_terminal_clusters": len(clusters),
        "cluster_sizes": sizes,
    }

    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""

    # ── (a) Positive control: auditory-selective cluster ─────────────────────
    log.info("=" * 70)
    log.info("(a) Positive control: identify auditory-selective terminal cluster")
    idx_aud, bins_aud = identify_cluster_by_proxy(clusters, audio_drive)
    log.info(f"  Winning cluster #{idx_aud}: {len(bins_aud)} bins  "
             f"mean_audio_drive={audio_drive[bins_aud].mean():.4f} "
             f"(overall mean={audio_drive.mean():.4f})")
    units_aud = top_units_for_cluster(bins_aud, sheet_av, args.top_pct_units)
    pattern_aud = sheet_av[:, units_aud]
    aud_path = save_localizer_embedding(pattern_aud, f"topoomni_auditory_localizer{tag_suffix}",
                                        args.embeddings_dir, args.bin_sec, args.skip_sec)
    summary["auditory_localizer"] = {
        "cluster_size": int(len(bins_aud)), "n_units_readout": int(len(units_aud)),
        "mean_audio_drive_in_cluster": float(audio_drive[bins_aud].mean()),
        "mean_audio_drive_overall": float(audio_drive.mean()),
        "embedding_path": str(aud_path),
    }

    # ── (b) AV-integration localizer ─────────────────────────────────────────
    log.info("=" * 70)
    log.info("(b) AV-integration localizer: identify integration-selective terminal cluster")
    idx_int, bins_int = identify_cluster_by_proxy(clusters, integration_drive)
    log.info(f"  Winning cluster #{idx_int}: {len(bins_int)} bins  "
             f"mean_integration_drive={integration_drive[bins_int].mean():.4f} "
             f"(overall mean={integration_drive.mean():.4f})")
    units_int = top_units_for_cluster(bins_int, sheet_av, args.top_pct_units)
    pattern_int = sheet_av[:, units_int]
    int_path = save_localizer_embedding(pattern_int, f"topoomni_integration_localizer{tag_suffix}",
                                        args.embeddings_dir, args.bin_sec, args.skip_sec)
    summary["integration_localizer"] = {
        "cluster_size": int(len(bins_int)), "n_units_readout": int(len(units_int)),
        "mean_integration_drive_in_cluster": float(integration_drive[bins_int].mean()),
        "mean_integration_drive_overall": float(integration_drive.mean()),
        "embedding_path": str(int_path),
        "same_cluster_as_auditory": bool(idx_int == idx_aud),
        "unit_overlap_with_auditory_readout": int(len(np.intersect1d(units_aud, units_int))),
    }

    out_json = Path(args.embeddings_dir) / f"_topoomni_localizer_cluster_summary{tag_suffix}.json"
    json.dump(summary, open(out_json, "w"), indent=2)
    log.info(f"Saved cluster summary: {out_json}")
    log.info("Done. Next: run rsa/searchlight.py with --model topoomni_auditory_localizer "
             "and --model topoomni_integration_localizer, --modality av.")


if __name__ == "__main__":
    main()
