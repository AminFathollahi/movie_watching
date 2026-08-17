"""
cluster/run_cluster.py
========================
Single-(model, modality) driver for the dual-clustering pipeline: HMM states
over reduced stimulus-embedding features (temporal axis) crossed against
HDBSCAN networks over reduced brain-vertex features (spatial axis), related
via a block-permutation interaction test. Mirrors subcortical_rsa.py's
structure (group-average loading, config-tag naming, skip-if-exists guard).

Spatial clustering is brain-only (modality-invariant), so it is computed
ONCE per spatial_config and cached under
  {OUTPUT_DIR}/group_average/_spatial/{spatial_config}/
then reused by every model/modality's interaction step.
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
sys.path.insert(0, str(ROOT / "rsa"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_cluster import (  # noqa: E402
    GROUP_AVG_CIFTI, GROUP_AVG_TRS, load_group_average, write_dlabel,
    preprocess_fmri, process_model_embeddings, get_run_bin_counts, align_and_assert_bins,
)
from reduce import reduce_temporal, reduce_spatial  # noqa: E402
from cluster_temporal import fit_hmm, sweep_states  # noqa: E402
from cluster_spatial import cluster_spatial, spatial_report  # noqa: E402
from interaction import interaction_matrix, block_permutation_test  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TIMING_CSV = "/home/amin/Research/Representation/Movie/data/movie_timing.csv"
EMBEDDINGS_DIR = "/home/amin/Research/Representation/Movie/outputs/model_embeddings"
OUTPUT_DIR = "/home/amin/Research/Representation/Movie/outputs/cluster"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--group-avg-cifti", default=GROUP_AVG_CIFTI)
    p.add_argument("--group-avg-trs", default=GROUP_AVG_TRS)
    p.add_argument("--timing-csv", default=TIMING_CSV)
    p.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    p.add_argument("--template-cifti", default=GROUP_AVG_CIFTI)
    p.add_argument("--output-dir", default=OUTPUT_DIR)

    p.add_argument("--mode", default="groupaverage", choices=["groupaverage", "persubject"])
    p.add_argument("--subject", default="group_average")
    p.add_argument("--subjects-list", default=None)

    p.add_argument("--model", required=True)
    p.add_argument("--modality", required=True, choices=["a", "v", "av"])

    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--skip-sec", type=float, default=5.0)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)

    p.add_argument("--temporal-reduction", default="pca", choices=["pca", "tphate"])
    p.add_argument("--temporal-n-components", type=int, default=50)
    p.add_argument("--adv-n-components", type=int, default=10)

    p.add_argument("--spatial-reduction", default="pca", choices=["pca", "phate"])
    p.add_argument("--spatial-n-components", type=int, default=100)

    p.add_argument("--spatial-cluster", default="hdbscan", choices=["hdbscan", "nmf"])
    p.add_argument("--min-cluster-size", type=int, default=100)
    p.add_argument("--min-samples", type=int, default=10)

    p.add_argument("--n-temporal-states", type=int, default=10)
    p.add_argument("--hmm-covariance", default="diag")
    p.add_argument("--hmm-n-init", type=int, default=10)

    p.add_argument("--select-states", action="store_true")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=15)

    p.add_argument("--temporal-source", default="embedding", choices=["embedding", "brain"])

    p.add_argument("--stability", action="store_true")

    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--perm-correction", default="fdr", choices=["fdr", "maxstat"])

    p.add_argument("--nmf-rank", type=int, default=20)

    return p.parse_args()


def _int_tag(x: float) -> int:
    return int(x)


def _spatial_config_tag(args) -> str:
    b, s, d = _int_tag(args.bin_sec), _int_tag(args.skip_sec), _int_tag(args.delay_sec)
    return (f"sreduce-{args.spatial_reduction}_snc{args.spatial_n_components}"
            f"_scluster-{args.spatial_cluster}_mcs{args.min_cluster_size}_ms{args.min_samples}"
            f"_bin{b}s_skip{s}s_delay{d}s")


def _config_tag(args, n_states: int) -> str:
    b, s, d = _int_tag(args.bin_sec), _int_tag(args.skip_sec), _int_tag(args.delay_sec)
    return (f"treduce-{args.temporal_reduction}_tnc{args.temporal_n_components}_K{n_states}"
            f"_sreduce-{args.spatial_reduction}_snc{args.spatial_n_components}"
            f"_scluster-{args.spatial_cluster}_bin{b}s_skip{s}s_delay{d}s")


# =============================================================================
# Spatial clustering — computed once, cached, reused across model/modality
# =============================================================================

def _get_spatial(args, X: np.ndarray, template_cifti: str):
    spatial_config = _spatial_config_tag(args)
    spatial_root = Path(args.output_dir) / "group_average" / "_spatial" / spatial_config
    spatial_root.mkdir(parents=True, exist_ok=True)

    labels_path = spatial_root / "spatial_vertex_labels.npy"
    dlabel_path = spatial_root / "spatial_vertex_labels.dlabel.nii"
    report_path = spatial_root / "spatial_report.json"

    if labels_path.exists() and dlabel_path.exists() and report_path.exists():
        log.info(f"Spatial clustering cached: {spatial_root}")
        vertex_labels = np.load(str(labels_path))
        report = json.loads(report_path.read_text())
        return vertex_labels, report, spatial_config

    log.info(f"Spatial clustering not cached — computing: {spatial_config}")
    spatial_feats, spatial_info = reduce_spatial(
        X, method=args.spatial_reduction, n_components=args.spatial_n_components,
        adv_n_components=args.adv_n_components)

    vertex_labels = cluster_spatial(
        spatial_feats, method=args.spatial_cluster,
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
        nmf_rank=args.nmf_rank)

    report = spatial_report(spatial_feats, vertex_labels)
    report["reduction"] = spatial_info
    report["reduced_spatial_shape"] = list(spatial_feats.shape)

    np.save(str(labels_path), vertex_labels)
    write_dlabel(vertex_labels, template_cifti, str(dlabel_path))
    report_path.write_text(json.dumps(report, indent=2))
    log.info(f"Spatial clustering saved: {spatial_root}")
    return vertex_labels, report, spatial_config


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.mode == "persubject":
        log.warning("mode=persubject is a stub this pass — not implemented. Exiting.")
        return

    bin_int, skip_int, delay_int = _int_tag(args.bin_sec), _int_tag(args.skip_sec), _int_tag(args.delay_sec)

    config = _config_tag(args, args.n_temporal_states)
    out_root = Path(args.output_dir) / "group_average" / f"{args.model}_{args.modality}" / config
    out_root.mkdir(parents=True, exist_ok=True)

    labels_out = out_root / "temporal_state_labels.npy"
    posteriors_out = out_root / "temporal_state_posteriors.npy"
    diagnostics_out = out_root / "temporal_hmm_diagnostics.json"
    interaction_out = out_root / "interaction_matrix.npy"
    manifest_out = out_root / "manifest.json"

    if all(p.exists() for p in [labels_out, posteriors_out, interaction_out, manifest_out]):
        log.info(f"Outputs already exist — skipping: {out_root}")
        return

    # ── 1. Load group-average brain data ────────────────────────────────────
    X, run_trs, bm_axis = load_group_average(args.group_avg_cifti, args.group_avg_trs)
    timing_df = pd.read_csv(args.timing_csv)

    # ── 2. Segment grid ──────────────────────────────────────────────────────
    seg_lengths = get_run_bin_counts(timing_df, run_trs, args.bin_sec, args.tr,
                                     delay_sec=args.delay_sec, skip_sec=args.skip_sec)
    n_seg = int(seg_lengths.sum())
    log.info(f"Segment grid: seg_lengths={seg_lengths.tolist()}  n_seg={n_seg}  "
             f"(equivalent TR count per segment: {int(round(args.bin_sec / args.tr))})")

    # ── 3. Bin fMRI to the segment grid ─────────────────────────────────────
    fmri_binned = preprocess_fmri(X, timing_df, run_trs, args.bin_sec, args.tr,
                                  args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    log.info(f"fmri_binned: {fmri_binned.shape}")

    # ── 4. Spatial clustering (cached, shared across model/modality) ───────
    vertex_labels, spatial_rep, spatial_config = _get_spatial(args, X, args.template_cifti)

    # ── 5. Temporal states (per model/modality) ─────────────────────────────
    emb_file = (Path(args.embeddings_dir) / args.model / f"bin{bin_int}s_skip{skip_int}s" /
                f"{args.model}_{args.modality}.npy")
    emb = process_model_embeddings(str(emb_file), timing_df, bin_sec=args.bin_sec, tr=args.tr,
                                   run_trs=run_trs, delay_sec=args.delay_sec, hrf=False,
                                   skip_sec=args.skip_sec, normalize=True)
    log.info(f"emb: {emb.shape}")
    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)

    if args.temporal_source == "embedding":
        temporal_source_data = emb
    else:
        log.warning("temporal-source=brain: interaction test will be descriptive-only (no permutation).")
        temporal_source_data = fmri_binned

    temporal_feats, temporal_info = reduce_temporal(
        temporal_source_data, method=args.temporal_reduction,
        n_components=args.temporal_n_components, adv_n_components=args.adv_n_components)

    # Diagnostic-only sweep — --select-states is wired for a future pass but
    # n_temporal_states is authoritative here regardless of its value.
    diagnostics = sweep_states(temporal_feats, seg_lengths, k_min=args.k_min, k_max=args.k_max,
                               n_init=args.hmm_n_init, covariance_type=args.hmm_covariance)
    diagnostics_out.write_text(json.dumps(diagnostics, indent=2))

    hmm_result = fit_hmm(temporal_feats, seg_lengths, n_states=args.n_temporal_states,
                         covariance_type=args.hmm_covariance, n_init=args.hmm_n_init)
    state_labels = hmm_result["labels"]
    posteriors = hmm_result["posteriors"]

    np.save(str(labels_out), state_labels)
    np.save(str(posteriors_out), posteriors)

    # ── 6. Interaction ───────────────────────────────────────────────────────
    if args.temporal_source == "embedding":
        M, pvals, perm_stats = block_permutation_test(
            fmri_binned, state_labels, vertex_labels, seg_lengths,
            n_perm=args.n_permutations, correction=args.perm_correction)
        np.save(str(out_root / "interaction_pvals_fdr.npy"), pvals)
        (out_root / "interaction_stats.json").write_text(json.dumps(perm_stats, indent=2))
        n_sig = int(perm_stats["n_sig_q05"])
    else:
        M, _, _ = interaction_matrix(fmri_binned, state_labels, vertex_labels)
        pvals = None
        n_sig = 0
    np.save(str(interaction_out), M)

    # ── 7. Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "args": vars(args),
        "spatial_config": spatial_config,
        "config": config,
        "shapes": {
            "X": list(X.shape),
            "fmri_binned": list(fmri_binned.shape),
            "emb": list(emb.shape),
            "temporal_features": list(temporal_feats.shape),
            "reduced_spatial": spatial_rep.get("reduced_spatial_shape"),
            "state_labels": list(state_labels.shape),
            "posteriors": list(posteriors.shape),
            "interaction_matrix": list(M.shape),
        },
        "temporal_cumulative_evr": temporal_info.get("cumulative_evr", [None])[-1]
                                    if temporal_info.get("cumulative_evr") else None,
        "spatial_cumulative_evr": (spatial_rep.get("reduction", {}).get("cumulative_evr", [None])[-1]
                                    if spatial_rep.get("reduction", {}).get("cumulative_evr") else None),
        "chosen_n_temporal_states": args.n_temporal_states,
        "n_networks": spatial_rep["n_networks"],
        "noise_fraction": spatial_rep["noise_fraction"],
        "n_interaction_cells_fdr_sig": n_sig,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest saved: {manifest_out}")
    log.info(f"SUMMARY model={args.model} modality={args.modality}: "
             f"n_networks={manifest['n_networks']}  noise_fraction={manifest['noise_fraction']:.4f}  "
             f"K={args.n_temporal_states}  n_sig_cells_fdr={n_sig}")


if __name__ == "__main__":
    main()
