"""Cluster embedding channels from low-dimensional embeddings of their movie-bin time series.

Channel analogue of ``vertex_clustering.py``: instead of one row
per cortical grayordinate and one column per fMRI TR, each row here is one
embedding channel (1,024 for PE-AV; 2,048 for each Omni-family layer-18
mean-pooled model) and each column is one of the 626 aligned 5-second movie
bins used throughout ``cf_modeling/deprecated/channel_cca_analysis.py``. All
reduction/clustering math (PCA, MDS, Isomap, t-SNE, FastICA, UMAP,
k-means/HDBSCAN/BIRCH) is imported unchanged from ``vertex_clustering.py``.
The only differences are the input load (channel embeddings, not a CIFTI
dtseries) and the label-write step (channels have no grayordinate axis, so
cluster labels are written to CSV instead of ``.dlabel.nii``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DATA, OUTPUTS  # noqa: E402
from cf_modeling.deprecated.channel_cca_analysis import MODEL_CONFIGS, _model_files  # noqa: E402
from rsa.shared.rsa_utils import process_model_embeddings  # noqa: E402
from io_cluster import write_channel_labels_csv  # noqa: E402
from vertex_clustering import (  # noqa: E402
    REDUCTIONS, CLUSTERERS, analysis_tag, reduction_tag, clustering_tag,
    zscore_timeseries_inplace, reduce_grayordinates,
    cluster_embedding, _label_names,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = str(OUTPUTS / "cluster")
FAMILIES = tuple(MODEL_CONFIGS)
EMBEDDINGS_DIR = str(OUTPUTS / "model_embeddings")
TIMING_CSV = str(DATA / "movie_timing.csv")
RUN_TRS = str(
    DATA / "preprocessed/average_sub/hedger_sg_psc/group_average_hedger_sg_psc_run_trs.npy"
)


def channel_model_selection_dir(output_dir, family: str, *,
                                regress_global: bool = False) -> Path:
    """Root of the channel reducer/cluster hyperparameter sweep for one family
    (written by ``channel_timeseries_model_selection.py``; read by it and by
    ``screen_temporal_differentiation.py``)."""
    tag = "norm-zscore_raw"
    if regress_global:
        tag += "_globalregressed"
    return Path(output_dir) / family / "_channel_timeseries_model_selection" / tag


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--timing-csv", default=TIMING_CSV)
    parser.add_argument("--run-trs", default=RUN_TRS)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--reductions", nargs="+", choices=REDUCTIONS,
                        default=list(REDUCTIONS))
    parser.add_argument("--clusterers", nargs="+", choices=CLUSTERERS,
                        default=list(CLUSTERERS))
    parser.add_argument("--n-components", type=int, choices=(2, 3), default=3)
    parser.add_argument("--n-landmarks", type=int, default=2_000,
                        help="Landmarks used by MDS, Isomap, and t-SNE (capped at channel count)")
    parser.add_argument("--extension-neighbors", type=int, default=8)
    parser.add_argument("--isomap-neighbors", type=int, default=15)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1_000)
    parser.add_argument("--mds-max-iterations", type=int, default=300)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--kmeans-clusters", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--birch-threshold", type=float, default=0.5)
    parser.add_argument("--birch-branching-factor", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-zscore-timeseries", action="store_true",
                        help="Do not z-score each channel across movie bins before reduction")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_channel_timeseries(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    """Load the (channels, movie bins) matrix analyzed in channel_cca_analysis.py."""
    model = MODEL_CONFIGS[args.family]
    intact_path, _, _ = _model_files(Path(args.embeddings_dir), model)
    timing = pd.read_csv(args.timing_csv)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    bins_by_channels = process_model_embeddings(
        str(intact_path), timing, args.bin_sec, args.tr, run_trs,
        delay_sec=args.delay_sec, hrf=False, skip_sec=args.skip_sec, normalize=True,
    )
    timeseries = np.ascontiguousarray(bins_by_channels.T.astype(np.float32))
    channel_ids = [f"channel_{i:04d}" for i in range(timeseries.shape[0])]
    return timeseries, channel_ids


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> Path:
    output_root = (
        Path(args.output_dir) / args.family / "_channel_timeseries" / analysis_tag(args)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    features_path = output_root / "channel_timeseries_features.npy"
    features_report_path = output_root / "channel_timeseries_features_report.json"
    channel_ids_path = output_root / "channel_ids.json"
    normalization = (
        "none" if args.no_zscore_timeseries else "within-channel z-score over movie bins"
    )

    if features_path.exists() and features_report_path.exists() and channel_ids_path.exists() and not args.force:
        preprocessing_report = json.loads(features_report_path.read_text())
        log.info("Loading cached channel features: %s", features_path)
        features = np.load(features_path)
        input_shape = preprocessing_report["input_shape_channels_by_bins"]
        n_constant = int(preprocessing_report["n_constant_timeseries"])
        channel_ids = json.loads(channel_ids_path.read_text())
    else:
        log.info("Loading channel time series for family: %s", args.family)
        timeseries, channel_ids = load_channel_timeseries(args)
        input_shape = [int(timeseries.shape[0]), int(timeseries.shape[1])]

        n_constant = 0
        if not args.no_zscore_timeseries:
            log.info("Z-scoring each of %d channel time series", timeseries.shape[0])
            n_constant = zscore_timeseries_inplace(timeseries)
            if n_constant:
                log.warning("Converted %d constant time series to zero", n_constant)

        features = timeseries
        preprocessing_report = {
            "family": args.family,
            "input_shape_channels_by_bins": input_shape,
            "timeseries_normalization": normalization,
            "n_constant_timeseries": n_constant,
        }
        np.save(features_path, features)
        _json_dump(features_report_path, preprocessing_report)
        channel_ids_path.write_text(json.dumps(channel_ids, indent=2) + "\n")

    if features.ndim != 2 or features.shape[0] != len(channel_ids) or not np.isfinite(features).all():
        raise ValueError(f"Channel-features cache is invalid: {features_path}")

    manifest: dict[str, Any] = {
        "analysis": "channel_timeseries_dimensionality_reduction_clustering",
        "family": args.family,
        "input_shape_channels_by_bins": input_shape,
        "timeseries_normalization": normalization,
        "n_constant_timeseries": n_constant,
        "feature_space": "original movie-bin time series; no preliminary PCA",
        "arguments": vars(args),
        "results": {},
    }

    for reduction in args.reductions:
        reduce_tag = reduction_tag(reduction, args)
        reduction_dir = output_root / reduce_tag
        reduction_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = reduction_dir / "channel_components.npy"
        reduction_report_path = reduction_dir / "reduction_report.json"

        if embedding_path.exists() and reduction_report_path.exists() and not args.force:
            log.info("Loading cached %s embedding", reduction)
            embedding = np.load(embedding_path)
            reduction_info = json.loads(reduction_report_path.read_text())
        else:
            log.info("Computing %s embedding", reduction)
            embedding, reduction_info = reduce_grayordinates(
                features,
                method=reduction,
                n_components=args.n_components,
                n_landmarks=args.n_landmarks,
                extension_neighbors=args.extension_neighbors,
                isomap_neighbors=args.isomap_neighbors,
                tsne_perplexity=args.tsne_perplexity,
                tsne_iterations=args.tsne_iterations,
                mds_max_iterations=args.mds_max_iterations,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                umap_metric=args.umap_metric,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            np.save(embedding_path, embedding)
            _json_dump(reduction_report_path, reduction_info)

        if embedding.shape != (features.shape[0], args.n_components):
            raise ValueError(f"Cached embedding has unexpected shape: {embedding_path}")
        manifest["results"][reduction] = {
            "embedding": str(embedding_path),
            "reduction_report": str(reduction_report_path),
            "clusterings": {},
        }

        for clusterer in args.clusterers:
            config = f"{reduce_tag}_{clustering_tag(clusterer, args)}"
            result_dir = reduction_dir / config
            result_dir.mkdir(parents=True, exist_ok=True)
            labels_path = result_dir / "channel_labels.npy"
            csv_path = result_dir / "channel_labels.csv"
            report_path = result_dir / "channel_report.json"

            complete = labels_path.exists() and csv_path.exists() and report_path.exists()
            if complete and not args.force:
                log.info("Clustering cached: %s / %s", reduction, clusterer)
            else:
                log.info("Clustering %s embedding with %s", reduction, clusterer)
                labels, report = cluster_embedding(embedding, clusterer, args)
                np.save(labels_path, labels)
                remapped = write_channel_labels_csv(
                    labels, channel_ids, str(csv_path),
                    label_names=_label_names(labels), map_name=config,
                )
                report.update(
                    reduction=reduction,
                    n_components=args.n_components,
                    raw_to_csv_key={
                        str(raw): int(key)
                        for key, raw in enumerate(
                            sorted(int(x) for x in np.unique(labels) if x != -1), start=1
                        )
                    },
                    csv_unassigned_key=0,
                    csv_unique_keys=[int(x) for x in np.unique(remapped)],
                    files={"raw_labels": str(labels_path), "csv": str(csv_path)},
                )
                _json_dump(report_path, report)

            manifest["results"][reduction]["clusterings"][clusterer] = {
                "labels": str(labels_path),
                "csv": str(csv_path),
                "report": str(report_path),
                "map_name": config,
            }
            _json_dump(manifest_path, manifest)

    log.info("Channel-timeseries sweep complete: %s", output_root)
    return output_root


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
