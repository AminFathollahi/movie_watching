"""Cluster grayordinates from low-dimensional embeddings of their fMRI time series.

This analysis is intentionally separate from ``run_cluster.py``.  The latter
crosses temporal stimulus states with one spatial brain parcellation; this
module compares spatial parcellations obtained from six reductions of the
same grayordinate-by-time matrix:

* PCA
* metric MDS (landmark fit + k-NN out-of-sample extension)
* Isomap (landmark fit + native out-of-sample transform)
* t-SNE (landmark fit + k-NN out-of-sample extension)
* FastICA
* UMAP (landmark fit + k-NN out-of-sample extension)

Only vertices inside the stimulus-driven mask (``--mask-cifti``, value >
``--mask-threshold``) enter reduction and clustering. Each embedding is
clustered with four-cluster k-means, HDBSCAN, and BIRCH without a requested
number of clusters. Every clustering is saved both as a raw NumPy label
vector and as a one-map Workbench ``.dlabel.nii``, both using the same
convention: 0 = outside the mask, 1..K = clusters, and a reserved key above K
for HDBSCAN noise among masked-in vertices when present.

MDS cannot be fit exactly to a typical 108k-grayordinate CIFTI because its
distance matrix is quadratic.  Scikit-learn t-SNE also has no ``transform``.
The landmark strategy keeps both methods tractable while still assigning an
embedding and a cluster label to every grayordinate.  The approximation is
recorded in each reduction report and in the run manifest.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import Birch, HDBSCAN, KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cifti_io import get_bm_axis, load_cifti_data, load_single_map  # noqa: E402
from io_cluster import GROUP_AVG_CIFTI, write_dlabel  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = "/home/amin/Research/Representation/Movie/outputs/cluster"
REDUCTIONS = ("pca", "mds", "isomap", "tsne", "fastica", "umap")
CLUSTERERS = ("kmeans", "hdbscan", "birch")
METRIC_SAMPLE_CAP = 5_000
MASK_CIFTI = (
    "/home/amin/Research/Representation/Movie/outputs/sitmulus_regressor_cifti/"
    "HCP_movie_stimulus_correlation_5sdelay_normalized.dscalar.nii"
)
MASK_THRESHOLD = 0.0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-cifti", default=GROUP_AVG_CIFTI,
                        help="Input dtseries; on disk it must have shape time x grayordinate")
    parser.add_argument("--template-cifti", default=None,
                        help="CIFTI providing the output BrainModelAxis; defaults to input")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--analysis-label", default="group_average",
                        help="First output path component (for example group_average or a subject ID)")
    parser.add_argument("--mask-cifti", default=MASK_CIFTI,
                        help="Single-map CIFTI; vertices with value > mask-threshold enter the pipeline")
    parser.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--reductions", nargs="+", choices=REDUCTIONS,
                        default=list(REDUCTIONS))
    parser.add_argument("--clusterers", nargs="+", choices=CLUSTERERS,
                        default=list(CLUSTERERS))
    parser.add_argument("--n-components", type=int, choices=(2, 3), default=3)
    parser.add_argument("--n-landmarks", type=int, default=2_000,
                        help="Landmarks used by MDS, Isomap, and t-SNE")
    parser.add_argument("--extension-neighbors", type=int, default=8,
                        help="Neighbors in the MDS/t-SNE out-of-sample extension")
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
                        help="Do not z-score each grayordinate across time before reduction")
    parser.add_argument("--force", action="store_true",
                        help="Recompute outputs even when all files for a result exist")
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "n_landmarks": args.n_landmarks,
        "extension_neighbors": args.extension_neighbors,
        "isomap_neighbors": args.isomap_neighbors,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_iterations": args.tsne_iterations,
        "mds_max_iterations": args.mds_max_iterations,
        "umap_neighbors": args.umap_neighbors,
        "kmeans_clusters": args.kmeans_clusters,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "birch_threshold": args.birch_threshold,
        "birch_branching_factor": args.birch_branching_factor,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError(f"Arguments must be positive: {', '.join(bad)}")
    if args.umap_min_dist < 0:
        raise ValueError("umap_min_dist must be non-negative")


def _float_tag(value: float) -> str:
    """Format a float for the repository's filename-safe config tags."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def analysis_tag(args: argparse.Namespace) -> str:
    norm = "raw" if args.no_zscore_timeseries else "zscore"
    return (
        f"norm-{norm}_prepca{args.pre_pca_components}_nc{args.n_components}"
        f"_landmarks{args.n_landmarks}"
    )


def reduction_tag(method: str, args: argparse.Namespace) -> str:
    """Reduction tag matching ``run_cluster.py``'s spatial config grammar.

    Method-specific values are included whenever changing them changes the
    embedding.  This prevents a resume run from silently reusing an embedding
    made under different landmark/manifold settings.
    """
    base = f"sreduce-{method}_snc{args.n_components}"
    if method == "mds":
        return (
            f"{base}_landmarks{args.n_landmarks}_extk{args.extension_neighbors}"
            f"_iter{args.mds_max_iterations}"
        )
    if method == "isomap":
        return f"{base}_landmarks{args.n_landmarks}_nn{args.isomap_neighbors}"
    if method == "tsne":
        return (
            f"{base}_landmarks{args.n_landmarks}_perp{_float_tag(args.tsne_perplexity)}"
            f"_extk{args.extension_neighbors}_iter{args.tsne_iterations}"
        )
    if method == "umap":
        return (
            f"{base}_landmarks{args.n_landmarks}_nn{args.umap_neighbors}"
            f"_mindist{_float_tag(args.umap_min_dist)}_metric-{args.umap_metric}"
            f"_extk{args.extension_neighbors}"
        )
    if method in ("pca", "fastica"):
        return base
    raise ValueError(f"Unknown reduction method: {method}")


def clustering_tag(method: str, args: argparse.Namespace) -> str:
    if method == "kmeans":
        return f"scluster-kmeans_k{args.kmeans_clusters}"
    if method == "hdbscan":
        return f"scluster-hdbscan_mcs{args.min_cluster_size}_ms{args.min_samples}"
    if method == "birch":
        return (
            f"scluster-birch_threshold{_float_tag(args.birch_threshold)}"
            f"_bf{args.birch_branching_factor}"
        )
    raise ValueError(f"Unknown clustering method: {method}")


def zscore_timeseries_inplace(timeseries: np.ndarray) -> int:
    """Z-score each row over time in-place; constant rows become all zero.

    Returns the number of constant rows.  In-place processing avoids a second
    full copy of the approximately 1.6 GB group-average matrix.
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected a 2-D grayordinate x time matrix, got {timeseries.shape}")
    if not np.issubdtype(timeseries.dtype, np.floating):
        raise TypeError("Timeseries must have a floating dtype for in-place z-scoring")
    if not np.isfinite(timeseries).all():
        raise ValueError("Input timeseries contains NaN or infinite values")

    means = timeseries.mean(axis=1, dtype=np.float64).astype(timeseries.dtype, copy=False)
    timeseries -= means[:, None]
    scales = timeseries.std(axis=1, dtype=np.float64).astype(timeseries.dtype, copy=False)
    constant = scales <= np.finfo(timeseries.dtype).eps
    scales[constant] = 1.0
    timeseries /= scales[:, None]
    timeseries[constant] = 0.0
    return int(constant.sum())


def load_stimulus_mask(mask_path: str, threshold: float, n_vertices: int) -> np.ndarray:
    """Boolean mask of stimulus-driven vertices (mask value > threshold)."""
    values = load_single_map(mask_path)
    if values.shape[0] != n_vertices:
        raise ValueError(
            f"Mask has {values.shape[0]} vertices, expected {n_vertices}: {mask_path}"
        )
    mask = values > threshold
    if not mask.any():
        raise ValueError(f"Stimulus mask selects zero vertices: {mask_path} > {threshold}")
    return mask


def expand_masked_labels(raw_labels: np.ndarray,
                         mask: np.ndarray) -> tuple[np.ndarray, dict[int, str], int | None]:
    """Place cluster labels fit on masked-in vertices back into full grayordinate length.

    Returns a length-``len(mask)`` array using ``write_dlabel``'s own convention
    (-1 = unassigned, remapped to key 0): vertices outside the mask are -1, and
    HDBSCAN noise (-1) among masked-in vertices is remapped to one past the
    largest cluster id so it always sorts into its own reserved key, after
    every real cluster and distinct from 0.
    """
    raw_labels = np.asarray(raw_labels)
    cluster_ids = sorted(int(v) for v in np.unique(raw_labels) if v != -1)
    noise_present = bool((raw_labels == -1).any())
    noise_value = (cluster_ids[-1] + 1) if cluster_ids else 0
    filled = np.where(raw_labels == -1, noise_value, raw_labels).astype(np.int64)
    full = np.full(mask.shape[0], -1, dtype=np.int64)
    full[mask] = filled
    names = {0: "not_stimulus_driven"}
    names.update({i + 1: f"cluster_{i + 1}" for i in range(len(cluster_ids))})
    noise_key = len(cluster_ids) + 1 if noise_present else None
    if noise_key is not None:
        names[noise_key] = "noise"
    return full, names, noise_key


def fit_preliminary_pca(timeseries: np.ndarray, n_components: int,
                        random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the shared randomized PCA used for denoising and tractability."""
    k = min(n_components, timeseries.shape[0], timeseries.shape[1])
    if k < 2:
        raise ValueError(f"Need at least two preliminary components, got {k}")
    model = PCA(n_components=k, svd_solver="randomized", random_state=random_state)
    features = model.fit_transform(timeseries).astype(np.float32, copy=False)
    info = {
        "method": "pca",
        "n_components": int(k),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(
            model.explained_variance_ratio_
        ).tolist(),
    }
    return features, info


def choose_landmarks(n_samples: int, n_landmarks: int,
                     random_state: int) -> np.ndarray:
    """Choose a deterministic, sorted simple-random landmark subset."""
    size = min(n_samples, n_landmarks)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=size, replace=False))


def _knn_extension(source: np.ndarray, landmark_indices: np.ndarray,
                   landmark_embedding: np.ndarray, n_neighbors: int,
                   n_jobs: int) -> np.ndarray:
    """Extend a landmark-only embedding to all rows by distance-weighted k-NN."""
    k = min(n_neighbors, len(landmark_indices))
    regressor = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=n_jobs)
    regressor.fit(source[landmark_indices], landmark_embedding)
    embedding = regressor.predict(source).astype(np.float32, copy=False)
    # Preserve fitted landmark coordinates exactly (also handles duplicate points).
    embedding[landmark_indices] = landmark_embedding
    return embedding


def _transform_in_chunks(model: Any, source: np.ndarray,
                         chunk_size: int = 10_000) -> np.ndarray:
    """Apply a manifold transform without materializing every query at once."""
    chunks = []
    for start in range(0, source.shape[0], chunk_size):
        chunks.append(model.transform(source[start:start + chunk_size]))
    return np.vstack(chunks).astype(np.float32, copy=False)


def reduce_grayordinates(
    features: np.ndarray,
    method: str,
    n_components: int = 3,
    n_landmarks: int = 2_000,
    extension_neighbors: int = 8,
    isomap_neighbors: int = 15,
    tsne_perplexity: float = 30.0,
    tsne_iterations: int = 1_000,
    mds_max_iterations: int = 300,
    fastica_algorithm: str = "parallel",
    fastica_fun: str = "logcosh",
    fastica_max_iterations: int = 1_000,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    random_state: int = 0,
    n_jobs: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce every grayordinate to two or three coordinates.

    ``features`` contains one row per vertex. The returned
    matrix always has the same number and order of rows.
    """
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("features must be a finite 2-D matrix")
    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    if n_components > features.shape[1]:
        raise ValueError("n_components exceeds the input feature dimension")
    if method not in REDUCTIONS:
        raise ValueError(f"Unknown reduction method: {method}")

    info: dict[str, Any] = {
        "method": method,
        "n_components": n_components,
        "input_shape": list(features.shape),
        "random_state": random_state,
    }

    if method == "pca":
        model = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
        embedding = model.fit_transform(features)
        info.update(
            implementation="randomized PCA fitted directly to input features",
            explained_variance_ratio=model.explained_variance_ratio_.tolist(),
            cumulative_explained_variance=np.cumsum(
                model.explained_variance_ratio_
            ).tolist(),
            normalized_reconstruction_mse=float(
                1.0 - np.sum(model.explained_variance_ratio_)
            ),
        )

    elif method == "fastica":
        model = FastICA(
            n_components=n_components,
            whiten="unit-variance",
            algorithm=fastica_algorithm,
            fun=fastica_fun,
            max_iter=fastica_max_iterations,
            tol=1e-4,
            random_state=random_state,
        )
        embedding = model.fit_transform(features)
        reconstructed = model.inverse_transform(embedding)
        denominator = float(np.var(features, dtype=np.float64))
        info.update(
            implementation="FastICA fitted directly to input features",
            algorithm=fastica_algorithm,
            fun=fastica_fun,
            max_iter=fastica_max_iterations,
            n_iter=int(model.n_iter_),
            normalized_reconstruction_mse=float(
                np.mean((features - reconstructed) ** 2, dtype=np.float64)
                / denominator
            ),
        )

    else:
        landmark_indices = choose_landmarks(
            features.shape[0], n_landmarks, random_state
        )
        landmarks = features[landmark_indices]
        if len(landmark_indices) <= n_components:
            raise ValueError("Need more landmarks than requested output components")
        info.update(
            n_landmarks=int(len(landmark_indices)),
            landmark_selection="simple random sample without replacement",
        )

        if method == "mds":
            model = MDS(
                n_components=n_components,
                metric=True,
                n_init=1,
                max_iter=mds_max_iterations,
                eps=1e-3,
                dissimilarity="euclidean",
                random_state=random_state,
                n_jobs=n_jobs,
            )
            landmark_embedding = model.fit_transform(landmarks)
            embedding = _knn_extension(
                features, landmark_indices, landmark_embedding,
                extension_neighbors, n_jobs,
            )
            info.update(
                implementation="metric MDS landmark fit with distance-weighted k-NN extension",
                extension_neighbors=min(extension_neighbors, len(landmark_indices)),
                stress=float(model.stress_),
                normalized_stress=float(
                    np.sqrt(model.stress_ / np.sum(pdist(landmarks) ** 2))
                ),
                n_iter=int(model.n_iter_),
            )

        elif method == "isomap":
            neighbors = min(isomap_neighbors, len(landmark_indices) - 1)
            model = Isomap(
                n_neighbors=neighbors,
                n_components=n_components,
                eigen_solver="auto",
                n_jobs=n_jobs,
            )
            landmark_embedding = model.fit_transform(landmarks)
            embedding = _transform_in_chunks(model, features)
            embedding[landmark_indices] = landmark_embedding
            info.update(
                implementation="landmark Isomap with native geodesic out-of-sample transform",
                isomap_neighbors=neighbors,
                reconstruction_error=float(model.reconstruction_error()),
            )

        elif method == "tsne":
            perplexity = min(float(tsne_perplexity), max(1.0, len(landmark_indices) - 1.0))
            tsne_kwargs: dict[str, Any] = {
                "n_components": n_components,
                "perplexity": perplexity,
                "init": "pca",
                "learning_rate": "auto",
                "method": "barnes_hut" if n_components <= 3 else "exact",
                "random_state": random_state,
                "n_jobs": n_jobs,
            }
            # scikit-learn renamed n_iter to max_iter after the repo's pinned 1.3.2.
            if "max_iter" in inspect.signature(TSNE).parameters:
                tsne_kwargs["max_iter"] = tsne_iterations
            else:
                tsne_kwargs["n_iter"] = tsne_iterations
            model = TSNE(**tsne_kwargs)
            landmark_embedding = model.fit_transform(landmarks)
            embedding = _knn_extension(
                features, landmark_indices, landmark_embedding,
                extension_neighbors, n_jobs,
            )
            info.update(
                implementation=(
                    f"{tsne_kwargs['method']} t-SNE landmark fit with "
                    "distance-weighted k-NN extension"
                ),
                extension_neighbors=min(extension_neighbors, len(landmark_indices)),
                perplexity=perplexity,
                kl_divergence=float(model.kl_divergence_),
                n_iter=int(model.n_iter_),
            )

        else:  # umap
            numba_cache = Path(tempfile.gettempdir()) / "movie_watching_numba_cache"
            numba_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
            try:
                from umap import UMAP
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise ImportError(
                    "UMAP reduction requires the 'umap-learn' package"
                ) from exc
            neighbors = min(int(umap_neighbors), len(landmark_indices) - 1)
            model = UMAP(
                n_neighbors=neighbors,
                n_components=n_components,
                min_dist=float(umap_min_dist),
                metric=umap_metric,
                random_state=random_state,
                n_jobs=n_jobs,
                transform_seed=random_state,
                low_memory=True,
            )
            landmark_embedding = model.fit_transform(landmarks)
            embedding = _knn_extension(
                features, landmark_indices, landmark_embedding,
                extension_neighbors, n_jobs,
            )
            info.update(
                implementation=(
                    "landmark UMAP with distance-weighted k-NN "
                    "out-of-sample extension"
                ),
                umap_neighbors=neighbors,
                min_dist=float(umap_min_dist),
                metric=umap_metric,
                extension_neighbors=min(extension_neighbors, len(landmark_indices)),
            )

    embedding = np.asarray(embedding, dtype=np.float32)
    expected = (features.shape[0], n_components)
    if embedding.shape != expected or not np.isfinite(embedding).all():
        raise RuntimeError(
            f"{method} returned invalid embedding {embedding.shape}; expected {expected}"
        )
    info["output_shape"] = list(embedding.shape)
    return embedding, info


def cluster_embedding(embedding: np.ndarray, method: str,
                      args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardize an embedding and cluster all of its rows."""
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("embedding must be a finite 2-D matrix")
    scaled = StandardScaler().fit_transform(embedding).astype(np.float32, copy=False)

    if method == "kmeans":
        estimator = KMeans(
            n_clusters=args.kmeans_clusters,
            n_init=20,
            random_state=args.random_state,
        )
        params = {"n_clusters": args.kmeans_clusters, "n_init": 20}
    elif method == "hdbscan":
        estimator = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        params = {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
        }
    elif method == "birch":
        estimator = Birch(
            threshold=args.birch_threshold,
            branching_factor=args.birch_branching_factor,
            n_clusters=None,
        )
        params = {
            "threshold": args.birch_threshold,
            "branching_factor": args.birch_branching_factor,
            "n_clusters": None,
        }
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = estimator.fit_predict(scaled).astype(np.int32, copy=False)
    report = clustering_report(scaled, labels, args.random_state)
    report.update(method=method, parameters=params, input_standardization="per component z-score")
    return labels, report


def clustering_report(features: np.ndarray, labels: np.ndarray,
                      random_state: int = 0) -> dict[str, Any]:
    """Return scalable cluster sizes and internal-validation diagnostics."""
    labels = np.asarray(labels)
    if labels.shape != (features.shape[0],):
        raise ValueError("labels length does not match feature rows")
    assigned = labels != -1
    unique = np.unique(labels[assigned])
    sizes = {str(int(key)): int(np.sum(labels == key)) for key in unique}
    report: dict[str, Any] = {
        "n_clusters": int(len(unique)),
        "n_assigned": int(assigned.sum()),
        "n_noise": int((~assigned).sum()),
        "noise_fraction": float(np.mean(~assigned)),
        "cluster_sizes_raw_labels": sizes,
        "metrics_sample_size": 0,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }
    if len(unique) < 2 or assigned.sum() <= len(unique):
        return report

    rng = np.random.default_rng(random_state)
    indices = np.flatnonzero(assigned)
    if len(indices) > METRIC_SAMPLE_CAP:
        indices = rng.choice(indices, size=METRIC_SAMPLE_CAP, replace=False)
    sample_labels = labels[indices]
    # A random subsample may omit tiny clusters; validation needs at least two.
    if len(np.unique(sample_labels)) < 2 or len(np.unique(sample_labels)) >= len(indices):
        return report
    sample = features[indices]
    report.update(
        metrics_sample_size=int(len(indices)),
        silhouette=float(silhouette_score(sample, sample_labels)),
        calinski_harabasz=float(calinski_harabasz_score(sample, sample_labels)),
        davies_bouldin=float(davies_bouldin_score(sample, sample_labels)),
    )
    return report


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _input_description(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _label_names(labels: np.ndarray) -> dict[int, str]:
    """Names keyed by the consecutive values written by write_dlabel."""
    n_clusters = len(np.unique(labels[labels != -1]))
    names = {0: "unassigned"}
    names.update({key: f"cluster_{key}" for key in range(1, n_clusters + 1)})
    return names


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_cifti)
    template_path = Path(args.template_cifti or args.input_cifti)
    for role, path in (("input", input_path), ("template", template_path)):
        if not path.exists():
            raise FileNotFoundError(f"{role.capitalize()} CIFTI does not exist: {path}")

    norm_tag = "raw" if args.no_zscore_timeseries else "zscore"
    output_root = (
        Path(args.output_dir) / args.analysis_label / "_vertex"
        / f"norm-{norm_tag}_raw" / f"fixed_nc{args.n_components}_landmarks{args.n_landmarks}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    input_info = _input_description(input_path)
    bm_axis = get_bm_axis(str(template_path))
    normalization = (
        "none" if args.no_zscore_timeseries else "within-vertex z-score over time"
    )

    log.info("Loading vertex time features: %s", input_path)
    features = load_cifti_data(str(input_path))
    if features.shape[0] != len(bm_axis):
        raise ValueError(
            f"Input has {features.shape[0]} vertices but template BrainModelAxis has {len(bm_axis)}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Input CIFTI contains NaN or infinite values")
    input_shape = [int(features.shape[0]), int(features.shape[1])]

    mask = load_stimulus_mask(args.mask_cifti, args.mask_threshold, features.shape[0])
    n_masked_in = int(mask.sum())
    log.info("Stimulus mask: %d/%d vertices masked in (%s > %s)",
             n_masked_in, len(mask), args.mask_cifti, args.mask_threshold)
    features = features[mask]

    n_constant = 0
    if not args.no_zscore_timeseries:
        log.info("Z-scoring each of %d masked-in vertex time profiles", features.shape[0])
        n_constant = zscore_timeseries_inplace(features)

    manifest: dict[str, Any] = {
        "analysis": "vertex_raw_feature_dimensionality_reduction_clustering",
        "analysis_label": args.analysis_label,
        "input": input_info,
        "template_cifti": str(template_path.resolve()),
        "input_shape_vertices_by_time": input_shape,
        "timeseries_normalization": normalization,
        "n_constant_timeseries": n_constant,
        "feature_space": "original fMRI time points; no preliminary PCA",
        "stimulus_mask": {
            "mask_cifti": str(Path(args.mask_cifti).resolve()),
            "mask_threshold": args.mask_threshold,
            "n_vertices_total": int(len(mask)),
            "n_masked_in": n_masked_in,
            "n_masked_out": int(len(mask) - n_masked_in),
        },
        "label_convention": "0 = outside stimulus mask; 1..K = clusters; "
                            "reserved key above K = HDBSCAN noise, when present",
        "arguments": vars(args),
        "results": {},
    }

    for reduction in args.reductions:
        reduce_tag = reduction_tag(reduction, args)
        reduction_dir = output_root / reduce_tag
        reduction_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = reduction_dir / "spatial_vertex_components.npy"
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
            labels_path = result_dir / "spatial_vertex_labels.npy"
            dlabel_path = result_dir / "spatial_vertex_labels.dlabel.nii"
            report_path = result_dir / "spatial_report.json"

            complete = labels_path.exists() and dlabel_path.exists() and report_path.exists()
            if complete and not args.force:
                log.info("Clustering cached: %s / %s", reduction, clusterer)
            else:
                log.info("Clustering %s embedding with %s", reduction, clusterer)
                labels, report = cluster_embedding(embedding, clusterer, args)
                full_raw, label_names, noise_label = expand_masked_labels(labels, mask)
                remapped = write_dlabel(
                    full_raw,
                    str(template_path),
                    str(dlabel_path),
                    label_names=label_names,
                    map_name=config,
                )
                np.save(labels_path, remapped)
                report.update(
                    reduction=reduction,
                    n_components=args.n_components,
                    n_masked_in=n_masked_in,
                    n_masked_out=int(len(mask) - n_masked_in),
                    dlabel_unassigned_key=0,
                    noise_label=noise_label,
                    label_table=label_names,
                    dlabel_unique_keys=[int(x) for x in np.unique(remapped)],
                    files={
                        "raw_labels": str(labels_path),
                        "dlabel": str(dlabel_path),
                    },
                )
                _json_dump(report_path, report)

            manifest["results"][reduction]["clusterings"][clusterer] = {
                "labels": str(labels_path),
                "dlabel": str(dlabel_path),
                "report": str(report_path),
                "map_name": config,
            }
            _json_dump(manifest_path, manifest)

    log.info("Vertex sweep complete: %s", output_root)
    return output_root


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
