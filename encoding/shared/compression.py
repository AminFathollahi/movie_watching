from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def spherical_channel_labels(
    train: np.ndarray, n_clusters: int, random_state: int = 0
) -> np.ndarray:
    """Cluster channels by their standardized training-set time courses."""
    from sklearn.cluster import KMeans

    train = np.array(train, dtype=np.float64, copy=True)
    if not 1 <= n_clusters <= train.shape[1]:
        raise ValueError("n_clusters must be between 1 and the channel count")
    if n_clusters == 1:
        return np.zeros(train.shape[1], dtype=int)
    channel_profiles = train.T
    channel_profiles -= channel_profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(channel_profiles, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    channel_profiles /= norms
    return KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=random_state,
    ).fit_predict(channel_profiles)


@dataclass
class FittedCompression:
    method: str
    dimension: int
    train: np.ndarray
    test: np.ndarray
    provenance: dict
    arrays: dict[str, np.ndarray]


def fit_compression(
    train: np.ndarray,
    test: np.ndarray,
    method: str,
    dimension: int | None = None,
    random_state: int = 0,
) -> FittedCompression:
    """Fit a representation transform on training samples and apply it to test."""
    train = np.asarray(train)
    test = np.asarray(test)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test must be 2D arrays with matching channels")
    if train.shape[0] < 2 or test.shape[0] < 1:
        raise ValueError("compression needs at least two train and one test sample")
    train_z, test_z, mean, scale = _standardize_train_test(train, test)
    n_features = train_z.shape[1]
    if method == "full":
        dimension = n_features
        return FittedCompression(
            method, dimension, train_z.astype(np.float32), test_z.astype(np.float32),
            {"method": method, "dimension": dimension},
            {"feature_mean": mean, "feature_scale": scale},
        )
    if dimension is None or not 1 <= dimension <= min(train_z.shape):
        raise ValueError("dimension must be between 1 and min(n_train, n_features)")

    arrays: dict[str, np.ndarray] = {
        "feature_mean": mean,
        "feature_scale": scale,
    }
    provenance = {
        "method": method,
        "dimension": int(dimension),
        "random_state": int(random_state),
    }

    if method == "pca":
        from sklearn.decomposition import PCA

        transform = PCA(n_components=dimension, svd_solver="full")
        train_out = transform.fit_transform(train_z)
        test_out = transform.transform(test_z)
        arrays.update(components=transform.components_, component_mean=transform.mean_)
        provenance["explained_variance_ratio"] = transform.explained_variance_ratio_.tolist()
    elif method == "random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        transform = GaussianRandomProjection(
            n_components=dimension, random_state=random_state
        )
        train_out = transform.fit_transform(train_z)
        test_out = transform.transform(test_z)
        arrays["components"] = np.asarray(transform.components_)
    elif method in {"cluster_mean", "cluster_pc1"}:
        labels = spherical_channel_labels(train_z, dimension, random_state)
        arrays["channel_labels"] = labels
        train_parts = []
        test_parts = []
        pc1 = np.zeros((dimension, n_features), dtype=np.float64)
        pc1_means = np.zeros(n_features, dtype=np.float64)
        for cluster in range(dimension):
            members = labels == cluster
            if method == "cluster_mean":
                train_parts.append(train_z[:, members].mean(axis=1))
                test_parts.append(test_z[:, members].mean(axis=1))
            else:
                from sklearn.decomposition import PCA

                component = PCA(n_components=1, svd_solver="full")
                train_parts.append(component.fit_transform(train_z[:, members])[:, 0])
                test_parts.append(component.transform(test_z[:, members])[:, 0])
                pc1[cluster, members] = component.components_[0]
                pc1_means[members] = component.mean_
        train_out = np.column_stack(train_parts)
        test_out = np.column_stack(test_parts)
        if method == "cluster_pc1":
            arrays["cluster_pc1_components"] = pc1
            arrays["cluster_pc1_means"] = pc1_means
        provenance["cluster_sizes"] = np.bincount(labels, minlength=dimension).tolist()
    else:
        raise ValueError(f"Unknown compression method: {method}")

    train_out, test_out, output_mean, output_scale = _standardize_train_test(
        train_out, test_out
    )
    arrays["output_mean"] = output_mean
    arrays["output_scale"] = output_scale
    return FittedCompression(
        method,
        int(dimension),
        train_out.astype(np.float32),
        test_out.astype(np.float32),
        provenance,
        arrays,
    )


def heldout_cluster_diagnostics(
    heldout: np.ndarray, labels: np.ndarray
) -> dict[str, float | int | list[int]]:
    """Measure coherence and reconstruction of frozen channel memberships."""
    heldout = np.asarray(heldout, dtype=np.float64)
    labels = np.asarray(labels)
    if heldout.ndim != 2 or labels.shape != (heldout.shape[1],):
        raise ValueError("heldout must be samples by channels with one label per channel")

    heldout_z, _, _, _ = _standardize_train_test(heldout, heldout)
    cluster_ids = np.unique(labels)
    coherence = []
    reconstruction = np.empty_like(heldout_z)
    cluster_means = []
    for cluster in cluster_ids:
        members = labels == cluster
        mean_series = heldout_z[:, members].mean(axis=1)
        reconstruction[:, members] = mean_series[:, None]
        cluster_means.append(mean_series)
        if members.sum() < 2:
            coherence.append(np.nan)
        else:
            corr = np.corrcoef(heldout_z[:, members], rowvar=False)
            upper = corr[np.triu_indices_from(corr, k=1)]
            coherence.append(float(np.nanmean(upper)))

    cluster_means = np.column_stack(cluster_means)
    if cluster_means.shape[1] > 1:
        between = np.corrcoef(cluster_means, rowvar=False)
        between_abs = np.abs(between[np.triu_indices_from(between, k=1)])
        mean_between = float(np.nanmean(between_abs))
    else:
        mean_between = 0.0
    singular = np.linalg.svd(cluster_means, compute_uv=False)
    singular_energy = np.square(singular).sum()
    effective_dim = (
        float((singular.sum() ** 2) / singular_energy)
        if singular_energy > 1e-12 else 0.0
    )
    return {
        "n_clusters": int(len(cluster_ids)),
        "cluster_sizes": [int((labels == cluster).sum()) for cluster in cluster_ids],
        "mean_within_cluster_correlation": float(np.nanmean(coherence)),
        "mean_absolute_between_cluster_correlation": mean_between,
        "reconstruction_mse": float(np.mean((heldout_z - reconstruction) ** 2)),
        "effective_dimension": effective_dim,
    }
