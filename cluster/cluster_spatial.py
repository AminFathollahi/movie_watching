"""
cluster/cluster_spatial.py
============================
Vertex-level spatial clustering into brain "networks". HDBSCAN (sklearn's
built-in, since 1.3 — no standalone `hdbscan` package) is the default and
only method run this pass. NMF is a documented alternative for signed fMRI
data (pos/neg split), wired but not run.
"""

import logging

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SILHOUETTE_SUBSAMPLE_CAP = 10_000  # ponytail: silhouette on 108k pts is O(n^2); cap the sample


def cluster_spatial(features: np.ndarray, method: str = "hdbscan",
                    min_cluster_size: int = 100, min_samples: int = 10,
                    nmf_rank: int = 20, random_state: int = 0):
    """Cluster vertices (V, k) into spatial networks. Returns labels (V,), noise=-1.
    """
    log.info(f"cluster_spatial: features={features.shape}  method={method}")

    if method == "hdbscan":
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(features)
    elif method == "nmf":
        # DOCUMENTED ALTERNATIVE (not run this pass): fMRI is signed, so split
        # into positive/negative parts before NMF (which requires X >= 0),
        # then take the argmax over W as the hard label. Rank selection would
        # use silhouette + cophenetic/consensus stability (not implemented —
        # nmf_rank is fixed here).
        from sklearn.decomposition import NMF
        X_split = np.concatenate([np.clip(features, 0, None), np.clip(-features, 0, None)], axis=1)
        nmf = NMF(n_components=nmf_rank, init="nndsvda", random_state=random_state, max_iter=500)
        W = nmf.fit_transform(X_split)  # soft graded membership (V, nmf_rank)
        labels = W.argmax(axis=1)       # hard label
        log.info(f"  NMF soft membership W={W.shape}; hard labels via argmax")
    else:
        raise ValueError(f"Unknown spatial cluster method: {method}")

    log.info(f"cluster_spatial: labels={labels.shape}  "
             f"n_unique={len(np.unique(labels))}  noise_frac={np.mean(labels == -1):.4f}")
    return labels.astype(int)


def spatial_report(features: np.ndarray, labels: np.ndarray, random_state: int = 0) -> dict:
    """Summarize a spatial clustering: n_networks, noise_fraction, silhouette, sizes."""
    non_noise = labels != -1
    uniq = np.unique(labels[non_noise])
    n_networks = len(uniq)
    noise_fraction = float(np.mean(labels == -1))
    sizes = {int(u): int(np.sum(labels == u)) for u in uniq}

    if n_networks >= 2 and non_noise.sum() >= 2:
        rng = np.random.default_rng(random_state)
        idx = np.where(non_noise)[0]
        if len(idx) > SILHOUETTE_SUBSAMPLE_CAP:
            idx = rng.choice(idx, size=SILHOUETTE_SUBSAMPLE_CAP, replace=False)
        silhouette = float(silhouette_score(features[idx], labels[idx]))
    else:
        silhouette = float("nan")

    report = {
        "n_networks": n_networks,
        "noise_fraction": noise_fraction,
        "silhouette_subsampled": silhouette,
        "network_sizes": sizes,
    }
    log.info(f"spatial_report: n_networks={n_networks}  noise_fraction={noise_fraction:.4f}  "
             f"silhouette={silhouette:.4f}")
    return report


def demo():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [20, 0], [0, 20]], dtype=float)
    true_labels = rng.integers(0, 3, 300)
    features = centers[true_labels] + rng.normal(scale=1.0, size=(300, 2))
    features = features.astype(np.float32)

    labels = cluster_spatial(features, method="hdbscan", min_cluster_size=20, min_samples=5)
    assert labels.shape == (300,)
    report = spatial_report(features, labels)
    assert report["n_networks"] >= 2, "expected HDBSCAN to find >=2 well-separated blobs"
    assert report["silhouette_subsampled"] > 0.5, "well-separated blobs should have high silhouette"
    print(f"[demo] cluster_spatial + spatial_report OK: {report}")


if __name__ == "__main__":
    demo()
