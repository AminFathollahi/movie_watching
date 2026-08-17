"""
cluster/reduce.py
==================
Dimensionality reduction for the temporal (stimulus embedding) and spatial
(brain vertex) axes of the cluster pipeline. PCA is the default and only
method run this pass; tphate/phate are wired but import lazily so their
absence never blocks a pca/pca run (they are not installed yet).
"""

import logging

import numpy as np
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def reduce_temporal(emb: np.ndarray, method: str = "pca",
                    n_components: int = 50, adv_n_components: int = 10):
    """Reduce (n_segments, n_features) stimulus embeddings for HMM state-fitting.

    Returns
    -------
    features : (n_segments, k) float32
    info     : dict with per-component and cumulative explained variance (pca only)
    """
    log.info(f"reduce_temporal: input={emb.shape}  method={method}")
    info = {"method": method}

    if method == "pca":
        k = min(n_components, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
        features = pca.fit_transform(emb)
        evr = pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        log.info(f"  PCA per-component EVR: {np.round(evr, 4).tolist()}")
        log.info(f"  PCA cumulative EVR: {np.round(cum_evr, 4).tolist()}")
        info.update(n_components=k, explained_variance_ratio=evr.tolist(),
                    cumulative_evr=cum_evr.tolist())
    elif method == "tphate":
        import tphate  # lazy — deferred this pass, not installed
        model = tphate.TPHATE(n_components=adv_n_components)
        features = model.fit_transform(emb)
        info.update(n_components=adv_n_components)
    else:
        raise ValueError(f"Unknown temporal reduction method: {method}")

    log.info(f"reduce_temporal: output={features.shape}")
    return features.astype(np.float32), info


def reduce_spatial(X: np.ndarray, method: str = "pca",
                   n_components: int = 100, adv_n_components: int = 10):
    """Reduce (V, T) vertex timeseries over T so each vertex -> k features.

    Returns
    -------
    features : (V, k) float32
    info     : dict with per-component and cumulative explained variance (pca only)
    """
    log.info(f"reduce_spatial: input={X.shape}  method={method}")
    info = {"method": method}

    if method == "pca":
        # PCA default capped at 100 for HDBSCAN density; this is BELOW the
        # variance-optimal k (EVR < ~full) — a deliberate density/speed tradeoff.
        k = min(n_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
        features = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        log.info(f"  PCA cumulative EVR (last 5): {np.round(cum_evr[-5:], 4).tolist()}")
        log.info(f"  PCA total cumulative EVR: {cum_evr[-1]:.4f}")
        info.update(n_components=k, explained_variance_ratio=evr.tolist(),
                    cumulative_evr=cum_evr.tolist())
    elif method == "phate":
        import phate  # lazy — deferred this pass, not installed
        model = phate.PHATE(n_components=adv_n_components, n_landmark=2000)
        features = model.fit_transform(X)
        info.update(n_components=adv_n_components)
    else:
        raise ValueError(f"Unknown spatial reduction method: {method}")

    log.info(f"reduce_spatial: output={features.shape}")
    return features.astype(np.float32), info


def demo():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(626, 2048)).astype(np.float32)
    feats, info = reduce_temporal(emb, method="pca", n_components=50)
    assert feats.shape == (626, 50)
    assert abs(info["cumulative_evr"][-1] - sum(info["explained_variance_ratio"])) < 1e-6
    print(f"[demo] reduce_temporal OK: {feats.shape}  cum_evr[-1]={info['cumulative_evr'][-1]:.4f}")

    X = rng.normal(size=(1000, 3655)).astype(np.float32)
    feats_s, info_s = reduce_spatial(X, method="pca", n_components=100)
    assert feats_s.shape == (1000, 100)
    print(f"[demo] reduce_spatial OK: {feats_s.shape}  cum_evr[-1]={info_s['cumulative_evr'][-1]:.4f}")


if __name__ == "__main__":
    demo()
