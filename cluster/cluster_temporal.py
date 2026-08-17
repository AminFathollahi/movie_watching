"""
cluster/cluster_temporal.py
============================
Gaussian HMM state-fitting on reduced temporal (stimulus embedding) features.
Run boundaries (lengths=) keep transitions from crossing between movie runs.
hmmlearn has no n_init kwarg — n_init is implemented by refitting with
different random_state and keeping the best-scoring restart.
"""

import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _fit_restarts(features, lengths, n_states, covariance_type, n_init, random_state, n_iter=200):
    """Fit n_init GaussianHMMs with different seeds; return them sorted best-first."""
    fits = []
    for i in range(n_init):
        model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                            n_iter=n_iter, random_state=random_state + i)
        model.fit(features, lengths)
        score = model.score(features, lengths)
        fits.append((score, model))
    fits.sort(key=lambda t: t[0], reverse=True)
    return fits


def fit_hmm(features: np.ndarray, lengths: np.ndarray, n_states: int = 10,
           covariance_type: str = "diag", n_init: int = 10, random_state: int = 0):
    """Fit a Gaussian HMM with run-boundary-respecting transitions.

    Returns
    -------
    dict with labels (n_seg,), posteriors (n_seg, K), model, loglik
    """
    log.info(f"fit_hmm: features={features.shape}  lengths={list(lengths)}  "
             f"n_states={n_states}  n_init={n_init}  covariance_type={covariance_type}")
    fits = _fit_restarts(features, lengths, n_states, covariance_type, n_init, random_state)
    best_score, best_model = fits[0]
    labels = best_model.predict(features, lengths)
    posteriors = best_model.predict_proba(features, lengths)
    log.info(f"fit_hmm: best_loglik={best_score:.2f}  labels={labels.shape}  posteriors={posteriors.shape}")
    return {"labels": labels, "posteriors": posteriors, "model": best_model, "loglik": best_score}


def sweep_states(features: np.ndarray, lengths: np.ndarray, k_min: int = 2, k_max: int = 15,
                 n_init: int = 10, covariance_type: str = "diag", random_state: int = 0):
    """DIAGNOSTIC sweep over K (n_states). Not branched on this pass.

    For each K: leave-one-run-out CV log-likelihood (primary), full-fit BIC
    (secondary), and restart-stability ARI between the best and 2nd-best
    restart of the full fit.

    Returns
    -------
    dict of {k: {...}} plus a "k_values" list, JSON-serializable.
    """
    lengths = np.asarray(lengths)
    n_runs = len(lengths)
    run_starts = np.concatenate([[0], np.cumsum(lengths)])[:-1]
    run_slices = [slice(int(s), int(s + l)) for s, l in zip(run_starts, lengths)]

    results = {}
    for k in range(k_min, k_max + 1):
        log.info(f"sweep_states: K={k}")

        # Leave-one-run-out CV log-likelihood
        cv_lls = []
        for held_idx in range(n_runs):
            train_slices = [run_slices[i] for i in range(n_runs) if i != held_idx]
            train_feats = np.concatenate([features[s] for s in train_slices], axis=0)
            train_lengths = [lengths[i] for i in range(n_runs) if i != held_idx]
            test_feats = features[run_slices[held_idx]]
            test_lengths = [lengths[held_idx]]

            model = GaussianHMM(n_components=k, covariance_type=covariance_type,
                                n_iter=200, random_state=random_state)
            try:
                model.fit(train_feats, train_lengths)
                ll = model.score(test_feats, test_lengths)
            except ValueError as e:
                log.warning(f"  K={k} held_run={held_idx}: fit failed ({e}); skipping")
                continue
            cv_lls.append(ll)

        # Full-fit BIC + restart-stability ARI
        fits = _fit_restarts(features, lengths, k, covariance_type, n_init, random_state)
        best_score, best_model = fits[0]
        bic = best_model.bic(features, lengths)

        if len(fits) >= 2:
            _, second_model = fits[1]
            labels_best = best_model.predict(features, lengths)
            labels_second = second_model.predict(features, lengths)
            restart_ari = float(adjusted_rand_score(labels_best, labels_second))
        else:
            restart_ari = float("nan")

        results[k] = {
            "cv_loglik_mean": float(np.mean(cv_lls)) if cv_lls else float("nan"),
            "cv_loglik_per_run": [float(x) for x in cv_lls],
            "bic": float(bic),
            "restart_ari": restart_ari,
        }
        log.info(f"  K={k}: cv_ll_mean={results[k]['cv_loglik_mean']:.2f}  "
                 f"bic={bic:.2f}  restart_ari={restart_ari:.4f}")

    results["k_values"] = list(range(k_min, k_max + 1))
    return results


def demo():
    rng = np.random.default_rng(0)
    lengths = [40, 35, 38]
    n_seg = sum(lengths)
    # 3 well-separated Gaussian blobs, sequential within each run -> HMM should recover 3 states
    true_states = np.concatenate([rng.integers(0, 3, l) for l in lengths])
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    features = centers[true_states] + rng.normal(scale=0.5, size=(n_seg, 2))
    features = features.astype(np.float32)

    result = fit_hmm(features, lengths, n_states=3, n_init=3, random_state=0)
    assert result["labels"].shape == (n_seg,)
    assert result["posteriors"].shape == (n_seg, 3)
    ari = adjusted_rand_score(true_states, result["labels"])
    assert ari > 0.8, f"expected HMM to recover well-separated states, ARI={ari:.3f}"
    print(f"[demo] fit_hmm OK: ARI vs ground truth = {ari:.3f}")

    sweep = sweep_states(features, lengths, k_min=2, k_max=4, n_init=2)
    assert set(sweep["k_values"]) == {2, 3, 4}
    for k in sweep["k_values"]:
        assert "cv_loglik_mean" in sweep[k] and "bic" in sweep[k]
    print(f"[demo] sweep_states OK: {[(k, round(sweep[k]['bic'], 1)) for k in sweep['k_values']]}")


if __name__ == "__main__":
    demo()
