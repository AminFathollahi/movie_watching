from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "cpu"):
        array = array.cpu().numpy()
    return np.asarray(array)


def _r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be matching samples-by-target arrays")
    residual = np.square(y_true - y_pred).sum(axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    total = np.square(centered).sum(axis=0)
    score = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(residual, total, out=score, where=total > 1e-12)
    score[total > 1e-12] = 1.0 - score[total > 1e-12]
    return score


def run_splitter(groups: np.ndarray):
    from sklearn.model_selection import PredefinedSplit

    groups = np.asarray(groups)
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("At least two training runs are required for inner CV")
    fold_ids = np.searchsorted(unique, groups)
    return PredefinedSplit(fold_ids)


def fit_group_ridge(
    train_bands: list[np.ndarray],
    y_train: np.ndarray,
    test_bands: list[np.ndarray],
    train_runs: np.ndarray,
    alphas: np.ndarray,
    n_iter: int = 20,
    backend: str = "torch_cuda",
    random_state: int = 0,
):
    import torch
    from himalaya.backend import set_backend
    from himalaya.ridge import GroupRidgeCV

    selected_backend = backend
    if backend == "torch_cuda" and not torch.cuda.is_available():
        selected_backend = "torch"
    set_backend(selected_backend, on_error="warn")

    y_train = np.asarray(y_train, dtype=np.float32)
    y_mean = y_train.mean(axis=0, keepdims=True)
    model = GroupRidgeCV(
        groups="input",
        solver="random_search",
        solver_params={
            "alphas": np.asarray(alphas),
            "n_iter": n_iter,
            "n_targets_batch": 2000,
            "n_targets_batch_refit": 2000,
            "progress_bar": False,
        },
        cv=run_splitter(train_runs),
        fit_intercept=False,
        random_state=random_state,
        Y_in_cpu=selected_backend == "torch_cuda",
    )
    model.fit(train_bands, y_train - y_mean)
    prediction = _to_numpy(model.predict(test_bands)).astype(np.float32) + y_mean
    return prediction, model, selected_backend, y_mean


def _pearson_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = y_true - y_true.mean(axis=0)
    y_pred = y_pred - y_pred.mean(axis=0)
    denominator = np.sqrt(np.square(y_true).sum(axis=0) * np.square(y_pred).sum(axis=0))
    score = np.full(denominator.shape, np.nan)
    np.divide((y_true * y_pred).sum(axis=0), denominator, out=score, where=denominator > 1e-12)
    return score


BANDS = ("a", "v", "j")
ALL_SUBSETS = ("a", "v", "j", "av", "aj", "vj", "avj")


def partition_variance(r2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gallant-lab variance partition of three feature spaces (A, V, J).

    Input: held-out R2 of the seven banded-ridge models fitted on every nonempty
    subset of {A, V, J}, keyed "a","v","j","av","aj","vj","avj".  Each model's R2
    is treated as the size of the union of the variance sets its bands explain
    (Lescroart 2015; Deniz 2019), and the seven Venn regions follow from
    inclusion-exclusion.  Negative values are retained; regions sum to R2(avj).
    """
    missing = set(ALL_SUBSETS) - set(r2)
    if missing:
        raise ValueError(f"partition_variance needs all seven subsets; missing {sorted(missing)}")
    a, v, j, av, aj, vj, avj = (np.asarray(r2[key], dtype=np.float64) for key in ALL_SUBSETS)
    pair_a_v, pair_a_j, pair_v_j = a + v - av, a + j - aj, v + j - vj
    triple = avj - a - v - j + pair_a_v + pair_a_j + pair_v_j
    return {
        "unique_a": avj - vj,
        "unique_v": avj - aj,
        "unique_j": avj - av,
        "shared_av_only": pair_a_v - triple,
        "shared_aj_only": pair_a_j - triple,
        "shared_vj_only": pair_v_j - triple,
        "shared_avj": triple,
    }


@dataclass
class FoldResult:
    test_indices: np.ndarray
    y_test: np.ndarray
    predictions: dict[str, np.ndarray]
    r2: dict[str, np.ndarray]
    provenance: dict
    arrays: dict[str, np.ndarray]


def evaluate_split(
    audio: np.ndarray | None,
    video: np.ndarray | None,
    joint: np.ndarray | None,
    targets: np.ndarray,
    run_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alphas: np.ndarray,
    label: str = "fold",
    subsets: tuple[str, ...] = ("av", "avj"),
    n_iter: int = 20,
    backend: str = "torch_cuda",
    model_random_state: int = 0,
) -> FoldResult:
    bad = set(subsets) - set(ALL_SUBSETS)
    if bad:
        raise ValueError(f"Unknown band subsets {sorted(bad)}; use {ALL_SUBSETS}")
    needed = set("".join(subsets))
    raw = {band: value for band, value in zip(BANDS, (audio, video, joint)) if band in needed}
    if len(raw) != len(needed):
        raise ValueError(f"subsets {subsets} need bands {sorted(needed)}; got {sorted(raw)}")
    arrays = [np.asarray(value) for value in (*raw.values(), targets)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("audio, video, joint, and targets must be 2D arrays")
    train_mask, test_mask = np.asarray(train_mask, bool), np.asarray(test_mask, bool)
    if len({value.shape[0] for value in arrays} | {len(run_ids), len(train_mask), len(test_mask)}) != 1:
        raise ValueError("all inputs must have the same sample count")
    if (train_mask & test_mask).any() or not test_mask.any():
        raise ValueError("train and test rows must be disjoint and test nonempty")
    run_ids = np.asarray(run_ids)
    train_runs = run_ids[train_mask]
    if np.unique(train_runs).size < 2:
        raise ValueError("Inner run-wise CV needs at least two training runs")

    train_x = {band: np.asarray(value[train_mask], dtype=np.float32) for band, value in raw.items()}
    test_x = {band: np.asarray(value[test_mask], dtype=np.float32) for band, value in raw.items()}
    y_train = np.asarray(targets[train_mask], dtype=np.float32)
    y_test = np.asarray(targets[test_mask], dtype=np.float32)

    predictions, r2, fitted_arrays = {}, {}, {}
    selected_backend = backend
    for subset in subsets:
        prediction, model, selected_backend, _ = fit_group_ridge(
            [train_x[band] for band in subset], y_train,
            [test_x[band] for band in subset], train_runs, alphas,
            n_iter=n_iter, backend=backend, random_state=model_random_state,
        )
        predictions[subset] = prediction
        r2[subset] = _r2_per_target(y_test, prediction).astype(np.float32)
        fitted_arrays[f"{subset}_best_alphas"] = _to_numpy(model.best_alphas_)
        fitted_arrays[f"{subset}_deltas"] = _to_numpy(model.deltas_)

    return FoldResult(
        test_indices=np.flatnonzero(test_mask),
        y_test=y_test,
        predictions=predictions,
        r2=r2,
        provenance={
            "fold": str(label),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_runs": [str(run) for run in np.unique(train_runs)],
            "subsets": list(subsets),
            "backend": selected_backend,
            "n_iter": int(n_iter),
            "alphas": np.asarray(alphas).tolist(),
            "model_random_state": int(model_random_state),
        },
        arrays=fitted_arrays,
    )
