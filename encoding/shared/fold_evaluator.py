from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from encoding.shared.compression import FittedCompression, fit_compression


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


def standardize_bands(
    train_bands: list[np.ndarray], test_bands: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray]]]:
    if len(train_bands) != len(test_bands) or not train_bands:
        raise ValueError("train_bands and test_bands must be nonempty and matched")
    train_out, test_out, stats = [], [], []
    for train, test in zip(train_bands, test_bands):
        train = np.asarray(train, dtype=np.float64)
        test = np.asarray(test, dtype=np.float64)
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
            raise ValueError("each train/test band must be 2D with matching features")
        mean = train.mean(axis=0, keepdims=True)
        scale = train.std(axis=0, keepdims=True)
        scale[scale < 1e-12] = 1.0
        train_out.append(((train - mean) / scale).astype(np.float32))
        test_out.append(((test - mean) / scale).astype(np.float32))
        stats.append({"mean": mean, "scale": scale})
    return train_out, test_out, stats


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


@dataclass
class FoldResult:
    test_indices: np.ndarray
    baseline_prediction: np.ndarray
    extended_prediction: np.ndarray
    baseline_r2: np.ndarray
    extended_r2: np.ndarray
    delta_r2: np.ndarray
    compression: FittedCompression
    provenance: dict
    arrays: dict[str, np.ndarray]


@dataclass
class EfficiencyFoldResult:
    test_indices: np.ndarray
    additive_prediction: np.ndarray
    joint_prediction: np.ndarray
    additive_r2: np.ndarray
    joint_r2: np.ndarray
    joint_minus_additive_r2: np.ndarray
    compressions: dict[str, FittedCompression]
    provenance: dict
    arrays: dict[str, np.ndarray]


def evaluate_outer_fold(
    audio: np.ndarray,
    video: np.ndarray,
    joint: np.ndarray,
    targets: np.ndarray,
    run_ids: np.ndarray,
    test_run,
    alphas: np.ndarray,
    method: str = "full",
    dimension: int | None = None,
    n_iter: int = 20,
    backend: str = "torch_cuda",
    random_state: int = 0,
    model_random_state: int = 0,
    baseline_cache: dict | None = None,
) -> FoldResult:
    """Evaluate A+V and A+V+J on one unseen run."""
    arrays = [np.asarray(value) for value in (audio, video, joint, targets)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("audio, video, joint, and targets must be 2D arrays")
    if len({value.shape[0] for value in arrays} | {len(run_ids)}) != 1:
        raise ValueError("all inputs must have the same sample count")
    run_ids = np.asarray(run_ids)
    test_mask = run_ids == test_run
    train_mask = ~test_mask
    if not test_mask.any() or np.unique(run_ids[train_mask]).size < 2:
        raise ValueError("Each outer fold needs a test run and at least two training runs")

    base_train, base_test, base_stats = standardize_bands(
        [audio[train_mask], video[train_mask]],
        [audio[test_mask], video[test_mask]],
    )
    compression = fit_compression(
        joint[train_mask], joint[test_mask], method, dimension, random_state
    )
    y_train = np.asarray(targets[train_mask], dtype=np.float32)
    y_test = np.asarray(targets[test_mask], dtype=np.float32)
    train_runs = run_ids[train_mask]

    if baseline_cache is None:
        baseline_prediction, baseline_model, selected_backend, _ = fit_group_ridge(
            base_train, y_train, base_test, train_runs, alphas,
            n_iter=n_iter, backend=backend, random_state=model_random_state,
        )
        baseline_cache = {
            "prediction": baseline_prediction,
            "best_alphas": _to_numpy(baseline_model.best_alphas_),
            "deltas": _to_numpy(baseline_model.deltas_),
            "backend": selected_backend,
        }
    else:
        baseline_prediction = baseline_cache["prediction"]

    extended_prediction, extended_model, selected_backend, _ = fit_group_ridge(
        [*base_train, compression.train],
        y_train,
        [*base_test, compression.test],
        train_runs,
        alphas,
        n_iter=n_iter,
        backend=backend,
        random_state=model_random_state,
    )
    baseline_r2 = _r2_per_target(y_test, baseline_prediction)
    extended_r2 = _r2_per_target(y_test, extended_prediction)
    fitted_arrays = {
        "baseline_best_alphas": np.asarray(baseline_cache["best_alphas"]),
        "baseline_deltas": np.asarray(baseline_cache["deltas"]),
        "extended_best_alphas": _to_numpy(extended_model.best_alphas_),
        "extended_deltas": _to_numpy(extended_model.deltas_),
    }
    for band, stat in zip(("audio", "video"), base_stats):
        fitted_arrays[f"{band}_mean"] = stat["mean"]
        fitted_arrays[f"{band}_scale"] = stat["scale"]

    return FoldResult(
        test_indices=np.flatnonzero(test_mask),
        baseline_prediction=baseline_prediction,
        extended_prediction=extended_prediction,
        baseline_r2=baseline_r2.astype(np.float32),
        extended_r2=extended_r2.astype(np.float32),
        delta_r2=(extended_r2 - baseline_r2).astype(np.float32),
        compression=compression,
        provenance={
            "test_run": str(test_run),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_runs": [str(run) for run in np.unique(train_runs)],
            "backend": selected_backend,
            "n_iter": int(n_iter),
            "alphas": np.asarray(alphas).tolist(),
            "compression": compression.provenance,
            "model_random_state": int(model_random_state),
        },
        arrays=fitted_arrays,
    )


def evaluate_compression_efficiency_fold(
    audio: np.ndarray,
    video: np.ndarray,
    joint: np.ndarray,
    targets: np.ndarray,
    run_ids: np.ndarray,
    test_run,
    alphas: np.ndarray,
    method: str,
    total_dimension: int,
    n_iter: int = 20,
    backend: str = "torch_cuda",
    random_state: int = 0,
    model_random_state: int = 0,
) -> EfficiencyFoldResult:
    """Compare compressed J with equally budgeted compressed A and V."""
    if method == "full" or total_dimension < 2:
        raise ValueError("efficiency comparison needs a compressed method and dimension >= 2")
    arrays = [np.asarray(value) for value in (audio, video, joint, targets)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("audio, video, joint, and targets must be 2D arrays")
    if len({value.shape[0] for value in arrays} | {len(run_ids)}) != 1:
        raise ValueError("all inputs must have the same sample count")

    run_ids = np.asarray(run_ids)
    test_mask = run_ids == test_run
    train_mask = ~test_mask
    if not test_mask.any() or np.unique(run_ids[train_mask]).size < 2:
        raise ValueError("Each outer fold needs a test run and at least two training runs")
    audio_dimension = total_dimension // 2
    video_dimension = total_dimension - audio_dimension
    compressions = {
        "audio": fit_compression(
            audio[train_mask], audio[test_mask], method, audio_dimension, random_state
        ),
        "video": fit_compression(
            video[train_mask], video[test_mask], method, video_dimension, random_state
        ),
        "joint": fit_compression(
            joint[train_mask], joint[test_mask], method, total_dimension, random_state
        ),
    }
    y_train = np.asarray(targets[train_mask], dtype=np.float32)
    y_test = np.asarray(targets[test_mask], dtype=np.float32)
    train_runs = run_ids[train_mask]
    additive_prediction, additive_model, selected_backend, _ = fit_group_ridge(
        [compressions["audio"].train, compressions["video"].train],
        y_train,
        [compressions["audio"].test, compressions["video"].test],
        train_runs,
        alphas,
        n_iter=n_iter,
        backend=backend,
        random_state=model_random_state,
    )
    joint_prediction, joint_model, _, _ = fit_group_ridge(
        [compressions["joint"].train],
        y_train,
        [compressions["joint"].test],
        train_runs,
        alphas,
        n_iter=n_iter,
        backend=backend,
        random_state=model_random_state,
    )
    additive_r2 = _r2_per_target(y_test, additive_prediction)
    joint_r2 = _r2_per_target(y_test, joint_prediction)
    fitted_arrays = {
        "additive_best_alphas": _to_numpy(additive_model.best_alphas_),
        "additive_deltas": _to_numpy(additive_model.deltas_),
        "joint_best_alphas": _to_numpy(joint_model.best_alphas_),
        "joint_deltas": _to_numpy(joint_model.deltas_),
    }
    for name, compression in compressions.items():
        for key, value in compression.arrays.items():
            fitted_arrays[f"{name}_{key}"] = value
    return EfficiencyFoldResult(
        test_indices=np.flatnonzero(test_mask),
        additive_prediction=additive_prediction,
        joint_prediction=joint_prediction,
        additive_r2=additive_r2.astype(np.float32),
        joint_r2=joint_r2.astype(np.float32),
        joint_minus_additive_r2=(joint_r2 - additive_r2).astype(np.float32),
        compressions=compressions,
        provenance={
            "test_run": str(test_run),
            "train_runs": [str(run) for run in np.unique(train_runs)],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "backend": selected_backend,
            "method": method,
            "total_dimension": int(total_dimension),
            "audio_dimension": int(audio_dimension),
            "video_dimension": int(video_dimension),
            "random_state": int(random_state),
            "model_random_state": int(model_random_state),
            "alphas": np.asarray(alphas).tolist(),
            "n_iter": int(n_iter),
        },
        arrays=fitted_arrays,
    )


def clip_metrics(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    extended_prediction: np.ndarray,
    clip_ids: np.ndarray,
) -> list[dict]:
    rows = []
    for clip in np.unique(clip_ids):
        mask = clip_ids == clip
        base_r2 = _r2_per_target(y_true[mask], baseline_prediction[mask])
        ext_r2 = _r2_per_target(y_true[mask], extended_prediction[mask])
        rows.append({
            "clip_id": str(clip),
            "n_samples": int(mask.sum()),
            "baseline_r2_mean": float(np.nanmean(base_r2)),
            "extended_r2_mean": float(np.nanmean(ext_r2)),
            "delta_r2_mean": float(np.nanmean(ext_r2 - base_r2)),
            "baseline_sse": float(np.square(y_true[mask] - baseline_prediction[mask]).sum()),
            "extended_sse": float(np.square(y_true[mask] - extended_prediction[mask]).sum()),
        })
    return rows


def clip_error_metrics(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    extended_prediction: np.ndarray,
    clip_ids: np.ndarray,
) -> list[dict]:
    rows = []
    for clip in np.unique(clip_ids):
        mask = clip_ids == clip
        baseline_mse = float(np.square(
            y_true[mask] - baseline_prediction[mask]
        ).mean())
        extended_mse = float(np.square(
            y_true[mask] - extended_prediction[mask]
        ).mean())
        rows.append({
            "clip_id": str(clip),
            "n_samples": int(mask.sum()),
            "baseline_mse": baseline_mse,
            "extended_mse": extended_mse,
            "mse_reduction": baseline_mse - extended_mse,
        })
    return rows


def paired_clip_inference(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    extended_prediction: np.ndarray,
    clip_ids: np.ndarray,
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    random_state: int = 0,
) -> dict:
    """Estimate uncertainty for paired held-out error reduction by clip."""
    if n_bootstrap < 1 or n_permutations < 1:
        raise ValueError("n_bootstrap and n_permutations must be positive")
    clip_ids = np.asarray(clip_ids)
    squared_error_gain = (
        np.square(y_true - baseline_prediction)
        - np.square(y_true - extended_prediction)
    )
    effects = np.array([
        squared_error_gain[clip_ids == clip].mean()
        for clip in np.unique(clip_ids)
    ])
    if len(effects) < 2:
        raise ValueError("At least two held-out clips are required for inference")
    rng = np.random.default_rng(random_state)
    bootstrap = effects[rng.integers(0, len(effects), (n_bootstrap, len(effects)))].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(effects)))
    null = (signs * effects).mean(axis=1)
    observed = float(effects.mean())
    return {
        "n_clips": int(len(effects)),
        "mean_mse_reduction": observed,
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p_greater": float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
    }
