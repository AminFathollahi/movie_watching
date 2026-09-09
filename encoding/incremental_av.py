"""Run leakage-safe A+V versus A+V+J encoding across held-out movie runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
sys.path.insert(0, str(ROOT))

from encoding.shared.encoding_utils import apply_hrf_to_segment, build_fmri_arrays, spm_hrf
from encoding.shared.fold_evaluator import (
    _r2_per_target,
    clip_error_metrics,
    clip_metrics,
    evaluate_compression_efficiency_fold,
    evaluate_outer_fold,
    paired_clip_inference,
)
from rsa.glasser import load_glasser_parcels


log = logging.getLogger(__name__)
REPEATED_VALIDATION_CLIPS = ("video5", "video9", "video14", "video18")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--fmri-suffix", default="raw")
    parser.add_argument("--subject", default="group_average")
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-name")
    parser.add_argument(
        "--joint-model-template",
        help="Joint-feature model name with a {run} placeholder for outer-fold-specific inputs.",
    )
    parser.add_argument(
        "--unimodal-model",
        help="Model supplying A and V; defaults to --model.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-sec", type=float, default=5.0)
    parser.add_argument("--skip-sec", type=float, default=5.0)
    parser.add_argument("--delay-sec", type=float, default=5.0)
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--hrf", action="store_true")
    parser.add_argument(
        "--exclude-video-ids",
        default=",".join(REPEATED_VALIDATION_CLIPS),
        help="Clips excluded from all fitting and main evaluation.",
    )
    parser.add_argument(
        "--roi-mask", action="append", default=[], metavar="NAME=PATH",
        help="Named nonzero CIFTI mask; may be repeated.",
    )
    parser.add_argument("--glasser-dlabel")
    parser.add_argument(
        "--parcel-roi", action="append", default=[], metavar="NAME=PARCELS",
        help="Named comma-separated bilateral Glasser parcel set; may be repeated.",
    )
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument(
        "--methods", nargs="+",
        choices=("full", "pca", "random_projection", "cluster_mean", "cluster_pc1"),
        default=("full", "pca", "random_projection", "cluster_mean", "cluster_pc1"),
    )
    parser.add_argument("--dimensions", nargs="+", type=int, default=(2, 4, 8, 16, 32, 64))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=9.0)
    parser.add_argument("--n-alphas", type=int, default=23)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--model-random-state", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument(
        "--compression-efficiency",
        action="store_true",
        help="Also compare compressed J with equally budgeted compressed A and V.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def sample_metadata(timing: pd.DataFrame, bin_sec: float, skip_sec: float) -> pd.DataFrame:
    rows = []
    for _, clip in timing.iterrows():
        duration = float(clip["duration_sec"])
        n_windows = (
            max(0, int(np.floor((duration - bin_sec) / skip_sec)) + 1)
            if duration >= bin_sec else 0
        )
        for window in range(n_windows):
            rows.append({
                "row_index": len(rows),
                "video_id": str(clip["video_id"]),
                "run_id": clip["run_id"],
                "window_index": window,
                "window_onset_sec": float(clip["onset_sec"]) + window * skip_sec,
            })
    return pd.DataFrame(rows)


def apply_hrf_by_clip(
    embeddings: np.ndarray, metadata: pd.DataFrame, bin_sec: float
) -> np.ndarray:
    output = np.empty_like(embeddings, dtype=np.float64)
    kernel = spm_hrf(bin_sec)
    for _, clip_rows in metadata.groupby("video_id", sort=False):
        indices = clip_rows["row_index"].to_numpy(dtype=int)
        output[indices] = apply_hrf_to_segment(
            np.asarray(embeddings[indices], dtype=np.float64), kernel,
        )
    return output


def _parse_named(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        name, payload = value.split("=", 1)
        if not name or not payload:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        parsed.append((name, payload))
    return parsed


def load_target_rois(args, fmri_path: Path, n_targets: int):
    rois: dict[str, np.ndarray] = {}
    for name, path in _parse_named(args.roi_mask):
        values = nib.load(path).get_fdata(dtype=np.float32).squeeze()
        if values.shape != (n_targets,):
            raise ValueError(f"ROI {name!r} has shape {values.shape}, expected {(n_targets,)}")
        rois[name] = np.flatnonzero(np.isfinite(values) & (values != 0))

    if args.parcel_roi:
        if not args.glasser_dlabel:
            raise ValueError("--parcel-roi requires --glasser-dlabel")
        fmri_axis = nib.load(str(fmri_path)).header.get_axis(1)
        parcels = load_glasser_parcels(args.glasser_dlabel, fmri_axis)
        short_names = {
            parcel.removeprefix("L_").removeprefix("R_").removesuffix("_ROI")
            for parcel in parcels
        }
        for name, payload in _parse_named(args.parcel_roi):
            requested = [item.strip() for item in payload.split(",") if item.strip()]
            missing = sorted(set(requested) - set(short_names))
            if missing:
                raise KeyError(f"Unknown Glasser parcels for {name!r}: {missing}")
            matching = [indices for parcel, indices in parcels.items()
                        if parcel.removeprefix("L_").removeprefix("R_").removesuffix("_ROI")
                        in requested]
            rois[name] = np.unique(np.concatenate(matching))

    if args.all_targets:
        rois["all"] = np.arange(n_targets)
    if not rois:
        raise ValueError("Specify --roi-mask, --parcel-roi, or --all-targets")
    empty = [name for name, indices in rois.items() if indices.size == 0]
    if empty:
        raise ValueError(f"Empty target ROIs: {empty}")

    target_indices = np.unique(np.concatenate(list(rois.values())))
    target_lookup = {int(index): column for column, index in enumerate(target_indices)}
    roi_columns = {
        name: np.array([target_lookup[int(index)] for index in indices], dtype=int)
        for name, indices in rois.items()
    }
    return target_indices, roi_columns


def compression_configs(args):
    configs = []
    if "full" in args.methods:
        configs.append(("full", None, 0))
    for method in args.methods:
        if method == "full":
            continue
        seeds = args.random_seeds if method in {"random_projection", "cluster_mean", "cluster_pc1"} else (0,)
        for dimension in args.dimensions:
            for seed in seeds:
                configs.append((method, dimension, seed))
    return configs


def _config_tag(method: str, dimension: int | None, seed: int) -> str:
    if method == "full":
        return "full"
    suffix = f"_seed{seed}" if method != "pca" else ""
    return f"{method}_d{dimension}{suffix}"


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _bh_qvalues(p_values) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def plot_summary(output_root: Path) -> Path:
    import matplotlib.pyplot as plt

    incremental = pd.read_csv(output_root / "inference.csv")
    efficiency_path = output_root / "efficiency_inference.csv"
    efficiency = pd.read_csv(efficiency_path) if efficiency_path.exists() else None
    rois = list(pd.unique(incremental["roi"]))
    n_columns = 2 if efficiency is not None else 1
    fig, axes = plt.subplots(
        len(rois), n_columns, figsize=(6.0 * n_columns, 3.8 * len(rois)),
        squeeze=False, constrained_layout=True,
    )
    colors = {
        "pca": "#2878B5", "random_projection": "#56B4E9",
        "cluster_mean": "#D95319", "cluster_pc1": "#E6A700",
    }

    def draw(axis, frame, title):
        compressed = frame[frame["method"] != "full"]
        for method, rows in compressed.groupby("method", sort=False):
            rows = rows.sort_values("dimension")
            color = colors.get(method, "0.4")
            axis.scatter(
                rows["dimension"], rows["mean_mse_reduction"],
                color=color, alpha=0.25, s=18,
            )
            means = rows.groupby("dimension", as_index=False)["mean_mse_reduction"].mean()
            axis.plot(
                means["dimension"], means["mean_mse_reduction"], marker="o",
                linewidth=1.8, color=color, label=method.replace("_", " "),
            )
        full = frame[frame["method"] == "full"]
        if len(full):
            axis.axhline(
                full["mean_mse_reduction"].mean(), color="black",
                linestyle="--", linewidth=1.2, label="full joint",
            )
        axis.axhline(0, color="0.65", linewidth=0.8)
        axis.set_xscale("log", base=2)
        axis.set_yscale("symlog", linthresh=0.01)
        axis.set(xlabel="Representation dimension", ylabel="Held-out MSE reduction (symlog)",
                 title=title)
        if len(compressed):
            axis.set_xticks(sorted(pd.unique(compressed["dimension"])))
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    for row_index, roi in enumerate(rois):
        draw(
            axes[row_index, 0], incremental[incremental["roi"] == roi],
            f"{roi}: incremental beyond full A+V",
        )
        if efficiency is not None:
            draw(
                axes[row_index, 1], efficiency[efficiency["roi"] == roi],
                f"{roi}: joint vs matched A/V budget",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=max(1, len(labels)), frameon=False)
    path = output_root / "incremental_av.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def aggregate_seed_inference(
    output_root: Path,
    metric_name: str,
    output_prefix: str,
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    random_state: int = 0,
) -> Path | None:
    tables = []
    for path in output_root.glob(f"*/{metric_name}"):
        frame = pd.read_csv(path)
        required = {"method", "dimension", "seed", "roi", "clip_id", "mse_reduction"}
        if required.issubset(frame):
            tables.append(frame[list(required)])
    if not tables:
        return None
    combined = pd.concat(tables, ignore_index=True)
    clips = combined.groupby(
        ["method", "dimension", "roi", "clip_id"], as_index=False,
    ).agg(
        mean_mse_reduction=("mse_reduction", "mean"),
        n_seeds=("seed", "nunique"),
    )
    rows = []
    for keys, frame in clips.groupby(["method", "dimension", "roi"], sort=False):
        effects = frame["mean_mse_reduction"].to_numpy()
        rng = np.random.default_rng(random_state)
        bootstrap = effects[
            rng.integers(0, len(effects), size=(n_bootstrap, len(effects)))
        ].mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(effects)))
        null = (signs * effects).mean(axis=1)
        observed = float(effects.mean())
        rows.append({
            "method": keys[0], "dimension": keys[1], "roi": keys[2],
            "n_clips": len(effects), "n_seeds": int(frame["n_seeds"].max()),
            "mean_mse_reduction": observed,
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "sign_flip_p_greater": float(
                (1 + np.sum(null >= observed)) / (n_permutations + 1)
            ),
        })
    inference = pd.DataFrame(rows)
    inference["sign_flip_q_bh"] = _bh_qvalues(inference["sign_flip_p_greater"])
    clip_path = output_root / f"{output_prefix}seed_averaged_clip_metrics.csv"
    inference_path = output_root / f"{output_prefix}seed_averaged_inference.csv"
    clips.to_csv(clip_path, index=False)
    inference.to_csv(inference_path, index=False)
    return inference_path


def _savez_atomic(path: Path, **arrays) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as archive:
            if any(key not in archive.files for key in required):
                return None
            return {key: archive[key] for key in required}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        log.warning("Ignoring incomplete cache file %s", path)
        return None


def _compact_fold_outputs(config_root: Path) -> None:
    keys_by_name = {
        "metrics.npz": (
            "test_indices", "baseline_r2", "extended_r2", "delta_r2",
        ),
        "efficiency_metrics.npz": (
            "test_indices", "additive_r2", "joint_r2",
            "joint_minus_additive_r2",
        ),
    }
    for name, keys in keys_by_name.items():
        for path in config_root.glob(f"run*/{name}"):
            arrays = _load_npz(path, keys)
            if arrays is not None:
                _savez_atomic(path, **arrays)


def _records_for_configuration(
    frame: pd.DataFrame | None, configuration: str, expected_rows: int,
) -> list[dict]:
    if frame is None or "configuration" not in frame:
        return []
    selected = frame.loc[frame["configuration"] == configuration]
    if len(selected) != expected_rows:
        return []
    return selected.to_dict("records")


def run(args) -> Path:
    if "avscramble" in args.model.lower() or (
        args.joint_model_template
        and "avscramble" in args.joint_model_template.lower()
    ):
        raise ValueError(
            "Global avscramble embeddings cross run boundaries. Re-extract joint "
            "features with audio permutations confined separately to each outer "
            "training and test partition before run-wise evaluation."
        )
    timing = pd.read_csv(args.timing_csv)
    metadata = sample_metadata(timing, args.bin_sec, args.skip_sec)
    unimodal_model = args.unimodal_model or args.model
    unimodal_root = (
        Path(args.embeddings_dir) / unimodal_model /
        f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s"
    )
    embedding_paths = {
        "a": unimodal_root / f"{unimodal_model}_a.npy",
        "v": unimodal_root / f"{unimodal_model}_v.npy",
    }
    if args.joint_model_template:
        joint_names = {
            run: args.joint_model_template.format(run=run)
            for run in sorted(pd.unique(metadata["run_id"]))
        }
    else:
        joint_names = {"default": args.model}
    joint_paths = {
        run: (
            Path(args.embeddings_dir) / name /
            f"bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s" /
            f"{name}_av.npy"
        )
        for run, name in joint_names.items()
    }
    embeddings = {name: np.load(path) for name, path in embedding_paths.items()}
    joint_embeddings = {run: np.load(path) for run, path in joint_paths.items()}
    expected = len(metadata)
    for modality, values in embeddings.items():
        if values.shape[0] != expected:
            raise ValueError(f"{modality} has {values.shape[0]} rows; timing describes {expected}")
        if args.hrf:
            embeddings[modality] = apply_hrf_by_clip(values, metadata, args.bin_sec)
    for run, values in joint_embeddings.items():
        if values.shape[0] != expected:
            raise ValueError(
                f"av_run{run} has {values.shape[0]} rows; timing describes {expected}"
            )
        if args.hrf:
            joint_embeddings[run] = apply_hrf_by_clip(values, metadata, args.bin_sec)

    excluded = {item.strip() for item in args.exclude_video_ids.split(",") if item.strip()}
    keep = ~metadata["video_id"].isin(excluded).to_numpy()
    fmri_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
    run_trs_path = Path(args.preprocessed_dir) / f"{args.subject}_{args.fmri_suffix}_run_trs.npy"
    targets, excluded_targets, _ = build_fmri_arrays(
        str(fmri_path), str(run_trs_path), timing, sorted(excluded), args.bin_sec, args.tr,
        delay_sec=args.delay_sec, skip_sec=args.skip_sec,
    )
    if targets.shape[0] != int(keep.sum()):
        raise ValueError(
            f"fMRI rows {targets.shape[0]} do not align to {int(keep.sum())} retained rows"
        )
    if excluded and excluded_targets.shape[0] != int((~keep).sum()):
        raise ValueError("Excluded fMRI rows do not align with excluded embedding rows")

    target_indices, roi_columns = load_target_rois(args, fmri_path, targets.shape[1])
    targets = targets[:, target_indices]
    metadata = metadata.loc[keep].sort_values(
        ["run_id", "row_index"], kind="stable"
    ).reset_index(drop=True)
    retained_rows = metadata["row_index"].to_numpy(dtype=int)
    metadata["analysis_index"] = np.arange(len(metadata))
    for modality in embeddings:
        embeddings[modality] = embeddings[modality][retained_rows]
    for run in joint_embeddings:
        joint_embeddings[run] = joint_embeddings[run][retained_rows]
    joint_dimension = next(iter(joint_embeddings.values())).shape[1]
    if any(values.shape[1] != joint_dimension for values in joint_embeddings.values()):
        raise ValueError("Outer-fold joint embeddings have inconsistent feature dimensions")

    output_root = (
        Path(args.output_dir) / args.subject / (args.output_name or args.model) /
        f"runwise_incremental_av_bin{args.bin_sec:g}s_skip{args.skip_sec:g}s"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cache_signature = {
        "analysis_version": 2,
        "subject": args.subject,
        "model": args.model,
        "unimodal_model": unimodal_model,
        "fmri": _file_identity(fmri_path),
        "timing": _file_identity(Path(args.timing_csv)),
        "embeddings": {
            **{name: _file_identity(path) for name, path in embedding_paths.items()},
            **{
                ("av" if run == "default" else f"av_run{run}"): _file_identity(path)
                for run, path in joint_paths.items()
            },
        },
        "bin_sec": args.bin_sec,
        "skip_sec": args.skip_sec,
        "delay_sec": args.delay_sec,
        "tr": args.tr,
        "hrf": args.hrf,
        "excluded_video_ids": sorted(excluded),
        "target_indices_sha256": hashlib.sha256(target_indices.tobytes()).hexdigest(),
        "alphas": np.logspace(args.alpha_min, args.alpha_max, args.n_alphas).tolist(),
        "n_iter": args.n_iter,
        "backend": args.backend,
        "model_random_state": args.model_random_state,
    }
    if args.joint_model_template:
        cache_signature["joint_model_template"] = args.joint_model_template
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        previous_signature = json.loads(manifest_path.read_text()).get("cache_signature")
        if previous_signature != cache_signature:
            raise ValueError(
                f"Existing results in {output_root} use different inputs or fitting "
                "settings; pass --force or choose another output directory"
            )
    metadata.to_csv(output_root / "samples.csv", index=False)
    np.save(output_root / "target_indices.npy", target_indices)
    (output_root / "rois.json").write_text(json.dumps(
        {name: target_indices[columns].tolist() for name, columns in roi_columns.items()}, indent=2
    ) + "\n")

    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    folds = list(pd.unique(metadata["run_id"]))
    configurations = [_config_tag(*config) for config in compression_configs(args)]
    manifest = {
        "status": "running",
        "subject": args.subject,
        "model": args.model,
        "output_name": args.output_name or args.model,
        "unimodal_model": unimodal_model,
        "joint_models": {str(run): name for run, name in joint_names.items()},
        "n_samples": len(metadata),
        "runs": [str(run) for run in folds],
        "excluded_video_ids": sorted(excluded),
        "n_targets": int(len(target_indices)),
        "response_preprocessing": (
            "Binned movie responses were z-scored within run using retained "
            "movie bins only. This response-only normalization uses the full "
            "held-out run; all feature transforms and predictor fits are "
            "training-only."
        ),
        "rois": {name: int(len(columns)) for name, columns in roi_columns.items()},
        "configurations": configurations,
        "cache_signature": cache_signature,
        "multiplicity": "Benjamini-Hochberg within each model output table",
        "arguments": vars(args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    existing_tables = {}
    for name in ("summary", "inference", "efficiency_summary", "efficiency_inference"):
        path = output_root / f"{name}.csv"
        existing_tables[name] = pd.read_csv(path) if path.exists() else None
    summary_rows = []
    inference_rows = []
    efficiency_summary_rows = []
    efficiency_inference_rows = []
    baseline_cache = {}
    for method, dimension, seed in compression_configs(args):
        tag = _config_tag(method, dimension, seed)
        config_root = output_root / tag
        config_root.mkdir(exist_ok=True)
        run_efficiency = args.compression_efficiency and method != "full"
        expected_summary = len(folds) * len(roi_columns)
        recovered = {
            "summary": _records_for_configuration(
                existing_tables["summary"], tag, expected_summary,
            ),
            "inference": _records_for_configuration(
                existing_tables["inference"], tag, len(roi_columns),
            ),
            "efficiency_summary": _records_for_configuration(
                existing_tables["efficiency_summary"], tag,
                expected_summary if run_efficiency else 0,
            ),
            "efficiency_inference": _records_for_configuration(
                existing_tables["efficiency_inference"], tag,
                len(roi_columns) if run_efficiency else 0,
            ),
        }
        complete = (
            (config_root / "folds.json").exists()
            and bool(recovered["summary"])
            and bool(recovered["inference"])
            and (
                not run_efficiency
                or (
                    bool(recovered["efficiency_summary"])
                    and bool(recovered["efficiency_inference"])
                )
            )
        )
        if complete and not args.force:
            summary_rows.extend(recovered["summary"])
            inference_rows.extend(recovered["inference"])
            efficiency_summary_rows.extend(recovered["efficiency_summary"])
            efficiency_inference_rows.extend(recovered["efficiency_inference"])
            _compact_fold_outputs(config_root)
            log.info("Reused complete configuration %s", tag)
            continue

        oof_base = np.full_like(targets, np.nan, dtype=np.float32)
        oof_extended = np.full_like(targets, np.nan, dtype=np.float32)
        if run_efficiency:
            oof_additive_compressed = np.full_like(targets, np.nan, dtype=np.float32)
            oof_joint_compressed = np.full_like(targets, np.nan, dtype=np.float32)
        provenance = []
        for test_run in folds:
            joint = joint_embeddings.get(test_run, joint_embeddings.get("default"))
            if joint is None:
                raise KeyError(f"No joint embeddings configured for held-out run {test_run}")
            fold_root = config_root / f"run{test_run}"
            result_path = fold_root / "metrics.npz"
            saved_result = None if args.force else _load_npz(
                result_path,
                (
                    "test_indices", "baseline_prediction", "extended_prediction",
                    "baseline_r2", "extended_r2", "delta_r2",
                ),
            )
            if saved_result is not None:
                test_indices = saved_result["test_indices"]
                base_prediction = saved_result["baseline_prediction"]
                extended_prediction = saved_result["extended_prediction"]
                base_r2 = saved_result["baseline_r2"]
                extended_r2 = saved_result["extended_r2"]
                delta_r2 = saved_result["delta_r2"]
                fold_provenance = json.loads((fold_root / "provenance.json").read_text())
                parameters_path = fold_root / "fitted_parameters.npz"
                if test_run not in baseline_cache and parameters_path.exists():
                    with np.load(parameters_path) as parameters:
                        baseline_cache[test_run] = {
                            "prediction": base_prediction,
                            "best_alphas": parameters["baseline_best_alphas"],
                            "deltas": parameters["baseline_deltas"],
                            "backend": fold_provenance["backend"],
                        }
            else:
                fold_root.mkdir(parents=True, exist_ok=True)
                log.info("Fitting %s, held-out run %s", tag, test_run)
                result = evaluate_outer_fold(
                    embeddings["a"], embeddings["v"], joint, targets,
                    metadata["run_id"].to_numpy(), test_run, alphas,
                    method=method, dimension=dimension, n_iter=args.n_iter,
                    backend=args.backend, random_state=seed,
                    model_random_state=args.model_random_state,
                    baseline_cache=baseline_cache.get(test_run),
                )
                test_indices = result.test_indices
                base_prediction = result.baseline_prediction
                extended_prediction = result.extended_prediction
                base_r2 = result.baseline_r2
                extended_r2 = result.extended_r2
                delta_r2 = result.delta_r2
                baseline_cache.setdefault(test_run, {
                    "prediction": result.baseline_prediction,
                    "best_alphas": result.arrays["baseline_best_alphas"],
                    "deltas": result.arrays["baseline_deltas"],
                    "backend": result.provenance["backend"],
                })
                _savez_atomic(
                    result_path,
                    test_indices=test_indices,
                    y_true=targets[test_indices],
                    baseline_prediction=base_prediction,
                    extended_prediction=extended_prediction,
                    baseline_error=targets[test_indices] - base_prediction,
                    extended_error=targets[test_indices] - extended_prediction,
                    baseline_r2=base_r2,
                    extended_r2=extended_r2,
                    delta_r2=delta_r2,
                )
                transform_arrays = {f"compression_{key}": value for key, value in result.compression.arrays.items()}
                _savez_atomic(
                    fold_root / "fitted_parameters.npz",
                    **result.arrays,
                    **transform_arrays,
                )
                fold_provenance = result.provenance
                (fold_root / "provenance.json").write_text(json.dumps(fold_provenance, indent=2) + "\n")

            oof_base[test_indices] = base_prediction
            oof_extended[test_indices] = extended_prediction
            provenance.append(fold_provenance)
            for roi, columns in roi_columns.items():
                summary_rows.append({
                    "configuration": tag,
                    "method": method,
                    "dimension": joint_dimension if dimension is None else dimension,
                    "seed": seed,
                    "test_run": test_run,
                    "roi": roi,
                    "n_targets": len(columns),
                    "baseline_r2_mean": float(np.nanmean(base_r2[columns])),
                    "extended_r2_mean": float(np.nanmean(extended_r2[columns])),
                    "delta_r2_mean": float(np.nanmean(delta_r2[columns])),
                    "fraction_delta_positive": float(np.nanmean(delta_r2[columns] > 0)),
                })

            if run_efficiency:
                efficiency_path = fold_root / "efficiency_metrics.npz"
                efficiency_provenance_path = fold_root / "efficiency_provenance.json"
                saved_efficiency = None if args.force else _load_npz(
                    efficiency_path,
                    (
                        "test_indices", "additive_prediction", "joint_prediction",
                        "additive_r2", "joint_r2", "joint_minus_additive_r2",
                    ),
                )
                if saved_efficiency is not None:
                    efficiency_indices = saved_efficiency["test_indices"]
                    additive_compressed = saved_efficiency["additive_prediction"]
                    joint_compressed = saved_efficiency["joint_prediction"]
                    additive_compressed_r2 = saved_efficiency["additive_r2"]
                    joint_compressed_r2 = saved_efficiency["joint_r2"]
                    efficiency_delta = saved_efficiency["joint_minus_additive_r2"]
                else:
                    log.info("Fitting efficiency %s, held-out run %s", tag, test_run)
                    efficiency = evaluate_compression_efficiency_fold(
                        embeddings["a"], embeddings["v"], joint, targets,
                        metadata["run_id"].to_numpy(), test_run, alphas,
                        method=method, total_dimension=dimension, n_iter=args.n_iter,
                        backend=args.backend, random_state=seed,
                        model_random_state=args.model_random_state,
                    )
                    efficiency_indices = efficiency.test_indices
                    additive_compressed = efficiency.additive_prediction
                    joint_compressed = efficiency.joint_prediction
                    additive_compressed_r2 = efficiency.additive_r2
                    joint_compressed_r2 = efficiency.joint_r2
                    efficiency_delta = efficiency.joint_minus_additive_r2
                    _savez_atomic(
                        efficiency_path,
                        test_indices=efficiency_indices,
                        y_true=targets[efficiency_indices],
                        additive_prediction=additive_compressed,
                        joint_prediction=joint_compressed,
                        additive_error=targets[efficiency_indices] - additive_compressed,
                        joint_error=targets[efficiency_indices] - joint_compressed,
                        additive_r2=additive_compressed_r2,
                        joint_r2=joint_compressed_r2,
                        joint_minus_additive_r2=efficiency_delta,
                    )
                    _savez_atomic(
                        fold_root / "efficiency_fitted_parameters.npz",
                        **efficiency.arrays,
                    )
                    efficiency_provenance_path.write_text(
                        json.dumps(efficiency.provenance, indent=2) + "\n"
                    )
                if not np.array_equal(efficiency_indices, test_indices):
                    raise RuntimeError("Efficiency and incremental fold indices differ")
                oof_additive_compressed[efficiency_indices] = additive_compressed
                oof_joint_compressed[efficiency_indices] = joint_compressed
                for roi, columns in roi_columns.items():
                    efficiency_summary_rows.append({
                        "configuration": tag,
                        "method": method,
                        "dimension": dimension,
                        "seed": seed,
                        "test_run": test_run,
                        "roi": roi,
                        "n_targets": len(columns),
                        "additive_compressed_r2_mean": float(
                            np.nanmean(additive_compressed_r2[columns])
                        ),
                        "joint_compressed_r2_mean": float(
                            np.nanmean(joint_compressed_r2[columns])
                        ),
                        "joint_minus_additive_r2_mean": float(
                            np.nanmean(efficiency_delta[columns])
                        ),
                    })

        if np.isnan(oof_base).any() or np.isnan(oof_extended).any():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {tag}")
        clip_rows = clip_metrics(
            targets, oof_base, oof_extended, metadata["video_id"].to_numpy()
        )
        for row in clip_rows:
            row["configuration"] = tag
            row["method"] = method
            row["dimension"] = joint_dimension if dimension is None else dimension
            row["seed"] = seed
        pd.DataFrame(clip_rows).to_csv(config_root / "clip_metrics.csv", index=False)
        roi_clip_rows = []
        for roi, columns in roi_columns.items():
            rows = clip_error_metrics(
                targets[:, columns], oof_base[:, columns], oof_extended[:, columns],
                metadata["video_id"].to_numpy(),
            )
            for row in rows:
                row["roi"] = roi
                row["configuration"] = tag
                row["method"] = method
                row["dimension"] = joint_dimension if dimension is None else dimension
                row["seed"] = seed
            roi_clip_rows.extend(rows)
        pd.DataFrame(roi_clip_rows).to_csv(
            config_root / "roi_clip_metrics.csv", index=False
        )
        for roi, columns in roi_columns.items():
            inference = paired_clip_inference(
                targets[:, columns], oof_base[:, columns], oof_extended[:, columns],
                metadata["video_id"].to_numpy(), n_bootstrap=args.n_bootstrap,
                n_permutations=args.n_permutations,
                random_state=args.model_random_state,
            )
            pooled_base = _r2_per_target(targets[:, columns], oof_base[:, columns])
            pooled_extended = _r2_per_target(
                targets[:, columns], oof_extended[:, columns]
            )
            inference.update({
                "configuration": tag,
                "method": method,
                "dimension": joint_dimension if dimension is None else dimension,
                "seed": seed,
                "roi": roi,
                "pooled_baseline_r2_mean": float(np.nanmean(pooled_base)),
                "pooled_extended_r2_mean": float(np.nanmean(pooled_extended)),
                "pooled_delta_r2_mean": float(np.nanmean(pooled_extended - pooled_base)),
            })
            inference_rows.append(inference)
        if run_efficiency:
            if np.isnan(oof_additive_compressed).any() or np.isnan(oof_joint_compressed).any():
                raise RuntimeError(f"Incomplete efficiency predictions for {tag}")
            efficiency_clips = clip_metrics(
                targets, oof_additive_compressed, oof_joint_compressed,
                metadata["video_id"].to_numpy(),
            )
            pd.DataFrame(efficiency_clips).to_csv(
                config_root / "efficiency_clip_metrics.csv", index=False
            )
            efficiency_roi_clip_rows = []
            for roi, columns in roi_columns.items():
                rows = clip_error_metrics(
                    targets[:, columns], oof_additive_compressed[:, columns],
                    oof_joint_compressed[:, columns],
                    metadata["video_id"].to_numpy(),
                )
                for row in rows:
                    row["roi"] = roi
                    row["configuration"] = tag
                    row["method"] = method
                    row["dimension"] = dimension
                    row["seed"] = seed
                efficiency_roi_clip_rows.extend(rows)
            pd.DataFrame(efficiency_roi_clip_rows).to_csv(
                config_root / "efficiency_roi_clip_metrics.csv", index=False
            )
            for roi, columns in roi_columns.items():
                inference = paired_clip_inference(
                    targets[:, columns], oof_additive_compressed[:, columns],
                    oof_joint_compressed[:, columns],
                    metadata["video_id"].to_numpy(), n_bootstrap=args.n_bootstrap,
                    n_permutations=args.n_permutations,
                    random_state=args.model_random_state,
                )
                inference.update({
                    "configuration": tag,
                    "method": method,
                    "dimension": dimension,
                    "seed": seed,
                    "roi": roi,
                })
                efficiency_inference_rows.append(inference)
        (config_root / "folds.json").write_text(json.dumps(provenance, indent=2) + "\n")
        pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)
        pd.DataFrame(inference_rows).to_csv(output_root / "inference.csv", index=False)
        if efficiency_summary_rows:
            pd.DataFrame(efficiency_summary_rows).to_csv(
                output_root / "efficiency_summary.csv", index=False
            )
            pd.DataFrame(efficiency_inference_rows).to_csv(
                output_root / "efficiency_inference.csv", index=False
            )
        _compact_fold_outputs(config_root)

    inference_frame = pd.DataFrame(inference_rows)
    inference_frame["sign_flip_q_bh"] = _bh_qvalues(
        inference_frame["sign_flip_p_greater"]
    )
    inference_frame.to_csv(output_root / "inference.csv", index=False)
    if efficiency_inference_rows:
        efficiency_inference_frame = pd.DataFrame(efficiency_inference_rows)
        efficiency_inference_frame["sign_flip_q_bh"] = _bh_qvalues(
            efficiency_inference_frame["sign_flip_p_greater"]
        )
        efficiency_inference_frame.to_csv(
            output_root / "efficiency_inference.csv", index=False
        )

    aggregate_seed_inference(
        output_root, "roi_clip_metrics.csv", "",
        args.n_bootstrap, args.n_permutations, args.model_random_state,
    )
    aggregate_seed_inference(
        output_root, "efficiency_roi_clip_metrics.csv", "efficiency_",
        args.n_bootstrap, args.n_permutations, args.model_random_state,
    )

    plot_summary(output_root)

    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Saved run-wise incremental AV results to %s", output_root)
    return output_root


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
