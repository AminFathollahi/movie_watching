#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_modeling.cf_naming import (ROI_MEAN_PREPROCESSING, ROI_MEAN_PREPROCESSING_OPTIONS, roi_mean_connectivity_names,
                                   roi_mean_preprocessing_label)
from shared.paths import DATA

log = logging.getLogger("roi_mean_partial_connectivity")
SURFACE_DIR = DATA / "vicsompy_data" / "hcp_999999_draw_NH" / "surfaces"
SURFACES = {"L": "bm_lh.gii", "R": "bm_rh.gii"}


def _runwise_standardize(values: np.ndarray, run_trs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if int(np.sum(run_trs)) != values.shape[0]:
        raise ValueError(f"run_trs sum to {int(np.sum(run_trs))}, but data have {values.shape[0]} TRs")
    result = np.empty_like(values)
    start = 0
    for length in np.asarray(run_trs, dtype=int):
        stop = start + int(length)
        chunk = values[start:stop]
        mean = np.nanmean(chunk, axis=0, keepdims=True)
        std = np.nanstd(chunk, axis=0, keepdims=True)
        safe = np.isfinite(std) & (std > np.finfo(np.float64).eps)
        result[start:stop] = np.where(safe, (chunk - mean) / std, 0.0)
        start = stop
    return np.nan_to_num(result, copy=False)


def _residual(target: np.ndarray, control: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    mean_target, mean_control = target[train].mean(0), control[train].mean()
    centered = control[train] - mean_control
    slope = (centered @ (target[train] - mean_target)) / (centered @ centered)
    return target[test] - mean_target - np.outer(control[test] - mean_control, slope)


def loro_partial_correlations(target_z: np.ndarray, first_z: np.ndarray, second_z: np.ndarray, run_trs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_z = np.asarray(target_z, dtype=np.float64)
    if target_z.ndim == 1:
        target_z = target_z[:, None]
    run = np.repeat(np.arange(len(run_trs)), np.asarray(run_trs, dtype=int))
    if target_z.shape[0] != run.size or first_z.size != run.size or second_z.size != run.size:
        raise ValueError("Targets and both ROI series must have run_trs.sum() TRs")
    series = {"first": first_z[:, None], "second": second_z[:, None]}
    totals = {"first": np.zeros(target_z.shape[1]), "second": np.zeros(target_z.shape[1])}
    for held_out in range(len(run_trs)):
        test, train = run == held_out, run != held_out
        for role, control in (("first", "second"), ("second", "first")):
            y = _residual(target_z, series[control][:, 0], train, test)
            x = _residual(series[role], series[control][:, 0], train, test)
            y, x = y - y.mean(0), x - x.mean(0)
            denominator = np.sqrt((y ** 2).sum(0) * (x ** 2).sum(0))
            totals[role] += test.sum() * np.where(denominator > 0, (y * x).sum(0) / np.where(denominator > 0, denominator, 1), 0.0)
    n = float(run.size)
    return ((totals["first"] / n).astype(np.float32), (totals["second"] / n).astype(np.float32))


def geodesic_exclusion(vertices: np.ndarray, coords: np.ndarray, faces: np.ndarray, source: np.ndarray, limit_mm: float) -> np.ndarray:
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1), axis=0)
    length = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    n = len(coords)
    graph = coo_matrix((np.r_[length, length], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    distance = dijkstra(graph, directed=False, indices=np.unique(source), limit=limit_mm, min_only=True)
    return distance[vertices] <= limit_mm


def _cortex_hemisphere_indices(brain_axis) -> dict[str, np.ndarray]:
    indices: dict[str, np.ndarray] = {}
    for structure, slc, _ in brain_axis.iter_structures():
        if "CORTEX_LEFT" in structure:
            indices["L"] = np.arange(len(brain_axis))[slc]
        elif "CORTEX_RIGHT" in structure:
            indices["R"] = np.arange(len(brain_axis))[slc]
    if set(indices) != {"L", "R"}:
        raise ValueError("Template must contain left and right cortical structures")
    return indices


def _load_mask(path: Path, expected: int) -> np.ndarray:
    mask = np.asarray(nib.load(path).dataobj)[0].astype(bool)
    if mask.size != expected:
        raise ValueError(f"{path} has {mask.size} grayordinates; expected {expected}")
    return mask


def _stream_mask_means(dataobj, masks: dict[str, np.ndarray], n_trs: int, n_gray: int, batch_size: int) -> dict[str, np.ndarray]:
    sums = {name: np.zeros(n_trs, dtype=np.float64) for name in masks}
    counts = {name: 0 for name in masks}
    for start in range(0, n_gray, batch_size):
        stop = min(start + batch_size, n_gray)
        active = {name: mask[start:stop] for name, mask in masks.items() if np.any(mask[start:stop])}
        if not active:
            continue
        block = np.asarray(dataobj[:, start:stop], dtype=np.float64)
        for name, local_mask in active.items():
            sums[name] += block[:, local_mask].sum(axis=1)
            counts[name] += int(local_mask.sum())
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"Empty ROI masks: {empty}")
    return {name: sums[name] / counts[name] for name in masks}


def _save_dscalar(data: np.ndarray, names: list[str], template, path: Path) -> None:
    header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(names), template.header.get_axis(1)))
    nib.save(nib.Cifti2Image(np.asarray(data, dtype=np.float32), header=header, nifti_header=template.nifti_header), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dtseries", type=Path, required=True)
    parser.add_argument("--run-trs", type=Path, required=True)
    parser.add_argument("--roi-a", required=True)
    parser.add_argument("--roi-b", required=True)
    parser.add_argument("--mask-a", type=Path, required=True)
    parser.add_argument("--mask-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocessing", choices=sorted(ROI_MEAN_PREPROCESSING_OPTIONS), default="raw")
    parser.add_argument("--supporting-dir", type=Path, default=None, help="Directory for the JSON provenance. Default: parent of --output-dir")
    parser.add_argument("--surface-dir", type=Path, default=SURFACE_DIR)
    parser.add_argument("--exclusion-mm", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = roi_mean_preprocessing_label(args.preprocessing)
    for path in (args.dtseries, args.run_trs, args.mask_a, args.mask_b):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supporting_dir = args.supporting_dir or args.output_dir.parent
    supporting_dir.mkdir(parents=True, exist_ok=True)

    image = nib.load(args.dtseries)
    if len(image.shape) != 2:
        raise ValueError(f"Expected a 2D dtseries, got {image.shape}")
    n_trs, n_gray = image.shape
    brain_axis = image.header.get_axis(1)
    hemispheres = _cortex_hemisphere_indices(brain_axis)
    run_trs = np.asarray(np.load(args.run_trs), dtype=int)
    if run_trs.sum() != n_trs:
        raise ValueError(f"run_trs sum {run_trs.sum()} != dtseries TRs {n_trs}")
    mask_a, mask_b = _load_mask(args.mask_a, n_gray), _load_mask(args.mask_b, n_gray)
    if np.any(mask_a & mask_b):
        raise ValueError("The two ROI masks overlap")

    means = _stream_mask_means(image.dataobj, {"a": mask_a, "b": mask_b}, n_trs, n_gray, args.batch_size)
    series = {key: _runwise_standardize(value, run_trs)[:, 0] for key, value in means.items()}

    excluded = np.zeros(n_gray, dtype=bool)
    for hemisphere, indices in hemispheres.items():
        surface = nib.load(args.surface_dir / SURFACES[hemisphere])
        coords, faces = (np.asarray(d.data) for d in surface.darrays[:2])
        vertices = brain_axis.vertex[indices]
        source = vertices[(mask_a | mask_b)[indices]]
        excluded[indices] = geodesic_exclusion(vertices, coords.astype(np.float64), faces.astype(np.int64), source, args.exclusion_mm)
    log.info("Excluded %d grayordinates inside or within %g mm of either ROI", int(excluded.sum()), args.exclusion_mm)

    a_given_b = np.full(n_gray, np.nan, dtype=np.float32)
    b_given_a = np.full(n_gray, np.nan, dtype=np.float32)
    for start in range(0, n_gray, args.batch_size):
        stop = min(start + args.batch_size, n_gray)
        keep = ~excluded[start:stop]
        if not keep.any():
            continue
        target_z = _runwise_standardize(np.asarray(image.dataobj[:, start:stop], dtype=np.float64)[:, keep], run_trs)
        columns = start + np.flatnonzero(keep)
        a_given_b[columns], b_given_a[columns] = loro_partial_correlations(target_z, series["a"], series["b"], run_trs)
        log.info("Processed %d/%d grayordinates", stop, n_gray)

    stem, names = roi_mean_connectivity_names(args.roi_a, args.roi_b, label)
    out_path = args.output_dir / f"{stem}.dscalar.nii"
    _save_dscalar(np.stack([a_given_b, b_given_a]), names, image, out_path)
    (supporting_dir / f"{stem}.json").write_text(json.dumps({
        "analysis": "roi_mean_timeseries_leave_one_run_out_partial_pearson_correlation",
        "analysis_scope": "ROI-mean fMRI time-series connectivity; not a connective-field model",
        "preprocessing_option": args.preprocessing,
        "preprocessing_label": label,
        "preprocessing": ROI_MEAN_PREPROCESSING[label],
        "dtseries": str(args.dtseries),
        "run_trs": run_trs.tolist(),
        "n_timepoints": int(n_trs),
        "roi_a": args.roi_a,
        "roi_b": args.roi_b,
        "roi_vertices": {args.roi_a: int(mask_a.sum()), args.roi_b: int(mask_b.sum())},
        "roi_means": "bilateral: both hemispheres pooled",
        "standardization": "within-run z-score of every vertex and both ROI means, all timepoints retained",
        "partial_correlation": (
            "for each held-out run, the target and the first ROI mean are regressed on the second ROI mean (intercept and slope from "
            "the other runs), the correlation of the held-out residuals is taken, and the runs are averaged weighted by TRs; "
            "the second map swaps the roles of the two ROIs"),
        "exclusion": f"grayordinates inside or within {args.exclusion_mm:g} mm of either ROI are NaN; "
                     "distance is the shortest path along the edges of the midthickness mesh (surface bm)",
        "n_excluded": int(excluded.sum()),
        "cifti": str(out_path),
        "map_names": names,
    }, indent=2) + "\n")
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
