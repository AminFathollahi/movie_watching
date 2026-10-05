#!/usr/bin/env python3
"""Extract bilateral anterior/posterior CCA ROIs on the cortical surface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, deque
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT as BASE  # noqa: E402

DEFAULT_MAP = (BASE / "outputs/rsa/raw/group_average/pe-av-small-16-frame_av"
               "/k100_delay5s_bin5s_skip5s_spearman"
               "/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy")
DEFAULT_TEMPLATE = (BASE / "data/preprocessed/average_sub/raw"
                    "/group_average_raw_cortex_59k.dtseries.nii")
DEFAULT_GLASSER = (BASE / "data/HCP_S1200_GroupAvg_v1"
                   "/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_"
                   "Group_Colors.59k_fs_LR.dlabel.nii")
DEFAULT_SURF = (BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
                "/CohortAvg.{hem}.midthickness_MSMAll.59k_fs_LR.surf.gii")
DEFAULT_OUTPUT = BASE / "outputs/cf_modeling"
DEFAULT_ROI_CONFIG = Path(__file__).with_name("roi_definitions.json")
N_SURF = 59292

# HCP-MMP areas in temporal cortex (plus temporo-parietal junction areas that
# form the posterior CCA at the default threshold). --temporal-rois can override.
TEMPORAL_ROIS = {
    "A1", "A4", "A5", "AAIC", "AVI", "EC", "FST", "LBelt", "MBelt",
    "PBelt", "PeEc", "PHA1", "PHA2", "PHA3", "PHT", "PI", "PoI1",
    "PoI2", "RI", "STSda", "STSdp", "STSva", "STSvp", "STGa", "TA2",
    "TE1a", "TE1m", "TE1p", "TE2a", "TE2p", "TF", "TGd", "TGv",
    "TPOJ1", "TPOJ2", "TPOJ3", "TSJ1", "TSJ2", "STV",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cca_islands")


def adjacency(faces: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in range(N_SURF)]
    for a, b, c in faces:
        adj[a].extend((b, c)); adj[b].extend((a, c)); adj[c].extend((a, b))
    return adj


def components(mask: np.ndarray, adj: list[list[int]]) -> list[np.ndarray]:
    seen = np.zeros(mask.size, dtype=bool)
    result = []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([int(start)])
        comp = []
        while queue:
            vertex = queue.popleft()
            comp.append(vertex)
            for neighbour in adj[vertex]:
                if mask[neighbour] and not seen[neighbour]:
                    seen[neighbour] = True
                    queue.append(neighbour)
        result.append(np.asarray(comp, dtype=np.int32))
    return result


def component_containing(mask: np.ndarray, seed: int,
                         adj: list[list[int]]) -> np.ndarray:
    """Return the connected suprathreshold component containing ``seed``."""
    if seed < 0 or seed >= mask.size or not mask[seed]:
        raise ValueError("Seed must be inside the supplied surface mask")
    seen = np.zeros(mask.size, dtype=bool)
    seen[seed] = True
    queue = deque([int(seed)])
    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbour in adj[vertex]:
            if mask[neighbour] and not seen[neighbour]:
                seen[neighbour] = True
                queue.append(int(neighbour))
    return np.asarray(result, dtype=np.int32)


def minimum_degree_core(mask: np.ndarray, adj: list[list[int]],
                        min_degree: int) -> np.ndarray:
    """Keep vertices with at least ``min_degree`` retained mesh neighbours."""
    mask = np.asarray(mask, dtype=bool).copy()
    if min_degree < 0:
        raise ValueError("Minimum degree cannot be negative")
    if min_degree == 0 or not np.any(mask):
        return mask
    neighbours = [np.asarray(sorted(set(items)), dtype=np.int32) for items in adj]
    degree = np.zeros(mask.size, dtype=np.int16)
    for vertex in np.flatnonzero(mask):
        degree[vertex] = int(np.sum(mask[neighbours[vertex]]))
    queue = deque(int(vertex) for vertex in np.flatnonzero(mask & (degree < min_degree)))
    while queue:
        vertex = queue.popleft()
        if not mask[vertex] or degree[vertex] >= min_degree:
            continue
        mask[vertex] = False
        for neighbour in neighbours[vertex]:
            if mask[neighbour]:
                degree[neighbour] -= 1
                if degree[neighbour] < min_degree:
                    queue.append(int(neighbour))
    return mask


def surface_ample_regions(
    values: np.ndarray,
    seed_components: list[np.ndarray],
    adj: list[list[int]],
    peak_fraction: float,
    min_degree: int = 3,
) -> list[dict]:
    """Grow peak-connected surface AMPLE ROIs with local mesh support."""
    if len(seed_components) != 2:
        raise ValueError("Exactly two seed components are required")
    if not 0.0 < peak_fraction < 1.0:
        raise ValueError("AMPLE peak fraction must be between zero and one")
    results = []
    candidate_masks = []
    peak_vertices = []
    for seed_component in seed_components:
        peak_vertex = int(seed_component[np.argmax(values[seed_component])])
        peak_value = float(values[peak_vertex])
        if not np.isfinite(peak_value) or peak_value <= 0:
            raise ValueError(
                "Surface AMPLE requires a finite positive peak in each seed island")
        ample_threshold = peak_fraction * peak_value
        raw_component = component_containing(
            np.isfinite(values) & (values > ample_threshold), peak_vertex, adj)
        raw_mask = np.zeros(values.size, dtype=bool)
        raw_mask[raw_component] = True
        candidate_masks.append(raw_mask)
        peak_vertices.append(peak_vertex)
        results.append({
            "peak_vertex": peak_vertex,
            "peak_value": peak_value,
            "peak_fraction": float(peak_fraction),
            "ample_threshold": float(ample_threshold),
            "min_within_mask_degree": int(min_degree),
            "n_vertices_before_support_filter": int(raw_component.size),
        })

    overlap = candidate_masks[0] & candidate_masks[1]
    if np.any(overlap):
        raise RuntimeError(
            f"Surface AMPLE peak regions overlap at {int(np.sum(overlap))} vertices; "
            "use a higher peak fraction")

    for result, candidate_mask, peak_vertex in zip(
            results, candidate_masks, peak_vertices):
        supported_mask = minimum_degree_core(candidate_mask, adj, min_degree)
        if not supported_mask[peak_vertex]:
            raise RuntimeError(
                f"AMPLE peak {peak_vertex} does not survive the {min_degree}-core "
                "support filter; lower --ample-min-degree or peak fraction")
        supported_component = component_containing(supported_mask, peak_vertex, adj)
        result.update({
            "vertices": supported_component,
            "raw_overlap_vertices": 0,
            "n_vertices_removed_by_support_filter": int(
                np.sum(candidate_mask) - supported_component.size),
        })
    if np.intersect1d(results[0]["vertices"], results[1]["vertices"]).size:
        raise RuntimeError("Surface AMPLE regions overlap at this peak fraction")
    return results


def merge_saddle_threshold(
    values: np.ndarray,
    first_component: np.ndarray,
    second_component: np.ndarray,
    adj: list[list[int]],
) -> float:
    """Return the superlevel-set value where two seed components first join.

    Vertices are activated from highest to lowest map value.  The returned
    value is the first activation level at which the two seed components are
    connected.  Because ROI masks use ``values > threshold``, using the
    returned value itself excludes every saddle-level vertex and therefore
    retains the largest strictly separated components.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.size != len(adj):
        raise ValueError("values and adjacency must describe the same surface")
    first_component = np.asarray(first_component, dtype=np.int32)
    second_component = np.asarray(second_component, dtype=np.int32)
    if first_component.size == 0 or second_component.size == 0:
        raise ValueError("Both seed components must contain at least one vertex")

    parent = np.arange(values.size, dtype=np.int32)
    rank = np.zeros(values.size, dtype=np.int8)
    active = np.zeros(values.size, dtype=bool)
    has_first = np.zeros(values.size, dtype=bool)
    has_second = np.zeros(values.size, dtype=bool)
    has_first[first_component] = True
    has_second[second_component] = True

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    def union(a: int, b: int) -> int:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return root_a
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        has_first[root_a] |= has_first[root_b]
        has_second[root_a] |= has_second[root_b]
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1
        return root_a

    for vertex in np.argsort(values)[::-1]:
        vertex = int(vertex)
        if not np.isfinite(values[vertex]):
            break
        active[vertex] = True
        for neighbour in adj[vertex]:
            if active[neighbour]:
                union(vertex, int(neighbour))
        root = find(vertex)
        if has_first[root] and has_second[root]:
            return float(values[vertex])
    raise RuntimeError("Seed components never merge on the finite surface graph")


def load_map(path: Path, map_index: int) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.asarray(np.load(path)).squeeze()
    else:
        data = np.asarray(nib.load(path).get_fdata())[map_index].squeeze()
    if data.ndim != 1:
        raise ValueError(f"Map must resolve to one vector, got shape {data.shape}")
    return data.astype(np.float32)


def resolve_threshold(values: np.ndarray, threshold: float | None,
                      top_percent: float | None,
                      registry_threshold: float) -> tuple[float, str]:
    """Resolve a fixed or upper-tail threshold and describe its provenance."""
    if threshold is not None and top_percent is not None:
        raise ValueError("Use either --threshold or --top-percent, not both")
    if top_percent is not None:
        if not 0.0 < top_percent < 100.0:
            raise ValueError("--top-percent must be between 0 and 100")
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Cannot percentile-threshold a map with no finite values")
        return float(np.percentile(finite, 100.0 - top_percent)), "top_percent"
    if threshold is None:
        threshold = registry_threshold
    return float(threshold), "fixed"


def suffixed_roi_names(base_names: dict[str, str], suffix: str | None) -> dict[str, str]:
    """Return variant-specific ROI names while preserving canonical defaults."""
    if not suffix:
        return dict(base_names)
    clean = suffix.strip().strip("_")
    if not clean or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in clean):
        raise ValueError("--roi-suffix may contain only letters, digits, and underscores")
    return {role: f"{name}_{clean}" for role, name in base_names.items()}


def surface_mask_to_cifti(mask_l: np.ndarray, mask_r: np.ndarray,
                          bm_axis) -> np.ndarray:
    """Convert full 59k L/R surface masks to the template's cortical axis."""
    grayordinates = np.zeros(len(bm_axis), dtype=np.float32)
    for structure, slc, part in bm_axis.iter_structures():
        if "CORTEX_LEFT" in structure:
            grayordinates[slc] = mask_l[part.vertex]
        elif "CORTEX_RIGHT" in structure:
            grayordinates[slc] = mask_r[part.vertex]
    return grayordinates


def save_dscalar(data: np.ndarray, map_names: list[str], bm_axis,
                 template: nib.Cifti2Image, path: Path) -> None:
    """Save one or more cortex masks as a CIFTI-2 dense scalar."""
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    image = nib.Cifti2Image(
        np.asarray(data, dtype=np.float32),
        header=header,
        nifti_header=template.nifti_header,
    )
    nib.save(image, path)


def save_island_dlabel(first: np.ndarray, second: np.ndarray, roi_names: dict[str, str],
                       roles: tuple[str, str], title: str,
                       bm_axis, template: nib.Cifti2Image, path: Path) -> None:
    """Save both non-overlapping functional ROIs as one categorical CIFTI map."""
    if np.any((first > 0) & (second > 0)):
        raise ValueError("Functional ROI masks overlap; cannot create island dlabel")
    keys = np.zeros(first.size, dtype=np.int32)
    keys[first > 0] = 1
    keys[second > 0] = 2
    labels = {
        0: ("UNLABELED", (0.0, 0.0, 0.0, 0.0)),
        1: (roi_names[roles[0]], (0.85, 0.16, 0.16, 1.0)),
        2: (roi_names[roles[1]], (0.16, 0.34, 0.85, 1.0)),
    }
    label_axis = nib.cifti2.LabelAxis([title], [labels])
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_axis))
    image = nib.Cifti2Image(keys[None, :], header=header,
                            nifti_header=template.nifti_header)
    nib.save(image, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, dest="map_path")
    parser.add_argument("--map-index", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--threshold", type=float, default=None,
                           help="Fixed map threshold; defaults to the ROI registry value")
    selection.add_argument("--top-percent", type=float, default=None,
                           help="Keep the highest P percent of finite cortical values")
    selection.add_argument(
        "--adaptive-hemisphere-saddle", action="store_true",
        help=("Choose a separate threshold per hemisphere by starting from two "
              "components at --adaptive-seed-top-percent and lowering the "
              "superlevel threshold to their strict pre-merge saddle"),
    )
    selection.add_argument(
        "--surface-ample", type=float, metavar="FRACTION",
        help=("Peak-relative AMPLE fraction (0--1), with peak-connected surface "
              "growth and local mesh support"),
    )
    parser.add_argument(
        "--adaptive-seed-top-percent", type=float, default=1.0,
        help="Initial global upper-tail percentage used to identify the two seed islands",
    )
    parser.add_argument(
        "--ample-seed-top-percent", type=float, default=1.0,
        help="Initial global upper-tail percentage used to locate the two AMPLE peaks",
    )
    parser.add_argument(
        "--ample-min-degree", type=int, default=3,
        help="Minimum number of retained surface-mesh neighbours per AMPLE vertex",
    )
    parser.add_argument("--roi-suffix", default=None,
                        help="Append a variant tag to both ROI names, e.g. peav_2pct")
    parser.add_argument("--min-verts", type=int, default=10)
    parser.add_argument("--min-temporal-fraction", type=float, default=0.5)
    parser.add_argument("--temporal-rois", nargs="+", default=sorted(TEMPORAL_ROIS))
    parser.add_argument("--template-cifti", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--glasser-dlabel", type=Path, default=DEFAULT_GLASSER)
    parser.add_argument("--surface-pattern", default=str(DEFAULT_SURF),
                        help="Format string containing {hem}, replaced by L/R")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.map_path, args.template_cifti, args.glasser_dlabel, args.roi_config):
        if not path.exists():
            raise FileNotFoundError(path)
    registry = json.loads(args.roi_config.read_text())
    roi_config = registry.get("cca_temporal_islands")
    if roi_config is None:
        raise KeyError("ROI registry must define 'cca_temporal_islands'")
    base_names = roi_config["names"]
    roles = ("anterior", "posterior")
    family = "cca"
    roi_names = suffixed_roi_names(base_names, args.roi_suffix)

    values = load_map(args.map_path, args.map_index)
    if args.adaptive_hemisphere_saddle:
        seed_threshold, _ = resolve_threshold(
            values, None, args.adaptive_seed_top_percent,
            float(roi_config["threshold"]))
        resolved_threshold = None
        selection_method = "adaptive_hemisphere_merge_saddle"
    elif args.surface_ample is not None:
        seed_threshold, _ = resolve_threshold(
            values, None, args.ample_seed_top_percent,
            float(roi_config["threshold"]))
        resolved_threshold = None
        selection_method = "surface_connected_ample"
    else:
        resolved_threshold, selection_method = resolve_threshold(
            values, args.threshold, args.top_percent, float(roi_config["threshold"]))
        seed_threshold = None
    template = nib.load(args.template_cifti)
    bm_axis = template.header.get_axis(1)
    cortex_count = sum(len(part.vertex) for name, _, part in bm_axis.iter_structures()
                       if "CORTEX" in name)
    if values.size != cortex_count:
        raise ValueError(f"Map has {values.size} values; template has {cortex_count} cortex grayordinates")

    dlabel = nib.load(args.glasser_dlabel)
    label_data = np.asarray(dlabel.get_fdata())[0]
    label_table = dlabel.header.get_axis(0).label[0]
    key_to_short = {int(key): name[2:-4] if name[1:2] == "_" and name.endswith("_ROI") else name
                    for key, (name, _) in label_table.items()}
    temporal = set(args.temporal_rois)
    selected: dict[str, list[dict]] = {"L": [], "R": []}
    thresholds_by_hemisphere: dict[str, float] = {}
    retained_percent_by_hemisphere: dict[str, float] = {}

    for structure, slc, part in bm_axis.iter_structures():
        if "CORTEX" not in structure:
            continue
        hem = "L" if "LEFT" in structure else "R"
        surface = nib.load(args.surface_pattern.format(hem=hem))
        xyz, faces = surface.darrays[0].data, surface.darrays[1].data
        full_values = np.full(N_SURF, -np.inf, dtype=np.float32)
        full_values[part.vertex] = values[slc]
        offset = 0 if hem == "L" else N_SURF

        surface_adjacency = adjacency(faces)

        def describe_components(threshold_value: float) -> list[dict]:
            candidates = []
            for comp in components(full_values > threshold_value, surface_adjacency):
                parcel_names = [key_to_short.get(int(x), "unknown")
                                for x in label_data[offset + comp]]
                temporal_fraction = np.mean([name in temporal for name in parcel_names])
                if len(comp) < args.min_verts or temporal_fraction < args.min_temporal_fraction:
                    continue
                candidates.append({
                    "vertices": comp,
                    "n_vertices": int(len(comp)),
                    "temporal_fraction": float(temporal_fraction),
                    "centroid_xyz": xyz[comp].mean(axis=0).astype(float).tolist(),
                    "parcels": dict(Counter(parcel_names).most_common()),
                })
            candidates.sort(key=lambda item: item["n_vertices"], reverse=True)
            return candidates

        if args.adaptive_hemisphere_saddle:
            seed_candidates = describe_components(seed_threshold)
            if len(seed_candidates) < 2:
                raise RuntimeError(
                    f"Only {len(seed_candidates)} qualifying temporal seed components "
                    f"in hemisphere {hem} at global top "
                    f"{args.adaptive_seed_top_percent:g}%; cannot estimate a merge saddle."
                )
            seed_components = [seed_candidates[0]["vertices"], seed_candidates[1]["vertices"]]
            hem_threshold = merge_saddle_threshold(
                full_values, seed_components[0], seed_components[1], surface_adjacency)
            thresholds_by_hemisphere[hem] = hem_threshold
            finite_hemisphere = np.asarray(values[slc])[np.isfinite(values[slc])]
            retained_percent_by_hemisphere[hem] = (
                100.0 * float(np.sum(finite_hemisphere > hem_threshold))
                / float(finite_hemisphere.size)
            )

            # Track the two seed islands at the strict pre-merge threshold.
            final_components = components(full_values > hem_threshold, surface_adjacency)
            vertex_to_component = {
                int(vertex): index
                for index, comp in enumerate(final_components)
                for vertex in comp
            }
            tracked = []
            for seed in seed_components:
                comp_index = vertex_to_component.get(int(seed[0]))
                if comp_index is None:
                    raise RuntimeError("Adaptive seed disappeared at its merge threshold")
                tracked.append(final_components[comp_index])
            if np.array_equal(np.sort(tracked[0]), np.sort(tracked[1])):
                raise RuntimeError(
                    f"Adaptive components are not separate in hemisphere {hem}; "
                    "strict superlevel invariant failed"
                )
            candidates = []
            for comp in tracked:
                parcel_names = [key_to_short.get(int(x), "unknown")
                                for x in label_data[offset + comp]]
                temporal_fraction = np.mean([name in temporal for name in parcel_names])
                candidates.append({
                    "vertices": comp,
                    "n_vertices": int(len(comp)),
                    "temporal_fraction": float(temporal_fraction),
                    "centroid_xyz": xyz[comp].mean(axis=0).astype(float).tolist(),
                    "parcels": dict(Counter(parcel_names).most_common()),
                })
            candidates.sort(key=lambda item: item["n_vertices"], reverse=True)
        elif args.surface_ample is not None:
            seed_candidates = describe_components(seed_threshold)
            if len(seed_candidates) < 2:
                raise RuntimeError(
                    f"Only {len(seed_candidates)} qualifying temporal AMPLE seed "
                    f"components in hemisphere {hem} at global top "
                    f"{args.ample_seed_top_percent:g}%")
            seed_components = [seed_candidates[0]["vertices"], seed_candidates[1]["vertices"]]
            ample_items = surface_ample_regions(
                full_values, seed_components, surface_adjacency,
                args.surface_ample, args.ample_min_degree)
            candidates = []
            for item in ample_items:
                comp = item["vertices"]
                parcel_names = [key_to_short.get(int(x), "unknown")
                                for x in label_data[offset + comp]]
                temporal_fraction = np.mean([name in temporal for name in parcel_names])
                candidates.append({
                    **item,
                    "n_vertices": int(len(comp)),
                    "temporal_fraction": float(temporal_fraction),
                    "centroid_xyz": xyz[comp].mean(axis=0).astype(float).tolist(),
                    "parcels": dict(Counter(parcel_names).most_common()),
                })
            candidates.sort(key=lambda item: item["n_vertices"], reverse=True)
            thresholds_by_hemisphere[hem] = float(seed_threshold)
            finite_hemisphere = np.asarray(values[slc])[np.isfinite(values[slc])]
            retained_percent_by_hemisphere[hem] = (
                100.0 * sum(item["n_vertices"] for item in candidates)
                / float(finite_hemisphere.size)
            )
        else:
            hem_threshold = resolved_threshold
            thresholds_by_hemisphere[hem] = hem_threshold
            finite_hemisphere = np.asarray(values[slc])[np.isfinite(values[slc])]
            retained_percent_by_hemisphere[hem] = (
                100.0 * float(np.sum(finite_hemisphere > hem_threshold))
                / float(finite_hemisphere.size)
            )
            candidates = describe_components(hem_threshold)

        if len(candidates) < 2:
            if args.surface_ample is not None:
                threshold_description = (
                    f"surface AMPLE fraction {args.surface_ample:g} from "
                    f"top-{args.ample_seed_top_percent:g}% seeds")
            elif args.adaptive_hemisphere_saddle:
                threshold_description = f"hemisphere threshold {hem_threshold:.10g}"
            else:
                threshold_description = (
                    f"{selection_method} threshold {hem_threshold:.10g}")
            raise RuntimeError(
                f"Only {len(candidates)} qualifying temporal components in hemisphere {hem} "
                f"at {threshold_description}; the requested map does not form two "
                "separable islands under the current component rule. Adjust the selection "
                "or use an explicit merged-island split."
            )
        selected[hem] = candidates[:2]

    bilateral = {roi_names[roles[0]]: {}, roi_names[roles[1]]: {}}
    for hem in ("L", "R"):
        # Surface coordinates use +Y anteriorly: anterior first, posterior second.
        ordered = sorted(
            selected[hem], key=lambda item: item["centroid_xyz"][1], reverse=True)
        bilateral[roi_names[roles[0]]][hem], bilateral[roi_names[roles[1]]][hem] = ordered

    masks_dir = args.output_base / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_map": str(args.map_path),
        "map_index": args.map_index,
        "selection_method": selection_method,
        "threshold": (thresholds_by_hemisphere
                      if (args.adaptive_hemisphere_saddle or args.surface_ample is not None)
                      else resolved_threshold),
        "top_percent": args.top_percent,
        "retained_percent_by_hemisphere": retained_percent_by_hemisphere,
        "n_above_threshold": (
            {
                hem: int(sum(item["n_vertices"] for item in selected[hem]))
                for hem in ("L", "R")
            }
            if args.surface_ample is not None
            else
            {
                hem: int(np.sum(np.isfinite(values[slc])
                                & (values[slc] > thresholds_by_hemisphere[hem])))
                for structure, slc, part in bm_axis.iter_structures()
                if "CORTEX" in structure
                for hem in [("L" if "LEFT" in structure else "R")]
            }
            if args.adaptive_hemisphere_saddle
            else int(np.sum(np.isfinite(values) & (values > resolved_threshold)))
        ),
        "roi_suffix": args.roi_suffix,
        "naming_family": family,
        "anatomical_assignment": (
            "higher centroid Y is anterior; lower centroid Y is posterior"),
        "roi_names": roi_names,
        "selection": {},
    }
    if args.adaptive_hemisphere_saddle:
        metadata["adaptive_rule"] = {
            "name": "strict_premerge_superlevel_saddle",
            "seed_top_percent_global": args.adaptive_seed_top_percent,
            "seed_threshold_global": seed_threshold,
            "comparison": "map_value > hemisphere_threshold",
            "rationale": (
                "Track the two largest temporal components from the global seed "
                "threshold and retain their maximum extent immediately before they merge."
            ),
        }
    if args.surface_ample is not None:
        metadata["ample_rule"] = {
            "name": "surface_connected_ample",
            "expanded_name": "activation mapping as a percentage of local excitation",
            "peak_fraction": args.surface_ample,
            "seed_top_percent_global": args.ample_seed_top_percent,
            "seed_threshold_global": seed_threshold,
            "comparison": "map_value > peak_fraction * local_peak",
            "overlap_policy": "reject the definition if peak regions overlap",
            "surface_support": f"iterative {args.ample_min_degree}-core",
            "rationale": (
                "Normalize each CCA island to its local peak, retain its connected "
                "superlevel component, and remove vertices lacking local mesh support."
            ),
        }
    saved_surface_masks = {}
    for roi, hemis in bilateral.items():
        metadata["selection"][roi] = {}
        saved_surface_masks[roi] = {}
        for hem, item in hemis.items():
            mask = np.zeros(N_SURF, dtype=bool)
            mask[item["vertices"]] = True
            saved_surface_masks[roi][hem] = mask
            out = masks_dir / f"{roi}_{hem}_mask.csv"
            pd.DataFrame({"mask": mask}).to_csv(out, index=False)
            clean = {key: value for key, value in item.items() if key != "vertices"}
            metadata["selection"][roi][hem] = clean
            log.info("%s %s: %d vertices, centroid=%s", roi, hem, item["n_vertices"],
                     np.round(item["centroid_xyz"], 1).tolist())

    cifti_masks = []
    cifti_names = []
    for roi in bilateral:
        cifti_mask = surface_mask_to_cifti(
            saved_surface_masks[roi]["L"], saved_surface_masks[roi]["R"], bm_axis)
        save_dscalar(cifti_mask[None, :], [roi], bm_axis, template,
                     masks_dir / f"{roi}_mask.dscalar.nii")
        cifti_masks.append(cifti_mask)
        cifti_names.append(roi)
    variant_tag = f"_{args.roi_suffix.strip().strip('_')}" if args.roi_suffix else ""
    save_dscalar(np.stack(cifti_masks), cifti_names, bm_axis, template,
                 masks_dir / f"{family}_islands_masks{variant_tag}.dscalar.nii")
    save_island_dlabel(
        cifti_masks[0], cifti_masks[1], roi_names, roles,
        "CCA anterior and posterior islands",
        bm_axis, template,
        masks_dir / f"{family}_islands_labels{variant_tag}.dlabel.nii",
    )

    metadata_path = masks_dir / f"{family}_islands{variant_tag}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Wrote masks and provenance to %s", masks_dir)


if __name__ == "__main__":
    main()
