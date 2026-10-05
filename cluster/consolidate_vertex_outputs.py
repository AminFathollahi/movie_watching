"""Consolidate selected vertex-clustering maps into review CIFTIs.

Creates six review dlabels, including one stable all-best file:

* one 2-D/3-D/latent-best file for each clustering algorithm
* one unmasked file containing every selected clustering
* ``best_vertex_clusterings.dlabel.nii`` with concise ``*_best`` map names
* one stimulus-regressor-masked file containing those same maps

Each map preserves its own label table. After all outputs pass a
round-trip validation, ``--delete-duplicates`` removes the individual source
dlabels while retaining raw label
arrays, reports, embeddings, and sweep tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUTPUTS  # noqa: E402

DEFAULT_SELECTION_ROOT = OUTPUTS / "cluster/group_average/_vertex/norm-zscore_raw"
DEFAULT_STIMULUS_MAP = (
    OUTPUTS / "sitmulus_regressor_cifti/"
    "HCP_movie_stimulus_correlation_5sdelay_normalized.dscalar.nii"
)
REDUCTION_ORDER = {name: i for i, name in enumerate(
    ("pca", "mds", "isomap", "tsne", "fastica", "umap")
)}
CLUSTER_ORDER = {name: i for i, name in enumerate(("kmeans", "hdbscan", "birch"))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--stimulus-map", type=Path, default=DEFAULT_STIMULUS_MAP)
    parser.add_argument("--delete-duplicates", action="store_true")
    return parser.parse_args()


def _method_from_reducer_tag(tag: str) -> str:
    for method in REDUCTION_ORDER:
        if tag.startswith(f"sreduce-{method}_"):
            return method
    raise ValueError(f"Cannot identify reduction method from {tag}")


def _component_count(tag: str) -> int:
    marker = "_snc"
    return int(tag.split(marker, 1)[1].split("_", 1)[0])


def _concise_map_name(record: dict, report: dict) -> str:
    reducer = _method_from_reducer_tag(record["reducer_tag"])
    dimensions = _component_count(record["reducer_tag"])
    clusterer = record["cluster_method"]
    params = report["parameters"]
    if clusterer == "kmeans":
        suffix = f"k{params['n_clusters']}"
    elif clusterer == "hdbscan":
        suffix = f"mcs{params['min_cluster_size']}_ms{params['min_samples']}"
    else:
        threshold = f"{params['threshold']:g}".replace(".", "p")
        suffix = f"threshold{threshold}_bf{params['branching_factor']}"
    best = "_bestdim" if "latent_best" in record.get("selection_role", "") else ""
    return f"{dimensions}d{best}_{reducer}_{clusterer}_{suffix}"


def _best_map_name(record: dict) -> str:
    role = record["selection_role"]
    role_name = {"display_2d": "2d", "display_3d": "3d", "latent_best": "best"}[role]
    return f"{record['reduction_method']}_{role_name}_{record['cluster_method']}_best"


def load_records(root: Path) -> list[dict]:
    manifest_path = root / "selected_maps_manifest.json"
    records = json.loads(manifest_path.read_text())
    enriched = []
    for record in records:
        report_path = Path(record["report"])
        dlabel_path = Path(record["dlabel"])
        report = json.loads(report_path.read_text())
        reducer = _method_from_reducer_tag(record["reducer_tag"])
        enriched.append({
            **record,
            "report_data": report,
            "source_dlabel": dlabel_path,
            "source_labels": report_path.parent / "spatial_vertex_labels.npy",
            "map_name": _concise_map_name(record, report),
            "n_components": _component_count(record["reducer_tag"]),
            "selection_role": record.get("selection_role", "display_unknown"),
            "reduction_method": reducer,
            "n_clusters": int(report["n_clusters"]),
            "noise_fraction": float(report["noise_fraction"]),
        })
    enriched.sort(key=lambda row: (
        row["n_components"], REDUCTION_ORDER[row["reduction_method"]],
        CLUSTER_ORDER[row["cluster_method"]],
    ))
    names = [row["map_name"] for row in enriched]
    if len(names) != len(set(names)):
        raise ValueError("Concise map names are not unique")
    return enriched


def write_multimap(records: list[dict], output_path: Path,
                   mask: np.ndarray | None = None,
                   map_suffix: str = "") -> None:
    if not records:
        raise ValueError(f"No maps supplied for {output_path.name}")
    arrays = []
    names = []
    tables = []
    brain_axis = None
    nifti_header = None
    for record in records:
        image = nib.load(str(record["source_dlabel"]))
        if image.shape[0] != 1:
            raise ValueError(f"Expected one source map: {record['source_dlabel']}")
        axis = image.header.get_axis(0)
        current_brain_axis = image.header.get_axis(1)
        if brain_axis is None:
            brain_axis = current_brain_axis
            nifti_header = image.nifti_header
        elif current_brain_axis != brain_axis:
            raise ValueError(f"BrainModelAxis mismatch: {record['source_dlabel']}")
        values = np.asarray(image.dataobj)[0].astype(np.float32, copy=False)
        if mask is not None:
            if mask.shape != values.shape:
                raise ValueError(f"Mask shape {mask.shape} != CIFTI map shape {values.shape}")
            values = values * mask
        arrays.append(values)
        names.append(record["map_name"] + map_suffix)
        tables.append(axis.label[0])
    label_axis = nib.cifti2.LabelAxis(names, tables)
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Cifti2Image(np.vstack(arrays), header=header, nifti_header=nifti_header),
        str(output_path),
    )


def validate_multimap(path: Path, expected: list[dict],
                      mask: np.ndarray | None = None,
                      map_suffix: str = "") -> None:
    image = nib.load(str(path))
    axis = image.header.get_axis(0)
    data = np.asarray(image.dataobj)
    expected_names = [record["map_name"] + map_suffix for record in expected]
    if image.shape != (len(expected), 108_441):
        raise ValueError(f"Unexpected shape for {path}: {image.shape}")
    if list(axis.name) != expected_names:
        raise ValueError(f"Map-name/order mismatch for {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in {path}")
    if mask is not None and np.any(data[:, mask == 0] != 0):
        raise ValueError(f"Nonzero labels outside stimulus mask in {path}")
    for index, record in enumerate(expected):
        data_keys = set(np.unique(data[index]).astype(int))
        table_keys = set(axis.label[index].keys())
        unexpected_unused = table_keys - data_keys
        if (
            not data_keys.issubset(table_keys)
            or 0 not in table_keys
            or (mask is None and unexpected_unused not in (set(), {0}))
        ):
            raise ValueError(f"Label-table mismatch in {path}, map {index}")


def write_map_index(rows: list[dict], output_path: Path) -> None:
    fields = [
        "cifti_file", "map_index_1based", "map_name", "n_components",
        "selection_role", "reduction_method", "clustering_method", "n_clusters", "noise_fraction",
        "reducer_tag", "full_fit_cluster_tag", "source_labels", "source_report",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_manifest(records: list[dict], review_dir: Path) -> None:
    """Write an authoritative manifest that points only at retained CIFTIs."""
    combined = []
    all_cifti = review_dir / "selected_clusterings_all-2d_3d_bestdim.dlabel.nii"
    best_cifti = review_dir / "best_vertex_clusterings.dlabel.nii"
    masked_cifti = review_dir / "selected_clusterings_all-2d_3d_stimregressor.dlabel.nii"
    algorithm_positions = {method: 0 for method in CLUSTER_ORDER}
    for all_index, record in enumerate(records, start=1):
        method = record["cluster_method"]
        algorithm_positions[method] += 1
        combined.append({
            "map_name": record["map_name"],
            "n_components": record["n_components"],
            "selection_role": record["selection_role"],
            "reduction_method": record["reduction_method"],
            "clustering_method": method,
            "n_clusters": record["n_clusters"],
            "noise_fraction": record["noise_fraction"],
            "all_cifti": str(all_cifti),
            "all_cifti_map_index_1based": all_index,
            "best_cifti": str(best_cifti),
            "best_cifti_map_name": _best_map_name(record),
            "best_cifti_map_index_1based": all_index,
            "algorithm_cifti": str(
                review_dir / f"selected_{method}_clusterings_2d_3d_bestdim.dlabel.nii"
            ),
            "algorithm_cifti_map_index_1based": algorithm_positions[method],
            "stimregressor_cifti": str(masked_cifti),
            "stimregressor_map_name": record["map_name"] + "_stimregressor",
            "stimregressor_cifti_map_index_1based": all_index,
            "source_labels": str(record["source_labels"]),
            "source_report": record["report"],
            "reducer_tag": record["reducer_tag"],
            "full_fit_cluster_tag": record["full_fit_cluster_tag"],
        })
    (review_dir / "combined_maps_manifest.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )


def delete_duplicate_dlabels(records: list[dict], selection_root: Path,
                             keep: set[Path]) -> list[Path]:
    targets = {Path(record["source_dlabel"]) for record in records}
    selected_maps = selection_root / "selected_maps"
    if selected_maps.exists():
        targets.update(selected_maps.rglob("*.dlabel.nii"))
    review_dir = selection_root / "review_ciftis"
    if review_dir.exists():
        targets.update(review_dir.rglob("*.dlabel.nii"))
    removed = []
    for path in sorted(targets):
        resolved = path.resolve()
        if resolved in keep or not path.exists():
            continue
        path.unlink()
        removed.append(path)
    archive = selected_maps / "superseded_unscaled_hdbscan"
    if archive.exists():
        shutil.rmtree(archive)
    return removed


def main() -> None:
    args = parse_args()
    records = load_records(args.selection_root)
    review_dir = args.selection_root / "review_ciftis"
    stimulus_image = nib.load(str(args.stimulus_map))
    if stimulus_image.shape != (1, 108_441):
        raise ValueError(f"Unexpected stimulus-map shape: {stimulus_image.shape}")
    source_brain_axis = nib.load(str(records[0]["source_dlabel"])).header.get_axis(1)
    if stimulus_image.header.get_axis(1) != source_brain_axis:
        raise ValueError("Stimulus map and selected cluster maps have different BrainModelAxis")
    stimulus_values = np.asarray(stimulus_image.dataobj)[0]
    stimulus_mask = (
        np.isfinite(stimulus_values) & (stimulus_values > 0)
    ).astype(np.float32)
    groups = {
        "selected_kmeans_clusterings_2d_3d_bestdim.dlabel.nii": [
            row for row in records if row["cluster_method"] == "kmeans"
        ],
        "selected_hdbscan_clusterings_2d_3d_bestdim.dlabel.nii": [
            row for row in records if row["cluster_method"] == "hdbscan"
        ],
        "selected_birch_clusterings_2d_3d_bestdim.dlabel.nii": [
            row for row in records if row["cluster_method"] == "birch"
        ],
        "selected_clusterings_all-2d_3d_bestdim.dlabel.nii": records,
        "best_vertex_clusterings.dlabel.nii": [
            {**row, "map_name": _best_map_name(row)} for row in records
        ],
    }
    index_rows = []
    outputs = []
    for filename, group in groups.items():
        path = review_dir / filename
        write_multimap(group, path)
        validate_multimap(path, group)
        outputs.append(path.resolve())
        for index, record in enumerate(group, start=1):
            index_rows.append({
                "cifti_file": filename,
                "map_index_1based": index,
                "map_name": record["map_name"],
                "n_components": record["n_components"],
                "selection_role": record["selection_role"],
                "reduction_method": record["reduction_method"],
                "clustering_method": record["cluster_method"],
                "n_clusters": record["n_clusters"],
                "noise_fraction": record["noise_fraction"],
                "reducer_tag": record["reducer_tag"],
                "full_fit_cluster_tag": record["full_fit_cluster_tag"],
                "source_labels": str(record["source_labels"]),
                "source_report": record["report"],
            })
    masked_filename = "selected_clusterings_all-2d_3d_stimregressor.dlabel.nii"
    masked_path = review_dir / masked_filename
    write_multimap(records, masked_path, mask=stimulus_mask,
                   map_suffix="_stimregressor")
    validate_multimap(expected=records, path=masked_path, mask=stimulus_mask,
                      map_suffix="_stimregressor")
    outputs.append(masked_path.resolve())
    for index, record in enumerate(records, start=1):
        index_rows.append({
            "cifti_file": masked_filename,
            "map_index_1based": index,
            "map_name": record["map_name"] + "_stimregressor",
            "n_components": record["n_components"],
            "selection_role": record["selection_role"],
            "reduction_method": record["reduction_method"],
            "clustering_method": record["cluster_method"],
            "n_clusters": record["n_clusters"],
            "noise_fraction": record["noise_fraction"],
            "reducer_tag": record["reducer_tag"],
            "full_fit_cluster_tag": record["full_fit_cluster_tag"],
            "source_labels": str(record["source_labels"]),
            "source_report": record["report"],
        })
    write_map_index(index_rows, review_dir / "map_index.csv")
    write_combined_manifest(records, review_dir)
    (review_dir / "README.md").write_text(
        "# Vertex-clustering maps\n\n"
        "Start with `best_vertex_clusterings.dlabel.nii`: it contains every selected "
        "2-D, 3-D, and latent-best map. Names follow "
        "`<reducer>_<2d|3d|best>_<clusterer>_best`. Use "
        "the three algorithm-specific CIFTIs for comparisons of only k-means, "
        "HDBSCAN, or BIRCH.\n\n"
        "`selected_clusterings_all-2d_3d_stimregressor.dlabel.nii` contains the "
        "same maps multiplied by the positive (`stimulus > 0`) binary stimulus-"
        "regressor mask. Its internal map names end in `_stimregressor`.\n\n"
        "Internal map names follow `<2d|3d>_<reducer>_<clusterer>_<key-parameter>`. "
        "`map_index.csv` and `combined_maps_manifest.json` give the full configuration, "
        "cluster count, noise fraction, raw-label path, and report for every map.\n\n"
        "The executed scatterplot notebook is "
        "`movie_watching/cluster/vertex_cluster_scatterplots.ipynb`; "
        "it generates:\n"
        "- Model selection plots: 1 dimensionality elbow figure + 9 cluster selection figures\n"
        "- 54 individual scatterplot figures (6 reducers × 3 clusterers × 3 roles)\n"
        "All figures are saved in `../figures/` with an index at `figures/scatterplot_index.csv`.\n"
    )
    (args.selection_root / "LOOK_HERE.md").write_text(
        "# Start here: selected vertex clusterings\n\n"
        "1. Open `review_ciftis/best_vertex_clusterings.dlabel.nii` for every "
        "selected unmasked 2-D, 3-D, and latent-best map.\n"
        "2. Open `review_ciftis/selected_clusterings_all-2d_3d_stimregressor.dlabel.nii` "
        "for the corresponding positive-stimulus-regressor-masked maps.\n"
        "3. Use the three `review_ciftis/selected_<algorithm>_clusterings_2d_3d_bestdim.dlabel.nii` "
        "files for algorithm-specific comparisons.\n"
        "4. Read `review_ciftis/map_index.csv` for exact map-to-configuration lookup.\n"
        "5. View the embedding scatterplots in `figures/`:\n"
        "   - Model selection: `dimensionality_selection_elbow_curves.png` and `cluster_selection_*.png`\n"
        "   - Individual scatterplots: `scatterplot_<reducer>_<clusterer>_<2d|3d|bestdim>.png` (54 total)\n"
        "   - Index: `figures/scatterplot_index.csv`\n"
        "   - Or inline in `movie_watching/cluster/vertex_cluster_scatterplots.ipynb`\n\n"
        "Individual one-map dlabels "
        "are redundant once consolidation has been validated.\n"
    )
    removed = []
    if args.delete_duplicates:
        removed = delete_duplicate_dlabels(
            records, args.selection_root, set(outputs)
        )
    print(json.dumps({
        "combined_ciftis": [str(path) for path in outputs],
        "map_counts": {name: len(group) for name, group in groups.items()},
        "stimregressor_map_count": len(records),
        "stimregressor_positive_grayordinates": int(stimulus_mask.sum()),
        "duplicates_deleted": len(removed),
    }, indent=2))


if __name__ == "__main__":
    main()
