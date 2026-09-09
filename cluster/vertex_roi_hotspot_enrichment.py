"""ROI and AV-hotspot enrichment for vertex clusters.

For the screened vertex clusterings, tests whether individual clusters
over-represent (a) named auditory/visual/audiovisual Glasser ROI groups or
(b) the top decile of the full-AV-embedding searchlight RSA map, against a
hypergeometric null over the stimulus-mask population (the population every
clustering was fit on). Reuses ``select_config``/``labels_path``/
``vertex_exclude_labels``/``_config_tag`` from ``channel_vertex_alignment.py``
and ``_roi_indices`` from ``heldout_roi_alignment.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, hypergeom

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLUSTER_DIR))

from channel_vertex_alignment import (  # noqa: E402
    _config_tag, labels_path, select_config, vertex_exclude_labels,
)
from heldout_roi_alignment import _roi_indices  # noqa: E402
from cifti_io import get_bm_axis, load_named_map  # noqa: E402

OUTPUT_DIR = Path("/home/amin/Research/Representation/Movie/outputs/cluster/_vertex_roi_hotspot_enrichment")
VERTEX_MS_DIR = Path(
    "/home/amin/Research/Representation/Movie/outputs/cluster/group_average/_vertex/norm-zscore_raw"
)
GLASSER_DLABEL = (
    "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/"
    "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"
)
AV_RSA_MAP = (
    "/home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average/pe-av-small-16-frame_av/"
    "rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii"
)
AV_RSA_MAP_NAME = "searchlight_spearman_rho"

ROI_GROUPS = {
    "auditory": "A1,MBelt,LBelt,PBelt,RI,A4,A5",
    "visual": "V1,MST,FFC",
    "audiovisual": "STGa,STSda,STSdp,STSva,STSvp,STV,TA2,TPOJ1,TPOJ2,TPOJ3",
}

CONFIGS = {
    "best_overall": dict(
        reducer_tag="sreduce-tsne_snc4_landmarks1000_perp50_extk4_iter1000",
        cluster_tag="scluster-hdbscan_mcs574_ms50",
    ),
    "full_coverage": dict(
        reducer_tag="sreduce-fastica_snc5_alg-deflation_fun-logcosh",
        cluster_tag="scluster-birch_threshold1p5_bf100",
    ),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex-model-selection-dir", default=VERTEX_MS_DIR, type=Path)
    parser.add_argument("--glasser-dlabel", default=GLASSER_DLABEL)
    parser.add_argument("--av-rsa-map", default=AV_RSA_MAP)
    parser.add_argument("--av-rsa-map-name", default=AV_RSA_MAP_NAME)
    parser.add_argument("--hotspot-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    return parser.parse_args(argv)


def load_roi_groups(glasser_dlabel: str, bm_axis) -> dict[str, np.ndarray]:
    args = SimpleNamespace(
        roi=[f"{name}={parcels}" for name, parcels in ROI_GROUPS.items()],
        glasser_dlabel=glasser_dlabel,
    )
    return _roi_indices(args, bm_axis)


def hotspot_indices(av_rsa_map: str, map_name: str, fraction: float, n_grayordinates: int) -> np.ndarray:
    rho = load_named_map(av_rsa_map, map_name)
    if rho.shape[0] != n_grayordinates:
        raise ValueError(f"AV RSA map has {rho.shape[0]} grayordinates, expected {n_grayordinates}")
    n_top = int(round(fraction * n_grayordinates))
    return np.argsort(rho)[-n_top:]


def enrichment_table(labels: np.ndarray, cluster_ids: list[int], population_mask: np.ndarray,
                     categories: dict[str, np.ndarray]) -> pd.DataFrame:
    pop_size = int(population_mask.sum())
    membership = {name: np.isin(np.arange(labels.shape[0]), idx) for name, idx in categories.items()}
    category_in_pop = {name: int((population_mask & mask).sum()) for name, mask in membership.items()}
    rows = []
    for cid in cluster_ids:
        in_cluster = labels == cid
        n_cluster = int(in_cluster.sum())
        for name, mask in membership.items():
            overlap = int((in_cluster & mask).sum())
            k_pop = category_in_pop[name]
            expected = n_cluster * k_pop / pop_size if pop_size else np.nan
            fold = overlap / expected if expected > 0 else np.nan
            p = float(hypergeom.sf(overlap - 1, pop_size, k_pop, n_cluster))
            rows.append({
                "cluster_id": cid, "n_cluster": n_cluster, "category": name,
                "n_category_in_population": k_pop, "overlap": overlap,
                "expected_overlap": expected, "fold_enrichment": fold, "p": p,
            })
    table = pd.DataFrame(rows)
    table["q_bh"] = false_discovery_control(table["p"], method="bh")
    return table


def run(args: argparse.Namespace) -> None:
    bm_axis = get_bm_axis(str(AV_RSA_MAP))
    n_grayordinates = len(bm_axis)
    roi_groups = load_roi_groups(args.glasser_dlabel, bm_axis)
    hotspot = hotspot_indices(args.av_rsa_map, args.av_rsa_map_name, args.hotspot_fraction, n_grayordinates)
    print(f"ROI group sizes: {({k: int(v.size) for k, v in roi_groups.items()})}")
    print(f"Hotspot: top {args.hotspot_fraction:.0%} = {hotspot.size} grayordinates")

    selected_csv = args.vertex_model_selection_dir / "selected_clusterings.csv"
    for config_name, spec in CONFIGS.items():
        row = select_config(selected_csv, role="latent_best",
                            reducer_tag=spec["reducer_tag"], cluster_tag=spec["cluster_tag"])
        labels_file = labels_path(args.vertex_model_selection_dir, row, "spatial_vertex_labels.npy")
        labels = np.load(labels_file)
        exclude = vertex_exclude_labels(labels_file)
        cluster_ids = sorted(int(c) for c in np.unique(labels) if c not in exclude)
        population_mask = ~np.isin(labels, exclude)
        tag = _config_tag(row, len(cluster_ids), "v")
        print(f"\n[{config_name}] {tag}: {len(cluster_ids)} clusters, "
              f"population={int(population_mask.sum())} vertices")

        roi_table = enrichment_table(labels, cluster_ids, population_mask, roi_groups)
        hotspot_table = enrichment_table(
            labels, cluster_ids, population_mask, {"av_hotspot": hotspot}
        )

        out_dir = args.output_dir / f"{config_name}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        roi_table.to_csv(out_dir / "roi_enrichment.csv", index=False)
        hotspot_table.to_csv(out_dir / "hotspot_overlap.csv", index=False)
        (out_dir / "manifest.json").write_text(pd.Series({
            "config_name": config_name, "reducer_tag": spec["reducer_tag"],
            "cluster_tag": spec["cluster_tag"], "labels_file": str(labels_file),
            "n_clusters": len(cluster_ids), "population_size": int(population_mask.sum()),
            "hotspot_fraction": args.hotspot_fraction, "hotspot_size": int(hotspot.size),
            "roi_group_sizes": {k: int(v.size) for k, v in roi_groups.items()},
            "null": "hypergeometric, one-sided over-representation",
            "correction": "BH-FDR within each table (ROI x cluster, hotspot x cluster separately)",
        }).to_json(indent=2))

        n_roi_sig = int((roi_table["q_bh"] < args.alpha).sum())
        n_hot_sig = int((hotspot_table["q_bh"] < args.alpha).sum())
        print(f"  ROI enrichment: {n_roi_sig}/{len(roi_table)} at q<{args.alpha}")
        print(f"  Hotspot overlap: {n_hot_sig}/{len(hotspot_table)} at q<{args.alpha}")
        print(roi_table.sort_values('q_bh').to_string(index=False))
        print(hotspot_table.sort_values('q_bh').to_string(index=False))


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
