"""Searchlight RSA restricted to one channel-clustering's channel subset.

Subsets the raw pe-av-small-16-frame AV embedding to the channels of one
channel cluster (indices taken directly from ``channel_labels.npy``, whose
row order matches this embedding file's column order — see
``channel_timeseries_clustering.load_channel_timeseries``), writes it as a
standalone embedding file, then calls ``searchlight.py`` unchanged as the
RSA engine. Same fMRI/searchlight configuration as the primary group-average
AV run in ``rsa/analysis.sh`` (k=100, bin5s/skip5s, delay5s, spearman,
raw preprocessing), so the output map is directly comparable to
``rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "pe-av-small-16-frame"
FULL_EMBEDDING = (
    f"/home/amin/Research/Representation/Movie/outputs/model_embeddings/{MODEL}/"
    f"bin5s_skip5s/{MODEL}_av.npy"
)
CHANNEL_LABELS = (
    "/home/amin/Research/Representation/Movie/outputs/cluster/peav/"
    "_channel_timeseries_model_selection/norm-zscore_raw/selected_maps/"
    "sreduce-mds_snc2_landmarks1000_extk8_iter300/"
    "sreduce-mds_snc2_landmarks1000_extk8_iter300_scluster-kmeans_k8/channel_labels.npy"
)
CLUSTER_ID = 7
REDUCER_TAG = "sreduce-mds_snc2_landmarks1000_extk8_iter300"
CLUSTER_TAG = "scluster-kmeans_k8"

PREPROCESSED_DIR = "/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw"
TIMING_CSV = "/home/amin/Research/Representation/Movie/data/movie_timing.csv"
TEMPLATE_CIFTI = (
    "/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/"
    "group_average_raw_cortex_59k.dtseries.nii"
)
HCP_DIR = "/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1"
LEFT_SURFACE = f"{HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE = f"{HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH = "/opt/workbench/bin_linux64/wb_command"
GEODESIC_CACHE_DIR = "/home/amin/Research/Representation/Movie/outputs/rsa/_geodesic_cache"
K = 100
BIN_SEC = 5.0
SKIP_SEC = 5.0
DELAY_SEC = 5.0
METHOD = "spearman"
TR = 1.0

OUTPUT_DIR = "/home/amin/Research/Representation/Movie/outputs/rsa/_channel_subset_searchlight"
EMBEDDINGS_SCRATCH_DIR = Path(OUTPUT_DIR) / "embeddings"
SUBSET_MODEL_NAME = "peav_cluster7"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--gpu-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def subset_columns(full: np.ndarray, labels: np.ndarray, cluster_id: int) -> tuple[np.ndarray, np.ndarray]:
    if full.shape[1] != labels.shape[0]:
        raise ValueError(
            f"Embedding has {full.shape[1]} channels, channel_labels has {labels.shape[0]}"
        )
    channel_indices = np.where(labels == cluster_id)[0]
    return full[:, channel_indices], channel_indices


def write_subset_embedding() -> tuple[Path, int]:
    labels = np.load(CHANNEL_LABELS)
    full = np.load(FULL_EMBEDDING)
    subset, channel_indices = subset_columns(full, labels, CLUSTER_ID)
    out_path = EMBEDDINGS_SCRATCH_DIR / SUBSET_MODEL_NAME / "bin5s_skip5s" / f"{SUBSET_MODEL_NAME}_av.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, subset)
    return out_path, channel_indices.size


def run(args: argparse.Namespace) -> Path:
    subset_path, n_channels = write_subset_embedding()
    print(f"Channel subset: cluster {CLUSTER_ID} of {REDUCER_TAG}/{CLUSTER_TAG} -> "
          f"{n_channels} channels, embedding saved to {subset_path}")

    combined_output = (
        Path(args.output_dir)
        / f"rsa_59k_{REDUCER_TAG}_{CLUSTER_TAG}_cluster{CLUSTER_ID}.dscalar.nii"
    )
    cmd = [
        sys.executable, str(ROOT / "rsa" / "searchlight.py"),
        "--preprocessed-dir", PREPROCESSED_DIR,
        "--fmri-suffix", "raw",
        "--timing-csv", TIMING_CSV,
        "--embeddings-dir", str(EMBEDDINGS_SCRATCH_DIR),
        "--template-cifti", TEMPLATE_CIFTI,
        "--output-dir", args.output_dir,
        "--subject", "group_average",
        "--model", SUBSET_MODEL_NAME,
        "--modality", "av",
        "--k", str(K),
        "--bin-sec", str(BIN_SEC),
        "--skip-sec", str(SKIP_SEC),
        "--delay-sec", str(DELAY_SEC),
        "--method", METHOD,
        "--tr", str(TR),
        "--left-surface", LEFT_SURFACE,
        "--right-surface", RIGHT_SURFACE,
        "--workbench", WORKBENCH,
        "--geodesic-cache-dir", GEODESIC_CACHE_DIR,
        "--combined-output", str(combined_output),
        "--gpu-batch-size", str(args.gpu_batch_size),
        "--n-blocks", "1",
    ]
    subprocess.run(cmd, check=True)
    return combined_output


def main(argv=None) -> None:
    output_path = run(parse_args(argv))
    print(f"Combined output: {output_path}")


if __name__ == "__main__":
    main()
