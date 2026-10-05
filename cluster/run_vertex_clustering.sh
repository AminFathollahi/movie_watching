#!/usr/bin/env bash
# Run the standalone vertex dimensionality-reduction/clustering sweep.
# This does not invoke or modify the temporal-state x spatial-network analysis.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"

DATA_BASE="${DATA_BASE:-$ROOT/data}"
OUTPUTS_BASE="${OUTPUTS_BASE:-$ROOT/outputs}"
GROUP_AVG_CIFTI="${GROUP_AVG_CIFTI:-${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii}"
OUTDIR="${OUTDIR:-${OUTPUTS_BASE}/cluster}"
CONDA_ENV="${CONDA_ENV:-movie}"

N_COMPONENTS="${N_COMPONENTS:-3}"
N_LANDMARKS="${N_LANDMARKS:-2000}"
KMEANS_CLUSTERS="${KMEANS_CLUSTERS:-4}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-100}"
MIN_SAMPLES="${MIN_SAMPLES:-10}"
BIRCH_THRESHOLD="${BIRCH_THRESHOLD:-0.5}"
REDUCTIONS="${REDUCTIONS:-pca mds isomap tsne fastica}"
CLUSTERERS="${CLUSTERERS:-kmeans hdbscan birch}"

read -r -a REDUCTION_ARGS <<< "$REDUCTIONS"
read -r -a CLUSTERER_ARGS <<< "$CLUSTERERS"

conda run --no-capture-output -n "$CONDA_ENV" python \
    "${SCRIPT_DIR}/vertex_clustering.py" \
    --input-cifti "$GROUP_AVG_CIFTI" \
    --template-cifti "$GROUP_AVG_CIFTI" \
    --output-dir "$OUTDIR" \
    --analysis-label group_average \
    --reductions "${REDUCTION_ARGS[@]}" \
    --clusterers "${CLUSTERER_ARGS[@]}" \
    --n-components "$N_COMPONENTS" \
    --n-landmarks "$N_LANDMARKS" \
    --kmeans-clusters "$KMEANS_CLUSTERS" \
    --min-cluster-size "$MIN_CLUSTER_SIZE" \
    --min-samples "$MIN_SAMPLES" \
    --birch-threshold "$BIRCH_THRESHOLD"
