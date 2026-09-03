#!/usr/bin/env bash
# Tune reducers/dimensionality first, then cluster selected 2-D/3-D/best-D embeddings.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_BASE="${DATA_BASE:-/home/amin/Research/Representation/Movie/data}"
OUTPUTS_BASE="${OUTPUTS_BASE:-/home/amin/Research/Representation/Movie/outputs}"
CONDA_ENV="${CONDA_ENV:-movie}"
TEMPLATE_CIFTI="${TEMPLATE_CIFTI:-${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii}"
PRE_PCA="${PRE_PCA:-${OUTPUTS_BASE}/cluster/group_average/_voxel_timeseries/norm-zscore_prepca50_nc3_landmarks2000/voxel_timeseries_prepca50.npy}"

conda run --no-capture-output -n "$CONDA_ENV" python \
    "${SCRIPT_DIR}/voxel_timeseries_model_selection.py" \
    --pre-pca "$PRE_PCA" \
    --template-cifti "$TEMPLATE_CIFTI" \
    --output-dir "${OUTPUTS_BASE}/cluster" \
    --analysis-label group_average "$@"
