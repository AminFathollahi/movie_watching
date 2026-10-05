#!/usr/bin/env bash
# Tune reducers/dimensionality first, then cluster selected 2-D/3-D/best-D embeddings.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="${DATA_BASE:-$ROOT/data}"
OUTPUTS_BASE="${OUTPUTS_BASE:-$ROOT/outputs}"
CONDA_ENV="${CONDA_ENV:-movie}"
TEMPLATE_CIFTI="${TEMPLATE_CIFTI:-${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii}"

conda run --no-capture-output -n "$CONDA_ENV" python \
    "${SCRIPT_DIR}/vertex_model_selection.py" \
    --input-cifti "$TEMPLATE_CIFTI" \
    --template-cifti "$TEMPLATE_CIFTI" \
    --output-dir "${OUTPUTS_BASE}/cluster" \
    --analysis-label group_average "$@"
