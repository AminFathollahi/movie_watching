#!/usr/bin/env bash
# rsa/run_topography_control.sh
# Runs rsa/topography_control.py: for a given true-coordinate run, replaces
# the k-NN neighbourhood with random same-size draws (coordinates ignored)
# and compares the rho distribution against the true run's saved results.
# Answers "does the true k-NN neighbourhood do any work beyond an arbitrary
# same-size unit sample?" -- see script docstring.
#
# LAYERS/TRUE_RUN_DIR must match an already-completed
# run_cca_seed_sheet_rsa_truecoords.sh run (this reads its saved *_rho.npy).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
CONDA_ENV="movie"

LAYERS="${LAYERS:-1,9,18,27,34,35}"
K="${K:-100}"
N_DRAWS="${N_DRAWS:-3}"
N_PERM="${N_PERM:-50}"  # small on purpose: actual_rho does not depend on n_perm,
                         # only the discarded null does -- this is a distribution
                         # comparison, not a significance test.
TRUE_RUN_DIR="${TRUE_RUN_DIR:-${OUTPUTS_BASE}/rsa/cca_seed_sheet_rsa/k100_multilayer_truecoords}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRUE_RUN_DIR}/topography_control}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-64}"
PERM_BATCH_SIZE="${PERM_BATCH_SIZE:-20}"
SEED="${SEED:-42}"

conda run --no-capture-output -n "$CONDA_ENV" python "rsa/topography_control.py" \
    --preprocessed-dir "${DATA_BASE}/preprocessed/average_sub/raw" --fmri-suffix raw \
    --timing-csv "${DATA_BASE}/movie_timing.csv" \
    --embeddings-dir "${OUTPUTS_BASE}/model_embeddings" \
    --masks-dir "${OUTPUTS_BASE}/cf_modeling/masks" \
    --layers "$LAYERS" --true-run-dir "$TRUE_RUN_DIR" --output-dir "$OUTPUT_DIR" \
    --k "$K" --n-draws "$N_DRAWS" --n-perm "$N_PERM" --seed "$SEED" \
    --gpu-batch-size "$GPU_BATCH_SIZE" --perm-batch-size "$PERM_BATCH_SIZE"
