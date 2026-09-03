#!/usr/bin/env bash
# rsa/run_cca_seed_sheet_rsa.sh
# Runs rsa/cca_seed_sheet_rsa.py: RSA between the two CCA seed ROIs (audio-
# preferring cca_a / video-preferring cca_p) and Topo-Omni's layer-18
# cortical-sheet units, plotted in the sheet's true 2D layout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
MASKS_DIR="${OUTPUTS_BASE}/cf_modeling/masks"
CONDA_ENV="movie"

SHEET_MODEL="${SHEET_MODEL:-topoomni_layer18_sheet_mp}"
K="${K:-50}"
# k-tagged output dir keeps k-sensitivity checkable: the original k=50 run
# lives at outputs/rsa/cca_seed_sheet_rsa/ (unsuffixed, untouched); every run
# through this script (any K, including 50) writes to a k${K}/ subdir so
# re-running never silently overwrites a prior K's results.
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_BASE}/rsa/cca_seed_sheet_rsa/k${K}}"
BIN_SEC="${BIN_SEC:-5.0}"
SKIP_SEC="${SKIP_SEC:-5.0}"
DELAY_SEC="${DELAY_SEC:-5.0}"
TR="${TR:-1.0}"
METHOD="${METHOD:-spearman}"
N_PERM="${N_PERM:-1000}"
SEED="${SEED:-42}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-512}"
PERM_BATCH_SIZE="${PERM_BATCH_SIZE:-100}"

conda run --no-capture-output -n "$CONDA_ENV" python "rsa/cca_seed_sheet_rsa.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" --fmri-suffix "$FMRI_SUFFIX" \
    --timing-csv "$TIMING_CSV" --embeddings-dir "$EMBEDDINGS_DIR" \
    --masks-dir "$MASKS_DIR" --output-dir "$OUTPUT_DIR" \
    --sheet-model "$SHEET_MODEL" --k "$K" \
    --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
    --method "$METHOD" --n-perm "$N_PERM" --seed "$SEED" \
    --gpu-batch-size "$GPU_BATCH_SIZE" --perm-batch-size "$PERM_BATCH_SIZE"
