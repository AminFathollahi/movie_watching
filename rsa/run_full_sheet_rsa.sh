#!/usr/bin/env bash
# rsa/run_full_sheet_rsa.sh
# Runs rsa/full_sheet_rsa.py: RSA between the two CCA seed ROIs and
# Topo-Omni's COMPLETE 304x512 (155,648-unit) cortical sheet (encoder +
# all 36 decoder layers) under TRUE (permute_coordinates seed=42)
# coordinates. See that script's module docstring for full methodology.
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

K="${K:-100}"
TRUE_COORDS_CACHE="${TRUE_COORDS_CACHE:-${OUTPUTS_BASE}/rsa/cca_seed_sheet_rsa/topoomni_true_coords_seed42.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_BASE}/rsa/cca_seed_sheet_rsa/fullsheet_k100_truecoords}"
BIN_SEC="${BIN_SEC:-5.0}"
SKIP_SEC="${SKIP_SEC:-5.0}"
DELAY_SEC="${DELAY_SEC:-5.0}"
TR="${TR:-1.0}"
METHOD="${METHOD:-spearman}"
N_PERM="${N_PERM:-1000}"
SEED="${SEED:-42}"
# 155,648 units is 12.7x the 12,288-unit multilayer truecoords run. Per-batch
# GPU memory there is dominated by (vertex_batch_size x n_pairs) intermediates
# -- independent of total unit count -- so this batch size is not shrunk for
# memory; it is kept at the same conservative value the truecoords run already
# validated under GPU contention (see rsa/run_cca_seed_sheet_rsa_truecoords.sh),
# now with the WHOLE card available (CF killed by the calling chain). The
# 12.7x unit-count increase instead means ~12.7x more loop iterations (wall
# time), not more peak memory.
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-64}"
PERM_BATCH_SIZE="${PERM_BATCH_SIZE:-20}"

conda run --no-capture-output -n "$CONDA_ENV" python "rsa/full_sheet_rsa.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" --fmri-suffix "$FMRI_SUFFIX" \
    --timing-csv "$TIMING_CSV" --embeddings-dir "$EMBEDDINGS_DIR" \
    --masks-dir "$MASKS_DIR" --output-dir "$OUTPUT_DIR" \
    --true-coords-cache "$TRUE_COORDS_CACHE" --k "$K" \
    --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
    --method "$METHOD" --n-perm "$N_PERM" --seed "$SEED" \
    --gpu-batch-size "$GPU_BATCH_SIZE" --perm-batch-size "$PERM_BATCH_SIZE"
