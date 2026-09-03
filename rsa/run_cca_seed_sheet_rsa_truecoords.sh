#!/usr/bin/env bash
# rsa/run_cca_seed_sheet_rsa_truecoords.sh
# Runs rsa/cca_seed_sheet_rsa_truecoords.py: RSA between the two CCA seed
# ROIs and Topo-Omni's cortical sheet under TRUE (trained, permute_coordinates
# seed=42) coordinates rather than the raster fallback used by
# run_cca_seed_sheet_rsa.sh. See that script's module docstring for full
# provenance of the true-coordinate lookup table.
#
# Two modes selected by LAYERS:
#   LAYERS=1,9,18,27,34,35 (default) -- primary deliverable: all 6 genuinely-
#       unimodal decoder layers stacked into true absolute rows.
#   LAYERS=18 -- single-layer comparison, directly against the raster k100
#       run in outputs/rsa/cca_seed_sheet_rsa/k100/.
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

LAYERS="${LAYERS:-1,9,18,27,34,35}"
K="${K:-100}"
TRUE_COORDS_CACHE="${TRUE_COORDS_CACHE:-${OUTPUTS_BASE}/rsa/cca_seed_sheet_rsa/topoomni_true_coords_seed42.npy}"
# Output dir tag mirrors the layer set so single-layer and multi-layer runs
# never collide; default here is the multi-layer primary deliverable.
if [ "$LAYERS" = "18" ]; then
    DEFAULT_TAG="k100_layer18_truecoords"
else
    DEFAULT_TAG="k100_multilayer_truecoords"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUTS_BASE}/rsa/cca_seed_sheet_rsa/${DEFAULT_TAG}}"
BIN_SEC="${BIN_SEC:-5.0}"
SKIP_SEC="${SKIP_SEC:-5.0}"
DELAY_SEC="${DELAY_SEC:-5.0}"
TR="${TR:-1.0}"
METHOD="${METHOD:-spearman}"
N_PERM="${N_PERM:-1000}"
SEED="${SEED:-42}"
# Small batch defaults: this machine typically has another GPU job running
# concurrently (cf_modeling per-subject fits); batch=512 (the raster script's
# default) reliably OOMs under that contention, batch=64/perm_batch=20 does not
# and still finishes in ~1-4 min per seed x modality on GPU.
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-64}"
PERM_BATCH_SIZE="${PERM_BATCH_SIZE:-20}"

conda run --no-capture-output -n "$CONDA_ENV" python "rsa/cca_seed_sheet_rsa_truecoords.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" --fmri-suffix "$FMRI_SUFFIX" \
    --timing-csv "$TIMING_CSV" --embeddings-dir "$EMBEDDINGS_DIR" \
    --masks-dir "$MASKS_DIR" --output-dir "$OUTPUT_DIR" \
    --layers "$LAYERS" --true-coords-cache "$TRUE_COORDS_CACHE" --k "$K" \
    --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
    --method "$METHOD" --n-perm "$N_PERM" --seed "$SEED" \
    --gpu-batch-size "$GPU_BATCH_SIZE" --perm-batch-size "$PERM_BATCH_SIZE"
