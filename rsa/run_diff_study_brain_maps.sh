#!/usr/bin/env bash
# rsa/run_diff_study_brain_maps.sh
# =========================================
# Runs rsa/searchlight.py (group-average, CPU -- these are small correlation
# jobs, not GPU model inference) for one model's whole-embedding intact +
# dummy-modality (clsav_from_a/v) conditions, so rsa/dummy_diff_maps.py has
# something to consolidate. Companion to the scramble diff-study, whose
# avscramble searchlight runs already exist on disk from earlier sessions.
#
# Usage:
#   bash rsa/run_diff_study_brain_maps.sh <model>
#   e.g. bash rsa/run_diff_study_brain_maps.sh pe-av-small-16-frame
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL="${1:?usage: run_diff_study_brain_maps.sh <model>}"

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
METHOD="spearman"
TR=1.0
GPU_BATCH_SIZE=512

# Intact (if not already run) + both dummy-modality conditions.
MODEL_NAMES=("$MODEL" "${MODEL}_clsav_from_a" "${MODEL}_clsav_from_v")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    COMBINED_OUT="${OUTPUT_DIR}/group_average/${MODEL_NAME}_av/rsa_59k_${FMRI_SUFFIX}_k${K}_delay${DELAY_SEC%.*}s_bin${BIN_SEC%.*}s_skip${SKIP_SEC%.*}s_${METHOD}_maps.dscalar.nii"
    if [ -f "$COMBINED_OUT" ]; then
        echo "=== searchlight: ${MODEL_NAME} / av -- already exists, skipping ==="
        continue
    fi
    EMB_PATH="${EMBEDDINGS_DIR}/${MODEL_NAME}/bin${BIN_SEC%.*}s_skip${SKIP_SEC%.*}s/${MODEL_NAME}_av.npy"
    if [ ! -f "$EMB_PATH" ]; then
        echo "=== searchlight: ${MODEL_NAME} / av -- SKIP, missing embedding ${EMB_PATH} ==="
        continue
    fi
    echo "=== searchlight: ${MODEL_NAME} / av ==="
    CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n movie python "${SCRIPT_DIR}/searchlight.py" \
        --preprocessed-dir   "$PREPROCESSED_DIR" \
        --fmri-suffix        "$FMRI_SUFFIX" \
        --timing-csv         "$TIMING_CSV" \
        --embeddings-dir     "$EMBEDDINGS_DIR" \
        --template-cifti     "$TEMPLATE_CIFTI" \
        --output-dir         "$OUTPUT_DIR" \
        --subject            "group_average" \
        --model              "$MODEL_NAME" \
        --modality           "av" \
        --k                  "$K" \
        --bin-sec            "$BIN_SEC" \
        --skip-sec           "$SKIP_SEC" \
        --delay-sec          "$DELAY_SEC" \
        --method             "$METHOD" \
        --tr                 "$TR" \
        --left-surface       "$LEFT_SURFACE" \
        --right-surface      "$RIGHT_SURFACE" \
        --workbench          "$WORKBENCH" \
        --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
        --combined-output    "$COMBINED_OUT" \
        --gpu-batch-size     "$GPU_BATCH_SIZE" \
        --n-blocks           1 \
        --normalize
done

echo "=== ALL DONE (${MODEL}) ==="
