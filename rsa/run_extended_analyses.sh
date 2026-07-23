#!/usr/bin/env bash
# rsa/run_extended_analyses.sh
# ===============================
# Runner for the "new analyses" pseudo-models that generalize the AV-
# integration story beyond plain RSA: _av_linear_resid (ridge residual vs.
# AudioMAE+VideoMAEv2 unimodal baseline, all NATIVE_AV_MODELS), and the
# own-unimodal geometric decompositions from
# compute_projection_residual_embeddings.py (_av_projection_resid,
# _av_linear_resid_encoder). Group-average only -- these pseudo-models are
# embedding-space transforms, not new subjects, and per explicit instruction
# this sweep is not run per-subject.
#
# Stages
#   embed     Generate the pseudo-model .npy embeddings on disk (CPU-only
#             ridge/projection math -- safe to run alongside a GPU
#             extraction job). Idempotent; skips any model whose native-AV
#             embedding isn't extracted yet.
#   plain     Group-average RSA searchlight (rsa/analysis.sh avg via
#             RSA_MODELS_OVERRIDE) for every pseudo-model found on disk.
#   partial   partial_rsa.py --run partial_corr_{model} for every
#             NATIVE_AV_MODELS entry (group-average only).
#   all       embed + plain + partial, in order.
#
# Usage
#   bash rsa/run_extended_analyses.sh [STAGE]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

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
RUN_TRS="${PREPROCESSED_DIR}/group_average_raw_run_trs.npy"
CONDA_ENV="movie"

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
METHOD="spearman"
TR=1.0
GPU_BATCH_SIZE=512

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

STAGE=${1:-all}

run_embed() {
    log "=== Extended analyses: generating _av_linear_resid embeddings ==="
    run_python "notebooks/feature_extraction/compute_linear_residual_embeddings.py" \
        --embeddings-dir "$EMBEDDINGS_DIR" --timing-csv "$TIMING_CSV" --run-trs "$RUN_TRS" \
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR"
    log "=== Extended analyses: generating _av_projection_resid / _av_linear_resid_encoder embeddings ==="
    run_python "notebooks/feature_extraction/compute_projection_residual_embeddings.py" \
        --embeddings-dir "$EMBEDDINGS_DIR" --timing-csv "$TIMING_CSV" --run-trs "$RUN_TRS" \
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR"
    log "=== Extended analyses embed stage complete ==="
}

# Discover every pseudo-model actually generated on disk -- avoids
# hand-maintaining a second copy of the two Python scripts' model lists.
_discover_models_str() {
    local MODELS_STR="" d name
    for d in "$EMBEDDINGS_DIR"/*_av_linear_resid "$EMBEDDINGS_DIR"/*_av_projection_resid "$EMBEDDINGS_DIR"/*_av_linear_resid_encoder; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        MODELS_STR="${MODELS_STR}${name}:av;"
    done
    echo "${MODELS_STR%;}"
}

run_plain() {
    local MODELS_STR
    MODELS_STR=$(_discover_models_str)
    if [ -z "$MODELS_STR" ]; then
        log "No pseudo-model embeddings found on disk -- run the 'embed' stage first."
        return 1
    fi
    log "=== Extended analyses PLAIN RSA (group-average): $(echo "$MODELS_STR" | tr ';' '\n' | wc -l) pseudo-models ==="
    BIN_SECS="$BIN_SEC" RSA_MODELS_OVERRIDE="$MODELS_STR" \
        bash "${SCRIPT_DIR}/analysis.sh" avg
    log "=== Extended analyses plain RSA complete ==="
}

run_partial() {
    log "=== Extended analyses PARTIAL RSA (partial_corr_*, group-average) ==="
    local MODELS RUN_KEY
    mapfile -t MODELS < <(run_python -c "
import sys; sys.path.insert(0, '.')
from rsa.shared.model_registry import NATIVE_AV_MODELS
print('\n'.join(NATIVE_AV_MODELS))
")
    for m in "${MODELS[@]}"; do
        RUN_KEY="partial_corr_${m}"
        log "  ${RUN_KEY}"
        run_python "${SCRIPT_DIR}/partial_rsa.py" \
            --run "$RUN_KEY" \
            --preprocessed-dir "$PREPROCESSED_DIR" \
            --fmri-suffix "$FMRI_SUFFIX" \
            --timing-csv "$TIMING_CSV" \
            --embeddings-dir "$EMBEDDINGS_DIR" \
            --template-cifti "$TEMPLATE_CIFTI" \
            --output-dir "$OUTPUT_DIR" \
            --subject group_average \
            --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
            --k "$K" --method "$METHOD" \
            --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
            --workbench "$WORKBENCH" \
            --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
            --gpu-batch-size "$GPU_BATCH_SIZE" \
            || log "  SKIP ${RUN_KEY}: partial_rsa.py failed (embeddings missing?)"
    done
    log "=== Extended analyses partial RSA complete ==="
}

case "$STAGE" in
    embed)   run_embed ;;
    plain)   run_plain ;;
    partial) run_partial ;;
    all)     run_embed; run_plain; run_partial ;;
    *) echo "Unknown STAGE: $STAGE (use: embed | plain | partial | all)"; exit 1 ;;
esac
log "Extended analyses (${STAGE}) complete."
