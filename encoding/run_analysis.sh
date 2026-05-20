#!/usr/bin/env bash
# encoding/run_analysis.sh
# ==========================
# Master runner for ridge encoding model analyses.
# Supports group-average and per-subject modes with GNU parallel.
#
# Usage:
#   bash run_analysis.sh                    # group-avg
#   bash run_analysis.sh avg               # group-avg (explicit)
#   bash run_analysis.sh persubject        # per-subject, all subjects
#   bash run_analysis.sh persubject 4      # per-subject, 4 parallel jobs
#   bash run_analysis.sh persubject 8 100610  # resume from subject 100610
#   bash run_analysis.sh all               # group-avg + per-subject
#
# Prerequisites:
#   conda activate analysis
#   GNU parallel: conda install -c conda-forge parallel

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data/Setareh"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# fMRI data
FMRI_CIFTI_AVG="${OUTPUTS_BASE}/preprocessed/group_average_gsr_zscore_cortex_59k.dtseries.nii"
PREPROCESSED_DIR="${OUTPUTS_BASE}/preprocessed"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# Timing
TIMING_CSV="${DATA_BASE}/Data/movie_timing.csv"

# Embeddings root
# Convention: {EMBEDDINGS_DIR}/{model_name}/{bin_sec}s/{model_name}_{modality}.npy
# Any entry whose embedding file is absent is silently skipped.
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# CIFTI template (for output header — must match the fMRI grayordinate space)
# Use the preprocessed group-average dtseries (59k, 108441 cortical grayordinates).
TEMPLATE_CIFTI="${FMRI_CIFTI_AVG}"

# Output root
OUTPUT_DIR="${OUTPUTS_BASE}/encoding"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
BIN_SEC=2.0
DELAY_SEC=5.0
HRF=false       # true → SPM HRF convolution; false → boxcar delay
NORMALIZE=true  # per-run z-score normalization of embeddings

# Ridge regularisation search
ALPHA_MIN=-2
ALPHA_MAX=9
N_ALPHAS=23
CHUNK_SIZE=2000    # vertices per batch (memory/speed tradeoff)

TEST_VIDEO_IDS="video5,video9,video14,video18"

# ── Model registry ─────────────────────────────────────────────────────────
# Format: "model_name:modalities"
#   model_name  — must match the subdirectory under EMBEDDINGS_DIR
#   modalities  — comma-separated list of embedding modality suffixes
#
# Embedding path: {EMBEDDINGS_DIR}/{model_name}/{BIN_SEC}s/{model_name}_{modality}.npy
# Add a new model by appending a line; it runs whenever its embeddings exist.
MODELS=(
    "pe-av-small-16-frame:v,a,av"
    "pe-av-base:v,a,av"
    "pe-av-large:v,a,av"
    "cav-mae-sync:v,a,av"
    "audiomae:a"
    "videomaev2-large:v"
    "wavlm-large:a"
    "whisper-large-v3:a"
    "pe-core-l14:v"
)

# ── Parallelisation ─────────────────────────────────────────────────────────
CONDA_ENV="analysis"
DEFAULT_BATCH_SIZE=8
# =============================================================================

MODE=${1:-avg}
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
START_FROM=${3:-""}

BIN_SEC_INT="${BIN_SEC%.*}"

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() { conda run -n "$CONDA_ENV" python "$@"; }

_hrf_flag()       { [ "$HRF"       = "true" ] && echo "--hrf"       || echo ""; }
_normalize_flag() { [ "$NORMALIZE" = "true" ] && echo "--normalize" || echo ""; }

_emb_exists() {
    local MODEL_NAME="$1" MOD="$2"
    [ -f "${EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_SEC_INT}s/${MODEL_NAME}_${MOD}.npy" ]
}

# =============================================================================
# GROUP-AVERAGE PIPELINE
# =============================================================================
run_avg() {
    log "=== Group-average encoding (${#MODELS[@]} models) ==="

    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_STR <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_STR"

        for MOD in "${MODS[@]}"; do
            if ! _emb_exists "$MODEL_NAME" "$MOD"; then
                log "  SKIP ${MODEL_NAME}/${MOD} — embedding not found"
                continue
            fi
            log "  ${MODEL_NAME} / ${MOD}"

            run_python "${SCRIPT_DIR}/run_encoding.py" \
                --fmri-cifti     "$FMRI_CIFTI_AVG" \
                --timing-csv     "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --output-dir     "$OUTPUT_DIR" \
                --subject        "group_average" \
                --model          "$MODEL_NAME" \
                --modality       "$MOD" \
                --bin-sec        "$BIN_SEC" \
                --delay-sec      "$DELAY_SEC" \
                --tr             "$TR" \
                --alpha-min      "$ALPHA_MIN" \
                --alpha-max      "$ALPHA_MAX" \
                --n-alphas       "$N_ALPHAS" \
                --chunk-size     "$CHUNK_SIZE" \
                --test-video-ids "$TEST_VIDEO_IDS" \
                $(_hrf_flag) $(_normalize_flag)
        done
    done

    log "=== Group-average encoding done ==="
}

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================

# Worker function — exported for GNU parallel
_run_one_subject() {
    local SUB="$1"
    local SCRIPT_DIR="$2"
    local CONDA_ENV="$3"
    local PREPROCESSED_DIR="$4"
    local OUTPUT_DIR="$5"
    local TIMING_CSV="$6"
    local EMBEDDINGS_DIR="$7"
    local TEMPLATE_CIFTI="$8"
    local MODELS_STR="$9"       # semicolon-separated "model_name:modalities" entries
    local BIN_SEC="${10}"
    local DELAY_SEC="${11}"
    local TR="${12}"
    local ALPHA_MIN="${13}"
    local ALPHA_MAX="${14}"
    local N_ALPHAS="${15}"
    local CHUNK_SIZE="${16}"
    local TEST_VIDEO_IDS="${17}"
    local HRF="${18}"
    local NORMALIZE="${19}"

    local BIN_SEC_INT="${BIN_SEC%.*}"
    local LOG_DIR="${OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/pipeline.log"

    local FMRI_PATH="${PREPROCESSED_DIR}/${SUB}_gsr_zscore_cortex_59k.dtseries.nii"
    if [ ! -f "$FMRI_PATH" ]; then
        echo "[$(date +%H:%M:%S)] ${SUB}: no preprocessed CIFTI — run preprocess_individual.py first" \
            | tee -a "$LOG"
        return 1
    fi

    local HRF_FLAG="";  [ "$HRF"       = "true" ] && HRF_FLAG="--hrf"
    local NORM_FLAG=""; [ "$NORMALIZE" = "true" ] && NORM_FLAG="--normalize"

    echo "[$(date +%H:%M:%S)] Starting ${SUB}" | tee -a "$LOG"

    IFS=';' read -ra MODEL_ENTRIES <<< "$MODELS_STR"
    for MODEL_ENTRY in "${MODEL_ENTRIES[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"

        for MOD in "${MODS[@]}"; do
            local EMB="${EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_SEC_INT}s/${MODEL_NAME}_${MOD}.npy"
            if [ ! -f "$EMB" ]; then
                echo "[$(date +%H:%M:%S)] ${SUB}: SKIP ${MODEL_NAME}/${MOD} — no embedding" \
                    | tee -a "$LOG"
                continue
            fi

            conda run -n "$CONDA_ENV" python \
                "${SCRIPT_DIR}/run_encoding.py" \
                --fmri-cifti     "$FMRI_PATH" \
                --timing-csv     "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --output-dir     "$OUTPUT_DIR" \
                --subject        "$SUB" \
                --model          "$MODEL_NAME" \
                --modality       "$MOD" \
                --bin-sec        "$BIN_SEC" \
                --delay-sec      "$DELAY_SEC" \
                --tr             "$TR" \
                --alpha-min      "$ALPHA_MIN" \
                --alpha-max      "$ALPHA_MAX" \
                --n-alphas       "$N_ALPHAS" \
                --chunk-size     "$CHUNK_SIZE" \
                --test-video-ids "$TEST_VIDEO_IDS" \
                $HRF_FLAG $NORM_FLAG \
                >> "$LOG" 2>&1
        done
    done

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} DONE" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB} FAILED (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_subject

run_persubject() {
    log "=== Per-subject encoding (${BATCH_SIZE} parallel jobs, ${#MODELS[@]} models) ==="

    local SUBJECTS
    SUBJECTS=$(cat "$SUBJECTS_LIST")
    if [ -n "$START_FROM" ]; then
        SUBJECTS=$(echo "$SUBJECTS" | awk "/$START_FROM/{found=1} found{print}")
    fi
    local N_TOTAL
    N_TOTAL=$(echo "$SUBJECTS" | wc -w)
    log "  Processing ${N_TOTAL} subjects ..."

    local MODELS_STR
    MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_subject {} \
            "$SCRIPT_DIR" "$CONDA_ENV" "$PREPROCESSED_DIR" "$OUTPUT_DIR" \
            "$TIMING_CSV" "$EMBEDDINGS_DIR" "$TEMPLATE_CIFTI" \
            "$MODELS_STR" "$BIN_SEC" "$DELAY_SEC" "$TR" \
            "$ALPHA_MIN" "$ALPHA_MAX" "$N_ALPHAS" "$CHUNK_SIZE" \
            "$TEST_VIDEO_IDS" "$HRF" "$NORMALIZE"
    else
        log "GNU parallel not found — running sequentially"
        log "  (install with: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB" \
                "$SCRIPT_DIR" "$CONDA_ENV" "$PREPROCESSED_DIR" "$OUTPUT_DIR" \
                "$TIMING_CSV" "$EMBEDDINGS_DIR" "$TEMPLATE_CIFTI" \
                "$MODELS_STR" "$BIN_SEC" "$DELAY_SEC" "$TR" \
                "$ALPHA_MIN" "$ALPHA_MAX" "$N_ALPHAS" "$CHUNK_SIZE" \
                "$TEST_VIDEO_IDS" "$HRF" "$NORMALIZE"
        done
    fi

    log "=== Per-subject encoding complete ==="
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    avg)        run_avg ;;
    persubject) run_persubject ;;
    all)        run_avg; run_persubject ;;
    *)
        echo "Unknown mode: $MODE. Use: avg | persubject | all" >&2; exit 1 ;;
esac

log "All encoding analyses complete."
