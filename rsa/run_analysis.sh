#!/usr/bin/env bash
# rsa/run_analysis.sh
# ====================
# Master runner for searchlight and Glasser parcel RSA analyses.
# Supports group-average and per-subject modes with GNU parallel.
#
# Usage:
#   bash run_analysis.sh                         # group-avg, all methods
#   bash run_analysis.sh avg                     # group-avg, all methods
#   bash run_analysis.sh avg searchlight         # group-avg, searchlight only
#   bash run_analysis.sh avg glasser             # group-avg, Glasser only
#   bash run_analysis.sh persubject              # per-subject, all methods
#   bash run_analysis.sh persubject all 4        # per-subject, 4 parallel jobs
#   bash run_analysis.sh persubject all 8 100610 # resume from subject 100610
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

# Embeddings root  (model subdirs live here)
# Convention: {EMBEDDINGS_DIR}/{model_name}/{bin_sec}s/{model_name}_{modality}.npy
# A model/modality combo is silently skipped if its embedding file is absent.
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# CIFTI template (for output header — must match the fMRI grayordinate space)
# Use the preprocessed group-average dtseries; its BrainModelAxis defines the
# 59k cortex structure (54216 LH + 54225 RH = 108441 grayordinates).
TEMPLATE_CIFTI="${FMRI_CIFTI_AVG}"

# Surface geometry (59k midthickness — correct for geodesic distances;
# the fMRI CIFTI vertex indices are in the range [0, 59291])
LEFT_SURFACE="${HCP_DIR}/S1200.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/S1200.R.midthickness_MSMAll.59k_fs_LR.surf.gii"

# Sphere surfaces for resampling (used by downstream tools / wb_view)
SPHERE_59K_LH="${HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
SPHERE_59K_RH="${HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"

# Glasser parcellation (59k dense-label CIFTI — covers all 59k vertices
# including medial wall; run_glasser.py maps to fMRI grayordinates by vertex index)
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"

# Connectome Workbench
WORKBENCH="/opt/workbench/bin_linux64/wb_command"

# Output root
OUTPUT_DIR="${OUTPUTS_BASE}/rsa"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
BIN_SEC=2.0
DELAY_SEC=5.0
HRF=false          # true → SPM HRF convolution; false → boxcar delay
NORMALIZE=true     # per-run z-score normalization of embeddings
BLOCKDIAG=false    # block-diagonal (per-segment) normalization
METHOD="spearman"
K=100              # searchlight neighbourhood size

# ── Model registry ─────────────────────────────────────────────────────────
# Format: "model_name:modalities"
#   model_name  — must match the subdirectory name under EMBEDDINGS_DIR
#   modalities  — comma-separated list of embedding modality suffixes
#
# Embedding path: {EMBEDDINGS_DIR}/{model_name}/{BIN_SEC}s/{model_name}_{modality}.npy
# Any entry whose embedding file does not exist is skipped with a warning.
# To add a new model: append a line and ensure its embeddings follow the convention.
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

MODE=${1:-avg}        # avg | persubject | all
METHOD_ARG=${2:-all}  # all | searchlight | glasser
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
START_FROM=${4:-""}   # resume from this subject ID (per-subject mode)

BIN_SEC_INT="${BIN_SEC%.*}"   # e.g. 2.0 → 2

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() { conda run -n "$CONDA_ENV" python "$@"; }

_hrf_flag()       { [ "$HRF"       = "true" ] && echo "--hrf"       || echo ""; }
_normalize_flag() { [ "$NORMALIZE" = "true" ] && echo "--normalize" || echo ""; }
_blockdiag_flag() { [ "$BLOCKDIAG" = "true" ] && echo "--blockdiag" || echo ""; }

# Check whether the embedding file for a given model/modality exists
_emb_exists() {
    local MODEL_NAME="$1" MOD="$2"
    local EMB="${EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_SEC_INT}s/${MODEL_NAME}_${MOD}.npy"
    [ -f "$EMB" ]
}

# =============================================================================
# GROUP-AVERAGE PIPELINE
# =============================================================================
_run_avg_one_model() {
    local MODEL_NAME="$1" MODALITIES_STR="$2"
    IFS=',' read -ra MODS <<< "$MODALITIES_STR"

    for MOD in "${MODS[@]}"; do
        if ! _emb_exists "$MODEL_NAME" "$MOD"; then
            log "  SKIP ${MODEL_NAME}/${MOD} — embedding not found"
            continue
        fi
        log "  ${MODEL_NAME} / ${MOD}"

        if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "searchlight" ]; then
            run_python "${SCRIPT_DIR}/searchlight/run_searchlight.py" \
                --fmri-cifti      "$FMRI_CIFTI_AVG" \
                --timing-csv      "$TIMING_CSV" \
                --embeddings-dir  "$EMBEDDINGS_DIR" \
                --template-cifti  "$TEMPLATE_CIFTI" \
                --output-dir      "$OUTPUT_DIR" \
                --subject         "group_average" \
                --model           "$MODEL_NAME" \
                --modality        "$MOD" \
                --k               "$K" \
                --bin-sec         "$BIN_SEC" \
                --delay-sec       "$DELAY_SEC" \
                --method          "$METHOD" \
                --tr              "$TR" \
                --left-surface    "$LEFT_SURFACE" \
                --right-surface   "$RIGHT_SURFACE" \
                --workbench       "$WORKBENCH" \
                $(_hrf_flag) $(_normalize_flag) $(_blockdiag_flag)
        fi

        if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "glasser" ]; then
            run_python "${SCRIPT_DIR}/glasser/run_glasser.py" \
                --fmri-cifti      "$FMRI_CIFTI_AVG" \
                --timing-csv      "$TIMING_CSV" \
                --embeddings-dir  "$EMBEDDINGS_DIR" \
                --template-cifti  "$TEMPLATE_CIFTI" \
                --output-dir      "$OUTPUT_DIR" \
                --subject         "group_average" \
                --model           "$MODEL_NAME" \
                --modality        "$MOD" \
                --bin-sec         "$BIN_SEC" \
                --delay-sec       "$DELAY_SEC" \
                --method          "$METHOD" \
                --tr              "$TR" \
                --glasser-dlabel  "$GLASSER_DLABEL" \
                $(_hrf_flag) $(_normalize_flag) $(_blockdiag_flag)
        fi
    done
}

run_avg() {
    log "=== Group-average RSA (${#MODELS[@]} models, method=${METHOD_ARG}) ==="
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_STR <<< "$MODEL_ENTRY"
        _run_avg_one_model "$MODEL_NAME" "$MODALITIES_STR"
    done
    log "=== Group-average RSA done ==="
}

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================

# Worker function — exported for GNU parallel
# Args: SUB SCRIPT_DIR CONDA_ENV PREPROCESSED_DIR OUTPUT_DIR
#       TIMING_CSV EMBEDDINGS_DIR TEMPLATE_CIFTI
#       LEFT_SURF RIGHT_SURF GLASSER_DLABEL WORKBENCH
#       MODELS_STR BIN_SEC DELAY_SEC TR K METHOD
#       HRF NORMALIZE BLOCKDIAG METHOD_ARG
_run_one_subject() {
    local SUB="$1"
    local SCRIPT_DIR="$2"
    local CONDA_ENV="$3"
    local PREPROCESSED_DIR="$4"
    local OUTPUT_DIR="$5"
    local TIMING_CSV="$6"
    local EMBEDDINGS_DIR="$7"
    local TEMPLATE_CIFTI="$8"
    local LEFT_SURF="$9"
    local RIGHT_SURF="${10}"
    local GLASSER_DLABEL="${11}"
    local WORKBENCH="${12}"
    local MODELS_STR="${13}"   # semicolon-separated "model_name:modalities" entries
    local BIN_SEC="${14}"
    local DELAY_SEC="${15}"
    local TR="${16}"
    local K="${17}"
    local METHOD="${18}"
    local HRF="${19}"
    local NORMALIZE="${20}"
    local BLOCKDIAG="${21}"
    local METHOD_ARG="${22}"

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
    local BD_FLAG="";   [ "$BLOCKDIAG" = "true" ] && BD_FLAG="--blockdiag"

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

            if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "searchlight" ]; then
                conda run -n "$CONDA_ENV" python \
                    "${SCRIPT_DIR}/searchlight/run_searchlight.py" \
                    --fmri-cifti    "$FMRI_PATH" \
                    --timing-csv    "$TIMING_CSV" \
                    --embeddings-dir "$EMBEDDINGS_DIR" \
                    --template-cifti "$TEMPLATE_CIFTI" \
                    --output-dir    "$OUTPUT_DIR" \
                    --subject       "$SUB" \
                    --model         "$MODEL_NAME" \
                    --modality      "$MOD" \
                    --k             "$K" \
                    --bin-sec       "$BIN_SEC" \
                    --delay-sec     "$DELAY_SEC" \
                    --method        "$METHOD" \
                    --tr            "$TR" \
                    --left-surface  "$LEFT_SURF" \
                    --right-surface "$RIGHT_SURF" \
                    --workbench     "$WORKBENCH" \
                    $HRF_FLAG $NORM_FLAG $BD_FLAG \
                    >> "$LOG" 2>&1
            fi

            if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "glasser" ]; then
                conda run -n "$CONDA_ENV" python \
                    "${SCRIPT_DIR}/glasser/run_glasser.py" \
                    --fmri-cifti    "$FMRI_PATH" \
                    --timing-csv    "$TIMING_CSV" \
                    --embeddings-dir "$EMBEDDINGS_DIR" \
                    --template-cifti "$TEMPLATE_CIFTI" \
                    --output-dir    "$OUTPUT_DIR" \
                    --subject       "$SUB" \
                    --model         "$MODEL_NAME" \
                    --modality      "$MOD" \
                    --bin-sec       "$BIN_SEC" \
                    --delay-sec     "$DELAY_SEC" \
                    --method        "$METHOD" \
                    --tr            "$TR" \
                    --glasser-dlabel "$GLASSER_DLABEL" \
                    $HRF_FLAG $NORM_FLAG $BD_FLAG \
                    >> "$LOG" 2>&1
            fi
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
    log "=== Per-subject RSA (${BATCH_SIZE} parallel jobs, ${#MODELS[@]} models) ==="

    local SUBJECTS
    SUBJECTS=$(cat "$SUBJECTS_LIST")
    if [ -n "$START_FROM" ]; then
        SUBJECTS=$(echo "$SUBJECTS" | awk "/$START_FROM/{found=1} found{print}")
    fi
    local N_TOTAL
    N_TOTAL=$(echo "$SUBJECTS" | wc -w)
    log "  Processing ${N_TOTAL} subjects ..."

    # Flatten MODELS array into a semicolon-separated string for export
    local MODELS_STR
    MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_subject {} \
            "$SCRIPT_DIR" "$CONDA_ENV" "$PREPROCESSED_DIR" "$OUTPUT_DIR" \
            "$TIMING_CSV" "$EMBEDDINGS_DIR" "$TEMPLATE_CIFTI" \
            "$LEFT_SURFACE" "$RIGHT_SURFACE" "$GLASSER_DLABEL" \
            "$WORKBENCH" "$MODELS_STR" "$BIN_SEC" "$DELAY_SEC" "$TR" \
            "$K" "$METHOD" "$HRF" "$NORMALIZE" "$BLOCKDIAG" "$METHOD_ARG"
    else
        log "GNU parallel not found — running sequentially"
        log "  (install with: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB" \
                "$SCRIPT_DIR" "$CONDA_ENV" "$PREPROCESSED_DIR" "$OUTPUT_DIR" \
                "$TIMING_CSV" "$EMBEDDINGS_DIR" "$TEMPLATE_CIFTI" \
                "$LEFT_SURFACE" "$RIGHT_SURFACE" "$GLASSER_DLABEL" \
                "$WORKBENCH" "$MODELS_STR" "$BIN_SEC" "$DELAY_SEC" "$TR" \
                "$K" "$METHOD" "$HRF" "$NORMALIZE" "$BLOCKDIAG" "$METHOD_ARG"
        done
    fi

    log "=== Per-subject RSA complete ==="
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

log "All RSA analyses complete."
