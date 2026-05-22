#!/usr/bin/env bash
# rsa/run_analysis.sh
# ====================
# Master runner for searchlight and Glasser parcel RSA analyses.
# Supports group-average and per-subject modes with GNU parallel.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# Raw 7T CIFTI files (used in streaming mode — preprocess on-the-fly)
CIFTI_DIR="/home/amin/Research/Representation/Movie/data/individual-59k"

# ── Streaming toggle ──────────────────────────────────────────────────────────
STREAM=true

# Preprocessing flags (Continuous Signal Cleaning Only)
SG_FILTER=true     # Removes scanner drift
PSC=true           # Normalizes baseline, preserves intrinsic variance
GSR=true           # Regresses global signal

# Automatically build PREPROCESSING_FLAG based on the toggles above
PREP_PARTS=()
[ "$SG_FILTER" = "true" ] && PREP_PARTS+=("sg")
[ "$PSC" = "true" ]       && PREP_PARTS+=("psc")
[ "$GSR" = "true" ]       && PREP_PARTS+=("gsr")

if [ ${#PREP_PARTS[@]} -eq 0 ]; then
    PREPROCESSING_FLAG="raw"
else
    PREPROCESSING_FLAG=$(IFS=_; echo "${PREP_PARTS[*]}")
fi

# fMRI data — disk mode inputs (pointing to continuous maps)
DELAY_SEC=5.0
FMRI_SUFFIX="${PREPROCESSING_FLAG}"
PREPROCESSED_DIR="${DATA_BASE}/average_sub/${PREPROCESSING_FLAG}"
FMRI_CIFTI_AVG="${PREPROCESSED_DIR}/group_average_${FMRI_SUFFIX}_cortex_59k.dtseries.nii"

SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# Timing
TIMING_CSV="/home/amin/Research/Representation/Movie/data/HCP Data/movie_timing.csv"

# Embeddings root
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# CIFTI template (59k grayordinate space)
# NOTE: template is only used for its BrainModelAxis (grayordinate structure).
# It does NOT need to match the per-subject preprocessing — the sg_psc group
# average is valid for all streaming runs regardless of per-subject preprocessing.
TEMPLATE_CIFTI="${DATA_BASE}/average_sub/sg_psc/group_average_sg_psc_cortex_59k.dtseries.nii"

# Surface geometry — group-average (59k midthickness)
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"

# Per-subject midthickness surfaces
MIDTHICKNESS_DIR="${DATA_BASE}/midthickness_1.6"

# Glasser parcellation (59k)
# NOTE: Ensure you are using the 59k version to match the template and surfaces.
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"

# Connectome Workbench
WORKBENCH="/opt/workbench/bin_linux64/wb_command"

# Output root
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/${PREPROCESSING_FLAG}"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
BIN_SEC=2.0
HRF=false
METHOD="spearman"
K=150 # Update this once evaluate_reliability.py finds the optimal k

# ── Model registry ─────────────────────────────────────────────────────────
MODELS=(
    "pe-av-small-16-frame:av"
    # "pe-av-small-16-frame:v,a,av"
    # "audiomae:a"
    # "videomaev2-large:v"
    # "wavlm-large:a"
    # "whisper-large-v3:a"
    # "pe-core-l14:v"
)

# ── Parallelisation ─────────────────────────────────────────────────────────
CONDA_ENV="analysis"
DEFAULT_BATCH_SIZE=8
# =============================================================================

MODE=${1:-avg}
METHOD_ARG=${2:-all}
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
START_FROM=${4:-""}

BIN_SEC_INT="${BIN_SEC%.*}"

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() { conda run -n "$CONDA_ENV" python "$@"; }

_hrf_flag() { [ "$HRF" = "true" ] && echo "--hrf" || echo ""; }

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
            run_python "${SCRIPT_DIR}/run_searchlight.py" \
                --preprocessed-dir "$PREPROCESSED_DIR" \
                --fmri-suffix      "$FMRI_SUFFIX" \
                --timing-csv       "$TIMING_CSV" \
                --embeddings-dir   "$EMBEDDINGS_DIR" \
                --template-cifti   "$TEMPLATE_CIFTI" \
                --output-dir       "$OUTPUT_DIR" \
                --subject          "group_average" \
                --model            "$MODEL_NAME" \
                --modality         "$MOD" \
                --k                "$K" \
                --bin-sec          "$BIN_SEC" \
                --delay-sec        "$DELAY_SEC" \
                --method           "$METHOD" \
                --tr               "$TR" \
                --left-surface     "$LEFT_SURFACE" \
                --right-surface    "$RIGHT_SURFACE" \
                --workbench        "$WORKBENCH" \
                $(_hrf_flag)
        fi

        if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "glasser" ]; then
            run_python "${SCRIPT_DIR}/run_glasser.py" \
                --preprocessed-dir "$PREPROCESSED_DIR" \
                --fmri-suffix      "$FMRI_SUFFIX" \
                --timing-csv       "$TIMING_CSV" \
                --embeddings-dir   "$EMBEDDINGS_DIR" \
                --template-cifti   "$TEMPLATE_CIFTI" \
                --output-dir       "$OUTPUT_DIR" \
                --subject          "group_average" \
                --model            "$MODEL_NAME" \
                --modality         "$MOD" \
                --bin-sec          "$BIN_SEC" \
                --delay-sec        "$DELAY_SEC" \
                --method           "$METHOD" \
                --tr               "$TR" \
                --glasser-dlabel   "$GLASSER_DLABEL" \
                $(_hrf_flag)
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

# Worker function — exported for GNU parallel.
_run_one_subject() {
    local SUB="$1"
    local SCRIPT_DIR="$2"
    local CONDA_ENV="$3"
    local PREPROCESSED_DIR="$4"
    local FMRI_SUFFIX="$5"
    local OUTPUT_DIR="$6"
    local TIMING_CSV="$7"
    local EMBEDDINGS_DIR="$8"
    local TEMPLATE_CIFTI="$9"
    local LEFT_SURF_DEFAULT="${10}"
    local RIGHT_SURF_DEFAULT="${11}"
    local GLASSER_DLABEL="${12}"
    local WORKBENCH="${13}"
    local MODELS_STR="${14}"
    local BIN_SEC="${15}"
    local DELAY_SEC="${16}"
    local TR="${17}"
    local K="${18}"
    local METHOD="${19}"
    local HRF="${20}"
    local METHOD_ARG="${21}"
    local STREAM="${22}"
    local CIFTI_DIR="${23}"
    local SG_FILTER="${24}"
    local PSC="${25}"
    local GSR="${26}"
    local MIDTHICKNESS_DIR="${27}"

    local BIN_SEC_INT="${BIN_SEC%.*}"
    local LOG_DIR="${OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/pipeline.log"

    local HRF_FLAG=""; [ "$HRF" = "true" ] && HRF_FLAG="--hrf"

    # ── Resume: derive fmri_tag + config strings (mirrors Python naming) ──────
    local FMRI_TAG_PARTS=()
    [ "$SG_FILTER" = "true" ] && FMRI_TAG_PARTS+=("sg")
    [ "$PSC"       = "true" ] && FMRI_TAG_PARTS+=("psc")
    [ "$GSR"       = "true" ] && FMRI_TAG_PARTS+=("gsr")
    local FMRI_TAG_LOCAL
    if [ ${#FMRI_TAG_PARTS[@]} -eq 0 ]; then
        FMRI_TAG_LOCAL="raw"
    else
        FMRI_TAG_LOCAL=$(IFS=_; echo "${FMRI_TAG_PARTS[*]}")
    fi
    # In disk mode the tag comes from the pre-built suffix, not the flag combo
    [ "$STREAM" = "false" ] && FMRI_TAG_LOCAL="$FMRI_SUFFIX"

    local DELAY_INT="${DELAY_SEC%.*}"
    # Config directory name (searchlight includes k; Glasser does not)
    local SL_CONFIG="k${K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}s_${METHOD}"
    local GL_CONFIG="delay${DELAY_INT}s_bin${BIN_SEC_INT}s_${METHOD}"

    # Per-subject midthickness surfaces (fall back to group-average if not found)
    local LEFT_SURF="$LEFT_SURF_DEFAULT"
    local RIGHT_SURF="$RIGHT_SURF_DEFAULT"
    local SUB_L="${MIDTHICKNESS_DIR}/${SUB}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    local SUB_R="${MIDTHICKNESS_DIR}/${SUB}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    if [ -f "$SUB_L" ] && [ -f "$SUB_R" ]; then
        LEFT_SURF="$SUB_L"
        RIGHT_SURF="$SUB_R"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: subject midthickness not found, using group-average surface" \
            | tee -a "$LOG"
    fi

    # Build fMRI input flags based on mode
    local FMRI_FLAGS
    if [ "$STREAM" = "true" ]; then
        local SG_FLAG="";  [ "$SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        local PSC_FLAG=""; [ "$PSC"       = "true" ] && PSC_FLAG="--psc"
        local GSR_FLAG="--gsr";     [ "$GSR"     = "false" ] && GSR_FLAG="--no-gsr"
        # No Z-score flag here, it's structurally enforced in rsa_utils.py
        FMRI_FLAGS="--raw-dir ${CIFTI_DIR} $SG_FLAG $PSC_FLAG $GSR_FLAG"
    else
        local FMRI_PATH="${PREPROCESSED_DIR}/${SUB}_${FMRI_SUFFIX}_cortex_59k.dtseries.nii"
        if [ ! -f "$FMRI_PATH" ]; then
            echo "[$(date +%H:%M:%S)] ${SUB}: no preprocessed CIFTI — run preprocess_individual.py first" \
                | tee -a "$LOG"
            return 1
        fi
        FMRI_FLAGS="--preprocessed-dir ${PREPROCESSED_DIR} --fmri-suffix ${FMRI_SUFFIX}"
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} (stream=${STREAM})" | tee -a "$LOG"

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
                local SL_OUT="${OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${SL_CONFIG}/rsa_59k_${FMRI_TAG_LOCAL}_k${K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${METHOD}_maps.dscalar.nii"
                if [ -f "$SL_OUT" ]; then
                    echo "[$(date +%H:%M:%S)] ${SUB}: searchlight ${MODEL_NAME}/${MOD} already done — skipping" \
                        | tee -a "$LOG"
                else
                    # shellcheck disable=SC2086
                    conda run -n "$CONDA_ENV" python \
                        "${SCRIPT_DIR}/run_searchlight.py" \
                        $FMRI_FLAGS \
                        --timing-csv       "$TIMING_CSV" \
                        --embeddings-dir   "$EMBEDDINGS_DIR" \
                        --template-cifti   "$TEMPLATE_CIFTI" \
                        --output-dir       "$OUTPUT_DIR" \
                        --subject          "$SUB" \
                        --model            "$MODEL_NAME" \
                        --modality         "$MOD" \
                        --k                "$K" \
                        --bin-sec          "$BIN_SEC" \
                        --delay-sec        "$DELAY_SEC" \
                        --method           "$METHOD" \
                        --tr               "$TR" \
                        --left-surface     "$LEFT_SURF" \
                        --right-surface    "$RIGHT_SURF" \
                        --workbench        "$WORKBENCH" \
                        $HRF_FLAG \
                        >> "$LOG" 2>&1
                fi
            fi

            if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "glasser" ]; then
                local GL_OUT="${OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${GL_CONFIG}/glasser_rsa_${FMRI_TAG_LOCAL}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${METHOD}_maps.dscalar.nii"
                if [ -f "$GL_OUT" ]; then
                    echo "[$(date +%H:%M:%S)] ${SUB}: glasser ${MODEL_NAME}/${MOD} already done — skipping" \
                        | tee -a "$LOG"
                else
                    # shellcheck disable=SC2086
                    conda run -n "$CONDA_ENV" python \
                        "${SCRIPT_DIR}/run_glasser.py" \
                        $FMRI_FLAGS \
                        --timing-csv       "$TIMING_CSV" \
                        --embeddings-dir   "$EMBEDDINGS_DIR" \
                        --template-cifti   "$TEMPLATE_CIFTI" \
                        --output-dir       "$OUTPUT_DIR" \
                        --subject          "$SUB" \
                        --model            "$MODEL_NAME" \
                        --modality         "$MOD" \
                        --bin-sec          "$BIN_SEC" \
                        --delay-sec        "$DELAY_SEC" \
                        --method           "$METHOD" \
                        --tr               "$TR" \
                        --glasser-dlabel   "$GLASSER_DLABEL" \
                        $HRF_FLAG \
                        >> "$LOG" 2>&1
                fi
            fi
        done
    done

    local STATUS=$?

    # Backup cleanup
    local CACHE_DIR="${OUTPUT_DIR}/_geodesic_cache"
    rm -f "${CACHE_DIR}/${SUB}_left_geodesic.dconn.nii"
    rm -f "${CACHE_DIR}/${SUB}_right_geodesic.dconn.nii"

    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} DONE" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB} FAILED (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_subject

run_persubject() {
    local MODE_TAG
    [ "$STREAM" = "true" ] && MODE_TAG="streaming" || MODE_TAG="disk"
    log "=== Per-subject RSA (${MODE_TAG}, ${BATCH_SIZE} parallel jobs, ${#MODELS[@]} models) ==="

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
            "$SCRIPT_DIR" "$CONDA_ENV" "$PREPROCESSED_DIR" "$FMRI_SUFFIX" "$OUTPUT_DIR" \
            "$TIMING_CSV" "$EMBEDDINGS_DIR" "$TEMPLATE_CIFTI" \
            "$LEFT_SURFACE" "$RIGHT_SURFACE" "$GLASSER_DLABEL" \
            "$WORKBENCH" "$MODELS_STR" "$BIN_SEC" "$DELAY_SEC" "$TR" \
            "$K" "$METHOD" "$HRF" "$METHOD_ARG" \
            "$STREAM" "$CIFTI_DIR" "$SG_FILTER" "$PSC" "$GSR" \
            "$MIDTHICKNESS_DIR"
    else
        log "GNU parallel not found — running sequentially"
        log "  (install with: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB" \
                "$SCRIPT_DIR" "$CONDA_ENV" "$PREPROCESSED_DIR" "$FMRI_SUFFIX" "$OUTPUT_DIR" \
                "$TIMING_CSV" "$EMBEDDINGS_DIR" "$TEMPLATE_CIFTI" \
                "$LEFT_SURFACE" "$RIGHT_SURFACE" "$GLASSER_DLABEL" \
                "$WORKBENCH" "$MODELS_STR" "$BIN_SEC" "$DELAY_SEC" "$TR" \
                "$K" "$METHOD" "$HRF" "$METHOD_ARG" \
                "$STREAM" "$CIFTI_DIR" "$SG_FILTER" "$PSC" "$GSR" \
                "$MIDTHICKNESS_DIR"
        done
    fi

    log "=== Per-subject RSA complete ==="
}

# =============================================================================
# GROUP STATS PIPELINE
# =============================================================================
run_group_stats() {
    local MODE_TAG
    [ "$STREAM" = "true" ] && MODE_TAG="streaming" || MODE_TAG="disk"
    log "=== Group stats (${MODE_TAG}, ${#MODELS[@]} models) ==="

    local MODELS_STR
    MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")

    IFS=';' read -ra MODEL_ENTRIES <<< "$MODELS_STR"
    for MODEL_ENTRY in "${MODEL_ENTRIES[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"

        for MOD in "${MODS[@]}"; do
            log "  Group stats: ${MODEL_NAME} / ${MOD}"
            run_python "${SCRIPT_DIR}/run_group_stats.py" \
                --output-dir    "$OUTPUT_DIR" \
                --model         "$MODEL_NAME" \
                --modality      "$MOD" \
                --k             "$K" \
                --bin-sec       "$BIN_SEC" \
                --delay-sec     "$DELAY_SEC" \
                --method        "$METHOD" \
                --fmri-tag      "$PREPROCESSING_FLAG" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --left-surface  "$LEFT_SURFACE" \
                --right-surface "$RIGHT_SURFACE"
        done
    done

    log "=== Group stats done ==="
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    avg)        run_avg ;;
    persubject) run_persubject; run_group_stats ;;
    groupstats) run_group_stats ;;
    all)        run_avg; run_persubject; run_group_stats ;;
    *)
        echo "Unknown mode: $MODE. Use: avg | persubject | groupstats | all" >&2; exit 1 ;;
esac

log "All RSA analyses complete."
