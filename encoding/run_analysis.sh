#!/usr/bin/env bash
# encoding/run_analysis.sh
# ==========================
# Master runner for ridge encoding model analyses.
# Supports group-average and per-subject modes with GNU parallel.
#
# Usage
# -----
#   bash encoding/run_analysis.sh [MODE] [BATCH_SIZE] [START_FROM]
#
#   MODE        preprocess   Preprocess all 175 subjects from SUBJECTS_LIST:
#                            per-subject CIFTIs → PREPROCESSED_INDIV_DIR
#                            group-average CIFTI → PREPROCESSED_DIR
#                            Respects SG_FILTER/PSC/GSR flags and resumes.
#               avg          Group-average encoding model (default)
#               persubject   Per-subject encoding → (no group-stats step yet)
#               all          avg + persubject
#
#   BATCH_SIZE  N            Parallel subjects (default 8)
#   START_FROM  SUBID        Resume per-subject from this subject ID
#
# Recommended workflow
#   # 0. Preprocess all 175 subjects (skip if using streaming mode)
#   bash encoding/run_analysis.sh preprocess
#
#   # 1. Group-average encoding
#   bash encoding/run_analysis.sh avg
#
#   # 2. Per-subject encoding
#   bash encoding/run_analysis.sh persubject 8
#
# Streaming vs disk mode
#   STREAM=true  — raw 7T CIFTIs preprocessed on-the-fly (SG→PSC→GSR→zscore)
#   STREAM=false — reads pre-saved CIFTIs from PREPROCESSED_INDIV_DIR
#
# Excluded subjects
#   Subjects without individual midthickness surfaces are listed in
#   data/excluded.txt and have been removed from data/subjects.txt.
#   175 subjects remain.
#
# Resume / skip
#   persubject: skips any subject/model whose output map already exists.
#   Delete the output file to force a rerun.
#
# Silence GNU parallel citation notice (run once)
#   parallel --citation

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# Raw 7T CIFTI files (used in streaming mode — preprocess on-the-fly)
CIFTI_DIR="/media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI"

# ── Streaming toggle ──────────────────────────────────────────────────────────
# false → disk mode (read pre-saved preprocessed CIFTIs from PREPROCESSED_INDIV_DIR)
# true  → streaming mode (preprocess raw CIFTIs from CIFTI_DIR on-the-fly)
STREAM=false

# Preprocessing flags
SG_FILTER=false   # Savitzky-Golay high-pass filter
PSC=false         # Percent signal change normalization
GSR=true          # Global signal regression
Z_SCORE=true      # Z-score per vertex (applied inside run_encoding.py; not by preprocess_individual.py)

# Automatically build PREPROCESSING_FLAG from SG_FILTER/PSC/GSR
# (Z_SCORE is NOT included — it is applied inside the Python analysis script)
PREP_PARTS=()
[ "$SG_FILTER" = "true" ] && PREP_PARTS+=("sg")
[ "$PSC"       = "true" ] && PREP_PARTS+=("psc")
[ "$GSR"       = "true" ] && PREP_PARTS+=("gsr")

if [ ${#PREP_PARTS[@]} -eq 0 ]; then
    PREPROCESSING_FLAG="raw"
else
    PREPROCESSING_FLAG=$(IFS=_; echo "${PREP_PARTS[*]}")
fi

# fMRI data paths
DELAY_SEC=5.0
FMRI_SUFFIX="${PREPROCESSING_FLAG}"
# Group-average preprocessed CIFTI (output of preprocess_individual.py --save-average)
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
# Per-subject preprocessed CIFTIs (output of preprocess_individual.py --save-individual)
# File pattern: {PREPROCESSED_INDIV_DIR}/{sub}_{PREPROCESSING_FLAG}_cortex_59k.dtseries.nii
PREPROCESSED_INDIV_DIR="${DATA_BASE}/preprocessed/${PREPROCESSING_FLAG}"

# Single authoritative subject list — 175 subjects with full 7T fMRI + midthickness.
# All pipeline stages (preprocess / encoding) must read from here.
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# Timing
TIMING_CSV="${DATA_BASE}/movie_timing.csv"

# Embeddings root
# Convention: {EMBEDDINGS_DIR}/{model_name}/{bin_sec}s/{model_name}_{modality}.npy
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# CIFTI template (59k grayordinate space).
# Must be a 59k cortex-only file (29696 L + 29716 R = 59412 vertices).
# The preprocessed group-average dtseries has 108441 grayordinates (full
# subcortical + cortical) and cannot be used as a template — nibabel will
# raise a shape mismatch error at save time.  Use the static HCP curvature
# dscalar which always exists and has the correct BrainModelAxis.
TEMPLATE_CIFTI="/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"

# Output root
OUTPUT_DIR="${OUTPUTS_BASE}/encoding"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
BIN_SEC=5.0
HRF=false       # true → SPM HRF convolution; false → boxcar delay
NORMALIZE=true  # per-run z-score normalization of embeddings

# Ridge regularisation search
ALPHA_MIN=-2
ALPHA_MAX=9
N_ALPHAS=23

# ── himalaya backend ───────────────────────────────────────────────────────
# torch_cuda uses GPU if CUDA is available (auto-detected); falls back to torch.
# Use "numpy" for CPU-only (slower but no GPU memory required).
BACKEND="torch_cuda"

TEST_VIDEO_IDS="video5,video9,video14,video18"

# ── Model registry ─────────────────────────────────────────────────────────
# Format: "model_name:modalities"
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
CONDA_ENV="movie"
DEFAULT_BATCH_SIZE=8
# =============================================================================

MODE=${1:-avg}
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
START_FROM=${3:-""}

BIN_SEC_INT="${BIN_SEC%.*}"

N_CPUS=$(nproc 2>/dev/null || echo 8)
N_JOBS_PER_SUBJECT=$(( N_CPUS / BATCH_SIZE ))
[ "$N_JOBS_PER_SUBJECT" -lt 1 ] && N_JOBS_PER_SUBJECT=1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
# PREPROCESSING PIPELINE
# =============================================================================
run_preprocess() {
    log "=== Preprocessing n=$(grep -cv '^\s*#' "$SUBJECTS_LIST") subjects → ${PREPROCESSED_INDIV_DIR} ==="
    log "  Flags: SG_FILTER=${SG_FILTER}  PSC=${PSC}  GSR=${GSR}  (${PREPROCESSING_FLAG})"
    log "  Note: Z_SCORE is applied inside run_encoding.py, not during preprocessing"
    log "  Subjects: ${SUBJECTS_LIST}"
    log "  Raw CIFTI dir: ${CIFTI_DIR}"

    local SG_FLAG="" PSC_FLAG="" GSR_FLAG="--no-gsr"
    [ "$SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
    [ "$PSC"       = "true" ] && PSC_FLAG="--psc"
    [ "$GSR"       = "true" ] && GSR_FLAG="--gsr"

    run_python "${SCRIPT_DIR}/../preprocess_individual.py" \
        --raw-dir        "$CIFTI_DIR" \
        --out-dir        "$PREPROCESSED_INDIV_DIR" \
        --subjects-list  "$SUBJECTS_LIST" \
        --tr             "$TR" \
        $SG_FLAG $PSC_FLAG $GSR_FLAG \
        --save-individual \
        --save-average

    # Move group average to PREPROCESSED_DIR so disk-mode avg encoding finds it
    local GA_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
    local GA_TRS_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
    if [ -f "$GA_SRC" ]; then
        mkdir -p "$PREPROCESSED_DIR"
        mv -f "$GA_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
        [ -f "$GA_TRS_SRC" ] && mv -f "$GA_TRS_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
        log "  Group average moved → ${PREPROCESSED_DIR}"
    fi

    log "=== Preprocessing complete ==="
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
                --tr               "$TR" \
                --alpha-min        "$ALPHA_MIN" \
                --alpha-max        "$ALPHA_MAX" \
                --n-alphas         "$N_ALPHAS" \
                --backend          "$BACKEND" \
                --test-video-ids   "$TEST_VIDEO_IDS" \
                $(_hrf_flag) $(_normalize_flag)
        done
    done

    log "=== Group-average encoding done ==="
}

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================

# Worker function — exported for GNU parallel.
# All config is read from exported _ENC_-prefixed env vars so paths with
# spaces are never mishandled by parallel's argument tokenisation.
# Only the subject ID is passed as a positional argument.
_run_one_subject() {
    local SUB="$1"

    local BIN_SEC_INT="${_ENC_BIN_SEC%.*}"
    local LOG_DIR="${_ENC_OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/pipeline.log"

    local HRF_FLAG="";  [ "$_ENC_HRF"       = "true" ] && HRF_FLAG="--hrf"
    local NORM_FLAG=""; [ "$_ENC_NORMALIZE"  = "true" ] && NORM_FLAG="--normalize"

    # Build fMRI input flags based on mode
    local FMRI_FLAGS
    if [ "$_ENC_STREAM" = "true" ]; then
        local SG_FLAG="";   [ "$_ENC_SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        local PSC_FLAG="";  [ "$_ENC_PSC"       = "true" ] && PSC_FLAG="--psc"
        local GSR_FLAG="--gsr";     [ "$_ENC_GSR"     = "false" ] && GSR_FLAG="--no-gsr"
        local ZSC_FLAG="--z-score"; [ "$_ENC_Z_SCORE" = "false" ] && ZSC_FLAG="--no-z-score"
        FMRI_FLAGS="--raw-dir ${_ENC_CIFTI_DIR} $SG_FLAG $PSC_FLAG $GSR_FLAG $ZSC_FLAG"
    else
        local FMRI_PATH="${_ENC_PREPROCESSED_INDIV_DIR}/${SUB}_${_ENC_FMRI_SUFFIX}_cortex_59k.dtseries.nii"
        if [ ! -f "$FMRI_PATH" ]; then
            echo "[$(date +%H:%M:%S)] ${SUB}: no preprocessed CIFTI — run preprocess mode first" \
                | tee -a "$LOG"
            return 1
        fi
        FMRI_FLAGS="--preprocessed-dir ${_ENC_PREPROCESSED_INDIV_DIR} --fmri-suffix ${_ENC_FMRI_SUFFIX}"
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} (stream=${_ENC_STREAM})" | tee -a "$LOG"

    local STATUS=0
    IFS=';' read -ra MODEL_ENTRIES <<< "$_ENC_MODELS_STR"
    for MODEL_ENTRY in "${MODEL_ENTRIES[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"

        for MOD in "${MODS[@]}"; do
            local EMB="${_ENC_EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_SEC_INT}s/${MODEL_NAME}_${MOD}.npy"
            if [ ! -f "$EMB" ]; then
                echo "[$(date +%H:%M:%S)] ${SUB}: SKIP ${MODEL_NAME}/${MOD} — no embedding" \
                    | tee -a "$LOG"
                continue
            fi

            # shellcheck disable=SC2086
            conda run --no-capture-output -n "$_ENC_CONDA_ENV" python \
                "${_ENC_SCRIPT_DIR}/run_encoding.py" \
                $FMRI_FLAGS \
                --timing-csv     "$_ENC_TIMING_CSV" \
                --embeddings-dir "$_ENC_EMBEDDINGS_DIR" \
                --template-cifti "$_ENC_TEMPLATE_CIFTI" \
                --output-dir     "$_ENC_OUTPUT_DIR" \
                --subject        "$SUB" \
                --model          "$MODEL_NAME" \
                --modality       "$MOD" \
                --bin-sec        "$_ENC_BIN_SEC" \
                --delay-sec      "$_ENC_DELAY_SEC" \
                --tr             "$_ENC_TR" \
                --alpha-min      "$_ENC_ALPHA_MIN" \
                --alpha-max      "$_ENC_ALPHA_MAX" \
                --n-alphas       "$_ENC_N_ALPHAS" \
                --backend        "$_ENC_BACKEND" \
                --test-video-ids "$_ENC_TEST_VIDEO_IDS" \
                $HRF_FLAG $NORM_FLAG \
                >> "$LOG" 2>&1 || STATUS=$?
        done
    done

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
    log "=== Per-subject encoding (${MODE_TAG}, ${BATCH_SIZE} parallel jobs, ${#MODELS[@]} models) ==="

    local SUBJECTS
    SUBJECTS=$(grep -v '^\s*#' "$SUBJECTS_LIST" \
               | sed 's/#.*//' \
               | awk '{print $1}' \
               | grep -v '^$')
    if [ -n "$START_FROM" ]; then
        SUBJECTS=$(echo "$SUBJECTS" | awk "/$START_FROM/{found=1} found{print}")
    fi
    local N_TOTAL
    N_TOTAL=$(echo "$SUBJECTS" | wc -l)
    log "  Processing ${N_TOTAL} subjects from ${SUBJECTS_LIST} ..."

    local MODELS_STR
    MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")

    export _ENC_SCRIPT_DIR="$SCRIPT_DIR"
    export _ENC_CONDA_ENV="$CONDA_ENV"
    export _ENC_PREPROCESSED_INDIV_DIR="$PREPROCESSED_INDIV_DIR"
    export _ENC_FMRI_SUFFIX="$FMRI_SUFFIX"
    export _ENC_OUTPUT_DIR="$OUTPUT_DIR"
    export _ENC_TIMING_CSV="$TIMING_CSV"
    export _ENC_EMBEDDINGS_DIR="$EMBEDDINGS_DIR"
    export _ENC_TEMPLATE_CIFTI="$TEMPLATE_CIFTI"
    export _ENC_MODELS_STR="$MODELS_STR"
    export _ENC_BIN_SEC="$BIN_SEC"
    export _ENC_DELAY_SEC="$DELAY_SEC"
    export _ENC_TR="$TR"
    export _ENC_ALPHA_MIN="$ALPHA_MIN"
    export _ENC_ALPHA_MAX="$ALPHA_MAX"
    export _ENC_N_ALPHAS="$N_ALPHAS"
    export _ENC_TEST_VIDEO_IDS="$TEST_VIDEO_IDS"
    export _ENC_HRF="$HRF"
    export _ENC_NORMALIZE="$NORMALIZE"
    export _ENC_BACKEND="$BACKEND"
    export _ENC_STREAM="$STREAM"
    export _ENC_CIFTI_DIR="$CIFTI_DIR"
    export _ENC_SG_FILTER="$SG_FILTER"
    export _ENC_PSC="$PSC"
    export _ENC_GSR="$GSR"
    export _ENC_Z_SCORE="$Z_SCORE"

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_subject {}
    else
        log "GNU parallel not found — running sequentially"
        log "  (install with: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB"
        done
    fi

    log "=== Per-subject encoding complete ==="
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    preprocess) run_preprocess ;;
    avg)        run_avg ;;
    persubject) run_persubject ;;
    all)        run_avg; run_persubject ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: preprocess | avg | persubject | all" >&2
        exit 1 ;;
esac

log "All encoding analyses complete."
