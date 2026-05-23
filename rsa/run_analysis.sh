#!/usr/bin/env bash
# rsa/run_analysis.sh
# ====================
# Master runner for searchlight and Glasser parcel RSA analyses.
# Supports group-average and per-subject modes with GNU parallel.
#
# Usage
# -----
#   bash rsa/run_analysis.sh [MODE] [METHOD] [BATCH_SIZE] [START_FROM]
#
#   MODE        avg          Group-average RSA only (default)
#               neighbors    Pre-compute geodesic k-NN caches for all subjects.
#                            Run this BEFORE persubject to front-load the
#                            ~80 min/hemisphere wb_command step.  Cached .npy
#                            files are reused by any future k ≤ K run too.
#               persubject   Per-subject RSA → group stats
#               groupstats   Re-run group stats on existing per-subject maps
#               all          avg + persubject + groupstats
#
#   METHOD      all          Searchlight + Glasser parcel RSA (default)
#               searchlight  Searchlight only
#               glasser      Glasser parcellation only
#               (ignored for 'neighbors' mode)
#
#   BATCH_SIZE  N            Parallel subjects (default 4)
#                            neighbors mode: each subject spawns 2 wb_command
#                            calls (L+R, sequential) — set to nproc/4 so each
#                            wb_command gets ~4 cores.
#                            persubject mode: N_JOBS_PER_SUBJECT = nproc/N
#                            searchlight threads per subject.
#   START_FROM  SUBID        Resume from this subject ID
#
# Recommended workflow
#   # 1. Pre-build all k=150 neighbour caches (run once; ~1.5 days at B=4)
#   bash rsa/run_analysis.sh neighbors all 4
#
#   # 2. Per-subject searchlight (all cache-hits → predictable ~100 min/subject)
#   bash rsa/run_analysis.sh persubject searchlight 4
#
#   # 3. Re-run group stats after adding subjects
#   bash rsa/run_analysis.sh groupstats
#
#   # Other
#   bash rsa/run_analysis.sh avg                           # group average only
#   bash rsa/run_analysis.sh persubject searchlight 4 100610  # resume from sub
#
# Performance notes
#   With 32 cores and per-vertex sequential time ~0.5 s (k=150, Spearman):
#     BATCH=1  n_jobs=32 → ~28 min/subject  (lowest per-subject latency)
#     BATCH=4  n_jobs=8  → ~55 min/subject  (4× throughput, balanced)
#     BATCH=8  n_jobs=4  → ~92 min/subject  (8× throughput, highest total CPUs)
#   Total wall time (~170 subjects) ≈ 65-80 hours at any batch size.
#   k' < K: if k=150 .npy files exist, k'=100 (or any k'<150) is derived
#   instantly from the k=150 cache (no wb_command rerun needed).
#
# Streaming vs disk mode
#   STREAM=true  (default) — raw 7T CIFTIs preprocessed on-the-fly (SG→PSC→GSR)
#   STREAM=false           — reads pre-saved CIFTIs from PREPROCESSED_DIR
#
# Per-subject midthickness surfaces
#   Uses {MIDTHICKNESS_DIR}/{sub}.L/R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii
#   if present, otherwise falls back to group-average midthickness.
#
# Resume / skip
#   neighbors  : skips subjects whose both hemisphere .npy files exist.
#   persubject : skips any subject/model whose output CIFTI already exists.
#   Delete the output CIFTI (or .npy) to force a rerun.
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
TIMING_CSV="/home/amin/Research/Representation/Movie/data/movie_timing.csv"

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
# Default batch size for persubject searchlight.
# With 32 cores: B=4 → n_jobs=8/subject → ~55 min/subject, 4 parallel.
# Change with the 3rd CLI argument: bash run_analysis.sh persubject all 8
DEFAULT_BATCH_SIZE=4
# Default batch size for the neighbors precompute.
# wb_command geodesic is the bottleneck; set to nproc/4 so each command
# gets ~4 cores.  Override with the 3rd CLI arg when running neighbors mode.
DEFAULT_NEIGHBORS_BATCH_SIZE=8
# =============================================================================

MODE=${1:-avg}
METHOD_ARG=${2:-all}
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
START_FROM=${4:-""}

BIN_SEC_INT="${BIN_SEC%.*}"

N_CPUS=$(nproc 2>/dev/null || echo 8)

# For 'neighbors' mode the batch size should default higher (wb_command is the
# bottleneck, not Python threads), so we rebind BATCH_SIZE here when needed.
if [ "$MODE" = "neighbors" ] && [ -z "${3:-}" ]; then
    BATCH_SIZE=$DEFAULT_NEIGHBORS_BATCH_SIZE
fi

# Joblib n_jobs per subject (searchlight only): divide available CPUs by
# parallel batch size so total threads ≈ nproc.
N_JOBS_PER_SUBJECT=$(( N_CPUS / BATCH_SIZE ))
[ "$N_JOBS_PER_SUBJECT" -lt 1 ] && N_JOBS_PER_SUBJECT=1

# Limit BLAS (OpenBLAS / MKL / BLIS) to 1 thread per process.
# Without this, each of the N_JOBS joblib threads may spawn its own BLAS thread pool
# for numpy matmul, resulting in BATCH_SIZE × N_JOBS × BLAS_threads >> nproc threads
# competing for the same cores (load averages 150+ instead of ~32).
# The per-vertex matmul shapes are too small for BLAS parallelism to help anyway.
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
# All config is read from exported env vars (set by run_persubject before
# calling parallel) so that paths with spaces are never mishandled by parallel's
# argument tokenisation.  Only the subject ID is passed as a positional arg.
_run_one_subject() {
    local SUB="$1"

    local BIN_SEC_INT="${_RSA_BIN_SEC%.*}"
    local LOG_DIR="${_RSA_OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/pipeline.log"

    local HRF_FLAG=""; [ "$_RSA_HRF" = "true" ] && HRF_FLAG="--hrf"

    # ── Resume: derive fmri_tag + config strings (mirrors Python naming) ──────
    local FMRI_TAG_PARTS=()
    [ "$_RSA_SG_FILTER" = "true" ] && FMRI_TAG_PARTS+=("sg")
    [ "$_RSA_PSC"       = "true" ] && FMRI_TAG_PARTS+=("psc")
    [ "$_RSA_GSR"       = "true" ] && FMRI_TAG_PARTS+=("gsr")
    local FMRI_TAG_LOCAL
    if [ ${#FMRI_TAG_PARTS[@]} -eq 0 ]; then
        FMRI_TAG_LOCAL="raw"
    else
        FMRI_TAG_LOCAL=$(IFS=_; echo "${FMRI_TAG_PARTS[*]}")
    fi
    [ "$_RSA_STREAM" = "false" ] && FMRI_TAG_LOCAL="$_RSA_FMRI_SUFFIX"

    local DELAY_INT="${_RSA_DELAY_SEC%.*}"
    local SL_CONFIG="k${_RSA_K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}s_${_RSA_METHOD}"
    local GL_CONFIG="delay${DELAY_INT}s_bin${BIN_SEC_INT}s_${_RSA_METHOD}"

    # ── Per-subject midthickness (fall back to group-average) ─────────────────
    local LEFT_SURF="$_RSA_LEFT_SURFACE"
    local RIGHT_SURF="$_RSA_RIGHT_SURFACE"
    local SUB_L="${_RSA_MIDTHICKNESS_DIR}/${SUB}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    local SUB_R="${_RSA_MIDTHICKNESS_DIR}/${SUB}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    if [ -f "$SUB_L" ] && [ -f "$SUB_R" ]; then
        LEFT_SURF="$SUB_L"
        RIGHT_SURF="$SUB_R"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: subject midthickness not found, using group-average surface" \
            | tee -a "$LOG"
    fi

    # ── fMRI input flags ───────────────────────────────────────────────────────
    local FMRI_FLAGS
    if [ "$_RSA_STREAM" = "true" ]; then
        local SG_FLAG="";  [ "$_RSA_SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        local PSC_FLAG=""; [ "$_RSA_PSC"       = "true" ] && PSC_FLAG="--psc"
        local GSR_FLAG="--gsr"; [ "$_RSA_GSR" = "false" ] && GSR_FLAG="--no-gsr"
        FMRI_FLAGS="--raw-dir ${_RSA_CIFTI_DIR} $SG_FLAG $PSC_FLAG $GSR_FLAG"
    else
        local FMRI_PATH="${_RSA_PREPROCESSED_DIR}/${SUB}_${_RSA_FMRI_SUFFIX}_cortex_59k.dtseries.nii"
        if [ ! -f "$FMRI_PATH" ]; then
            echo "[$(date +%H:%M:%S)] ${SUB}: no preprocessed CIFTI — run preprocess_individual.py first" \
                | tee -a "$LOG"
            return 1
        fi
        FMRI_FLAGS="--preprocessed-dir ${_RSA_PREPROCESSED_DIR} --fmri-suffix ${_RSA_FMRI_SUFFIX}"
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} (stream=${_RSA_STREAM})" | tee -a "$LOG"

    # Track any conda run failure across all models/modalities.
    # IMPORTANT: do NOT use `local STATUS=$?` after the loops — a false-condition
    # `if` block with no `else` resets $? to 0 and would mask real failures.
    local STATUS=0

    IFS=';' read -ra MODEL_ENTRIES <<< "$_RSA_MODELS_STR"
    for MODEL_ENTRY in "${MODEL_ENTRIES[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"

        for MOD in "${MODS[@]}"; do
            local EMB="${_RSA_EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_SEC_INT}s/${MODEL_NAME}_${MOD}.npy"
            if [ ! -f "$EMB" ]; then
                echo "[$(date +%H:%M:%S)] ${SUB}: SKIP ${MODEL_NAME}/${MOD} — no embedding" \
                    | tee -a "$LOG"
                continue
            fi

            if [ "$_RSA_METHOD_ARG" = "all" ] || [ "$_RSA_METHOD_ARG" = "searchlight" ]; then
                local SL_OUT="${_RSA_OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${SL_CONFIG}/rsa_59k_${FMRI_TAG_LOCAL}_k${_RSA_K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${_RSA_METHOD}_maps.dscalar.nii"
                if [ -f "$SL_OUT" ]; then
                    echo "[$(date +%H:%M:%S)] ${SUB}: searchlight ${MODEL_NAME}/${MOD} already done — skipping" \
                        | tee -a "$LOG"
                else
                    # shellcheck disable=SC2086
                    conda run --no-capture-output -n "$_RSA_CONDA_ENV" python \
                        "${_RSA_SCRIPT_DIR}/run_searchlight.py" \
                        $FMRI_FLAGS \
                        --timing-csv       "$_RSA_TIMING_CSV" \
                        --embeddings-dir   "$_RSA_EMBEDDINGS_DIR" \
                        --template-cifti   "$_RSA_TEMPLATE_CIFTI" \
                        --output-dir       "$_RSA_OUTPUT_DIR" \
                        --subject          "$SUB" \
                        --model            "$MODEL_NAME" \
                        --modality         "$MOD" \
                        --k                "$_RSA_K" \
                        --bin-sec          "$_RSA_BIN_SEC" \
                        --delay-sec        "$_RSA_DELAY_SEC" \
                        --method           "$_RSA_METHOD" \
                        --tr               "$_RSA_TR" \
                        --left-surface     "$LEFT_SURF" \
                        --right-surface    "$RIGHT_SURF" \
                        --workbench        "$_RSA_WORKBENCH" \
                        $HRF_FLAG \
                        >> "$LOG" 2>&1 || STATUS=$?
                fi
            fi

            if [ "$_RSA_METHOD_ARG" = "all" ] || [ "$_RSA_METHOD_ARG" = "glasser" ]; then
                local GL_OUT="${_RSA_OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${GL_CONFIG}/glasser_rsa_${FMRI_TAG_LOCAL}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${_RSA_METHOD}_maps.dscalar.nii"
                if [ -f "$GL_OUT" ]; then
                    echo "[$(date +%H:%M:%S)] ${SUB}: glasser ${MODEL_NAME}/${MOD} already done — skipping" \
                        | tee -a "$LOG"
                else
                    # shellcheck disable=SC2086
                    conda run --no-capture-output -n "$_RSA_CONDA_ENV" python \
                        "${_RSA_SCRIPT_DIR}/run_glasser.py" \
                        $FMRI_FLAGS \
                        --timing-csv       "$_RSA_TIMING_CSV" \
                        --embeddings-dir   "$_RSA_EMBEDDINGS_DIR" \
                        --template-cifti   "$_RSA_TEMPLATE_CIFTI" \
                        --output-dir       "$_RSA_OUTPUT_DIR" \
                        --subject          "$SUB" \
                        --model            "$MODEL_NAME" \
                        --modality         "$MOD" \
                        --bin-sec          "$_RSA_BIN_SEC" \
                        --delay-sec        "$_RSA_DELAY_SEC" \
                        --method           "$_RSA_METHOD" \
                        --tr               "$_RSA_TR" \
                        --glasser-dlabel   "$_RSA_GLASSER_DLABEL" \
                        $HRF_FLAG \
                        >> "$LOG" 2>&1 || STATUS=$?
                fi
            fi
        done
    done

    local CACHE_DIR="${_RSA_OUTPUT_DIR}/_geodesic_cache"
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

# =============================================================================
# NEIGHBOURS PRE-COMPUTE PIPELINE
# =============================================================================

# Worker — exported for GNU parallel.
# Computes geodesic k-NN .npy caches for one subject (both hemispheres).
# Skips subjects whose both hemisphere files already exist.
_run_one_neighbors() {
    local SUB="$1"

    local CACHE_DIR="${_RSA_OUTPUT_DIR}/_geodesic_cache"
    local NPY_L="${CACHE_DIR}/${SUB}_left_neighbors_k${_RSA_K}.npy"
    local NPY_R="${CACHE_DIR}/${SUB}_right_neighbors_k${_RSA_K}.npy"

    # Resume: skip if both hemispheres already cached
    if [ -f "$NPY_L" ] && [ -f "$NPY_R" ]; then
        echo "[$(date +%H:%M:%S)] ${SUB}: k-NN k=${_RSA_K} already cached — skipping"
        return 0
    fi

    local LOG_DIR="${_RSA_OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/neighbors.log"

    # Per-subject midthickness (fall back to group-average)
    local LEFT_SURF="$_RSA_LEFT_SURFACE"
    local RIGHT_SURF="$_RSA_RIGHT_SURFACE"
    local SUB_L="${_RSA_MIDTHICKNESS_DIR}/${SUB}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    local SUB_R="${_RSA_MIDTHICKNESS_DIR}/${SUB}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    if [ -f "$SUB_L" ] && [ -f "$SUB_R" ]; then
        LEFT_SURF="$SUB_L"
        RIGHT_SURF="$SUB_R"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: no per-subject midthickness — using group-average surface" \
            | tee -a "$LOG"
    fi

    echo "[$(date +%H:%M:%S)] ${SUB}: computing k=${_RSA_K} neighbor cache ..." | tee -a "$LOG"

    local STATUS=0
    conda run --no-capture-output -n "$_RSA_CONDA_ENV" python \
        "${_RSA_SCRIPT_DIR}/precompute_neighbors.py" \
        --subject       "$SUB" \
        --left-surface  "$LEFT_SURF" \
        --right-surface "$RIGHT_SURF" \
        --workbench     "$_RSA_WORKBENCH" \
        --cache-dir     "$CACHE_DIR" \
        --k             "$_RSA_K" \
        >> "$LOG" 2>&1 || STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ${SUB}: neighbors DONE" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: neighbors FAILED (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_neighbors

run_precompute_neighbors() {
    log "=== Pre-compute k=${K} neighbors (${BATCH_SIZE} parallel subjects) ==="
    log "  Each subject: wb_command geodesic (~70 min/hem) + k-NN extraction (~10 min/hem)"
    log "  Subjects with cached .npy files are skipped automatically."

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
    log "  ${N_TOTAL} subjects (BATCH_SIZE=${BATCH_SIZE})"

    # Count already-cached
    local CACHE_DIR="${OUTPUT_DIR}/_geodesic_cache"
    local N_CACHED=0
    for SUB in $SUBJECTS; do
        [ -f "${CACHE_DIR}/${SUB}_left_neighbors_k${K}.npy" ] && \
        [ -f "${CACHE_DIR}/${SUB}_right_neighbors_k${K}.npy" ] && \
        N_CACHED=$(( N_CACHED + 1 ))
    done
    log "  ${N_CACHED} subjects already cached; $((N_TOTAL - N_CACHED)) to compute"

    # Export env vars for the worker
    export _RSA_SCRIPT_DIR="$SCRIPT_DIR"
    export _RSA_CONDA_ENV="$CONDA_ENV"
    export _RSA_OUTPUT_DIR="$OUTPUT_DIR"
    export _RSA_LEFT_SURFACE="$LEFT_SURFACE"
    export _RSA_RIGHT_SURFACE="$RIGHT_SURFACE"
    export _RSA_WORKBENCH="$WORKBENCH"
    export _RSA_K="$K"
    export _RSA_MIDTHICKNESS_DIR="$MIDTHICKNESS_DIR"

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_neighbors {}
    else
        log "GNU parallel not found — running sequentially"
        for SUB in $SUBJECTS; do
            _run_one_neighbors "$SUB"
        done
    fi

    log "=== Neighbor precompute complete ==="
}

run_persubject() {
    local MODE_TAG
    [ "$STREAM" = "true" ] && MODE_TAG="streaming" || MODE_TAG="disk"
    log "=== Per-subject RSA (${MODE_TAG}, ${BATCH_SIZE} parallel jobs, ${N_JOBS_PER_SUBJECT} searchlight threads/subject, ${#MODELS[@]} models) ==="

    local SUBJECTS
    # Strip full-line comments (^#), inline comments (#...), blank lines;
    # take only the first field (subject ID) so inline notes don't become tokens.
    SUBJECTS=$(grep -v '^\s*#' "$SUBJECTS_LIST" \
               | sed 's/#.*//' \
               | awk '{print $1}' \
               | grep -v '^$')
    if [ -n "$START_FROM" ]; then
        SUBJECTS=$(echo "$SUBJECTS" | awk "/$START_FROM/{found=1} found{print}")
    fi
    local N_TOTAL
    N_TOTAL=$(echo "$SUBJECTS" | wc -l)
    log "  Processing ${N_TOTAL} subjects ..."

    # Export all config as _RSA_-prefixed env vars so _run_one_subject reads
    # them from the environment.  GNU parallel inherits exported vars, so no
    # positional arg passing is needed — paths with spaces work correctly.
    local MODELS_STR
    MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")
    export _RSA_SCRIPT_DIR="$SCRIPT_DIR"
    export _RSA_CONDA_ENV="$CONDA_ENV"
    export _RSA_PREPROCESSED_DIR="$PREPROCESSED_DIR"
    export _RSA_FMRI_SUFFIX="$FMRI_SUFFIX"
    export _RSA_OUTPUT_DIR="$OUTPUT_DIR"
    export _RSA_TIMING_CSV="$TIMING_CSV"
    export _RSA_EMBEDDINGS_DIR="$EMBEDDINGS_DIR"
    export _RSA_TEMPLATE_CIFTI="$TEMPLATE_CIFTI"
    export _RSA_LEFT_SURFACE="$LEFT_SURFACE"
    export _RSA_RIGHT_SURFACE="$RIGHT_SURFACE"
    export _RSA_GLASSER_DLABEL="$GLASSER_DLABEL"
    export _RSA_WORKBENCH="$WORKBENCH"
    export _RSA_MODELS_STR="$MODELS_STR"
    export _RSA_BIN_SEC="$BIN_SEC"
    export _RSA_DELAY_SEC="$DELAY_SEC"
    export _RSA_TR="$TR"
    export _RSA_K="$K"
    export _RSA_METHOD="$METHOD"
    export _RSA_HRF="$HRF"
    export _RSA_METHOD_ARG="$METHOD_ARG"
    export _RSA_STREAM="$STREAM"
    export _RSA_CIFTI_DIR="$CIFTI_DIR"
    export _RSA_SG_FILTER="$SG_FILTER"
    export _RSA_PSC="$PSC"
    export _RSA_GSR="$GSR"
    export _RSA_MIDTHICKNESS_DIR="$MIDTHICKNESS_DIR"
    export _RSA_N_JOBS="$N_JOBS_PER_SUBJECT"

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
    neighbors)  run_precompute_neighbors ;;
    persubject) run_persubject; run_group_stats ;;
    groupstats) run_group_stats ;;
    all)        run_avg; run_persubject; run_group_stats ;;
    *)
        echo "Unknown mode: $MODE. Use: avg | neighbors | persubject | groupstats | all" >&2; exit 1 ;;
esac

log "All RSA analyses complete."
