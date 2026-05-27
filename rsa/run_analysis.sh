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
#   MODE        avg            Group-average RSA only (default)
#               preprocess     Preprocess every subject listed in SUBJECTS_LIST:
#                              per-subject CIFTIs → PREPROCESSED_INDIV_DIR
#                              group-average CIFTI → PREPROCESSED_DIR
#                              Respects SG_FILTER/PSC/GSR flags and resumes.
#               neighbors_avg  Pre-compute the geodesic k-NN cache for the
#                              group-average midthickness surface only.
#               neighbors      Pre-compute geodesic k-NN caches for the group
#                              average and every subject in SUBJECTS_LIST into
#                              GEODESIC_CACHE_DIR.  Caches are reused by any
#                              future run with k ≤ K_cached.
#               persubject     Per-subject RSA → group statistics
#               groupstats     Re-run group statistics on existing per-subject maps
#               all            avg + persubject + groupstats
#
#   METHOD      all          Searchlight + Glasser parcel RSA (default)
#               searchlight  Searchlight only
#               glasser      Glasser parcellation only
#               (ignored for 'neighbors' modes)
#
#   BATCH_SIZE  N          Number of subjects processed in parallel (default 4).
#                          In persubject mode, joblib threads per subject are
#                          set to nproc / BATCH_SIZE.
#   START_FROM  SUBID      Resume the per-subject loop starting at this subject ID.
#
# Recommended workflow
#   # 0. Preprocess all subjects + group average (skip in streaming mode)
#   bash rsa/run_analysis.sh preprocess
#
#   # 1a. Build the group-average midthickness surface from SUBJECTS_LIST
#   bash make_average.sh
#
#   # 1b. Build the group-average geodesic cache
#   bash rsa/run_analysis.sh neighbors_avg
#
#   # 1c. Build per-subject neighbour caches for every subject in SUBJECTS_LIST
#   bash rsa/run_analysis.sh neighbors all 4
#
#   # 2. Per-subject searchlight
#   bash rsa/run_analysis.sh persubject searchlight 4
#
#   # 3. Re-run group stats after adding subjects
#   bash rsa/run_analysis.sh groupstats
#
#   # Group average only
#   bash rsa/run_analysis.sh avg
#
#   # Resume per-subject from a given subject ID
#   bash rsa/run_analysis.sh persubject searchlight 4 100610
#
# Caching
#   Neighbour caches are stored as
#     {GEODESIC_CACHE_DIR}/{subject}_{hem}_neighbors_k{K_cached}.npy
#   A run requesting k will (1) use the exact-k cache if present, otherwise
#   (2) derive k from any cached k' > k, otherwise (3) run wb_command and
#   write a new cache file.  The same logic applies to group_average.
#
# Streaming vs disk mode
#   STREAM=true   — raw 7T CIFTIs are preprocessed on-the-fly (SG → PSC → GSR).
#   STREAM=false  — reads pre-saved CIFTIs from PREPROCESSED_DIR.
#
# Per-subject midthickness surfaces
#   Uses {MIDTHICKNESS_DIR}/{sub}.L/R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii
#   when available; otherwise falls back to the group-average midthickness.
#
# Subject list
#   SUBJECTS_LIST is the single authoritative roster used by every stage.
#   Subjects missing individual midthickness surfaces are tracked in
#   data/excluded.txt and removed from data/subjects.txt.
#
# Resume / skip
#   neighbors  — subjects whose both-hemisphere .npy files exist are skipped.
#   persubject — any subject/model whose output CIFTI already exists is skipped.
#   Delete the output CIFTI (or .npy) to force a rerun.
#
# GNU parallel
#   To silence the citation notice once, run: parallel --citation

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

# Streaming toggle
#   true  → preprocess raw CIFTIs from CIFTI_DIR on-the-fly
#   false → read pre-saved CIFTIs from PREPROCESSED_INDIV_DIR / PREPROCESSED_DIR
STREAM=true

# Preprocessing flags
SG_FILTER=false    # Savitzky-Golay high-pass filter
PSC=false           # percent signal change normalization
GSR=false           # global signal regression

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
# Group-average preprocessed CIFTI (output of preprocess_individual.py --save-average)
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
# Per-subject preprocessed CIFTIs (output of preprocess_individual.py --save-individual)
# File pattern: {PREPROCESSED_INDIV_DIR}/{sub}_{PREPROCESSING_FLAG}_cortex_59k.dtseries.nii
PREPROCESSED_INDIV_DIR="${DATA_BASE}/preprocessed/${PREPROCESSING_FLAG}"

# Single authoritative subject list.  All pipeline stages
# (preprocess / neighbors / searchlight) must read from here.
# Subjects without individual midthickness surfaces are tracked in
# data/excluded.txt and removed from this list.
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# Timing
TIMING_CSV="/home/amin/Research/Representation/Movie/data/movie_timing.csv"

# Embeddings root
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# CIFTI template (59k grayordinate space).
# Used only for its BrainModelAxis (grayordinate structure); the data values
# are irrelevant, so the same template is valid for every PREPROCESSING_FLAG.
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/sg_psc/group_average_sg_psc_cortex_59k.dtseries.nii"

# Surface geometry — group-average midthickness (59k)
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"

# Per-subject midthickness surfaces (used when available for the searchlight loop)
MIDTHICKNESS_DIR="${DATA_BASE}/midthickness_1.6"

# Glasser parcellation, 59k version (must match TEMPLATE_CIFTI and the surfaces)
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"

# Connectome Workbench
WORKBENCH="/opt/workbench/bin_linux64/wb_command"

# Output root (per-preprocessing)
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/${PREPROCESSING_FLAG}"

# Shared geodesic neighbour cache.  Preprocessing-agnostic and stored outside
# OUTPUT_DIR so that the k_max derivation logic in run_searchlight.py can
# reuse existing caches across all PREPROCESSING_FLAG values.
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
BIN_SEC=5.0
HRF=false
METHOD="spearman"
K=100 

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

# ── GPU acceleration ─────────────────────────────────────────────────────
# true  → GPU-batched searchlight (full-k vertices on CUDA; partial-k on CPU)
# false → CPU joblib parallelism (use when GPU unavailable or for per-subject parallel)
USE_GPU=true
GPU_BATCH_SIZE=512   # vertices per GPU batch; reduce if OOM

# ── Parallelisation ─────────────────────────────────────────────────────────
CONDA_ENV="movie"
# Default number of subjects processed in parallel.
# persubject mode divides nproc evenly across subjects for joblib threads.
DEFAULT_BATCH_SIZE=4
# Default batch size for the neighbors precompute, where wb_command is the
# bottleneck rather than Python-level threads.
DEFAULT_NEIGHBORS_BATCH_SIZE=8
# =============================================================================

MODE=${1:-avg}
METHOD_ARG=${2:-all}
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
START_FROM=${4:-""}

BIN_SEC_INT="${BIN_SEC%.*}"

N_CPUS=$(nproc 2>/dev/null || echo 8)

# In 'neighbors' mode use the neighbors-specific default when no explicit
# BATCH_SIZE was given on the command line.
if [ "$MODE" = "neighbors" ] && [ -z "${3:-}" ]; then
    BATCH_SIZE=$DEFAULT_NEIGHBORS_BATCH_SIZE
fi

# Joblib n_jobs per subject (searchlight): divide CPUs by parallel batch size.
N_JOBS_PER_SUBJECT=$(( N_CPUS / BATCH_SIZE ))
[ "$N_JOBS_PER_SUBJECT" -lt 1 ] && N_JOBS_PER_SUBJECT=1

# Pin BLAS / OMP backends to a single thread per process.  Without this, every
# joblib worker may spawn its own BLAS thread pool, oversubscribing the cores.
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
# PREPROCESSING PIPELINE
# =============================================================================
# Runs preprocess_individual.py for every subject in SUBJECTS_LIST.  Flags
# (SG_FILTER / PSC / GSR) are driven by the CONFIG section so that disk mode
# and streaming mode see identical preprocessing.
#
# Outputs:
#   Per-subject CIFTIs → PREPROCESSED_INDIV_DIR  ({sub}_{flag}_cortex_59k.dtseries.nii)
#   Group-average CIFTI → PREPROCESSED_DIR       (group_average_{flag}_cortex_59k.dtseries.nii)
#
# preprocess_individual.py resumes by skipping subjects whose outputs already
# exist; delete the output CIFTI to force a rerun.
# =============================================================================
run_preprocess() {
    log "Preprocessing $(grep -cv '^\s*#' "$SUBJECTS_LIST") subjects -> ${PREPROCESSED_INDIV_DIR}"
    log "  Flags: SG_FILTER=${SG_FILTER}  PSC=${PSC}  GSR=${GSR}  (${PREPROCESSING_FLAG})"
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

    # The group-average is written to PREPROCESSED_INDIV_DIR by default;
    # move it to PREPROCESSED_DIR so disk-mode avg RSA finds it.
    local GA_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
    local GA_TRS_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
    if [ -f "$GA_SRC" ]; then
        mkdir -p "$PREPROCESSED_DIR"
        mv -f "$GA_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
        [ -f "$GA_TRS_SRC" ] && mv -f "$GA_TRS_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
        log "  Group average moved to ${PREPROCESSED_DIR}"
    fi

    log "Preprocessing complete."
}

# =============================================================================
# GROUP-AVERAGE PIPELINE
# =============================================================================
_run_avg_one_model() {
    local MODEL_NAME="$1" MODALITIES_STR="$2"
    IFS=',' read -ra MODS <<< "$MODALITIES_STR"

    local DELAY_INT="${DELAY_SEC%.*}"
    local SL_CONFIG="k${K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}s_${METHOD}"

    for MOD in "${MODS[@]}"; do
        if ! _emb_exists "$MODEL_NAME" "$MOD"; then
            log "  skip ${MODEL_NAME}/${MOD}: embedding not found"
            continue
        fi
        log "  ${MODEL_NAME} / ${MOD}"

        # Combined dscalar accumulating searchlight + Glasser maps for this
        # config; per-k path keeps separate k runs from overwriting each other.
        # Combined dscalar in the parent directory with the full name
        local COMBINED_OUT="${OUTPUT_DIR}/group_average/${MODEL_NAME}/rsa_59k_${FMRI_SUFFIX}_k${K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${METHOD}_maps.dscalar.nii"

        if [ "$METHOD_ARG" = "all" ] || [ "$METHOD_ARG" = "searchlight" ]; then
            local GPU_FLAG="--no-gpu"; [ "$USE_GPU" = "true" ] && GPU_FLAG="--gpu"
            run_python "${SCRIPT_DIR}/run_searchlight.py" \
                --preprocessed-dir   "$PREPROCESSED_DIR" \
                --fmri-suffix        "$FMRI_SUFFIX" \
                --timing-csv         "$TIMING_CSV" \
                --embeddings-dir     "$EMBEDDINGS_DIR" \
                --template-cifti     "$TEMPLATE_CIFTI" \
                --output-dir         "$OUTPUT_DIR" \
                --subject            "group_average" \
                --model              "$MODEL_NAME" \
                --modality           "$MOD" \
                --k                  "$K" \
                --bin-sec            "$BIN_SEC" \
                --delay-sec          "$DELAY_SEC" \
                --method             "$METHOD" \
                --tr                 "$TR" \
                --left-surface       "$LEFT_SURFACE" \
                --right-surface      "$RIGHT_SURFACE" \
                --workbench          "$WORKBENCH" \
                --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
                --combined-output    "$COMBINED_OUT" \
                $GPU_FLAG --gpu-batch-size "$GPU_BATCH_SIZE" \
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
                --combined-output  "$COMBINED_OUT" \
                $(_hrf_flag)
        fi
    done
}

run_avg() {
    log "Group-average RSA (${#MODELS[@]} models, method=${METHOD_ARG})"
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_STR <<< "$MODEL_ENTRY"
        _run_avg_one_model "$MODEL_NAME" "$MODALITIES_STR"
    done
    log "Group-average RSA complete."
}

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================

# Worker exported for GNU parallel.  All config is passed via exported
# _RSA_-prefixed env vars so that paths containing spaces are not mistokenised
# by parallel; only the subject ID is passed positionally.
_run_one_subject() {
    local SUB="$1"

    local BIN_SEC_INT="${_RSA_BIN_SEC%.*}"
    local LOG_DIR="${_RSA_OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/pipeline.log"

    local HRF_FLAG=""; [ "$_RSA_HRF" = "true" ] && HRF_FLAG="--hrf"

    # Derive fmri_tag + config strings to mirror Python output naming.
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

    # Per-subject midthickness (fall back to group-average if unavailable).
    local LEFT_SURF="$_RSA_LEFT_SURFACE"
    local RIGHT_SURF="$_RSA_RIGHT_SURFACE"
    local SUB_L="${_RSA_MIDTHICKNESS_DIR}/${SUB}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    local SUB_R="${_RSA_MIDTHICKNESS_DIR}/${SUB}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    if [ -f "$SUB_L" ] && [ -f "$SUB_R" ]; then
        LEFT_SURF="$SUB_L"
        RIGHT_SURF="$SUB_R"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: per-subject midthickness not found; using group-average surface" \
            | tee -a "$LOG"
    fi

    # fMRI input flags.
    local FMRI_FLAGS
    if [ "$_RSA_STREAM" = "true" ]; then
        local SG_FLAG="";  [ "$_RSA_SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        local PSC_FLAG=""; [ "$_RSA_PSC"       = "true" ] && PSC_FLAG="--psc"
        local GSR_FLAG="--gsr"; [ "$_RSA_GSR" = "false" ] && GSR_FLAG="--no-gsr"
        FMRI_FLAGS="--raw-dir ${_RSA_CIFTI_DIR} $SG_FLAG $PSC_FLAG $GSR_FLAG"
    else
        local FMRI_PATH="${_RSA_PREPROCESSED_DIR}/${SUB}_${_RSA_FMRI_SUFFIX}_cortex_59k.dtseries.nii"
        if [ ! -f "$FMRI_PATH" ]; then
            echo "[$(date +%H:%M:%S)] ${SUB}: preprocessed CIFTI not found; run 'preprocess' mode first" \
                | tee -a "$LOG"
            return 1
        fi
        FMRI_FLAGS="--preprocessed-dir ${_RSA_PREPROCESSED_DIR} --fmri-suffix ${_RSA_FMRI_SUFFIX}"
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} (stream=${_RSA_STREAM})" | tee -a "$LOG"

    # Aggregate exit status across all model/modality runs.  Do NOT replace
    # this with `local STATUS=$?` after the loops: a false-condition `if`
    # block with no `else` resets $? to 0 and would mask real failures.
    local STATUS=0

    IFS=';' read -ra MODEL_ENTRIES <<< "$_RSA_MODELS_STR"
    for MODEL_ENTRY in "${MODEL_ENTRIES[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"

        for MOD in "${MODS[@]}"; do
            local EMB="${_RSA_EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_SEC_INT}s/${MODEL_NAME}_${MOD}.npy"
            if [ ! -f "$EMB" ]; then
                echo "[$(date +%H:%M:%S)] ${SUB}: skip ${MODEL_NAME}/${MOD}: embedding not found" \
                    | tee -a "$LOG"
                continue
            fi

            # Combined dscalar accumulating searchlight + Glasser maps for this
            # subject/model/config; per-k path keeps separate k runs distinct.
            # Combined dscalar in the parent directory with the full name
            local COMBINED_OUT="${_RSA_OUTPUT_DIR}/${SUB}/${MODEL_NAME}/rsa_59k_${FMRI_TAG_LOCAL}_k${_RSA_K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${_RSA_METHOD}_maps.dscalar.nii"
            if [ "$_RSA_METHOD_ARG" = "all" ] || [ "$_RSA_METHOD_ARG" = "searchlight" ]; then
                # Update SL_OUT to include _searchlight
                local SL_OUT="${_RSA_OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${SL_CONFIG}/rsa_59k_${FMRI_TAG_LOCAL}_k${_RSA_K}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${_RSA_METHOD}_searchlight.dscalar.nii"
                # Skip only when both the individual file and the combined file
                # exist; the individual file alone implies the combined output
                # still needs updating.
                if [ -f "$SL_OUT" ] && [ -f "$COMBINED_OUT" ]; then
                    echo "[$(date +%H:%M:%S)] ${SUB}: searchlight ${MODEL_NAME}/${MOD} already complete; skipping" \
                        | tee -a "$LOG"
                else
                    local GPU_FLAG="--no-gpu"   # per-subject: disable GPU to allow parallel CPU jobs
                    # shellcheck disable=SC2086
                    conda run --no-capture-output -n "$_RSA_CONDA_ENV" python \
                        "${_RSA_SCRIPT_DIR}/run_searchlight.py" \
                        $FMRI_FLAGS \
                        --timing-csv         "$_RSA_TIMING_CSV" \
                        --embeddings-dir     "$_RSA_EMBEDDINGS_DIR" \
                        --template-cifti     "$_RSA_TEMPLATE_CIFTI" \
                        --output-dir         "$_RSA_OUTPUT_DIR" \
                        --subject            "$SUB" \
                        --model              "$MODEL_NAME" \
                        --modality           "$MOD" \
                        --k                  "$_RSA_K" \
                        --bin-sec            "$_RSA_BIN_SEC" \
                        --delay-sec          "$_RSA_DELAY_SEC" \
                        --method             "$_RSA_METHOD" \
                        --tr                 "$_RSA_TR" \
                        --left-surface       "$LEFT_SURF" \
                        --right-surface      "$RIGHT_SURF" \
                        --workbench          "$_RSA_WORKBENCH" \
                        --geodesic-cache-dir "$_RSA_GEODESIC_CACHE_DIR" \
                        --combined-output    "$COMBINED_OUT" \
                        $GPU_FLAG --gpu-batch-size 512 \
                        $HRF_FLAG \
                        >> "$LOG" 2>&1 || STATUS=$?
                fi
            fi

            if [ "$_RSA_METHOD_ARG" = "all" ] || [ "$_RSA_METHOD_ARG" = "glasser" ]; then
                # Combined dscalar in the parent directory with the full name
                local GL_OUT="${_RSA_OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${GL_CONFIG}/rsa_59k_${FMRI_TAG_LOCAL}_delay${DELAY_INT}s_bin${BIN_SEC_INT}_${_RSA_METHOD}_glasser.dscalar.nii"
                local GL_REPORT="${_RSA_OUTPUT_DIR}/${SUB}/${MODEL_NAME}/${GL_CONFIG}/ranked_report.csv"
                if [ -f "$GL_OUT" ] && [ -f "$GL_REPORT" ] && [ -f "$COMBINED_OUT" ]; then
                    echo "[$(date +%H:%M:%S)] ${SUB}: Glasser ${MODEL_NAME}/${MOD} already complete; skipping" \
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
                        --combined-output  "$COMBINED_OUT" \
                        $HRF_FLAG \
                        >> "$LOG" 2>&1 || STATUS=$?
                fi
            fi
        done
    done

    # Remove any partial per-subject dconn files left in the cache directory.
    # run_searchlight.py deletes these after extracting k-NN, but a crash mid-
    # run can leave them behind.
    rm -f "${_RSA_GEODESIC_CACHE_DIR}/${SUB}_left_geodesic.dconn.nii"
    rm -f "${_RSA_GEODESIC_CACHE_DIR}/${SUB}_right_geodesic.dconn.nii"

    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} complete" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB} failed (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_subject

# =============================================================================
# NEIGHBOUR PRE-COMPUTE PIPELINE
# =============================================================================

# _find_kmax_npy CACHE_DIR SUBJECT HEM K_REQ
# Print the path to the smallest cache file with k > K_REQ, or "" if none.
# Cache filenames follow the pattern: {subject}_{hem}_neighbors_k{K}.npy
_find_kmax_npy() {
    local cache_dir="$1" subject="$2" hem="$3" k_req="$4"
    local best_k=0 best_f=""
    for f in "${cache_dir}/${subject}_${hem}_neighbors_k"*.npy; do
        [ -f "$f" ] || continue
        local kc
        kc=$(basename "$f" | grep -oP '(?<=_k)\d+(?=\.npy)') || continue
        [ -z "$kc" ] && continue
        if [ "$kc" -gt "$k_req" ] && { [ "$best_k" -eq 0 ] || [ "$kc" -lt "$best_k" ]; }; then
            best_k="$kc"; best_f="$f"
        fi
    done
    echo "$best_f"
}
export -f _find_kmax_npy

# Worker exported for GNU parallel.  Per-hemisphere status is evaluated with
# the same three-level convention used by Python get_neighbors():
#   exact         -- exact-k cache file present
#   kmax:<file>   -- a cache with k' > k is present; Python derives k on demand
#   missing       -- neither; wb_command must run for that hemisphere
# precompute_neighbors.py inspects each hemisphere internally, so the shell
# only needs to short-circuit when both hemispheres are already covered.
_run_one_neighbors() {
    local SUB="$1"

    local CACHE_DIR="${_RSA_GEODESIC_CACHE_DIR}"

    _sub_hem_status() {
        local hem="$1"
        local exact="${CACHE_DIR}/${SUB}_${hem}_neighbors_k${_RSA_K}.npy"
        if [ -f "$exact" ]; then
            echo "exact"
            return
        fi
        local kmax
        kmax=$(_find_kmax_npy "$CACHE_DIR" "$SUB" "$hem" "$_RSA_K")
        if [ -n "$kmax" ]; then
            echo "kmax:$(basename "$kmax")"
            return
        fi
        echo "missing"
    }

    local STATUS_L STATUS_R
    STATUS_L=$(_sub_hem_status "left")
    STATUS_R=$(_sub_hem_status "right")

    if [[ "$STATUS_L" != "missing" ]] && [[ "$STATUS_R" != "missing" ]]; then
        echo "[$(date +%H:%M:%S)] ${SUB}: L=${STATUS_L}, R=${STATUS_R}; skipping wb_command"
        return 0
    fi

    local LOG_DIR="${_RSA_OUTPUT_DIR}/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/neighbors.log"

    local LEFT_SURF="$_RSA_LEFT_SURFACE"
    local RIGHT_SURF="$_RSA_RIGHT_SURFACE"
    local SUB_L="${_RSA_MIDTHICKNESS_DIR}/${SUB}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    local SUB_R="${_RSA_MIDTHICKNESS_DIR}/${SUB}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    if [ -f "$SUB_L" ] && [ -f "$SUB_R" ]; then
        LEFT_SURF="$SUB_L"
        RIGHT_SURF="$SUB_R"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: per-subject midthickness not found; using group-average surface" \
            | tee -a "$LOG"
    fi

    echo "[$(date +%H:%M:%S)] ${SUB}: L=${STATUS_L}, R=${STATUS_R}; computing k=${_RSA_K} neighbour cache for missing hemisphere(s)" \
        | tee -a "$LOG"

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
        echo "[$(date +%H:%M:%S)] ${SUB}: neighbour cache complete" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB}: neighbour cache failed (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_neighbors

# =============================================================================
# NEIGHBOUR PRE-COMPUTE — GROUP-AVERAGE ONLY
# =============================================================================
# Builds the geodesic k-NN cache for the group-average midthickness surface.
# Run this after make_average.sh (re)generates the group-average surface and
# before the first 'avg' RSA run.  Uses the K and GEODESIC_CACHE_DIR values
# defined in the CONFIG section.
#
#   bash rsa/run_analysis.sh neighbors_avg
#
# The full 'neighbors' mode invokes this first and then adds per-subject
# caches; use 'neighbors_avg' to build only the group-average cache.
# =============================================================================
run_precompute_neighbors_avg() {
    log "Group-average k=${K} neighbour cache"
    log "  Surface L: ${LEFT_SURFACE}"
    log "  Surface R: ${RIGHT_SURFACE}"
    log "  Cache dir: ${GEODESIC_CACHE_DIR}"

    mkdir -p "$GEODESIC_CACHE_DIR"

    # Evaluate each hemisphere independently so a partial cache (one
    # hemisphere computed, the other not) is handled correctly.
    # Status: "exact" | "kmax:<file>" | "missing"
    _hem_status() {
        local subject="$1" hem="$2" k_req="$3"
        local exact="${GEODESIC_CACHE_DIR}/${subject}_${hem}_neighbors_k${k_req}.npy"
        if [ -f "$exact" ]; then
            echo "exact"
            return
        fi
        local kmax
        kmax=$(_find_kmax_npy "$GEODESIC_CACHE_DIR" "$subject" "$hem" "$k_req")
        if [ -n "$kmax" ]; then
            echo "kmax:$(basename "$kmax")"
            return
        fi
        echo "missing"
    }

    local STATUS_L STATUS_R
    STATUS_L=$(_hem_status "group_average" "left"  "$K")
    STATUS_R=$(_hem_status "group_average" "right" "$K")

    log "  left  hemisphere: ${STATUS_L}"
    log "  right hemisphere: ${STATUS_R}"

    if [[ "$STATUS_L" != "missing" ]] && [[ "$STATUS_R" != "missing" ]]; then
        if [[ "$STATUS_L" == "exact" ]] && [[ "$STATUS_R" == "exact" ]]; then
            log "  Both hemispheres: exact k=${K} cache present."
        else
            log "  Both hemispheres covered by a larger-k cache;"
            log "  k=${K} will be derived on the first avg RSA run."
        fi
        log "Group-average neighbour cache complete."
        return 0
    fi

    # precompute_neighbors.py checks each hemisphere internally and only
    # runs wb_command for the missing one(s).
    local MISSING_HEMS=()
    [[ "$STATUS_L" == "missing" ]] && MISSING_HEMS+=("left")
    [[ "$STATUS_R" == "missing" ]] && MISSING_HEMS+=("right")
    log "  Computing missing hemisphere(s): ${MISSING_HEMS[*]}"

    conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/precompute_neighbors.py" \
        --subject       "group_average" \
        --left-surface  "$LEFT_SURFACE" \
        --right-surface "$RIGHT_SURFACE" \
        --workbench     "$WORKBENCH" \
        --cache-dir     "$GEODESIC_CACHE_DIR" \
        --k             "$K"

    log "Group-average neighbour cache complete."
}

run_precompute_neighbors() {
    log "Neighbour pre-compute (k=${K}, ${SUBJECTS_LIST})"
    log "  Cache dir: ${GEODESIC_CACHE_DIR}  (shared across all PREPROCESSING_FLAG values)"
    log "  Subjects with both-hemisphere .npy caches are skipped."

    # Group-average cache.
    run_precompute_neighbors_avg

    # Per-subject caches.
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

    local N_CACHED=0
    for SUB in $SUBJECTS; do
        [ -f "${GEODESIC_CACHE_DIR}/${SUB}_left_neighbors_k${K}.npy" ] && \
        [ -f "${GEODESIC_CACHE_DIR}/${SUB}_right_neighbors_k${K}.npy" ] && \
        N_CACHED=$(( N_CACHED + 1 ))
    done
    log "  ${N_TOTAL} subjects: ${N_CACHED} already cached, $((N_TOTAL - N_CACHED)) to compute (BATCH_SIZE=${BATCH_SIZE})"

    # Export env vars for the worker
    export _RSA_SCRIPT_DIR="$SCRIPT_DIR"
    export _RSA_CONDA_ENV="$CONDA_ENV"
    export _RSA_OUTPUT_DIR="$OUTPUT_DIR"
    export _RSA_GEODESIC_CACHE_DIR="$GEODESIC_CACHE_DIR"
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
    log "  Processing ${N_TOTAL} subjects from ${SUBJECTS_LIST} ..."

    # Export all config as _RSA_-prefixed env vars so _run_one_subject reads
    # them from the environment.  GNU parallel inherits exported vars, so no
    # positional arg passing is needed — paths with spaces work correctly.
    local MODELS_STR
    MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")
    export _RSA_SCRIPT_DIR="$SCRIPT_DIR"
    export _RSA_CONDA_ENV="$CONDA_ENV"
    export _RSA_PREPROCESSED_DIR="$PREPROCESSED_INDIV_DIR"
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
    export _RSA_GEODESIC_CACHE_DIR="$GEODESIC_CACHE_DIR"
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
    preprocess)     run_preprocess ;;
    avg)            run_avg ;;
    neighbors_avg)  run_precompute_neighbors_avg ;;
    neighbors)      run_precompute_neighbors ;;
    persubject)     run_persubject; run_group_stats ;;
    groupstats)     run_group_stats ;;
    all)            run_avg; run_persubject; run_group_stats ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: preprocess | avg | neighbors_avg | neighbors | persubject | groupstats | all" >&2
        exit 1 ;;
esac

log "All RSA analyses complete."
