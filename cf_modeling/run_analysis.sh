#!/usr/bin/env bash
# cf_modeling/run_analysis.sh
# ============================
# Master runner for CF (cortical field) modeling analyses.
# Supports group-average and per-subject modes with GNU parallel.
#
# Usage
# -----
#   bash cf_modeling/run_analysis.sh [MODE] [BATCH_SIZE] [START_FROM]
#
#   MODE        preprocess   Preprocess all 175 subjects from SUBJECTS_LIST:
#                            per-subject CIFTIs → PREPROCESSED_INDIV_DIR
#                            group-average CIFTI → PREPROCESSED_DIR
#                            Respects SG_FILTER/PSC/ flags and resumes.
#               avg          Group-average CF modeling for all ROI pairs
#               persubject   Per-subject CF modeling → group stats
#               all          avg + persubject  (default)
#
#   BATCH_SIZE  N            Parallel subjects per ROI pair (default 8)
#   START_FROM  SUBID        Resume per-subject from this subject ID
#
# Recommended workflow
#   # 0. Preprocess all 175 subjects (skip if using streaming mode)
#   bash cf_modeling/run_analysis.sh preprocess
#
#   # 1. Group-average CF modeling
#   bash cf_modeling/run_analysis.sh avg
#
#   # 2. Per-subject CF modeling
#   bash cf_modeling/run_analysis.sh persubject 8
#
# Disk mode (default, STREAM=false)
#   Reads pre-saved preprocessed CIFTIs from PREPROCESSED_INDIV_DIR.
#   Run 'preprocess' mode first.
#
# Streaming mode (STREAM=true)
#   Preprocesses raw 7T CIFTIs on-the-fly; no CIFTI is saved.
#   Set CIFTI_DIR below and configure SG_FILTER/PSC/Z_SCORE.
#
# Excluded subjects
#   Subjects without individual midthickness surfaces are listed in
#   data/excluded.txt and have been removed from data/subjects.txt.
#   175 subjects remain.
#
# Resume / skip
#   persubject: skips any subject whose R2_nc maps already exist.
#   Delete the output .npy files to force a rerun.
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
PYCORTEX_STORE="${DATA_BASE}/hedger2026"
export PYCORTEX_FILESTORE="$PYCORTEX_STORE"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# ── Streaming toggle ──────────────────────────────────────────────────────────
# false → disk mode (read pre-saved preprocessed CIFTIs from PREPROCESSED_INDIV_DIR)
# true  → streaming mode (preprocess raw CIFTIs from CIFTI_DIR on-the-fly)
STREAM=false

# Preprocessing flags
SG_FILTER=true    # Savitzky-Golay high-pass filter
PSC=true          # Percent signal change normalization



# Automatically build PREPROCESSING_FLAG from SG_FILTER/PSC
# (Z_SCORE is NOT included — it is applied inside the Python analysis script)
PREP_PARTS=()
[ "$SG_FILTER" = "true" ] && PREP_PARTS+=("sg")
[ "$PSC"       = "true" ] && PREP_PARTS+=("psc")


if [ ${#PREP_PARTS[@]} -eq 0 ]; then
    PREPROCESSING_FLAG="raw"
else
    PREPROCESSING_FLAG=$(IFS=_; echo "${PREP_PARTS[*]}")
fi

# fMRI data paths
FMRI_SUFFIX="${PREPROCESSING_FLAG}"
# Group-average preprocessed CIFTI (output of preprocess_individual.py --save-average)
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
# Per-subject preprocessed CIFTIs (output of preprocess_individual.py --save-individual)
# File pattern: {PREPROCESSED_INDIV_DIR}/{sub}_{PREPROCESSING_FLAG}_cortex_59k.dtseries.nii
PREPROCESSED_INDIV_DIR="${DATA_BASE}/preprocessed/${PREPROCESSING_FLAG}"

# Group-average CIFTI — used as fMRI input and CIFTI template for group-average mode
FMRI_GROUP_CIFTI="${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"

# Single authoritative subject list — 175 subjects with full 7T fMRI + midthickness.
# All pipeline stages (preprocess / cf_modeling) must read from here.
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"


# Glasser HCP-MMP1 59k_fs_LR dlabel (used by extract_geometry.py)
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"

# Output and RSA roots
OUTPUT_BASE="${OUTPUTS_BASE}/cf_modeling"
RSA_BASE="${OUTPUTS_BASE}/rsa"

# ── Parallelisation ─────────────────────────────────────────────────────────
CONDA_ENV="vicsompy_av"
DEFAULT_BATCH_SIZE=8    # 8 jobs × 2 BLAS threads ≈ 1 job per physical core

# ── Analysis ROI pairs ───────────────────────────────────────────────────────
# Format: "ROI_A:ROI_B"
# Per-subject pairs (computationally expensive; parallelised across subjects)
PERSUBJECT_PAIRS=(
    "A5:FFC"
    "V1:3b"
    "TA2:MST"
)
# Group-average pairs
AVG_PAIRS=(
    # "A1:V1"
    "A5:FFC"
    "3b:V1"
)
# =============================================================================

MODE=${1:-all}
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
START_FROM=${3:-""}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================
run_preprocess() {
    log "=== Preprocessing n=$(grep -cv '^\s*#' "$SUBJECTS_LIST") subjects → ${PREPROCESSED_INDIV_DIR} ==="
    log "  Flags: SG_FILTER=${SG_FILTER}  PSC=${PSC}  (${PREPROCESSING_FLAG})"
    log "  Note: Z_SCORE is applied inside run_cfmodeling.py, not during preprocessing"
    log "  Subjects: ${SUBJECTS_LIST}"
    log "  Raw CIFTI dir: ${CIFTI_DIR}"

    local SG_FLAG="" PSC_FLAG="" 
    [ "$SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
    [ "$PSC"       = "true" ] && PSC_FLAG="--psc" 

    run_python "${SCRIPT_DIR}/../preprocess_individual.py" \
        --raw-dir        "$CIFTI_DIR" \
        --out-dir        "$PREPROCESSED_INDIV_DIR" \
        --subjects-list  "$SUBJECTS_LIST" \
        --tr             1.0 \
        $SG_FLAG $PSC_FLAG \
        --save-individual \
        --save-average

    # Move group average to PREPROCESSED_DIR so disk-mode avg CF modeling finds it
    local GA_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
    local GA_TRS_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
    if [ -f "$GA_SRC" ]; then
        mkdir -p "$PREPROCESSED_DIR"
        mv -f "$GA_SRC"     "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
        [ -f "$GA_TRS_SRC" ] && \
        mv -f "$GA_TRS_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
        log "  Group average moved → ${PREPROCESSED_DIR}"
    fi

    log "=== Preprocessing complete ==="
}

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================

# Worker function — exported for GNU parallel.
# ROI_A and ROI_B are positional (they vary per ROI-pair call).
# All other config is read from exported _CF_-prefixed env vars.
#
# Args: SUB ROI_A ROI_B
_run_one_subject() {
    local SUB="$1"
    local ROI_A="$2"
    local ROI_B="$3"

    local OUT_DIR="${_CF_OUTPUT_BASE}/per_subject/${ROI_A}_${ROI_B}/subjects/${SUB}"
    mkdir -p "$OUT_DIR"
    local LOG="${OUT_DIR}/pipeline.log"

    # Skip if null-corrected R² maps already exist
    if [ -f "${OUT_DIR}/R2_${ROI_A}_nc.npy" ] && \
       [ -f "${OUT_DIR}/R2_${ROI_B}_nc.npy" ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} ${ROI_A}×${ROI_B}: already done — skipping"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} ${ROI_A}×${ROI_B} (stream=${_CF_STREAM})" \
        | tee -a "$LOG"

    local STATUS=0
    if [ "$_CF_STREAM" = "true" ]; then
        local SG_FLAG="";   [ "$_CF_SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        local PSC_FLAG="";  [ "$_CF_PSC"       = "true" ] && PSC_FLAG="--psc"
        
        

        # shellcheck disable=SC2086
        conda run --no-capture-output -n "$_CF_CONDA_ENV" python \
            "${_CF_SCRIPT_DIR}/run_cfmodeling.py" \
            --mode       per_subject \
            --roi-a      "$ROI_A" \
            --roi-b      "$ROI_B" \
            --subject    "$SUB" \
            --raw-dir    "$_CF_CIFTI_DIR" \
            --output-base "$_CF_OUTPUT_BASE" \
            
            >> "$LOG" 2>&1 || STATUS=$?
    else
        local FMRI_PATH="${_CF_PREPROCESSED_INDIV_DIR}/${SUB}_${_CF_FMRI_SUFFIX}_cortex_59k.dtseries.nii"
        if [ ! -f "$FMRI_PATH" ]; then
            echo "[$(date +%H:%M:%S)] ${SUB}: no preprocessed CIFTI — run preprocess mode first" \
                | tee -a "$LOG"
            return 1
        fi

        conda run --no-capture-output -n "$_CF_CONDA_ENV" python \
            "${_CF_SCRIPT_DIR}/run_cfmodeling.py" \
            --mode             per_subject \
            --roi-a            "$ROI_A" \
            --roi-b            "$ROI_B" \
            --subject          "$SUB" \
            --preprocessed-dir "$_CF_PREPROCESSED_INDIV_DIR" \
            --fmri-suffix      "$_CF_FMRI_SUFFIX" \
            --output-base      "$_CF_OUTPUT_BASE" \
            >> "$LOG" 2>&1 || STATUS=$?
    fi

    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} ${ROI_A}×${ROI_B} DONE" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB} ${ROI_A}×${ROI_B} FAILED (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_subject

run_persubject_pair() {
    local ROI_A="$1"
    local ROI_B="$2"
    local OUT="${OUTPUT_BASE}/per_subject/${ROI_A}_${ROI_B}"

    local MODE_TAG
    [ "$STREAM" = "true" ] && MODE_TAG="streaming" || MODE_TAG="disk"
    log "======================================================"
    log "  Per-subject: ${ROI_A} × ${ROI_B}  [${MODE_TAG}]"
    log "  Subjects: ${SUBJECTS_LIST}"
    log "  Output:   ${OUT}"
    log "  Parallel: ${BATCH_SIZE} jobs"
    log "======================================================"

    # ── Step 01: geometry + LBOEs (one-time, cached) ─────────────────────────
    local SUB_A_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_A" | tr '[:upper:]' '[:lower:]').pkl"
    local SUB_B_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_B" | tr '[:upper:]' '[:lower:]').pkl"

    if [ -f "$SUB_A_PKL" ] && [ -f "$SUB_B_PKL" ]; then
        log "[01] Subsurfaces cached — skipping"
    else
        log "[01] Building ${ROI_A} + ${ROI_B} subsurfaces + LBOEs (59k_fs_LR) ..."
        run_python "${SCRIPT_DIR}/extract_geometry.py" \
            --mode           per_subject \
            --roi_a          "$ROI_A" \
            --roi_b          "$ROI_B" \
            --pycortex_store "$PYCORTEX_STORE" \
            --glasser_dlabel "$GLASSER_DLABEL" \
            --output_base    "$OUTPUT_BASE"
        log "[01] Done"
    fi

    # ── Steps 02-04: per-subject in parallel ──────────────────────────────────
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
    log "[02-04] Processing ${N_TOTAL} subjects from ${SUBJECTS_LIST} (${BATCH_SIZE} parallel, ${MODE_TAG}) ..."

    export _CF_SCRIPT_DIR="$SCRIPT_DIR"
    export _CF_CONDA_ENV="$CONDA_ENV"
    export _CF_OUTPUT_BASE="$OUTPUT_BASE"
    export _CF_PREPROCESSED_INDIV_DIR="$PREPROCESSED_INDIV_DIR"
    export _CF_FMRI_SUFFIX="$FMRI_SUFFIX"
    export _CF_STREAM="$STREAM"
    export _CF_CIFTI_DIR="$CIFTI_DIR"
    export _CF_SG_FILTER="$SG_FILTER"
    export _CF_PSC="$PSC"
    
    export _CF_Z_SCORE="$Z_SCORE"

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_subject {} "$ROI_A" "$ROI_B"
    else
        log "GNU parallel not found — running sequentially"
        log "  (install with: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB" "$ROI_A" "$ROI_B"
        done
    fi
    log "[02-04] All subjects done"

    # ── Step 05: aggregate + integration maps ────────────────────────────────
    log "[05] Aggregating subjects → integration maps ..."
    run_python "${SCRIPT_DIR}/integration_maps.py" \
        --mode           per_subject \
        --roi_a          "$ROI_A" \
        --roi_b          "$ROI_B" \
        --pycortex_store "$PYCORTEX_STORE" \
        --output_base    "$OUTPUT_BASE" \
        --template_cifti "$FMRI_GROUP_CIFTI"
    log "[05] Done"

    # ── Step 06: group statistics ─────────────────────────────────────────────
    log "[06] Group statistics ..."
    run_python "${SCRIPT_DIR}/overlap.py" \
        --mode           per_subject \
        --roi_a         "$ROI_A" \
        --roi_b         "$ROI_B" \
        --output_base   "$OUTPUT_BASE" \
        --template_cifti "$FMRI_GROUP_CIFTI"
    log "[06] Done"

    log "Per-subject ${ROI_A}×${ROI_B} complete → ${OUT}"
}

run_persubject() {
    log "=== Per-subject CF modeling (${#PERSUBJECT_PAIRS[@]} ROI pairs) ==="
    for PAIR in "${PERSUBJECT_PAIRS[@]}"; do
        IFS=':' read -r ROI_A ROI_B <<< "$PAIR"
        run_persubject_pair "$ROI_A" "$ROI_B"
    done
    log "=== Per-subject CF modeling complete ==="
}

# =============================================================================
# GROUP-AVERAGE PIPELINE
# =============================================================================
run_avg_pair() {
    local ROI_A="$1"
    local ROI_B="$2"
    local OUT="${OUTPUT_BASE}/group_average/${ROI_A}_${ROI_B}"

    log "======================================================"
    log "  Group-average: ${ROI_A} × ${ROI_B}"
    log "  fMRI: ${FMRI_GROUP_CIFTI}"
    log "  Output: ${OUT}"
    log "======================================================"

    # ── Step 01: geometry + LBOEs (one-time, cached) ─────────────────────────
    local SUB_A_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_A" | tr '[:upper:]' '[:lower:]').pkl"
    local SUB_B_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_B" | tr '[:upper:]' '[:lower:]').pkl"

    if [ -f "$SUB_A_PKL" ] && [ -f "$SUB_B_PKL" ]; then
        log "[01] Subsurfaces cached — skipping"
    else
        log "[01] Building ${ROI_A} + ${ROI_B} subsurfaces + LBOEs (59k_fs_LR) ..."
        run_python "${SCRIPT_DIR}/extract_geometry.py" \
            --mode          group_average \
            --roi_a         "$ROI_A" \
            --roi_b         "$ROI_B" \
            --pycortex_store "$PYCORTEX_STORE" \
            --glasser_dlabel "$GLASSER_DLABEL" \
            --output_base   "$OUTPUT_BASE"
        log "[01] Done"
    fi

    log "[02-04] Prepare data, fit banded ridge, null-correct ..."
    run_python "${SCRIPT_DIR}/run_cfmodeling.py" \
        --mode             group_average \
        --roi-a            "$ROI_A" \
        --roi-b            "$ROI_B" \
        --output-base      "$OUTPUT_BASE" \
        --preprocessed-dir "$PREPROCESSED_DIR" \
        --fmri-suffix      "$FMRI_SUFFIX" \
        --template-cifti   "$FMRI_GROUP_CIFTI"
    log "[02-04] Done"

    log "[05] Integration maps ..."
    run_python "${SCRIPT_DIR}/integration_maps.py" \
        --mode           group_average \
        --roi_a          "$ROI_A" \
        --roi_b          "$ROI_B" \
        --pycortex_store "$PYCORTEX_STORE" \
        --output_base    "$OUTPUT_BASE" \
        --template_cifti "$FMRI_GROUP_CIFTI"
    log "[05] Done"

    log "[06] RSA spatial overlap ..."
    run_python "${SCRIPT_DIR}/overlap.py" \
        --mode          group_average \
        --roi_a         "$ROI_A" \
        --roi_b         "$ROI_B" \
        --output_base   "$OUTPUT_BASE" \
        --template_cifti "$FMRI_GROUP_CIFTI" \
        --rsa_base      "$RSA_BASE"
    log "[06] Done"

    log "Group-average ${ROI_A}×${ROI_B} complete → ${OUT}"
}

run_avg() {
    log "=== Group-average CF modeling (${#AVG_PAIRS[@]} ROI pairs) ==="
    for PAIR in "${AVG_PAIRS[@]}"; do
        IFS=':' read -r ROI_A ROI_B <<< "$PAIR"
        run_avg_pair "$ROI_A" "$ROI_B"
    done
    log "=== Group-average CF modeling complete ==="
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    preprocess)              run_preprocess ;;
    avg|groupaverage)        run_avg ;;
    persubject)              run_persubject ;;
    all)                     run_avg; run_persubject ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: preprocess | avg | persubject | all" >&2
        exit 1 ;;
esac

log "All CF modeling analyses complete."
