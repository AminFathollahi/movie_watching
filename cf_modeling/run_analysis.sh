#!/bin/bash
# cf_modeling/run_analysis.sh
# ===========================
# Master runner for CF modeling (group_average and per_subject).
#
# Prerequisites
# -------------
#   1. conda activate cfmod
#   2. GNU parallel: conda install -c conda-forge parallel
#
# DISK MODE (default, STREAM=false):
#   Requires pre-saved preprocessed CIFTIs in FMRI_OUT_DIR.
#   Runs scripts 02 → 03 → 04 sequentially per subject.
#
# STREAMING MODE (set STREAM=true below):
#   Preprocesses raw CIFTIs on-the-fly (full-run mode per vicsompy convention).
#   No preprocessed CIFTI or X/Y arrays are saved; only R2_nc maps are written.
#   Uses run_subject_stream.py (combines 02+03+04 in one Python process).
#   Set CIFTI_DIR to the raw 7T CIFTI directory and configure preprocessing flags.
#
# Usage
# -----
#   bash run_analysis.sh                        # all analyses (both modes)
#   bash run_analysis.sh persubject             # per-subject pipeline only
#   bash run_analysis.sh groupaverage           # group-average pipeline only
#   bash run_analysis.sh persubject 8           # override parallel batch size
#   bash run_analysis.sh persubject 8 100610    # resume from subject ID

set -euo pipefail

# =============================================================================
# PATHS  — edit these for your system
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data/Setareh"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

# Raw 7T CIFTI files (used for streaming mode and subject discovery)
CIFTI_DIR="/media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI"

# ── Streaming toggle ──────────────────────────────────────────────────────────
# false → disk mode: reads pre-saved preprocessed CIFTIs from FMRI_OUT_DIR
# true  → streaming mode: preprocesses raw CIFTIs on-the-fly, no CIFTI saved
STREAM=false

# Preprocessing flags (streaming mode only; ignored in disk mode)
# These must match the full-run preprocessing convention (SG→PSC→GSR→zscore per run)
SG_FILTER=true    # --sg_filter / "" (Savitzky-Golay high-pass)
PSC=true          # --psc / ""       (percent signal change)
GSR=true          # --gsr / --no-gsr
Z_SCORE=true      # --z_score / --no-z_score

# Preprocessed CIFTI directory (disk mode only — output of preprocess_individual.py)
FMRI_OUT_DIR="/home/amin/Research/Representation/Movie/outputs/preprocessed"

# Preprocessing suffix (disk mode only — must match what preprocess_individual.py used)
# Default: --sg-filter --psc --gsr --z-score  →  sg_psc_gsr_zscore
FMRI_SUFFIX="sg_psc_gsr_zscore"

# Derived paths — group-average CIFTI is used as template for CIFTI saving
FMRI_GROUP_CIFTI="${FMRI_OUT_DIR}/group_average_${FMRI_SUFFIX}_cortex_59k.dtseries.nii"

# Glasser HCP-MMP1 59k_fs_LR dlabel (both modes use same parcellation)
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"

OUTPUT_BASE="/home/amin/Research/Representation/Movie/outputs/cf_modeling"
RSA_BASE="/home/amin/Research/Representation/Movie/outputs/searchlight_rsa_output"

CONDA_ENV="cfmod"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_BATCH_SIZE=8    # 8 jobs × 2 BLAS threads ≈ 1 job per physical core

# =============================================================================
# CLI
# =============================================================================
MODE=${1:-"all"}               # all | persubject | groupaverage
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
START_FROM=${3:-""}            # resume from this subject ID (per-subject only)

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() {
    conda run -n "$CONDA_ENV" python "$@"
}

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================

# Per-subject worker — called by GNU parallel.
# Disk mode: runs scripts 02 → 03 → 04 for one subject.
# Streaming mode: runs run_subject_stream.py (02+03+04 in one Python process,
#                 no X/Y arrays or preprocessed CIFTI saved).
# Skips if R2_nc maps already exist.
# Args: SUB SCRIPT_DIR CONDA_ENV OUTPUT_BASE ROI_A ROI_B
#       FMRI_OUT_DIR FMRI_SUFFIX STREAM CIFTI_DIR SG_FILTER PSC GSR Z_SCORE
_run_one_subject() {
    local SUB="$1"
    local SCRIPT_DIR="$2"
    local CONDA_ENV="$3"
    local OUTPUT_BASE="$4"
    local ROI_A="$5"
    local ROI_B="$6"
    local FMRI_OUT_DIR="$7"
    local FMRI_SUFFIX="$8"
    local STREAM="${9}"
    local CIFTI_DIR="${10}"
    local SG_FILTER="${11}"
    local PSC="${12}"
    local GSR="${13}"
    local Z_SCORE="${14}"

    local OUT_DIR="${OUTPUT_BASE}/per_subject/${ROI_A}_${ROI_B}/subjects/${SUB}"
    mkdir -p "$OUT_DIR"
    local LOG="${OUT_DIR}/pipeline.log"

    # Skip if null-corrected R² maps already exist
    if [ -f "${OUT_DIR}/R2_${ROI_A}_nc.npy" ] && \
       [ -f "${OUT_DIR}/R2_${ROI_B}_nc.npy" ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} already done — skipping"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} (stream=${STREAM})" | tee -a "$LOG"

    if [ "$STREAM" = "true" ]; then
        # Streaming: preprocess raw CIFTI on-the-fly, no preprocessed CIFTI saved
        local SG_FLAG="";  [ "$SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        local PSC_FLAG=""; [ "$PSC"       = "true" ] && PSC_FLAG="--psc"
        local GSR_FLAG="--gsr";    [ "$GSR"     = "false" ] && GSR_FLAG="--no-gsr"
        local ZSC_FLAG="--z-score"; [ "$Z_SCORE" = "false" ] && ZSC_FLAG="--no-z-score"

        # shellcheck disable=SC2086
        conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/run_cfmodeling.py" \
            --mode per_subject \
            --roi-a "$ROI_A" --roi-b "$ROI_B" \
            --subject "$SUB" \
            --raw-dir "$CIFTI_DIR" \
            --output-base "$OUTPUT_BASE" \
            $SG_FLAG $PSC_FLAG $GSR_FLAG $ZSC_FLAG >> "$LOG" 2>&1 \
            || { echo "[$(date +%H:%M:%S)] ${SUB} FAILED (streaming)" | tee -a "$LOG"; return 1; }
    else
        # Disk mode: read pre-saved preprocessed CIFTI
        conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/run_cfmodeling.py" \
            --mode per_subject \
            --roi-a "$ROI_A" --roi-b "$ROI_B" \
            --subject "$SUB" \
            --preprocessed-dir "$FMRI_OUT_DIR" \
            --fmri-suffix "$FMRI_SUFFIX" \
            --output-base "$OUTPUT_BASE" >> "$LOG" 2>&1 \
            || { echo "[$(date +%H:%M:%S)] ${SUB} FAILED" | tee -a "$LOG"; return 1; }
    fi

    echo "[$(date +%H:%M:%S)] ${SUB} DONE" | tee -a "$LOG"
}
export -f _run_one_subject


run_persubject_analysis() {
    local ROI_A="$1"
    local ROI_B="$2"
    local OUT="${OUTPUT_BASE}/per_subject/${ROI_A}_${ROI_B}"

    local MODE_TAG
    [ "$STREAM" = "true" ] && MODE_TAG="streaming" || MODE_TAG="disk"
    log "======================================================"
    log "  Per-subject: ${ROI_A} × ${ROI_B}  [${MODE_TAG}]"
    if [ "$STREAM" = "true" ]; then
        log "  Raw CIFTI: ${CIFTI_DIR}"
    else
        log "  FMRI: ${FMRI_OUT_DIR} (suffix: ${FMRI_SUFFIX})"
    fi
    log "  Output: ${OUT}"
    log "  Parallel: ${BATCH_SIZE} jobs"
    log "======================================================"

    # -- Step 01: geometry + LBOEs (one-time, cached) -------------------------
    local SUB_A_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_A" | tr '[:upper:]' '[:lower:]').pkl"
    local SUB_B_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_B" | tr '[:upper:]' '[:lower:]').pkl"

    if [ -f "$SUB_A_PKL" ] && [ -f "$SUB_B_PKL" ]; then
        log "[01] Subsurfaces cached — skipping"
    else
        log "[01] Building ${ROI_A} + ${ROI_B} subsurfaces + LBOEs (59k_fs_LR) …"
        run_python "${SCRIPT_DIR}/01_extract_geometry.py" \
            --mode per_subject \
            --roi_a "$ROI_A" --roi_b "$ROI_B" \
            --hcp_dir "$HCP_DIR" \
            --glasser_dlabel "$GLASSER_DLABEL" \
            --output_base "$OUTPUT_BASE"
        log "[01] Done"
    fi

    # -- Steps 02-04: per-subject in parallel ----------------------------------
    local SUBJECTS
    SUBJECTS=$(ls "$CIFTI_DIR"/ | grep "MOVIE1" | sed 's/_.*//' | sort -u)
    if [ -n "$START_FROM" ]; then
        SUBJECTS=$(echo "$SUBJECTS" | awk "/$START_FROM/{found=1} found{print}")
    fi
    local N_TOTAL
    N_TOTAL=$(echo "$SUBJECTS" | wc -w)
    log "[02-04] Processing ${N_TOTAL} subjects (${BATCH_SIZE} in parallel, ${MODE_TAG}) …"

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_subject {} \
            "$SCRIPT_DIR" "$CONDA_ENV" "$OUTPUT_BASE" \
            "$ROI_A" "$ROI_B" "$FMRI_OUT_DIR" "$FMRI_SUFFIX" \
            "$STREAM" "$CIFTI_DIR" "$SG_FILTER" "$PSC" "$GSR" "$Z_SCORE"
    else
        log "GNU parallel not found — running sequentially"
        log "(install: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB" \
                "$SCRIPT_DIR" "$CONDA_ENV" "$OUTPUT_BASE" \
                "$ROI_A" "$ROI_B" "$FMRI_OUT_DIR" "$FMRI_SUFFIX" \
                "$STREAM" "$CIFTI_DIR" "$SG_FILTER" "$PSC" "$GSR" "$Z_SCORE"
        done
    fi
    log "[02-04] All subjects done"

    # -- Step 05: aggregate + integration maps ---------------------------------
    log "[05] Aggregating subjects → integration maps …"
    run_python "${SCRIPT_DIR}/05_integration_maps.py" \
        --mode per_subject \
        --roi_a "$ROI_A" --roi_b "$ROI_B" \
        --output_base "$OUTPUT_BASE" \
        --template_cifti "$FMRI_GROUP_CIFTI"
    log "[05] Done"

    # -- Step 06: group statistics --------------------------------------------
    log "[06] Group statistics …"
    run_python "${SCRIPT_DIR}/06_summary.py" \
        --mode per_subject \
        --roi_a "$ROI_A" --roi_b "$ROI_B" \
        --output_base "$OUTPUT_BASE" \
        --template_cifti "$FMRI_GROUP_CIFTI"
    log "[06] Done"

    log "Per-subject ${ROI_A}×${ROI_B} complete → ${OUT}"
}


# =============================================================================
# GROUP-AVERAGE PIPELINE
# =============================================================================

run_groupaverage_analysis() {
    local ROI_A="$1"
    local ROI_B="$2"
    local OUT="${OUTPUT_BASE}/group_average/${ROI_A}_${ROI_B}"

    log "======================================================"
    log "  Group-average: ${ROI_A} × ${ROI_B}"
    log "  FMRI: ${FMRI_GROUP_CIFTI}"
    log "  Output: ${OUT}"
    log "======================================================"

    local COMMON="--mode group_average --roi_a $ROI_A --roi_b $ROI_B --output_base $OUTPUT_BASE"

    # -- Step 01: geometry + LBOEs (one-time, cached) -------------------------
    local SUB_A_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_A" | tr '[:upper:]' '[:lower:]').pkl"
    local SUB_B_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_B" | tr '[:upper:]' '[:lower:]').pkl"

    if [ -f "$SUB_A_PKL" ] && [ -f "$SUB_B_PKL" ]; then
        log "[01] Subsurfaces cached — skipping"
    else
        log "[01] Building ${ROI_A} + ${ROI_B} subsurfaces + LBOEs (59k_fs_LR) …"
        run_python "${SCRIPT_DIR}/01_extract_geometry.py" \
            $COMMON \
            --hcp_dir "$HCP_DIR" \
            --glasser_dlabel "$GLASSER_DLABEL"
        log "[01] Done"
    fi

    log "[02-04] Prepare data, fit banded ridge, null-correct …"
    run_python "${SCRIPT_DIR}/run_cfmodeling.py" \
        --mode group_average \
        --roi-a "$ROI_A" --roi-b "$ROI_B" \
        --output-base "$OUTPUT_BASE" \
        --preprocessed-dir "$FMRI_OUT_DIR" \
        --fmri-suffix "$FMRI_SUFFIX" \
        --template-cifti "$FMRI_GROUP_CIFTI"
    log "[02-04] Done"

    log "[05] Integration maps …"
    run_python "${SCRIPT_DIR}/05_integration_maps.py" \
        $COMMON \
        --template_cifti "$FMRI_GROUP_CIFTI"
    log "[05] Done"

    log "[06] RSA spatial overlap …"
    run_python "${SCRIPT_DIR}/06_summary.py" \
        $COMMON \
        --template_cifti "$FMRI_GROUP_CIFTI" \
        --rsa_base "$RSA_BASE"
    log "[06] Done"

    log "Group-average ${ROI_A}×${ROI_B} complete → ${OUT}"
}


# =============================================================================
# ANALYSIS DEFINITIONS  — add ROI pairs here
# =============================================================================

run_all_persubject() {
    run_persubject_analysis "A5"  "FFC"
    run_persubject_analysis "V1"  "3b"
    run_persubject_analysis "TA2" "MST"
}

run_all_groupaverage() {
    run_groupaverage_analysis "A1" "V1"
    run_groupaverage_analysis "A5" "FFC"
    run_groupaverage_analysis "3b" "V1"
}


# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    persubject)
        run_all_persubject
        ;;
    groupaverage)
        run_all_groupaverage
        ;;
    all)
        run_all_persubject
        run_all_groupaverage
        ;;
    *)
        echo "Unknown mode: $MODE. Use: all | persubject | groupaverage" >&2
        exit 1
        ;;
esac

log "All analyses complete."
