#!/bin/bash
# run_analysis.sh
# ===============
# Master runner for all CF modeling analyses.
#
# Usage:
#   bash run_analysis.sh                    # run all defined analyses
#   bash run_analysis.sh persubject         # only per-subject pipeline
#   bash run_analysis.sh groupaverage       # only group-average pipeline
#   bash run_analysis.sh persubject 8       # override batch size
#   bash run_analysis.sh persubject 1 100610  # resume from subject 100610
#
# Prerequisites:
#   conda activate cfmod
#   GNU parallel: conda install -c conda-forge parallel

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data/Setareh"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"
CIFTI_DIR="/media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI"

GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"
TEMPLATE_CIFTI="${HCP_DIR}/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"

PER_SUBJECT_OUT="/home/amin/Research/Representation/Movie/outputs/cf_modeling/per_subject"
GROUP_AVG_OUT="/home/amin/Research/Representation/Movie/outputs/cf_modeling/group_average"
RSA_BASE="/home/amin/Research/Representation/Movie/outputs/searchlight_rsa_output"

CONDA_ENV="cfmod"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DEFAULT_BATCH_SIZE=8   # 8 jobs × 2 BLAS threads = 16 threads = 1 per physical core

# =============================================================================
# CLI
# =============================================================================
MODE=${1:-"all"}         # all | persubject | groupaverage
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
START_FROM=${3:-""}      # resume from this subject ID (per-subject only)

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() {
    conda run -n "$CONDA_ENV" python "$@"
}

# Top-level worker — called by parallel with all values as positional args
# Args: SUB SCRIPT_DIR CONDA_ENV PER_SUBJECT_OUT ROI_A ROI_B CIFTI_DIR
_run_one_subject() {
    local SUB="$1"
    local SCRIPT_DIR="$2"
    local CONDA_ENV="$3"
    local PER_SUBJECT_OUT="$4"
    local ROI_A="$5"
    local ROI_B="$6"
    local CIFTI_DIR="$7"

    local LOG_DIR="${PER_SUBJECT_OUT}/${ROI_A}_${ROI_B}/subjects/${SUB}"
    mkdir -p "$LOG_DIR"
    local LOG="${LOG_DIR}/pipeline.log"

    local NC_A="${LOG_DIR}/R2_${ROI_A}_nc.npy"
    local NC_B="${LOG_DIR}/R2_${ROI_B}_nc.npy"
    if [ -f "$NC_A" ] && [ -f "$NC_B" ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} already done — skipping"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB}" | tee -a "$LOG"

    conda run -n "$CONDA_ENV" \
        python "${SCRIPT_DIR}/per_subject/02_fit_subject.py" \
            --subject "$SUB" \
            --roi_a "$ROI_A" --roi_b "$ROI_B" \
            --output_base "$PER_SUBJECT_OUT" \
            --cifti_dir "$CIFTI_DIR" >> "$LOG" 2>&1

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} DONE" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] ${SUB} FAILED (exit $STATUS)" | tee -a "$LOG"
        return $STATUS
    fi
}
export -f _run_one_subject

# =============================================================================
# PER-SUBJECT PIPELINE
# =============================================================================
run_persubject_analysis() {
    local ROI_A="$1"
    local ROI_B="$2"
    local OUT="${PER_SUBJECT_OUT}/${ROI_A}_${ROI_B}"

    log "======================================================"
    log "  Per-subject: ${ROI_A} × ${ROI_B}"
    log "  Output: ${OUT}"
    log "  Parallel: ${BATCH_SIZE} jobs"
    log "======================================================"

    # -- Step 01: geometry + LBOEs (one-time) ---------------------------------
    local SUB_A_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_A" | tr '[:upper:]' '[:lower:]').pkl"
    local SUB_B_PKL="${OUT}/subsurfaces/sub_$(echo "$ROI_B" | tr '[:upper:]' '[:lower:]').pkl"

    if [ -f "$SUB_A_PKL" ] && [ -f "$SUB_B_PKL" ]; then
        log "[01] Subsurfaces cached — skipping"
    else
        log "[01] Building ${ROI_A} + ${ROI_B} subsurfaces + LBOEs …"
        run_python "${SCRIPT_DIR}/per_subject/01_extract_geometry.py" \
            --roi_a "$ROI_A" --roi_b "$ROI_B" \
            --hcp_dir "$HCP_DIR" \
            --glasser_dlabel "$GLASSER_DLABEL" \
            --output_base "$PER_SUBJECT_OUT"
        log "[01] Done"
    fi

    # -- Steps 02-04: per-subject (parallel) ----------------------------------
    local SUBJECTS
    SUBJECTS=$(ls "$CIFTI_DIR"/ | grep "MOVIE1" | sed 's/_.*//' | sort -u)
    if [ -n "$START_FROM" ]; then
        SUBJECTS=$(echo "$SUBJECTS" | awk "/$START_FROM/{found=1} found{print}")
    fi
    local N_TOTAL
    N_TOTAL=$(echo "$SUBJECTS" | wc -w)
    log "[02-04] Processing ${N_TOTAL} subjects (${BATCH_SIZE} in parallel) …"

    if command -v parallel &>/dev/null; then
        echo "$SUBJECTS" | parallel --jobs "$BATCH_SIZE" --line-buffer \
            _run_one_subject {} \
            "$SCRIPT_DIR" "$CONDA_ENV" "$PER_SUBJECT_OUT" "$ROI_A" "$ROI_B" "$CIFTI_DIR"
    else
        log "GNU parallel not found — running sequentially (install: conda install -c conda-forge parallel)"
        for SUB in $SUBJECTS; do
            _run_one_subject "$SUB" \
                "$SCRIPT_DIR" "$CONDA_ENV" "$PER_SUBJECT_OUT" "$ROI_A" "$ROI_B" "$CIFTI_DIR"
        done
    fi

    log "[02-04] All subjects done"

    # -- Step 05: group average -----------------------------------------------
    log "[05] Group average …"
    run_python "${SCRIPT_DIR}/per_subject/05_group_average.py" --no_pycortex \
        --roi_a "$ROI_A" --roi_b "$ROI_B" \
        --hcp_dir "$HCP_DIR" --cifti_dir "$CIFTI_DIR" \
        --output_base "$PER_SUBJECT_OUT"
    log "[05] Done"

    # -- Step 06: group stats -------------------------------------------------
    log "[06] Group statistics + figures …"
    MPLBACKEND=Agg run_python "${SCRIPT_DIR}/per_subject/06_group_stats.py" \
        --roi_a "$ROI_A" --roi_b "$ROI_B" \
        --hcp_dir "$HCP_DIR" --cifti_dir "$CIFTI_DIR" \
        --output_base "$PER_SUBJECT_OUT"
    log "[06] Done"

    log "Per-subject ${ROI_A}×${ROI_B} complete → ${OUT}"
}

# =============================================================================
# GROUP-AVERAGE PIPELINE
# =============================================================================
run_groupaverage_analysis() {
    local ROI_A="$1"
    local ROI_B="$2"
    local OUT="${GROUP_AVG_OUT}/${ROI_A}_${ROI_B}"

    log "======================================================"
    log "  Group-average: ${ROI_A} × ${ROI_B}"
    log "  Output: ${OUT}"
    log "======================================================"

    local COMMON="--roi_a $ROI_A --roi_b $ROI_B --output_base $GROUP_AVG_OUT"

    log "[01] Building subsurfaces + LBOEs …"
    run_python "${SCRIPT_DIR}/group_average/01_extract_geometry.py" \
        $COMMON \
        --hcp_dir "$HCP_DIR" \
        --template_cifti "$TEMPLATE_CIFTI"
    log "[01] Done"

    log "[02] Preparing HCP group-average timeseries …"
    run_python "${SCRIPT_DIR}/group_average/02_prep_hcp_timeseries.py" \
        $COMMON \
        --data_base "$DATA_BASE"
    log "[02] Done"

    log "[03] Fitting banded ridge …"
    run_python "${SCRIPT_DIR}/group_average/03_fit_banded_ridge.py" \
        $COMMON \
        --template_cifti "$TEMPLATE_CIFTI"
    log "[03] Done"

    log "[04] Null-corrected maps …"
    run_python "${SCRIPT_DIR}/group_average/04_null_corrected_maps.py" \
        $COMMON \
        --template_cifti "$TEMPLATE_CIFTI"
    log "[04] Done"

    log "[05] Integration maps …"
    run_python "${SCRIPT_DIR}/group_average/05_integration_maps.py" \
        $COMMON \
        --template_cifti "$TEMPLATE_CIFTI"
    log "[05] Done"

    log "[06] RSA spatial overlap …"
    run_python "${SCRIPT_DIR}/group_average/06_rsa_overlap.py" \
        $COMMON \
        --template_cifti "$TEMPLATE_CIFTI" \
        --rsa_base "$RSA_BASE"
    log "[06] Done"

    log "Group-average ${ROI_A}×${ROI_B} complete → ${OUT}"
}

# =============================================================================
# ANALYSIS DEFINITIONS
# =============================================================================
run_all_persubject() {
    run_persubject_analysis "A5"  "FFC"
    run_persubject_analysis "V1"  "3b"
    run_persubject_analysis "TA2" "MST"
}

run_all_groupaverage() {
    run_groupaverage_analysis "A1" "V1"
    run_groupaverage_analysis "A5" "FFC"
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
