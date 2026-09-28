#!/usr/bin/env bash
# cf_modeling/analysis.sh
# ============================
# Master runner for CF (connective field) modeling analyses.
# Imports vicsompy directly from its source repo (no pip install required).
#
# Usage
# -----
#   bash cf_modeling/analysis.sh [MODE] [BATCH_SIZE] [START_FROM]
#
#   MODE        masks        Generate per-ROI CSV mask files from Glasser dlabel
#                            (optional — 01_extract_geometry.py reads dlabel directly)
#               geometry     Build Subsurfaces + LBOEs for all ROI pairs
#               preprocess   Preprocess all subjects: SG→PSC→GSR per run,
#                            save per-subject + group-average CIFTIs
#               avg          Group-average CF modeling for all AVG_PAIRS
#               cca          Extract CCA temporal islands, then run their
#                            group-average geometry/model/post-processing
#               cca_peav_2pct  Run the suffixed PE-AV top-2% CCA variant
#               cca_peav_0p104 Run the lowest three-decimal PE-AV threshold
#                              that keeps anterior/posterior islands separate
#               cca_1pct_all   Run all saved top-1% CCA variants (PE-AV and
#                              Omni-3B/Topo-Omni layers 18/27, last-token)
#               cca_prepare_variants
#                              Prepare requested top-1% and AMPLE sensitivity masks
#               cca_nemotron_18_mp_lboe_sensitivity
#                              Fit Nemotron-18 MP top-1% at 50 and 100 LBOEs
#               cca_hemi_saddle_selected
#                              Run adaptive per-hemisphere pre-merge ROIs for
#                              PE-AV and the highest-max-rho feasible LM map
#               cca_raw_corr_all
#                              Backfill raw-r maps/dlabels for completed CCA fits
#               persubject   Per-subject CF modeling for all PERSUBJECT_PAIRS
#               all          geometry + avg + persubject  (default)
#
#   BATCH_SIZE  N            Parallel subjects per ROI pair (default 8)
#   START_FROM  SUBID        Resume per-subject from this subject ID
#
# Recommended workflow
#   # 0. (Optional) Generate ROI CSV mask files
#   bash cf_modeling/analysis.sh masks
#
#   # 1. Build subsurfaces + LBOEs (one-time, cached)
#   bash cf_modeling/analysis.sh geometry
#
#   # 2. Preprocess all 175 subjects (skip if using streaming mode)
#   bash cf_modeling/analysis.sh preprocess
#
#   # 3. Group-average CF modeling
#   bash cf_modeling/analysis.sh avg
#
#   # 4. Per-subject CF modeling
#   bash cf_modeling/analysis.sh persubject 8
#
# Disk mode (default, STREAM=false)
#   Reads pre-saved preprocessed CIFTIs from PREPROCESSED_INDIV_DIR.
#   Run 'preprocess' mode first.
#
# Streaming mode (STREAM=true)
#   Preprocesses raw 7T CIFTIs on-the-fly; no CIFTI is saved.
#   Set CIFTI_DIR below and configure SG_FILTER/PSC/GSR.
#
# Excluded subjects
#   Subjects without individual midthickness surfaces are listed in
#   data/excluded.txt and have been removed from data/subjects.txt.
#   175 subjects remain.
#
# Resume / skip
#   geometry:    skips if both sub_{roi}.pkl caches already exist.
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
HCP_DIR="${MOVIE_HCP_DIR:-/home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1}"
PYCORTEX_STORE="${DATA_BASE}/hedger2026"
export PYCORTEX_FILESTORE="$PYCORTEX_STORE"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# ── vicsompy source repo (direct import; no pip install) ──────────────────────
VICSOMPY_REPO="/home/amin/Research/Representation/Movie/Vicarious_somatotopy"

# ── Pycortex subject ──────────────────────────────────────────────────────────
CX_SUB="hcp_999999_draw_NH"
SURF_TYPE="fiducial"           # midthickness (sphere not available in our subject)

# ── Glasser HCP-MMP1 59k_fs_LR dlabel ────────────────────────────────────────
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"

# ── LBOE count ────────────────────────────────────────────────────────────────
# Capped at min(N_LBOE, n_L-2, n_R-2) by 01_extract_geometry.py.
N_LBOE=100

# ── Streaming toggle ──────────────────────────────────────────────────────────
# false → disk mode (read pre-saved preprocessed CIFTIs from PREPROCESSED_INDIV_DIR)
# true  → streaming mode (preprocess raw CIFTIs from CIFTI_DIR on-the-fly)
STREAM=false

# Raw CIFTI directory (streaming mode only)
CIFTI_DIR="${MOVIE_RAW_CIFTI_DIR:-/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/individual-59k}"

# ── Preprocessing flags ───────────────────────────────────────────────────────
# Applied per run: SG high-pass → PSC (with pre-SG mean) → GSR
SG_FILTER=true    # Savitzky-Golay high-pass filter
PSC=true          # Percent signal change (uses pre-SG mean for normalisation)
GSR=false          # Global signal regression

# Automatically build PREPROCESSING_FLAG from SG_FILTER/PSC/GSR
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
FMRI_SUFFIX="${PREPROCESSING_FLAG}"
# Per-subject preprocessed CIFTIs (output of preprocess_individual.py --save-individual)
# File pattern: {PREPROCESSED_INDIV_DIR}/{sub}_{PREPROCESSING_FLAG}_cortex_59k.dtseries.nii
PREPROCESSED_INDIV_DIR="${DATA_BASE}/preprocessed/${PREPROCESSING_FLAG}"

# ── Group-average CIFTI ───────────────────────────────────────────────────────
# Two options; comment/uncomment to switch:
#
#   OUR_AVG   — mean of 175 individually preprocessed subjects (from preprocess mode)
#   HEDGER    — Hedger et al. pseudo-subject 999999, 4 runs concatenated
#               (built once by: conda run -n movie python cf_modeling/concat_hedger_average.py)
#
# Our own group average (175-subject mean):
#PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
#FMRI_GROUP_CIFTI="${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
#
# Hedger pseudo-subject 999999 (default — matches Hedger et al. 2025 exactly):
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/hedger_sg_psc"
# Cortex-only template (108441 grayords) — used for integration_maps / viz template
FMRI_GROUP_CIFTI="${PREPROCESSED_DIR}/group_average_hedger_sg_psc_cortex_59k.dtseries.nii"
# Full-brain CIFTI (170494 grayords) — used for fitting, matches Hedger's pipeline exactly
FMRI_GROUP_CIFTI_FULLBRAIN="${PREPROCESSED_DIR}/group_average_hedger_sg_psc_fullbrain.dtseries.nii"
FMRI_GROUP_RUN_TRS="${PREPROCESSED_DIR}/group_average_hedger_sg_psc_run_trs.npy"

RSA_PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
RSA_GROUP_CIFTI="${RSA_PREPROCESSED_DIR}/group_average_raw_cortex_59k.dtseries.nii"
RSA_GROUP_RUN_TRS="${RSA_PREPROCESSED_DIR}/group_average_raw_run_trs.npy"

# Single authoritative subject list — 175 subjects with full 7T fMRI + midthickness.
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# Output and RSA roots
OUTPUT_BASE="${OUTPUTS_BASE}/cf_modeling"
# External HDD for per-subject runs whose un-pruned footprint (~340MB/subject
# x 175 subjects) does not fit the SSD. Mirrors the OUTPUT_BASE/{mode}/{roi}/
# subjects/{sub} layout exactly (only the base path differs), so every
# downstream script (integration_maps.py, overlap.py, ...) needs no changes —
# they read/write under whatever --output-base they are given.
HDD_OUTPUT_BASE="/media/amin/ADATA HD710 PRO/cf_modeling"
RSA_BASE="${OUTPUTS_BASE}/rsa"
RSA_MODEL="pe-av-small-16-frame"   # RSA model name used in overlap.py output filenames

# ROI CSV masks directory (output of 00_make_roi_masks.py)
MASKS_DIR="${OUTPUT_BASE}/masks"
ROI_DEFINITIONS="${SCRIPT_DIR}/roi_definitions.json"

# ── himalaya modeling parameters ─────────────────────────────────────────────
BACKEND="torch_cuda"    # torch_cuda | torch | numpy
N_ITER=20               # random-search iterations for alpha selection
N_TARGETS_BATCH=20000   # targets processed per GPU batch

# ── Parallelisation ──────────────────────────────────────────────────────────
CONDA_ENV="movie"
DEFAULT_BATCH_SIZE=8    # 8 jobs × 2 BLAS threads ≈ 1 job per physical core

# ── Analysis ROI pairs ───────────────────────────────────────────────────────
# Format: "ROI_A:ROI_B"   (use Glasser short names, e.g. 3b V1 A1 TA2 MST A5 FFC)
# Per-subject pairs (computationally expensive; parallelised across subjects)
PERSUBJECT_PAIRS=(
    "A5:FFC"
    "V1:3b"
    "TA2:MST"
)
# Group-average pairs
AVG_PAIRS=(
    "3b:V1"
    "A1:V1"
    "A5:FFC"
    
)

# All unique ROIs across all pairs — used for geometry + mask steps
_all_rois() {
    printf '%s\n' "${PERSUBJECT_PAIRS[@]}" "${AVG_PAIRS[@]}" \
        | tr ':' '\n' | sort -u
}

# =============================================================================
# SHELL SETTINGS
# =============================================================================

MODE=${1:-all}
BATCH_SIZE=${2:-$DEFAULT_BATCH_SIZE}
START_FROM=${3:-""}

# Parallelism control: one BLAS thread per Python process so parallel jobs
# don't fight for CPU cores.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Prevent CUDA OOM from single-large-allocation failures
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# HELPERS
# =============================================================================

log() { echo "[$(date +%H:%M:%S)] $*"; }

qualify_lboe_roi() {
    local ROI="$1"
    local COUNT="$2"
    if [[ "$ROI" =~ _lboe([0-9]+)$ ]]; then
        if [ "${BASH_REMATCH[1]}" != "$COUNT" ]; then
            echo "ROI '$ROI' conflicts with requested lboe${COUNT}" >&2
            return 2
        fi
        printf '%s\n' "$ROI"
    else
        printf '%s_lboe%s\n' "$ROI" "$COUNT"
    fi
}

source_mask_roi() {
    printf '%s\n' "$1" | sed -E 's/_lboe[0-9]+$//'
}

run_python() {
    conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

run_bivariate_workbench_export() {
    local ROI_A="$1"
    local ROI_B="$2"
    local CF_MODE="$3"   # group_average | per_subject
    log "  Exporting Workbench bivariate map (${ROI_A}×${ROI_B}, ${CF_MODE}) ..."
    run_python "${SCRIPT_DIR}/export_bivariate_cifti.py" \
        --mode           "$CF_MODE" \
        --roi-a          "$ROI_A" \
        --roi-b          "$ROI_B" \
        --output-base    "$OUTPUT_BASE" \
        --template-cifti "$FMRI_GROUP_CIFTI" \
        --bins           32 \
        --vmin           0 \
        --vmax           0.4
}

run_roi_mean_partial_connectivity() {
    local ROI_A="$1"
    local ROI_P="$2"
    local PREPROCESSING="$3"
    local DTC_SERIES="$4"
    local RUN_TRS="$5"
    local COLLECTION_TAG="${6:-${ROI_A}_${ROI_P}}"
    local PREPROCESSING_LABEL="${PREPROCESSING}_per_run_zscore"
    local OUT_DIR="${OUTPUT_BASE}/group_average/roi_mean_timeseries_connectivity_${COLLECTION_TAG}/cifti_maps"
    local PARTIAL_CIFTI="${OUT_DIR}/roi_mean_timeseries_partial_pearson_r_${ROI_A}_${ROI_P}_${PREPROCESSING_LABEL}.dscalar.nii"
    local MASK_ROI_A MASK_ROI_P
    MASK_ROI_A=$(source_mask_roi "$ROI_A")
    MASK_ROI_P=$(source_mask_roi "$ROI_P")
    log "  ROI-mean partial Pearson-r (${ROI_A}|${ROI_P}; ${PREPROCESSING_LABEL}) ..."
    run_python "${SCRIPT_DIR}/roi_mean_partial_connectivity.py" \
        --dtseries   "$DTC_SERIES" \
        --run-trs    "$RUN_TRS" \
        --roi-a      "$ROI_A" \
        --roi-p      "$ROI_P" \
        --mask-a     "${MASKS_DIR}/${MASK_ROI_A}_mask.dscalar.nii" \
        --mask-p     "${MASKS_DIR}/${MASK_ROI_P}_mask.dscalar.nii" \
        --output-dir "$OUT_DIR" \
        --preprocessing "$PREPROCESSING"
    log "  Exporting bilateral/within-hemisphere/L/R partial-r bivariate dlabels ..."
    run_python "${SCRIPT_DIR}/export_partial_bivariate_cifti.py" \
        --partial-cifti "$PARTIAL_CIFTI" \
        --roi-a         "$ROI_A" \
        --roi-p         "$ROI_P" \
        --output-dir    "$OUT_DIR" \
        --preprocessing-label "$PREPROCESSING_LABEL" \
        --bins          32 \
        --vmin          0 \
        --vmax          0.4
}

run_roi_mean_raw_connectivity() {
    local ROI_A="$1"
    local ROI_P="$2"
    local PREPROCESSING="$3"
    local DTC_SERIES="$4"
    local RUN_TRS="$5"
    local COLLECTION_TAG="${6:-${ROI_A}_${ROI_P}}"
    local PREPROCESSING_LABEL="${PREPROCESSING}_per_run_zscore"
    local OUT_DIR="${OUTPUT_BASE}/group_average/roi_mean_timeseries_connectivity_${COLLECTION_TAG}/cifti_maps"
    local ZERO_ORDER_CIFTI="${OUT_DIR}/roi_mean_timeseries_zero_order_pearson_r_${ROI_A}_${ROI_P}_${PREPROCESSING_LABEL}.dscalar.nii"
    local BIVARIATE_STEM="${OUT_DIR}/bivariate_roi_mean_timeseries_zero_order_pearson_r_${ROI_A}_${ROI_P}_${PREPROCESSING_LABEL}"
    local MASK_ROI_A MASK_ROI_P
    MASK_ROI_A=$(source_mask_roi "$ROI_A")
    MASK_ROI_P=$(source_mask_roi "$ROI_P")
    if [ -f "$ZERO_ORDER_CIFTI" ] && \
       [ -f "${BIVARIATE_STEM}_bilateral_32bin.dlabel.nii" ] && \
       [ -f "${BIVARIATE_STEM}_within_hemisphere_32bin.dlabel.nii" ] && \
       [ -f "${BIVARIATE_STEM}_L_32bin.dlabel.nii" ] && \
       [ -f "${BIVARIATE_STEM}_R_32bin.dlabel.nii" ]; then
        log "  Zero-order ROI-mean Pearson-r maps already exist for ${ROI_A} x ${ROI_P} (${PREPROCESSING_LABEL}) — skipping"
        return 0
    fi
    log "  Zero-order ROI-mean Pearson-r maps (${ROI_A}, ${ROI_P}; ${PREPROCESSING_LABEL}; not CF maps) ..."
    run_python "${SCRIPT_DIR}/roi_mean_raw_connectivity.py" \
        --dtseries   "$DTC_SERIES" \
        --run-trs    "$RUN_TRS" \
        --roi-a      "$ROI_A" \
        --roi-p      "$ROI_P" \
        --mask-a     "${MASKS_DIR}/${MASK_ROI_A}_mask.dscalar.nii" \
        --mask-p     "${MASKS_DIR}/${MASK_ROI_P}_mask.dscalar.nii" \
        --output-dir "$OUT_DIR" \
        --preprocessing "$PREPROCESSING"
    log "  Exporting zero-order Pearson-r bilateral/within-hemisphere/L/R bivariate dlabels ..."
    run_python "${SCRIPT_DIR}/export_raw_corr_bivariate_cifti.py" \
        --zero-order-cifti "$ZERO_ORDER_CIFTI" \
        --roi-a      "$ROI_A" \
        --roi-p      "$ROI_P" \
        --output-dir "$OUT_DIR" \
        --preprocessing-label "$PREPROCESSING_LABEL" \
        --bins       32 \
        --vmin       0 \
        --vmax       0.4
}

run_roi_mean_connectivity_preprocessing_variants() {
    local ROI_A="$1"
    local ROI_P="$2"
    local MASK_ROI_A MASK_ROI_P COLLECTION_TAG
    MASK_ROI_A=$(source_mask_roi "$ROI_A")
    MASK_ROI_P=$(source_mask_roi "$ROI_P")
    if [ "$MASK_ROI_A" = a5 ] && [ "$MASK_ROI_P" = ffc ]; then
        COLLECTION_TAG="A5_glasser_FFC_glasser"
    elif [ "$MASK_ROI_A" = cca_a_peav_1pct ] && [ "$MASK_ROI_P" = cca_p_peav_1pct ]; then
        COLLECTION_TAG="cca_a_peav_top_1pct_cca_p_peav_top_1pct"
    else
        COLLECTION_TAG="${MASK_ROI_A}_${MASK_ROI_P}"
    fi
    run_roi_mean_raw_connectivity "$MASK_ROI_A" "$MASK_ROI_P" \
        "sg_psc" "$FMRI_GROUP_CIFTI" "$FMRI_GROUP_RUN_TRS" "$COLLECTION_TAG"
    run_roi_mean_partial_connectivity "$MASK_ROI_A" "$MASK_ROI_P" \
        "sg_psc" "$FMRI_GROUP_CIFTI" "$FMRI_GROUP_RUN_TRS" "$COLLECTION_TAG"
    run_roi_mean_raw_connectivity "$MASK_ROI_A" "$MASK_ROI_P" \
        "raw" "$RSA_GROUP_CIFTI" "$RSA_GROUP_RUN_TRS" "$COLLECTION_TAG"
    run_roi_mean_partial_connectivity "$MASK_ROI_A" "$MASK_ROI_P" \
        "raw" "$RSA_GROUP_CIFTI" "$RSA_GROUP_RUN_TRS" "$COLLECTION_TAG"
}

# =============================================================================
# STEP 00: OPTIONAL CSV MASK GENERATION
# =============================================================================
# Generates {MASKS_DIR}/{roi}_{L|R}_mask.csv from the Glasser dlabel.
# Only needed if you want standalone mask files (e.g. for external tools).
# 01_extract_geometry.py reads the dlabel directly and does NOT require these.

run_masks() {
    local ALL_ROIS
    ALL_ROIS=$(_all_rois | tr '\n' ' ')
    log "=== [00] Generate ROI CSV masks (Glasser → ${MASKS_DIR}) ==="
    log "  ROIs: ${ALL_ROIS}"
    # shellcheck disable=SC2086
    run_python "${SCRIPT_DIR}/00_make_roi_masks.py" \
        --glasser-dlabel "$GLASSER_DLABEL" \
        --rois $ALL_ROIS \
        --masks-dir      "$MASKS_DIR"
    log "=== [00] ROI masks done ==="
}

# =============================================================================
# STEP 01: GEOMETRY (Subsurfaces + LBOEs) — per ROI pair
# =============================================================================
# Builds StableSubsurface objects and computes up to N_LBOE LBOEs.
# Caches results in {output_base}/{mode}/{ROI_A}_{ROI_B}/subsurfaces/.
# Re-running skips cached pairs automatically.

_run_geometry_for_pair() {
    local ROI_A="$1"
    local ROI_B="$2"
    local CF_MODE="$3"   # group_average | per_subject

    local SUB_A_PKL="${OUTPUT_BASE}/${CF_MODE}/${ROI_A}_${ROI_B}/subsurfaces/sub_$(echo "$ROI_A" | tr '[:upper:]' '[:lower:]').pkl"
    local SUB_B_PKL="${OUTPUT_BASE}/${CF_MODE}/${ROI_A}_${ROI_B}/subsurfaces/sub_$(echo "$ROI_B" | tr '[:upper:]' '[:lower:]').pkl"

    if [ -f "$SUB_A_PKL" ] && [ -f "$SUB_B_PKL" ]; then
        log "  [01] ${CF_MODE} ${ROI_A}×${ROI_B}: subsurfaces cached — skipping"
        return 0
    fi

    log "  [01] Building ${CF_MODE} ${ROI_A}×${ROI_B} subsurfaces + LBOEs ..."
    run_python "${SCRIPT_DIR}/01_extract_geometry.py" \
        --mode           "$CF_MODE" \
        --roi-a          "$ROI_A" \
        --roi-b          "$ROI_B" \
        --n-lboe         "$N_LBOE" \
        --pycortex-store "$PYCORTEX_STORE" \
        --cx-sub         "$CX_SUB" \
        --surf-type      "$SURF_TYPE" \
        --glasser-dlabel "$GLASSER_DLABEL" \
        --masks-dir      "$MASKS_DIR" \
        --output-base    "$OUTPUT_BASE" \
        --vicsompy-repo  "$VICSOMPY_REPO"
    log "  [01] ${CF_MODE} ${ROI_A}×${ROI_B}: done"
}

run_geometry() {
    log "=== [01] Build all subsurfaces + LBOEs ==="
    for PAIR in "${AVG_PAIRS[@]}"; do
        IFS=':' read -r ROI_A ROI_B <<< "$PAIR"
        _run_geometry_for_pair \
            "$(qualify_lboe_roi "$ROI_A" "$N_LBOE")" \
            "$(qualify_lboe_roi "$ROI_B" "$N_LBOE")" \
            "group_average"
    done
    for PAIR in "${PERSUBJECT_PAIRS[@]}"; do
        IFS=':' read -r ROI_A ROI_B <<< "$PAIR"
        _run_geometry_for_pair \
            "$(qualify_lboe_roi "$ROI_A" "$N_LBOE")" \
            "$(qualify_lboe_roi "$ROI_B" "$N_LBOE")" \
            "per_subject"
    done
    log "=== [01] Geometry complete ==="
}

# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================
# Applies per-run: SG high-pass → PSC (pre-SG mean) → GSR.
# Saves concatenated CIFTI + run_trs.npy per subject and group average.

run_preprocess() {
    log "=== Preprocessing n=$(grep -cv '^\s*#' "$SUBJECTS_LIST") subjects → ${PREPROCESSED_INDIV_DIR} ==="
    log "  Flags: SG_FILTER=${SG_FILTER}  PSC=${PSC}  GSR=${GSR}  (${PREPROCESSING_FLAG})"
    log "  Note: Z-scoring (per run) is applied inside 02_fit_cf_model.py — not here"
    log "  Subjects: ${SUBJECTS_LIST}"

    local SG_FLAG="" PSC_FLAG="" GSR_FLAG=""
    [ "$SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
    [ "$PSC"       = "true" ] && PSC_FLAG="--psc"
    [ "$GSR"       = "true" ] && GSR_FLAG="--gsr"

    # shellcheck disable=SC2086
    run_python "${SCRIPT_DIR}/../preprocess_individual.py" \
        --raw-dir        "$CIFTI_DIR" \
        --out-dir        "$PREPROCESSED_INDIV_DIR" \
        --subjects-list  "$SUBJECTS_LIST" \
        --tr             1.0 \
        $SG_FLAG $PSC_FLAG $GSR_FLAG \
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
# Config is read from exported _CF_-prefixed env vars.
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
    if [ -f "${OUT_DIR}/cf_model_null_corrected_split_r2_${ROI_A}.npy" ] && \
       [ -f "${OUT_DIR}/cf_model_null_corrected_split_r2_${ROI_B}.npy" ]; then
        echo "[$(date +%H:%M:%S)] ${SUB} ${ROI_A}×${ROI_B}: already done — skipping"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Starting ${SUB} ${ROI_A}×${ROI_B} (stream=${_CF_STREAM})" \
        | tee -a "$LOG"

    local STATUS=0
    if [ "$_CF_STREAM" = "true" ]; then
        local SG_FLAG="" PSC_FLAG="" GSR_FLAG=""
        [ "$_CF_SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
        [ "$_CF_PSC"       = "true" ] && PSC_FLAG="--psc"
        [ "$_CF_GSR"       = "true" ] && GSR_FLAG="--gsr"

        # shellcheck disable=SC2086
        conda run --no-capture-output -n "$_CF_CONDA_ENV" python \
            "${_CF_SCRIPT_DIR}/02_fit_cf_model.py" \
            --mode          per_subject \
            --roi-a         "$ROI_A" \
            --roi-b         "$ROI_B" \
            --subject       "$SUB" \
            --raw-dir       "$_CF_CIFTI_DIR" \
            --output-base   "$_CF_OUTPUT_BASE" \
            --backend       "$_CF_BACKEND" \
            --n-iter        "$_CF_N_ITER" \
            --vicsompy-repo "$_CF_VICSOMPY_REPO" \
            $SG_FLAG $PSC_FLAG $GSR_FLAG \
            >> "$LOG" 2>&1 || STATUS=$?
    else
        local FMRI_PATH="${_CF_PREPROCESSED_INDIV_DIR}/${SUB}_${_CF_FMRI_SUFFIX}_cortex_59k.dtseries.nii"
        if [ ! -f "$FMRI_PATH" ]; then
            echo "[$(date +%H:%M:%S)] ${SUB}: no preprocessed CIFTI — run preprocess mode first" \
                | tee -a "$LOG"
            return 1
        fi

        conda run --no-capture-output -n "$_CF_CONDA_ENV" python \
            "${_CF_SCRIPT_DIR}/02_fit_cf_model.py" \
            --mode             per_subject \
            --roi-a            "$ROI_A" \
            --roi-b            "$ROI_B" \
            --subject          "$SUB" \
            --preprocessed-dir "$_CF_PREPROCESSED_INDIV_DIR" \
            --fmri-suffix      "$_CF_FMRI_SUFFIX" \
            --output-base      "$_CF_OUTPUT_BASE" \
            --backend          "$_CF_BACKEND" \
            --n-iter           "$_CF_N_ITER" \
            --vicsompy-repo    "$_CF_VICSOMPY_REPO" \
            >> "$LOG" 2>&1 || STATUS=$?
    fi

    if [ $STATUS -eq 0 ]; then
        # betas_*.npy are per-vertex CF model weights (~170-270MB each x2 ROIs).
        # Neither integration_maps.py nor overlap.py read them for per-subject
        # aggregation (only R2_*/product_map*.npy) — pruning them keeps disk
        # usage near ~6MB/subject instead of ~340MB/subject, for output
        # targets too small to hold the un-pruned run (e.g. a near-full SSD).
        # Kept by default: deleting them makes a run non-reinspectable (no
        # per-subject CF weight matrices without a ~30min/subject refit).
        # Opt in with _CF_PRUNE_BETAS=true when the output disk is the
        # constraint.
        if [ "${_CF_PRUNE_BETAS:-false}" = "true" ]; then
            rm -f "${OUT_DIR}"/betas_*.npy
        fi
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
    log "  Subjects:  ${SUBJECTS_LIST}"
    log "  Output:    ${OUT}"
    log "  Parallel:  ${BATCH_SIZE} jobs"
    log "======================================================"

    # ── Step 01: geometry + LBOEs (one-time, cached) ─────────────────────────
    _run_geometry_for_pair "$ROI_A" "$ROI_B" "per_subject"

    # ── Steps 02: per-subject in parallel ─────────────────────────────────────
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
    log "[02] Processing ${N_TOTAL} subjects (${BATCH_SIZE} parallel, ${MODE_TAG}) ..."

    export _CF_SCRIPT_DIR="$SCRIPT_DIR"
    export _CF_CONDA_ENV="$CONDA_ENV"
    export _CF_OUTPUT_BASE="$OUTPUT_BASE"
    export _CF_PREPROCESSED_INDIV_DIR="$PREPROCESSED_INDIV_DIR"
    export _CF_FMRI_SUFFIX="$FMRI_SUFFIX"
    export _CF_STREAM="$STREAM"
    export _CF_CIFTI_DIR="${CIFTI_DIR:-}"
    export _CF_SG_FILTER="$SG_FILTER"
    export _CF_PSC="$PSC"
    export _CF_GSR="$GSR"
    export _CF_BACKEND="$BACKEND"
    export _CF_N_ITER="$N_ITER"
    export _CF_VICSOMPY_REPO="$VICSOMPY_REPO"

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
    log "[02] All subjects done"

    # ── Post-processing: integration maps ────────────────────────────────────
    log "[03] Aggregating subjects → integration maps ..."
    run_python "${SCRIPT_DIR}/integration_maps.py" \
        --mode           per_subject \
        --roi-a          "$ROI_A" \
        --roi-b          "$ROI_B" \
        --pycortex-store "$PYCORTEX_STORE" \
        --output-base    "$OUTPUT_BASE" \
        --template-cifti "$FMRI_GROUP_CIFTI"
    run_bivariate_workbench_export "$ROI_A" "$ROI_B" "per_subject"
    log "[03] Done"

    # ── Group statistics ──────────────────────────────────────────────────────
    log "[04] Group statistics ..."
    run_python "${SCRIPT_DIR}/overlap.py" \
        --mode           per_subject \
        --roi-a          "$ROI_A" \
        --roi-b          "$ROI_B" \
        --output-base    "$OUTPUT_BASE" \
        --template-cifti "$FMRI_GROUP_CIFTI"
    log "[04] Done"

    log "Per-subject ${ROI_A}×${ROI_B} complete → ${OUT}"
}

run_persubject() {
    log "=== Per-subject CF modeling (${#PERSUBJECT_PAIRS[@]} ROI pairs) ==="
    for PAIR in "${PERSUBJECT_PAIRS[@]}"; do
        IFS=':' read -r ROI_A ROI_B <<< "$PAIR"
        run_persubject_pair \
            "$(qualify_lboe_roi "$ROI_A" "$N_LBOE")" \
            "$(qualify_lboe_roi "$ROI_B" "$N_LBOE")"
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
    log "  fMRI:   ${FMRI_GROUP_CIFTI}"
    log "  Output: ${OUT}"
    log "======================================================"

    # ── Step 01: geometry + LBOEs (one-time, cached) ─────────────────────────
    _run_geometry_for_pair "$ROI_A" "$ROI_B" "group_average"

    # ── Step 02: fit CF model ─────────────────────────────────────────────────
    log "[02] Fit CF model (group_average ${ROI_A}×${ROI_B}) ..."
    run_python "${SCRIPT_DIR}/02_fit_cf_model.py" \
        --mode                  group_average \
        --roi-a                 "$ROI_A" \
        --roi-b                 "$ROI_B" \
        --preprocessed-dir      "$PREPROCESSED_DIR" \
        --fmri-suffix           "$FMRI_SUFFIX" \
        --fmri-fullbrain-path   "$FMRI_GROUP_CIFTI_FULLBRAIN" \
        --template-cifti        "$FMRI_GROUP_CIFTI" \
        --output-base           "$OUTPUT_BASE" \
        --backend               "$BACKEND" \
        --n-iter                "$N_ITER" \
        --n-targets-batch       "$N_TARGETS_BATCH" \
        --vicsompy-repo         "$VICSOMPY_REPO"
    log "[02] Done"

    # ── Post-processing: integration maps ────────────────────────────────────
    log "[03] Integration maps ..."
    run_python "${SCRIPT_DIR}/integration_maps.py" \
        --mode           group_average \
        --roi-a          "$ROI_A" \
        --roi-b          "$ROI_B" \
        --pycortex-store "$PYCORTEX_STORE" \
        --output-base    "$OUTPUT_BASE" \
        --template-cifti "$FMRI_GROUP_CIFTI"
    run_bivariate_workbench_export "$ROI_A" "$ROI_B" "group_average"
    if [[ "$ROI_A" == cca_a* && "$ROI_B" == cca_p* ]]; then
        run_roi_mean_connectivity_preprocessing_variants "$ROI_A" "$ROI_B"
    fi
    log "[03] Done"

    # ── RSA spatial overlap ───────────────────────────────────────────────────
    log "[04] RSA spatial overlap ..."
    run_python "${SCRIPT_DIR}/overlap.py" \
        --mode           group_average \
        --roi-a          "$ROI_A" \
        --roi-b          "$ROI_B" \
        --output-base    "$OUTPUT_BASE" \
        --template-cifti "$FMRI_GROUP_CIFTI" \
        --rsa-base       "$RSA_BASE" \
        --rsa-model      "$RSA_MODEL"
    log "[04] Done"

    log "Group-average ${ROI_A}×${ROI_B} complete → ${OUT}"
}

run_avg() {
    log "=== Group-average CF modeling (${#AVG_PAIRS[@]} ROI pairs) ==="
    for PAIR in "${AVG_PAIRS[@]}"; do
        IFS=':' read -r ROI_A ROI_B <<< "$PAIR"
        run_avg_pair \
            "$(qualify_lboe_roi "$ROI_A" "$N_LBOE")" \
            "$(qualify_lboe_roi "$ROI_B" "$N_LBOE")"
    done
    log "=== Group-average CF modeling complete ==="
}

# =============================================================================
# FUNCTIONAL CCA ISLAND PIPELINE
# =============================================================================

prepare_cca_variant() {
    local TAG="$1"
    local SOURCE_MAP="$2"
    shift 2
    log "=== Prepare CCA ROIs (${TAG}) ==="
    run_python "${SCRIPT_DIR}/run_cca_islands.py" \
        --map              "$SOURCE_MAP" \
        --roi-suffix       "$TAG" \
        --roi-config       "$ROI_DEFINITIONS" \
        --glasser-dlabel   "$GLASSER_DLABEL" \
        --template-cifti   "$FMRI_GROUP_CIFTI" \
        --output-base      "$OUTPUT_BASE" \
        "$@"
}

run_existing_visualization() {
    local ROI_A="$1"
    local ROI_B="$2"
    log "[05] Render existing pycortex visualizations (${ROI_A}×${ROI_B}) ..."
    CCA_VIZ_ROI_A="$ROI_A" CCA_VIZ_ROI_P="$ROI_B" MPLBACKEND=Agg \
        conda run --no-capture-output -n "$CONDA_ENV" ipython -c \
        "import json, os, sys; nb=json.load(open('${SCRIPT_DIR}/viz.ipynb')); code='\\n'.join(''.join(nb['cells'][i]['source']) for i in (1,2,3)); code='\\n'.join(line for line in code.splitlines() if not line.startswith('%')); exec(compile(code, '${SCRIPT_DIR}/viz.ipynb', 'exec')); cortex.options.config.set('webgl', 'colormaps', os.path.join(sys.prefix, 'share', 'pycortex', 'colormaps')); plot_all_for_pair(os.environ['CCA_VIZ_ROI_A'], os.environ['CCA_VIZ_ROI_P'])"
    log "[05] Done"
}

run_cca_fit_pair() {
    local SOURCE_ROI_A="$1"
    local SOURCE_ROI_P="$2"
    local FIT_ROI_A FIT_ROI_P
    FIT_ROI_A=$(qualify_lboe_roi "$SOURCE_ROI_A" "$N_LBOE")
    FIT_ROI_P=$(qualify_lboe_roi "$SOURCE_ROI_P" "$N_LBOE")
    run_avg_pair "$FIT_ROI_A" "$FIT_ROI_P"
    run_existing_visualization "$FIT_ROI_A" "$FIT_ROI_P"
}

run_cca() {
    local CCA_CONFIG ROI_A ROI_B CCA_TAG CCA_THRESHOLD CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS
    CCA_CONFIG=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["cca_temporal_islands"]; d=c["default_variant"]; print(c["names"]["anterior"], c["names"]["posterior"], d["tag"], c["threshold"], c["max_lboe_per_hemisphere"], c["n_targets_batch"], c["backend"], c["cpu_threads"])' "$ROI_DEFINITIONS")
    read -r ROI_A ROI_B CCA_TAG CCA_THRESHOLD CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS <<< "$CCA_CONFIG"
    ROI_A="${ROI_A}_${CCA_TAG}"
    ROI_B="${ROI_B}_${CCA_TAG}"

    log "=== Extract PE-AV fixed-threshold CCA ROIs (${CCA_TAG}; threshold=${CCA_THRESHOLD}) ==="
    run_python "${SCRIPT_DIR}/run_cca_islands.py" \
        --map             "${RSA_BASE}/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy" \
        --threshold       "$CCA_THRESHOLD" \
        --roi-suffix      "$CCA_TAG" \
        --roi-config      "$ROI_DEFINITIONS" \
        --glasser-dlabel  "$GLASSER_DLABEL" \
        --output-base     "$OUTPUT_BASE"

    # The study used 200 to retain fine spatial fields. For our profile-recovery
    # goal, geometry keeps that richest basis where possible and applies its
    # existing shared-hemisphere safety cap for smaller functional ROIs.
    local OLD_N_LBOE="$N_LBOE"
    local OLD_TARGET_BATCH="$N_TARGETS_BATCH"
    local OLD_BACKEND="$BACKEND"
    local OLD_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
    local OLD_OMP_NUM_THREADS="$OMP_NUM_THREADS"
    local OLD_MKL_NUM_THREADS="$MKL_NUM_THREADS"
    N_LBOE="$CCA_N_LBOE"
    N_TARGETS_BATCH="$CCA_TARGET_BATCH"
    BACKEND="$CCA_BACKEND"
    export OPENBLAS_NUM_THREADS="$CCA_CPU_THREADS"
    export OMP_NUM_THREADS="$CCA_CPU_THREADS"
    export MKL_NUM_THREADS="$CCA_CPU_THREADS"
    run_cca_fit_pair "$ROI_A" "$ROI_B"
    N_LBOE="$OLD_N_LBOE"
    N_TARGETS_BATCH="$OLD_TARGET_BATCH"
    BACKEND="$OLD_BACKEND"
    export OPENBLAS_NUM_THREADS="$OLD_OPENBLAS_NUM_THREADS"
    export OMP_NUM_THREADS="$OLD_OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$OLD_MKL_NUM_THREADS"
}

# Suffixed top-2% PE-AV definition.  This is deliberately separate from the
# base `cca` entry point so established unsuffixed outputs remain untouched.
run_cca_peav_2pct() {
    local TAG="peav_2pct"
    local ROI_A="cca_a_${TAG}"
    local ROI_B="cca_p_${TAG}"
    local PEAV_MAP="${RSA_BASE}/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"

    log "=== Extract PE-AV top-2% CCA ROIs (${TAG}) ==="
    run_python "${SCRIPT_DIR}/run_cca_islands.py" \
        --map              "$PEAV_MAP" \
        --top-percent      2 \
        --roi-suffix       "$TAG" \
        --roi-config       "$ROI_DEFINITIONS" \
        --glasser-dlabel   "$GLASSER_DLABEL" \
        --template-cifti   "$FMRI_GROUP_CIFTI" \
        --output-base      "$OUTPUT_BASE"

    local CCA_CONFIG CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS
    CCA_CONFIG=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["cca_temporal_islands"]; print(c["max_lboe_per_hemisphere"], c["n_targets_batch"], c["backend"], c["cpu_threads"])' "$ROI_DEFINITIONS")
    read -r CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS <<< "$CCA_CONFIG"

    local OLD_N_LBOE="$N_LBOE"
    local OLD_TARGET_BATCH="$N_TARGETS_BATCH"
    local OLD_BACKEND="$BACKEND"
    local OLD_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
    local OLD_OMP_NUM_THREADS="$OMP_NUM_THREADS"
    local OLD_MKL_NUM_THREADS="$MKL_NUM_THREADS"
    N_LBOE="$CCA_N_LBOE"
    N_TARGETS_BATCH="$CCA_TARGET_BATCH"
    BACKEND="$CCA_BACKEND"
    export OPENBLAS_NUM_THREADS="$CCA_CPU_THREADS"
    export OMP_NUM_THREADS="$CCA_CPU_THREADS"
    export MKL_NUM_THREADS="$CCA_CPU_THREADS"

    run_cca_fit_pair "$ROI_A" "$ROI_B"

    N_LBOE="$OLD_N_LBOE"
    N_TARGETS_BATCH="$OLD_TARGET_BATCH"
    BACKEND="$OLD_BACKEND"
    export OPENBLAS_NUM_THREADS="$OLD_OPENBLAS_NUM_THREADS"
    export OMP_NUM_THREADS="$OLD_OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$OLD_MKL_NUM_THREADS"
}

# The PE-AV anterior/posterior components merge at 0.103635184467 in the limiting
# (right) hemisphere.  Threshold 0.104 is therefore the lowest safe value at
# three-decimal precision and keeps this boundary definition reproducible.
run_cca_peav_0p104() {
    local TAG="peav_0p104"
    local ROI_A="cca_a_${TAG}"
    local ROI_B="cca_p_${TAG}"
    local PEAV_MAP="${RSA_BASE}/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"

    log "=== Extract PE-AV threshold-0.104 CCA ROIs (${TAG}) ==="
    run_python "${SCRIPT_DIR}/run_cca_islands.py" \
        --map              "$PEAV_MAP" \
        --threshold        0.104 \
        --roi-suffix       "$TAG" \
        --roi-config       "$ROI_DEFINITIONS" \
        --glasser-dlabel   "$GLASSER_DLABEL" \
        --output-base      "$OUTPUT_BASE"

    local CCA_CONFIG CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS
    CCA_CONFIG=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["cca_temporal_islands"]; print(c["max_lboe_per_hemisphere"], c["n_targets_batch"], c["backend"], c["cpu_threads"])' "$ROI_DEFINITIONS")
    read -r CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS <<< "$CCA_CONFIG"

    local OLD_N_LBOE="$N_LBOE"
    local OLD_TARGET_BATCH="$N_TARGETS_BATCH"
    local OLD_BACKEND="$BACKEND"
    local OLD_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
    local OLD_OMP_NUM_THREADS="$OMP_NUM_THREADS"
    local OLD_MKL_NUM_THREADS="$MKL_NUM_THREADS"
    N_LBOE="$CCA_N_LBOE"
    N_TARGETS_BATCH="$CCA_TARGET_BATCH"
    BACKEND="$CCA_BACKEND"
    export OPENBLAS_NUM_THREADS="$CCA_CPU_THREADS"
    export OMP_NUM_THREADS="$CCA_CPU_THREADS"
    export MKL_NUM_THREADS="$CCA_CPU_THREADS"

    run_cca_fit_pair "$ROI_A" "$ROI_B"

    N_LBOE="$OLD_N_LBOE"
    N_TARGETS_BATCH="$OLD_TARGET_BATCH"
    BACKEND="$OLD_BACKEND"
    export OPENBLAS_NUM_THREADS="$OLD_OPENBLAS_NUM_THREADS"
    export OMP_NUM_THREADS="$OLD_OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$OLD_MKL_NUM_THREADS"
}

# Prepare and fit one top-1% CCA definition.
run_cca_1pct_variant() {
    local TAG="$1"
    local SOURCE_MAP="$2"
    local REQUESTED_LBOE="${3:-}"
    local REQUESTED_BACKEND="${4:-}"
    local ROI_A="cca_a_${TAG}"
    local ROI_B="cca_p_${TAG}"

    prepare_cca_variant "$TAG" "$SOURCE_MAP" --top-percent 1

    local CCA_CONFIG CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS
    CCA_CONFIG=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["cca_temporal_islands"]; print(c["max_lboe_per_hemisphere"], c["n_targets_batch"], c["backend"], c["cpu_threads"])' "$ROI_DEFINITIONS")
    read -r CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS <<< "$CCA_CONFIG"
    if [ -n "$REQUESTED_LBOE" ]; then
        CCA_N_LBOE="$REQUESTED_LBOE"
    fi
    if [ -n "$REQUESTED_BACKEND" ]; then
        CCA_BACKEND="$REQUESTED_BACKEND"
    fi

    local OLD_N_LBOE="$N_LBOE"
    local OLD_TARGET_BATCH="$N_TARGETS_BATCH"
    local OLD_BACKEND="$BACKEND"
    local OLD_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
    local OLD_OMP_NUM_THREADS="$OMP_NUM_THREADS"
    local OLD_MKL_NUM_THREADS="$MKL_NUM_THREADS"
    N_LBOE="$CCA_N_LBOE"
    N_TARGETS_BATCH="$CCA_TARGET_BATCH"
    BACKEND="$CCA_BACKEND"
    export OPENBLAS_NUM_THREADS="$CCA_CPU_THREADS"
    export OMP_NUM_THREADS="$CCA_CPU_THREADS"
    export MKL_NUM_THREADS="$CCA_CPU_THREADS"

    run_cca_fit_pair "$ROI_A" "$ROI_B"

    N_LBOE="$OLD_N_LBOE"
    N_TARGETS_BATCH="$OLD_TARGET_BATCH"
    BACKEND="$OLD_BACKEND"
    export OPENBLAS_NUM_THREADS="$OLD_OPENBLAS_NUM_THREADS"
    export OMP_NUM_THREADS="$OLD_OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$OLD_MKL_NUM_THREADS"
}

run_cca_1pct_all() {
    local RSA_SETTINGS="k100_delay5s_bin5s_skip5s_spearman"
    local RSA_FILENAME="rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"
    local TAG MODEL_DIR
    local -a CCA_1PCT_VARIANTS=(
        "peav_1pct:pe-av-small-16-frame_av"
        "omni3b_layer_18_lt_1pct:omni3b_layer18_lt_av"
        "omni3b_layer_27_lt_1pct:omni3b_layer27_lt_av"
        "topoomni_layer_18_lt_1pct:topoomni_layer18_lt_av"
        "topoomni_layer_27_lt_1pct:topoomni_layer27_lt_av"
    )

    log "=== Run all ${#CCA_1PCT_VARIANTS[@]} top-1% CCA variants ==="
    for VARIANT in "${CCA_1PCT_VARIANTS[@]}"; do
        IFS=':' read -r TAG MODEL_DIR <<< "$VARIANT"
        run_cca_1pct_variant \
            "$TAG" \
            "${RSA_BASE}/raw/group_average/${MODEL_DIR}/${RSA_SETTINGS}/${RSA_FILENAME}"
    done
    log "=== All top-1% CCA variants complete ==="
}

# Per-subject CF modeling on the PE-AV top-1% CCA islands (already-verified
# non-overlapping anterior/posterior masks: cca_a_peav_1pct / cca_p_peav_1pct).
# Reuses run_persubject_pair — the same per-subject machinery PERSUBJECT_PAIRS
# uses for anatomical ROIs — rather than a parallel pipeline. Masks are NOT
# regenerated (prepare_cca_variant / run_cca_islands.py are skipped) since
# they already exist under masks/. LBOE basis size is passed as $1 (default
# N_LBOE=100) so a uniform-basis fallback to 50 is a one-argument rerun.
#
# Outputs go to HDD_OUTPUT_BASE (external drive) — 175 subjects x ~340MB
# un-pruned betas does not fit the SSD's few-GB headroom. Betas are KEPT
# (no _CF_PRUNE_BETAS) since the HDD has room; inputs (raw CIFTIs) are
# untouched and still read from CIFTI_DIR on the SSD. Geometry (LBOEs) is
# identical to the already-built SSD cache for this ROI pair — copied over
# once rather than recomputed.
run_persubject_cca_peav_1pct() {
    local REQUESTED_LBOE="${1:-$N_LBOE}"
    local ROI_A ROI_B
    ROI_A=$(qualify_lboe_roi "cca_a_peav_1pct" "$REQUESTED_LBOE")
    ROI_B=$(qualify_lboe_roi "cca_p_peav_1pct" "$REQUESTED_LBOE")

    local SSD_GEOM_DIR="${OUTPUT_BASE}/per_subject/${ROI_A}_${ROI_B}/subsurfaces"
    local HDD_GEOM_DIR="${HDD_OUTPUT_BASE}/per_subject/${ROI_A}_${ROI_B}/subsurfaces"
    if [ -f "${SSD_GEOM_DIR}/sub_${ROI_A}.pkl" ] && [ ! -f "${HDD_GEOM_DIR}/sub_${ROI_A}.pkl" ]; then
        log "  Seeding HDD geometry cache from existing SSD build ..."
        mkdir -p "$HDD_GEOM_DIR"
        cp "$SSD_GEOM_DIR"/* "$HDD_GEOM_DIR"/
    fi

    local OLD_N_LBOE="$N_LBOE"
    local OLD_OUTPUT_BASE="$OUTPUT_BASE"
    local OLD_STREAM="$STREAM"
    N_LBOE="$REQUESTED_LBOE"
    OUTPUT_BASE="$HDD_OUTPUT_BASE"
    # Disk mode needs preprocessed CIFTIs in PREPROCESSED_INDIV_DIR for every
    # subject; only 3/175 exist there. Streaming preprocesses raw CIFTIs
    # on-the-fly from CIFTI_DIR (all 175 present) so no separate preprocess
    # pass (and its own disk cost) is needed first.
    STREAM=true
    run_persubject_pair "$ROI_A" "$ROI_B"
    N_LBOE="$OLD_N_LBOE"
    OUTPUT_BASE="$OLD_OUTPUT_BASE"
    STREAM="$OLD_STREAM"
}

run_cca_prepare_variants() {
    local RSA_SETTINGS="k100_delay5s_bin5s_skip5s_spearman"
    local RSA_FILENAME="rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"
    local TAG MODEL_DIR
    local -a TOP1_VARIANTS=(
        "nemotron_layer_18_mp_1pct:nemotron_layer18_mp_av"
        "topoomni_layer_18_mp_1pct:topoomni_layer18_mp_av"
        "topoomni_layer_18_lt_1pct:topoomni_layer18_lt_av"
    )
    local -a AMPLE_VARIANTS=(
        "nemotron_layer_18_mp:nemotron_layer18_mp_av"
        "peav:pe-av-small-16-frame_av"
        "topoomni_layer_18_mp:topoomni_layer18_mp_av"
        "topoomni_layer_18_lt:topoomni_layer18_lt_av"
    )
    local -a AMPLE_PERCENTAGES=(70 75 80 90)

    for VARIANT in "${TOP1_VARIANTS[@]}"; do
        IFS=':' read -r TAG MODEL_DIR <<< "$VARIANT"
        prepare_cca_variant \
            "$TAG" \
            "${RSA_BASE}/raw/group_average/${MODEL_DIR}/${RSA_SETTINGS}/${RSA_FILENAME}" \
            --top-percent 1
    done
    local PERCENT FRACTION
    for VARIANT in "${AMPLE_VARIANTS[@]}"; do
        IFS=':' read -r TAG MODEL_DIR <<< "$VARIANT"
        for PERCENT in "${AMPLE_PERCENTAGES[@]}"; do
            if [[ "$TAG" == "topoomni_layer_18_mp" && "$PERCENT" == 70 ]]; then
                continue
            fi
            FRACTION="0.${PERCENT}"
            prepare_cca_variant \
                "${TAG}_ample${PERCENT}" \
                "${RSA_BASE}/raw/group_average/${MODEL_DIR}/${RSA_SETTINGS}/${RSA_FILENAME}" \
                --surface-ample "$FRACTION" \
                --ample-seed-top-percent 1 \
                --ample-min-degree 3
        done
    done
    log "=== Requested CCA masks prepared; no models fitted ==="
}

# Fit one top-1% definition at 50 and 100 requested LBOEs.
run_cca_top1_lboe_sensitivity() {
    local TAG="$1"
    local MODEL_DIR="$2"
    local SOURCE_MAP="${RSA_BASE}/raw/group_average/${MODEL_DIR}/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"
    run_cca_1pct_variant "$TAG" "$SOURCE_MAP" 50
    run_cca_1pct_variant "$TAG" "$SOURCE_MAP" 100
}

run_cca_peav_1pct_lboe_sensitivity() {
    run_cca_top1_lboe_sensitivity "peav_1pct" "pe-av-small-16-frame_av"
}

run_cca_nemotron_18_mp_lboe_sensitivity() {
    run_cca_top1_lboe_sensitivity \
        "nemotron_layer_18_mp_1pct" \
        "nemotron_layer18_mp_av"
}

# Fit an adaptive, per-hemisphere pre-merge definition.
run_cca_hemi_saddle_variant() {
    local TAG="$1"
    local SOURCE_MAP="$2"
    local ROI_A="cca_a_${TAG}"
    local ROI_B="cca_p_${TAG}"

    log "=== Extract adaptive hemisphere-saddle CCA ROIs (${TAG}) ==="
    run_python "${SCRIPT_DIR}/run_cca_islands.py" \
        --map                        "$SOURCE_MAP" \
        --adaptive-hemisphere-saddle \
        --adaptive-seed-top-percent 1 \
        --roi-suffix                 "$TAG" \
        --roi-config                 "$ROI_DEFINITIONS" \
        --glasser-dlabel             "$GLASSER_DLABEL" \
        --output-base                "$OUTPUT_BASE"

    local CCA_CONFIG CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS
    CCA_CONFIG=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["cca_temporal_islands"]; print(c["max_lboe_per_hemisphere"], c["n_targets_batch"], c["backend"], c["cpu_threads"])' "$ROI_DEFINITIONS")
    read -r CCA_N_LBOE CCA_TARGET_BATCH CCA_BACKEND CCA_CPU_THREADS <<< "$CCA_CONFIG"

    local OLD_N_LBOE="$N_LBOE"
    local OLD_TARGET_BATCH="$N_TARGETS_BATCH"
    local OLD_BACKEND="$BACKEND"
    local OLD_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
    local OLD_OMP_NUM_THREADS="$OMP_NUM_THREADS"
    local OLD_MKL_NUM_THREADS="$MKL_NUM_THREADS"
    N_LBOE="$CCA_N_LBOE"
    N_TARGETS_BATCH="$CCA_TARGET_BATCH"
    BACKEND="$CCA_BACKEND"
    export OPENBLAS_NUM_THREADS="$CCA_CPU_THREADS"
    export OMP_NUM_THREADS="$CCA_CPU_THREADS"
    export MKL_NUM_THREADS="$CCA_CPU_THREADS"

    run_cca_fit_pair "$ROI_A" "$ROI_B"

    N_LBOE="$OLD_N_LBOE"
    N_TARGETS_BATCH="$OLD_TARGET_BATCH"
    BACKEND="$OLD_BACKEND"
    export OPENBLAS_NUM_THREADS="$OLD_OPENBLAS_NUM_THREADS"
    export OMP_NUM_THREADS="$OLD_OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$OLD_MKL_NUM_THREADS"
}

run_cca_hemi_saddle_selected() {
    local RSA_SETTINGS="k100_delay5s_bin5s_skip5s_spearman"
    local RSA_FILENAME="rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy"

    # LM candidate ranking by maximum finite RSA rho:
    #   Nemotron-18 LT (.531097) > Nemotron-9 LT (.522923) >
    #   Topo-Omni-18 LT (.477923) > Omni-3B-18 LT (.476612) >
    #   Topo-Omni-27 LT (.384452) > Omni-3B-27 LT (.381878).
    # Nemotron-18 forms clean bilateral seed islands and is therefore selected.
    run_cca_hemi_saddle_variant \
        "peav_hemi_saddle" \
        "${RSA_BASE}/raw/group_average/pe-av-small-16-frame_av/${RSA_SETTINGS}/${RSA_FILENAME}"
    run_cca_hemi_saddle_variant \
        "nemotron_layer_18_lt_hemi_saddle" \
        "${RSA_BASE}/raw/group_average/nemotron_layer18_lt_av/${RSA_SETTINGS}/${RSA_FILENAME}"
}

run_cca_raw_corr_all() {
    local PREP PAIR FIRST SECOND ROI_A ROI_P
    log "=== Backfill ordinary CCA correlation maps for completed fits ==="
    for PREP in "${OUTPUT_BASE}"/group_average/cca_a*/prep; do
        [ -d "$PREP" ] || continue
        mapfile -t FILES < <(find "$PREP" -maxdepth 1 -type f -name 'cf_model_null_corrected_split_r2_cca_*.npy' -printf '%f\n' | sort)
        if [ "${#FILES[@]}" -eq 2 ]; then
            FIRST="${FILES[0]#cf_model_null_corrected_split_r2_}"; FIRST="${FIRST%.npy}"
            SECOND="${FILES[1]#cf_model_null_corrected_split_r2_}"; SECOND="${SECOND%.npy}"
            if [[ "$FIRST" == cca_a* ]]; then ROI_A="$FIRST"; ROI_P="$SECOND"; else ROI_A="$SECOND"; ROI_P="$FIRST"; fi
        else
            PAIR="$(basename "$(dirname "$PREP")")"
            [[ "$PAIR" == cca_a*_cca_p* ]] || continue
            ROI_A="${PAIR%%_cca_p*}"
            ROI_P="cca_p${PAIR#*_cca_p}"
        fi
        [[ "$ROI_A" == cca_a* && "$ROI_P" == cca_p* ]] || continue
        run_roi_mean_connectivity_preprocessing_variants "$ROI_A" "$ROI_P"
    done
    log "=== Raw CCA correlation backfill complete ==="
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    masks)                     run_masks ;;
    geometry)                  run_geometry ;;
    preprocess)                run_preprocess ;;
    avg|groupaverage)          run_avg ;;
    cca)                       run_cca ;;
    cca_peav_2pct)             run_cca_peav_2pct ;;
    cca_peav_0p104)            run_cca_peav_0p104 ;;
    cca_1pct_all)              run_cca_1pct_all ;;
    cca_prepare_variants)      run_cca_prepare_variants ;;
    cca_peav_1pct_lboe_sensitivity) run_cca_peav_1pct_lboe_sensitivity ;;
    cca_nemotron_18_mp_lboe_sensitivity) run_cca_nemotron_18_mp_lboe_sensitivity ;;
    cca_hemi_saddle_selected)  run_cca_hemi_saddle_selected ;;
    cca_raw_corr_all)          run_cca_raw_corr_all ;;
    persubject)                run_persubject ;;
    persubject_cca_peav_1pct)  run_persubject_cca_peav_1pct "$N_LBOE" ;;
    all)                       run_geometry; run_avg; run_persubject ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: masks | geometry | preprocess | avg | cca | cca_peav_2pct | cca_peav_0p104 | cca_1pct_all | cca_prepare_variants | cca_peav_1pct_lboe_sensitivity | cca_nemotron_18_mp_lboe_sensitivity | cca_hemi_saddle_selected | cca_raw_corr_all | persubject | persubject_cca_peav_1pct | all" >&2
        exit 1 ;;
esac

log "All CF modeling analyses complete."
