#!/usr/bin/env bash
# rsa/group_avg_multiscale.sh
# ================================
# Group-average searchlight RSA at multiple temporal bin sizes.
#
# fMRI   : group-average, SG + PSC + GSR (continuous)
# Model  : pe-av-small-16-frame / av
# Bins   : 2 s, 5 s, 10 s  (run sequentially)
# Geodesic cache is shared across all bin sizes — computed once for left and
# right hemispheres and reused via _geodesic_cache/group_average_{hem}_neighbors_k{K}.npy
#
# Usage:
#   conda activate analysis
#   bash rsa/group_avg_multiscale.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# Base preprocessing flags pointing to the continuous map directory
PREPROCESSING_FLAG="sg_psc"
FMRI_SUFFIX="${PREPROCESSING_FLAG}"
PREPROCESSED_DIR="${DATA_BASE}/average_sub/${PREPROCESSING_FLAG}"
FMRI_CIFTI="${PREPROCESSED_DIR}/group_average_${FMRI_SUFFIX}_cortex_59k.dtseries.nii"

TIMING_CSV="${DATA_BASE}/segmented_stimulus/filtered/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${FMRI_CIFTI}"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/${PREPROCESSING_FLAG}"

LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"

MODEL="pe-av-small-16-frame"
MODALITY="av"

# IMPORTANT: Update K based on the results of evaluate_reliability.py
K=100
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"
export MODEL_NORM="${MODEL_NORM:-center}"
CONDA_ENV="analysis"
# =============================================================================

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Preflight checks
for f in "$FMRI_CIFTI" "$TIMING_CSV" "$LEFT_SURFACE" "$RIGHT_SURFACE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f" >&2; exit 1
    fi
done

log "Group-average multiscale RSA"
log "  fMRI  : $FMRI_CIFTI"
log "  Model : ${MODEL} / ${MODALITY}"
log "  Bins  : 1 2 seconds"
log "  Geodesic cache: ${OUTPUT_DIR}/_geodesic_cache/group_average_{hem}_neighbors_k${K}.npy"

for BIN_SEC in 1 2  ; do
    EMB="${EMBEDDINGS_DIR}/${MODEL}/${BIN_SEC}s/${MODEL}_${MODALITY}.npy"
    if [ ! -f "$EMB" ]; then
        log "  SKIP bin=${BIN_SEC}s — embedding not found: $EMB"
        continue
    fi

    log "--- bin_sec=${BIN_SEC}s ---"
    conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/searchlight.py" \
        --preprocessed-dir "$PREPROCESSED_DIR" \
        --fmri-suffix      "$FMRI_SUFFIX" \
        --timing-csv       "$TIMING_CSV" \
        --embeddings-dir   "$EMBEDDINGS_DIR" \
        --template-cifti   "$TEMPLATE_CIFTI" \
        --output-dir       "$OUTPUT_DIR" \
        --subject          "group_average" \
        --model            "$MODEL" \
        --modality         "$MODALITY" \
        --k                "$K" \
        --bin-sec          "$BIN_SEC" \
        --delay-sec        "$DELAY_SEC" \
        --method           "$METHOD" \
        --tr               "$TR" \
        --left-surface     "$LEFT_SURFACE" \
        --right-surface    "$RIGHT_SURFACE" \
        --workbench        "$WORKBENCH" \
        --model-norm       "$MODEL_NORM"
    log "--- bin_sec=${BIN_SEC}s done ---"
done

log "All bin sizes complete."
