#!/usr/bin/env bash
# encoding/run_roi_av_profile.sh
# ================================
# Standalone runner for encoding/roi_av_profile.py: per-ROI audio/video/joint
# variance decomposition on held-out movie runs, for the cca_a/cca_p
# cross-modal-alignment ROIs plus the A5 (auditory) / FFC (visual) unimodal
# anchors. Follows the path/argument conventions of encoding/analysis.sh
# (read there for run_incremental_av()) without editing that file directly.
#
# Stages
#   av    --band-config av: audio vs video bands, both configured models.
#   text  --band-config text: transcript vs caption bands. Only
#         pe-av-small-16-frame ships _transcript_t/_caption_t embeddings.
#   all   av + text.
#
# Usage
#   bash encoding/run_roi_av_profile.sh [STAGE]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
HCP_DIR="${MOVIE_HCP_DIR:-/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1}"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
OUTPUT_DIR="${OUTPUTS_BASE}/encoding/roi_av_profile"
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"
MASKS_DIR="${OUTPUTS_BASE}/cf_modeling/masks"
CONDA_ENV="movie"

BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
ALPHA_MIN=-2
ALPHA_MAX=9
N_ALPHAS=23
N_ITER=20
BACKEND="torch"   # CPU -- GPU is in use by another job
MODEL_RANDOM_STATE=0
N_BOOTSTRAP=10000
N_PERMUTATIONS=10000

AV_MODELS=(pe-av-small-16-frame nemotron_layer18_mp)
TEXT_MODELS=(pe-av-small-16-frame)   # only model with _transcript_t/_caption_t on disk

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

ROI_ARGS=(
    --roi-mask "cca_a=${MASKS_DIR}/cca_a_mask.dscalar.nii"
    --roi-mask "cca_p=${MASKS_DIR}/cca_p_mask.dscalar.nii"
    --glasser-dlabel "$GLASSER_DLABEL"
    --parcel-roi "a5=A5"
    --parcel-roi "ffc=FFC"
)

run_roi_av_profile() {
    local MODEL_NAME="$1" BAND_CONFIG="$2"
    log "roi_av_profile: ${MODEL_NAME} / ${BAND_CONFIG}"
    run_python "${SCRIPT_DIR}/roi_av_profile.py" \
        --preprocessed-dir "$PREPROCESSED_DIR" \
        --fmri-suffix "$FMRI_SUFFIX" \
        --subject group_average \
        --timing-csv "$TIMING_CSV" \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --model "$MODEL_NAME" \
        --band-config "$BAND_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
        --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX" --n-alphas "$N_ALPHAS" --n-iter "$N_ITER" \
        --backend "$BACKEND" --model-random-state "$MODEL_RANDOM_STATE" \
        --n-bootstrap "$N_BOOTSTRAP" --n-permutations "$N_PERMUTATIONS" \
        "${ROI_ARGS[@]}"
}

run_stage_av() {
    local MODEL_NAME
    for MODEL_NAME in "${AV_MODELS[@]}"; do
        run_roi_av_profile "$MODEL_NAME" av
    done
}

run_stage_text() {
    local MODEL_NAME
    for MODEL_NAME in "${TEXT_MODELS[@]}"; do
        run_roi_av_profile "$MODEL_NAME" text
    done
}

STAGE=${1:-all}
case "$STAGE" in
    av)   run_stage_av ;;
    text) run_stage_text ;;
    all)  run_stage_av; run_stage_text ;;
    *) echo "Unknown STAGE: $STAGE (use: av | text | all)"; exit 1 ;;
esac
log "roi_av_profile (${STAGE}) complete."
