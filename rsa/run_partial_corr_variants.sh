#!/usr/bin/env bash
# Group-average partial Spearman RSA fits completing three five-map
# partial-correlation families (from_unimodals / text_aligned_models /
# own_unimodal), plus consolidation into their three combined CIFTIs.
#
# Usage:
#   bash rsa/run_partial_corr_variants.sh all
#   bash rsa/run_partial_corr_variants.sh fits
#   bash rsa/run_partial_corr_variants.sh fits_own_unimodal
#   bash rsa/run_partial_corr_variants.sh consolidate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

PREPROCESSED_AVG_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"
GROUP_ROOT="${OUTPUT_DIR}/group_average"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"
TEMPLATE_CIFTI="${PREPROCESSED_AVG_DIR}/group_average_raw_cortex_59k.dtseries.nii"
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"

CONDA_ENV="movie"
FMRI_TAG="raw"
K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"
export MODEL_NORM="${MODEL_NORM:-center}"
GPU_BATCH_SIZE=384

RUN_KEYS=(
    partial_corr_audiomae_given_videomae
    partial_corr_videomae_given_audiomae
    partial_corr_pe-av-small-16-frame_wavlm_only
    partial_corr_pe-av-small-16-frame_pecore_only
    cross_family_specialist_pe-av-small-16-frame
    partial_corr_wavlm_given_pecore
    partial_corr_pecore_given_wavlm
)

# own_unimodal family: PE-AV's own audio/video streams as nuisance. The
# both-nuisance map (peav_av_given_both) already exists as
# integration_pe-av-small-16-frame and is intentionally NOT re-run here.
OWN_UNIMODAL_RUN_KEYS=(
    partial_corr_pe-av-small-16-frame_own_a_only
    partial_corr_pe-av-small-16-frame_own_v_only
    partial_corr_peav_a_given_peav_v
    partial_corr_peav_v_given_peav_a
)

STAGE="${1:-all}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

common_args() {
    printf '%s\n' \
        --timing-csv "$TIMING_CSV" \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --template-cifti "$TEMPLATE_CIFTI" \
        --output-dir "$OUTPUT_DIR" \
        --bin-sec "$BIN_SEC" \
        --skip-sec "$SKIP_SEC" \
        --delay-sec "$DELAY_SEC" \
        --tr "$TR" \
        --k "$K" \
        --method "$METHOD" \
        --model-norm "$MODEL_NORM" \
        --left-surface "$LEFT_SURFACE" \
        --right-surface "$RIGHT_SURFACE" \
        --workbench "$WORKBENCH" \
        --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
        --gpu-batch-size "$GPU_BATCH_SIZE"
}

run_fits() {
    mapfile -t args < <(common_args)
    for run_key in "${RUN_KEYS[@]}"; do
        log "Group-average partial Spearman RSA: run=${run_key}"
        run_python "${SCRIPT_DIR}/partial_rsa.py" \
            "${args[@]}" \
            --run "$run_key" \
            --preprocessed-dir "$PREPROCESSED_AVG_DIR" \
            --fmri-suffix "$FMRI_TAG" \
            --subject group_average \
            --n-blocks 1 \
            --force
        log "Done: ${run_key}"
    done
}

run_fits_own_unimodal() {
    mapfile -t args < <(common_args)
    for run_key in "${OWN_UNIMODAL_RUN_KEYS[@]}"; do
        log "Group-average partial Spearman RSA: run=${run_key}"
        run_python "${SCRIPT_DIR}/partial_rsa.py" \
            "${args[@]}" \
            --run "$run_key" \
            --preprocessed-dir "$PREPROCESSED_AVG_DIR" \
            --fmri-suffix "$FMRI_TAG" \
            --subject group_average \
            --n-blocks 1 \
            --force
        log "Done: ${run_key}"
    done
}

run_consolidate() {
    for variant in from_unimodals text_aligned_models own_unimodal; do
        log "Consolidating partial-corr variant: ${variant}"
        run_python "${SCRIPT_DIR}/partial_corr_variants.py" \
            --variant "$variant" \
            --rsa-root "$GROUP_ROOT" \
            --template-cifti "$TEMPLATE_CIFTI"
    done
}

case "$STAGE" in
    fits)              run_fits ;;
    fits_own_unimodal) run_fits_own_unimodal ;;
    consolidate)       run_consolidate ;;
    all)               run_fits; run_fits_own_unimodal; run_consolidate ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Use: fits | fits_own_unimodal | consolidate | all" >&2
        exit 2
        ;;
esac

log "Partial-corr variants stage '${STAGE}' complete"
