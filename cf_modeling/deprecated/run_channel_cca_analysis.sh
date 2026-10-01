#!/usr/bin/env bash
# Per-channel modality-axis vs CCA-axis correlation (zero-order + partial),
# one model's own top-1% CCA-A/CCA-P mask per model. See
# cf_modeling/deprecated/channel_cca_analysis.py's module docstring for the analysis.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MOVIE_ROOT="/home/amin/Research/Representation/Movie"
OUTPUTS="${MOVIE_ROOT}/outputs"
DATA="${MOVIE_ROOT}/data"
CHANNEL_ROOT="${CHANNEL_CCA_ROOT:-${OUTPUTS}/cf_modeling/deprecated/channel_cca_preference}"
ANALYSIS_ROOT="${CHANNEL_ROOT}/results"
STAGE="${1:-all}"

run_channel_analysis() {
    local mask_a="$1"
    local mask_p="$2"
    shift 2
    conda run --no-capture-output -n movie python \
        "${ROOT}/cf_modeling/deprecated/channel_cca_analysis.py" \
        --embeddings-dir "${OUTPUTS}/model_embeddings" \
        --dtseries "${DATA}/preprocessed/average_sub/hedger_sg_psc/group_average_hedger_sg_psc_cortex_59k.dtseries.nii" \
        --run-trs "${DATA}/preprocessed/average_sub/hedger_sg_psc/group_average_hedger_sg_psc_run_trs.npy" \
        --timing-csv "${DATA}/movie_timing.csv" \
        --mask-a "${mask_a}" \
        --mask-p "${mask_p}" \
        --output-dir "${ANALYSIS_ROOT}" \
        "$@"
}

analyze_channels() {
    run_channel_analysis \
        "${OUTPUTS}/cf_modeling/deprecated/masks/cca_a_peav_1pct_mask.dscalar.nii" \
        "${OUTPUTS}/cf_modeling/deprecated/masks/cca_p_peav_1pct_mask.dscalar.nii" \
        --models peav \
        --roi-set top1pct_peav
}

analyze_topoomni_sheet() {
    run_channel_analysis \
        "${OUTPUTS}/cf_modeling/deprecated/masks/cca_a_topoomni_layer_18_mp_1pct_mask.dscalar.nii" \
        "${OUTPUTS}/cf_modeling/deprecated/masks/cca_p_topoomni_layer_18_mp_1pct_mask.dscalar.nii" \
        --models topoomni_layer18_sheet_mp \
        --roi-set top1pct_topoomni_layer_18_mp
}

analyze_nemotron_own_rois() {
    run_channel_analysis \
        "${OUTPUTS}/cf_modeling/deprecated/masks/cca_a_nemotron_layer_18_mp_1pct_mask.dscalar.nii" \
        "${OUTPUTS}/cf_modeling/deprecated/masks/cca_p_nemotron_layer_18_mp_1pct_mask.dscalar.nii" \
        --models nemotron_layer18_mp \
        --roi-set top1pct_nemotron_layer_18_mp
}

analyze_own_masks() {
    analyze_channels
    analyze_nemotron_own_rois
    analyze_topoomni_sheet
}

case "$STAGE" in
    analyze) analyze_own_masks ;;
    topoomni) analyze_topoomni_sheet ;;
    nemotron-own-rois) analyze_nemotron_own_rois ;;
    all) analyze_own_masks ;;
    *) echo "Use: $0 [analyze|topoomni|nemotron-own-rois|all]" >&2; exit 2 ;;
esac
