#!/usr/bin/env bash
# DEPRECATED 2026-09-02 — legacy 4-class channel_sensitivity input; see rsa/channel_class_rsa.py header.
echo "DEPRECATED: run_channel_class_rsa.sh is retired (legacy 4-class labels). Refusing to run." >&2; exit 1
# Run model-own-mask channel-class searchlight RSA into one CIFTI.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOVIE_ROOT="${MOVIE_ROOT:-$(dirname "$ROOT")}"
DATA="${MOVIE_ROOT}/data"
OUTPUTS="${MOVIE_ROOT}/outputs"
HCP="${DATA}/HCP_S1200_GroupAvg_v1"
CHANNEL_ROOT="${CHANNEL_CCA_ROOT:-${OUTPUTS}/cf_modeling/channel_cca_preference}"
ANALYSES="${CHANNEL_ROOT}/results/analyses"
RESULTS="${CHANNEL_ROOT}/results/rsa"
COMBINED="${RESULTS}/channel_significance_class_rsa_own_top1pct_3models_k100_perm500.dscalar.nii"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-128}"
PERM_BATCH_SIZE="${PERM_BATCH_SIZE:-100}"

conda run --no-capture-output -n movie python "${ROOT}/rsa/channel_class_rsa.py" \
    --analysis peav top1pct_peav clsav \
        "${OUTPUTS}/model_embeddings/pe-av-small-16-frame/bin5s_skip5s/pe-av-small-16-frame_av.npy" \
        "${ANALYSES}/peav/top1pct_peav/channel_sensitivity_top1pct_peav.csv" \
    --analysis nemotron_layer18_mp top1pct_nemotron_layer_18_mp mp \
        "${OUTPUTS}/model_embeddings/nemotron_layer18_mp/bin5s_skip5s/nemotron_layer18_mp_av.npy" \
        "${ANALYSES}/nemotron_layer18_mp/top1pct_nemotron_layer_18_mp/channel_sensitivity_top1pct_nemotron_layer_18_mp.csv" \
    --analysis topoomni_layer18_sheet_mp top1pct_topoomni_layer_18_mp mp \
        "${OUTPUTS}/model_embeddings/topoomni_layer18_sheet_mp/bin5s_skip5s/topoomni_layer18_sheet_mp_av.npy" \
        "${ANALYSES}/topoomni_layer18_sheet_mp/top1pct_topoomni_layer_18_mp/channel_sensitivity_top1pct_topoomni_layer_18_mp.csv" \
    --reference-rsa peav \
        "${OUTPUTS}/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy" \
    --reference-rsa nemotron_layer18_mp \
        "${OUTPUTS}/rsa/raw/group_average/nemotron_layer18_mp_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy" \
    --reference-rsa topoomni_layer18_sheet_mp \
        "${OUTPUTS}/rsa/raw/group_average/topoomni_layer18_sheet_mp_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy" \
    --dtseries "${DATA}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii" \
    --run-trs "${DATA}/preprocessed/average_sub/raw/group_average_raw_run_trs.npy" \
    --timing-csv "${DATA}/movie_timing.csv" \
    --template-cifti "${DATA}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii" \
    --left-surface "${HCP}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii" \
    --right-surface "${HCP}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii" \
    --workbench /opt/workbench/bin_linux64/wb_command \
    --geodesic-cache-dir "${OUTPUTS}/rsa/_geodesic_cache" \
    --output-dir "${RESULTS}" \
    --output-cifti "${COMBINED}" \
    --k 100 --bin-sec 5 --skip-sec 5 --delay-sec 5 \
    --method spearman --gpu-batch-size "${GPU_BATCH_SIZE}" \
    --n-permutations 500 --permutation-seed 20260831 \
    --perm-batch-size "${PERM_BATCH_SIZE}"
