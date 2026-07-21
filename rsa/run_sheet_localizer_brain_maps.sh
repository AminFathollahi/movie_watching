#!/usr/bin/env bash
# rsa/run_sheet_localizer_brain_maps.sh
# =========================================
# Runs rsa/searchlight.py for every per-cluster + combined embedding produced
# by rsa/topoomni_sheet_localizer.py's save_significant_cluster_readouts()
# (see its summary JSON, name from rsa/localizer_naming.py's
# summary_json_name()), then merges every result's rho/fdr maps into ONE
# combined dscalar.nii with descriptive per-cluster map names
# (rsa/label_sheet_localizer_maps.py).
#
# Every raw per-cluster/all-clusters searchlight output AND the final merged
# CIFTI for one analysis live together under ONE folder:
# group_average/<analysis>/raw/<model>_av/... for the raw searchlight runs,
# group_average/<analysis>/<analysis>_all_clusters_maps.dscalar.nii for the
# merged deliverable, group_average/<analysis>/summary.json for context.
#
# Usage:
#   bash rsa/run_sheet_localizer_brain_maps.sh <speech> <driver> <sheet>
# (av_integration was retired in favor of run_av_separability_brain_maps.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

KIND="${1:?usage: run_sheet_localizer_brain_maps.sh <speech> <driver> <sheet>}"
DRIVER="${2:?usage: run_sheet_localizer_brain_maps.sh <speech> <driver> <sheet>}"
SHEET="${3:?usage: run_sheet_localizer_brain_maps.sh <speech> <driver> <sheet>}"
case "$KIND" in
    speech) ;;
    *) echo "kind must be speech"; exit 1 ;;
esac
SUFFIX="${SUFFIX:-}"
BASE_NAME="localizer_${KIND}_drv-${DRIVER}_sheet-${SHEET}${SUFFIX}"

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"
ANALYSIS_DIR="${OUTPUT_DIR}/group_average/${BASE_NAME}"

K=100
BIN_SEC="${BIN_SEC:-5.0}"
SKIP_SEC="${SKIP_SEC:-5.0}"
DELAY_SEC=5.0
METHOD="spearman"
TR=1.0
GPU_BATCH_SIZE=512

SUMMARY_JSON="${EMBEDDINGS_DIR}/_localizer_summary_drv-${DRIVER}_sheet-${SHEET}${SUFFIX}.json"
[ -f "$SUMMARY_JSON" ] || { echo "Missing $SUMMARY_JSON -- run topoomni_sheet_localizer.py first."; exit 1; }

mkdir -p "$ANALYSIS_DIR"
cp "$SUMMARY_JSON" "$ANALYSIS_DIR/summary.json"

COMBINED_ONLY="${COMBINED_ONLY:-}"
if [ -n "$COMBINED_ONLY" ]; then
    MODELS="${BASE_NAME}_all"
else
    MODELS=$(python3 -c "
import json
s = json.load(open('${SUMMARY_JSON}'))
b = s['${KIND}_localizer']
names = [f\"${BASE_NAME}_c{r['cluster_idx']}\" for r in b['significant_clusters']]
names.append('${BASE_NAME}_all')
print(' '.join(names))
")
fi

echo "Models to run ($KIND, drv=$DRIVER, sheet=$SHEET): $MODELS"

for MODEL_NAME in $MODELS; do
    echo "=== searchlight: ${MODEL_NAME} / av ==="
    COMBINED_OUT="${ANALYSIS_DIR}/raw/${MODEL_NAME}_av/rsa_59k_${FMRI_SUFFIX}_k${K}_delay${DELAY_SEC%.*}s_bin${BIN_SEC%.*}s_skip${SKIP_SEC%.*}s_${METHOD}_maps.dscalar.nii"
    CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n movie python "${SCRIPT_DIR}/searchlight.py" \
        --preprocessed-dir   "$PREPROCESSED_DIR" \
        --fmri-suffix        "$FMRI_SUFFIX" \
        --timing-csv         "$TIMING_CSV" \
        --embeddings-dir     "$EMBEDDINGS_DIR" \
        --template-cifti     "$TEMPLATE_CIFTI" \
        --output-dir         "$OUTPUT_DIR" \
        --subject            "group_average" \
        --model              "$MODEL_NAME" \
        --modality           "av" \
        --k                  "$K" \
        --bin-sec            "$BIN_SEC" \
        --skip-sec           "$SKIP_SEC" \
        --delay-sec          "$DELAY_SEC" \
        --method             "$METHOD" \
        --tr                 "$TR" \
        --left-surface       "$LEFT_SURFACE" \
        --right-surface      "$RIGHT_SURFACE" \
        --workbench          "$WORKBENCH" \
        --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
        --combined-output    "$COMBINED_OUT" \
        --gpu-batch-size     "$GPU_BATCH_SIZE" \
        --n-blocks           1 \
        --normalize
done

echo "=== Merging all maps into one combined CIFTI ==="
conda run --no-capture-output -n movie python "${SCRIPT_DIR}/label_sheet_localizer_maps.py" "$KIND" "$DRIVER" "$SHEET" "$SUFFIX"
echo "=== ALL DONE ($KIND drv=$DRIVER sheet=$SHEET suffix=${SUFFIX:-none}) -- see ${ANALYSIS_DIR} ==="
