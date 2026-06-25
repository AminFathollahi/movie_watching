#!/usr/bin/env bash
# rsa/run_crossnobis.sh
# =====================
# Usage wrapper for crossnobis_searchlight.py.
# Run crossnobis RSA on one subject (or group_average) using the repeated
# end-of-run clips (video5/9/14/18).  Results are saved per-subject; aggregate
# across subjects with group_stats.py (--method rho_a).
#
# Usage
# -----
#   bash rsa/run_crossnobis.sh [SUBJECT]
#
#   SUBJECT   Subject ID (default: group_average)
#
# Example — single subject
#   bash rsa/run_crossnobis.sh 100610
#
# Example — all subjects via GNU parallel
#   cat data/subjects.txt | parallel -j4 bash rsa/run_crossnobis.sh {}

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Config (mirrors analysis.sh) ───────────────────────────────────────────
DATA_BASE="/home/amin/Research/Representation/Movie/data"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"

MODEL="pe-av-small-16-frame"
MODALITY="av"
K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
CONDA_ENV="movie"

SUBJECT="${1:-group_average}"

# ── Preprocessed dir: group_average lives in average_sub/, subjects in raw/ ─
if [ "$SUBJECT" = "group_average" ]; then
    PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${FMRI_SUFFIX}"
else
    PREPROCESSED_DIR="${DATA_BASE}/preprocessed/${FMRI_SUFFIX}"
fi

# ── Per-subject midthickness (falls back to group-average) ─────────────────
MIDTHICKNESS_DIR="${DATA_BASE}/midthickness_1.6"
SUB_L="${MIDTHICKNESS_DIR}/${SUBJECT}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
SUB_R="${MIDTHICKNESS_DIR}/${SUBJECT}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
if [ -f "$SUB_L" ] && [ -f "$SUB_R" ]; then
    LEFT_SURFACE="$SUB_L"
    RIGHT_SURFACE="$SUB_R"
fi

echo "[$(date +%H:%M:%S)] crossnobis: subject=${SUBJECT}  model=${MODEL}/${MODALITY}"

conda run --no-capture-output -n "$CONDA_ENV" python \
    "${SCRIPT_DIR}/crossnobis_searchlight.py" \
    --preprocessed-dir   "$PREPROCESSED_DIR" \
    --fmri-suffix        "$FMRI_SUFFIX" \
    --timing-csv         "$TIMING_CSV" \
    --embeddings-dir     "$EMBEDDINGS_DIR" \
    --template-cifti     "$TEMPLATE_CIFTI" \
    --left-surface       "$LEFT_SURFACE" \
    --right-surface      "$RIGHT_SURFACE" \
    --workbench          "$WORKBENCH" \
    --output-dir         "$OUTPUT_DIR" \
    --subject            "$SUBJECT" \
    --model              "$MODEL" \
    --modality           "$MODALITY" \
    --k                  "$K" \
    --bin-sec            "$BIN_SEC" \
    --skip-sec           "$SKIP_SEC" \
    --delay-sec          "$DELAY_SEC" \
    --tr                 "$TR" \
    --geodesic-cache-dir "$GEODESIC_CACHE_DIR"

echo "[$(date +%H:%M:%S)] crossnobis done: ${SUBJECT}"
