#!/usr/bin/env bash
# Cleanly rerun the canonical 5 s group-average encoding analysis for every
# native AV model that has matched A, V, and AV embeddings.  The master runner
# appends superadditivity/conjunction maps after all three modalities finish.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ROOT="/home/amin/Research/Representation/Movie/outputs/encoding/group_average"
CONFIG="delay5s_norm_bin5s_skip5s"
CLEAR_EXISTING=false
WAIT_FOR_PID=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --clear-existing)
            CLEAR_EXISTING=true
            shift
            ;;
        --wait-for-pid)
            WAIT_FOR_PID="${2:?--wait-for-pid requires a PID}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

MODELS=(
    "pe-av-small-16-frame:a,v,av"
    "cav-mae-sync:a,v,av"
)

for L in 35 34 27 18 9 1; do
    case "$L" in
        9|18|27|34) SUFFIX="_mp" ;;
        *)          SUFFIX="" ;;
    esac
    MODELS+=("omni3b_layer${L}${SUFFIX}:a,v,av")
    MODELS+=("topoomni_layer${L}${SUFFIX}:a,v,av")
    MODELS+=("topoomni_layer${L}_sheet${SUFFIX}:a,v,av")
done

for L in 9 18 27 35 36; do
    MODELS+=("nemotron_layer${L}_mp:a,v,av")
done
MODELS+=("nemotron_layer35_lt:a,v,av")

if [ -n "$WAIT_FOR_PID" ]; then
    echo "Waiting for PID ${WAIT_FOR_PID} before native-AV encoding rerun ..."
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
        sleep 30
    done
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
        echo "GPU compute process still active; waiting 30 s ..."
        sleep 30
    done
fi

if [ "$CLEAR_EXISTING" = "true" ]; then
    echo "Removing stale native-AV encoding maps for ${CONFIG} ..."
    for ENTRY in "${MODELS[@]}"; do
        MODEL="${ENTRY%%:*}"
        OUT_DIR="${OUTPUT_ROOT}/${MODEL}/${CONFIG}"
        [ -d "$OUT_DIR" ] || continue
        for MOD in a v av; do
            case "$MOD" in
                a) MOD_NAME="audio" ;;
                v) MOD_NAME="visual" ;;
                av) MOD_NAME="audiovisual" ;;
            esac
            rm -f \
                "${OUT_DIR}/encoding_pearson_r_${MOD_NAME}.dscalar.nii" \
                "${OUT_DIR}/encoding_r2_${MOD_NAME}.dscalar.nii" \
                "${OUT_DIR}/encoding_pearson_r_${MOD_NAME}_sigmap_uncorr.dscalar.nii" \
                "${OUT_DIR}/encoding_pearson_r_${MOD_NAME}_sigmap_fdr.dscalar.nii" \
                "${OUT_DIR}/encoding_pearson_r_${MOD_NAME}_fdr_mask.dscalar.nii" \
                "${OUT_DIR}/encoding_pearson_r_${MOD_NAME}_fdr_lh.border" \
                "${OUT_DIR}/encoding_pearson_r_${MOD_NAME}_fdr_rh.border"
        done
        rm -f \
            "${OUT_DIR}/encoding_pearson_r_audiovisual_conjunction.mask.nii" \
            "${OUT_DIR}/n_test.npy"
    done
fi

MODELS_STR=$(IFS=';'; echo "${MODELS[*]}")
export ENCODING_MODELS_OVERRIDE="$MODELS_STR"
export BIN_SECS="5.0"

exec bash "${SCRIPT_DIR}/analysis.sh" avg
