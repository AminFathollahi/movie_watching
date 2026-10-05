#!/usr/bin/env bash
# encoding/run_diff_study.sh
# ============================
# Encoding-currency analogue of rsa/run_diff_study.sh. The dummy conditions
# (clsav_from_a, clsav_from_v) are CONTROLS paired against a native-AV
# baseline, not independent models, so they have their own model list.
# The global temporal scramble is not run: its embeddings mix training and
# held-out clips and variance_partition.py rejects them.
#
# Hypothesis under test: in true integration regions, BOTH the joint-only
# model's held-out Pearson r (r_j) AND the direct joint gain (unique_j:
# A+V+J minus A+V) are high for the intact embedding and WEAKEN under the
# dummy conditions.
#
# Stages
#   partition    variance_partition.py for the intact model and every dummy
#                condition (always with the intact base model's A and V
#                bands and the condition's J), for every split x feature
#                scaling variant.
#   consolidate  encoding/diff_maps.py intact-vs-dummy contrasts.
#   all          partition + consolidate.
#
# Usage
#   bash encoding/run_diff_study.sh [STAGE]
#
# Env overrides (LARGE compute job: each fit is a cross-validated banded ridge):
#   DIFF_STUDY_MODELS       space-separated subset of BASE_MODELS.
#   DIFF_STUDY_CONDITIONS   space-separated subset of {clsav_from_a,clsav_from_v}.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
OUTPUT_DIR="${OUTPUTS_BASE}/encoding"
CONDA_ENV="movie"

BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
N_ITER=20
BACKEND="torch_cuda"
VARIANTS=(loro:demean)

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Same native-AV roster as rsa/run_diff_study.sh and encoding/diff_maps.py.
BASE_MODELS=(
    pe-av-small-16-frame
    nemotron_layer9_mp nemotron_layer18_mp nemotron_layer27_mp nemotron_layer36_mp
    omni3b_layer9_mp omni3b_layer18_mp omni3b_layer27_mp
    topoomni_layer9_mp topoomni_layer18_mp topoomni_layer27_mp
    topoomni_layer9_sheet_mp topoomni_layer18_sheet_mp topoomni_layer27_sheet_mp
)
CONDITIONS=(clsav_from_a clsav_from_v)

if [ -n "${DIFF_STUDY_MODELS:-}" ]; then
    read -ra BASE_MODELS <<< "$DIFF_STUDY_MODELS"
fi
if [ -n "${DIFF_STUDY_CONDITIONS:-}" ]; then
    read -ra CONDITIONS <<< "$DIFF_STUDY_CONDITIONS"
fi

STAGE=${1:-partition}

run_diff_study_partition() {
    local BASE COND MODEL
    log "=== Diff-study variance partition (group-average) ==="
    for BASE in "${BASE_MODELS[@]}"; do
        for COND in "" "${CONDITIONS[@]}"; do
            MODEL="${BASE}${COND:+_$COND}"
            for VARIANT in "${VARIANTS[@]}"; do
                log "  ${MODEL} (${VARIANT%%:*}, ${VARIANT##*:})"
                run_python "${SCRIPT_DIR}/variance_partition.py" \
                    --split "${VARIANT%%:*}" --feature-scaling "${VARIANT##*:}" \
                    --preprocessed-dir "$PREPROCESSED_DIR" \
                    --fmri-suffix "$FMRI_SUFFIX" \
                    --timing-csv "$TIMING_CSV" \
                    --embeddings-dir "$EMBEDDINGS_DIR" \
                    --template-cifti "$TEMPLATE_CIFTI" \
                    --output-dir "$OUTPUT_DIR" \
                    --subject group_average \
                    --model "$MODEL" --audio-model "$BASE" --video-model "$BASE" --tag unimodal_own \
                    --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                    --n-iter "$N_ITER" --backend "$BACKEND" \
                    || log "  SKIP ${MODEL}: variance_partition.py failed (embeddings missing?)"
            done
        done
    done
    log "=== Diff-study variance partition complete ==="
}

case "$STAGE" in
    partition)   run_diff_study_partition ;;
    consolidate) run_python "${SCRIPT_DIR}/diff_maps.py" ;;
    all)         run_diff_study_partition; run_python "${SCRIPT_DIR}/diff_maps.py" ;;
    *) echo "Unknown STAGE: $STAGE (use: partition | consolidate | all)"; exit 1 ;;
esac
log "Diff study (${STAGE}) complete."
