#!/usr/bin/env bash
# subcortical/run_diff_study.sh
# ================================
# Subcortical analogue of rsa/run_diff_study.sh + encoding/run_diff_study.sh.
# Drives the Move-1 partial-RSA integration contrast (subcortical_partial_rsa.py,
# new this pass -- subcortical/ had zero partial-RSA infra before) for every
# native-AV model's intact/scramble/dummy conditions, then consolidates via
# subcortical/diff_maps.py (binding + modality_presence_diff, same naming as
# the cortical and encoding versions).
#
# Unlike rsa/encoding, subcortical/analysis.sh's default MODELS array ALREADY
# includes every scramble/dummy condition for every native-mm layer/pooling
# combo (see its NATIVE_MM_LAYER_BASES block) -- so the "plain" stage here
# just reuses that existing sweep rather than building a parallel model list.
#
# Stages
#   plain        Group-average plain subcortical searchlight RSA for every
#                native-AV model + its scramble/dummy conditions. Reuses
#                subcortical/analysis.sh groupavg as-is.
#   partial      subcortical_partial_rsa.py (Move-1 integration contrast) for
#                every condition, using the SAME rsa/shared/model_registry.py
#                PARTIAL_RSA_RUNS keys as the cortical script (target/nuisance
#                are just embedding lookups, agnostic to brain structure).
#   consolidate  subcortical/diff_maps.py.
#   all          plain + partial + consolidate.
#
# Usage
#   bash subcortical/run_diff_study.sh [STAGE]
#
# Env overrides (partial stage; this is real compute -- a nuisance projection and a
# 19-structure searchlight per run):
#   DIFF_STUDY_MODELS       space-separated subset of BASE_MODELS.
#   DIFF_STUDY_CONDITIONS   space-separated subset of {intact,avscramble,clsav_from_a,clsav_from_v}.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"

CIFTI_DIR="${DATA_BASE}/individual-59k"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
OUTPUT_DIR="${OUTPUTS_BASE}/subcortical"
NEIGHBOR_CACHE_DIR="${OUTPUT_DIR}/_neighbor_cache"
GROUP_AVERAGE_DIR="${OUTPUT_DIR}/group_average_cache"
TEMPLATE_CIFTI="${OUTPUT_DIR}/subcortical_template.dscalar.nii"
CONDA_ENV="movie"

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Same native-AV roster as rsa/run_diff_study.sh / encoding/run_diff_study.sh.
BASE_MODELS=(
    pe-av-small-16-frame
    nemotron_layer9_mp nemotron_layer9_lt
    nemotron_layer18_mp nemotron_layer18_lt
    nemotron_layer27_mp nemotron_layer27_lt
    nemotron_layer36_mp nemotron_layer36_lt
    omni3b_layer9_mp omni3b_layer9_lt
    omni3b_layer18_mp omni3b_layer18_lt
    omni3b_layer27_mp omni3b_layer27_lt
    topoomni_layer9_mp topoomni_layer9_lt
    topoomni_layer18_mp topoomni_layer18_lt
    topoomni_layer27_mp topoomni_layer27_lt
    topoomni_layer9_sheet_mp topoomni_layer9_sheet_lt
    topoomni_layer18_sheet_mp topoomni_layer18_sheet_lt
    topoomni_layer27_sheet_mp topoomni_layer27_sheet_lt
)
CONDITIONS=(intact avscramble clsav_from_a clsav_from_v)

if [ -n "${DIFF_STUDY_MODELS:-}" ]; then
    read -ra BASE_MODELS <<< "$DIFF_STUDY_MODELS"
fi
if [ -n "${DIFF_STUDY_CONDITIONS:-}" ]; then
    read -ra CONDITIONS <<< "$DIFF_STUDY_CONDITIONS"
fi

STAGE=${1:-plain}

# Same run-key derivation as rsa/run_diff_study.sh's _run_key() -- identical
# rsa/shared/model_registry.py registry, so the logic must match exactly.
_run_key() {
    local base="$1" cond="$2" short="$1" suffix=""
    if [[ "$base" == *_mp ]]; then
        short="${base%_mp}"
    elif [[ "$base" == *_lt ]]; then
        short="${base%_lt}"
        suffix="_lasttoken"
    fi
    if [ "$cond" = "intact" ]; then
        echo "integration_${short}${suffix}"
    else
        echo "integration_${short}${suffix}_${cond}"
    fi
}
# pe-av's intact integration run is registered under the special "run_B" key
# (see rsa/shared/model_registry.py), not "integration_pe-av-small-16-frame".
_run_key_special() {
    local base="$1" cond="$2"
    if [ "$cond" = "intact" ] && [ "$base" = "pe-av-small-16-frame" ]; then
        echo "run_B"
    else
        _run_key "$base" "$cond"
    fi
}

run_diff_study_plain() {
    log "=== Diff-study PLAIN subcortical RSA (reusing analysis.sh groupavg's default sweep) ==="
    bash "${SCRIPT_DIR}/analysis.sh" groupavg
    log "=== Diff-study plain subcortical RSA complete ==="
}

run_diff_study_partial() {
    local BASE COND RUN_KEY
    log "=== Diff-study PARTIAL subcortical RSA (Move-1 integration contrast) ==="
    for BASE in "${BASE_MODELS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            RUN_KEY=$(_run_key_special "$BASE" "$COND")
            log "  ${BASE}/${COND} -> ${RUN_KEY}"
            run_python "${SCRIPT_DIR}/subcortical_partial_rsa.py" \
                --run "$RUN_KEY" \
                --raw-dir "$CIFTI_DIR" \
                --subjects-list "$SUBJECTS_LIST" \
                --subject group_average \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --output-dir "$OUTPUT_DIR" \
                --group-average-dir "$GROUP_AVERAGE_DIR" \
                --neighbor-cache-dir "$NEIGHBOR_CACHE_DIR" \
                --workbench "$WORKBENCH" \
                --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
                --method "$METHOD" --tr "$TR" \
                || log "  SKIP ${RUN_KEY}: subcortical_partial_rsa.py failed (run undefined or embeddings missing)"
        done
    done
    log "=== Diff-study partial subcortical RSA complete ==="
}

case "$STAGE" in
    plain)       run_diff_study_plain ;;
    partial)     run_diff_study_partial ;;
    consolidate) run_python "${SCRIPT_DIR}/diff_maps.py"; bash "${SCRIPT_DIR}/analysis.sh" visualize ;;
    all)         run_diff_study_plain; run_diff_study_partial; run_python "${SCRIPT_DIR}/diff_maps.py"; bash "${SCRIPT_DIR}/analysis.sh" visualize ;;
    *) echo "Unknown STAGE: $STAGE (use: plain | partial | consolidate | all)"; exit 1 ;;
esac
log "Diff study (${STAGE}) complete."
