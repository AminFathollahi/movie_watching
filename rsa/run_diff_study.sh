#!/usr/bin/env bash
# rsa/run_diff_study.sh
# =======================
# Dedicated runner for the scramble/dummy-modality diff study across every
# native-AV multimodal model (pe-av, nemotron, omni3b, topoomni[+sheet]).
#
# Why this is a separate script instead of more rsa/analysis.sh MODELS
# entries: avscramble/clsav_from_a/clsav_from_v are CONTROL CONDITIONS paired
# against a native-AV baseline, not independent models to explore. Dumping
# them into analysis.sh's general model sweep runs the searchlight RSA but
# computes no diff against the native condition and no re-run of the
# partial-RSA integration contrast -- exactly the gap this script closes.
#
# Hypothesis under test (AV-integration claim): in true integration regions,
# BOTH plain RSA alignment AND partial RSA (av controlling for the model's
# own a/v) should be high for the native/intact condition and WEAKEN under
# scramble (mismatched real a/v) and dummy (one real modality + a fixed
# content-free placeholder for the other) -- while non-integration regions
# stay flat across all three conditions.
#
# Stages
#   plain        persubject searchlight + group_stats (mean rho + t/FDR
#                across subjects) + group-average searchlight, for every
#                {base_model}_{condition} in BASE_MODELS x CONDITIONS.
#                Reuses rsa/analysis.sh itself via RSA_MODELS_OVERRIDE so the
#                well-tested persubject/GNU-parallel/group_stats machinery
#                isn't reimplemented; SKIP_NOISE_CEILING/SKIP_CROSSNOBIS skip
#                the two model-independent steps the main pipeline already
#                covers.
#   partial      Re-runs rsa/partial_rsa.py's integration contrast (group-
#                average only, matching how the intact integration runs are
#                computed today) for every condition, via the run keys added
#                to rsa/shared/model_registry.py::PARTIAL_RSA_RUNS.
#   consolidate  Per-subject PAIRED significance test (rsa/scramble_paired_stats.py,
#                which is generic despite its name -- works for any two
#                model/modality RSA outputs) for every condition against its
#                native baseline, then the group-average diff consolidators
#                (rsa/scramble_diff_maps.py, rsa/dummy_diff_maps.py).
#   all          plain + partial + consolidate, in order.
#
# Usage
#   bash rsa/run_diff_study.sh [STAGE]
#
# Env overrides (stage entire model list or subset it for a staged rollout --
# this is a LARGE compute job: ~25 models x 3 conditions x 175 subjects for
# the plain stage alone):
#   DIFF_STUDY_MODELS       space-separated subset of BASE_MODELS.
#   DIFF_STUDY_CONDITIONS   space-separated subset of {avscramble,clsav_from_a,clsav_from_v}.
#
# Example: run just one model's plain-RSA stage first to sanity-check paths:
#   DIFF_STUDY_MODELS="pe-av-small-16-frame" bash rsa/run_diff_study.sh plain
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

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
CONDA_ENV="movie"

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
METHOD="spearman"
MODEL_NORM="${MODEL_NORM:-center}"
export MODEL_NORM
TR=1.0
GPU_BATCH_SIZE=512

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Base native-AV models: every mp/lt readout with avscramble + both dummy
# conditions confirmed on disk. nemotron has no "_lt" integration contrast
# (its "_mp" readout is already genuinely joint, unlike omni3b/topoomni where
# "_mp" was originally a circular (a+v)/2 -- see model_registry.py), so its
# "_lt" embeddings are swept for plain RSA but skipped by the partial stage. ─
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
CONDITIONS=(avscramble clsav_from_a clsav_from_v)

if [ -n "${DIFF_STUDY_MODELS:-}" ]; then
    read -ra BASE_MODELS <<< "$DIFF_STUDY_MODELS"
fi
if [ -n "${DIFF_STUDY_CONDITIONS:-}" ]; then
    read -ra CONDITIONS <<< "$DIFF_STUDY_CONDITIONS"
fi

STAGE=${1:-plain}

# ── Derive a condition's PARTIAL_RSA_RUNS key from its base model name ─────
# "_mp" models -> integration_{short}_{cond}; "_lt" models -> the "lasttoken"
# variant (nuisance pulled from the corresponding "_mp" condition instead of
# a nonexistent "_lt_a"/"_lt_v" -- see model_registry.py's lasttoken helpers).
_run_key() {
    local base="$1" cond="$2" short="$1" suffix=""
    if [[ "$base" == *_mp ]]; then
        short="${base%_mp}"
    elif [[ "$base" == *_lt ]]; then
        short="${base%_lt}"
        suffix="_lasttoken"
    fi
    echo "integration_${short}${suffix}_${cond}"
}

run_diff_study_plain() {
    local MODELS_STR="" BASE COND
    for BASE in "${BASE_MODELS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            MODELS_STR="${MODELS_STR}${BASE}_${COND}:av;"
        done
    done
    MODELS_STR="${MODELS_STR%;}"
    log "=== Diff-study PLAIN RSA: ${#BASE_MODELS[@]} models x ${#CONDITIONS[@]} conditions ==="
    BIN_SECS="$BIN_SEC" MODEL_NORMS="$MODEL_NORM" RSA_MODELS_OVERRIDE="$MODELS_STR" \
        SKIP_NOISE_CEILING=true SKIP_CROSSNOBIS=true \
        bash "${SCRIPT_DIR}/analysis.sh" persubject
    BIN_SECS="$BIN_SEC" MODEL_NORMS="$MODEL_NORM" RSA_MODELS_OVERRIDE="$MODELS_STR" \
        bash "${SCRIPT_DIR}/analysis.sh" avg
    log "=== Diff-study plain RSA complete ==="
}

run_diff_study_partial() {
    local BASE COND RUN_KEY
    log "=== Diff-study PARTIAL RSA (integration contrast, group-average) ==="
    for BASE in "${BASE_MODELS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            RUN_KEY=$(_run_key "$BASE" "$COND")
            log "  ${RUN_KEY}"
            run_python "${SCRIPT_DIR}/partial_rsa.py" \
                --run "$RUN_KEY" \
                --preprocessed-dir "$PREPROCESSED_DIR" \
                --fmri-suffix "$FMRI_SUFFIX" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --output-dir "$OUTPUT_DIR" \
                --subject group_average \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                --k "$K" --method "$METHOD" --model-norm "$MODEL_NORM" \
                --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
                --workbench "$WORKBENCH" \
                --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
                --gpu-batch-size "$GPU_BATCH_SIZE" \
                || log "  SKIP ${RUN_KEY}: partial_rsa.py failed (run undefined or embeddings missing)"
        done
    done
    log "=== Diff-study partial RSA complete ==="
}

run_diff_study_paired_stats() {
    local BASE COND
    log "=== Diff-study per-subject PAIRED stats (native vs each condition) ==="
    for BASE in "${BASE_MODELS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            log "  ${BASE} vs ${BASE}_${COND}"
            run_python "${SCRIPT_DIR}/scramble_paired_stats.py" \
                --output-dir "$OUTPUT_DIR" \
                --intact-model "$BASE" \
                --scrambled-model "${BASE}_${COND}" \
                --modality av --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
                --delay-sec "$DELAY_SEC" --method "$METHOD" --model-norm "$MODEL_NORM" --fmri-tag "$FMRI_SUFFIX" \
                --template-cifti "$TEMPLATE_CIFTI" \
                || log "  SKIP ${BASE}_${COND}: paired stats failed (per-subject maps missing?)"
        done
    done
    log "=== Diff-study paired stats complete ==="
}

run_diff_study_consolidate() {
    run_diff_study_paired_stats
    log "=== Diff-study group-average consolidation ==="
    run_python "${SCRIPT_DIR}/scramble_diff_maps.py" || log "  WARNING: scramble_diff_maps.py failed"
    run_python "${SCRIPT_DIR}/dummy_diff_maps.py"    || log "  WARNING: dummy_diff_maps.py failed"
    log "=== Diff-study consolidation complete ==="
}

case "$STAGE" in
    plain)       run_diff_study_plain ;;
    partial)     run_diff_study_partial ;;
    consolidate) run_diff_study_consolidate ;;
    all)         run_diff_study_plain; run_diff_study_partial; run_diff_study_consolidate ;;
    *) echo "Unknown STAGE: $STAGE (use: plain | partial | consolidate | all)"; exit 1 ;;
esac
log "Diff study (${STAGE}) complete."
