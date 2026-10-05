#!/usr/bin/env bash
# cluster/cluster.sh
# ====================
# Master runner for the dual-clustering pipeline (stimulus temporal states
# x brain spatial networks). Mirrors subcortical/analysis.sh conventions.
#
# Usage
#   bash cluster/cluster.sh MODE
#
#   MODE   groupaverage   PRIMARY FIRST PASS. Loop MODELS x modalities, run
#                          run_cluster.py --mode groupaverage for each.
#          persubject     Same loop but --mode persubject (stub this pass).
#          temporal_differentiation  Screen every candidate clustering (vertex and
#                          channel) for mean/max off-diagonal |Pearson r| among its
#                          own cluster mean profiles.
#          channel_model_selection  Reducer/cluster hyperparameter sweep over
#                          channel movie-bin time series, no preliminary PCA.
#                          nemotron_layer18_mp and peav only, run in parallel.
#
# Every default below is overridable via env var and reachable by editing
# this file only.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"

# =============================================================================
# CONFIG
# =============================================================================
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"

GROUP_AVG_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
GROUP_AVG_TRS="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_run_trs.npy"
TEMPLATE_CIFTI="$GROUP_AVG_CIFTI"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
CONDA_ENV="movie"

OUTDIR="${OUTPUTS_BASE}/cluster"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# task shorthand "nemotron:a,v,av" -> on-disk name nemotron_layer27_mp
# (verified present; a/v/av all (626,2048) at bin5s_skip5s). One-line edit
# here to swap the layer or add more models.
#
# _avscramble / _clsav_from_a / _clsav_from_v (modality=av) are AV-integration
# CONTROL conditions, not new stimuli: avscramble pairs real audio with a
# randomly-permuted video (breaks temporal correspondence, tests binding
# sensitivity); clsav_from_a/clsav_from_v blank out one real modality (tests
# whether a network needs BOTH modalities simultaneously present). Reuses
# run_cluster.py unchanged -- these are just other {model}/{modality} on-disk
# embedding names. Consumed by cluster/av_integration.py.
MODELS=(
    "nemotron_layer27_mp:a,v,av"
    "nemotron_layer27_mp_avscramble:av"
    "nemotron_layer27_mp_clsav_from_a:av"
    "nemotron_layer27_mp_clsav_from_v:av"
    "pe-av-small-16-frame:a,v,av"
    "pe-av-small-16-frame_avscramble:av"
    "pe-av-small-16-frame_clsav_from_a:av"
    "pe-av-small-16-frame_clsav_from_v:av"
)

BIN_SEC=5  SKIP_SEC=5  DELAY_SEC=5  TR=1.0
TEMPORAL_REDUCTION=pca  SPATIAL_REDUCTION=pca  SPATIAL_CLUSTER=hdbscan
TEMPORAL_NCOMP=50  SPATIAL_NCOMP=100  ADV_NCOMP=10
N_STATES=10  HMM_COVARIANCE=diag  HMM_N_INIT=10
MIN_CLUSTER_SIZE=100  MIN_SAMPLES=10
N_PERM=1000  PERM_CORRECTION=fdr
CHANNEL_MODEL_SELECTION_FAMILIES=("peav" "nemotron_layer18_mp")

MODE=${1:-groupaverage}
run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_groupaverage() {
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Group-average clustering: ${MODEL_NAME}/${MOD}"
            run_python "${SCRIPT_DIR}/run_cluster.py" \
                --group-avg-cifti "$GROUP_AVG_CIFTI" \
                --group-avg-trs "$GROUP_AVG_TRS" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --output-dir "$OUTDIR" \
                --mode groupaverage \
                --model "$MODEL_NAME" --modality "$MOD" \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                --temporal-reduction "$TEMPORAL_REDUCTION" --temporal-n-components "$TEMPORAL_NCOMP" \
                --adv-n-components "$ADV_NCOMP" \
                --spatial-reduction "$SPATIAL_REDUCTION" --spatial-n-components "$SPATIAL_NCOMP" \
                --spatial-cluster "$SPATIAL_CLUSTER" \
                --min-cluster-size "$MIN_CLUSTER_SIZE" --min-samples "$MIN_SAMPLES" \
                --n-temporal-states "$N_STATES" --hmm-covariance "$HMM_COVARIANCE" --hmm-n-init "$HMM_N_INIT" \
                --n-permutations "$N_PERM" --perm-correction "$PERM_CORRECTION"
        done
    done
    log "Group-average clustering complete."
}

run_persubject() {
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Per-subject clustering (stub): ${MODEL_NAME}/${MOD}"
            run_python "${SCRIPT_DIR}/run_cluster.py" \
                --group-avg-cifti "$GROUP_AVG_CIFTI" \
                --group-avg-trs "$GROUP_AVG_TRS" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --output-dir "$OUTDIR" \
                --mode persubject --subjects-list "$SUBJECTS_LIST" \
                --model "$MODEL_NAME" --modality "$MOD" \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR"
        done
    done
    log "Per-subject clustering complete."
}

run_temporal_differentiation() {
    log "Cluster mean-timeseries temporal-differentiation screen (vertex + channel)"
    run_python "${SCRIPT_DIR}/screen_temporal_differentiation.py"
}

run_channel_model_selection() {
    local FAMILY
    for FAMILY in "${CHANNEL_MODEL_SELECTION_FAMILIES[@]}"; do
        log "Channel model selection: ${FAMILY}"
        run_python "${SCRIPT_DIR}/channel_timeseries_model_selection.py" --family "$FAMILY" &
    done
    wait
    log "Channel model selection complete."
}

case "$MODE" in
    groupaverage) run_groupaverage ;;
    persubject)   run_persubject ;;
    temporal_differentiation) run_temporal_differentiation ;;
    channel_model_selection) run_channel_model_selection ;;
    *) echo "Unknown MODE: $MODE (use: groupaverage | persubject | temporal_differentiation | channel_model_selection)"; exit 1 ;;
esac
