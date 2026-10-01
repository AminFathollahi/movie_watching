#!/usr/bin/env bash
# Classical PE-AV partial Spearman RSA: group average, subjects, C2F, TFCE-FWE.
#
# Usage:
#   bash rsa/run_peav_partial_analysis.sh all [BATCH_SIZE] [START_FROM]
#   bash rsa/run_peav_partial_analysis.sh group_average
#   bash rsa/run_peav_partial_analysis.sh persubject [BATCH_SIZE] [START_FROM]
#   bash rsa/run_peav_partial_analysis.sh groupstats
#   bash rsa/run_peav_partial_analysis.sh tfce

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

RAW_DIR="${DATA_BASE}/individual-59k"
PREPROCESSED_AVG_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"
TEMPLATE_CIFTI="${PREPROCESSED_AVG_DIR}/group_average_raw_cortex_59k.dtseries.nii"
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
MIDTHICKNESS_DIR="/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Movie/data/midthickness_1.6"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"

CONDA_ENV="movie"
RUN_KEY="partial_corr_pe-av-small-16-frame"
MODEL="pe-av-small-16-frame"
MODALITY="av"
ANALYSIS_LABEL="pe-av-small-16-frame_av_partial_corr"
FMRI_TAG="raw"
K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"
MODEL_NORM="${MODEL_NORM:-center}"
N_BLOCKS=16
BLOCKS_SWEEP=(4 8 16)
N_BOOTSTRAP=2000
N_PERMUTATIONS=5000
GPU_BATCH_SIZE=384

STAGE="${1:-all}"
BATCH_SIZE="${2:-1}"
START_FROM="${3:-}"
N_CPUS="$(nproc 2>/dev/null || echo 8)"
N_JOBS_PER_SUBJECT=$((N_CPUS / BATCH_SIZE))
[ "$N_JOBS_PER_SUBJECT" -lt 1 ] && N_JOBS_PER_SUBJECT=1

PARALLEL_BIN="$(conda run --no-capture-output -n "$CONDA_ENV" which parallel 2>/dev/null \
                || command -v parallel 2>/dev/null || true)"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log() { echo "[$(date +%H:%M:%S)] $*"; }
run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

common_args() {
    printf '%s\n' \
        --run "$RUN_KEY" \
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

run_group_average() {
    log "Rerunning group-average PE-AV classical partial Spearman RSA"
    mapfile -t args < <(common_args)
    run_python "${SCRIPT_DIR}/partial_rsa.py" \
        "${args[@]}" \
        --preprocessed-dir "$PREPROCESSED_AVG_DIR" \
        --fmri-suffix "$FMRI_TAG" \
        --subject group_average \
        --n-blocks 1 \
        --force
}

_run_partial_subject() {
    local sub="$1"
    local left_surface="$_PEAV_LEFT_SURFACE"
    local right_surface="$_PEAV_RIGHT_SURFACE"
    local sub_l="${_PEAV_MIDTHICKNESS_DIR}/${sub}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    local sub_r="${_PEAV_MIDTHICKNESS_DIR}/${sub}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
    if [ -f "$sub_l" ] && [ -f "$sub_r" ]; then
        left_surface="$sub_l"
        right_surface="$sub_r"
    fi

    local log_dir="${_PEAV_OUTPUT_DIR}/subject_data/${sub}"
    local log_file="${log_dir}/partial_peav.log"
    mkdir -p "$log_dir"
    echo "[$(date +%H:%M:%S)] ${sub}: starting classical partial Spearman RSA" | tee -a "$log_file"

    conda run --no-capture-output -n "$_PEAV_CONDA_ENV" python \
        "${_PEAV_SCRIPT_DIR}/partial_rsa.py" \
        --run "$_PEAV_RUN_KEY" \
        --raw-dir "$_PEAV_RAW_DIR" \
        --fmri-suffix "$_PEAV_FMRI_TAG" \
        --timing-csv "$_PEAV_TIMING_CSV" \
        --embeddings-dir "$_PEAV_EMBEDDINGS_DIR" \
        --template-cifti "$_PEAV_TEMPLATE_CIFTI" \
        --output-dir "$_PEAV_OUTPUT_DIR" \
        --subject "$sub" \
        --bin-sec "$_PEAV_BIN_SEC" \
        --skip-sec "$_PEAV_SKIP_SEC" \
        --delay-sec "$_PEAV_DELAY_SEC" \
        --tr "$_PEAV_TR" \
        --k "$_PEAV_K" \
        --method "$_PEAV_METHOD" \
        --model-norm "$_PEAV_MODEL_NORM" \
        --left-surface "$left_surface" \
        --right-surface "$right_surface" \
        --workbench "$_PEAV_WORKBENCH" \
        --geodesic-cache-dir "$_PEAV_GEODESIC_CACHE_DIR" \
        --gpu-batch-size "$_PEAV_GPU_BATCH_SIZE" \
        --n-jobs "$_PEAV_N_JOBS" \
        --n-blocks "$_PEAV_N_BLOCKS" \
        >> "$log_file" 2>&1
    echo "[$(date +%H:%M:%S)] ${sub}: complete" | tee -a "$log_file"
}
export -f _run_partial_subject

run_persubject() {
    local subjects
    subjects="$(grep -v '^\s*#' "$SUBJECTS_LIST" | sed 's/#.*//' | awk 'NF {print $1}')"
    if [ -n "$START_FROM" ]; then
        subjects="$(echo "$subjects" | awk -v start="$START_FROM" '$0 == start {found=1} found')"
    fi
    local n_subjects
    n_subjects="$(echo "$subjects" | awk 'NF' | wc -l)"
    log "Per-subject partial RSA: ${n_subjects} subjects, batch=${BATCH_SIZE}, threads/subject=${N_JOBS_PER_SUBJECT}"

    export _PEAV_SCRIPT_DIR="$SCRIPT_DIR"
    export _PEAV_CONDA_ENV="$CONDA_ENV"
    export _PEAV_RUN_KEY="$RUN_KEY"
    export _PEAV_RAW_DIR="$RAW_DIR"
    export _PEAV_FMRI_TAG="$FMRI_TAG"
    export _PEAV_TIMING_CSV="$TIMING_CSV"
    export _PEAV_EMBEDDINGS_DIR="$EMBEDDINGS_DIR"
    export _PEAV_TEMPLATE_CIFTI="$TEMPLATE_CIFTI"
    export _PEAV_OUTPUT_DIR="$OUTPUT_DIR"
    export _PEAV_BIN_SEC="$BIN_SEC"
    export _PEAV_SKIP_SEC="$SKIP_SEC"
    export _PEAV_DELAY_SEC="$DELAY_SEC"
    export _PEAV_TR="$TR"
    export _PEAV_K="$K"
    export _PEAV_METHOD="$METHOD"
    export _PEAV_MODEL_NORM="$MODEL_NORM"
    export _PEAV_LEFT_SURFACE="$LEFT_SURFACE"
    export _PEAV_RIGHT_SURFACE="$RIGHT_SURFACE"
    export _PEAV_MIDTHICKNESS_DIR="$MIDTHICKNESS_DIR"
    export _PEAV_WORKBENCH="$WORKBENCH"
    export _PEAV_GEODESIC_CACHE_DIR="$GEODESIC_CACHE_DIR"
    export _PEAV_GPU_BATCH_SIZE="$GPU_BATCH_SIZE"
    export _PEAV_N_JOBS="$N_JOBS_PER_SUBJECT"
    export _PEAV_N_BLOCKS="$N_BLOCKS"

    if [ -n "$PARALLEL_BIN" ]; then
        echo "$subjects" | "$PARALLEL_BIN" --jobs "$BATCH_SIZE" --line-buffer \
            _run_partial_subject {}
    else
        for sub in $subjects; do
            _run_partial_subject "$sub"
        done
    fi
}

stats_common_args() {
    printf '%s\n' \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --modality "$MODALITY" \
        --analysis-label "$ANALYSIS_LABEL" \
        --k "$K" \
        --bin-sec "$BIN_SEC" \
        --skip-sec "$SKIP_SEC" \
        --delay-sec "$DELAY_SEC" \
        --method "$METHOD" \
        --model-norm "$MODEL_NORM" \
        --fmri-tag "$FMRI_TAG" \
        --template-cifti "$TEMPLATE_CIFTI" \
        --left-surface "$LEFT_SURFACE" \
        --right-surface "$RIGHT_SURFACE" \
        --workbench "$WORKBENCH"
}

run_groupstats() {
    mapfile -t args < <(stats_common_args)
    for n_blocks in "${BLOCKS_SWEEP[@]}"; do
        log "Random-effects + C2F aggregation: n_blocks=${n_blocks}"
        run_python "${SCRIPT_DIR}/group_stats.py" \
            "${args[@]}" \
            --n-blocks "$n_blocks" \
            --n-bootstrap "$N_BOOTSTRAP"
    done
}

run_tfce() {
    mapfile -t args < <(stats_common_args)
    log "TFCE-FWE aggregation: ${N_PERMUTATIONS} sign-flip permutations"
    run_python "${SCRIPT_DIR}/tfce_groupstats.py" \
        "${args[@]}" \
        --n-permutations "$N_PERMUTATIONS" \
        --n-jobs -1
}

case "$STAGE" in
    group_average) run_group_average ;;
    persubject)    run_persubject ;;
    groupstats)    run_groupstats ;;
    tfce)          run_tfce ;;
    all)           run_group_average; run_persubject; run_groupstats; run_tfce ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Use: group_average | persubject | groupstats | tfce | all" >&2
        exit 2
        ;;
esac

log "PE-AV partial RSA stage '${STAGE}' complete"
