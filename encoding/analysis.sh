#!/usr/bin/env bash
# encoding/analysis.sh
# ==========================
# Master runner for the banded-ridge encoding analyses (group-average data).
#
# Usage
# -----
#   bash encoding/analysis.sh preprocess
#   bash encoding/analysis.sh variance_partition [--models M...] [--bins B...]
#                             [--variants split:scaling...] [--controls SET...|none|all]
#                             [--subsets S...] [--response-scaling run|none]
#   bash encoding/analysis.sh screen [--audio-models M...] [--video-models M...]
#                             [--bins B...] [--variants split:scaling...]
#   bash encoding/analysis.sh factorial_interaction [--models M...] [--bins 5]
#
#   preprocess           Preprocess all 175 subjects from SUBJECTS_LIST:
#                        per-subject CIFTIs → PREPROCESSED_INDIV_DIR
#                        group-average CIFTI → PREPROCESSED_DIR
#                        Respects SG_FILTER/PSC/GSR flags and resumes.
#   variance_partition   Seven banded-ridge models and the A/V/J variance
#                        partition for each model: once with A, V, J from the
#                        model itself (tag unimodal_own), once per entry of
#                        OWN_SETS (A and V from the entry's models, tagged with
#                        the entry's tag), and once per control set in
#                        CONTROL_SETS (A and V from the set's models, tagged
#                        with the set name); J is always the model's.
#   screen               Single-feature ridge (tag "screen") for each audio
#                        and video model, in that model's own directory.
#   factorial_interaction  Crossed-pair interaction representation
#
# Defaults: --models pe-av-small-16-frame nemotron_layer18_mp, --bins 5 2 1,
# --variants loro:demean, --controls all, --subsets all seven (a v j av aj vj
# avj); a fit with fewer subsets adds its maps to those already stored.
#
# Excluded subjects
#   Subjects without individual midthickness surfaces are listed in
#   data/excluded.txt and have been removed from data/subjects.txt.
#   175 subjects remain.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

# Raw 7T CIFTI files (input to the preprocess mode)
CIFTI_DIR="/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/individual-59k"

# Preprocessing flags
SG_FILTER=false   # Savitzky-Golay high-pass filter
PSC=false         # Percent signal change normalization
GSR=false         # Global signal regression

# Automatically build PREPROCESSING_FLAG from SG_FILTER/PSC/GSR
PREP_PARTS=()
[ "$SG_FILTER" = "true" ] && PREP_PARTS+=("sg")
[ "$PSC"       = "true" ] && PREP_PARTS+=("psc")
[ "$GSR"       = "true" ] && PREP_PARTS+=("gsr")

if [ ${#PREP_PARTS[@]} -eq 0 ]; then
    PREPROCESSING_FLAG="raw"
else
    PREPROCESSING_FLAG=$(IFS=_; echo "${PREP_PARTS[*]}")
fi

# fMRI data paths
DELAY_SEC=5.0
FMRI_SUFFIX="${PREPROCESSING_FLAG}"
# Group-average preprocessed CIFTI (output of preprocess_individual.py --save-average)
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
# Per-subject preprocessed CIFTIs (output of preprocess_individual.py --save-individual)
# File pattern: {PREPROCESSED_INDIV_DIR}/{sub}_{PREPROCESSING_FLAG}_cortex_59k.dtseries.nii
PREPROCESSED_INDIV_DIR="${DATA_BASE}/preprocessed/${PREPROCESSING_FLAG}"

# Single authoritative subject list — 175 subjects with full 7T fMRI + midthickness.
# All pipeline stages (preprocess / encoding) must read from here.
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# Timing
TIMING_CSV="${DATA_BASE}/movie_timing.csv"

# Embeddings root
# Convention: {EMBEDDINGS_DIR}/{model_name}/bin{B}s_skip{S}s/{model_name}_{modality}.npy
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# CIFTI template whose brain-model axis defines the output grayordinates.
TEMPLATE_CIFTI="/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"

# Output root
OUTPUT_DIR="${OUTPUTS_BASE}/encoding"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
# Window durations (seconds) are swept per invocation with SKIP_SEC = BIN_SEC
# (no overlap), matching the project convention.

# ── Backend ───────────────────────────────────────────────────────────────
# torch_cuda: GPU-accelerated (requires himalaya ≥0.4.11 — older versions
# have a device-mismatch bug when n_samples < n_features)
BACKEND="torch_cuda"

PARTITION_N_ITER=20
PARTITION_MODEL_RANDOM_STATE=0

# Control sets: "name audio_model video_model"; the name is the output tag.
CONTROL_SETS=(
    "mae dasheng-0.6b-d75 videomaev2-large-d75"
    "mae-large dasheng-1.2b videomaev2-giant-d50"
    "latent openbeats-large-i2 vjepa2-vitl-d75"
    "large-mixed dasheng-1.2b vjepa2-vitg-d75"
    "speech-wavlm wavlm-large-d75 vjepa2-vitl-d75"
    "speech-w2vbert w2v-bert-2.0-d75 vjepa2-vitl-d75"
    "text-contrastive clap-larger pe-core-l14"
    "text-asr whisper-large-v3 pe-core-l14"
)

# Further own sets beyond unimodal_own: "model tag audio_model video_model".
OWN_SETS=(
    "pe-av-small-16-frame dummy_av pe-av-small-16-frame_dummy_av pe-av-small-16-frame_dummy_av"
)

MODELS=(pe-av-small-16-frame nemotron_layer18_mp)
BINS=(5 2 1)
VARIANTS=(loro:demean)
LOCO_EXCLUDED="video9,video14,video18"
CONTROLS=(all)
SUBSETS=()
RESPONSE_SCALING=(run)
AUDIO_MODELS=()
VIDEO_MODELS=()
read -ra PAIRING_SEEDS <<< "${PAIRING_SEEDS:-0}"
SEGMENTED_DIR="${MOVIE_SEGMENTED_DIR:-/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/segmented_stimulus/filtered}"

CONDA_ENV="movie"
# =============================================================================

MODE=${1:-variance_partition}
shift || true

parse_options() {
    local target=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --models) MODELS=(); target=MODELS ;;
            --bins) BINS=(); target=BINS ;;
            --variants) VARIANTS=(); target=VARIANTS ;;
            --controls) CONTROLS=(); target=CONTROLS ;;
            --subsets) SUBSETS=(); target=SUBSETS ;;
            --response-scaling) RESPONSE_SCALING=(); target=RESPONSE_SCALING ;;
            --audio-models) AUDIO_MODELS=(); target=AUDIO_MODELS ;;
            --video-models) VIDEO_MODELS=(); target=VIDEO_MODELS ;;
            --*) echo "Unknown option: $1" >&2; exit 1 ;;
            *)
                [ -n "$target" ] || { echo "Unexpected argument: $1" >&2; exit 1; }
                eval "$target+=(\"\$1\")" ;;
        esac
        shift
    done
}
parse_options "$@"

# BIN_SEC/SKIP_SEC are set per iteration of the BINS sweep in DISPATCH below.

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() { conda run -n "$CONDA_ENV" python "$@"; }

# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================
run_preprocess() {
    log "=== Preprocessing n=$(grep -cv '^\s*#' "$SUBJECTS_LIST") subjects → ${PREPROCESSED_INDIV_DIR} ==="
    log "  Flags: SG_FILTER=${SG_FILTER}  PSC=${PSC}  GSR=${GSR}  (${PREPROCESSING_FLAG})"
    log "  Subjects: ${SUBJECTS_LIST}"
    log "  Raw CIFTI dir: ${CIFTI_DIR}"

    local SG_FLAG="" PSC_FLAG="" GSR_FLAG="--no-gsr"
    [ "$SG_FILTER" = "true" ] && SG_FLAG="--sg-filter"
    [ "$PSC"       = "true" ] && PSC_FLAG="--psc"
    [ "$GSR"       = "true" ] && GSR_FLAG="--gsr"

    run_python "${SCRIPT_DIR}/../preprocess_individual.py" \
        --raw-dir        "$CIFTI_DIR" \
        --out-dir        "$PREPROCESSED_INDIV_DIR" \
        --subjects-list  "$SUBJECTS_LIST" \
        --tr             "$TR" \
        $SG_FLAG $PSC_FLAG $GSR_FLAG \
        --save-individual \
        --save-average

    # Move group average to PREPROCESSED_DIR so the group-average fits find it
    local GA_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
    local GA_TRS_SRC="${PREPROCESSED_INDIV_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
    if [ -f "$GA_SRC" ]; then
        mkdir -p "$PREPROCESSED_DIR"
        mv -f "$GA_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"
        [ -f "$GA_TRS_SRC" ] && mv -f "$GA_TRS_SRC" "${PREPROCESSED_DIR}/group_average_${PREPROCESSING_FLAG}_run_trs.npy"
        log "  Group average moved → ${PREPROCESSED_DIR}"
    fi

    log "=== Preprocessing complete ==="
}

FAILURES=0

# Fit every variant with the given extra variance_partition.py arguments; a
# failed fit is logged and the remaining fits continue.
fit_variants() {
    local VARIANT
    for VARIANT in "${VARIANTS[@]}"; do
        log "Fit (${VARIANT%%:*}, ${VARIANT##*:}) $*"
        local SPLIT_ARGS=()
        [ "${VARIANT%%:*}" = loco ] && SPLIT_ARGS=(--exclude-video-ids "$LOCO_EXCLUDED")
        [ ${#SUBSETS[@]} -gt 0 ] && SPLIT_ARGS+=(--subsets "${SUBSETS[@]}")
        run_python "${SCRIPT_DIR}/variance_partition.py" \
            --split "${VARIANT%%:*}" --feature-scaling "${VARIANT##*:}" "${SPLIT_ARGS[@]}" \
            --response-scaling "${RESPONSE_SCALING[0]}" \
            --preprocessed-dir "$PREPROCESSED_DIR" \
            --fmri-suffix "$FMRI_SUFFIX" \
            --subject group_average \
            --timing-csv "$TIMING_CSV" \
            --embeddings-dir "$EMBEDDINGS_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --template-cifti "$TEMPLATE_CIFTI" \
            --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
            --delay-sec "$DELAY_SEC" --tr "$TR" \
            --n-iter "$PARTITION_N_ITER" \
            --model-random-state "$PARTITION_MODEL_RANDOM_STATE" \
            --backend "$BACKEND" \
            "$@" || { log "FAILED: $*"; FAILURES=$((FAILURES + 1)); }
    done
}

run_variance_partition() {
    local MODEL_NAME NAME ENTRY OWN_MODEL SET AUDIO_MODEL VIDEO_MODEL TAGGED
    local SETS=()
    for NAME in "${CONTROLS[@]}"; do
        case "$NAME" in
            none) ;;
            all) SETS=("${CONTROL_SETS[@]}") ;;
            *)
                for ENTRY in "${CONTROL_SETS[@]}"; do
                    [ "${ENTRY%% *}" = "$NAME" ] && { SETS+=("$ENTRY"); continue 2; }
                done
                echo "Unknown control set: $NAME" >&2; exit 1 ;;
        esac
    done
    for MODEL_NAME in "${MODELS[@]}"; do
        fit_variants --model "$MODEL_NAME" --tag unimodal_own
        for ENTRY in "${OWN_SETS[@]}"; do
            read -r OWN_MODEL SET AUDIO_MODEL VIDEO_MODEL <<< "$ENTRY"
            [ "$OWN_MODEL" = "$MODEL_NAME" ] || continue
            fit_variants --model "$MODEL_NAME" --tag "$SET" \
                --audio-model "$AUDIO_MODEL" --video-model "$VIDEO_MODEL"
        done
        for ENTRY in "${SETS[@]}"; do
            read -r SET AUDIO_MODEL VIDEO_MODEL <<< "$ENTRY"
            fit_variants --model "$MODEL_NAME" --tag "$SET" \
                --audio-model "$AUDIO_MODEL" --video-model "$VIDEO_MODEL"
        done
    done
}

run_screen() {
    local BAND FLAG NAMES MODEL_NAME BIN_DIR="bin${BIN_SEC%.*}s_skip${SKIP_SEC%.*}s"
    for BAND in a v; do
        if [ "$BAND" = a ]; then
            FLAG=--audio-model; NAMES=("${AUDIO_MODELS[@]}")
        else
            FLAG=--video-model; NAMES=("${VIDEO_MODELS[@]}")
        fi
        for MODEL_NAME in "${NAMES[@]}"; do
            if [ ! -f "${EMBEDDINGS_DIR}/${MODEL_NAME}/${BIN_DIR}/${MODEL_NAME}_${BAND}.npy" ]; then
                log "SKIP ${MODEL_NAME}: no ${BAND} embeddings at ${BIN_DIR}"
                continue
            fi
            fit_variants --subsets "$BAND" --tag screen "$FLAG" "$MODEL_NAME"
        done
    done
}

run_interaction_control_fit() {
    local MODEL_NAME="$1"
    local OUTPUT_NAME="$2"
    local RESULT_DIR="$3"
    local JOINT_TEMPLATE="${4:-}"
    local TEMPLATE_ARGS=()
    if [ -n "$JOINT_TEMPLATE" ]; then
        TEMPLATE_ARGS=(--joint-model-template "$JOINT_TEMPLATE")
    fi
    run_python "${SCRIPT_DIR}/variance_partition.py" \
        --split loro --feature-scaling demean \
        --preprocessed-dir "$PREPROCESSED_DIR" \
        --fmri-suffix "$FMRI_SUFFIX" \
        --subject group_average \
        --timing-csv "$TIMING_CSV" \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --model "$MODEL_NAME" --tag unimodal_own \
        --output-name "$OUTPUT_NAME" \
        --output-dir "$RESULT_DIR" \
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
        --delay-sec "$DELAY_SEC" --tr "$TR" \
        --n-iter "$PARTITION_N_ITER" \
        --model-random-state "$PARTITION_MODEL_RANDOM_STATE" \
        --backend "$BACKEND" \
        --template-cifti "$TEMPLATE_CIFTI" \
        "${TEMPLATE_ARGS[@]}"
}

run_factorial_interaction() {
    if [ "$BIN_SEC" != "5" ] && [ "$BIN_SEC" != "5.0" ]; then
        log "Factorial interaction currently requires 5-second segments"
        return 1
    fi
    export MOVIE_SEGMENTED_DIR="$SEGMENTED_DIR"
    local MODEL_NAME SEED RUN OUTPUT_NAME TEMPLATE
    for MODEL_NAME in "${MODELS[@]}"; do
        for SEED in "${PAIRING_SEEDS[@]}"; do
            for RUN in 1 2 3 4; do
                if [ "$MODEL_NAME" = "pe-av-small-16-frame" ]; then
                    conda run --no-capture-output -n avtransformer python \
                        "${SCRIPT_DIR}/../notebooks/feature_extraction/pe_av_extract_scramble.py" \
                        --factorial-run "$RUN" --seed "$SEED" --timing-csv "$TIMING_CSV"
                else
                    conda run --no-capture-output -n avtransformer python \
                        "${SCRIPT_DIR}/../notebooks/feature_extraction/nemotron_extract_scramble.py" \
                        --factorial-run "$RUN" --seed "$SEED" --timing-csv "$TIMING_CSV"
                fi
            done
            OUTPUT_NAME="${MODEL_NAME}_interaction_seed${SEED}"
            TEMPLATE="${MODEL_NAME}_interaction_run{run}_seed${SEED}"
            run_interaction_control_fit \
                "$MODEL_NAME" "$OUTPUT_NAME" "${OUTPUT_DIR}/factorial_interaction" "$TEMPLATE"
        done
    done
}

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    preprocess) run_preprocess ;;
    variance_partition|screen|factorial_interaction)
        for BIN_SEC in "${BINS[@]}"; do
            SKIP_SEC="$BIN_SEC"
            log "=== BIN_SEC=${BIN_SEC}s SKIP_SEC=${SKIP_SEC}s ==="
            case "$MODE" in
                variance_partition) run_variance_partition ;;
                screen) run_screen ;;
                factorial_interaction) run_factorial_interaction ;;
            esac
        done
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: preprocess | variance_partition | screen | factorial_interaction" >&2
        exit 1 ;;
esac

log "All encoding analyses complete."
[ "$FAILURES" -eq 0 ] || { log "${FAILURES} fit(s) failed"; exit 1; }
