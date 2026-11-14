#!/usr/bin/env bash
# deprecated/cka/analysis.sh
# ==========================
# Master runner for the searchlight centered kernel alignment (CKA) analyses
# (group-average and single-subject data).
#
# Usage
# -----
#   bash deprecated/cka/analysis.sh run|subjects|aggregate|noise-ceiling [--models M...] [--bins B...]
#                         [--variants SCALING...] [--max-vertices N] [--output-dir DIR]
#                         [--controls own|TAG...]   audio and video from the model itself
#                         (own, the default) or from a control pair of CONTROL_SETS, named by its tag
#                         [--limit N] [--k N]   N grayordinates per searchlight
#                         [--space cortex|subcortex]   subcortex: `run` on the subcortical voxels
#                         (subcortical/subcortical_avj.py cka, removed), written under outputs/subcortical/group_average
#
#   run          CKA of the group-average responses with the joint, audio and
#                video embeddings (tag TAG), and the semi-partial and
#                commonality maps, for each model, bin and variant. A run whose
#                outputs exist for every selected model is skipped (the
#                commonality file is written last), so a repeated call resumes.
#   subjects     The maps of `run` from each subject's own responses
#                (resumable). Needs RAW_DIR.
#   aggregate    Mean and standard error of the subject maps.
#   noise-ceiling  Lower and upper bounds on the subject-mean CKA from the
#                binned subject responses (cached in CACHE_DIR, about 24 GB).
#
# Defaults: --models pe-av-small-16-frame nemotron_layer18_mpcat, --bins 5,
# --variants center. --max-vertices 0 evaluates every grayordinate.
# With GPU_LOCK set to a file, every GPU step runs under flock on it.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"
EXTERNAL_BASE="$ROOT/external"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

PREPROCESSING_FLAG="raw"
SUBJECT="group_average"
# Group-average preprocessed CIFTI and run lengths
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
# Raw 7T CIFTI files of the individual subjects (input to subjects)
RAW_DIR="${EXTERNAL_BASE}/data/individual-59k"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"

# Convention: {EMBEDDINGS_DIR}/{model}/bin{B}s_skip{S}s/{model}_av.npy
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# Grayordinate template (cortex, 108441 grayordinates)
TEMPLATE_CIFTI="${PREPROCESSED_DIR}/${SUBJECT}_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"

# Surface geometry and geodesic neighbour cache (shared with deprecated/rsa/analysis.sh)
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"

# Output root and the binned subject responses of noise-ceiling
OUTPUT_DIR="${OUTPUTS_BASE}/cka"
CACHE_DIR="${OUTPUTS_BASE}/cka/raw/intermediate/subject_responses"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
DELAY_SEC=5.0
K=100
# Window stride equals the window length (no overlap).

MODELS=(pe-av-small-16-frame nemotron_layer18_mpcat)
# Control pairs: "audio_model video_model [model glob]"; the output tag is derived from the two models
# (encoding/shared/tags.py). The optional glob restricts the models the pair is run for (the Qwen2.5-VL pair
# is never run for Nemotron, whose backbone is of the same family). Same pairs as encoding/analysis.sh.
CONTROL_SETS=(
    "pe-av-small-16-frame_dummy_av pe-av-small-16-frame_dummy_av pe-av*"
    "dasheng-1.2b videomaev2-giant-d50"
    "dasheng-0.6b-d75 videomaev2-large-d75"
    "openbeats-large-i2 vjepa2-vitl-d75"
    "w2v-bert-2.0-d75 vjepa2-vitg-d75"
    "languagebind-audio languagebind-video"
    "clap-larger viclip-l14"
    "whisper-large-v3 qwen2.5-vl-3b-d75 pe-av*"
    "pe-av-small-16-frame pe-av-small-16-frame nemotron*"
    "nemotron_layer18_mpcat nemotron_layer18_mpcat pe-av*"
)
CONTROLS=(own)
SPACE=cortex
BINS=(5)
VARIANTS=(center)
MAX_VERTICES=0
LIMIT=0

CONDA_ENV="movie"
# =============================================================================

MODE=${1:-run}
shift || true

parse_options() {
    local target=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --models) MODELS=(); target=MODELS ;;
            --bins) BINS=(); target=BINS ;;
            --variants) VARIANTS=(); target=VARIANTS ;;
            --controls) CONTROLS=(); target=CONTROLS ;;
            --max-vertices) MAX_VERTICES="${2:?--max-vertices needs a value}"; target=""; shift ;;
            --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; target=""; shift ;;
            --limit) LIMIT="${2:?--limit needs a value}"; target=""; shift ;;
            --k) K="${2:?--k needs a value}"; target=""; shift ;;
            --space) SPACE="${2:?--space needs a value}"; target=""; shift ;;
            --*) echo "Unknown option: $1" >&2; exit 1 ;;
            *)
                [ -n "$target" ] || { echo "Unexpected argument: $1" >&2; exit 1; }
                eval "$target+=(\"\$1\")" ;;
        esac
        shift
    done
}
parse_options "$@"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

gpu_python() {
    if [ -n "${GPU_LOCK:-}" ]; then flock "$GPU_LOCK" conda run --no-capture-output -n "$CONDA_ENV" python "$@"; else run_python "$@"; fi
}

# The commonality file is the last one a run writes.
run_done() {  # bin variant tag models...
    local bin=$1 variant=$2 tag=$3 model
    shift 3
    [ "$MODE" = run ] && [ "$MAX_VERTICES" -eq 0 ] || return 1
    local folder="$OUTPUT_DIR/$PREPROCESSING_FLAG/$SUBJECT" space=59k
    [ "$SPACE" = subcortex ] && folder="$OUTPUTS_BASE/subcortical/group_average" space=subcortical
    for model in "$@"; do
        [ -f "$folder/$model/cka_${space}_${PREPROCESSING_FLAG}_k${K}_delay${DELAY_SEC%.*}s_bin${bin}s_skip${bin}s_noncv_${variant}_${tag}_commonality.dscalar.nii" ] || return 1
    done
}

tag_of() { run_python "${REPO_DIR}/encoding/shared/tags.py" "$1" "$2"; }

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    run|subjects|aggregate|noise-ceiling) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: run | subjects | aggregate | noise-ceiling" >&2
        exit 1 ;;
esac

FAILURES=0
for BIN_SEC in "${BINS[@]}"; do
    SKIP_SEC="$BIN_SEC"
    log "=== BIN_SEC=${BIN_SEC}s SKIP_SEC=${SKIP_SEC}s ==="
    COMMON=(
        --preprocessed-dir "$PREPROCESSED_DIR" --fmri-suffix "$PREPROCESSING_FLAG" --subject "$SUBJECT"
        --timing-csv "$TIMING_CSV" --template-cifti "$TEMPLATE_CIFTI"
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR"
    )
    INDIVIDUAL=(--subjects-list "$SUBJECTS_LIST" --limit "$LIMIT")
    EXTRA=()
    case "$MODE" in
        noise-ceiling)
            log "CKA noise ceiling bin ${BIN_SEC}"
            gpu_python "${SCRIPT_DIR}/cka_searchlight.py" noise-ceiling "${COMMON[@]}" "${INDIVIDUAL[@]}" --raw-dir "$RAW_DIR" \
                --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
                --workbench "$WORKBENCH" --geodesic-cache-dir "$GEODESIC_CACHE_DIR" --k "$K" --max-vertices "$MAX_VERTICES" \
                || { log "FAILED: noise-ceiling bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
            continue ;;
        subjects) EXTRA=("${INDIVIDUAL[@]}" --raw-dir "$RAW_DIR") ;;
        aggregate) EXTRA=("${INDIVIDUAL[@]}") ;;
    esac
    for VARIANT in "${VARIANTS[@]}"; do
      for NAME in "${CONTROLS[@]}"; do
        # Runs: "models|tag|extra arguments"; own uses each model's own tag.
        RUNS=()
        if [ "$NAME" = own ]; then
            for MODEL in "${MODELS[@]}"; do RUNS+=("${MODEL}|$(tag_of "$MODEL" "$MODEL")|"); done
        else
            FOUND=""
            for ENTRY in "${CONTROL_SETS[@]}"; do
                read -r AUDIO_MODEL VIDEO_MODEL PATTERN <<< "$ENTRY"
                [ "$(tag_of "$AUDIO_MODEL" "$VIDEO_MODEL")" = "$NAME" ] || continue
                FOUND=1
                SELECTED=()
                for MODEL in "${MODELS[@]}"; do [[ "$MODEL" == ${PATTERN:-*} ]] && SELECTED+=("$MODEL"); done
                [ "${#SELECTED[@]}" -eq 0 ] || RUNS+=("${SELECTED[*]}|${NAME}|--audio-model ${AUDIO_MODEL} --video-model ${VIDEO_MODEL}")
            done
            [ -n "$FOUND" ] || { echo "Unknown control pair: $NAME" >&2; exit 1; }
        fi
        for RUN in "${RUNS[@]}"; do
            IFS='|' read -r RUN_MODELS RUN_TAG RUN_EXTRA <<< "$RUN"
            read -ra RUN_MODEL_ARR <<< "$RUN_MODELS"
            read -ra RUN_EXTRA_ARR <<< "$RUN_EXTRA"
            if run_done "$BIN_SEC" "$VARIANT" "$RUN_TAG" "${RUN_MODEL_ARR[@]}"; then
                log "done: ${VARIANT} ${RUN_TAG} bin ${BIN_SEC} (${RUN_MODELS})"
                continue
            fi
            log "CKA ${MODE} (${VARIANT}, ${RUN_TAG}) ${RUN_MODELS}"
            if [ "$SPACE" = subcortex ]; then
                [ "$MODE" = run ] || { echo "--space subcortex supports run only" >&2; exit 1; }
                gpu_python "${REPO_DIR}/subcortical/subcortical_avj.py" cka \
                    --models "${RUN_MODEL_ARR[@]}" --group-average-dir "$OUTPUTS_BASE/subcortical/group_average_cache" \
                    --timing-csv "$TIMING_CSV" --embeddings-dir "$EMBEDDINGS_DIR" \
                    --template-cifti "$OUTPUTS_BASE/subcortical/subcortical_template.dscalar.nii" \
                    --output-dir "$OUTPUTS_BASE/subcortical" --neighbor-cache-dir "$OUTPUTS_BASE/subcortical/_neighbor_cache" \
                    --workbench "$WORKBENCH" --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                    --model-norm "$VARIANT" --tag "$RUN_TAG" "${RUN_EXTRA_ARR[@]}" \
                    || { log "FAILED: ${MODE} ${VARIANT} ${RUN_TAG} bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
                continue
            fi
            gpu_python "${SCRIPT_DIR}/cka_searchlight.py" "$MODE" "${COMMON[@]}" "${EXTRA[@]}" \
                --models "${RUN_MODEL_ARR[@]}" --embeddings-dir "$EMBEDDINGS_DIR" --output-dir "$OUTPUT_DIR" \
                --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
                --workbench "$WORKBENCH" --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
                --k "$K" --feature-scaling "$VARIANT" --tag "$RUN_TAG" "${RUN_EXTRA_ARR[@]}" --max-vertices "$MAX_VERTICES" \
                || { log "FAILED: ${MODE} ${VARIANT} ${RUN_TAG} bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
        done
      done
    done
done

log "All CKA analyses complete."
[ "$FAILURES" -eq 0 ] || { log "${FAILURES} step(s) failed"; exit 1; }
