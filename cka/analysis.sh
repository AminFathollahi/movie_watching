#!/usr/bin/env bash
# cka/analysis.sh
# ==========================
# Master runner for the searchlight centered kernel alignment (CKA) analyses
# (group-average data).
#
# Usage
# -----
#   bash cka/analysis.sh partitions [--bins B...]
#   bash cka/analysis.sh run|noncv|subjects|aggregate|commonality [--models M...] [--bins B...]
#                         [--variants SCALING...] [--max-vertices N] [--output-dir DIR]
#                         [--controls own|SET...]   audio and video from the model itself
#                         (own, the default) or from a control set of CONTROL_SETS
#                         [--limit N] [--k N]   N grayordinates per searchlight
#
#   partitions   Averages the preprocessed responses of N_PARTITIONS disjoint
#                subject groups from RAW_DIR into one array per bin
#                (resumable; written to PARTITIONS_DIR).
#   run          Non-cross-validated CKA, cross-validated CKA with
#                and without whitening by the window noise covariance of the
#                joint, audio and video embeddings (tag TAG), and the
#                semi-partial and commonality maps, for each model, bin and
#                variant. Needs the partitions of the same bin.
#   noncv        Only the non-cross-validated CKA, semi-partial and commonality maps of `run`.
#   subjects     Cross-validated CKA of the joint, audio and video embeddings
#                for every subject against the other subjects (resumable).
#                Needs RAW_DIR and the noise model written by `run`.
#   aggregate    Semi-partial and commonality maps of every subject, and the
#                mean, standard error and random-effects maps over subjects.
#   commonality  Commonality maps of the stored group-average CKA maps (no searchlight).
#
# Defaults: --models pe-av-small-16-frame nemotron_layer18_mp, --bins 5,
# --variants center. --max-vertices 0 evaluates every grayordinate.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"
EXTERNAL_BASE="$ROOT/external"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

PREPROCESSING_FLAG="raw"
SUBJECT="group_average"
# Group-average preprocessed CIFTI and run lengths
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/${PREPROCESSING_FLAG}"
# Raw 7T CIFTI files of the individual subjects (input to partitions)
RAW_DIR="${EXTERNAL_BASE}/data/individual-59k"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"

# Convention: {EMBEDDINGS_DIR}/{model}/bin{B}s_skip{S}s/{model}_av.npy
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"

# Grayordinate template (cortex, 108441 grayordinates)
TEMPLATE_CIFTI="${PREPROCESSED_DIR}/${SUBJECT}_${PREPROCESSING_FLAG}_cortex_59k.dtseries.nii"

# Surface geometry and geodesic neighbour cache (shared with rsa/analysis.sh)
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"

# Output root; the partition arrays (6.8 GB each) stay off the home SSD
OUTPUT_DIR="${OUTPUTS_BASE}/cka"
PARTITIONS_DIR="${EXTERNAL_BASE}/outputs/cka/partitions"

# ── Analysis parameters ────────────────────────────────────────────────────
TR=1.0
DELAY_SEC=5.0
K=100
N_PARTITIONS=25
# Window stride equals the window length (no overlap).

MODELS=(pe-av-small-16-frame nemotron_layer18_mp)
# Control sets: "name audio_model video_model"; the name is the output tag.
# Same sets as encoding/analysis.sh.
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
CONTROLS=(own)
BINS=(5)
VARIANTS=(center)
TAG="unimodal_own"
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

# =============================================================================
# DISPATCH
# =============================================================================
case "$MODE" in
    partitions|run|noncv|subjects|aggregate|commonality) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: partitions | run | noncv | subjects | aggregate | commonality" >&2
        exit 1 ;;
esac

FAILURES=0
for BIN_SEC in "${BINS[@]}"; do
    SKIP_SEC="$BIN_SEC"
    log "=== BIN_SEC=${BIN_SEC}s SKIP_SEC=${SKIP_SEC}s ==="
    COMMON=(
        --preprocessed-dir "$PREPROCESSED_DIR" --fmri-suffix "$PREPROCESSING_FLAG" --subject "$SUBJECT"
        --timing-csv "$TIMING_CSV" --template-cifti "$TEMPLATE_CIFTI"
        --partitions-dir "$PARTITIONS_DIR" --n-partitions "$N_PARTITIONS"
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR"
    )
    INDIVIDUAL=(--subjects-list "$SUBJECTS_LIST" --limit "$LIMIT")
    EXTRA=()
    case "$MODE" in
        partitions)
            run_python "${SCRIPT_DIR}/cka_searchlight.py" partitions "${COMMON[@]}" "${INDIVIDUAL[@]}" --raw-dir "$RAW_DIR" \
                || { log "FAILED: partitions bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
            continue ;;
        subjects) EXTRA=("${INDIVIDUAL[@]}" --raw-dir "$RAW_DIR") ;;
        aggregate) EXTRA=("${INDIVIDUAL[@]}") ;;
    esac
    for VARIANT in "${VARIANTS[@]}"; do
      for NAME in "${CONTROLS[@]}"; do
        SOURCE=(--tag "$TAG")
        if [ "$NAME" != own ]; then
            ENTRY=""
            for SET in "${CONTROL_SETS[@]}"; do [ "${SET%% *}" = "$NAME" ] && ENTRY="$SET"; done
            [ -n "$ENTRY" ] || { echo "Unknown control set: $NAME" >&2; exit 1; }
            read -r SET AUDIO_MODEL VIDEO_MODEL <<< "$ENTRY"
            SOURCE=(--tag "$SET" --audio-model "$AUDIO_MODEL" --video-model "$VIDEO_MODEL")
        fi
        log "CKA ${MODE} (${VARIANT}, ${NAME}) ${MODELS[*]}"
        run_python "${SCRIPT_DIR}/cka_searchlight.py" "$MODE" "${COMMON[@]}" "${EXTRA[@]}" \
            --models "${MODELS[@]}" --embeddings-dir "$EMBEDDINGS_DIR" --output-dir "$OUTPUT_DIR" \
            --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
            --workbench "$WORKBENCH" --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
            --k "$K" --feature-scaling "$VARIANT" "${SOURCE[@]}" --max-vertices "$MAX_VERTICES" \
            || { log "FAILED: ${MODE} ${VARIANT} ${NAME} bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
      done
    done
done

log "All CKA analyses complete."
[ "$FAILURES" -eq 0 ] || { log "${FAILURES} step(s) failed"; exit 1; }
