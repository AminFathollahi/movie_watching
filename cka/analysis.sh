#!/usr/bin/env bash
# cka/analysis.sh
# ==========================
# Master runner for the searchlight centered kernel alignment (CKA) analyses
# (group-average data).
#
# Usage
# -----
#   bash cka/analysis.sh partitions [--bins B...]
#   bash cka/analysis.sh run [--models M...] [--bins B...] [--variants SCALING...]
#                         [--max-vertices N] [--output-dir DIR]
#
#   partitions   Averages the preprocessed responses of N_PARTITIONS disjoint
#                subject groups from RAW_DIR into one array per bin
#                (resumable; written to PARTITIONS_DIR).
#   run          Non-cross-validated CKA (diagnostics), cross-validated CKA with
#                and without autoregressive whitening, their log-likelihoods and
#                the log-likelihood differences between the models, for each
#                model, bin and variant. Needs the partitions of the same bin.
#
# Defaults: --models pe-av-small-16-frame nemotron_layer18_mp, --bins 5,
# --variants center. --max-vertices 0 evaluates every grayordinate.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG — all paths and analysis parameters defined here
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
EXTERNAL_BASE="/media/amin/ADATA HD710 PRO/Research/Representation/Movie"
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
BINS=(5)
VARIANTS=(center)
MAX_VERTICES=0

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
            --max-vertices) MAX_VERTICES="${2:?--max-vertices needs a value}"; target=""; shift ;;
            --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; target=""; shift ;;
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
    partitions|run) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Use: partitions | run" >&2
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
    case "$MODE" in
        partitions)
            run_python "${SCRIPT_DIR}/cka_searchlight.py" partitions "${COMMON[@]}" \
                --raw-dir "$RAW_DIR" --subjects-list "$SUBJECTS_LIST" \
                || { log "FAILED: partitions bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
            ;;
        run)
            for VARIANT in "${VARIANTS[@]}"; do
                log "CKA (${VARIANT}) ${MODELS[*]}"
                run_python "${SCRIPT_DIR}/cka_searchlight.py" run "${COMMON[@]}" \
                    --models "${MODELS[@]}" --embeddings-dir "$EMBEDDINGS_DIR" --output-dir "$OUTPUT_DIR" \
                    --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
                    --workbench "$WORKBENCH" --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
                    --k "$K" --feature-scaling "$VARIANT" --max-vertices "$MAX_VERTICES" \
                    || { log "FAILED: ${VARIANT} bin ${BIN_SEC}"; FAILURES=$((FAILURES + 1)); }
            done
            ;;
    esac
done

log "All CKA analyses complete."
[ "$FAILURES" -eq 0 ] || { log "${FAILURES} step(s) failed"; exit 1; }
