#!/usr/bin/env bash
# Run a resumable batch of AV-pairing permutations and aggregate completed maps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

usage() {
    cat <<'EOF'
Usage:
  bash rsa/run_av_scramble_permutations.sh --batch-size N [options]
  bash rsa/run_av_scramble_permutations.sh --seeds S... [options]
  bash rsa/run_av_scramble_permutations.sh --aggregate-only [options]

Options:
  --batch-size N             Run the first N missing seeds in 1..TOTAL.
  --total-permutations N     Target null size (default: 500).
  --seeds S...               Run explicit seeds; each must be in 1..TOTAL.
  --alpha A                  Threshold for saved masks (default: 0.01).
  --aggregate-only           Rebuild outputs without generating permutations.
  --require-complete         Fail aggregation unless all TOTAL maps exist.
  --no-aggregate             Do not rebuild outputs after this batch.
  --dry-run                  Print the selected seeds and exit.
  -h, --help                 Show this help.

Examples:
  bash rsa/run_av_scramble_permutations.sh --batch-size 50
  bash rsa/run_av_scramble_permutations.sh --batch-size 100
  bash rsa/run_av_scramble_permutations.sh --seeds 401 450 499
  bash rsa/run_av_scramble_permutations.sh --aggregate-only --require-complete
EOF
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"
PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${PREPROCESSED_DIR}/group_average_raw_cortex_59k.dtseries.nii"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"

INTACT_MODEL="pe-av-small-16-frame"
SCRAMBLE_BASE="pe-av-small-16-frame_avscramble"
MODALITY="av"
K=100
BIN_SEC=5.0
DELAY_SEC=5.0
METHOD="spearman"
export MODEL_NORM="${MODEL_NORM:-center}"

ALPHA=0.01
TOTAL_PERMUTATIONS=500
BATCH_SIZE=""
AGGREGATE_ONLY=false
REQUIRE_COMPLETE=false
DO_AGGREGATE=true
DRY_RUN=false
SEEDS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
        --total-permutations) TOTAL_PERMUTATIONS="${2:?--total-permutations requires a value}"; shift 2 ;;
        --alpha) ALPHA="${2:?--alpha requires a value}"; shift 2 ;;
        --aggregate-only) AGGREGATE_ONLY=true; shift ;;
        --require-complete) REQUIRE_COMPLETE=true; shift ;;
        --no-aggregate) DO_AGGREGATE=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --seeds)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$TOTAL_PERMUTATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "TOTAL must be a positive integer." >&2; exit 2; }
if [ -n "$BATCH_SIZE" ]; then
    [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be a positive integer." >&2; exit 2; }
fi
if [ "$AGGREGATE_ONLY" = false ] && [ -z "$BATCH_SIZE" ] && [ ${#SEEDS[@]} -eq 0 ]; then
    echo "Specify --batch-size, --seeds, or --aggregate-only." >&2
    usage >&2
    exit 2
fi
if [ -n "$BATCH_SIZE" ] && [ ${#SEEDS[@]} -gt 0 ]; then
    echo "Use either --batch-size or --seeds, not both." >&2
    exit 2
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

model_for_seed() {
    if [ "$1" -eq 42 ]; then echo "$SCRAMBLE_BASE"; else echo "${SCRAMBLE_BASE}_seed${1}"; fi
}

rho_path_for_seed() {
    local model
    model="$(model_for_seed "$1")"
    echo "${OUTPUT_DIR}/group_average/${model}_${MODALITY}/k${K}_delay5s_bin5s_skip5s_${METHOD}_${MODEL_NORM}/rsa_59k_${FMRI_SUFFIX}_k${K}_delay5s_bin5s_skip5s_${METHOD}_${MODEL_NORM}_searchlight.npy"
}

if [ -n "$BATCH_SIZE" ]; then
    for SEED in $(seq 1 "$TOTAL_PERMUTATIONS"); do
        if [ ! -f "$(rho_path_for_seed "$SEED")" ]; then
            SEEDS+=("$SEED")
            [ ${#SEEDS[@]} -ge "$BATCH_SIZE" ] && break
        fi
    done
fi

for SEED in "${SEEDS[@]}"; do
    [[ "$SEED" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid seed: $SEED" >&2; exit 2; }
    [ "$SEED" -le "$TOTAL_PERMUTATIONS" ] || { echo "Seed $SEED exceeds target $TOTAL_PERMUTATIONS." >&2; exit 2; }
done

if [ "$AGGREGATE_ONLY" = false ]; then
    log "=== Batch has ${#SEEDS[@]} seed(s); target is ${TOTAL_PERMUTATIONS} ==="
    log "selected seeds: ${SEEDS[*]:-(none; target already complete)}"
fi
if [ "$DRY_RUN" = true ]; then
    exit 0
fi

for SEED in "${SEEDS[@]}"; do
    MODEL="$(model_for_seed "$SEED")"
    EMB_PATH="${EMBEDDINGS_DIR}/${MODEL}/bin5s_skip5s/${MODEL}_av.npy"
    RHO_PATH="$(rho_path_for_seed "$SEED")"

    if [ -f "$RHO_PATH" ]; then
        log "seed=${SEED}: complete; skipping."
        continue
    fi

    if [ ! -f "$EMB_PATH" ]; then
        log "seed=${SEED}: extracting scrambled AV embedding."
        conda run --no-capture-output -n avtransformer \
            python notebooks/feature_extraction/pe_av_extract_scramble.py \
            --scramble-av --seed "$SEED"
    else
        log "seed=${SEED}: embedding exists; skipping extraction."
    fi

    log "seed=${SEED}: running group-average searchlight RSA."
    BIN_SECS="$BIN_SEC" DELAY_SEC="$DELAY_SEC" K="$K" METHOD="$METHOD" MODEL_NORMS="$MODEL_NORM" \
        RSA_MODELS_OVERRIDE="${MODEL}:${MODALITY}" \
        bash "${SCRIPT_DIR}/analysis.sh" avg searchlight
done

if [ "$DO_AGGREGATE" = true ]; then
    log "=== Aggregating completed permutations in 1..${TOTAL_PERMUTATIONS} ==="
    INFERENCE_ARGS=(
        --output-dir "$OUTPUT_DIR"
        --intact-model "$INTACT_MODEL"
        --scramble-base "$SCRAMBLE_BASE"
        --total-permutations "$TOTAL_PERMUTATIONS"
        --modality "$MODALITY"
        --k "$K"
        --bin-sec "$BIN_SEC"
        --delay-sec "$DELAY_SEC"
        --method "$METHOD"
        --model-norm "$MODEL_NORM"
        --fmri-tag "$FMRI_SUFFIX"
        --template-cifti "$TEMPLATE_CIFTI"
        --alpha "$ALPHA"
    )
    [ "$REQUIRE_COMPLETE" = true ] && INFERENCE_ARGS+=(--require-complete)
    conda run --no-capture-output -n movie \
        python "${SCRIPT_DIR}/av_scramble_permutation_inference.py" "${INFERENCE_ARGS[@]}"
fi

log "=== AV-scramble batch complete ==="
