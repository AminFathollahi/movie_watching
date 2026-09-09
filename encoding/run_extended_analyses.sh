#!/usr/bin/env bash
# encoding/run_extended_analyses.sh
# ====================================
# Encoding-currency analogue of rsa/run_extended_analyses.sh: group-average
# encoding (encoding.py, modality=av) for every _av_linear_resid_unimodal /
# _av_projection_resid_own / _av_linear_resid_encoder pseudo-model already
# generated on disk (run "bash rsa/run_extended_analyses.sh embed" first).
# Group-average only, per explicit instruction -- no per-subject sweep.
#
# Usage
#   bash encoding/run_extended_analyses.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
BIN_SEC=5.0

log() { echo "[$(date +%H:%M:%S)] $*"; }

_discover_models_str() {
    local MODELS_STR="" d name
    for d in "$EMBEDDINGS_DIR"/*_av_linear_resid_unimodal "$EMBEDDINGS_DIR"/*_av_projection_resid_own "$EMBEDDINGS_DIR"/*_av_linear_resid_encoder; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        MODELS_STR="${MODELS_STR}${name}:av;"
    done
    echo "${MODELS_STR%;}"
}

MODELS_STR=$(_discover_models_str)
if [ -z "$MODELS_STR" ]; then
    log "No pseudo-model embeddings found on disk -- run 'bash rsa/run_extended_analyses.sh embed' first."
    exit 1
fi
log "=== Extended analyses PLAIN encoding (group-average): $(echo "$MODELS_STR" | tr ';' '\n' | wc -l) pseudo-models ==="
BIN_SECS="$BIN_SEC" ENCODING_MODELS_OVERRIDE="$MODELS_STR" \
    bash "${SCRIPT_DIR}/analysis.sh" avg
log "=== Extended analyses encoding complete ==="
