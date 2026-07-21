#!/usr/bin/env bash
# subcortical/analysis.sh
# ========================
# Master runner for the subcortical RSA extension. Mirrors rsa/analysis.sh
# conventions (subject-list handling, output-dir layout, conda env) but the
# subcortical voxel grid is subject-invariant, so neighbor precompute runs
# ONCE (no per-subject loop) — see subcortical_io.py.
#
# Usage
#   bash subcortical/analysis.sh MODE
#
#   MODE   neighbors     Precompute + cache k=100 neighbor arrays for every
#                         structure (voxel-graph geodesic; purist SUIT
#                         surface geodesic for cerebellum). Run once before
#                         anything else.
#          groupavg      PRIMARY FIRST PASS. Build the group-average
#                         subcortical timeseries (reuses SUBJECTS_LIST) and
#                         run the k=100 searchlight + nuclei ROI-RSA on it.
#          noiseceiling  Vertex-wise inter-subject noise ceiling on a subset
#                         of subjects (streaming mode — reads raw CIFTIs
#                         directly, mirrors the cortical noise-ceiling
#                         invocation). Mandatory reliability gate (subcortex.txt §4).
#          persubject    Per-subject searchlight + nuclei RSA for every subject
#                         in SUBJECTS_LIST (streaming preprocessing each time —
#                         subcortex has no PREPROCESSED_INDIV_DIR cache).
#                         Sequential (single GPU); resumable via subcortical_rsa.py's
#                         existing skip-if-outputs-exist check.
#          groupstats    Across-subject t-test (rsa/group_stats.py, unmodified
#                         inference code, --fname-prefix rsa_subcortical) over
#                         every per-subject map produced by `persubject`.
#

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

CIFTI_DIR="${DATA_BASE}/individual-59k"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"          # same authoritative roster as rsa/analysis.sh
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
CONDA_ENV="movie"

OUTPUT_DIR="${OUTPUTS_BASE}/subcortical"
NEIGHBOR_CACHE_DIR="${OUTPUT_DIR}/_neighbor_cache"
GROUP_AVERAGE_DIR="${OUTPUT_DIR}/group_average_cache"

# Template CIFTI built by subcortical_io.make_subcortical_template
TEMPLATE_CIFTI="${OUTPUT_DIR}/subcortical_template.dscalar.nii"

# ── Analysis parameters (match established cortical run per subcortex.txt §1) ──
MODEL="pe-av-small-16-frame"
MODALITY="av"
K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"

# Noise-ceiling subject subset (fast streaming-mode reliability check; not
# the full 176-subject roster — mirrors how the cortical noise ceiling was run).
NC_SUBJECTS="${NC_SUBJECTS:-100610 102311 102816 104416 105923 111514 114823 118225 125525 130518}"

MODE=${1:-groupavg}
run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

ensure_template() {
    if [ ! -f "$TEMPLATE_CIFTI" ]; then
        log "Building subcortical template CIFTI ..."
        run_python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from subcortical_io import make_subcortical_template, SUBCORTICAL_STRUCTURES
from pathlib import Path
make_subcortical_template(SUBCORTICAL_STRUCTURES, Path('${TEMPLATE_CIFTI}'), Path('${CIFTI_DIR}'))
"
    fi
}

run_neighbors() {
    log "Precomputing subcortical neighbor caches (k=${K}) ..."
    run_python "${SCRIPT_DIR}/precompute_neighbors.py" \
        --raw-dir "$CIFTI_DIR" --subject 132118 --k "$K" \
        --cache-dir "$NEIGHBOR_CACHE_DIR" --workbench "$WORKBENCH"
    log "Neighbor precompute complete."
}

run_groupavg() {
    ensure_template
    log "Group-average subcortical RSA: ${MODEL}/${MODALITY}"
    run_python "${SCRIPT_DIR}/subcortical_rsa.py" \
        --raw-dir "$CIFTI_DIR" \
        --subjects-list "$SUBJECTS_LIST" \
        --subject "group_average" \
        --timing-csv "$TIMING_CSV" \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --template-cifti "$TEMPLATE_CIFTI" \
        --output-dir "$OUTPUT_DIR" \
        --group-average-dir "$GROUP_AVERAGE_DIR" \
        --neighbor-cache-dir "$NEIGHBOR_CACHE_DIR" \
        --workbench "$WORKBENCH" \
        --model "$MODEL" --modality "$MODALITY" \
        --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
        --method "$METHOD" --tr "$TR"
    log "Group-average subcortical RSA complete."
}

run_noiseceiling() {
    ensure_template
    log "Subcortical noise ceiling (streaming, N=$(echo $NC_SUBJECTS | wc -w) subjects) ..."
    run_python "${SCRIPT_DIR}/subcortical_noise_ceiling.py" \
        --raw-dir "$CIFTI_DIR" \
        --subjects $NC_SUBJECTS \
        --timing-csv "$TIMING_CSV" \
        --template-cifti "$TEMPLATE_CIFTI" \
        --neighbor-cache-dir "$NEIGHBOR_CACHE_DIR" \
        --workbench "$WORKBENCH" \
        --output-dir "${OUTPUT_DIR}/noise_ceiling" \
        --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
        --method "$METHOD" --tr "$TR"
    log "Noise ceiling complete."
}

run_persubject() {
    ensure_template
    local SUBJECTS
    SUBJECTS=$(grep -v '^\s*#' "$SUBJECTS_LIST" | sed 's/#.*//' | awk '{print $1}' | grep -v '^$')
    local N_TOTAL; N_TOTAL=$(echo "$SUBJECTS" | wc -l)
    log "Per-subject subcortical RSA: ${N_TOTAL} subjects (sequential, streaming preprocessing) ..."
    local i=0
    for SUB in $SUBJECTS; do
        i=$((i + 1))
        log "  [$i/$N_TOTAL] ${SUB}"
        run_python "${SCRIPT_DIR}/subcortical_rsa.py" \
            --raw-dir "$CIFTI_DIR" \
            --subject "$SUB" \
            --timing-csv "$TIMING_CSV" \
            --embeddings-dir "$EMBEDDINGS_DIR" \
            --template-cifti "$TEMPLATE_CIFTI" \
            --output-dir "$OUTPUT_DIR" \
            --neighbor-cache-dir "$NEIGHBOR_CACHE_DIR" \
            --workbench "$WORKBENCH" \
            --model "$MODEL" --modality "$MODALITY" \
            --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
            --method "$METHOD" --tr "$TR" \
            || log "  WARNING: ${SUB} failed — continuing"
    done
    log "Per-subject subcortical RSA complete."
}

run_groupstats() {
    ensure_template
    log "Subcortical group stats (across-subject t-test): ${MODEL}/${MODALITY}"
    run_python "${SCRIPT_DIR}/../rsa/group_stats.py" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" --modality "$MODALITY" \
        --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
        --method "$METHOD" \
        --fmri-tag "raw" \
        --fname-prefix "rsa_subcortical" \
        --template-cifti "$TEMPLATE_CIFTI" \
        --left-surface "unused" --right-surface "unused" \
        --n-blocks 16
    log "Subcortical group stats complete."
}

case "$MODE" in
    neighbors)    run_neighbors ;;
    groupavg)     run_groupavg ;;
    noiseceiling) run_noiseceiling ;;
    persubject)   run_persubject ;;
    groupstats)   run_groupstats ;;
    *) echo "Unknown MODE: $MODE (use: neighbors | groupavg | noiseceiling | persubject | groupstats)"; exit 1 ;;
esac
