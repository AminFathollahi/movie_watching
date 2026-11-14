#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"
OUTPUTS_BASE="$ROOT/outputs"

CIFTI_DIR="${DATA_BASE}/individual-59k"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
CONDA_ENV="movie"

OUTPUT_DIR="${OUTPUTS_BASE}/subcortical"
NEIGHBOR_CACHE_DIR="${OUTPUT_DIR}/_neighbor_cache"
TEMPLATE_CIFTI="${OUTPUT_DIR}/subcortical_template.dscalar.nii"

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"

NC_SUBJECTS="${NC_SUBJECTS:-100610 102311 102816 104416 105923 111514 114823 118225 125525 130518}"

MODE=${1:-neighbors}
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
    run_visualize
}

run_visualize() {
    ensure_template
    log "Rebuilding per-structure subcortical Workbench bundle ..."
    run_python "${SCRIPT_DIR}/subcortical_visualization.py" \
        --output-dir "$OUTPUT_DIR" \
        --template "$TEMPLATE_CIFTI" \
        --wb-command "$WORKBENCH"
    log "Workbench bundle complete: ${OUTPUT_DIR}/workbench_visualization/subcortical_wb_view.spec"
}

case "$MODE" in
    neighbors)    run_neighbors ;;
    noiseceiling) run_noiseceiling ;;
    visualize)    run_visualize ;;
    *) echo "Unknown MODE: $MODE (use: neighbors | noiseceiling | visualize)"; exit 1 ;;
esac
