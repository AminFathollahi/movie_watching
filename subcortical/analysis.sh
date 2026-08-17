#!/usr/bin/env bash
# subcortical/analysis.sh
# ========================
# Master runner for the subcortical RSA extension. Mirrors rsa/analysis.sh
# conventions (subject-list handling, output-dir layout, conda env) but the
# subcortical voxel grid is subject-invariant, so neighbor precompute runs
# ONCE (no per-subject loop) — see subcortical_io.py.
#
# Usage
#   bash subcortical/analysis.sh MODE [N_BLOCKS]
#
#   N_BLOCKS   groupstats df control (default 16), same convention as
#              rsa/analysis.sh's positional N_BLOCKS override.
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

# ── Model registry (mirrors rsa/analysis.sh's MODELS + LAYERS convention) ──
# "model:modality,modality" — uncomment/add entries to sweep more.
MODELS=(
    "pe-av-small-16-frame:av"
    "pe-av-small-16-frame_avscramble:av"
    "pe-av-small-16-frame_clsav_from_a:av"
    "pe-av-small-16-frame_clsav_from_v:av"
)

# omni3b / topoomni layer sweep — mirrors rsa/analysis.sh's LAYERS block (this
# was previously missing the omni3b intact entries -- only topoomni's intact
# was added here, so omni3b_layer*_mp/lt never got a plain group-average RSA
# run outside of its scramble/dummy conditions).
LAYERS=(35 27 18 9 1)
OMNI3B_MODALITIES="av"
TOPOOMNI_MODALITIES="av"
if [ "${SKIP_LAYER_SWEEP:-false}" != "true" ]; then
    for L in "${LAYERS[@]}"; do
        case "$L" in
            9|18|27) SUFFIX="_mp" ;;
            *)       SUFFIX="" ;;
        esac
        MODELS+=("omni3b_layer${L}${SUFFIX}:${OMNI3B_MODALITIES}")
        MODELS+=("topoomni_layer${L}${SUFFIX}:${TOPOOMNI_MODALITIES}")
        MODELS+=("topoomni_layer${L}_sheet${SUFFIX}:${TOPOOMNI_MODALITIES}")
        # Layers 9/18/27 also have a "_lt" (last-token) intact readout, needed
        # by diff_maps.py's mp+lt roster (nemotron has no _lt -- see below).
        if [ "$SUFFIX" = "_mp" ]; then
            MODELS+=("omni3b_layer${L}_lt:${OMNI3B_MODALITIES}")
            MODELS+=("topoomni_layer${L}_lt:${TOPOOMNI_MODALITIES}")
            MODELS+=("topoomni_layer${L}_sheet_lt:${TOPOOMNI_MODALITIES}")
        fi
    done

    # nemotron (omni-embed-nemotron-3b): no bare layer1/2/4/35 probes exist on
    # disk (unlike omni3b/topoomni) -- only the 9/18/27/36 depth-sweep. Its
    # native "_av" readout ("_mp") is already genuinely joint from the start,
    # so "_lt" is never used as an INTEGRATION target (see model_registry.py)
    # -- but diff_maps.py's plain-RSA roster still tracks "_lt" as a
    # depth-comparability probe, so both poolings get an intact plain run.
    NEMOTRON_MODALITIES="av"
    for L in 9 18 27 36; do
        MODELS+=("nemotron_layer${L}_mp:${NEMOTRON_MODALITIES}")
        MODELS+=("nemotron_layer${L}_lt:${NEMOTRON_MODALITIES}")
    done
fi

# ── Layer-swept native multimodal models: avscramble + avdummy ────────────
# nemotron / omni3b / topoomni(+_sheet) each build their own joint AV
# embedding at a given layer, pooled two ways (_mp mean-pool, _lt last-token
# — both are independent conditions per the mp/lt naming migration, kept
# side by side rather than picking one). Only layers with a pooled variant
# on disk have scramble/dummy embeddings (9/18/27; nemotron also has 36).
NATIVE_MM_LAYER_BASES=(
    nemotron_layer9 nemotron_layer18 nemotron_layer27 nemotron_layer36
    omni3b_layer9 omni3b_layer18 omni3b_layer27
    topoomni_layer9 topoomni_layer18 topoomni_layer27
    topoomni_layer9_sheet topoomni_layer18_sheet topoomni_layer27_sheet
)
POOLING=(mp lt)
if [ "${SKIP_SCRAMBLE_DUMMY_SWEEP:-false}" != "true" ]; then
    for BASE in "${NATIVE_MM_LAYER_BASES[@]}"; do
        for POOL in "${POOLING[@]}"; do
            NAME="${BASE}_${POOL}"
            MODELS+=("${NAME}_avscramble:av")
            MODELS+=("${NAME}_clsav_from_a:av")
            MODELS+=("${NAME}_clsav_from_v:av")
        done
    done
fi

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
METHOD="spearman"

# groupstats df control: df = min(n_subjects-1, N_BLOCKS-1)
N_BLOCKS=16

# Noise-ceiling subject subset (fast streaming-mode reliability check; not
# the full 176-subject roster — mirrors how the cortical noise ceiling was run).
NC_SUBJECTS="${NC_SUBJECTS:-100610 102311 102816 104416 105923 111514 114823 118225 125525 130518}"

MODE=${1:-groupavg}
N_BLOCKS_ARG=${2:-""}
[ -n "$N_BLOCKS_ARG" ] && N_BLOCKS="$N_BLOCKS_ARG"
run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
refresh_visualization_if_enabled() {
    if [ "${REFRESH_VISUALIZATION:-true}" = "true" ]; then
        run_visualize
    fi
}

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
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Group-average subcortical RSA: ${MODEL_NAME}/${MOD}"
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
                --model "$MODEL_NAME" --modality "$MOD" \
                --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
                --method "$METHOD" --tr "$TR"
        done
    done
    log "Group-average subcortical RSA complete."
    refresh_visualization_if_enabled
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
    refresh_visualization_if_enabled
}

run_persubject() {
    ensure_template
    local SUBJECTS
    SUBJECTS=$(grep -v '^\s*#' "$SUBJECTS_LIST" | sed 's/#.*//' | awk '{print $1}' | grep -v '^$')
    local N_TOTAL; N_TOTAL=$(echo "$SUBJECTS" | wc -l)
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Per-subject subcortical RSA: ${MODEL_NAME}/${MOD} — ${N_TOTAL} subjects (sequential, streaming preprocessing) ..."
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
                    --model "$MODEL_NAME" --modality "$MOD" \
                    --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
                    --method "$METHOD" --tr "$TR" \
                    || log "  WARNING: ${SUB} failed — continuing"
            done
        done
    done
    log "Per-subject subcortical RSA complete."
}

run_groupstats() {
    ensure_template
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Subcortical group stats (across-subject t-test): ${MODEL_NAME}/${MOD} (n_blocks=${N_BLOCKS})"
            run_python "${SCRIPT_DIR}/../rsa/group_stats.py" \
                --output-dir "$OUTPUT_DIR" \
                --model "$MODEL_NAME" --modality "$MOD" \
                --k "$K" --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" \
                --method "$METHOD" \
                --fmri-tag "raw" \
                --fname-prefix "rsa_subcortical" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --left-surface "unused" --right-surface "unused" \
                --n-blocks "$N_BLOCKS" \
                || log "  WARNING: group stats skipped for ${MODEL_NAME}/${MOD}"
        done
    done
    log "Subcortical group stats complete."
    refresh_visualization_if_enabled
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
    groupavg)     run_groupavg ;;
    noiseceiling) run_noiseceiling ;;
    persubject)   run_persubject ;;
    groupstats)   run_groupstats ;;
    visualize)    run_visualize ;;
    *) echo "Unknown MODE: $MODE (use: neighbors | groupavg | noiseceiling | persubject | groupstats | visualize)"; exit 1 ;;
esac
