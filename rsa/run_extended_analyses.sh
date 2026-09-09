#!/usr/bin/env bash
# Run and consolidate the group-average residualized AV RSA suite.
#
# Stages
#   embed       Generate linear- and projection-residual embeddings.
#   rsa         Run normal searchlight RSA for both residual embeddings.
#   partial     Run partial-correlation RSA against the same specialist pair.
#   consolidate Combine the three searchlight maps into one CIFTI per model.
#   all         Run all stages in dependency order.
#
# Usage
#   bash rsa/run_extended_analyses.sh [embed|rsa|partial|consolidate|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
HCP_DIR="${DATA_BASE}/HCP_S1200_GroupAvg_v1"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
LEFT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii"
RIGHT_SURFACE="${HCP_DIR}/GroupAverage_59k/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii"
WORKBENCH="/opt/workbench/bin_linux64/wb_command"
OUTPUT_DIR="${OUTPUTS_BASE}/rsa/raw"
GROUP_ROOT="${OUTPUT_DIR}/group_average"
GEODESIC_CACHE_DIR="${OUTPUTS_BASE}/rsa/_geodesic_cache"
RUN_TRS="${PREPROCESSED_DIR}/group_average_raw_run_trs.npy"
CONDA_ENV="movie"

K=100
BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
METHOD="spearman"
TR=1.0
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-512}"

CONFIG="k100_delay5s_bin5s_skip5s_spearman"
NORMAL_MAPS_FILENAME="rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_maps.dscalar.nii"
RESIDUALIZED_MAPS_FILENAME="rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_residualized_maps.dscalar.nii"

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

load_models() {
    mapfile -t RESIDUAL_MODELS < <(run_python -c "
from rsa.shared.model_registry import RESIDUALIZED_AV_MODELS
print('\\n'.join(RESIDUALIZED_AV_MODELS))
")
}

run_embed() {
    log "Generating linear-residual embeddings"
    run_python "notebooks/feature_extraction/compute_linear_residual_embeddings.py" \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --timing-csv "$TIMING_CSV" \
        --run-trs "$RUN_TRS" \
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
        --delay-sec "$DELAY_SEC" --tr "$TR"

    log "Generating projection-residual embeddings"
    run_python "notebooks/feature_extraction/compute_projection_residual_embeddings.py" \
        --embeddings-dir "$EMBEDDINGS_DIR" \
        --timing-csv "$TIMING_CSV" \
        --run-trs "$RUN_TRS" \
        --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
        --delay-sec "$DELAY_SEC" --tr "$TR" \
        --skip-encoder
}

run_rsa() {
    local models_str
    models_str=$(run_python -c "
from pathlib import Path
import nibabel as nib
from rsa.shared.model_registry import RESIDUALIZED_AV_MODELS

embeddings = Path('$EMBEDDINGS_DIR')
group_root = Path('$GROUP_ROOT')
normal_filename = '$NORMAL_MAPS_FILENAME'
residualized_filename = '$RESIDUALIZED_MAPS_FILENAME'
missing = []
for model in RESIDUALIZED_AV_MODELS:
    consolidated = group_root / f'{model}_av' / residualized_filename
    consolidated_names = []
    if consolidated.is_file():
        try:
            consolidated_names = list(nib.load(consolidated).header.get_axis(0).name)
        except Exception:
            pass
    for suffix in ('linear_resid_unimodal', 'projection_resid_own'):
        scalar_name = suffix
        if scalar_name in consolidated_names:
            continue
        residual_model = f'{model}_av_{suffix}'
        embedding = embeddings / residual_model / 'bin5s_skip5s' / f'{residual_model}_av.npy'
        if not embedding.is_file():
            continue
        output = group_root / f'{residual_model}_av' / normal_filename
        names = []
        if output.is_file():
            try:
                names = list(nib.load(output).header.get_axis(0).name)
            except Exception:
                pass
        if 'searchlight_spearman_rho' not in names:
            missing.append(f'{residual_model}:av')
print(';'.join(missing))
")
    if [ -z "$models_str" ]; then
        log "All residual-embedding searchlight maps already exist"
        return 0
    fi

    log "Running group-average searchlight RSA for residual embeddings"
    BIN_SECS="$BIN_SEC" GPU_BATCH_SIZE="$GPU_BATCH_SIZE" RSA_MODELS_OVERRIDE="$models_str" \
        bash "${SCRIPT_DIR}/analysis.sh" avg searchlight
}

run_partial() {
    local model run_key failed=0
    mapfile -t RESIDUAL_MODELS < <(run_python -c "
from pathlib import Path
import nibabel as nib
from rsa.shared.model_registry import RESIDUALIZED_AV_MODELS

group_root = Path('$GROUP_ROOT')
config = '$CONFIG'
residualized_filename = '$RESIDUALIZED_MAPS_FILENAME'
for model in RESIDUALIZED_AV_MODELS:
    consolidated = group_root / f'{model}_av' / residualized_filename
    names = []
    if consolidated.is_file():
        try:
            names = list(nib.load(consolidated).header.get_axis(0).name)
        except Exception:
            pass
    legacy = group_root / f'{model}_av_partial_corr' / config / 'partial_corr_r_searchlight.dscalar.nii'
    if 'partial_correlation' not in names and not legacy.is_file():
        print(model)
")
    if [ "${#RESIDUAL_MODELS[@]}" -eq 0 ]; then
        log "All partial-correlation maps are consolidated"
        return 0
    fi
    log "Running group-average partial-correlation RSA"
    for model in "${RESIDUAL_MODELS[@]}"; do
        run_key="partial_corr_${model}"
        log "  ${run_key}"
        run_python "${SCRIPT_DIR}/partial_rsa.py" \
            --run "$run_key" \
            --preprocessed-dir "$PREPROCESSED_DIR" \
            --fmri-suffix "$FMRI_SUFFIX" \
            --timing-csv "$TIMING_CSV" \
            --embeddings-dir "$EMBEDDINGS_DIR" \
            --template-cifti "$TEMPLATE_CIFTI" \
            --output-dir "$OUTPUT_DIR" \
            --subject group_average \
            --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
            --delay-sec "$DELAY_SEC" --tr "$TR" \
            --k "$K" --method "$METHOD" \
            --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE" \
            --workbench "$WORKBENCH" \
            --geodesic-cache-dir "$GEODESIC_CACHE_DIR" \
            --gpu-batch-size "$GPU_BATCH_SIZE" \
            || failed=1
    done
    return "$failed"
}

run_consolidate() {
    load_models
    local model failed=0
    log "Consolidating residualized RSA maps"
    for model in "${RESIDUAL_MODELS[@]}"; do
        run_python "${SCRIPT_DIR}/residualized_maps.py" \
            --model "$model" \
            --rsa-root "$GROUP_ROOT" \
            --config "$CONFIG" \
            --normal-maps-filename "$NORMAL_MAPS_FILENAME" \
            --template-cifti "$TEMPLATE_CIFTI" \
            || failed=1
    done
    return "$failed"
}

STAGE=${1:-all}
case "$STAGE" in
    embed)       run_embed ;;
    rsa|plain)   run_rsa ;;
    partial)     run_partial ;;
    consolidate) run_consolidate ;;
    all)         run_embed; run_rsa; run_partial; run_consolidate ;;
    *)
        echo "Unknown stage: $STAGE (use: embed | rsa | partial | consolidate | all)"
        exit 1
        ;;
esac

log "Residualized RSA stage complete: ${STAGE}"
