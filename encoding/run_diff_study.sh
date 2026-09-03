#!/usr/bin/env bash
# encoding/run_diff_study.sh
# ============================
# Encoding-currency analogue of rsa/run_diff_study.sh. Same reasoning: scramble
# (avscramble) and dummy (clsav_from_a/_v) are CONTROL CONDITIONS paired
# against a native-AV baseline, not independent models, so they get their own
# model list and runner instead of encoding/analysis.sh's general MODELS sweep.
#
# Hypothesis under test (AV-integration claim, encoding-currency version): in
# true integration regions, BOTH plain predictive alignment (encoding_r2_audiovisual)
# AND residual AV-band variance (variance_partition_r2_av_residual_band)
# should be high for the native/intact condition and WEAKEN under scramble
# and dummy.
#
# Stages
#   plain     Group-average plain encoding (encoding.py, modality=av) for
#             every {base_model}_{condition} in BASE_MODELS x CONDITIONS.
#             Reuses encoding/analysis.sh via ENCODING_MODELS_OVERRIDE.
#   avresid   Group-average banded-ridge unique-AV-variance
#             (encoding/variance_partition.py) for every condition. Scramble
#             uses the condition's OWN a/v (--nuisance-model defaults to
#             --model); dummy conditions point --nuisance-model at the
#             INTACT base model with a single real --nuisance-modalities
#             (the placeholder modality can't be a nuisance band).
#   consolidate  encoding/diff_maps.py (binding + modality_presence_diff).
#   all       plain + avresid + consolidate.
#
# Usage
#   bash encoding/run_diff_study.sh [STAGE]
#
# Env overrides (this is a LARGE compute job -- each avresid run is a
# cross-validated banded-ridge fit, not a cheap correlation):
#   DIFF_STUDY_MODELS       space-separated subset of BASE_MODELS.
#   DIFF_STUDY_CONDITIONS   space-separated subset of {avscramble,clsav_from_a,clsav_from_v}.
#
# Example: sanity-check one model before committing to the full sweep:
#   DIFF_STUDY_MODELS="pe-av-small-16-frame" bash encoding/run_diff_study.sh plain
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"

PREPROCESSED_DIR="${DATA_BASE}/preprocessed/average_sub/raw"
FMRI_SUFFIX="raw"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
TEMPLATE_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
OUTPUT_DIR="${OUTPUTS_BASE}/encoding"
CONDA_ENV="movie"

BIN_SEC=5.0
SKIP_SEC=5.0
DELAY_SEC=5.0
TR=1.0
ALPHA_MIN=-2
ALPHA_MAX=9
N_ALPHAS=23
N_ITER=20
BACKEND="torch_cuda"
TEST_VIDEO_IDS="video5,video9,video14,video18"

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Same native-AV roster as rsa/run_diff_study.sh and encoding/diff_maps.py.
BASE_MODELS=(
    pe-av-small-16-frame
    nemotron_layer9_mp nemotron_layer18_mp nemotron_layer27_mp nemotron_layer36_mp
    omni3b_layer9_mp omni3b_layer18_mp omni3b_layer27_mp
    topoomni_layer9_mp topoomni_layer18_mp topoomni_layer27_mp
    topoomni_layer9_sheet_mp topoomni_layer18_sheet_mp topoomni_layer27_sheet_mp
)
CONDITIONS=(avscramble clsav_from_a clsav_from_v)

if [ -n "${DIFF_STUDY_MODELS:-}" ]; then
    read -ra BASE_MODELS <<< "$DIFF_STUDY_MODELS"
fi
if [ -n "${DIFF_STUDY_CONDITIONS:-}" ]; then
    read -ra CONDITIONS <<< "$DIFF_STUDY_CONDITIONS"
fi

STAGE=${1:-plain}

run_diff_study_plain() {
    # Includes each BASE's own intact run (needed by diff_maps.py's
    # diff_intact_minus_scrambled/dummy_plain_rsa) alongside every
    # scramble/dummy condition -- mirrors run_diff_study_avresid's
    # "also ensure the intact base model's own AVresid exists" check.
    local MODELS_STR="" BASE COND
    for BASE in "${BASE_MODELS[@]}"; do
        MODELS_STR="${MODELS_STR}${BASE}:av;"
        for COND in "${CONDITIONS[@]}"; do
            MODELS_STR="${MODELS_STR}${BASE}_${COND}:av;"
        done
    done
    MODELS_STR="${MODELS_STR%;}"
    log "=== Diff-study PLAIN encoding: ${#BASE_MODELS[@]} models x (intact + ${#CONDITIONS[@]} conditions) ==="
    BIN_SECS="$BIN_SEC" ENCODING_MODELS_OVERRIDE="$MODELS_STR" \
        bash "${SCRIPT_DIR}/analysis.sh" avg
    log "=== Diff-study plain encoding complete ==="
}

run_diff_study_avresid() {
    local BASE COND MODEL NUISANCE_MODEL NUISANCE_MODS
    log "=== Diff-study AVresid (group-average) ==="
    for BASE in "${BASE_MODELS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            MODEL="${BASE}_${COND}"
            case "$COND" in
                avscramble)
                    NUISANCE_MODEL="$MODEL"; NUISANCE_MODS="a,v" ;;
                clsav_from_a)
                    NUISANCE_MODEL="$BASE"; NUISANCE_MODS="a" ;;
                clsav_from_v)
                    NUISANCE_MODEL="$BASE"; NUISANCE_MODS="v" ;;
            esac
            log "  ${MODEL}  (nuisance=${NUISANCE_MODEL}/${NUISANCE_MODS})"
            run_python "${SCRIPT_DIR}/variance_partition.py" \
                --preprocessed-dir "$PREPROCESSED_DIR" \
                --fmri-suffix "$FMRI_SUFFIX" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --output-dir "$OUTPUT_DIR" \
                --subject group_average \
                --model "$MODEL" \
                --nuisance-model "$NUISANCE_MODEL" --nuisance-modalities "$NUISANCE_MODS" \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                --normalize \
                --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX" --n-alphas "$N_ALPHAS" --n-iter "$N_ITER" \
                --backend "$BACKEND" \
                --test-video-ids "$TEST_VIDEO_IDS" \
                || log "  SKIP ${MODEL}: variance_partition.py failed (embeddings missing?)"
        done
        # Also ensure the intact base model's own AVresid exists (needed by
        # diff_maps.py's binding/modality_presence_diff); cheap to re-check,
        # variance_partition.py has no internal skip-if-exists so only run if
        # the output is actually missing.
        INTACT_OUT="${OUTPUT_DIR}/group_average/${BASE}/delay5s_norm_bin5s_skip5s/variance_partition_r2_av_residual_band.dscalar.nii"
        if [ ! -f "$INTACT_OUT" ]; then
            log "  ${BASE} (intact)"
            run_python "${SCRIPT_DIR}/variance_partition.py" \
                --preprocessed-dir "$PREPROCESSED_DIR" \
                --fmri-suffix "$FMRI_SUFFIX" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --output-dir "$OUTPUT_DIR" \
                --subject group_average \
                --model "$BASE" \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                --normalize \
                --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX" --n-alphas "$N_ALPHAS" --n-iter "$N_ITER" \
                --backend "$BACKEND" \
                --test-video-ids "$TEST_VIDEO_IDS" \
                || log "  SKIP ${BASE} (intact): variance_partition.py failed"
        fi
    done
    log "=== Diff-study AVresid complete ==="
}

case "$STAGE" in
    plain)       run_diff_study_plain ;;
    avresid)     run_diff_study_avresid ;;
    consolidate) run_python "${SCRIPT_DIR}/diff_maps.py" ;;
    all)         run_diff_study_plain; run_diff_study_avresid; run_python "${SCRIPT_DIR}/diff_maps.py" ;;
    *) echo "Unknown STAGE: $STAGE (use: plain | avresid | consolidate | all)"; exit 1 ;;
esac
log "Diff study (${STAGE}) complete."
