#!/usr/bin/env bash
# cluster/cluster.sh
# ====================
# Master runner for the dual-clustering pipeline (stimulus temporal states
# x brain spatial networks). Mirrors subcortical/analysis.sh conventions.
#
# Usage
#   bash cluster/cluster.sh MODE
#
#   MODE   groupaverage   PRIMARY FIRST PASS. Loop MODELS x modalities, run
#                          run_cluster.py --mode groupaverage for each.
#          persubject     Same loop but --mode persubject (stub this pass).
#          channel_stability  Evaluate AV channel clusters across movie runs.
#          heldout_roi_alignment  Select channel-to-ROI matches on training
#                          runs and evaluate them on the unseen run.
#          profile_correlations  Screen every candidate clustering for temporal
#                          differentiation and save full cluster x cluster mean-
#                          profile correlation matrices for the best-differentiated
#                          solutions.
#          modality_preference  Test each channel cluster's audio/video preference
#                          (AV vs A/V and clsav-ablated AV) against a channel-label
#                          permutation null.
#          vertex_roi_hotspot_enrichment  Auditory/visual/audiovisual ROI and
#                          AV-RSA-hotspot enrichment per vertex cluster (screened
#                          best-overall and best-full-coverage solutions).
#
# Every default below is overridable via env var and reachable by editing
# this file only.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIG
# =============================================================================
DATA_BASE="/home/amin/Research/Representation/Movie/data"
OUTPUTS_BASE="/home/amin/Research/Representation/Movie/outputs"
HCP_DIR="${MOVIE_HCP_DIR:-/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1}"

GROUP_AVG_CIFTI="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
GROUP_AVG_TRS="${DATA_BASE}/preprocessed/average_sub/raw/group_average_raw_run_trs.npy"
TEMPLATE_CIFTI="$GROUP_AVG_CIFTI"
TIMING_CSV="${DATA_BASE}/movie_timing.csv"
EMBEDDINGS_DIR="${OUTPUTS_BASE}/model_embeddings"
CONDA_ENV="movie"

OUTDIR="${OUTPUTS_BASE}/cluster"
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

# task shorthand "nemotron:a,v,av" -> on-disk name nemotron_layer27_mp
# (verified present; a/v/av all (626,2048) at bin5s_skip5s). One-line edit
# here to swap the layer or add more models.
#
# _avscramble / _clsav_from_a / _clsav_from_v (modality=av) are AV-integration
# CONTROL conditions, not new stimuli: avscramble pairs real audio with a
# randomly-permuted video (breaks temporal correspondence, tests binding
# sensitivity); clsav_from_a/clsav_from_v blank out one real modality (tests
# whether a network needs BOTH modalities simultaneously present). Reuses
# run_cluster.py unchanged -- these are just other {model}/{modality} on-disk
# embedding names. Consumed by cluster/av_integration.py.
MODELS=(
    "nemotron_layer27_mp:a,v,av"
    "nemotron_layer27_mp_avscramble:av"
    "nemotron_layer27_mp_clsav_from_a:av"
    "nemotron_layer27_mp_clsav_from_v:av"
    "pe-av-small-16-frame:a,v,av"
    "pe-av-small-16-frame_avscramble:av"
    "pe-av-small-16-frame_clsav_from_a:av"
    "pe-av-small-16-frame_clsav_from_v:av"
)

BIN_SEC=5  SKIP_SEC=5  DELAY_SEC=5  TR=1.0
TEMPORAL_REDUCTION=pca  SPATIAL_REDUCTION=pca  SPATIAL_CLUSTER=hdbscan
TEMPORAL_NCOMP=50  SPATIAL_NCOMP=100  ADV_NCOMP=10
N_STATES=10  HMM_COVARIANCE=diag  HMM_N_INIT=10
MIN_CLUSTER_SIZE=100  MIN_SAMPLES=10
N_PERM=1000  PERM_CORRECTION=fdr
CHANNEL_STABILITY_MODELS=("pe-av-small-16-frame" "nemotron_layer18_mp")
CHANNEL_STABILITY_CLUSTERS=(2 4 8 16 32)
CHANNEL_STABILITY_RANDOM_SEEDS=(0 1 2 3 4)
CHANNEL_STABILITY_N_NULL=1000
HELDOUT_ALIGNMENT_MODELS=("pe-av-small-16-frame:32" "nemotron_layer18_mp:32")
HELDOUT_ALIGNMENT_RANDOM_SEEDS=(0 1 2 3 4)
HELDOUT_ALIGNMENT_N_SHIFTS=5000
HELDOUT_ALIGNMENT_N_RANDOM_PARTITIONS=2000
GLASSER_DLABEL="${HCP_DIR}/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"
PROFILE_CORRELATIONS_TOP_N=10
MODALITY_PREFERENCE_FAMILIES=("peav" "nemotron_layer18_mp")
MODALITY_PREFERENCE_TOP_N=10
MODALITY_PREFERENCE_N_PERMUTATIONS=10000

MODE=${1:-groupaverage}
run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_groupaverage() {
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Group-average clustering: ${MODEL_NAME}/${MOD}"
            run_python "${SCRIPT_DIR}/run_cluster.py" \
                --group-avg-cifti "$GROUP_AVG_CIFTI" \
                --group-avg-trs "$GROUP_AVG_TRS" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --output-dir "$OUTDIR" \
                --mode groupaverage \
                --model "$MODEL_NAME" --modality "$MOD" \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR" \
                --temporal-reduction "$TEMPORAL_REDUCTION" --temporal-n-components "$TEMPORAL_NCOMP" \
                --adv-n-components "$ADV_NCOMP" \
                --spatial-reduction "$SPATIAL_REDUCTION" --spatial-n-components "$SPATIAL_NCOMP" \
                --spatial-cluster "$SPATIAL_CLUSTER" \
                --min-cluster-size "$MIN_CLUSTER_SIZE" --min-samples "$MIN_SAMPLES" \
                --n-temporal-states "$N_STATES" --hmm-covariance "$HMM_COVARIANCE" --hmm-n-init "$HMM_N_INIT" \
                --n-permutations "$N_PERM" --perm-correction "$PERM_CORRECTION"
        done
    done
    log "Group-average clustering complete."
}

run_persubject() {
    for MODEL_ENTRY in "${MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME MODALITIES_ENTRY <<< "$MODEL_ENTRY"
        IFS=',' read -ra MODS <<< "$MODALITIES_ENTRY"
        for MOD in "${MODS[@]}"; do
            log "Per-subject clustering (stub): ${MODEL_NAME}/${MOD}"
            run_python "${SCRIPT_DIR}/run_cluster.py" \
                --group-avg-cifti "$GROUP_AVG_CIFTI" \
                --group-avg-trs "$GROUP_AVG_TRS" \
                --template-cifti "$TEMPLATE_CIFTI" \
                --timing-csv "$TIMING_CSV" \
                --embeddings-dir "$EMBEDDINGS_DIR" \
                --output-dir "$OUTDIR" \
                --mode persubject --subjects-list "$SUBJECTS_LIST" \
                --model "$MODEL_NAME" --modality "$MOD" \
                --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" --delay-sec "$DELAY_SEC" --tr "$TR"
        done
    done
    log "Per-subject clustering complete."
}

run_channel_stability() {
    for MODEL_NAME in "${CHANNEL_STABILITY_MODELS[@]}"; do
        log "Channel stability: ${MODEL_NAME}/av"
        run_python "${SCRIPT_DIR}/channel_stability.py" \
            --embedding-path "${EMBEDDINGS_DIR}/${MODEL_NAME}/bin${BIN_SEC}s_skip${SKIP_SEC}s/${MODEL_NAME}_av.npy" \
            --timing-csv "$TIMING_CSV" \
            --output-dir "${OUTDIR}/channel_stability/${MODEL_NAME}/bin${BIN_SEC}s_skip${SKIP_SEC}s" \
            --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
            --clusters "${CHANNEL_STABILITY_CLUSTERS[@]}" \
            --random-seeds "${CHANNEL_STABILITY_RANDOM_SEEDS[@]}" \
            --n-null "$CHANNEL_STABILITY_N_NULL"
    done
}

run_heldout_roi_alignment() {
    local MODEL_ENTRY MODEL_NAME N_CLUSTERS
    for MODEL_ENTRY in "${HELDOUT_ALIGNMENT_MODELS[@]}"; do
        IFS=':' read -r MODEL_NAME N_CLUSTERS <<< "$MODEL_ENTRY"
        log "Held-out ROI alignment: ${MODEL_NAME}/av k=${N_CLUSTERS}"
        run_python "${SCRIPT_DIR}/heldout_roi_alignment.py" \
            --embedding-path "${EMBEDDINGS_DIR}/${MODEL_NAME}/bin${BIN_SEC}s_skip${SKIP_SEC}s/${MODEL_NAME}_av.npy" \
            --timing-csv "$TIMING_CSV" \
            --fmri-path "$GROUP_AVG_CIFTI" \
            --run-trs "$GROUP_AVG_TRS" \
            --glasser-dlabel "$GLASSER_DLABEL" \
            --roi "auditory=A1,MBelt,LBelt,PBelt,RI,A4,A5" \
            --roi "posterior_temporal=STGa,STSda,STSdp,STSva,STSvp,STV,TA2,TPOJ1,TPOJ2,TPOJ3" \
            --output-dir "${OUTDIR}/heldout_roi_alignment/${MODEL_NAME}/k${N_CLUSTERS}_bin${BIN_SEC}s_skip${SKIP_SEC}s" \
            --n-clusters "$N_CLUSTERS" \
            --random-seeds "${HELDOUT_ALIGNMENT_RANDOM_SEEDS[@]}" \
            --n-shifts "$HELDOUT_ALIGNMENT_N_SHIFTS" \
            --n-random-partitions "$HELDOUT_ALIGNMENT_N_RANDOM_PARTITIONS" \
            --bin-sec "$BIN_SEC" --skip-sec "$SKIP_SEC" \
            --delay-sec "$DELAY_SEC" --tr "$TR"
    done
}

run_profile_correlations() {
    log "Cluster mean-timeseries screen + profile-correlation matrices (top ${PROFILE_CORRELATIONS_TOP_N})"
    run_python "${SCRIPT_DIR}/screen_temporal_differentiation.py" --top-n "$PROFILE_CORRELATIONS_TOP_N"
}

run_modality_preference() {
    local FAMILY_ARGS=()
    for FAMILY in "${MODALITY_PREFERENCE_FAMILIES[@]}"; do
        FAMILY_ARGS+=(--family "$FAMILY")
    done
    log "Channel modality preference: ${MODALITY_PREFERENCE_FAMILIES[*]}"
    run_python "${SCRIPT_DIR}/channel_modality_preference.py" \
        "${FAMILY_ARGS[@]}" \
        --top-n "$MODALITY_PREFERENCE_TOP_N" \
        --n-permutations "$MODALITY_PREFERENCE_N_PERMUTATIONS"
}

run_vertex_roi_hotspot_enrichment() {
    log "Vertex cluster ROI/AV-hotspot enrichment"
    run_python "${SCRIPT_DIR}/vertex_roi_hotspot_enrichment.py" --glasser-dlabel "$GLASSER_DLABEL"
}

case "$MODE" in
    groupaverage) run_groupaverage ;;
    persubject)   run_persubject ;;
    channel_stability) run_channel_stability ;;
    heldout_roi_alignment) run_heldout_roi_alignment ;;
    profile_correlations) run_profile_correlations ;;
    modality_preference) run_modality_preference ;;
    vertex_roi_hotspot_enrichment) run_vertex_roi_hotspot_enrichment ;;
    *) echo "Unknown MODE: $MODE (use: groupaverage | persubject | channel_stability | heldout_roi_alignment | profile_correlations | modality_preference | vertex_roi_hotspot_enrichment)"; exit 1 ;;
esac
