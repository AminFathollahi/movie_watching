#!/bin/bash
# make_average.sh
# ================
# Build group-average midthickness and inflated surfaces from the 175 subjects
# listed in data/subjects.txt.
#
# Uses ONLY subjects in subjects.txt — never globs the full directory — so stray
# surface files (e.g., 995174 which is present in midthickness_1.6/ but absent
# from subjects.txt) are excluded automatically.
#
# Usage
# -----
#   bash make_average.sh
#
# Output
# ------
#   GroupAverage_59k/CohortAvg.{L,R}.midthickness_MSMAll.59k_fs_LR.surf.gii
#   GroupAverage_59k/CohortAvg.{L,R}.inflated_MSMAll.59k_fs_LR.surf.gii
#
# After running, build the geodesic k-NN cache for the new group-average surface:
#   bash rsa/analysis.sh neighbors_avg

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MOVIE_ROOT:-$(dirname "$REPO_DIR")}"
DATA_BASE="$ROOT/data"

# Authoritative subject list (n=175; excludes subjects without midthickness)
SUBJECTS_LIST="${DATA_BASE}/subjects.txt"

MID_DIR="${DATA_BASE}/midthickness_1.6"
INF_DIR="${DATA_BASE}/Inflated_1.6"
OUT_DIR="${DATA_BASE}/GroupAverage_59k"

WORKBENCH="${WORKBENCH:-/opt/workbench/bin_linux64/wb_command}"

# =============================================================================
# HELPERS
# =============================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Load subject IDs from subjects.txt (skip blank lines and # comments)
load_subjects() {
    grep -v '^\s*#' "$SUBJECTS_LIST" | sed 's/#.*//' | awk '{print $1}' | grep -v '^$'
}

# =============================================================================
# average_surfaces HEMI SURF_TYPE DATA_DIR
#   HEMI      : L or R
#   SURF_TYPE : midthickness | inflated
#   DATA_DIR  : directory containing per-subject .surf.gii files
# =============================================================================
average_surfaces() {
    local hemi="$1"
    local surf_type="$2"
    local data_dir="$3"
    local out_file="${OUT_DIR}/CohortAvg.${hemi}.${surf_type}_MSMAll.59k_fs_LR.surf.gii"

    log "Building ${hemi} ${surf_type} average from subjects in ${SUBJECTS_LIST} ..."

    local surf_args=""
    local n_found=0
    local missing=()

    while IFS= read -r sub; do
        # Pattern: {sub}.{L|R}.{surf_type}_1.6mm_MSMAll.59k_fs_LR.surf.gii
        local f="${data_dir}/${sub}.${hemi}.${surf_type}_1.6mm_MSMAll.59k_fs_LR.surf.gii"
        if [ -f "$f" ]; then
            surf_args="${surf_args} -surf ${f}"
            n_found=$(( n_found + 1 ))
        else
            missing+=("$sub")
        fi
    done < <(load_subjects)

    if [ "$n_found" -eq 0 ]; then
        log "ERROR: no ${hemi} ${surf_type} files found — check MID_DIR / INF_DIR paths"
        return 1
    fi

    if [ "${#missing[@]}" -gt 0 ]; then
        log "WARNING: ${#missing[@]} subject(s) missing ${hemi} ${surf_type}: ${missing[*]}"
    fi

    log "  Averaging ${n_found} surfaces → ${out_file}"
    # shellcheck disable=SC2086
    "$WORKBENCH" -surface-average "$out_file" $surf_args
    log "  Saved: ${out_file}"
}

# =============================================================================
# MAIN
# =============================================================================
mkdir -p "$OUT_DIR"

N_SUBS=$(load_subjects | wc -l)
log "=== Group-average surfaces — ${N_SUBS} subjects from ${SUBJECTS_LIST} ==="

average_surfaces "L" "midthickness" "$MID_DIR"
average_surfaces "R" "midthickness" "$MID_DIR"

average_surfaces "L" "inflated" "$INF_DIR"
average_surfaces "R" "inflated" "$INF_DIR"

log "=== All group averages generated → ${OUT_DIR} ==="
log "Next step: bash rsa/analysis.sh neighbors_avg"
