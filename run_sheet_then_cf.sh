#!/usr/bin/env bash
# run_sheet_then_cf.sh
# =====================
# Detached chain: stop the running per-subject CF fit, extract Topo-Omni's
# FULL cortical sheet (intact AV pass), run k=100 sheet searchlight RSA
# against the two CCA seed ROIs, then relaunch CF exactly as it was running.
#
# Launch (survives session death):
#   mkdir -p logs && nohup setsid bash run_sheet_then_cf.sh \
#       > logs/chain_bootstrap.log 2>&1 &
#
# Not `set -e`: step 2 (extraction) and step 3 (RSA) failures are handled
# explicitly so the GPU is never left idle -- a failure always falls through
# to step 4 (relaunch CF).
set -u -o pipefail
cd "$(dirname "$0")"

mkdir -p logs
LOGFILE="logs/sheet_then_cf_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== sheet_then_cf chain starting -- log=$LOGFILE ==="

# ── Step 1: kill the running CF job, wait for the GPU to clear ─────────────
log "Step 1: killing CF job (persubject_cca_peav_1pct)"
pkill -f "cf_modeling/run_analysis.sh persubject_cca_peav_1pct" || true
pkill -f "02_fit_cf_model.py --mode per_subject" || true

WAITED=0
while [ "$WAITED" -lt 300 ]; do
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \r')
    if [ -z "$PIDS" ]; then
        log "Step 1: GPU clear after ${WAITED}s"
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
done
if [ "$WAITED" -ge 300 ]; then
    log "Step 1: WARNING -- GPU still occupied after 300s wait (pids: $PIDS), proceeding anyway"
fi

# ── Step 2: full-sheet extraction ───────────────────────────────────────────
log "Step 2: full-sheet extraction (topo_omni env, BIN_SEC=5.0 SKIP_SEC=5.0)"
STEP2_OK=0
if BIN_SEC=5.0 SKIP_SEC=5.0 conda run --no-capture-output -n topo_omni \
    python "notebooks/feature_extraction/topo_omni_extract_full_sheet.py"; then
    STEP2_OK=1
    log "Step 2: OK"
else
    log "Step 2: FAILED -- skipping RSA, falling through to CF relaunch"
fi

# ── Step 3: full-sheet searchlight RSA ──────────────────────────────────────
if [ "$STEP2_OK" -eq 1 ]; then
    log "Step 3: full-sheet searchlight RSA (movie env)"
    if bash "rsa/run_full_sheet_rsa.sh"; then
        log "Step 3: OK"
    else
        log "Step 3: FAILED"
    fi
else
    log "Step 3: SKIPPED (step 2 failed)"
fi

# ── Step 4: relaunch CF exactly as it was running ───────────────────────────
log "Step 4: relaunching CF (persubject_cca_peav_1pct)"
bash cf_modeling/run_analysis.sh persubject_cca_peav_1pct 1
log "=== chain finished (CF run_analysis.sh exited) ==="
