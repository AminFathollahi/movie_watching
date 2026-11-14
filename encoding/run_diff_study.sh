#!/usr/bin/env bash
# Dummy-modality contrasts: variance-partition fits of every base model and of its dummy-modality conditions (clsav_from_a, clsav_from_v),
# with the intact model's A and V, then the contrast maps of diff_maps.py. A runner like the others (skip rule, --status, --dry-run).
# Usage: bash encoding/run_diff_study.sh [partition|consolidate|all] [options]
# Environment: DIFF_STUDY_MODELS (base models, default the roster of diff_maps.py), DIFF_STUDY_CONDITIONS.
source "$(dirname "${BASH_SOURCE[0]}")/runner_lib.sh"
STAGE=partition
case "${1:-}" in partition|consolidate|all) STAGE=$1; shift ;; esac
SPACES=cortex BINS=5 SCALINGS=demean MODELS=${DIFF_STUDY_MODELS:-} CONDITIONS=${DIFF_STUDY_CONDITIONS:-}

code=0
if [ "$STAGE" != consolidate ]; then runner diff_study diff_study "$@"; code=$?; fi
case " $* " in *" --status "*|*" --dry-run "*) exit $code ;; esac
if [ "$STAGE" != partition ]; then resolve_python; "$PYTHON" "$RUNNER_DIR/diff_maps.py" || code=1; fi
exit $code
