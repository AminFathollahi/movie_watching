#!/bin/bash

# Simple checker for PE-AV model downloads

echo "======================================"
echo "PE-AV Model Download Status"
echo "======================================"
echo ""

BASE_DIR="/home/amin/hf_models/manual"

check_model() {
    local model_name=$1
    local model_dir="${BASE_DIR}/facebook--${model_name}"
    
    echo "Checking: facebook/${model_name}"
    
    if [ ! -d "$model_dir" ]; then
        echo "  Status: ✗ NOT DOWNLOADED (directory missing)"
        echo "  Location: $model_dir"
        echo ""
        return 1
    fi
    
    # Check for key files
    local has_config=false
    local has_model=false
    
    if [ -f "$model_dir/config.json" ]; then
        has_config=true
    fi
    
    if [ -f "$model_dir/model.safetensors" ] || [ -f "$model_dir/pytorch_model.bin" ]; then
        has_model=true
    fi
    
    if $has_config && $has_model; then
        echo "  Status: ✓ COMPLETE"
        echo "  Location: $model_dir"
        du -sh "$model_dir" 2>/dev/null | awk '{print "  Size: " $1}'
    else
        echo "  Status: ⚠ INCOMPLETE"
        echo "  Location: $model_dir"
        [ ! $has_config ] && echo "  Missing: config.json"
        [ ! $has_model ] && echo "  Missing: model weights"
    fi
    
    echo ""
}

# Check all models
check_model "pe-av-small"
check_model "pe-av-base"
check_model "pe-av-base-16-frame"
check_model "pe-av-large"
check_model "pe-av-large-16-frame"

echo "======================================"
echo "Total disk usage:"
du -sh "${BASE_DIR}" 2>/dev/null
echo "======================================"
