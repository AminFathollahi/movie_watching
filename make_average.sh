#!/bin/bash

# Define directories (Updated to match your actual capitalizations)
MID_DIR="/home/amin/Research/Representation/Movie/data/midthickness_1.6"
INF_DIR="/home/amin/Research/Representation/Movie/data/Inflated_1.6"
OUT_DIR="/home/amin/Research/Representation/Movie/data/GroupAverage_59k"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Function to compute average for a specific hemisphere and surface type
average_surfaces() {
    local hemi=$1
    local surf_type=$2
    local data_dir=$3
    local out_file="$OUT_DIR/CohortAvg.${hemi}.${surf_type}_MSMAll.59k_fs_LR.surf.gii"

    echo "Building average for ${hemi} ${surf_type}..."

    # Fixed wildcard: Removed the extra dot between hemisphere and surface type
    local surf_args=""
    for file in "$data_dir"/*."$hemi"."$surf_type"*.surf.gii; do
        if [ -f "$file" ]; then
            surf_args="$surf_args -surf $file"
        fi
    done

    # Check if we found files
    if [ -z "$surf_args" ]; then
        echo "No files found for $hemi $surf_type in $data_dir"
        return
    fi

    # Execute the surface average command
    wb_command -surface-average "$out_file" $surf_args
    
    echo "Saved to $out_file"
}

# Run the averaging for Left and Right hemispheres for Midthickness
# (Make sure your midthickness folder is actually lowercase 'm')
average_surfaces "L" "midthickness" "$MID_DIR"
average_surfaces "R" "midthickness" "$MID_DIR"

# Run the averaging for Left and Right hemispheres for Inflated
average_surfaces "L" "inflated" "$INF_DIR"
average_surfaces "R" "inflated" "$INF_DIR"

echo "All group averages generated successfully."
