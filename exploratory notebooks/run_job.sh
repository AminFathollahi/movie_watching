#!/bin/bash

# 1. Define the name of your python script
SCRIPT_NAME="ridge_encoding.py"

echo "=========================================="
echo "Starting Analysis Job: $SCRIPT_NAME"
echo "Start Time: $(date)"
echo "=========================================="

# 2. Run the Python Script
# We use the full path to your python if you use a specific environment (e.g., conda)
# Otherwise 'python' or 'python3' is fine.
python "$SCRIPT_NAME"

# Capture the exit status of the python script
EXIT_STATUS=$?

echo "=========================================="
echo "Job Finished at: $(date)"
echo "Exit Status: $EXIT_STATUS"
echo "=========================================="

# 3. Shutdown Logic
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Script finished successfully. Shutting down in 1 minute..."
    echo "Press Ctrl+C to cancel shutdown."
    
    # Sleep for 60 seconds to give you a chance to cancel if you are watching
    sleep 60
    
    # Shutdown command (requires sudo privileges or root user)
    sudo shutdown -h now
else
    echo "❌ Script CRASHED or failed. Computer will NOT shut down so you can see the error."
fi
