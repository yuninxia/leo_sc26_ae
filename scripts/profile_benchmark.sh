#!/bin/bash
# Profile a GPU application with HPCToolkit for Leo analysis
#
# Usage:
#   ./scripts/profile_benchmark.sh <name> <executable> [args...]
#
# Environment:
#   GPU_VENDOR=nvidia|amd  (auto-detected if not set)
#
# Example:
#   ./scripts/profile_benchmark.sh gaussian ./gaussian -s 2048
#   GPU_VENDOR=amd ./scripts/profile_benchmark.sh vecadd ./vecadd
#
# This creates:
#   - hpctoolkit-<name>-measurements/
#   - hpctoolkit-<name>-database/

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <name> <executable> [args...]"
    echo "Example: $0 gaussian ./gaussian -s 2048"
    echo "Set GPU_VENDOR=amd for AMD GPUs"
    exit 1
fi

NAME=$1
EXECUTABLE=$2
shift 2
ARGS="$@"

MEASUREMENTS="hpctoolkit-${NAME}-measurements"
DATABASE="hpctoolkit-${NAME}-database"

# Auto-detect GPU vendor if not set
if [ -z "$GPU_VENDOR" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_VENDOR="nvidia"
    elif command -v rocm-smi &> /dev/null; then
        GPU_VENDOR="amd"
    else
        echo "Warning: Could not detect GPU vendor, defaulting to nvidia"
        GPU_VENDOR="nvidia"
    fi
fi

# Set GPU event based on vendor
if [ "$GPU_VENDOR" = "amd" ]; then
    GPU_EVENT="gpu=rocm,pc"
else
    GPU_EVENT="gpu=cuda,pc"
fi

echo "=== Profiling $EXECUTABLE $ARGS ==="
echo "GPU vendor: $GPU_VENDOR"
echo "Output: $MEASUREMENTS, $DATABASE"
echo

# Check if hpcrun is available
if ! command -v hpcrun &> /dev/null; then
    echo "Error: hpcrun not found. Load HPCToolkit first:"
    echo "  module load hpctoolkit/2026.0.0"
    exit 1
fi

# Clean up old data
rm -rf "$MEASUREMENTS" "$DATABASE"

# Step 1: Profile with hpcrun
echo "Step 1/3: Running hpcrun..."
hpcrun -o "$MEASUREMENTS" -e "$GPU_EVENT" "$EXECUTABLE" $ARGS

# Step 2: Generate structure info
echo "Step 2/3: Running hpcstruct..."
hpcstruct --gpucfg yes "$MEASUREMENTS"

# Step 3: Create database
echo "Step 3/3: Running hpcprof..."
hpcprof -o "$DATABASE" "$MEASUREMENTS"

echo
echo "=== Profiling complete ==="
echo "Run Leo analysis with:"
echo "  uv run python scripts/analyze_benchmark.py $MEASUREMENTS"
