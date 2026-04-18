#!/bin/bash
# Run Leo analysis on profiling results using Apptainer on Aurora.
#
# Usage:
#   ./run_leo_aurora.sh <dir> --arch <arch> [--top-n N] [--sif /path/to/image.sif]
#
# The <dir> can be either:
#   - A measurements directory (hpctoolkit-*-measurements*)
#   - A kernel output directory containing a measurements directory
#
# Examples:
#   ./run_leo_aurora.sh results/per-kernel/Basic_DAXPY/intel --arch intel
#   ./run_leo_aurora.sh results/per-kernel/Basic_DAXPY/intel --arch intel --top-n 2
#
# Prerequisites:
#   - .env file in the Leo repo root (contains API keys, HuggingFace token, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIF="${SIF:-$SCRIPT_DIR/leo-rajaperf-intel.sif}"
ENV_FILE="$LEO_ROOT/.env"

# ============================================
# Parse arguments
# ============================================
INPUT_DIR=""
ARCH=""
TOP_N=5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"; shift 2 ;;
        --top-n)
            TOP_N="$2"; shift 2 ;;
        --sif)
            SIF="$2"; shift 2 ;;
        --*)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            fi
            shift ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$ARCH" ]]; then
    echo "Usage: $0 <dir> --arch <arch> [--top-n N] [--sif /path/to/image.sif]"
    echo ""
    echo "Examples:"
    echo "  $0 results/per-kernel/Basic_DAXPY/intel --arch intel"
    echo "  $0 results/per-kernel/Basic_DAXPY/intel --arch intel --top-n 2"
    exit 1
fi

# Resolve to absolute path
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"

# Auto-detect: if user passed a measurements dir, use its parent as BIND_DIR
# so the sibling database dir is also visible inside the container.
if [[ "$(basename "$INPUT_DIR")" == hpctoolkit-*-measurements* ]]; then
    BIND_DIR="$(dirname "$INPUT_DIR")"
    MEAS_NAME="$(basename "$INPUT_DIR")"
else
    BIND_DIR="$INPUT_DIR"
    # Find the measurements directory name inside the given dir
    MEAS_NAME=$(ls -td "$INPUT_DIR"/hpctoolkit-*-measurements 2>/dev/null | head -1 | xargs basename 2>/dev/null)
    if [[ -z "$MEAS_NAME" ]]; then
        MEAS_NAME=$(ls -td "$INPUT_DIR"/hpctoolkit-*-measurements-* 2>/dev/null | head -1 | xargs basename 2>/dev/null)
    fi
    if [[ -z "$MEAS_NAME" ]]; then
        echo "ERROR: No hpctoolkit measurements directory found in $INPUT_DIR"
        exit 1
    fi
fi

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: SIF image not found: $SIF"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "WARNING: No .env file found at $ENV_FILE"
fi

echo "============================================"
echo " Leo Analysis on Aurora"
echo "============================================"
echo "Data dir:     $BIND_DIR"
echo "Measurements: $MEAS_NAME"
echo "Architecture: $ARCH"
echo "Top-N:        $TOP_N"
echo "SIF:          $SIF"
echo ""

mkdir -p "$HOME/tmp"

BIND_FLAGS="--bind $BIND_DIR:/data:ro"
if [[ -f "$ENV_FILE" ]]; then
    BIND_FLAGS="$BIND_FLAGS --bind $ENV_FILE:/opt/leo/.env:ro"
fi

apptainer exec --cleanenv --no-home --writable-tmpfs \
    $BIND_FLAGS \
    --bind "$HOME/tmp:/tmp/leo-tmp" \
    --env UV_CACHE_DIR=/tmp/leo-tmp/uv-cache \
    "$SIF" \
    bash -c "cd /opt/leo && uv run python scripts/analyze_benchmark.py /data/$MEAS_NAME --arch $ARCH --top-n $TOP_N ${EXTRA_ARGS[*]}"
