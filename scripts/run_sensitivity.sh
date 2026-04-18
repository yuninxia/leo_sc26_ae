#!/bin/bash
# Run latency sensitivity analysis inside the universal Docker container.
#
# Usage:
#   ./scripts/run_sensitivity.sh <results_subdir> <arch> [extra_args...]
#   ./scripts/run_sensitivity.sh batch                  # run all 9 (3 vendors × 3 benchmarks)
#
# Examples:
#   ./scripts/run_sensitivity.sh amd-minibude-20260220-151739 mi300
#   ./scripts/run_sensitivity.sh intel-xsbench-20260218-134304 pvc --scales 0.25 0.5 1.0 2.0 4.0
#   ./scripts/run_sensitivity.sh batch

set -euo pipefail

# Paths
LEO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$LEO_DIR/results}"

run_one() {
    local RESULTS_SUBDIR="$1"
    local ARCH="$2"
    shift 2
    local EXTRA_ARGS="$@"

    # Find measurements directory
    local MEAS_DIR
    MEAS_DIR=$(ls -d "${RESULTS_DIR}/${RESULTS_SUBDIR}"/hpctoolkit-*-measurements 2>/dev/null | head -1)
    if [ -z "$MEAS_DIR" ]; then
        echo "Error: No measurements directory found in ${RESULTS_DIR}/${RESULTS_SUBDIR}/"
        return 1
    fi
    local MEAS_NAME
    MEAS_NAME=$(basename "$MEAS_DIR")

    echo ""
    echo "=== Latency Sensitivity Analysis ==="
    echo "Results:  ${RESULTS_SUBDIR}"
    echo "Arch:     ${ARCH}"
    echo "Meas:     ${MEAS_NAME}"
    echo ""

    docker run --rm \
        --entrypoint bash \
        -v "${LEO_DIR}/src:/opt/leo/src:ro" \
        -v "${LEO_DIR}/scripts:/opt/leo/scripts:ro" \
        -v "${RESULTS_DIR}/${RESULTS_SUBDIR}:/data:ro" \
        leo-base-universal:latest \
        -c "cd /opt/leo && .venv/bin/python3 scripts/latency_sensitivity.py /data/${MEAS_NAME} --arch ${ARCH} ${EXTRA_ARGS}"
}

# --- Batch mode: 3 vendors × 3 benchmarks ---
if [ "${1:-}" = "batch" ]; then
    shift
    EXTRA_ARGS="${@}"

    echo "=========================================="
    echo " Batch: 3 vendors × 3 benchmarks"
    echo "=========================================="

    # AMD MI300
    run_one amd-minibude-20260220-151739                mi300  $EXTRA_ARGS
    run_one amd-lulesh-20260220-190435                  mi300  $EXTRA_ARGS
    run_one amd-xsbench-20260220-175339                 mi300  $EXTRA_ARGS

    # Intel PVC
    run_one intel-minibude-20260222-163954               pvc   $EXTRA_ARGS
    run_one intel-lulesh-20260218-153334                 pvc   $EXTRA_ARGS
    run_one intel-xsbench-20260218-134304                pvc   $EXTRA_ARGS

    # NVIDIA H100
    run_one nvidia-gilgamesh-minibude-20260222-174228    h100  $EXTRA_ARGS
    run_one nvidia-gilgamesh-lulesh-20260222-190913      h100  $EXTRA_ARGS
    run_one nvidia-gilgamesh-xsbench-20260222-182940     h100  $EXTRA_ARGS

    echo ""
    echo "=========================================="
    echo " Batch complete: 9 runs"
    echo "=========================================="
    exit 0
fi

# --- Single mode ---
if [ $# -lt 2 ]; then
    echo "Usage: $0 <results_subdir> <arch> [extra_args...]"
    echo "       $0 batch [extra_args...]"
    exit 1
fi

run_one "$@"
