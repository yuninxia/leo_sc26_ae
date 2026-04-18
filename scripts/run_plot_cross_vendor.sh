#!/bin/bash
# Generate cross-vendor dependency chain comparison figure using Leo + matplotlib.
# Runs inside the universal Docker container.
#
# Usage:
#   ./scripts/run_plot_cross_vendor.sh [KERNEL] [OUTPUT]
#   ./scripts/run_plot_cross_vendor.sh --all [OUTPUT]
#
# Examples:
#   ./scripts/run_plot_cross_vendor.sh                                  # defaults to CONVECTION3DPA
#   ./scripts/run_plot_cross_vendor.sh Apps_CONVECTION3DPA output.pdf
#   ./scripts/run_plot_cross_vendor.sh --all /tmp/all_kernels.pdf       # all kernels, one page each

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TOP_CHAINS="${TOP_CHAINS:-2}"

if [ "${1:-}" = "--all" ]; then
  OUTPUT="${2:-/tmp/cross_vendor_all_kernels.pdf}"
  OUTPUT_DIR="$(dirname "$OUTPUT")"
  OUTPUT_FILE="$(basename "$OUTPUT")"
  KERNEL_ARG="--all"
  echo "Mode:    all kernels (one page each)"
else
  KERNEL="${1:-Apps_CONVECTION3DPA}"
  OUTPUT="${2:-/tmp/cross_vendor_chains.pdf}"
  OUTPUT_DIR="$(dirname "$OUTPUT")"
  OUTPUT_FILE="$(basename "$OUTPUT")"
  KERNEL_ARG="--kernel $KERNEL"
  echo "Kernel:  $KERNEL"
fi

echo "Output:  $OUTPUT"
echo "Chains:  $TOP_CHAINS per vendor"

docker run --rm \
  --entrypoint="" \
  -v "$(pwd):/opt/leo:rw" \
  -v "$(pwd)/tests/data/pc:/data/pc:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e MPLCONFIGDIR=/tmp/mpl \
  -u "$(id -u):$(id -g)" \
  leo-base-universal:latest \
  bash -c "cd /opt/leo && uv run python scripts/plot_cross_vendor_chains.py \
    $KERNEL_ARG \
    --data-dir /data/pc/per-kernel \
    --top-chains $TOP_CHAINS \
    -o /output/$OUTPUT_FILE"

echo "Done: $OUTPUT"
