#!/bin/bash
# Kripke LTimes baseline-vs-optimized comparison — NVIDIA H100/GH200.
#
# Top-level entry point that delegates to optimized/run_nvidia.sh.
# Matches the AE-appendix naming convention used by lulesh/quicksilver/xsbench.
#
# Usage:
#   ./run_compare_nvidia.sh         # 5 runs (baseline + optimized)
#   ./run_compare_nvidia.sh 10      # 10 runs each
#
# Requires: leo-kripke-nvidia Docker image + NVIDIA GPU.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/optimized/run_nvidia.sh" "$@"
