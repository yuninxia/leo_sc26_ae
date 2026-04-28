#!/bin/bash
# Kripke LTimes baseline-vs-optimized comparison — AMD MI300A.
#
# Top-level entry point that delegates to optimized/run_amd.sh (the underlying
# script that builds and runs both versions inside leo-kripke-amd). This
# wrapper exists so the AE artifact's HPC-app comparisons share the
# `run_compare_<vendor>.sh` naming convention referenced by the appendix
# (`bash benchmarks/<name>/run_compare_<vendor>.sh`).
#
# Usage:
#   ./run_compare_amd.sh            # 5 runs (baseline + optimized)
#   ./run_compare_amd.sh 10         # 10 runs each
#
# Requires: leo-kripke-amd Docker image + AMD GPU.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/optimized/run_amd.sh" "$@"
