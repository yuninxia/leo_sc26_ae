#!/bin/bash
# Task 3 entry point: run a RAJAPerf kernel (original vs optimized) on one or all vendors.
#
# This is a thin wrapper over benchmarks/rajaperf-h100/run_compare.sh. It translates
# `--vendor {nvidia|amd|intel|all}` into the `--docker leo-rajaperf-<vendor>` form that
# the underlying script expects, so reviewers have a single uniform CLI per Section VI.
#
# Usage:
#   bash scripts/evaluation/run_workload_rajaperf.sh --kernel MASS3DEA --vendor nvidia
#   bash scripts/evaluation/run_workload_rajaperf.sh --kernel MASS3DEA --vendor all
#   bash scripts/evaluation/run_workload_rajaperf.sh --help
#
# Prerequisites: the vendor-specific Docker image (leo-rajaperf-<vendor>) must exist.
# Portable build (any GPU host):
#     docker build -f scripts/evaluation/docker/Dockerfile.rajaperf-<vendor> \
#                  -t leo-rajaperf-<vendor> .
# Or authors'-side SSH helper (Rice cluster only):
#     bash scripts/evaluation/build_containers.sh <vendor> --workload rajaperf
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INNER="$LEO_ROOT/benchmarks/rajaperf-h100/run_compare.sh"

KERNEL=""
VENDOR=""
EXTRA_ARGS=()

print_help() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel)     KERNEL="$2"; shift 2 ;;
    --vendor)     VENDOR="$2"; shift 2 ;;
    -h|--help)    print_help; exit 0 ;;
    *)            EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$VENDOR" ]]; then
  echo "ERROR: --vendor is required (nvidia|amd|intel|all)" >&2
  print_help
  exit 1
fi

if [[ ! -x "$INNER" ]]; then
  echo "ERROR: inner script not found or not executable: $INNER" >&2
  exit 1
fi

run_one() {
  local v="$1"
  local image="leo-rajaperf-${v}"
  echo "=== Running RAJAPerf compare on ${v} (image: ${image}) ==="
  local args=(--docker "$image")
  [[ -n "$KERNEL" ]] && args+=(--kernel "$KERNEL")
  args+=("${EXTRA_ARGS[@]}")
  bash "$INNER" "${args[@]}"
}

case "$VENDOR" in
  nvidia|amd|intel)
    run_one "$VENDOR"
    ;;
  all)
    for v in nvidia amd intel; do
      run_one "$v" || echo "WARN: ${v} run failed — continuing" >&2
    done
    ;;
  *)
    echo "ERROR: --vendor must be one of: nvidia, amd, intel, all (got: $VENDOR)" >&2
    exit 1
    ;;
esac
