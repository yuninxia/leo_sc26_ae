#!/bin/bash
# Measure per-kernel GPU time for RAJAPerf on Intel PVC using unitrace.
# Saves raw unitrace output per kernel for manual analysis.
#
# Usage:
#   ./run_unitrace_intel.sh                                    # All 15 kernels
#   ./run_unitrace_intel.sh --kernels Apps_LTIMES Apps_MASS3DEA
#   ./run_unitrace_intel.sh --npasses 200

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-rajaperf-intel-unitrace"
MACHINE="headroom"

ALL_KERNELS=(
    Apps_MASS3DEA Apps_LTIMES_NOVIEW Apps_LTIMES
    Polybench_3MM Polybench_2MM Polybench_GEMM
    Apps_PRESSURE Apps_ENERGY Apps_FIR
    Apps_ZONAL_ACCUMULATION_3D Apps_VOL3D Apps_DEL_DOT_VEC_2D
    Apps_DIFFUSION3DPA Apps_CONVECTION3DPA Apps_MASS3DPA
)

SELECTED_KERNELS=()
NPASSES=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernels|-k) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SELECTED_KERNELS+=("$1"); shift; done ;;
        --npasses) NPASSES="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ ${#SELECTED_KERNELS[@]} -eq 0 ]] && SELECTED_KERNELS=("${ALL_KERNELS[@]}")

OPT_SRC="$SCRIPT_DIR/optimized/src/apps"
OUT_DIR="$SCRIPT_DIR/unitrace_raw_intel"
mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

echo "============================================"
echo " RAJAPerf Per-Kernel Timing (Intel unitrace)"
echo " Kernels: ${#SELECTED_KERNELS[@]}"
echo " Passes:  $NPASSES"
echo " Output:  $OUT_DIR/"
echo "============================================"
echo ""

ssh "$MACHINE" docker run --rm \
    --device=/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -e ZE_AFFINITY_MASK=0 \
    -e ZE_ENABLE_TRACING_LAYER=1 \
    -v "$OPT_SRC:/opt/rajaperf-opt-src:ro" \
    -v "$SCRIPT_DIR/unitrace_inner.sh:/opt/unitrace_inner.sh:ro" \
    -v "$OUT_DIR:/data" \
    --entrypoint bash "$DOCKER_IMAGE" \
    /opt/unitrace_inner.sh "$NPASSES" "${SELECTED_KERNELS[@]}" 2>&1

echo ""
echo "Raw outputs saved to: $OUT_DIR/"
echo "Each kernel has: <kernel>_orig.txt and <kernel>_opt.txt"
