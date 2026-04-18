#!/bin/bash
# Measure per-kernel GPU time for all RAJAPerf kernels on NVIDIA H100 using nsys.
# Outputs JSON with original/optimized ns and speedup.
#
# Usage:
#   ./run_nsys_nvidia.sh                                    # All 15 kernels
#   ./run_nsys_nvidia.sh --kernels Apps_LTIMES Apps_MASS3DEA
#   ./run_nsys_nvidia.sh --npasses 200

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-rajaperf-nvidia"
MACHINE="gilgamesh"
GPU_DEVICE=0

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
        --device)  GPU_DEVICE="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ ${#SELECTED_KERNELS[@]} -eq 0 ]] && SELECTED_KERNELS=("${ALL_KERNELS[@]}")

OPT_SRC="$SCRIPT_DIR/optimized/src/apps"
RESULTS="$SCRIPT_DIR/nsys_results_nvidia.json"

echo "============================================"
echo " RAJAPerf Per-Kernel Timing (NVIDIA nsys)"
echo " Kernels: ${#SELECTED_KERNELS[@]}"
echo " Passes:  $NPASSES"
echo "============================================"
echo ""

ssh "$MACHINE" docker run --rm \
    --gpus "device=$GPU_DEVICE" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$OPT_SRC:/opt/rajaperf-opt-src:ro" \
    -v /packages/cuda/12.9.0:/opt/cuda-host:ro \
    -v "$SCRIPT_DIR/nsys_inner.sh:/opt/nsys_inner.sh:ro" \
    --entrypoint bash "$DOCKER_IMAGE" \
    /opt/nsys_inner.sh "$NPASSES" "${SELECTED_KERNELS[@]}" \
    2>"$SCRIPT_DIR/nsys_nvidia.log" | tee "$RESULTS"

echo ""
echo "Results: $RESULTS"
echo "Log: $SCRIPT_DIR/nsys_nvidia.log"
