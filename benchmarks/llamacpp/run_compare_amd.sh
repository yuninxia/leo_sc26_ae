#!/bin/bash
# llama.cpp baseline-vs-optimized — AMD MI300A (Qwen2.5-1.5B Q4_K_M, mmq kernel).
# Mirrors run_compare_nvidia.sh; applies the LEO-derived MMQ rewrite by
# copying optimized/mmq_amd_optimized.cuh over ggml/src/ggml-cuda/mmq.cuh,
# rebuilding inside leo-llamacpp-amd, and re-running llama-bench.
#
# Usage:
#   ./run_compare_amd.sh                                    # defaults
#   ./run_compare_amd.sh --model /path/to/Qwen.gguf         # specify model
#   ./run_compare_amd.sh --runs 5                           # llama-bench reps
#   ./run_compare_amd.sh --device 1                         # AMD GPU id

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-llamacpp-amd"
MODEL="${MODEL_DIR:-$HOME/models}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
OPT_MMQ="$SCRIPT_DIR/optimized/mmq_amd_optimized.cuh"
RUNS=10
GPU_DEVICE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker) DOCKER_IMAGE="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --runs)   RUNS="$2"; shift 2 ;;
        --device) GPU_DEVICE="$2"; shift 2 ;;
        *)        shift ;;
    esac
done

echo "============================================"
echo " llama.cpp Original vs Optimized (AMD)"
echo " Container: $DOCKER_IMAGE"
echo " Model:     $(basename "$MODEL")"
echo " Runs:      $RUNS"
echo " GPU:       $GPU_DEVICE"
echo "============================================"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL"
    echo "       Run benchmarks/llamacpp/download_model.sh first."
    exit 1
fi
if [ ! -f "$OPT_MMQ" ]; then
    echo "ERROR: optimized header not found at $OPT_MMQ"
    exit 1
fi

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e ROCR_VISIBLE_DEVICES=$GPU_DEVICE \
    -v "$MODEL":/workspace/model.gguf:ro \
    -v "$OPT_MMQ":/workspace/mmq_optimized.cuh:ro \
    --entrypoint bash "$DOCKER_IMAGE" -c "
set -e

echo '=== BASELINE pp128 ==='
llama-bench -m /workspace/model.gguf -p 128 -n 0 -ngl 99 -r $RUNS 2>&1 | tail -5

echo ''
echo '=== BASELINE pp512 ==='
llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r $RUNS 2>&1 | tail -5

echo ''
echo '=== Applying LEO optimization (MMQ rewrite) ==='
cd /opt/llama.cpp
cp /workspace/mmq_optimized.cuh ggml/src/ggml-cuda/mmq.cuh
echo 'Replaced mmq.cuh'
cmake --build build -j\$(nproc) 2>&1 | tail -1
cp build/bin/llama-bench /opt/llamacpp/bin/
cp build/bin/libggml*.so* /opt/llamacpp/lib/
ldconfig

echo ''
echo '=== OPTIMIZED pp128 ==='
llama-bench -m /workspace/model.gguf -p 128 -n 0 -ngl 99 -r $RUNS 2>&1 | tail -5

echo ''
echo '=== OPTIMIZED pp512 ==='
llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r $RUNS 2>&1 | tail -5
"
