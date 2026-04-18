#!/bin/bash
# llama.cpp optimization benchmark — NVIDIA GH200 (mmq kernel)
# Measures baseline vs LEO-guided optimization (ids_dst[j] -> j).
#
# Usage:
#   ./run_compare_nvidia.sh                                    # defaults
#   ./run_compare_nvidia.sh --docker leo-llamacpp-nvidia-arm   # specify image
#   ./run_compare_nvidia.sh --model /path/to/model.gguf        # specify model
#   ./run_compare_nvidia.sh --runs 5                           # number of llama-bench repeats

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-llamacpp-nvidia-arm"
MODEL="${MODEL_DIR:-$HOME/models}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
RUNS=3
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
echo " llama.cpp Original vs Optimized (NVIDIA)"
echo " Container: $DOCKER_IMAGE"
echo " Model:     $(basename "$MODEL")"
echo " Runs:      $RUNS"
echo " GPU:       $GPU_DEVICE"
echo "============================================"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL"
    exit 1
fi

docker run --rm --gpus "device=$GPU_DEVICE" \
    -v "$MODEL":/workspace/model.gguf:ro \
    --entrypoint bash "$DOCKER_IMAGE" -c "
set -e

echo '=== BASELINE (Qwen2.5-1.5B, Q4_K_M, pp512) ==='
llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r $RUNS 2>&1 | tail -5

echo ''
echo '=== Applying LEO optimization (ids_dst[j] -> j) ==='
cd /opt/llama.cpp
sed -i 's/dst\[ids_dst\[j\]\*stride + i\]/dst[j*stride + i]/' ggml/src/ggml-cuda/mmq.cuh
echo \"Changed \$(grep -c 'dst\[j\*stride' ggml/src/ggml-cuda/mmq.cuh) lines\"
cmake --build build -j\$(nproc) 2>&1 | tail -1
cp build/bin/llama-bench /opt/llamacpp/bin/

echo ''
echo '=== OPTIMIZED ==='
llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r $RUNS 2>&1 | tail -5
"
