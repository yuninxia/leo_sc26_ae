#!/bin/bash
# Measure per-kernel timing for llama.cpp mul_mat_q on AMD MI300A using rocprofv3
set -e

MODEL="${MODEL_DIR:-$HOME/models}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
OPT_MMQ="${LEO_ROOT:-$(pwd)}/benchmarks/llamacpp/optimized/mmq_amd_optimized.cuh"

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$MODEL":/workspace/model.gguf:ro \
    -v "$OPT_MMQ":/workspace/mmq_optimized.cuh:ro \
    --entrypoint bash leo-llamacpp-amd -c '
set -e

echo "========== BASELINE per-kernel (rocprofv3) =========="
rocprofv3 --kernel-trace --stats -- llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r 1 2>&1 | grep -i "mul_mat_q" | head -5
echo ""
echo "--- All top kernels ---"
rocprofv3 --kernel-trace --stats -- llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r 1 2>&1 | grep "KERNEL_DISPATCH" | sort -t"|" -k5 -rn | head -10

echo ""
echo "========== APPLYING OPTIMIZATION =========="
cd /opt/llama.cpp
cp /workspace/mmq_optimized.cuh ggml/src/ggml-cuda/mmq.cuh
echo "Replaced mmq.cuh"
cmake --build build -j$(nproc) 2>&1 | tail -1
# Must copy shared libs too - the MMQ code lives in libggml-hip.so
cp build/bin/llama-bench /opt/llamacpp/bin/
cp build/bin/libggml*.so* /opt/llamacpp/lib/
ldconfig

echo ""
echo "========== OPTIMIZED per-kernel (rocprofv3) =========="
rocprofv3 --kernel-trace --stats -- llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r 1 2>&1 | grep -i "mul_mat_q" | head -5
echo ""
echo "--- All top kernels ---"
rocprofv3 --kernel-trace --stats -- llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r 1 2>&1 | grep "KERNEL_DISPATCH" | sort -t"|" -k5 -rn | head -10
'
