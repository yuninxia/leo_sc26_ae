#!/bin/bash
# Test llama.cpp AMD with both pp128 and pp512, baseline + optimized
set -e

MODEL="${MODEL_DIR:-$HOME/models}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
OPT_MMQ="${LEO_ROOT:-$(pwd)}/benchmarks/llamacpp/optimized/mmq_amd_optimized.cuh"

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$MODEL":/workspace/model.gguf:ro \
    -v "$OPT_MMQ":/workspace/mmq_optimized.cuh:ro \
    --entrypoint bash leo-llamacpp-amd -c '
set -e

echo "========== BASELINE pp128 =========="
llama-bench -m /workspace/model.gguf -p 128 -n 0 -ngl 99 -r 10 2>&1 | tail -5

echo ""
echo "========== BASELINE pp512 =========="
llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r 10 2>&1 | tail -5

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
echo "========== OPTIMIZED pp128 =========="
llama-bench -m /workspace/model.gguf -p 128 -n 0 -ngl 99 -r 10 2>&1 | tail -5

echo ""
echo "========== OPTIMIZED pp512 =========="
llama-bench -m /workspace/model.gguf -p 512 -n 0 -ngl 99 -r 10 2>&1 | tail -5
'
