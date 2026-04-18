#!/bin/bash
# Minimal test to verify AMD mmq optimization takes effect
# Key fix: must copy freshly built libggml-hip.so in addition to llama-bench
set -e

MODEL="${MODEL_DIR:-$HOME/models}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
OPT_MMQ="${LEO_ROOT:-$(pwd)}/benchmarks/llamacpp/optimized/mmq_amd_optimized.cuh"

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$MODEL":/workspace/model.gguf:ro \
    -v "$OPT_MMQ":/workspace/mmq_optimized.cuh:ro \
    --entrypoint bash leo-llamacpp-amd -c '
set -e

echo "========== APPLYING OPTIMIZATION =========="
cd /opt/llama.cpp
cp /workspace/mmq_optimized.cuh ggml/src/ggml-cuda/mmq.cuh

# Clean build
rm -rf build
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_MFMA=ON 2>&1 | tail -3
cmake --build build -j$(nproc) 2>&1 | tail -3

# Copy BOTH binary AND shared libraries
cp build/bin/llama-bench /opt/llamacpp/bin/
cp build/bin/libggml*.so* /opt/llamacpp/lib/
ldconfig
echo "Copied llama-bench + all libggml shared libs"

echo ""
echo "========== DEBUG OUTPUT =========="
llama-bench -m /workspace/model.gguf -p 128 -n 0 -ngl 99 -r 1 2>&1 | grep -i "DEBUG\|pp128"

echo ""
echo "========== KERNEL NAMES (rocprof) =========="
rocprof --stats -- llama-bench -m /workspace/model.gguf -p 128 -n 0 -ngl 99 -r 1 2>&1 | grep -i "mul_mat_q" | head -5
'
