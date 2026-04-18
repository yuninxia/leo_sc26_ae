#!/bin/bash
# Profile a CUDA kernel with HPCToolkit + Leo.
# Run inside leo-astra-nvidia container on gilgamesh:
#   docker run --rm --gpus all ... leo-astra-nvidia:latest -c \
#     '/opt/scripts/verify_leo_profile.sh /path/to/kernel.cu v3 256 8192 200'
set -e

KERNEL_CU="${1:-/opt/astra/test/rms/rms_v1.cu}"
TAG="${2:-default}"
B="${3:-256}"
D="${4:-8192}"
ITERS="${5:-200}"
WORK=/tmp/leo_profile_test

CACHE=$WORK/hpcstruct-cache
mkdir -p "$WORK" "$CACHE" && cd "$WORK"

# Torch paths (no Python needed — pybind11 parts are stripped from kernel)
TORCH_DIR=$(python3.11 -c "import torch; print(torch.utils.cmake_prefix_path + '/../..')")
export LD_LIBRARY_PATH=$TORCH_DIR/lib:$LD_LIBRARY_PATH

echo "============================================"
echo " Kernel:  $KERNEL_CU"
echo " Tag:     $TAG"
echo " Shape:   ${B}x${D}, $ITERS iters"
echo " Torch:   $TORCH_DIR"
echo "============================================"

# 1. Compile: driver + kernel → standalone executable with -lineinfo
echo -e "\n[1/5] Compiling with -lineinfo..."

# Strip pybind11/Python parts from kernel .cu (they're for Python, not needed here)
# Replace torch/extension.h → torch/torch.h, remove PYBIND11_MODULE block
cp "$KERNEL_CU" $WORK/kernel.cu
sed -i 's|#include <torch/extension.h>|#include <torch/torch.h>|' $WORK/kernel.cu
sed -i '/PYBIND11_MODULE/,/^}/d' $WORK/kernel.cu

nvcc /opt/scripts/astra_profile_driver.cu $WORK/kernel.cu \
    -I$TORCH_DIR/include \
    -I$TORCH_DIR/include/torch/csrc/api/include \
    -L$TORCH_DIR/lib \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
    -gencode arch=compute_90,code=sm_90 \
    -O3 -lineinfo --use_fast_math -std=c++17 \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC \
    -o $WORK/kernel_profile
echo "  OK: $(ls -lh $WORK/kernel_profile | awk '{print $5}')"

# 2. Quick sanity test
echo -e "\n[2/5] Quick test..."
$WORK/kernel_profile 32 4096 5

# 3. Profile with hpcrun
echo -e "\n[3/5] hpcrun -e gpu=cuda,pc ..."
MEAS=$WORK/hpctoolkit-${TAG}-measurements
rm -rf $MEAS $WORK/hpctoolkit-${TAG}-database
hpcrun -o $MEAS -e gpu=cuda,pc $WORK/kernel_profile $B $D $ITERS
echo "  gpubins:"
ls -lh $MEAS/gpubins/ 2>/dev/null || echo "  (none)"

# 4. hpcstruct + hpcprof
echo -e "\n[4/5] hpcstruct + hpcprof..."
export HPCTOOLKIT_HPCSTRUCT_CACHE=$CACHE
hpcstruct --gpucfg yes $MEAS
hpcprof $MEAS  # default output: hpctoolkit-kernel-database (Leo auto-discovers this)

# 5. Leo analysis
echo -e "\n[5/5] Leo analysis..."
cd /opt/leo && uv run python scripts/analyze_benchmark.py $MEAS --arch h100 --top-n 2

echo -e "\n============================================"
echo " Done. Results in $WORK/"
echo "============================================"
