#!/bin/bash
# QuickSilver optimization benchmark — AMD MI300A (Velocity-Bench HIP)
# Single container: copies optimized headers, rebuilds, runs per-kernel profiling.
#
# Usage:
#   ./run_compare_amd.sh              # per-kernel timing with rocprofv3
#   ./run_compare_amd.sh --device 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-quicksilver-amd"
QS_ARGS="-i /opt/quicksilver/examples/CORAL2_Benchmark/Problem1/Coral2_P1.inp -x 16 -y 16 -z 16 -N 20"

GPU_DEVICE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --device) GPU_DEVICE="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " QuickSilver Original vs Optimized (AMD)"
echo " Container: $DOCKER_IMAGE (Velocity-Bench HIP)"
echo " Per-kernel timing with rocprofv3"
echo " GPU:  $GPU_DEVICE"
echo "============================================"
echo ""

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e ROCR_VISIBLE_DEVICES=$GPU_DEVICE \
    -v "$SCRIPT_DIR/optimized-amd:/opt/qs-opt:ro" \
    --entrypoint bash "$DOCKER_IMAGE" -c '
set -e

SRC=/opt/velocity-bench/QuickSilver/HIP/src
QS=/opt/quicksilver/bin/qs
QS_ARGS="'"$QS_ARGS"'"

cd /tmp

echo "=== ORIGINAL (rocprofv3 per-kernel) ==="
rocprofv3 --kernel-trace --stats -S -- $QS $QS_ARGS 2>&1 | grep -E "CycleTracking|Kernel Name|Total"

echo ""
echo "=== Building optimized version ==="
for f in CollisionEvent.hh MacroscopicCrossSection.hh NuclearData.hh MCT.hh; do
    [ -f /opt/qs-opt/$f ] && cp /opt/qs-opt/$f $SRC/$f && echo "  Replaced $f"
done
cd $SRC && make clean > /dev/null 2>&1
make ROCM_PATH=/opt/rocm CXXFLAGS="-I/opt/rocm/include/ -ggdb -fno-omit-frame-pointer --offload-arch=gfx942" -sj 2>&1 | tail -2
cp qs $QS
echo "Build complete."

echo ""
echo "=== OPTIMIZED (rocprofv3 per-kernel) ==="
cd /tmp
rocprofv3 --kernel-trace --stats -S -- $QS $QS_ARGS 2>&1 | grep -E "CycleTracking|Kernel Name|Total"

echo ""
echo "============================================"
echo " Compare CycleTrackingKernel total time"
echo " Speedup = original_ns / optimized_ns"
echo "============================================"
'
