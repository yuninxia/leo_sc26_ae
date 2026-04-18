#!/bin/bash
# QuickSilver optimization benchmark — NVIDIA (Velocity-Bench CUDA)
# Single container: copies optimized headers, rebuilds, runs per-kernel profiling.
#
# Usage:
#   ./run_compare_nvidia.sh                          # run locally
#   ./run_compare_nvidia.sh --docker leo-quicksilver-nvidia-arm   # specify image
#   ./run_compare_nvidia.sh --machine gilgamesh      # run via SSH
#   ./run_compare_nvidia.sh --device 1               # specify GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-quicksilver-nvidia"
QS_ARGS="-i /opt/quicksilver/examples/CORAL2_Benchmark/Problem1/Coral2_P1.inp -x 16 -y 16 -z 16 -N 20"

GPU_DEVICE=0
MACHINE=""  # empty = run locally

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)  GPU_DEVICE="$2"; shift 2 ;;
        --machine) MACHINE="$2"; shift 2 ;;
        --docker)  DOCKER_IMAGE="$2"; shift 2 ;;
        *)         shift ;;  # ignore unknown options
    esac
done

echo "============================================"
echo " QuickSilver Original vs Optimized (NVIDIA)"
echo " Container: $DOCKER_IMAGE"
echo " Per-kernel timing with nsys"
echo " GPU:  $GPU_DEVICE"
echo " Machine: ${MACHINE:-localhost}"
echo "============================================"
echo ""

# Find host CUDA for nsys
run_on_target() {
    if [ -n "$MACHINE" ]; then
        ssh "$MACHINE" "$@"
    else
        eval "$@"
    fi
}

NSYS_MOUNT=""
for cuda_path in /usr/local/cuda /packages/cuda/12.9.0; do
    if run_on_target "test -d $cuda_path" 2>/dev/null; then
        NSYS_MOUNT="-v $cuda_path:/opt/cuda-host:ro"
        break
    fi
done

run_on_target "docker run --rm \
    --gpus device=$GPU_DEVICE \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e QS_DEVICE=GPU \
    -v $SCRIPT_DIR/optimized:/opt/qs-opt:ro \
    $NSYS_MOUNT \
    --entrypoint bash $DOCKER_IMAGE -c '
set -e
if [ -x /opt/cuda-host/bin/nsys ]; then
    export PATH=/opt/cuda-host/bin:\$PATH
fi

SRC=/opt/velocity-bench/QuickSilver/CUDA/src
QS=/opt/quicksilver/bin/qs
QS_ARGS=\"$QS_ARGS\"

cd /tmp

echo \"=== ORIGINAL (nsys per-kernel) ===\"
nsys profile --stats=true -f true -o orig \$QS \$QS_ARGS 2>&1 \
    | sed -n \"/cuda_gpu_kern_sum/,/Executing/p\" | head -15

echo \"\"
echo \"=== Building optimized version ===\"
for f in CollisionEvent.hh MacroscopicCrossSection.hh NuclearData.hh MCT.hh; do
    [ -f /opt/qs-opt/\$f ] && cp /opt/qs-opt/\$f \$SRC/\$f && echo \"  Replaced \$f\"
done
cd \$SRC && make clean > /dev/null 2>&1 && make USE_SM=90 CUDA_PATH=/usr/local/cuda -sj 2>&1 | tail -2
cp qs \$QS
echo \"Build complete.\"

echo \"\"
echo \"=== OPTIMIZED (nsys per-kernel) ===\"
cd /tmp
nsys profile --stats=true -f true -o opt \$QS \$QS_ARGS 2>&1 \
    | sed -n \"/cuda_gpu_kern_sum/,/Executing/p\" | head -15

echo \"\"
echo \"============================================\"
echo \" Compare CycleTrackingKernel total time\"
echo \" Speedup = original_ns / optimized_ns\"
echo \"============================================\"
'"
