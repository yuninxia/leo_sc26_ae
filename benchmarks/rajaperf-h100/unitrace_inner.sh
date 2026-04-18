#!/bin/bash
# Inner script for Intel unitrace measurement.
# Saves full unitrace output per kernel to /data/<kernel>_{orig,opt}.txt
# Args: NPASSES KERNEL1 KERNEL2 ...
set -e

EXEC=/opt/rajaperf/bin/raja-perf.exe
NPASSES=$1
shift
KERNELS=("$@")

mkdir -p /data

for KERNEL in "${KERNELS[@]}"; do
    echo "Processing $KERNEL ..."

    # ---- Original ----
    cd /tmp
    unitrace -d $EXEC --variants Base_SYCL --checkrun 1 --npasses $NPASSES --kernels $KERNEL \
        > /data/${KERNEL}_orig.txt 2>&1
    echo "  original done"

    # ---- Build optimized ----
    KBASE=${KERNEL#Apps_}
    KBASE=$(echo "$KBASE" | sed "s/^Polybench_/POLYBENCH_/")
    cp /opt/rajaperf-opt-src/${KBASE}*.cpp /opt/RAJAPerf/src/apps/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.cpp /opt/RAJAPerf/src/polybench/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.hpp /opt/RAJAPerf/src/polybench/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.hpp /opt/RAJAPerf/src/apps/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.cpp /opt/RAJAPerf/src/polybench/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.hpp /opt/RAJAPerf/src/polybench/ 2>/dev/null || true
    cd /opt/rajaperf-build && make -j$(nproc) > /dev/null 2>&1

    # ---- Optimized ----
    cd /tmp
    unitrace -d /opt/rajaperf-build/bin/raja-perf.exe --variants Base_SYCL --checkrun 1 --npasses $NPASSES --kernels $KERNEL \
        > /data/${KERNEL}_opt.txt 2>&1
    echo "  optimized done"

    # ---- Restore ----
    cd /opt/RAJAPerf
    git checkout -- src/apps/${KBASE}*.cpp src/apps/${KBASE}*.hpp 2>/dev/null || true
    git checkout -- src/polybench/${KBASE}*.cpp src/polybench/${KBASE}*.hpp 2>/dev/null || true
    cd /opt/rajaperf-build && make -j$(nproc) > /dev/null 2>&1
done

echo ""
echo "All done. Raw outputs saved to /data/"
ls -la /data/*.txt
