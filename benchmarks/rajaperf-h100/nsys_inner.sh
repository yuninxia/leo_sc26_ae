#!/bin/bash
# Inner script to run inside Docker container
# Args: NPASSES KERNEL1 KERNEL2 ...
set -e

export PATH=/opt/cuda-host/bin:$PATH
EXEC=/opt/rajaperf/bin/raja-perf.exe
NPASSES=$1
shift
KERNELS=("$@")

extract_top_kernel_ns() {
    sed -n "/cuda_gpu_kern_sum/,/Executing/p" "$1" \
        | grep -E "^\s+[0-9]" | head -1 | awk '{print $2}'
}

echo "{"

FIRST=true
for KERNEL in "${KERNELS[@]}"; do
    echo "  Processing $KERNEL ..." >&2

    cd /tmp && rm -f *.nsys-rep *.sqlite 2>/dev/null || true

    # ---- Original ----
    nsys profile --stats=true -f true -o orig \
        $EXEC --variants Base_CUDA --checkrun 1 --npasses $NPASSES --kernels $KERNEL \
        > /tmp/orig.log 2>&1
    ORIG_NS=$(extract_top_kernel_ns /tmp/orig.log)

    # ---- Build optimized ----
    KBASE=${KERNEL#Apps_}
    # Polybench kernels: RAJAPerf CLI uses "Polybench_GEMM" but files are "POLYBENCH_GEMM-Cuda.cpp"
    KBASE=$(echo "$KBASE" | sed 's/^Polybench_/POLYBENCH_/')
    cp /opt/rajaperf-opt-src/${KBASE}*.cpp /opt/RAJAPerf/src/apps/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.hpp /opt/RAJAPerf/src/apps/ 2>/dev/null || true
    # Polybench kernels live in src/polybench/ in some RAJAPerf versions
    cp /opt/rajaperf-opt-src/${KBASE}*.cpp /opt/RAJAPerf/src/polybench/ 2>/dev/null || true
    cp /opt/rajaperf-opt-src/${KBASE}*.hpp /opt/RAJAPerf/src/polybench/ 2>/dev/null || true
    cd /opt/rajaperf-build && make -j$(nproc) > /dev/null 2>&1

    # ---- Optimized ----
    cd /tmp && rm -f *.nsys-rep *.sqlite 2>/dev/null || true
    nsys profile --stats=true -f true -o opt \
        /opt/rajaperf-build/bin/raja-perf.exe --variants Base_CUDA --checkrun 1 --npasses $NPASSES --kernels $KERNEL \
        > /tmp/opt.log 2>&1
    OPT_NS=$(extract_top_kernel_ns /tmp/opt.log)

    # ---- Restore original source ----
    cd /opt/RAJAPerf
    git checkout -- src/apps/${KBASE}*.cpp src/apps/${KBASE}*.hpp 2>/dev/null || true
    git checkout -- src/polybench/${KBASE}*.cpp src/polybench/${KBASE}*.hpp 2>/dev/null || true
    cd /opt/rajaperf-build && make -j$(nproc) > /dev/null 2>&1

    # ---- Output ----
    if [ -n "$ORIG_NS" ] && [ -n "$OPT_NS" ]; then
        SPEEDUP=$(awk "BEGIN { s=$ORIG_NS/$OPT_NS; printf \"%.2f\", (s<1.0) ? 1.00 : s }")
    else
        SPEEDUP="null"
    fi

    $FIRST || echo ","
    FIRST=false
    printf '  "%s": {"original_ns": %s, "optimized_ns": %s, "speedup": %s}' \
        "$KERNEL" "${ORIG_NS:-null}" "${OPT_NS:-null}" "$SPEEDUP"
done

echo ""
echo "}"
