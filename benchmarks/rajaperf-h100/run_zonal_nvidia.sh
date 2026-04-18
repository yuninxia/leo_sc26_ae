#!/bin/bash
# Quick 10-run ZONAL_ACCUMULATION_3D benchmark on NVIDIA
set -e

cp -r /opt/rajaperf-optimized/src/apps/ZONAL_ACCUMULATION_3D-Cuda.cpp /opt/rajaperf-build/src/apps/
cd /opt/rajaperf-build && make -j72 > /dev/null 2>&1 && echo "BUILD OK"

extract_median() {
    # Run binary, get per-pass times (skip warmup pass 1), return median
    local bin="$1"
    "$bin" -k Apps_ZONAL_ACCUMULATION_3D -v Base_CUDA --npasses 11 --checkrun 1 --show-progress 2>&1 \
        | grep PASSED | grep -oP 'tuning -- \K[0-9.eE+-]+' \
        | tail -n+2 \
        | python3 -c "import sys,statistics; vals=sorted(float(l) for l in sys.stdin); print(f'{statistics.median(vals)*1000:.6f}')"
}

echo "run orig_ms opt_ms"
for run in $(seq 1 10); do
    orig=$(extract_median /opt/rajaperf/bin/raja-perf.exe)
    opt=$(extract_median /opt/rajaperf-build/bin/raja-perf.exe)
    echo "$run $orig $opt"
done
