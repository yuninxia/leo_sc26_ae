#!/bin/bash
# A/B comparison of original vs optimized cuJSON parse performance.
# Usage: bench_compare.sh [NWARMUP] [NRUNS]
#   Defaults: 5 warmup, 20 measurement runs

DATA=/opt/cuJSON/dataset/bench_large.json
ORIG=/opt/cuJSON/bin/cujson_standard_original
OPT=/opt/cuJSON/bin/cujson_standard_optimized
NWARMUP=${1:-5}
NRUNS=${2:-20}

echo "=== Warmup ($NWARMUP runs each) ==="
for i in $(seq 1 $NWARMUP); do
    $ORIG $DATA > /dev/null 2>&1
    $OPT  $DATA > /dev/null 2>&1
    echo "  warmup $i/$NWARMUP done"
done

echo ""
echo "=== Original ($NRUNS runs) ==="
for i in $(seq 1 $NRUNS); do
    $ORIG $DATA 2>&1 | grep -o 'took [0-9.]* ms'
    if (( i % 5 == 0 )); then echo "  --- original $i/$NRUNS done ---"; fi
done

echo ""
echo "=== Optimized ($NRUNS runs) ==="
for i in $(seq 1 $NRUNS); do
    $OPT $DATA 2>&1 | grep -o 'took [0-9.]* ms'
    if (( i % 5 == 0 )); then echo "  --- optimized $i/$NRUNS done ---"; fi
done
