#!/bin/bash
# Quick test: build optimized LULESH, verify correctness, run interleaved pairs
set -e

OPT_SRC=/opt/lulesh-opt-src
ORIG=/opt/lulesh/bin/lulesh2.0

echo "=== Building optimized LULESH ==="
mkdir -p /opt/lulesh-opt-build && cd /opt/lulesh-opt-build
cp $OPT_SRC/*.cc $OPT_SRC/*.h $OPT_SRC/Makefile $OPT_SRC/README .

# Apply same Makefile patch as Dockerfile
sed -i 's|$(CXX) -fopenmp=libomp.*-o $@|$(CXX) $(CXXFLAGS) $(OBJECTS2.0) $(LDFLAGS) -lstdc++ -lm -o $@|' Makefile
sed -i 's|-L/usr/local/cuda/nvvm/libdevice||' Makefile

make clean 2>/dev/null || true
make CXX="icpx" \
     CXXFLAGS="-DUSE_MPI=0 -g -O2 -fiopenmp -fopenmp-targets=spir64" \
     LDFLAGS="" \
     OMPFLAGS="" 2>&1

if [ ! -x lulesh2.0 ]; then
    echo "BUILD FAILED"
    exit 1
fi
OPT=/opt/lulesh-opt-build/lulesh2.0
echo "Build OK"

ARGS="-s 10 -i 10"

echo ""
echo "=== Verification ==="
echo "--- Original ---"
$ORIG $ARGS 2>&1 | grep -E "Final Origin Energy|MaxAbsDiff|MaxRelDiff"
echo "--- Optimized ---"
$OPT $ARGS 2>&1 | grep -E "Final Origin Energy|MaxAbsDiff|MaxRelDiff"

echo ""
echo "=== Warmup ==="
$ORIG $ARGS > /dev/null 2>&1
$OPT $ARGS > /dev/null 2>&1
echo "Done"

echo ""
echo "=== 10 interleaved pairs ==="
for r in $(seq 1 10); do
    if [ $((r % 2)) -eq 1 ]; then
        ot=$($ORIG $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
        pt=$($OPT $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
    else
        pt=$($OPT $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
        ot=$($ORIG $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
    fi
    printf "  Round %2d: orig=%ss  opt=%ss\n" "$r" "$ot" "$pt"
done
