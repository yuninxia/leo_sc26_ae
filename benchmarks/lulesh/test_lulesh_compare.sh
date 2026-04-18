#!/bin/bash
# Compare original vs optimized LULESH on Intel PVC
# Run inside the leo-lulesh-intel Docker container
set -e

ORIG=/opt/lulesh/bin/lulesh2.0
OPT_SRC=/opt/lulesh-opt-src

# Build optimized version using the same Makefile approach as the Docker image
echo "=== Building optimized LULESH ==="
mkdir -p /opt/lulesh-opt-build && cd /opt/lulesh-opt-build
cp $OPT_SRC/*.cc $OPT_SRC/*.h $OPT_SRC/Makefile $OPT_SRC/README .

# Apply the same Makefile patch as the Dockerfile
sed -i 's|$(CXX) -fopenmp=libomp.*-o $@|$(CXX) $(CXXFLAGS) $(OBJECTS2.0) $(LDFLAGS) -lstdc++ -lm -o $@|' Makefile
sed -i 's|-L/usr/local/cuda/nvvm/libdevice||' Makefile

# Build with same flags as Docker image
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

# Test args - use moderate size for quick comparison
ARGS="-s 30 -i 100"

echo ""
echo "=== Verification (correctness check) ==="
echo "--- Original ---"
$ORIG $ARGS 2>&1 | grep -E "Final Origin Energy|MaxAbsDiff|MaxRelDiff|Elapsed"
echo "--- Optimized ---"
$OPT $ARGS 2>&1 | grep -E "Final Origin Energy|MaxAbsDiff|MaxRelDiff|Elapsed"

echo ""
echo "=== Warmup ==="
$ORIG $ARGS > /dev/null 2>&1
$OPT $ARGS > /dev/null 2>&1
echo "Done"

echo ""
echo "=== 5 interleaved pairs ==="
for r in $(seq 1 5); do
    if [ $((r % 2)) -eq 1 ]; then
        ot=$($ORIG $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
        pt=$($OPT $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
    else
        pt=$($OPT $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
        ot=$($ORIG $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
    fi
    printf "  Round %d: orig=%ss  opt=%ss\n" "$r" "$ot" "$pt"
done
