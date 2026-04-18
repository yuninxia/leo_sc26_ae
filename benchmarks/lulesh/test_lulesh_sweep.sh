#!/bin/bash
# Sweep TEAMS/THREADS configurations for both original and optimized LULESH
# Tests each config with 5 warmups + 20 interleaved pairs
set -e

ORIG_SRC=/opt/lulesh-src/LULESH/omp_4.0
OPT_SRC=/opt/lulesh-opt-src
ARGS="-s 30 -i 100"

# Configurations to sweep: "TEAMS THREADS"
CONFIGS=(
    "128 64"
    "256 32"
    "512 16"
    "512 32"
    "1024 16"
    "1024 32"
)

WARMUPS=5
ROUNDS=20

patch_makefile() {
    sed -i 's|$(CXX) -fopenmp=libomp.*-o $@|$(CXX) $(CXXFLAGS) $(OBJECTS2.0) $(LDFLAGS) -lstdc++ -lm -o $@|' Makefile
    sed -i 's|-L/usr/local/cuda/nvvm/libdevice||' Makefile
}

build_variant() {
    local src_dir=$1
    local build_dir=$2
    local teams=$3
    local threads=$4

    mkdir -p "$build_dir" && cd "$build_dir"
    cp "$src_dir"/*.cc "$src_dir"/*.h "$src_dir"/Makefile "$src_dir"/README .
    patch_makefile
    make clean 2>/dev/null || true
    make CXX="icpx" \
         CXXFLAGS="-DUSE_MPI=0 -g -O2 -fiopenmp -fopenmp-targets=spir64 -DTEAMS=$teams -DTHREADS=$threads" \
         LDFLAGS="" \
         OMPFLAGS="" 2>&1

    if [ ! -x lulesh2.0 ]; then
        echo "BUILD FAILED ($build_dir, TEAMS=$teams THREADS=$threads)"
        exit 1
    fi
}

echo "=============================================="
echo "LULESH TEAMS/THREADS Sweep"
echo "Configs: ${#CONFIGS[@]}, Warmups: $WARMUPS, Rounds: $ROUNDS"
echo "Problem: $ARGS"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
    read -r teams threads <<< "$cfg"
    echo ""
    echo "====== Config: TEAMS=$teams THREADS=$threads ======"

    # Build both variants with this config
    echo "--- Building original (TEAMS=$teams THREADS=$threads) ---"
    build_variant "$ORIG_SRC" /opt/sweep-orig-build "$teams" "$threads"
    ORIG=/opt/sweep-orig-build/lulesh2.0

    echo "--- Building optimized (TEAMS=$teams THREADS=$threads) ---"
    build_variant "$OPT_SRC" /opt/sweep-opt-build "$teams" "$threads"
    OPT=/opt/sweep-opt-build/lulesh2.0

    # Warmup
    echo "--- Warmup ($WARMUPS each) ---"
    for w in $(seq 1 $WARMUPS); do
        $ORIG $ARGS > /dev/null 2>&1
        $OPT $ARGS > /dev/null 2>&1
    done
    echo "Done"

    # Interleaved measurement
    echo "--- $ROUNDS interleaved pairs ---"
    for r in $(seq 1 $ROUNDS); do
        if [ $((r % 2)) -eq 1 ]; then
            ot=$($ORIG $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
            pt=$($OPT  $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
        else
            pt=$($OPT  $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
            ot=$($ORIG $ARGS 2>&1 | grep "Elapsed time" | awk '{print $4}')
        fi
        printf "  Round %2d: orig=%ss  opt=%ss\n" "$r" "$ot" "$pt"
    done
done

echo ""
echo "====== Sweep complete ======"
