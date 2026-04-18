#!/bin/bash
# Compare original (upstream) vs optimized (fork) LULESH on Intel PVC.
#
# Runs inside Docker container (leo-lulesh-intel) which already has the
# upstream LULESH pre-built. Only the optimized fork needs to be built.
#
# Usage:
#   ./run_compare_intel.sh                          # Build opt, compare (5 pairs)
#   ./run_compare_intel.sh --skip-build             # Reuse optimized build
#   ./run_compare_intel.sh --size 45 --iters 200    # Custom problem size
#   ./run_compare_intel.sh --pairs 20               # More measurement pairs
#   ./run_compare_intel.sh --device 0               # Use GPU device 0
#   ./run_compare_intel.sh --diff                   # Show source diff

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-lulesh-intel"

# Defaults
SKIP_BUILD=false
SHOW_DIFF=false
SIZE=30
ITERS=100
PAIRS=5
GPU_DEVICE=0

# ==================== Argument parsing ====================
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --vendor)      shift 2 ;;  # ignored, always intel
        --skip-build)  SKIP_BUILD=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --diff)        SHOW_DIFF=true; shift ;;
        --size|-s)     SIZE="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --iters|-i)    ITERS="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --pairs)       PAIRS="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --device)      GPU_DEVICE="$2"; PASSTHROUGH_ARGS+=("--device" "$2"); shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==================== Show diff (runs locally, no Docker needed) ====================
if [ "$SHOW_DIFF" = true ]; then
    echo "Source differences between upstream and fork (omp_4.0):"
    echo ""
    diff -ru "$SCRIPT_DIR/fork/omp_4.0/" "$SCRIPT_DIR/optimized/" 2>/dev/null || true
    exit 0
fi

# ==================== Docker re-launch ====================
if [ ! -f /.dockerenv ]; then
    echo "Launching inside Docker: $DOCKER_IMAGE (device=$GPU_DEVICE)"
    echo ""

    exec docker run --rm \
        --device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
        -e ZE_AFFINITY_MASK=$GPU_DEVICE \
        -e SYCL_CACHE_PERSISTENT=1 \
        -v /tmp/sycl-cache:/root/.cache \
        -v "$SCRIPT_DIR":/opt/lulesh-compare:ro \
        --entrypoint /bin/bash \
        "$DOCKER_IMAGE" \
        -c "
            cp -r /opt/lulesh-compare/fork/omp_4.0 /opt/lulesh-original
            cp -r /opt/lulesh-compare/optimized /opt/lulesh-optimized
            cp /opt/lulesh-compare/run_compare_intel.sh /opt/run_compare_intel.sh
            chmod +x /opt/run_compare_intel.sh
            cd /opt
            bash /opt/run_compare_intel.sh --vendor intel ${PASSTHROUGH_ARGS[*]}
        "
fi

# ==================== Inside Docker ====================
SRC_ORIGINAL="/opt/lulesh-original"
SRC_OPTIMIZED="/opt/lulesh-optimized"

# ==================== Build both versions ====================
if [ "$SKIP_BUILD" = false ]; then
    echo "============================================"
    echo " Building original + optimized LULESH (Intel PVC)"
    echo "============================================"
    echo ""

    BUILD_ARGS='CXX=icpx CXXFLAGS="-DUSE_MPI=0 -g -O2 -fiopenmp -fopenmp-targets=spir64" LDFLAGS="" OMPFLAGS=""'

    for variant in original optimized; do
        src="/opt/lulesh-${variant}"
        cd "$src"
        sed -i 's|$(CXX) -fopenmp=libomp.*-o $@|$(CXX) $(CXXFLAGS) $(OBJECTS2.0) $(LDFLAGS) -lstdc++ -lm -o $@|' Makefile
        sed -i 's|-L/usr/local/cuda/nvvm/libdevice||' Makefile
        echo "  [${variant}] make clean && make ..."
        make clean > /dev/null 2>&1 || true
        if eval "make $BUILD_ARGS" > make.log 2>&1; then
            echo "  [${variant}] OK"
        else
            echo "  [${variant}] FAILED (see make.log)"
            tail -20 make.log
            exit 1
        fi
    done
    echo ""
fi

BIN_ORIGINAL="$SRC_ORIGINAL/lulesh2.0"
BIN_OPTIMIZED="$SRC_OPTIMIZED/lulesh2.0"

if [ ! -x "$BIN_OPTIMIZED" ]; then
    echo "ERROR: Optimized binary not found: $BIN_OPTIMIZED"
    echo "       Run without --skip-build to build first."
    exit 1
fi

# ==================== Verification ====================
LULESH_ARGS="-s $SIZE -i $ITERS"

echo "============================================"
echo " LULESH Original vs Optimized (Intel PVC)"
echo " Args:      $LULESH_ARGS"
echo " Pairs:     $PAIRS (consecutive)"
echo " Original:  $BIN_ORIGINAL"
echo " Optimized: $BIN_OPTIMIZED"
echo "============================================"
echo ""

echo "--- Verification ---"
tmpfile_orig="/tmp/lulesh_verify_orig.out"
tmpfile_opt="/tmp/lulesh_verify_opt.out"

$BIN_ORIGINAL $LULESH_ARGS > "$tmpfile_orig" 2>&1 || true
$BIN_OPTIMIZED $LULESH_ARGS > "$tmpfile_opt" 2>&1 || true

orig_energy=$(grep "Final Origin Energy" "$tmpfile_orig" | awk '{print $NF}')
opt_energy=$(grep "Final Origin Energy" "$tmpfile_opt" | awk '{print $NF}')

echo "  Original:  Final Origin Energy = $orig_energy"
echo "  Optimized: Final Origin Energy = $opt_energy"

if [ "$orig_energy" = "$opt_energy" ]; then
    echo "  Verification: PASS (energies match)"
else
    echo "  Verification: WARNING (energies differ: $orig_energy vs $opt_energy)"
fi
echo ""

# ==================== Warmup ====================
echo "--- Warmup (1 run each) ---"
$BIN_ORIGINAL $LULESH_ARGS > /dev/null 2>&1 || true
$BIN_OPTIMIZED $LULESH_ARGS > /dev/null 2>&1 || true
echo "  Done"
echo ""

# ==================== Consecutive measurement ====================
run_batch() {
    local bin="$1"
    local label="$2"
    local n="$3"
    local times=()
    local sum=0

    for ((r=1; r<=n; r++)); do
        local tmpfile="/tmp/lulesh_${label}_${r}.out"
        "$bin" $LULESH_ARGS > "$tmpfile" 2>&1 || true
        local t
        t=$(grep "Hourglass kernel time" "$tmpfile" | awk '{print $4}')
        if [ -z "$t" ]; then
            t=$(grep "Elapsed time" "$tmpfile" | awk '{print $4}')
        fi
        if [ -n "$t" ]; then
            times+=("$t")
            sum=$(awk "BEGIN { printf \"%.6f\", $sum + $t }")
            printf "  [%s] run %d/%d: %ss\n" "$label" "$r" "$n" "$t"
        else
            printf "  [%s] run %d/%d: FAILED to parse output\n" "$label" "$r" "$n"
        fi
        rm -f "$tmpfile"
    done

    if [ ${#times[@]} -gt 0 ]; then
        local avg=$(awk "BEGIN { printf \"%.6f\", $sum / ${#times[@]} }")
        local mn="${times[0]}"
        for t in "${times[@]}"; do
            mn=$(awk "BEGIN { print ($t < $mn) ? $t : $mn }")
        done
        printf "  [%s] mean=%ss  min=%ss  (%d runs)\n" "$label" "$avg" "$mn" "${#times[@]}"
        eval "${label^^}_MEAN=$avg"
        eval "${label^^}_MIN=$mn"
        eval "${label^^}_N=${#times[@]}"
    fi
}

echo "--- Original: $PAIRS consecutive runs ---"
run_batch "$BIN_ORIGINAL" "original" "$PAIRS"
echo ""
echo "--- Optimized: $PAIRS consecutive runs ---"
run_batch "$BIN_OPTIMIZED" "optimized" "$PAIRS"
echo ""

# ==================== Results ====================
echo "============================================"
echo " Results (Hourglass kernel time)"
echo "============================================"

if [ -n "$ORIGINAL_MEAN" ] && [ -n "$OPTIMIZED_MEAN" ]; then
    speedup_mean=$(awk "BEGIN { printf \"%.2f\", $ORIGINAL_MEAN / $OPTIMIZED_MEAN }")
    speedup_min=$(awk "BEGIN { printf \"%.2f\", $ORIGINAL_MIN / $OPTIMIZED_MIN }")

    printf "%-12s %12s %12s %10s\n" "" "Original" "Optimized" "Speedup"
    printf "%-12s %12s %12s %10s\n" "----------" "----------" "----------" "--------"
    printf "%-12s %10ss %10ss %9sx\n" "mean" "$ORIGINAL_MEAN" "$OPTIMIZED_MEAN" "$speedup_mean"
    printf "%-12s %10ss %10ss %9sx\n" "min" "$ORIGINAL_MIN" "$OPTIMIZED_MIN" "$speedup_min"
    echo ""
    echo "  $PAIRS runs each, -s $SIZE -i $ITERS"
else
    echo "Could not compute speedup (runs failed)"
fi
