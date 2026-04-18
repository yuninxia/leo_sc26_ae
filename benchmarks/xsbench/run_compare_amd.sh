#!/bin/bash
# Compare original (upstream) vs optimized (fork) XSBench on AMD MI300A.
#
# Runs inside Docker container (leo-xsbench-amd) which already has the
# upstream XSBench pre-built. Only the optimized fork needs to be built.
#
# Usage:
#   ./run_compare_amd.sh                          # Build opt, compare (3 runs)
#   ./run_compare_amd.sh --skip-build             # Reuse optimized build
#   ./run_compare_amd.sh --lookups 170000000      # More lookups for stable timing
#   ./run_compare_amd.sh --runs 5                 # Multiple runs
#   ./run_compare_amd.sh --device 1               # Use GPU device 1
#   ./run_compare_amd.sh --diff                   # Show source diff

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="leo-xsbench-amd"

# Defaults
SKIP_BUILD=false
SHOW_DIFF=false
SIZE="large"
GRID="hash"
METHOD="event"
LOOKUPS=""
RUNS=3
GPU_DEVICE=0

# ==================== Argument parsing ====================
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --vendor)      shift 2 ;;  # ignored, always amd
        --skip-build)  SKIP_BUILD=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --diff)        SHOW_DIFF=true; shift ;;
        --size)        SIZE="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --grid|-G)     GRID="$2"; PASSTHROUGH_ARGS+=("--grid" "$2"); shift 2 ;;
        --method|-m)   METHOD="$2"; PASSTHROUGH_ARGS+=("--method" "$2"); shift 2 ;;
        --lookups|-l)  LOOKUPS="$2"; PASSTHROUGH_ARGS+=("--lookups" "$2"); shift 2 ;;
        --runs)        RUNS="$2"; PASSTHROUGH_ARGS+=("--runs" "$2"); shift 2 ;;
        --device)      GPU_DEVICE="$2"; PASSTHROUGH_ARGS+=("--device" "$2"); shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==================== Show diff (runs locally, no Docker needed) ====================
if [ "$SHOW_DIFF" = true ]; then
    echo "Source differences between original/ and optimized/ (HIP):"
    echo ""
    diff -ru "$SCRIPT_DIR/original/hip/" "$SCRIPT_DIR/optimized/hip/" 2>/dev/null || true
    exit 0
fi

# ==================== Docker re-launch ====================
if [ ! -f /.dockerenv ]; then
    echo "Launching inside Docker: $DOCKER_IMAGE (device=$GPU_DEVICE)"
    echo ""

    exec docker run --rm \
        --device=/dev/kfd --device=/dev/dri --group-add video \
        -e ROCR_VISIBLE_DEVICES=$GPU_DEVICE \
        -v "$SCRIPT_DIR":/opt/xsbench-compare:ro \
        --entrypoint /bin/bash \
        "$DOCKER_IMAGE" \
        -c "
            cp -r /opt/xsbench-compare/optimized /opt/xsbench-optimized
            cp -r /opt/xsbench-compare/original /opt/xsbench-original
            cp /opt/xsbench-compare/run_compare_amd.sh /opt/run_compare_amd.sh
            chmod +x /opt/run_compare_amd.sh
            cd /opt
            bash /opt/run_compare_amd.sh --vendor amd ${PASSTHROUGH_ARGS[*]}
        "
fi

# ==================== Inside Docker ====================
SRC_ORIGINAL="/opt/xsbench-original/hip"
SRC_OPTIMIZED="/opt/xsbench-optimized/hip"

# ==================== Build both versions ====================
if [ "$SKIP_BUILD" = false ]; then
    echo "============================================"
    echo " Building original + optimized XSBench (AMD HIP)"
    echo "============================================"
    echo ""

    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    GPU_TARGET="${GPU_TARGET:-gfx942}"
    MAKE_CMD="make COMPILER=amd OPTIMIZE=yes CFLAGS=\"-std=c++14 -O3 --offload-arch=${GPU_TARGET} -g -fno-omit-frame-pointer -include cstring\""

    cd "$SRC_ORIGINAL"
    echo "  [original] make clean && make ..."
    make clean > /dev/null 2>&1 || true
    if eval "$MAKE_CMD" > make.log 2>&1; then
        echo "  [original] OK"
    else
        echo "  [original] FAILED (see make.log)"
        tail -20 make.log
        exit 1
    fi

    cd "$SRC_OPTIMIZED"
    echo "  [optimized] make clean && make ..."
    make clean > /dev/null 2>&1 || true
    if eval "$MAKE_CMD" > make.log 2>&1; then
        echo "  [optimized] OK"
    else
        echo "  [optimized] FAILED (see make.log)"
        tail -20 make.log
        exit 1
    fi
    echo ""
fi

BIN_ORIGINAL="$SRC_ORIGINAL/XSBench"
BIN_OPTIMIZED="$SRC_OPTIMIZED/XSBench"

if [ ! -x "$BIN_ORIGINAL" ]; then
    echo "ERROR: Original binary not found: $BIN_ORIGINAL"
    echo "       Run without --skip-build to build first."
    exit 1
fi

if [ ! -x "$BIN_OPTIMIZED" ]; then
    echo "ERROR: Optimized binary not found: $BIN_OPTIMIZED"
    echo "       Run without --skip-build to build first."
    exit 1
fi

# ==================== Run and compare ====================
XSBENCH_ARGS="-s $SIZE -m $METHOD -G $GRID"
if [ -n "$LOOKUPS" ]; then
    XSBENCH_ARGS="$XSBENCH_ARGS -l $LOOKUPS"
fi

echo "============================================"
echo " XSBench Original vs Optimized (AMD)"
echo " Args:      $XSBENCH_ARGS"
echo " Runs:      $RUNS"
echo " Original:  $BIN_ORIGINAL"
echo " Optimized: $BIN_OPTIMIZED"
echo "============================================"
echo ""

run_xsbench() {
    local bin="$1"
    local label="$2"
    local runs="$3"

    local total_runtime=0
    local runtimes=()
    local checksum=""

    for ((r=1; r<=runs; r++)); do
        # Run binary, write output to temp file to avoid subshell issues with HIP
        local tmpfile="/tmp/xsbench_${label}_${r}.out"
        "$bin" $XSBENCH_ARGS > "$tmpfile" 2>&1 || true

        local runtime
        # Prefer kernel-only time if available, else fall back to end-to-end Runtime
        runtime=$(grep -oP 'Kernel time: \K[0-9.]+' "$tmpfile" | tail -1 || echo "")
        if [ -z "$runtime" ]; then
            runtime=$(grep -oP 'Runtime:\s+\K[0-9.]+' "$tmpfile" | tail -1 || echo "")
        fi
        checksum=$(grep -oP 'Verification checksum: \K\d+' "$tmpfile" | tail -1 || echo "")
        local valid
        valid=$(grep -oP 'Verification checksum: \d+ \(\K\w+' "$tmpfile" || echo "")

        if [ -n "$runtime" ]; then
            runtimes+=("$runtime")
            total_runtime=$(awk "BEGIN { printf \"%.3f\", $total_runtime + $runtime }")
            echo "  [$label] run $r/$runs: ${runtime}s (checksum: $checksum $valid)"
        else
            echo "  [$label] run $r/$runs: FAILED to parse output"
            tail -5 "$tmpfile"
        fi
        rm -f "$tmpfile"
    done

    if [ ${#runtimes[@]} -gt 0 ]; then
        local avg_runtime
        avg_runtime=$(awk "BEGIN { printf \"%.3f\", $total_runtime / ${#runtimes[@]} }")
        # Find min
        local min_runtime="${runtimes[0]}"
        for rt in "${runtimes[@]}"; do
            min_runtime=$(awk "BEGIN { print ($rt < $min_runtime) ? $rt : $min_runtime }")
        done
        echo "  [$label] avg=${avg_runtime}s  min=${min_runtime}s  (${#runtimes[@]} runs)"

        # Return via global vars
        eval "${label^^}_AVG=$avg_runtime"
        eval "${label^^}_MIN=$min_runtime"
        eval "${label^^}_CHECKSUM=$checksum"
    fi
}

run_xsbench "$BIN_ORIGINAL" "original" "$RUNS"
echo ""
run_xsbench "$BIN_OPTIMIZED" "optimized" "$RUNS"
echo ""

# ==================== Results ====================
echo "============================================"
echo " Results"
echo "============================================"

if [ -n "$ORIGINAL_AVG" ] && [ -n "$OPTIMIZED_AVG" ]; then
    SPEEDUP_AVG=$(awk "BEGIN { printf \"%.2f\", $ORIGINAL_AVG / $OPTIMIZED_AVG }")
    SPEEDUP_MIN=$(awk "BEGIN { printf \"%.2f\", $ORIGINAL_MIN / $OPTIMIZED_MIN }")
    printf "%-12s %12s %12s %10s\n" "" "Original" "Optimized" "Speedup"
    printf "%-12s %12s %12s %10s\n" "----------" "----------" "----------" "--------"
    printf "%-12s %10ss %10ss %9sx\n" "avg" "$ORIGINAL_AVG" "$OPTIMIZED_AVG" "$SPEEDUP_AVG"
    printf "%-12s %10ss %10ss %9sx\n" "min" "$ORIGINAL_MIN" "$OPTIMIZED_MIN" "$SPEEDUP_MIN"
    echo ""
    echo "  Checksums: original=$ORIGINAL_CHECKSUM optimized=$OPTIMIZED_CHECKSUM"
    if [ "$ORIGINAL_CHECKSUM" = "$OPTIMIZED_CHECKSUM" ]; then
        echo "  Verification: PASS (checksums match)"
    else
        echo "  Verification: FAIL (checksums differ!)"
    fi
else
    echo "Could not compute speedup (one or both runs failed)"
fi

echo ""
echo "  Source diff:"
diff -rq "/opt/xsbench-original/hip/" "/opt/xsbench-optimized/hip/" 2>/dev/null | \
    sed 's|^|    |' || echo "    (no differences)"
