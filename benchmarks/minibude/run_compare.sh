#!/bin/bash
# Compare original (upstream) vs optimized (fork) miniBUDE builds.
#
# Runs inside Docker container (leo-minibude-amd) which already has the
# upstream miniBUDE pre-built. Only the optimized fork needs to be built.
#
# Directory structure:
#   benchmarks/minibude/
#     original/            # upstream UoB-HPC/miniBUDE (for diff only)
#     optimized/           # fork with Leo optimizations applied
#     run_compare.sh       # this script
#
# Usage:
#   ./run_compare.sh --docker leo-minibude-amd                      # Build opt, compare
#   ./run_compare.sh --docker leo-minibude-amd --skip-build         # Reuse optimized build
#   ./run_compare.sh --docker leo-minibude-amd --deck bm1 -w 64     # Custom params
#   ./run_compare.sh --docker leo-minibude-amd --profile            # + HPCToolkit/Leo
#   ./run_compare.sh --diff                                         # Show source diff

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
VENDOR=""
SKIP_BUILD=false
PROFILE=false
SHOW_DIFF=false
DOCKER_IMAGE=""
DECK="bm2"
WGSIZE=256
PPWI=2
ITERATIONS=8
JOBS=$(nproc 2>/dev/null || echo 8)

# ==================== Argument parsing ====================
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --vendor)      VENDOR="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --profile)     PROFILE=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --diff)        SHOW_DIFF=true; shift ;;
        --deck)        DECK="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        -w|--wgsize)   WGSIZE="$2"; PASSTHROUGH_ARGS+=("--wgsize" "$2"); shift 2 ;;
        -p|--ppwi)     PPWI="$2"; PASSTHROUGH_ARGS+=("--ppwi" "$2"); shift 2 ;;
        -i|--iter)     ITERATIONS="$2"; PASSTHROUGH_ARGS+=("--iter" "$2"); shift 2 ;;
        --jobs|-j)     JOBS="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --docker)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "ERROR: --docker requires an image name (e.g., leo-minibude-amd)"
                exit 1
            fi
            DOCKER_IMAGE="$2"; shift 2
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==================== Show diff (runs locally, no Docker needed) ====================
if [ "$SHOW_DIFF" = true ]; then
    echo "Source differences between original/ and optimized/:"
    echo ""
    diff -ru "$SCRIPT_DIR/original/src/" "$SCRIPT_DIR/optimized/src/" 2>/dev/null || true
    exit 0
fi

# ==================== Vendor detection ====================
if [ -z "$VENDOR" ] && [ -n "$DOCKER_IMAGE" ]; then
    if [[ "$DOCKER_IMAGE" == *intel* ]]; then VENDOR="intel"
    elif [[ "$DOCKER_IMAGE" == *amd* ]]; then VENDOR="amd"
    elif [[ "$DOCKER_IMAGE" == *nvidia* ]]; then VENDOR="nvidia"
    fi
fi

if [ -z "$VENDOR" ]; then
    echo "ERROR: --docker <image> required (e.g., leo-minibude-amd)"
    echo ""
    echo "Usage:"
    echo "  ./run_compare.sh --docker leo-minibude-amd"
    echo "  ./run_compare.sh --docker leo-minibude-amd --deck bm1 -w 64"
    echo "  ./run_compare.sh --docker leo-minibude-amd --profile"
    echo "  ./run_compare.sh --diff"
    exit 1
fi

# ==================== Docker re-launch ====================
if [ -n "$DOCKER_IMAGE" ] && [ ! -f /.dockerenv ]; then
    echo "Launching inside Docker: $DOCKER_IMAGE"
    echo ""

    DOCKER_DEVICE_FLAGS=()
    case "$VENDOR" in
        nvidia) DOCKER_DEVICE_FLAGS+=(--gpus all) ;;
        intel)  DOCKER_DEVICE_FLAGS+=(--device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path) ;;
        amd)    DOCKER_DEVICE_FLAGS+=(--device=/dev/kfd --device=/dev/dri --group-add video) ;;
    esac

    # Persistent Docker volume for the CMake build directory
    BUILD_VOLUME="minibude-build-${VENDOR}"

    exec docker run --rm \
        "${DOCKER_DEVICE_FLAGS[@]}" \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e SYCL_CACHE_PERSISTENT=1 \
        -v /tmp/sycl-cache:/root/.cache \
        -v "$SCRIPT_DIR":/opt/minibude-compare:ro \
        -v "$LEO_ROOT/src":/opt/leo-src:ro \
        -v "$LEO_ROOT/scripts":/opt/leo-scripts:ro \
        -v "${BUILD_VOLUME}":/opt/build-optimized \
        --entrypoint /bin/bash \
        "$DOCKER_IMAGE" \
        -c "
            cp -r /opt/minibude-compare/optimized /opt/minibude-optimized
            cp -r /opt/minibude-compare/original /opt/minibude-original
            cp /opt/minibude-compare/run_compare.sh /opt/run_compare.sh
            chmod +x /opt/run_compare.sh

            if echo '${PASSTHROUGH_ARGS[*]}' | grep -q -- '--profile'; then
                cp -r /opt/leo-src/leo/* /opt/leo/src/leo/ 2>/dev/null || true
                cp /opt/leo-scripts/analyze_benchmark.py /opt/leo/scripts/ 2>/dev/null || true
            fi

            cd /opt
            bash /opt/run_compare.sh --vendor $VENDOR ${PASSTHROUGH_ARGS[*]}
        "
fi

# ==================== Inside Docker: paths ====================
BIN_ORIGINAL="/opt/minibude/bin/hip-bude"
case "$VENDOR" in
    nvidia) BIN_ORIGINAL="/opt/minibude/bin/cuda-bude" ;;
    intel)  BIN_ORIGINAL="/opt/minibude/bin/sycl-bude" ;;
esac

if [ ! -x "$BIN_ORIGINAL" ]; then
    echo "ERROR: Original binary not found: $BIN_ORIGINAL"
    exit 1
fi

SRC_OPTIMIZED="/opt/minibude-optimized"
BUILD_OPTIMIZED="/opt/build-optimized"

# ==================== Build optimized version ====================
if [ "$SKIP_BUILD" = false ]; then
    echo "============================================"
    echo " Building optimized miniBUDE ($VENDOR)"
    echo " Original: using container's pre-built binary"
    echo "============================================"
    echo ""

    mkdir -p "$BUILD_OPTIMIZED"
    cd "$BUILD_OPTIMIZED"

    if [ ! -f CMakeCache.txt ]; then
        echo "  [optimized] cmake ..."
        case "$VENDOR" in
            amd)
                ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
                GPU_TARGET="${GPU_TARGET:-gfx942}"
                cmake "$SRC_OPTIMIZED" \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_INSTALL_PREFIX=/opt/minibude-opt \
                    -DMODEL=hip \
                    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/hipcc" \
                    -DCMAKE_CXX_FLAGS="--offload-arch=${GPU_TARGET}" \
                    -DCXX_EXTRA_FLAGS="-g -fno-omit-frame-pointer" \
                    > cmake.log 2>&1
                ;;
            nvidia)
                cmake "$SRC_OPTIMIZED" \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DMODEL=cuda \
                    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
                    -DCUDA_ARCH=sm_90 \
                    -DCMAKE_CUDA_FLAGS="-lineinfo" \
                    -DCXX_EXTRA_FLAGS="-g -fno-omit-frame-pointer" \
                    > cmake.log 2>&1
                ;;
            intel)
                cmake "$SRC_OPTIMIZED" \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DMODEL=sycl \
                    -DSYCL_COMPILER=ONEAPI-ICPX \
                    -DCMAKE_CXX_COMPILER=icpx \
                    -DCXX_EXTRA_FLAGS="-g -fno-omit-frame-pointer" \
                    > cmake.log 2>&1
                ;;
        esac
    else
        echo "  [optimized] cmake already configured (reusing)"
    fi

    echo "  [optimized] make -j$JOBS ..."
    if make -j"$JOBS" > make.log 2>&1; then
        echo "  [optimized] OK"
    else
        echo "  [optimized] FAILED (see $BUILD_OPTIMIZED/make.log)"
        tail -20 "$BUILD_OPTIMIZED/make.log"
        exit 1
    fi
    echo ""
fi

case "$VENDOR" in
    amd)    BIN_OPTIMIZED="$BUILD_OPTIMIZED/hip-bude" ;;
    nvidia) BIN_OPTIMIZED="$BUILD_OPTIMIZED/cuda-bude" ;;
    intel)  BIN_OPTIMIZED="$BUILD_OPTIMIZED/sycl-bude" ;;
esac

if [ ! -x "$BIN_OPTIMIZED" ]; then
    echo "ERROR: Optimized binary not found: $BIN_OPTIMIZED"
    echo "       Run without --skip-build to build first."
    exit 1
fi

# ==================== Run and compare ====================
DECK_PATH="/opt/miniBUDE/data/$DECK"
BUDE_ARGS="--deck $DECK_PATH -w $WGSIZE -p $PPWI -i $ITERATIONS"

echo "============================================"
echo " miniBUDE Original vs Optimized ($VENDOR)"
echo " Deck:    $DECK (wgsize=$WGSIZE, ppwi=$PPWI)"
echo " Iters:   $ITERATIONS"
echo " Original: $BIN_ORIGINAL"
echo " Optimized: $BIN_OPTIMIZED"
echo "============================================"
echo ""

run_minibude() {
    local bin="$1"
    local label="$2"

    echo "  [$label] Running: $(basename $bin) $BUDE_ARGS"
    local out
    out=$("$bin" $BUDE_ARGS 2>&1) || true

    # Extract from "best: { min_ms: 1664.43, ... avg_ms: 1672.35, ... }"
    local best_line min_ms avg_ms
    best_line=$(echo "$out" | grep '^best:' || echo "")
    min_ms=$(echo "$best_line" | grep -oP 'min_ms:\s*\K[0-9.]+' || echo "")
    avg_ms=$(echo "$best_line" | grep -oP 'avg_ms:\s*\K[0-9.]+' || echo "")

    if [ -n "$min_ms" ]; then
        echo "  [$label] min=${min_ms}ms  avg=${avg_ms}ms"
    else
        echo "  [$label] FAILED to parse output"
        echo "$out" | tail -5
    fi

    # Return via global vars
    eval "${label^^}_MIN=$min_ms"
    eval "${label^^}_AVG=$avg_ms"
}

run_minibude "$BIN_ORIGINAL" "original"
echo ""
run_minibude "$BIN_OPTIMIZED" "optimized"
echo ""

# ==================== Results ====================
echo "============================================"
echo " Results"
echo "============================================"

if [ -n "$ORIGINAL_MIN" ] && [ -n "$OPTIMIZED_MIN" ]; then
    SPEEDUP_MIN=$(awk "BEGIN { printf \"%.2f\", $ORIGINAL_MIN / $OPTIMIZED_MIN }")
    SPEEDUP_AVG=$(awk "BEGIN { printf \"%.2f\", $ORIGINAL_AVG / $OPTIMIZED_AVG }")
    printf "%-12s %12s %12s %10s\n" "" "Original" "Optimized" "Speedup"
    printf "%-12s %12s %12s %10s\n" "----------" "----------" "----------" "--------"
    printf "%-12s %10sms %10sms %9sx\n" "min" "$ORIGINAL_MIN" "$OPTIMIZED_MIN" "$SPEEDUP_MIN"
    printf "%-12s %10sms %10sms %9sx\n" "avg" "$ORIGINAL_AVG" "$OPTIMIZED_AVG" "$SPEEDUP_AVG"
else
    echo "Could not compute speedup (one or both runs failed)"
fi

echo ""
echo "  Source diff:"
diff -rq "/opt/minibude-original/src/" "/opt/minibude-optimized/src/" 2>/dev/null | \
    sed 's|^|    |' || echo "    (no differences)"

# ==================== Profiling (optional) ====================
if [ "$PROFILE" = true ]; then
    if ! command -v hpcrun &>/dev/null; then
        echo ""
        echo "ERROR: hpcrun not found."
        exit 1
    fi

    declare -A HPCRUN_EVENTS LEO_ARCH
    HPCRUN_EVENTS[amd]="gpu=rocm,pc=hw@25"
    HPCRUN_EVENTS[nvidia]="gpu=cuda,pc"
    HPCRUN_EVENTS[intel]="gpu=level0,pc"
    LEO_ARCH[amd]="mi300"; LEO_ARCH[nvidia]="h100"; LEO_ARCH[intel]="pvc"

    EVENT="${HPCRUN_EVENTS[$VENDOR]}"
    ARCH="${LEO_ARCH[$VENDOR]}"
    PROFILE_DIR="/opt/profiles"
    mkdir -p "$PROFILE_DIR"

    echo ""
    echo "============================================"
    echo " HPCToolkit Profiling + Leo Analysis"
    echo "============================================"

    for label in original optimized; do
        if [ "$label" = "original" ]; then BIN="$BIN_ORIGINAL"; else BIN="$BIN_OPTIMIZED"; fi
        MEAS_DIR="$PROFILE_DIR/minibude-${label}-measurements"
        DB_DIR="$PROFILE_DIR/minibude-${label}-database"

        echo ""
        echo "--- $label ---"
        echo "  hpcrun..."
        rm -rf "$MEAS_DIR" "$DB_DIR"
        hpcrun -o "$MEAS_DIR" -e "$EVENT" "$BIN" $BUDE_ARGS 2>&1 | tail -1

        echo "  hpcstruct..."
        hpcstruct --gpucfg yes "$MEAS_DIR" 2>&1 | tail -1 || true

        echo "  hpcprof..."
        hpcprof -o "$DB_DIR" "$MEAS_DIR" 2>&1 | tail -1

        echo "  Leo analysis..."
        cd /opt/leo 2>/dev/null || true
        uv run python scripts/analyze_benchmark.py \
            "$MEAS_DIR" --arch "$ARCH" --top-n 1 2>&1 | \
            tee "$PROFILE_DIR/minibude-${label}-leo.txt"
    done
fi
