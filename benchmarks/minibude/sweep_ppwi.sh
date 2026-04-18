#!/bin/bash
# Sweep PPWI × deck for original vs optimized miniBUDE.
# Finds the configuration with the best optimization speedup.
#
# Usage:
#   ./sweep_ppwi.sh --docker leo-minibude-amd          # Full sweep on AMD
#   ./sweep_ppwi.sh --docker leo-minibude-intel         # Full sweep on Intel
#   ./sweep_ppwi.sh --docker leo-minibude-nvidia        # Full sweep on NVIDIA
#   ./sweep_ppwi.sh --docker leo-minibude-amd --deck bm2_long  # Single deck
#   ./sweep_ppwi.sh --docker leo-minibude-amd --ppwi 1,2,4     # Subset of PPWI
#
# Output: CSV table + best speedup summary

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DOCKER_IMAGE=""
DECKS="bm1 bm2 bm2_long"
PPWI_VALUES="all"
WGSIZE=256
ITERATIONS=4
VENDOR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)    DOCKER_IMAGE="$2"; shift 2 ;;
        --deck)      DECKS="$2"; shift 2 ;;
        --ppwi)      PPWI_VALUES="$2"; shift 2 ;;
        -w|--wgsize) WGSIZE="$2"; shift 2 ;;
        -i|--iter)   ITERATIONS="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Detect vendor from image name
if [[ "$DOCKER_IMAGE" == *intel* ]]; then VENDOR="intel"
elif [[ "$DOCKER_IMAGE" == *amd* ]]; then VENDOR="amd"
elif [[ "$DOCKER_IMAGE" == *nvidia* ]]; then VENDOR="nvidia"
fi

if [ -z "$DOCKER_IMAGE" ] || [ -z "$VENDOR" ]; then
    echo "Usage: $0 --docker <leo-minibude-{amd,nvidia,intel}>"
    exit 1
fi

# Select binary names per vendor
case "$VENDOR" in
    amd)    BIN_NAME="hip-bude" ;;
    nvidia) BIN_NAME="cuda-bude" ;;
    intel)  BIN_NAME="sycl-bude" ;;
esac

# Docker device flags
DOCKER_DEVICE_FLAGS=""
case "$VENDOR" in
    nvidia) DOCKER_DEVICE_FLAGS="--gpus all -e CUDA_VISIBLE_DEVICES=0" ;;
    intel)  DOCKER_DEVICE_FLAGS="--device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0" ;;
    amd)    DOCKER_DEVICE_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video -e ROCR_VISIBLE_DEVICES=0" ;;
esac

BUILD_VOLUME="minibude-build-${VENDOR}"

echo "============================================"
echo " MiniBUDE PPWI Sweep: $VENDOR"
echo " Decks:      $DECKS"
echo " PPWI:       $PPWI_VALUES"
echo " Wgsize:     $WGSIZE"
echo " Iterations: $ITERATIONS"
echo " Image:      $DOCKER_IMAGE"
echo "============================================"
echo ""

# Build the in-container script
CONTAINER_SCRIPT=$(cat <<'INNER_EOF'
#!/bin/bash
set -e

BIN_ORIGINAL="/opt/minibude/bin/BIN_NAME_PLACEHOLDER"
BIN_OPTIMIZED="/opt/build-optimized/BIN_NAME_PLACEHOLDER"

if [ ! -x "$BIN_OPTIMIZED" ]; then
    echo "Building optimized version..."
    mkdir -p /opt/build-optimized && cd /opt/build-optimized
    if [ ! -f CMakeCache.txt ]; then
        VENDOR_PLACEHOLDER_CMAKE
    fi
    make -j$(nproc) > /dev/null 2>&1
    echo "Build OK"
    echo ""
fi

DECKS="DECKS_PLACEHOLDER"
PPWI_VALUES="PPWI_PLACEHOLDER"
WGSIZE=WGSIZE_PLACEHOLDER
ITERATIONS=ITER_PLACEHOLDER

# Header
echo "deck,ppwi,original_avg_ms,original_min_ms,optimized_avg_ms,optimized_min_ms,speedup_avg,speedup_min"

for deck in $DECKS; do
    DECK_PATH="/opt/miniBUDE/data/$deck"
    if [ ! -d "$DECK_PATH" ]; then
        echo "# SKIP: $deck not found" >&2
        continue
    fi

    # Determine PPWI list
    if [ "$PPWI_VALUES" = "all" ]; then
        PPWI_LIST="1 2 4 8 16 32 64 128"
    else
        PPWI_LIST=$(echo "$PPWI_VALUES" | tr ',' ' ')
    fi

    for ppwi in $PPWI_LIST; do
        # Run original
        ORIG_OUT=$("$BIN_ORIGINAL" --deck "$DECK_PATH" -w "$WGSIZE" -p "$ppwi" -i "$ITERATIONS" 2>&1) || true
        ORIG_AVG=$(echo "$ORIG_OUT" | grep -oP 'avg_ms:\s*\K[0-9.]+' | tail -1)
        ORIG_MIN=$(echo "$ORIG_OUT" | grep -oP 'min_ms:\s*\K[0-9.]+' | tail -1)

        # Run optimized
        OPT_OUT=$("$BIN_OPTIMIZED" --deck "$DECK_PATH" -w "$WGSIZE" -p "$ppwi" -i "$ITERATIONS" 2>&1) || true
        OPT_AVG=$(echo "$OPT_OUT" | grep -oP 'avg_ms:\s*\K[0-9.]+' | tail -1)
        OPT_MIN=$(echo "$OPT_OUT" | grep -oP 'min_ms:\s*\K[0-9.]+' | tail -1)

        if [ -n "$ORIG_AVG" ] && [ -n "$OPT_AVG" ]; then
            SPEEDUP_AVG=$(awk "BEGIN { printf \"%.2f\", $ORIG_AVG / $OPT_AVG }")
            SPEEDUP_MIN=$(awk "BEGIN { printf \"%.2f\", $ORIG_MIN / $OPT_MIN }")
            echo "$deck,$ppwi,$ORIG_AVG,$ORIG_MIN,$OPT_AVG,$OPT_MIN,$SPEEDUP_AVG,$SPEEDUP_MIN"
        else
            echo "$deck,$ppwi,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL"
        fi
    done
done
INNER_EOF
)

# Substitute placeholders
CONTAINER_SCRIPT="${CONTAINER_SCRIPT//BIN_NAME_PLACEHOLDER/$BIN_NAME}"
CONTAINER_SCRIPT="${CONTAINER_SCRIPT//DECKS_PLACEHOLDER/$DECKS}"
CONTAINER_SCRIPT="${CONTAINER_SCRIPT//PPWI_PLACEHOLDER/$PPWI_VALUES}"
CONTAINER_SCRIPT="${CONTAINER_SCRIPT//WGSIZE_PLACEHOLDER/$WGSIZE}"
CONTAINER_SCRIPT="${CONTAINER_SCRIPT//ITER_PLACEHOLDER/$ITERATIONS}"

# Vendor-specific cmake command
case "$VENDOR" in
    amd)
        CMAKE_CMD='cmake /opt/minibude-optimized -DCMAKE_BUILD_TYPE=Release -DMODEL=hip -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_CXX_FLAGS="--offload-arch=gfx942" -DCXX_EXTRA_FLAGS="-g -fno-omit-frame-pointer" > /dev/null 2>&1'
        ;;
    nvidia)
        CMAKE_CMD='cmake /opt/minibude-optimized -DCMAKE_BUILD_TYPE=Release -DMODEL=cuda -DCMAKE_CUDA_FLAGS="-lineinfo" -DCXX_EXTRA_FLAGS="-g -fno-omit-frame-pointer" > /dev/null 2>&1'
        ;;
    intel)
        CMAKE_CMD='cmake /opt/minibude-optimized -DCMAKE_BUILD_TYPE=Release -DMODEL=sycl -DCMAKE_CXX_COMPILER=icpx -DCXX_EXTRA_FLAGS="-g -fno-omit-frame-pointer" > /dev/null 2>&1'
        ;;
esac
CONTAINER_SCRIPT="${CONTAINER_SCRIPT//VENDOR_PLACEHOLDER_CMAKE/$CMAKE_CMD}"

# Write temp script
TMPSCRIPT=$(mktemp /tmp/sweep-ppwi-XXXXXX.sh)
echo "$CONTAINER_SCRIPT" > "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
trap "rm -f $TMPSCRIPT" EXIT

# Run in Docker
CSV_OUTPUT=$(docker run --rm \
    --shm-size=4g \
    $DOCKER_DEVICE_FLAGS \
    -v "$SCRIPT_DIR/optimized:/opt/minibude-optimized:ro" \
    -v "${BUILD_VOLUME}:/opt/build-optimized" \
    -v "$TMPSCRIPT:/opt/sweep.sh:ro" \
    --entrypoint bash \
    "$DOCKER_IMAGE" \
    /opt/sweep.sh 2>&1)

echo "$CSV_OUTPUT"

# Print summary table
echo ""
echo "============================================"
echo " Summary Table"
echo "============================================"
printf "%-10s %5s %12s %12s %10s\n" "Deck" "PPWI" "Orig(ms)" "Opt(ms)" "Speedup"
printf "%-10s %5s %12s %12s %10s\n" "--------" "----" "----------" "----------" "--------"

BEST_SPEEDUP="0"
BEST_CONFIG=""

echo "$CSV_OUTPUT" | grep -v '^#' | grep -v '^deck,' | while IFS=',' read -r deck ppwi orig_avg orig_min opt_avg opt_min spd_avg spd_min; do
    [ -z "$deck" ] && continue
    [[ "$deck" == *"FAIL"* ]] && continue
    [[ "$orig_avg" == "FAIL" ]] && continue
    printf "%-10s %5s %12s %12s %9sx\n" "$deck" "$ppwi" "$orig_avg" "$opt_avg" "$spd_avg"
done

# Find best
BEST=$(echo "$CSV_OUTPUT" | grep -v '^#' | grep -v '^deck,' | grep -v 'FAIL' | \
    awk -F',' '{ if ($7+0 > max+0) { max=$7; line=$0 } } END { print line }')

if [ -n "$BEST" ]; then
    echo ""
    IFS=',' read -r deck ppwi orig_avg orig_min opt_avg opt_min spd_avg spd_min <<< "$BEST"
    echo "Best speedup: ${spd_avg}x at deck=$deck, PPWI=$ppwi (orig=${orig_avg}ms, opt=${opt_avg}ms)"
fi
