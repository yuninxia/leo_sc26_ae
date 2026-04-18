#!/bin/bash
# Root-cause validation: profile ORIGINAL and OPTIMIZED RAJAPerf kernels,
# run Leo on both, and compare before/after stall patterns.
#
# Usage:
#   ./run_validation.sh amd --kernels Apps_LTIMES
#   ./run_validation.sh nvidia --kernels Apps_LTIMES Apps_MASS3DEA
#   ./run_validation.sh intel --kernels Apps_LTIMES
#   ./run_validation.sh all --kernels Apps_LTIMES           # All 3 vendors in parallel
#
# Output:
#   results/validation/<KERNEL>/<vendor>/
#     original/   — hpctoolkit measurements + Leo output
#     optimized/  — hpctoolkit measurements + Leo output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$LEO_ROOT/results/validation"

# ============================================
# Vendor configuration
# ============================================
declare -A MACHINES DOCKER_GPU_FLAGS GPU_EVENTS GPU_ARCH VARIANTS GPU_VISIBLE_ENV GPUCFG_FLAGS IMAGES

MACHINES[amd]="odyssey"
MACHINES[nvidia]="gilgamesh"
MACHINES[intel]="headroom"

DOCKER_GPU_FLAGS[amd]="--device=/dev/kfd --device=/dev/dri"
DOCKER_GPU_FLAGS[nvidia]="--gpus all"
DOCKER_GPU_FLAGS[intel]="--device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path"

GPU_EVENTS[amd]="gpu=rocm,pc=hw@25"
GPU_EVENTS[nvidia]="gpu=cuda,pc"
GPU_EVENTS[intel]="gpu=level0,pc"

GPU_ARCH[amd]="mi300"
GPU_ARCH[nvidia]="h100"
GPU_ARCH[intel]="pvc"

VARIANTS[amd]="Base_HIP"
VARIANTS[nvidia]="Base_CUDA"
VARIANTS[intel]="Base_SYCL"

GPU_VISIBLE_ENV[amd]="-e ROCR_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia]="-e CUDA_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[intel]="-e ZE_AFFINITY_MASK=0 -e ZE_ENABLE_TRACING_LAYER=1"

GPUCFG_FLAGS[amd]="--gpucfg yes"
GPUCFG_FLAGS[nvidia]="--gpucfg yes"
GPUCFG_FLAGS[intel]="--gpucfg yes"

IMAGES[amd]="leo-rajaperf-amd"
IMAGES[nvidia]="leo-rajaperf-nvidia"
IMAGES[intel]="leo-rajaperf-intel"

# Optimized source directory
OPT_SRC="$LEO_ROOT/benchmarks/rajaperf-h100/optimized/src/apps"

# ============================================
# Parse arguments
# ============================================
VENDOR="${1:-}"
shift || true

SELECTED_KERNELS=()
NPASSES=${NPASSES:-200}

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernels|-k)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_KERNELS+=("$1")
                shift
            done
            ;;
        --npasses)  NPASSES="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$VENDOR" || ${#SELECTED_KERNELS[@]} -eq 0 ]]; then
    echo "Usage: $0 <amd|nvidia|intel|all> --kernels <K1> [K2 ...]"
    echo ""
    echo "Profiles original + optimized RAJAPerf kernels and runs Leo on both."
    echo "Compares before/after stall patterns for root-cause validation."
    echo ""
    echo "Examples:"
    echo "  $0 amd --kernels Apps_LTIMES"
    echo "  $0 all --kernels Apps_LTIMES              # All 3 vendors in parallel"
    echo "  $0 all --kernels Apps_LTIMES Apps_MASS3DEA # Multiple kernels, all vendors"
    exit 1
fi

# Determine vendors to run
if [[ "$VENDOR" == "all" ]]; then
    ALL_VENDORS=(amd nvidia intel)
else
    ALL_VENDORS=("$VENDOR")
fi

# ============================================
# In-container script (shared across vendors)
# ============================================
INNER_SCRIPT=$(cat << 'INNEREOF'
#!/bin/bash
set -e

export LD_LIBRARY_PATH="/opt/dyninst/lib:/opt/dyninst/lib64:$LD_LIBRARY_PATH"
export ZE_ENABLE_TRACING_LAYER=1

KERNEL="$1"
VARIANT="$2"
EVENT="$3"
NPASSES="$4"
GPUCFG="$5"
ARCH="$6"

EXEC="/opt/rajaperf/bin/raja-perf.exe"
EXEC_NAME="raja-perf.exe"

run_profile_and_leo() {
    local LABEL="$1"
    local OUTDIR="$2"
    local BIN="$3"

    echo ""
    echo "--- $LABEL: profiling $KERNEL ---"

    local WORKDIR="/tmp/validation-${LABEL}"
    rm -rf "$WORKDIR"
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"

    echo "  hpcrun ..."
    hpcrun -e "$EVENT" "$BIN" \
        --variants "$VARIANT" \
        --checkrun 1 \
        --npasses "$NPASSES" \
        --kernels "$KERNEL" 2>&1 | tail -3

    MEAS_DIR=$(ls -d hpctoolkit-${EXEC_NAME}-measurements* 2>/dev/null | head -1)
    if [[ -z "$MEAS_DIR" ]]; then
        echo "  ERROR: no measurements found"
        return 1
    fi

    echo "  hpcstruct ..."
    hpcstruct $GPUCFG "$MEAS_DIR" 2>&1 | tail -2

    echo "  hpcprof ..."
    hpcprof "$MEAS_DIR" 2>&1 | tail -2

    # Copy measurements + database to output volume
    echo "  Copying results to output..."
    cp -r "$WORKDIR/hpctoolkit-"*"-measurements"* "$OUTDIR/" 2>/dev/null || echo "  WARNING: measurements copy failed"
    cp -r "$WORKDIR/hpctoolkit-"*"-database"* "$OUTDIR/" 2>/dev/null || echo "  WARNING: database copy failed"

    # Verify what was copied
    echo "  Output contents:"
    ls -d "$OUTDIR"/hpctoolkit-* 2>/dev/null | sed 's|^|    |' || echo "    (empty)"

    echo "  [$LABEL] done"
}

# Original
run_profile_and_leo "original" "/data/original" "$EXEC"

# Optimized
if [[ -x /opt/rajaperf-opt/bin/raja-perf.exe ]]; then
    run_profile_and_leo "optimized" "/data/optimized" "/opt/rajaperf-opt/bin/raja-perf.exe"
else
    echo "ERROR: optimized binary not found"
    exit 1
fi

echo ""
echo "=== PROFILING COMPLETE ==="
echo "Run validation_compare.py to analyze before/after stall cycles."
INNEREOF
)

# ============================================
# Run one vendor for all kernels
# ============================================
run_vendor() {
    local V="$1"
    local MACHINE="${MACHINES[$V]}"
    local EVENT="${GPU_EVENTS[$V]}"
    local ARCH="${GPU_ARCH[$V]}"
    local VARIANT="${VARIANTS[$V]}"
    local DOCKER_FLAGS="${DOCKER_GPU_FLAGS[$V]}"
    local VIS_ENV="${GPU_VISIBLE_ENV[$V]}"
    local GPUCFG="${GPUCFG_FLAGS[$V]}"
    local IMAGE="${IMAGES[$V]}"

    echo "[$V] Starting on $MACHINE (${#SELECTED_KERNELS[@]} kernels)..."

    for KERNEL in "${SELECTED_KERNELS[@]}"; do
        echo ""
        echo "[$V] ============================================"
        echo "[$V]  Validating: $KERNEL"
        echo "[$V] ============================================"

        KERNEL_DIR="$RESULTS_BASE/$KERNEL/$V"
        ORIG_DIR="$KERNEL_DIR/original"
        OPT_DIR="$KERNEL_DIR/optimized"
        mkdir -p "$ORIG_DIR" "$OPT_DIR"
        chmod 777 "$ORIG_DIR" "$OPT_DIR"

        KBASE="${KERNEL#Apps_}"

        # Write inner script to a path on the shared filesystem
        TMPSCRIPT="$KERNEL_DIR/.run_inner.sh"
        echo "$INNER_SCRIPT" > "$TMPSCRIPT"
        chmod +x "$TMPSCRIPT"

        # Docker command — mount the inner script from shared filesystem
        DOCKER_CMD="docker run --rm \
            $DOCKER_FLAGS \
            $VIS_ENV \
            -e PYTHONUNBUFFERED=1 \
            -v $ORIG_DIR:/data/original \
            -v $OPT_DIR:/data/optimized \
            -v $TMPSCRIPT:/opt/run.sh:ro \
            -v $OPT_SRC:/opt/rajaperf-opt-src:ro \
            --entrypoint bash \
            $IMAGE \
            -c '
                echo \"Rebuilding RAJAPerf with optimized $KERNEL sources...\"
                cp /opt/rajaperf-opt-src/${KBASE}*.cpp /opt/RAJAPerf/src/apps/ 2>/dev/null || true
                cp /opt/rajaperf-opt-src/${KBASE}*.hpp /opt/RAJAPerf/src/apps/ 2>/dev/null || true
                cd /opt/rajaperf-build
                make -j\$(nproc) 2>&1 | tail -5
                mkdir -p /opt/rajaperf-opt/bin
                cp bin/raja-perf.exe /opt/rajaperf-opt/bin/
                bash /opt/run.sh \"$KERNEL\" \"$VARIANT\" \"$EVENT\" \"$NPASSES\" \"$GPUCFG\" \"$ARCH\"
            '"

        # Run locally or via SSH
        if [[ "$MACHINE" == "$(hostname)" || "$MACHINE" == "odyssey" ]]; then
            eval "$DOCKER_CMD" 2>&1 | tee "$KERNEL_DIR/validation.log"
        else
            ssh "$MACHINE" "$DOCKER_CMD" 2>&1 | tee "$KERNEL_DIR/validation.log"
        fi

        rm -f "$TMPSCRIPT"

        echo "[$V] Results: $KERNEL_DIR/{original,optimized}/leo_output.txt"
    done

    echo ""
    echo "[$V] All kernels done."
}

# ============================================
# Launch vendors
# ============================================
echo "============================================"
echo " Root-Cause Validation"
echo "============================================"
echo "Vendors:   ${ALL_VENDORS[*]}"
echo "Kernels:   ${SELECTED_KERNELS[*]}"
echo "Passes:    $NPASSES"
echo "Output:    $RESULTS_BASE/"
echo ""

if [[ ${#ALL_VENDORS[@]} -eq 1 ]]; then
    # Single vendor: run directly
    run_vendor "${ALL_VENDORS[0]}"
else
    # Multiple vendors: run in parallel
    PIDS=()
    for V in "${ALL_VENDORS[@]}"; do
        LOGFILE="$RESULTS_BASE/.validation-log-${V}.txt"
        run_vendor "$V" > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        echo "Launched $V on ${MACHINES[$V]} (PID ${PIDS[-1]}, log: $LOGFILE)"
    done

    echo ""
    echo "Waiting for all vendors..."
    FAILURES=0
    for i in "${!ALL_VENDORS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "${ALL_VENDORS[$i]}: DONE"
        else
            echo "${ALL_VENDORS[$i]}: FAILED"
            FAILURES=$((FAILURES + 1))
        fi
    done

    echo ""
    echo "============================================"
    echo " Logs"
    echo "============================================"
    for V in "${ALL_VENDORS[@]}"; do
        echo "--- $V ---"
        grep -E "COMPARISON|BEFORE|AFTER|Stall|global_load|s_waitcnt|LDG|send|ds_read|SUPPORTED|Supported" \
            "$RESULTS_BASE/.validation-log-${V}.txt" 2>/dev/null | head -20
        echo ""
    done
fi

echo "============================================"
echo " Validation complete"
echo "============================================"
echo "Results in: $RESULTS_BASE/"
for KERNEL in "${SELECTED_KERNELS[@]}"; do
    for V in "${ALL_VENDORS[@]}"; do
        echo "  $KERNEL/$V/{original,optimized}/leo_output.txt"
    done
done
