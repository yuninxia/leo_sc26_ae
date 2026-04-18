#!/bin/bash
# Run the full HPCToolkit + Leo evaluation pipeline for each RAJAPerf kernel separately.
#
# Creates per-kernel directories with interpretable names:
#   results/per-kernel/<KERNEL_NAME>/<vendor>/
#     hpctoolkit-raja-perf.exe-measurements/
#     hpctoolkit-raja-perf.exe-database/
#
# Usage:
#   ./run_per_kernel.sh <vendor>                     # Run all 29 shared kernels (Apps+Polybench)
#   ./run_per_kernel.sh <vendor> --kernels Basic_DAXPY Apps_EDGE3D  # Run specific kernels
#   ./run_per_kernel.sh all                          # Run all vendors sequentially
#
# Examples:
#   ./run_per_kernel.sh nvidia
#   ./run_per_kernel.sh amd --kernels Basic_DAXPY Polybench_GEMM Stream_TRIAD
#   ./run_per_kernel.sh all --kernels Basic_DAXPY

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$LEO_ROOT/results/per-kernel"

# 20 kernels shared across all 3 vendors (NVIDIA, AMD, Intel)
# Categories: Apps (16, real HPC applications) + Polybench (4)
# Excludes Basic (trivial), Stream (bandwidth-only), Lcals, Algorithm (only 1 shared: REDUCE_SUM).
# Excludes 9 Polybench kernels with insufficient PC samples (0 stall cycles on AMD MI300A).
# See docs/RAJAPERF_SHARED_KERNELS.md for the full 60-kernel list.
SHARED_KERNELS=(
    Apps_CONVECTION3DPA
    Apps_DEL_DOT_VEC_2D
    Apps_DIFFUSION3DPA
    Apps_EDGE3D
    Apps_ENERGY
    Apps_FIR
    Apps_LTIMES
    Apps_LTIMES_NOVIEW
    Apps_MASS3DEA
    Apps_MASS3DPA
    Apps_MASS3DPA_ATOMIC
    Apps_MASSVEC3DPA
    Apps_MATVEC_3D_STENCIL
    Apps_PRESSURE
    Apps_VOL3D
    Apps_ZONAL_ACCUMULATION_3D
    Polybench_2MM
    Polybench_3MM
    Polybench_FLOYD_WARSHALL
    Polybench_GEMM
)

# ============================================
# Parse arguments
# ============================================
VENDOR="${1:-}"
shift || true

SELECTED_KERNELS=()
SKIP_LEO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernels|-k)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_KERNELS+=("$1")
                shift
            done
            ;;
        --skip-leo)
            SKIP_LEO=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$VENDOR" ]]; then
    echo "Usage: $0 <amd|nvidia|nvidia-arm|intel|all> [--kernels K1 K2 ...]"
    echo ""
    echo "Runs HPCToolkit + Leo for each RAJAPerf kernel separately."
    echo ""
    echo "Output structure:"
    echo "  results/per-kernel/<KERNEL_NAME>/<vendor>/"
    echo "    hpctoolkit-raja-perf.exe-measurements/"
    echo "    hpctoolkit-raja-perf.exe-database/"
    echo ""
    echo "Options:"
    echo "  --kernels, -k <names>   Only run these kernels (default: all 29 shared)"
    echo "  --skip-leo              Skip Leo analysis (profiling only)"
    echo ""
    echo "Examples:"
    echo "  $0 nvidia                          # All 30 kernels on NVIDIA"
    echo "  $0 amd --kernels Basic_DAXPY       # Single kernel on AMD"
    echo "  $0 all --kernels Basic_DAXPY       # Single kernel on all vendors"
    exit 1
fi

# Use all shared kernels if none specified
if [[ ${#SELECTED_KERNELS[@]} -eq 0 ]]; then
    SELECTED_KERNELS=("${SHARED_KERNELS[@]}")
fi

# Determine which vendors to run
if [[ "$VENDOR" == "all" ]]; then
    VENDORS=(nvidia-arm amd intel)
else
    VENDORS=("$VENDOR")
fi

# ============================================
# Run evaluation — one parallel stream per vendor
# ============================================
TOTAL=$((${#SELECTED_KERNELS[@]} * ${#VENDORS[@]}))

echo "============================================"
echo " Per-Kernel RAJAPerf Evaluation"
echo "============================================"
echo "Vendors:  ${VENDORS[*]}"
echo "Kernels:  ${#SELECTED_KERNELS[@]}"
echo "Total:    $TOTAL runs"
echo "Output:   $RESULTS_BASE/"
echo ""
echo "Each vendor runs on a separate machine in parallel."
echo ""

# Function: run all kernels for one vendor (sequential within vendor)
run_vendor() {
    local V="$1"
    local LOGFILE="$RESULTS_BASE/.log-${V}.txt"
    local DONE=0
    local SKIP=0
    local FAIL=0

    echo "[${V}] Starting ${#SELECTED_KERNELS[@]} kernels..." | tee "$LOGFILE"

    for KERNEL in "${SELECTED_KERNELS[@]}"; do
        KERNEL_DIR="$RESULTS_BASE/$KERNEL/$V"

        # Skip if already has results
        if compgen -G "$KERNEL_DIR/hpctoolkit-*-database" > /dev/null 2>&1; then
            echo "[${V}] SKIP $KERNEL (exists)" >> "$LOGFILE"
            SKIP=$((SKIP + 1))
            continue
        fi

        echo "[${V}] Running $KERNEL ..." | tee -a "$LOGFILE"

        SKIP_LEO_FLAG=""
        if [ "$SKIP_LEO" = "true" ]; then
            SKIP_LEO_FLAG="--skip-leo"
        fi

        if "$SCRIPT_DIR/run_evaluation.sh" "$V" \
            --kernels "$KERNEL" \
            --npasses ${NPASSES:-100} \
            --output-dir "$KERNEL_DIR" \
            $SKIP_LEO_FLAG >> "$LOGFILE" 2>&1; then
            DONE=$((DONE + 1))
            echo "[${V}] OK $KERNEL" | tee -a "$LOGFILE"
        else
            FAIL=$((FAIL + 1))
            echo "[${V}] FAILED $KERNEL" | tee -a "$LOGFILE"
        fi
    done

    echo "[${V}] Finished: $DONE done, $SKIP skipped, $FAIL failed" | tee -a "$LOGFILE"
}

# Launch all vendors in parallel
PIDS=()
for V in "${VENDORS[@]}"; do
    run_vendor "$V" &
    PIDS+=($!)
    echo "Launched $V (PID ${PIDS[-1]})"
done

echo ""
echo "Waiting for all vendors to finish..."
echo "  Logs: $RESULTS_BASE/.log-<vendor>.txt"
echo ""

# Wait for all and collect exit codes
FAILURES=0
for i in "${!VENDORS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "${VENDORS[$i]}: DONE"
    else
        echo "${VENDORS[$i]}: FAILED"
        FAILURES=$((FAILURES + 1))
    fi
done

# ============================================
# Summary
# ============================================
echo ""
echo "============================================"
echo " Summary"
echo "============================================"
for V in "${VENDORS[@]}"; do
    if [ -f "$RESULTS_BASE/.log-${V}.txt" ]; then
        tail -1 "$RESULTS_BASE/.log-${V}.txt"
    fi
done

echo ""
echo "Results in: $RESULTS_BASE/"
echo ""
echo "Directory structure:"
echo "  $RESULTS_BASE/"
for KERNEL in "${SELECTED_KERNELS[@]:0:3}"; do
    echo "    $KERNEL/"
    for V in "${VENDORS[@]}"; do
        if compgen -G "$RESULTS_BASE/$KERNEL/$V/hpctoolkit-*-database" > /dev/null 2>&1; then
            echo "      $V/  (done)"
        else
            echo "      $V/"
        fi
    done
done
echo "    ..."
