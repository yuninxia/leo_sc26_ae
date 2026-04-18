#!/bin/bash
# Run HPCToolkit profiling for RAJAPerf kernels on Aurora (Intel PVC) via Apptainer.
#
# Creates per-kernel directories:
#   results/per-kernel/<KERNEL_NAME>/intel/
#     hpctoolkit-raja-perf.exe-measurements/
#     hpctoolkit-raja-perf.exe-database/
#
# Usage:
#   ./run_per_kernel_aurora.sh                              # Run all 60 shared kernels
#   ./run_per_kernel_aurora.sh --kernels Basic_DAXPY        # Run one kernel
#   ./run_per_kernel_aurora.sh --sif /path/to/image.sif     # Custom SIF path
#
# Prerequisites:
#   module load apptainer
#   apptainer build --fakeroot leo-rajaperf-intel.sif docker://jssonxia/leo-rajaperf-intel:latest
#
# PBS submission example:
#   qsub -l select=1 -l walltime=04:00:00 -A <project> -q workq ./run_per_kernel_aurora.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="${RESULTS_BASE:-$SCRIPT_DIR/../../results/per-kernel}"

# Default SIF image location (can be overridden with --sif)
SIF="${SIF:-$SCRIPT_DIR/leo-rajaperf-intel.sif}"

# 60 kernels shared across all 3 vendors
SHARED_KERNELS=(
    Algorithm_REDUCE_SUM
    Apps_CONVECTION3DPA Apps_DEL_DOT_VEC_2D Apps_DIFFUSION3DPA Apps_EDGE3D
    Apps_ENERGY Apps_FIR Apps_LTIMES Apps_LTIMES_NOVIEW Apps_MASS3DEA
    Apps_MASS3DPA Apps_MASS3DPA_ATOMIC Apps_MASSVEC3DPA Apps_MATVEC_3D_STENCIL
    Apps_PRESSURE Apps_VOL3D Apps_ZONAL_ACCUMULATION_3D
    Basic_ARRAY_OF_PTRS Basic_COPY8 Basic_DAXPY Basic_EMPTY Basic_IF_QUAD
    Basic_INIT3 Basic_INIT_VIEW1D Basic_INIT_VIEW1D_OFFSET Basic_MAT_MAT_SHARED
    Basic_MULADDSUB Basic_NESTED_INIT Basic_PI_REDUCE Basic_REDUCE3_INT Basic_TRAP_INT
    Lcals_DIFF_PREDICT Lcals_EOS Lcals_FIRST_DIFF Lcals_FIRST_MIN Lcals_FIRST_SUM
    Lcals_GEN_LIN_RECUR Lcals_HYDRO_1D Lcals_HYDRO_2D Lcals_INT_PREDICT
    Lcals_PLANCKIAN Lcals_TRIDIAG_ELIM
    Polybench_2MM Polybench_3MM Polybench_ADI Polybench_ATAX Polybench_FDTD_2D
    Polybench_FLOYD_WARSHALL Polybench_GEMM Polybench_GEMVER Polybench_GESUMMV
    Polybench_HEAT_3D Polybench_JACOBI_1D Polybench_JACOBI_2D Polybench_MVT
    Stream_ADD Stream_COPY Stream_DOT Stream_MUL Stream_TRIAD
)

# ============================================
# Parse arguments
# ============================================
SELECTED_KERNELS=()
NPASSES=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernels|-k)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_KERNELS+=("$1")
                shift
            done
            ;;
        --sif)
            SIF="$2"; shift 2 ;;
        --npasses)
            NPASSES="$2"; shift 2 ;;
        --results)
            RESULTS_BASE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#SELECTED_KERNELS[@]} -eq 0 ]]; then
    SELECTED_KERNELS=("${SHARED_KERNELS[@]}")
fi

# Validate SIF exists
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: SIF image not found: $SIF"
    echo ""
    echo "Build it first:"
    echo "  module load apptainer"
    echo "  apptainer build --fakeroot leo-rajaperf-intel.sif docker://jssonxia/leo-rajaperf-intel:latest"
    exit 1
fi

# ============================================
# Run kernels
# ============================================
TOTAL=${#SELECTED_KERNELS[@]}
DONE=0
SKIP=0
FAIL=0

# Persistent hpcstruct cache (shared across all kernel runs)
CACHE_DIR="$HOME/.hpctoolkit/hpcstruct-cache-intel"
mkdir -p "$CACHE_DIR"

# Log file (tee to both terminal and file)
LOGFILE="$RESULTS_BASE/.log-intel-aurora.txt"
mkdir -p "$RESULTS_BASE"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================"
echo " Per-Kernel RAJAPerf on Aurora (Intel PVC)"
echo "============================================"
echo "SIF:      $SIF"
echo "Kernels:  $TOTAL"
echo "Passes:   $NPASSES"
echo "Cache:    $CACHE_DIR"
echo "Log:      $LOGFILE"
echo "Output:   $RESULTS_BASE/"
echo ""

for KERNEL in "${SELECTED_KERNELS[@]}"; do
    OUTDIR="$RESULTS_BASE/$KERNEL/intel"

    # Skip if already has results
    if compgen -G "$OUTDIR/hpctoolkit-*-database" > /dev/null 2>&1; then
        echo "SKIP $KERNEL (exists)"
        SKIP=$((SKIP + 1))
        continue
    fi

    mkdir -p "$OUTDIR"
    echo "[$((DONE + FAIL + 1))/$TOTAL] Running $KERNEL ..."

    if apptainer exec \
        --cleanenv \
        --no-home \
        --bind "$OUTDIR:/workspace" \
        --bind "$CACHE_DIR:/opt/hpcstruct-cache" \
        "$SIF" \
        bash -c '
            set -e
            export LD_LIBRARY_PATH="/opt/dyninst/lib:/opt/dyninst/lib64:$LD_LIBRARY_PATH"
            export ZE_ENABLE_TRACING_LAYER=1
            export HPCTOOLKIT_HPCSTRUCT_CACHE=/opt/hpcstruct-cache
            cd /workspace

            hpcrun -e "gpu=level0,pc" /opt/rajaperf/bin/raja-perf.exe \
                --variants Base_SYCL --checkrun 1 --npasses '"$NPASSES"' \
                --kernels '"$KERNEL"'

            MEAS=$(ls -td hpctoolkit-raja-perf.exe-measurements* | head -1)
            hpcstruct --gpucfg yes "$MEAS"
            hpcprof "$MEAS"
        '; then
        DONE=$((DONE + 1))
        echo "OK $KERNEL"
    else
        FAIL=$((FAIL + 1))
        echo "FAILED $KERNEL"
    fi
done

echo ""
echo "============================================"
echo " Summary: $DONE done, $SKIP skipped, $FAIL failed"
echo "============================================"
