#!/bin/bash
# Run full evaluation workflow: profile a workload with HPCToolkit, then analyze with Leo
#
# Usage:
#   ./run_evaluation.sh <vendor> [--workload <name>] [options]
#
# Examples:
#   ./run_evaluation.sh amd                                   # RAJAPerf (default)
#   ./run_evaluation.sh amd --kernels DAXPY MULADDSUB
#   ./run_evaluation.sh nvidia --workload llamacpp --model /path/to/model.gguf
#   ./run_evaluation.sh nvidia --kernels Polybench --top-n 3
#
# The script SSHes to the target machine and runs the workflow inside the Docker container:
#   1. hpcrun  -e gpu=<rocm|cuda|level0>,pc[=hw] <workload-binary> ...
#   2. hpcstruct --gpucfg yes <measurements>/
#   3. hpcprof <measurements>/
#   4. Leo analysis on the resulting database

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENDOR="${1:-}"
shift || true

# ============================================
# Vendor configuration
# ============================================
declare -A MACHINES DOCKER_GPU_FLAGS GPU_EVENTS GPU_ARCH VARIANTS
MACHINES[amd]="odyssey"
MACHINES[nvidia]="illyad"
MACHINES[nvidia-gilgamesh]="gilgamesh"
MACHINES[nvidia-athena]="athena"
MACHINES[nvidia-voltar]="voltar"
MACHINES[nvidia-hopper1]="hopper1"
MACHINES[nvidia-hopper2]="hopper2"
MACHINES[intel]="headroom"

DOCKER_GPU_FLAGS[amd]="--device=/dev/kfd --device=/dev/dri"
DOCKER_GPU_FLAGS[nvidia]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-gilgamesh]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-athena]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-voltar]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-hopper1]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-hopper2]="--gpus all"
DOCKER_GPU_FLAGS[intel]="--device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path"

# HPCToolkit event strings
GPU_EVENTS[amd]="gpu=rocm,pc=hw@25"
GPU_EVENTS[nvidia]="gpu=cuda,pc"
GPU_EVENTS[nvidia-gilgamesh]="gpu=cuda,pc"
GPU_EVENTS[nvidia-athena]="gpu=cuda,pc"
GPU_EVENTS[nvidia-voltar]="gpu=cuda,pc"
GPU_EVENTS[nvidia-hopper1]="gpu=cuda,pc"
GPU_EVENTS[nvidia-hopper2]="gpu=cuda,pc"
GPU_EVENTS[intel]="gpu=level0,pc"

# Leo architecture names (use specific GPU models for accurate analysis)
GPU_ARCH[amd]="mi300"
GPU_ARCH[nvidia]="h100"
GPU_ARCH[nvidia-gilgamesh]="h100"
GPU_ARCH[nvidia-athena]="a100"
GPU_ARCH[nvidia-voltar]="a100"
GPU_ARCH[nvidia-hopper1]="gh200"
GPU_ARCH[nvidia-hopper2]="gh200"
GPU_ARCH[intel]="pvc"

# RAJAPerf variant to run (GPU-specific)
VARIANTS[amd]="Base_HIP"
VARIANTS[nvidia]="Base_CUDA"
VARIANTS[nvidia-gilgamesh]="Base_CUDA"
VARIANTS[nvidia-athena]="Base_CUDA"
VARIANTS[nvidia-voltar]="Base_CUDA"
VARIANTS[nvidia-hopper1]="Base_CUDA"
VARIANTS[nvidia-hopper2]="Base_CUDA"
VARIANTS[intel]="Base_SYCL"

# CabanaMD device type per vendor
declare -A CABANA_DEVICE
CABANA_DEVICE[amd]="HIP"
CABANA_DEVICE[nvidia]="CUDA"
CABANA_DEVICE[nvidia-gilgamesh]="CUDA"
CABANA_DEVICE[nvidia-athena]="CUDA"
CABANA_DEVICE[nvidia-voltar]="CUDA"
CABANA_DEVICE[nvidia-hopper1]="CUDA"
CABANA_DEVICE[nvidia-hopper2]="CUDA"
CABANA_DEVICE[intel]=""

# Limit profiling to a single GPU (avoids multi-die contention during PC sampling)
declare -A GPU_VISIBLE_ENV
GPU_VISIBLE_ENV[amd]="-e ROCR_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia]="-e CUDA_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia-gilgamesh]="-e CUDA_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia-athena]="-e CUDA_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia-voltar]="-e CUDA_VISIBLE_DEVICES=2"
GPU_VISIBLE_ENV[nvidia-hopper1]="-e CUDA_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia-hopper2]="-e CUDA_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[intel]="-e ZE_AFFINITY_MASK=0"

# hpcstruct GPU CFG flags
declare -A GPUCFG_FLAGS
GPUCFG_FLAGS[amd]="--gpucfg yes"
GPUCFG_FLAGS[nvidia]="--gpucfg yes"
GPUCFG_FLAGS[nvidia-gilgamesh]="--gpucfg yes"
GPUCFG_FLAGS[nvidia-athena]="--gpucfg yes"
GPUCFG_FLAGS[nvidia-voltar]="--gpucfg yes"
GPUCFG_FLAGS[nvidia-hopper1]="--gpucfg yes"
GPUCFG_FLAGS[nvidia-hopper2]="--gpucfg yes"
GPUCFG_FLAGS[intel]="--gpucfg yes"

# Singularity/Apptainer GPU flags (used when Docker is not available)
declare -A SIF_FLAGS
SIF_FLAGS[amd]="--rocm"
SIF_FLAGS[nvidia]="--nv"
SIF_FLAGS[nvidia-gilgamesh]="--nv"
SIF_FLAGS[nvidia-athena]="--nv"
SIF_FLAGS[nvidia-voltar]="--nv"
SIF_FLAGS[nvidia-hopper1]="--nv"
SIF_FLAGS[nvidia-hopper2]="--nv"
SIF_FLAGS[intel]="--bind /dev/dri"

# ============================================
# Workload configuration
# ============================================
# Workload executables (inside container)
declare -A WORKLOAD_EXEC
WORKLOAD_EXEC[rajaperf]="/opt/rajaperf/bin/raja-perf.exe"
WORKLOAD_EXEC[llamacpp]="/opt/llamacpp/bin/llama-bench"
WORKLOAD_EXEC[babelstream]="/opt/babelstream/bin/babelstream"  # resolved per-vendor below
WORKLOAD_EXEC[minibude]="/opt/minibude/bin/bude"              # resolved per-vendor below
WORKLOAD_EXEC[xsbench]="/opt/xsbench/bin/XSBench"
WORKLOAD_EXEC[kripke]="/opt/kripke/bin/kripke.exe"
WORKLOAD_EXEC[arborx]="/opt/arborx-build/benchmarks/cluster/ArborX_Benchmark_DBSCAN.exe"
WORKLOAD_EXEC[qmcpack]="/opt/qmcpack/bin/qmcpack"
WORKLOAD_EXEC[cabanamd]="/opt/cabanamd/bin/cbnMD"
WORKLOAD_EXEC[amrwind]="/opt/amr-wind/bin/amr_wind"
WORKLOAD_EXEC[quicksilver]="/opt/quicksilver/bin/qs"
WORKLOAD_EXEC[quicksilver-native]="/opt/quicksilver/bin/qs"
WORKLOAD_EXEC[lulesh]="/opt/lulesh/bin/lulesh2.0"
WORKLOAD_EXEC[hipkittens-gemm]="/opt/hipkittens/bin/hipkittens_gemm_bench"
WORKLOAD_EXEC[hipkittens-gemm-1024]="/opt/hipkittens/bin/hipkittens_gemm_1024_bench"
WORKLOAD_EXEC[hipkittens-gemm-2048]="/opt/hipkittens/bin/hipkittens_gemm_2048_bench"
WORKLOAD_EXEC[hipkittens-gemm-4096]="/opt/hipkittens/bin/hipkittens_gemm_4096_bench"
WORKLOAD_EXEC[hipkittens-gemm-16384]="/opt/hipkittens/bin/hipkittens_gemm_16384_bench"
WORKLOAD_EXEC[hipkittens-layernorm]="/opt/hipkittens/bin/hipkittens_layernorm_bench"
WORKLOAD_EXEC[hipkittens-layernorm-1024]="/opt/hipkittens/bin/hipkittens_layernorm_1024_bench"
WORKLOAD_EXEC[hipkittens-layernorm-2048]="/opt/hipkittens/bin/hipkittens_layernorm_2048_bench"
WORKLOAD_EXEC[hipkittens-layernorm-8192]="/opt/hipkittens/bin/hipkittens_layernorm_8192_bench"
WORKLOAD_EXEC[hipkittens-layernorm-16384]="/opt/hipkittens/bin/hipkittens_layernorm_16384_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm]="/opt/hipkittens/bin/hipkittens_rmsnorm_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-8192]="/opt/hipkittens/bin/hipkittens_rmsnorm_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-16384]="/opt/hipkittens/bin/hipkittens_rmsnorm_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-65536]="/opt/hipkittens/bin/hipkittens_rmsnorm_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-opt]="/opt/hipkittens/bin/hipkittens_rmsnorm_optimized_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-pipe]="/opt/hipkittens/bin/hipkittens_rmsnorm_pipelined_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-pipe-8192]="/opt/hipkittens/bin/hipkittens_rmsnorm_pipelined_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-pipe-16384]="/opt/hipkittens/bin/hipkittens_rmsnorm_pipelined_bench"
WORKLOAD_EXEC[hipkittens-rmsnorm-pipe-65536]="/opt/hipkittens/bin/hipkittens_rmsnorm_pipelined_bench"
WORKLOAD_EXEC[thunderkittens-gemm]="/opt/thunderkittens/bin/tk_gemm_bench"
WORKLOAD_EXEC[thunderkittens-gemm-8192]="/opt/thunderkittens/bin/tk_gemm_bench"
WORKLOAD_EXEC[thunderkittens-fp8gemm]="/opt/thunderkittens/bin/tk_fp8_gemm_bench"
WORKLOAD_EXEC[thunderkittens-fp8gemm-8192]="/opt/thunderkittens/bin/tk_fp8_gemm_bench"
WORKLOAD_EXEC[thunderkittens-attn]="/opt/thunderkittens/bin/tk_attn_bench"
WORKLOAD_EXEC[thunderkittens-attn-8192]="/opt/thunderkittens/bin/tk_attn_bench"
WORKLOAD_EXEC[thunderkittens-layernorm]="/opt/thunderkittens/bin/tk_layernorm_bench"
WORKLOAD_EXEC[thunderkittens-layernorm-8192]="/opt/thunderkittens/bin/tk_layernorm_bench"
WORKLOAD_EXEC[thunderkittens-rotary]="/opt/thunderkittens/bin/tk_rotary_bench"
WORKLOAD_EXEC[thunderkittens-rotary-8192]="/opt/thunderkittens/bin/tk_rotary_bench"
WORKLOAD_EXEC[thunderkittens-mamba2]="/opt/thunderkittens/bin/tk_mamba2_bench"
WORKLOAD_EXEC[thunderkittens-mamba2-8192]="/opt/thunderkittens/bin/tk_mamba2_bench"

# Workload arguments (passed after executable path)
declare -A WORKLOAD_ARGS
WORKLOAD_ARGS[hipkittens-rmsnorm-8192]="8192"
WORKLOAD_ARGS[hipkittens-rmsnorm-16384]="16384"
WORKLOAD_ARGS[hipkittens-rmsnorm-65536]="65536"
WORKLOAD_ARGS[hipkittens-rmsnorm-pipe-8192]="8192"
WORKLOAD_ARGS[hipkittens-rmsnorm-pipe-16384]="16384"
WORKLOAD_ARGS[hipkittens-rmsnorm-pipe-65536]="65536"
WORKLOAD_ARGS[thunderkittens-gemm-8192]="8192"
WORKLOAD_ARGS[thunderkittens-fp8gemm-8192]="8192"
WORKLOAD_ARGS[thunderkittens-attn-8192]="8192"
WORKLOAD_ARGS[thunderkittens-layernorm-8192]="8192"
WORKLOAD_ARGS[thunderkittens-rotary-8192]="8192"
WORKLOAD_ARGS[thunderkittens-mamba2-8192]="8192"
WORKLOAD_ARGS[minibude]="--deck /opt/miniBUDE/data/bm2 -p 2 -w 256"
WORKLOAD_ARGS[babelstream]="--arraysize $((2**28))"
WORKLOAD_ARGS[xsbench]="-m event -G hash"
WORKLOAD_ARGS[lulesh]="-i 100"  # Intel/NVIDIA default; AMD overridden below
WORKLOAD_EXEC[cujson]="/opt/cuJSON/bin/cujson_standard_optimized"
WORKLOAD_ARGS[cujson]="/opt/cuJSON/dataset/bench_large.json"
WORKLOAD_EXEC[minitest]="/opt/minitest/bin/single.ompoffload.icpx.intelgpu"
# FlashInfer MLA/GQA attention benchmarks (built at runtime from host-mounted source)
WORKLOAD_EXEC[flashinfer-mla]="/opt/flashinfer-bench/mla_bench"
WORKLOAD_ARGS[flashinfer-mla]="8 2048 16 512 64 1"
WORKLOAD_EXEC[flashinfer-mla-prefill]="/opt/flashinfer-bench/mla_bench"
WORKLOAD_ARGS[flashinfer-mla-prefill]="8 2048 16 512 64 64"

# ============================================
# Parse options
# ============================================
WORKLOAD="rajaperf"
RAJAPERF_ARGS=""
RAJAPERF_KERNELS_SET=false
LEO_TOP_N=2
MODEL_PATH=""
LLAMACPP_ARGS=""
OUTPUT_DIR=""
SKIP_LEO=false
BENCH_ONLY=false
HPCSTRUCT_JOBS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --workload)
            WORKLOAD="$2"; shift 2 ;;
        --kernels|-k)
            shift
            KERNEL_ARGS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                KERNEL_ARGS="$KERNEL_ARGS $1"
                shift
            done
            RAJAPERF_ARGS="$RAJAPERF_ARGS --kernels $KERNEL_ARGS"
            RAJAPERF_KERNELS_SET=true
            ;;
        --size)
            RAJAPERF_ARGS="$RAJAPERF_ARGS --size $2"
            shift 2
            ;;
        --checkrun)
            RAJAPERF_ARGS="$RAJAPERF_ARGS --checkrun $2"
            shift 2
            ;;
        --npasses)
            RAJAPERF_ARGS="$RAJAPERF_ARGS --npasses $2"
            shift 2
            ;;
        --top-n)
            LEO_TOP_N="$2"
            shift 2
            ;;
        --arch)
            GPU_ARCH[$VENDOR]="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --llama-args)
            LLAMACPP_ARGS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-leo)
            SKIP_LEO=true
            shift
            ;;
        --bench-only)
            BENCH_ONLY=true
            shift
            ;;
        --hpcstruct-jobs|-j)
            HPCSTRUCT_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================
# RAJAPerf default kernel filter
# ============================================
# Keep Apps (real HPC applications) + Polybench.
# Excludes Basic (trivial), Stream (bandwidth-only), Lcals, Algorithm (only 1 shared: REDUCE_SUM).
# See docs/RAJAPERF_SHARED_KERNELS.md for the full kernel list.
# Override with --kernels to run specific kernels or groups.
if [ "$WORKLOAD" = "rajaperf" ] && [ "$RAJAPERF_KERNELS_SET" = "false" ]; then
    RAJAPERF_ARGS="--kernels Apps Polybench $RAJAPERF_ARGS"
fi

# ============================================
# Validation
# ============================================
if [[ -z "$VENDOR" || ! "${MACHINES[$VENDOR]+_}" ]]; then
    echo "Usage: $0 <amd|nvidia|nvidia-gilgamesh|nvidia-athena|nvidia-voltar|nvidia-hopper1|nvidia-hopper2|intel> [--workload <name>] [options]"
    echo ""
    echo "Workloads:"
    echo "  rajaperf (default)  RAJAPerf benchmark suite"
    echo "  llamacpp            llama.cpp LLM inference benchmark"
    echo "  babelstream         BabelStream memory bandwidth (Copy/Mul/Add/Triad/Dot)"
    echo "  minibude            miniBUDE molecular docking (compute-bound)"
    echo "  xsbench             XSBench nuclear cross-section lookup"
    echo "  arborx              ArborX DBSCAN clustering benchmark (Kokkos, 10M 3D points)"
    echo "  qmcpack             QMCPACK quantum Monte Carlo"
    echo "  cabanamd            CabanaMD molecular dynamics (Kokkos)"
    echo "  amrwind             AMR-Wind incompressible CFD (AMReX)"
    echo "  quicksilver         QuickSilver Monte Carlo particle transport"
    echo "  quicksilver-native  QuickSilver LLNL native HIP build (AMD only)"
    echo "  hipkittens-gemm     HipKittens BF16 GEMM 8192x8192 (AMD only, default)"
    echo "  hipkittens-gemm-1024  HipKittens BF16 GEMM 1024x1024 (AMD only)"
    echo "  hipkittens-gemm-2048  HipKittens BF16 GEMM 2048x2048 (AMD only)"
    echo "  hipkittens-gemm-4096  HipKittens BF16 GEMM 4096x4096 (AMD only)"
    echo "  hipkittens-gemm-16384 HipKittens BF16 GEMM 16384x16384 (AMD only)"
    echo "  hipkittens-layernorm  HipKittens BF16 LayerNorm B=16 N=4096 D=2048 (AMD only, default)"
    echo "  hipkittens-layernorm-1024  HipKittens BF16 LayerNorm N=1024 (AMD only)"
    echo "  hipkittens-layernorm-2048  HipKittens BF16 LayerNorm N=2048 (AMD only)"
    echo "  hipkittens-layernorm-8192  HipKittens BF16 LayerNorm N=8192 (AMD only)"
    echo "  hipkittens-layernorm-16384 HipKittens BF16 LayerNorm N=16384 (AMD only)"
    echo "  hipkittens-rmsnorm  HipKittens BF16 RMSNorm B=16 N=4096 D=128 (AMD only)"
    echo "  hipkittens-rmsnorm-16384  HipKittens BF16 RMSNorm N=16384 (AMD only)"
    echo "  hipkittens-rmsnorm-65536  HipKittens BF16 RMSNorm N=65536 (AMD only)"
    echo "  hipkittens-rmsnorm-opt  HipKittens BF16 RMSNorm optimized (shared memory staging, AMD only)"
    echo "  hipkittens-rmsnorm-pipe HipKittens BF16 RMSNorm pipelined (multi-row, AMD only)"
    echo "  hipkittens-rmsnorm-pipe-16384  HipKittens BF16 RMSNorm pipelined N=16384 (AMD only)"
    echo "  hipkittens-rmsnorm-pipe-65536  HipKittens BF16 RMSNorm pipelined N=65536 (AMD only)"
    echo "  thunderkittens-gemm       ThunderKittens BF16 GEMM 4096x4096 (NVIDIA H100 only)"
    echo "  thunderkittens-gemm-8192  ThunderKittens BF16 GEMM 8192x8192"
    echo "  thunderkittens-fp8gemm    ThunderKittens FP8 GEMM 4096x4096"
    echo "  thunderkittens-attn       ThunderKittens MHA B=4 H=16 N=4096 D=128"
    echo "  thunderkittens-layernorm  ThunderKittens LayerNorm B=16 N=4096 D=1024"
    echo "  thunderkittens-rotary     ThunderKittens Rotary B=4 H=32 N=4096 D=128"
    echo "  thunderkittens-mamba2     ThunderKittens Mamba2 B=4 H=16 N=4096 D=64"
    echo "  cujson              cuJSON GPU JSON parser (NVIDIA only, 100MB synthetic dataset)"
    echo "  lulesh              LULESH 2.0 hydrodynamics (OpenMP offload)"
    echo "  minitest            Simple OpenMP offload minitest"
    echo ""
    echo "RAJAPerf options:"
    echo "  --kernels, -k <names>   Kernels/groups to run (default: Apps Polybench)"
    echo "  --size <int>            Problem size"
    echo "  --checkrun <int>        Run each kernel N times (default: 1)"
    echo "  --npasses <int>         Number of passes (default: 1)"
    echo ""
    echo "llama.cpp options:"
    echo "  --model <path>          Path to GGUF model file (required)"
    echo "  --llama-args <args>     Additional llama-bench arguments"
    echo ""
    echo "Common options:"
    echo "  --workload <name>       Workload to profile (default: rajaperf)"
    echo "  --top-n <int>           Leo: analyze top N kernels (default: 5)"
    echo "  --arch <name>           Override Leo GPU architecture"
    echo "  --output-dir <path>     Override results directory path"
    echo "  --hpcstruct-jobs, -j <int>  Threads for hpcstruct binary analysis"
    echo ""
    echo "Target machines:"
    echo "  amd              -> odyssey    (MI300A)"
    echo "  nvidia           -> illyad     (H100)"
    echo "  nvidia-gilgamesh -> gilgamesh  (H100)"
    echo "  nvidia-athena    -> athena     (A100)"
    echo "  nvidia-voltar    -> voltar     (A100-80GB, GPU index 2)"
    echo "  nvidia-hopper1   -> hopper1    (GH200, ARM)"
    echo "  nvidia-hopper2   -> hopper2    (GH200, ARM)"
    echo "  intel            -> headroom   (PVC)"
    exit 1
fi

# Validate workload-specific requirements
if [ "$WORKLOAD" = "llamacpp" ] && [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --model is required for llamacpp workload"
    echo "Example: $0 $VENDOR --workload llamacpp --model /path/to/model.gguf"
    exit 1
fi

MACHINE="${MACHINES[$VENDOR]}"
DOCKER_GPU="${DOCKER_GPU_FLAGS[$VENDOR]}"
EVENT="${GPU_EVENTS[$VENDOR]}"
ARCH="${GPU_ARCH[$VENDOR]}"
VARIANT="${VARIANTS[$VENDOR]}"
GPUCFG="${GPUCFG_FLAGS[$VENDOR]}"
# BabelStream and miniBUDE binaries are named {model}-stream and {model}-bude
if [ "$WORKLOAD" = "babelstream" ]; then
    case "$VENDOR" in
        nvidia*) WORKLOAD_EXEC[babelstream]="/opt/babelstream/bin/cuda-stream" ;;
        amd)     WORKLOAD_EXEC[babelstream]="/opt/babelstream/bin/hip-stream" ;;
        intel)   WORKLOAD_EXEC[babelstream]="/opt/babelstream/bin/sycl2020-acc-stream" ;;
    esac
elif [ "$WORKLOAD" = "minibude" ]; then
    case "$VENDOR" in
        nvidia*) WORKLOAD_EXEC[minibude]="/opt/minibude/bin/cuda-bude" ;;
        amd)     WORKLOAD_EXEC[minibude]="/opt/minibude/bin/hip-bude" ;;
        intel)   WORKLOAD_EXEC[minibude]="/opt/minibude/bin/sycl-bude" ;;
    esac
fi
# All hipkittens/thunderkittens variants share one Docker image per framework
WORKLOAD_FOR_IMAGE="$WORKLOAD"
if [[ "$WORKLOAD" == hipkittens-* ]]; then
    WORKLOAD_FOR_IMAGE="hipkittens"
elif [[ "$WORKLOAD" == thunderkittens-* ]]; then
    WORKLOAD_FOR_IMAGE="thunderkittens"
elif [[ "$WORKLOAD" == flashinfer-* ]]; then
    WORKLOAD_FOR_IMAGE="flashinfer"
fi
# Map vendor to Docker image suffix (nvidia-gilgamesh shares nvidia images)
VENDOR_FOR_IMAGE="$VENDOR"
if [[ "$VENDOR" == nvidia-gilgamesh || "$VENDOR" == nvidia-athena || "$VENDOR" == nvidia-voltar || "$VENDOR" == nvidia-hopper1 || "$VENDOR" == nvidia-hopper2 ]]; then
    VENDOR_FOR_IMAGE="nvidia"
fi
# Vendor-specific workload arg overrides
if [ "$WORKLOAD" = "lulesh" ] && [[ "$VENDOR" == amd* ]]; then
    WORKLOAD_ARGS[lulesh]="-s 60 -i 1000"  # AMD needs larger config for stable timing
fi

IMAGE_NAME="leo-${WORKLOAD_FOR_IMAGE}-${VENDOR_FOR_IMAGE}"

echo "============================================"
echo " Leo Evaluation: $VENDOR on $MACHINE"
echo "============================================"
echo "Workload:     $WORKLOAD"
echo "Container:    $IMAGE_NAME:latest"
echo "GPU event:    $EVENT"
echo "Architecture: $ARCH"
if [ "$WORKLOAD" = "rajaperf" ]; then
    echo "Variant:      $VARIANT"
    echo "RAJAPerf args: ${RAJAPERF_ARGS:-<defaults>}"
elif [ "$WORKLOAD" = "llamacpp" ]; then
    echo "Model:        $MODEL_PATH"
fi
echo "Leo top-N:    $LEO_TOP_N"
echo ""

# ============================================
# Build the in-container script
# ============================================
EXEC_PATH="${WORKLOAD_EXEC[$WORKLOAD]}"
EXEC_NAME="$(basename "$EXEC_PATH")"
MEAS_GLOB="hpctoolkit-${EXEC_NAME}-measurements*"
DB_GLOB="hpctoolkit-${EXEC_NAME}-database*"

TMPSCRIPT="$LEO_ROOT/.eval-run-$$.sh"
trap "rm -f $TMPSCRIPT" EXIT

# Determine the profiling command based on workload
if [ "$WORKLOAD" = "rajaperf" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH \\
    --variants \"$VARIANT\" \\
    --checkrun 1 \\
    $RAJAPERF_ARGS"
elif [ "$WORKLOAD" = "llamacpp" ]; then
    # -p 512: prefill 512 tokens, -n 0: skip generation (decode crashes under hpcrun on Intel SYCL)
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH \\
    -m /workspace/model.gguf \\
    -p 512 -n 0 -ngl 99 \\
    $LLAMACPP_ARGS"
elif [ "$WORKLOAD" = "cabanamd" ]; then
    CABANA_DEV="${CABANA_DEVICE[$VENDOR]}"
    CABANA_DEV_FLAG=""
    if [ -n "$CABANA_DEV" ]; then
        CABANA_DEV_FLAG="--device-type $CABANA_DEV"
    fi
    # Run single-rank: CabanaMD calls MPI_Init internally; single-rank works without a launcher.
    # Avoid mpiexec on all vendors: OpenMPI hits btl_self RDMA segfault in Docker,
    # and Intel MPI's I_MPI_OFFLOAD GPU pinning interferes with Kokkos SYCL device discovery.
    # Use in.lj.bench (no VTK dumps, 200 steps) if available; fall back to in.lj for old images.
    PROFILE_CMD="if [ -f /opt/cabanamd-src/input/in.lj.bench ]; then CABANA_IL=/opt/cabanamd-src/input/in.lj.bench; else CABANA_IL=/opt/cabanamd-src/input/in.lj; fi && hpcrun -e \"$EVENT\" $EXEC_PATH -il \$CABANA_IL $CABANA_DEV_FLAG"
elif [ "$WORKLOAD" = "amrwind" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH /opt/amr-wind-benchmarks/inputs.abl"
elif [ "$WORKLOAD" = "quicksilver" ]; then
    # SYCL backend needs QS_DEVICE=GPU to select the GPU device
    PROFILE_CMD="QS_DEVICE=GPU hpcrun -e \"$EVENT\" $EXEC_PATH -i /opt/quicksilver/examples/CORAL2_Benchmark/Problem1/Coral2_P1.inp -x 16 -y 16 -z 16 -N 20"
elif [ "$WORKLOAD" = "quicksilver-native" ]; then
    # LLNL native HIP build (tutorial settings: -fgpu-rdc, -DHAVE_HIP)
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH -i /opt/quicksilver/examples/CORAL2_Benchmark/Problem1/Coral2_P1.inp -x 16 -y 16 -z 16 -N 20"
elif [ "$WORKLOAD" = "babelstream" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH --arraysize $((2**28))"
elif [ "$WORKLOAD" = "minibude" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH --deck /opt/miniBUDE/data/bm2 -p 2 -w 256"
elif [ "$WORKLOAD" = "xsbench" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH -m event -G hash"
elif [ "$WORKLOAD" = "kripke" ]; then
    # Sn transport: 32^3 zones, 32 energy groups, 10 iterations (~1-2 min on GPU)
    # HSA_XNACK=1 required for CHAI THIN_GPU_ALLOCATE on MI300A (page fault handling)
    PROFILE_CMD="HSA_XNACK=1 hpcrun -e \"$EVENT\" $EXEC_PATH --zones 32,32,32 --groups 32 --niter 10"
elif [ "$WORKLOAD" = "arborx" ]; then
    # DBSCAN benchmark with 10M uniform 3D points, repeated 50x for PC sampling coverage.
    # ROCprofiler stochastic sampling locks onto one dispatch; repeating ensures all kernels
    # get dispatched multiple times so the sampler captures them across iterations.
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH --eps 0.042 --filename /data/points.arborx --core-min-size 2 --verbose --repeat 50"
elif [ "$WORKLOAD" = "qmcpack" ]; then
    # NiO-S4 VMC: 16 atoms, 192 electrons — large enough for QMCPACK's own GPU kernels
    # H5 file at /opt/leo-host/benchmarks/qmcpack/ (135MB, downloaded from ANL Box)
    PROFILE_CMD="bash /opt/leo-host/scripts/evaluation/docker/qmcpack_prep_input.sh && OMP_NUM_THREADS=4 hpcrun -e \"$EVENT\" $EXEC_PATH nio_gpu_profile.xml"
elif [[ "$WORKLOAD" == hipkittens-* ]]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH ${WORKLOAD_ARGS[$WORKLOAD]:-}"
elif [[ "$WORKLOAD" == thunderkittens-* ]]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH ${WORKLOAD_ARGS[$WORKLOAD]:-}"
elif [ "$WORKLOAD" = "lulesh" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH ${WORKLOAD_ARGS[lulesh]}"
elif [ "$WORKLOAD" = "cujson" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH ${WORKLOAD_ARGS[$WORKLOAD]:-}"
elif [ "$WORKLOAD" = "minitest" ]; then
    PROFILE_CMD="hpcrun -e \"$EVENT\" $EXEC_PATH"
elif [[ "$WORKLOAD" == flashinfer-* ]]; then
    # FlashInfer: build from host-mounted source at runtime, then profile from /workspace
    FLASHINFER_BUILD_CMD="cp -r /opt/leo-host/benchmarks/flashinfer /opt/flashinfer-bench && make -C /opt/flashinfer-bench mla_bench CUDA_ARCH=90"
    if [ "$WORKLOAD" = "flashinfer-mla-prefill" ]; then
        PROFILE_CMD="$FLASHINFER_BUILD_CMD && MLA_MODE=prefill hpcrun -e \"$EVENT\" /opt/flashinfer-bench/mla_bench ${WORKLOAD_ARGS[$WORKLOAD]:-8 2048 16 512 64 1}"
    else
        PROFILE_CMD="$FLASHINFER_BUILD_CMD && hpcrun -e \"$EVENT\" /opt/flashinfer-bench/mla_bench ${WORKLOAD_ARGS[$WORKLOAD]:-8 2048 16 512 64 1}"
    fi
fi

WARGS="${WORKLOAD_ARGS[$WORKLOAD]:-}"

cat > "$TMPSCRIPT" <<EOF
#!/bin/bash
set -e

# Ensure dyninst libs are found (installed to lib/, not lib64/)
export LD_LIBRARY_PATH="/opt/dyninst/lib:/opt/dyninst/lib64:\$LD_LIBRARY_PATH"

# Intel Level Zero tracing layer (required for hpcrun PC sampling with GTPin)
export ZE_ENABLE_TRACING_LAYER=1

# Load PC sampling library in shared linker namespace for llamacpp on Intel.
# Without this, dlmopen(LM_ID_NEWLM) creates an isolated namespace where
# environ is uninitialized, causing getenv() segfaults in the PC sampling library.
export HPCRUN_LEVEL0_PC_VISIBLE=1

# Intel MPI GPU-aware support: Cabana passes SYCL device pointers to MPI_Send,
# so Intel MPI must detect and stage GPU buffers via Level Zero.
export I_MPI_OFFLOAD=1
# Use shared-memory fabric only for single-node Docker runs
export I_MPI_FABRICS=shm

# OpenMPI btl_self RDMA crashes in Docker; force eager (copy) protocol for self-sends
export OMPI_MCA_btl_self_eager_limit=1073741824

# Enable hpcstruct cache to avoid re-analyzing unchanged GPU binaries
# Only set if the cache directory is writable (NFS root_squash may prevent this)
if [ -w /opt/hpcstruct-cache ]; then
    export HPCTOOLKIT_HPCSTRUCT_CACHE=/opt/hpcstruct-cache
fi

WORKDIR=/workspace
mkdir -p \$WORKDIR && cd \$WORKDIR

if [ "$BENCH_ONLY" = "true" ]; then
echo ""
echo "=== Benchmark Only (no HPCToolkit) ==="
echo "Command: $EXEC_PATH $WARGS"
echo ""
$EXEC_PATH $WARGS
else

echo ""
echo "=== Step 1: Profile $WORKLOAD with HPCToolkit ==="
echo "Command: $PROFILE_CMD"
echo ""

$PROFILE_CMD

# Find the measurements directory
MEAS_DIR=\$(ls -td $MEAS_GLOB 2>/dev/null | head -1)
if [ -z "\$MEAS_DIR" ]; then
    echo "ERROR: No measurements directory found (pattern: $MEAS_GLOB)"
    exit 1
fi
echo ""
echo "Measurements: \$MEAS_DIR"

echo ""
echo "=== Step 2: hpcstruct ==="
HPCSTRUCT_J_FLAG=""
if [ -n "$HPCSTRUCT_JOBS" ]; then
    HPCSTRUCT_J_FLAG="-j $HPCSTRUCT_JOBS"
fi
hpcstruct $GPUCFG \$HPCSTRUCT_J_FLAG "\$MEAS_DIR" || echo "WARNING: hpcstruct exited with code \$? (continuing anyway)"

echo ""
echo "=== Step 3: hpcprof ==="
hpcprof "\$MEAS_DIR"

# Find the database directory
DB_DIR=\$(ls -td $DB_GLOB 2>/dev/null | head -1)
if [ -z "\$DB_DIR" ]; then
    echo "ERROR: No database directory found (pattern: $DB_GLOB)"
    exit 1
fi
echo ""
echo "Database: \$DB_DIR"

if [ "$SKIP_LEO" = "false" ]; then
echo ""
echo "=== Step 4: Leo Analysis ==="
# Sync Leo source from host mount (container may have stale version)
if [ -d /opt/leo-host/src ]; then
    cp -r /opt/leo-host/src/leo/* /opt/leo/src/leo/ 2>/dev/null || true
    cp /opt/leo-host/scripts/analyze_benchmark.py /opt/leo/scripts/ 2>/dev/null || true
fi
cd /opt/leo
uv run python scripts/analyze_benchmark.py \\
    "\$WORKDIR/\$MEAS_DIR" \\
    --arch "$ARCH" \\
    --top-n $LEO_TOP_N 2>&1 | tee \$WORKDIR/leo_output.txt
else
echo ""
echo "=== Step 4: Leo Analysis (skipped) ==="
fi

fi

# Make all output files accessible to the host user (Docker runs as root,
# but NFS root_squash maps root->nobody, so files end up owned by nobody.
# chmod makes them removable/readable by the mounting user.)
chmod -R a+rwX /workspace 2>/dev/null || true
chmod -R a+rwX /opt/hpcstruct-cache 2>/dev/null || true

echo ""
echo "=== Done ==="
EOF

chmod +x "$TMPSCRIPT"

# ============================================
# Run on target machine
# ============================================
echo "Launching container on $MACHINE..."
echo ""

# Build docker run command with optional mounts
DOCKER_MOUNTS="-v $LEO_ROOT:/opt/leo-host:ro"

# Mount persistent hpcstruct cache (survives across container runs)
CACHE_DIR="$HOME/.hpctoolkit/hpcstruct-cache-$VENDOR"
mkdir -p "$CACHE_DIR"
chmod 777 "$CACHE_DIR"
DOCKER_MOUNTS="$DOCKER_MOUNTS -v $CACHE_DIR:/opt/hpcstruct-cache"

# Mount persistent workspace to keep measurements/database after container exits
if [ -n "$OUTPUT_DIR" ]; then
    RESULTS_DIR="$OUTPUT_DIR"
else
    RESULTS_DIR="$LEO_ROOT/results/${VENDOR}-${WORKLOAD}-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RESULTS_DIR"
chmod 777 "$RESULTS_DIR"
DOCKER_MOUNTS="$DOCKER_MOUNTS -v $RESULTS_DIR:/workspace"
echo "Results dir:  $RESULTS_DIR"
echo ""

# Mount model file for llama.cpp
if [ "$WORKLOAD" = "llamacpp" ] && [ -n "$MODEL_PATH" ]; then
    DOCKER_MOUNTS="$DOCKER_MOUNTS -v $MODEL_PATH:/workspace/model.gguf:ro"
fi

# Mount data file for ArborX DBSCAN benchmark
if [ "$WORKLOAD" = "arborx" ]; then
    ARBORX_DATA="$LEO_ROOT/benchmarks/arborx/data/points_10M.arborx"
    if [ ! -f "$ARBORX_DATA" ]; then
        echo "ERROR: ArborX data file not found: $ARBORX_DATA"
        echo "Generate with: python benchmarks/arborx/generate_data.py --n 10000000 --output $ARBORX_DATA"
        exit 1
    fi
    DOCKER_MOUNTS="$DOCKER_MOUNTS -v $ARBORX_DATA:/data/points.arborx:ro"
fi

# MPI workloads need larger shared memory for inter-process communication.
# Non-MPI workloads (amrwind) must NOT use --shm-size on AMD because it
# triggers GPU hangs with HPCToolkit's ROCm PC sampling.
SHM_FLAG=""
if [ "$WORKLOAD" != "amrwind" ]; then
    SHM_FLAG="--shm-size=4g"
fi

DOCKER_ENV="${GPU_VISIBLE_ENV[$VENDOR]}"

# Check if Docker is available on the target machine
if ssh "$MACHINE" "docker ps &>/dev/null" 2>/dev/null; then
    # Docker is available — use it
    ssh "$MACHINE" "docker run --rm \
        $SHM_FLAG \
        $DOCKER_ENV \
        $DOCKER_GPU \
        $DOCKER_MOUNTS \
        $IMAGE_NAME:latest \
        /opt/leo-host/.eval-run-$$.sh"
else
    # Docker not available — fall back to Singularity/Apptainer
    echo "Docker not available on $MACHINE, using Singularity..."

    # Determine the .sif file path (uses docker vendor name, e.g., "nvidia" not "nvidia-voltar")
    SIF_FILE="$LEO_ROOT/containers/leo-${WORKLOAD_FOR_IMAGE}-${VENDOR_FOR_IMAGE}.sif"
    if ! ssh "$MACHINE" "test -f '$SIF_FILE'" 2>/dev/null; then
        echo "ERROR: Singularity image not found: $SIF_FILE"
        echo "Build it first with: ./build_containers.sh $VENDOR --workload $WORKLOAD_FOR_IMAGE --sif"
        exit 1
    fi

    # Convert Docker mounts (-v src:dst[:opt]) to Singularity binds (--bind src:dst[:opt])
    SINGULARITY_BINDS=""
    while IFS= read -r mount; do
        [ -z "$mount" ] && continue
        SINGULARITY_BINDS="$SINGULARITY_BINDS --bind $mount"
    done < <(echo "$DOCKER_MOUNTS" | grep -oP '(?<=-v )\S+')

    # Convert Docker env vars (-e VAR=VAL) to Singularity env (--env VAR=VAL)
    SINGULARITY_ENV=""
    while IFS= read -r envvar; do
        [ -z "$envvar" ] && continue
        SINGULARITY_ENV="$SINGULARITY_ENV --env $envvar"
    done < <(echo "$DOCKER_ENV" | grep -oP '(?<=-e )\S+')

    # GPU access flags for Singularity
    SINGULARITY_GPU="${SIF_FLAGS[$VENDOR]}"

    ssh "$MACHINE" "singularity exec \
        $SINGULARITY_GPU \
        $SINGULARITY_BINDS \
        $SINGULARITY_ENV \
        $SIF_FILE \
        /opt/leo-host/.eval-run-$$.sh"
fi
