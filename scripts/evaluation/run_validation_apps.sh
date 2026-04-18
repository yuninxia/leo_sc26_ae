#!/bin/bash
# Root-cause validation for HPC application kernels (non-RAJAPerf).
# Profiles original and optimized versions, runs Leo on both.
#
# Usage:
#   ./run_validation_apps.sh nvidia --apps kripke lulesh xsbench minibude
#   ./run_validation_apps.sh nvidia --apps kripke    # single app
#   ./run_validation_apps.sh amd --apps kripke lulesh xsbench
#
# Output:
#   results/validation/<APP>/<vendor>/
#     original/leo_output.txt
#     optimized/leo_output.txt
#     validation.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$LEO_ROOT/results/validation"
BENCH_DIR="$LEO_ROOT/benchmarks"

# Vendor config
declare -A MACHINES DOCKER_GPU_FLAGS GPU_EVENTS GPU_ARCH GPU_VISIBLE_ENV
MACHINES[amd]="odyssey"
MACHINES[nvidia]="gilgamesh"

DOCKER_GPU_FLAGS[amd]="--device=/dev/kfd --device=/dev/dri"
DOCKER_GPU_FLAGS[nvidia]="--gpus all"

GPU_EVENTS[amd]="gpu=rocm,pc=hw@25"
GPU_EVENTS[nvidia]="gpu=cuda,pc"

GPU_ARCH[amd]="mi300"
GPU_ARCH[nvidia]="h100"

GPU_VISIBLE_ENV[amd]="-e ROCR_VISIBLE_DEVICES=0"
GPU_VISIBLE_ENV[nvidia]="-e CUDA_VISIBLE_DEVICES=0"

# Parse arguments
VENDOR="${1:-}"
shift || true

SELECTED_APPS=()
NPASSES=${NPASSES:-200}

while [[ $# -gt 0 ]]; do
    case $1 in
        --apps|-a)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_APPS+=("$1")
                shift
            done
            ;;
        --npasses) NPASSES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$VENDOR" || ${#SELECTED_APPS[@]} -eq 0 ]]; then
    echo "Usage: $0 <amd|nvidia> --apps <app1> [app2 ...]"
    echo ""
    echo "Available apps: kripke lulesh xsbench minibude llamacpp"
    exit 1
fi

MACHINE="${MACHINES[$VENDOR]}"
EVENT="${GPU_EVENTS[$VENDOR]}"
ARCH="${GPU_ARCH[$VENDOR]}"
DOCKER_FLAGS="${DOCKER_GPU_FLAGS[$VENDOR]}"
VIS_ENV="${GPU_VISIBLE_ENV[$VENDOR]}"

echo "============================================"
echo " HPC App Validation"
echo "============================================"
echo "Vendor:  $VENDOR ($MACHINE)"
echo "Apps:    ${SELECTED_APPS[*]}"
echo "Event:   $EVENT"
echo "Output:  $RESULTS_BASE/"
echo ""

# Common inner script for profiling + Leo
make_inner_script() {
    local APP="$1"
    local EXEC="$2"
    local ARGS="$3"
    local OPT_CMD="$4"  # Command to apply optimization and rebuild

    cat << INNEREOF
#!/bin/bash
set -e

EVENT="$EVENT"
ARCH="$ARCH"
EXEC="$EXEC"
EXEC_NAME="\$(basename \$EXEC)"

export LD_LIBRARY_PATH="/opt/dyninst/lib:/opt/dyninst/lib64:\$LD_LIBRARY_PATH"
export ZE_ENABLE_TRACING_LAYER=1

profile_and_leo() {
    local LABEL="\$1"
    local OUTDIR="\$2"
    local BIN="\$3"

    echo ""
    echo "--- \$LABEL: profiling $APP ---"
    local WORKDIR="/tmp/validation-\${LABEL}"
    rm -rf "\$WORKDIR"
    mkdir -p "\$WORKDIR"
    cd "\$WORKDIR"

    echo "  hpcrun ..."
    hpcrun -e "\$EVENT" \$BIN $ARGS 2>&1 | tail -3

    MEAS_DIR=\$(ls -d hpctoolkit-\${EXEC_NAME}-measurements* 2>/dev/null | head -1)
    if [[ -z "\$MEAS_DIR" ]]; then
        echo "  ERROR: no measurements found"
        return 1
    fi

    echo "  hpcstruct ..."
    hpcstruct --gpucfg yes "\$MEAS_DIR" 2>&1 | tail -2

    echo "  hpcprof ..."
    hpcprof "\$MEAS_DIR" 2>&1 | tail -2

    echo "  Copying measurements..."
    cp -r "\$WORKDIR/hpctoolkit-"* "\$OUTDIR/" 2>/dev/null || true

    echo "  Leo analysis ..."
    cd /opt/leo
    PYTHONUNBUFFERED=1 uv run python scripts/analyze_benchmark.py \\
        "\$WORKDIR/\$MEAS_DIR" \\
        --arch "\$ARCH" --top-n 2 2>&1 | tee "\$WORKDIR/leo_output.txt" | tail -30 \\
        || echo "  WARNING: Leo failed"
    cp "\$WORKDIR/leo_output.txt" "\$OUTDIR/" 2>/dev/null || true

    echo "  [\$LABEL] done"
}

# Original
profile_and_leo "original" "/data/original" "\$EXEC"

# Apply optimization
echo ""
echo "--- Applying optimization for $APP ---"
$OPT_CMD

# Optimized
profile_and_leo "optimized" "/data/optimized" "\$EXEC"

echo ""
echo "=== COMPARISON ==="
echo "--- BEFORE (original) ---"
grep -A15 "STALL ANALYSIS" /tmp/validation-original/leo_output.txt 2>/dev/null | head -18
echo ""
echo "--- AFTER (optimized) ---"
grep -A15 "STALL ANALYSIS" /tmp/validation-optimized/leo_output.txt 2>/dev/null | head -18
INNEREOF
}

# Per-app validation
for APP in "${SELECTED_APPS[@]}"; do
    echo "============================================"
    echo " Validating: $APP ($VENDOR)"
    echo "============================================"

    APP_DIR="$RESULTS_BASE/$APP/$VENDOR"
    ORIG_DIR="$APP_DIR/original"
    OPT_DIR="$APP_DIR/optimized"
    mkdir -p "$ORIG_DIR" "$OPT_DIR"

    # App-specific configuration
    case "$APP" in
        kripke)
            IMAGE="leo-kripke-${VENDOR}"
            EXEC="/opt/kripke/bin/kripke.exe"
            ARGS="--zones 32,32,32 --groups 32 --niter 10"
            [[ "$VENDOR" == "amd" ]] && ARGS="HSA_XNACK=1 $ARGS"
            MOUNTS="-v $BENCH_DIR/kripke/optimized:/opt/kripke-opt:ro"
            # Optimization: replace LTimes.h and rebuild
            OPT_CMD="cp /opt/kripke-opt/LTimes.h /opt/kripke-src/src/Kripke/Arch/LTimes.h && cd /opt/kripke-build && make -j\$(nproc) 2>&1 | tail -3 && cp kripke.exe /opt/kripke/bin/kripke.exe"
            ;;
        lulesh)
            IMAGE="leo-lulesh-${VENDOR}"
            EXEC="/opt/lulesh/bin/lulesh2.0"
            ARGS="-s 45 -i 200"
            MOUNTS="-v $BENCH_DIR/lulesh/optimized:/opt/lulesh-opt-src:ro"
            if [[ "$VENDOR" == "nvidia" ]]; then
                OPT_CMD="cp /opt/lulesh-opt-src/*.cc /opt/lulesh-opt-src/*.h /opt/lulesh-src/LULESH/omp_4.0/ 2>/dev/null || true && cd /opt/lulesh-src/LULESH/omp_4.0 && sed -i 's|-L/usr/local/cuda/nvvm/libdevice||' Makefile && sed -i 's|\$(CXX) -fopenmp=libomp.*-o \$@|\$(CXX) \$(CXXFLAGS) \$(OBJECTS2.0) \$(LDFLAGS) -lstdc++ -lm -o \$@|' Makefile && make clean > /dev/null 2>&1 && make CXX=nvc++ CXXFLAGS='-DUSE_MPI=0 -fast -gopt -mp=gpu -gpu=cc90' LDFLAGS='' OMPFLAGS='' > /dev/null 2>&1 && cp lulesh2.0 /opt/lulesh/bin/lulesh2.0"
            else
                OPT_CMD="cp /opt/lulesh-opt-src/*.cc /opt/lulesh-opt-src/*.h /opt/lulesh-src/ 2>/dev/null || true && cd /opt/lulesh-src && sed -i 's|-L/usr/local/cuda/nvvm/libdevice||' Makefile && sed -i 's|\$(CXX) -fopenmp=libomp.*-o \$@|\$(CXX) \$(CXXFLAGS) \$(OBJECTS2.0) \$(LDFLAGS) -lstdc++ -lm -o \$@|' Makefile && make clean > /dev/null 2>&1 && make CXX=amdclang++ CXXFLAGS='-DUSE_MPI=0 -g -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx942' LDFLAGS='' OMPFLAGS='' > /dev/null 2>&1 && cp lulesh2.0 /opt/lulesh/bin/lulesh2.0"
            fi
            ;;
        xsbench)
            IMAGE="leo-xsbench-${VENDOR}"
            EXEC="/opt/xsbench/bin/XSBench"
            ARGS="-m event -G hash"
            if [[ "$VENDOR" == "nvidia" ]]; then
                MOUNTS="-v $BENCH_DIR/xsbench/optimized/cuda:/opt/xsbench-opt-src:ro"
                OPT_CMD="cp /opt/xsbench-opt-src/Simulation.cu /opt/xsbench-opt-src/XSbench_header.h /opt/xsbench-opt-src/GridInit.cu /opt/XSBench/cuda/ 2>/dev/null || true && cd /opt/XSBench/cuda && make clean > /dev/null 2>&1 && make > /dev/null 2>&1 && cp XSBench /opt/xsbench/bin/XSBench"
            else
                MOUNTS="-v $BENCH_DIR/xsbench/optimized/hip:/opt/xsbench-opt-src:ro"
                OPT_CMD="cp /opt/xsbench-opt-src/*.cpp /opt/xsbench-opt-src/*.h /opt/XSBench/hip/ 2>/dev/null || true && cd /opt/XSBench/hip && make clean > /dev/null 2>&1 && make COMPILER=amd OPTIMIZE=yes CFLAGS='-std=c++14 -O3 --offload-arch=gfx942 -g -fno-omit-frame-pointer -include cstring' > /dev/null 2>&1 && cp XSBench /opt/xsbench/bin/XSBench"
            fi
            ;;
        minibude)
            IMAGE="leo-minibude-${VENDOR}"
            EXEC="/opt/minibude/bin/cuda-bude"
            ARGS="--deck /opt/miniBUDE/data/bm2 -p 2 -w 256"
            MOUNTS="-v $BENCH_DIR/minibude/optimized/src/cuda:/opt/minibude-opt-src:ro"
            OPT_CMD="echo 'miniBUDE: rebuilding with optimized source...' && cp /opt/minibude-opt-src/fasten.hpp /opt/miniBUDE/src/cuda/fasten.hpp 2>/dev/null || true && cd /opt/minibude-build && cmake --build . 2>&1 | tail -3 && cp cuda-bude /opt/minibude/bin/cuda-bude 2>/dev/null || true"
            ;;
        llamacpp)
            IMAGE="leo-llamacpp-${VENDOR}"
            EXEC="/opt/llamacpp/bin/llama-bench"
            ARGS="-m /workspace/model.gguf -p 512 -n 0 -ngl 99"
            # Find model file
            MODEL_FILE=$(find "$LEO_ROOT/results" -name "model.gguf" -path "*nvidia*llamacpp*" 2>/dev/null | head -1)
            if [[ -z "$MODEL_FILE" ]]; then
                MODEL_FILE=$(find "$LEO_ROOT/results" -name "model.gguf" 2>/dev/null | head -1)
            fi
            if [[ -z "$MODEL_FILE" ]]; then
                echo "WARNING: No model.gguf found for llama.cpp — skipping"
                continue
            fi
            MODEL_DIR=$(dirname "$MODEL_FILE")
            if [[ "$VENDOR" == "nvidia" ]]; then
                MOUNTS="-v $BENCH_DIR/llamacpp/optimized:/opt/llamacpp-opt:ro -v $MODEL_FILE:/workspace/model.gguf:ro"
                OPT_CMD="cd /opt/llama.cpp && git apply /opt/llamacpp-opt/nvidia_mmq_optimize.patch 2>&1 | tail -3 && cmake --build build -j\$(nproc) 2>&1 | tail -3 && cp build/bin/llama-bench /opt/llamacpp/bin/llama-bench"
            else
                MOUNTS="-v $BENCH_DIR/llamacpp/optimized:/opt/llamacpp-opt:ro -v $MODEL_FILE:/workspace/model.gguf:ro"
                OPT_CMD="cd /opt/llama.cpp && git apply /opt/llamacpp-opt/amd_mmq_optimize.patch 2>&1 | tail -3 && cmake --build build -j\$(nproc) 2>&1 | tail -3 && cp build/bin/llama-bench /opt/llamacpp/bin/llama-bench"
            fi
            ;;
        *)
            echo "Unknown app: $APP"
            continue
            ;;
    esac

    # Write inner script
    TMPSCRIPT="$APP_DIR/.run_inner.sh"
    make_inner_script "$APP" "$EXEC" "$ARGS" "$OPT_CMD" > "$TMPSCRIPT"
    chmod +x "$TMPSCRIPT"

    # Docker command
    DOCKER_CMD="docker run --rm \
        $DOCKER_FLAGS \
        $VIS_ENV \
        -e PYTHONUNBUFFERED=1 \
        -v $ORIG_DIR:/data/original \
        -v $OPT_DIR:/data/optimized \
        -v $TMPSCRIPT:/opt/run.sh:ro \
        $MOUNTS \
        --entrypoint bash \
        $IMAGE \
        -c '$OPT_CMD && bash /opt/run.sh'"

    # Actually we need to run original FIRST, then optimize, then optimized
    DOCKER_CMD="docker run --rm \
        $DOCKER_FLAGS \
        $VIS_ENV \
        -e PYTHONUNBUFFERED=1 \
        -v $ORIG_DIR:/data/original \
        -v $OPT_DIR:/data/optimized \
        -v $TMPSCRIPT:/opt/run.sh:ro \
        $MOUNTS \
        --entrypoint bash \
        $IMAGE \
        -c 'bash /opt/run.sh'"

    # Run locally or via SSH
    if [[ "$MACHINE" == "$(hostname)" || "$MACHINE" == "odyssey" ]]; then
        eval "$DOCKER_CMD" 2>&1 | tee "$APP_DIR/validation.log"
    else
        ssh "$MACHINE" "$DOCKER_CMD" 2>&1 | tee "$APP_DIR/validation.log"
    fi

    rm -f "$TMPSCRIPT"
    echo ""
    echo "Results: $APP_DIR/{original,optimized}/leo_output.txt"
    echo ""
done

echo "============================================"
echo " App validation complete"
echo "============================================"
