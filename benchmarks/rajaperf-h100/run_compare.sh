#!/bin/bash
# Compare original (upstream) vs optimized (fork) RAJAPerf builds.
#
# Runs inside Docker containers (leo-rajaperf-{intel,amd,nvidia}) which
# already have the upstream RAJAPerf pre-built. Only the optimized fork
# needs to be built, saving ~50% build time.
#
# Directory structure:
#   benchmarks/rajaperf/
#     original/            # upstream LLNL RAJAPerf source (for diff only)
#     optimized/           # fork with Leo optimizations applied
#     run_compare.sh       # this script
#
# Usage:
#   ./run_compare.sh --docker leo-rajaperf-intel                           # Build opt, compare all
#   ./run_compare.sh --docker leo-rajaperf-intel --kernel MASS3DEA         # Compare one kernel
#   ./run_compare.sh --docker leo-rajaperf-intel --skip-build              # Reuse optimized build
#   ./run_compare.sh --docker leo-rajaperf-intel --kernel MASS3DEA --profile
#   ./run_compare.sh --docker leo-rajaperf-intel --warmup 5 --nruns 5      # Custom warmup/runs
#   ./run_compare.sh --list-optimized                                      # Show optimized kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
VENDOR=""
KERNEL_FILTER=""
SKIP_BUILD=false
PROFILE=false
LIST_OPTIMIZED=false
DOCKER_IMAGE=""
PROFILER=""
WARMUP=5
NRUNS=5
PASSES_PER_RUN=10
JOBS=$(nproc 2>/dev/null || echo 8)

# ==================== Argument parsing ====================
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --vendor)         VENDOR="$2"; shift 2 ;;
        --kernel)         KERNEL_FILTER="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --skip-build)     SKIP_BUILD=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --profile)        PROFILE=true; PASSTHROUGH_ARGS+=("$1"); shift ;;
        --profiler)       PROFILER="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --list-optimized) LIST_OPTIMIZED=true; shift ;;
        --warmup)         WARMUP="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --nruns)          NRUNS="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --passes-per-run) PASSES_PER_RUN="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --jobs|-j)        JOBS="$2"; PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --docker)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "ERROR: --docker requires an image name (e.g., leo-rajaperf-intel)"
                exit 1
            fi
            DOCKER_IMAGE="$2"; shift 2
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==================== List optimized kernels (runs locally, no Docker needed) ====================
if [ "$LIST_OPTIMIZED" = true ]; then
    echo "Optimized kernels (files differing between original/ and optimized/):"
    echo ""
    for suffix in Sycl Hip Cuda; do
        files=$(diff -rq "$SCRIPT_DIR/original/src/" "$SCRIPT_DIR/optimized/src/" 2>/dev/null | \
                grep -- "-${suffix}.cpp differ" | \
                sed "s|.*/||; s|-${suffix}.cpp differ||" || true)
        if [ -n "$files" ]; then
            echo "  $suffix backend:"
            echo "$files" | while read -r f; do echo "    $f"; done
        fi
    done
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
    echo "ERROR: --docker <image> required (e.g., leo-rajaperf-intel)"
    exit 1
fi

declare -A RP_VARIANT RP_SUFFIX
RP_VARIANT[intel]="Base_SYCL";  RP_SUFFIX[intel]="Sycl"
RP_VARIANT[amd]="Base_HIP";    RP_SUFFIX[amd]="Hip"
RP_VARIANT[nvidia]="Base_CUDA"; RP_SUFFIX[nvidia]="Cuda"
VARIANT="${RP_VARIANT[$VENDOR]}"
SUFFIX="${RP_SUFFIX[$VENDOR]}"

# ==================== Docker re-launch ====================
if [ -n "$DOCKER_IMAGE" ] && [ ! -f /.dockerenv ]; then
    echo "Launching inside Docker: $DOCKER_IMAGE"
    echo ""

    DOCKER_DEVICE_FLAGS=()
    case "$VENDOR" in
        nvidia) DOCKER_DEVICE_FLAGS+=(--gpus all)
                # Mount host CUDA for nsys profiler access
                if [ "$PROFILER" = "nsys" ]; then
                    NSYS_BIN=$(which nsys 2>/dev/null || find /opt/cuda* /usr/local/cuda* /packages/cuda* -name nsys -type f 2>/dev/null | head -1)
                    if [ -n "$NSYS_BIN" ]; then
                        CUDA_HOST_DIR=$(dirname "$(dirname "$NSYS_BIN")")
                        DOCKER_DEVICE_FLAGS+=(-v "$CUDA_HOST_DIR:/opt/cuda-host:ro")
                    else
                        echo "WARNING: nsys not found on host; profiler mode may fail" >&2
                    fi
                fi
                ;;
        intel)  DOCKER_DEVICE_FLAGS+=(--device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path) ;;
        amd)    DOCKER_DEVICE_FLAGS+=(--device=/dev/kfd --device=/dev/dri --group-add video) ;;
    esac

    # Persistent Docker volume for the CMake build directory.
    # After the first full build (~10 min), subsequent runs with only
    # one changed .cpp file take ~10 seconds (incremental recompile + relink).
    BUILD_VOLUME="rajaperf-build-${VENDOR}"

    DOCKER_OUTPUT=$(docker run --rm \
        "${DOCKER_DEVICE_FLAGS[@]}" \
        -e SYCL_CACHE_PERSISTENT=1 \
        -v /tmp/sycl-cache:/root/.cache \
        -v "$SCRIPT_DIR":/opt/rajaperf-compare:ro \
        -v "$LEO_ROOT/src":/opt/leo-src:ro \
        -v "$LEO_ROOT/scripts":/opt/leo-scripts:ro \
        -v "${BUILD_VOLUME}":/opt/build-optimized \
        "$DOCKER_IMAGE" \
        -c "
            # Add host CUDA tools to PATH (for nsys profiler)
            [ -d /opt/cuda-host/bin ] && export PATH=/opt/cuda-host/bin:\$PATH

            # Copy optimized source to writable location (original uses container's pre-built binary)
            cp -r /opt/rajaperf-compare/optimized /opt/rajaperf-optimized
            # Copy original source for diff discovery only
            cp -r /opt/rajaperf-compare/original /opt/rajaperf-original
            cp /opt/rajaperf-compare/run_compare.sh /opt/run_compare.sh
            chmod +x /opt/run_compare.sh

            # Sync Leo source if profiling
            if echo '${PASSTHROUGH_ARGS[*]}' | grep -q -- '--profile'; then
                cp -r /opt/leo-src/leo/* /opt/leo/src/leo/ 2>/dev/null || true
                cp /opt/leo-scripts/analyze_benchmark.py /opt/leo/scripts/ 2>/dev/null || true
            fi

            cd /opt
            bash /opt/run_compare.sh --vendor $VENDOR ${PASSTHROUGH_ARGS[*]}
        " 2>&1)

    # Print Docker output (everything except CSV markers)
    echo "$DOCKER_OUTPUT" | grep -v '^=== CSV_'

    # Extract CSVs from Docker output and save to host. Postprocess (drop
    # cold-start passes, compute robust stats) runs inside the container — its
    # python3 is Ubuntu-22.04 stock 3.10+ which has the PEP 585 generics
    # postprocess.py uses. Avoiding host python3 sidesteps RHEL 8 (3.6.8) and
    # x86/ARM venv-mismatch concerns on GH200.
    echo "$DOCKER_OUTPUT" | sed -n '/^=== CSV_RAW_BEGIN ===/,/^=== CSV_RAW_END ===/{ /^===/d; p }' \
        > "$SCRIPT_DIR/rajaperf-compare-raw.csv"
    echo "$DOCKER_OUTPUT" | sed -n '/^=== CSV_SUMMARY_BEGIN ===/,/^=== CSV_SUMMARY_END ===/{ /^===/d; p }' \
        > "$SCRIPT_DIR/rajaperf-compare-summary.csv"
    echo "$DOCKER_OUTPUT" | sed -n '/^=== CSV_CLEAN_BEGIN ===/,/^=== CSV_CLEAN_END ===/{ /^===/d; p }' \
        > "$SCRIPT_DIR/rajaperf-compare-clean.csv"

    RAW_LINES=$(wc -l < "$SCRIPT_DIR/rajaperf-compare-raw.csv")
    CLEAN_LINES=$(wc -l < "$SCRIPT_DIR/rajaperf-compare-clean.csv")
    if [ "$RAW_LINES" -gt 1 ]; then
        echo ""
        echo "  CSV saved to:"
        echo "    $SCRIPT_DIR/rajaperf-compare-raw.csv ($((RAW_LINES - 1)) samples)"
        echo "    $SCRIPT_DIR/rajaperf-compare-summary.csv"
        if [ "$CLEAN_LINES" -gt 1 ]; then
            echo "    $SCRIPT_DIR/rajaperf-compare-clean.csv (cold-start passes dropped, $((CLEAN_LINES - 1)) samples)"
        fi
    fi

    exit 0
fi

# ==================== Inside Docker: paths ====================
# Original: container's pre-built raja-perf.exe (from upstream LLNL/RAJAPerf)
# Optimized: build from our fork source
BIN_ORIGINAL=$(which raja-perf.exe 2>/dev/null || echo "")
if [ -z "$BIN_ORIGINAL" ]; then
    echo "ERROR: raja-perf.exe not found in container. Is this a leo-rajaperf-* image?"
    exit 1
fi

SRC_ORIGINAL="/opt/rajaperf-original"
SRC_OPTIMIZED="/opt/rajaperf-optimized"
BUILD_OPTIMIZED="/opt/build-optimized"

# ==================== Build optimized version ====================
if [ "$SKIP_BUILD" = false ]; then
    echo "============================================"
    echo " Building optimized RAJAPerf ($VENDOR)"
    echo " Original: using container's pre-built binary"
    echo "============================================"
    echo ""

    mkdir -p "$BUILD_OPTIMIZED"
    cd "$BUILD_OPTIMIZED"

    # Only run cmake if not already configured
    if [ ! -f CMakeCache.txt ]; then
        echo "  [optimized] cmake ..."
        case "$VENDOR" in
            intel)
                cmake "$SRC_OPTIMIZED" \
                    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                    -DCMAKE_CXX_COMPILER=icpx \
                    -DCMAKE_C_COMPILER=icx \
                    -DCMAKE_CXX_FLAGS="-fsycl -fsycl-unnamed-lambda" \
                    -DENABLE_SYCL=ON -DENABLE_CUDA=OFF -DENABLE_HIP=OFF -DENABLE_OPENMP=OFF \
                    -DRAJA_ENABLE_SYCL=ON -DRAJA_ENABLE_DESUL_ATOMICS=ON \
                    -DBLT_CXX_STD=c++17 -DCMAKE_CXX_STANDARD=17 \
                    > cmake.log 2>&1
                ;;
            amd)
                ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
                GPU_TARGET="${GPU_TARGET:-gfx942}"
                cmake "$SRC_OPTIMIZED" \
                    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                    -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
                    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/bin/hipcc" \
                    -DENABLE_HIP=ON -DENABLE_OPENMP=OFF -DENABLE_CUDA=OFF \
                    -DRAJA_ENABLE_HIP=ON \
                    -DGPU_TARGETS="$GPU_TARGET" -DAMDGPU_TARGETS="$GPU_TARGET" \
                    -DCMAKE_HIP_ARCHITECTURES="$GPU_TARGET" \
                    -DBLT_CXX_STD=c++17 -DCMAKE_CXX_STANDARD=17 \
                    > cmake.log 2>&1
                ;;
            nvidia)
                GPU_ARCH="${GPU_ARCH:-90}"
                cmake "$SRC_OPTIMIZED" \
                    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                    -DENABLE_CUDA=ON -DENABLE_HIP=OFF -DENABLE_OPENMP=OFF \
                    -DRAJA_ENABLE_CUDA=ON \
                    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
                    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
                    -DCMAKE_CUDA_FLAGS="-lineinfo" \
                    -DBLT_CXX_STD=c++17 -DCMAKE_CXX_STANDARD=17 \
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

BIN_OPTIMIZED="$BUILD_OPTIMIZED/bin/raja-perf.exe"
if [ ! -x "$BIN_OPTIMIZED" ]; then
    echo "ERROR: Optimized binary not found: $BIN_OPTIMIZED"
    echo "       Run without --skip-build to build first."
    exit 1
fi

# ==================== Discover optimized kernels ====================
# Extract kernel names from differing filenames, then resolve to actual
# RAJAPerf kernel names (filenames use POLYBENCH_2MM but registry uses Polybench_2MM)
DIFF_KERNELS=$(diff -rq "$SRC_ORIGINAL/src/" "$SRC_OPTIMIZED/src/" 2>/dev/null | \
    grep -- "-${SUFFIX}.cpp differ" | \
    sed "s|.*/||; s|-${SUFFIX}.cpp differ||" || true)

ALL_KERNELS=$("$BIN_ORIGINAL" --print-kernels 2>&1 | grep -E '^[A-Za-z]' || true)
OPTIMIZED_KERNELS=""
for dk in $DIFF_KERNELS; do
    actual=$(echo "$ALL_KERNELS" | grep -i "${dk}$" | head -1)
    if [ -n "$actual" ]; then
        OPTIMIZED_KERNELS="${OPTIMIZED_KERNELS:+$OPTIMIZED_KERNELS
}$actual"
    else
        OPTIMIZED_KERNELS="${OPTIMIZED_KERNELS:+$OPTIMIZED_KERNELS
}$dk"
    fi
done

if [ -z "$OPTIMIZED_KERNELS" ]; then
    echo "No optimized kernels found (no ${SUFFIX} files differ between original/ and optimized/)"
    exit 0
fi

if [ -n "$KERNEL_FILTER" ]; then
    OPTIMIZED_KERNELS=$(echo "$OPTIMIZED_KERNELS" | grep -i "$KERNEL_FILTER" || true)
    if [ -z "$OPTIMIZED_KERNELS" ]; then
        echo "No optimized kernels match filter '$KERNEL_FILTER'"
        exit 0
    fi
fi

KERNEL_COUNT=$(echo "$OPTIMIZED_KERNELS" | wc -l)

echo "============================================"
echo " RAJAPerf Original vs Optimized ($VENDOR)"
echo " Variant: $VARIANT"
echo " Kernels: $KERNEL_COUNT"
echo " Warmup:  $WARMUP passes (1 invocation, discarded)"
echo " Measure: $NRUNS invocations × $PASSES_PER_RUN passes = $((NRUNS * PASSES_PER_RUN)) samples"
echo "          (each pass internally runs kernel ~50 reps)"
echo " Metric:  min (primary) + mean ± stddev"
echo " Original binary: $BIN_ORIGINAL"
echo " Optimized binary: $BIN_OPTIMIZED"
echo "============================================"
echo ""

# ==================== Run and compare ====================
#
# Methodology (based on GPU microbenchmark best practices):
#
#   1. Warmup: 1 invocation with --npasses $WARMUP (discard output)
#      - Stabilizes GPU frequency, warms caches, JIT compilation
#
#   2. Measurement: $NRUNS invocations, each with --npasses $PASSES_PER_RUN
#      - RAJAPerf internally auto-repeats kernel ~50 times per pass
#      - Each invocation reports avg time across $PASSES_PER_RUN passes
#      - Total kernel executions: $NRUNS × $PASSES_PER_RUN × ~50 reps
#
#   3. Statistics: report min (primary), mean ± stddev (secondary)
#      - Min = best-case steady-state (recommended for GPU benchmarks)
#      - Stddev captures inter-run variance (thermal, DVFS, contention)
#      - Speedup = orig_min / opt_min
#
# Default: 5 warmup + 5×10 = 50 measurement passes → 12 process invocations total
#

# ==================== Vendor GPU profiler functions ====================

# Extract top kernel GPU execution time (nanoseconds) from profiler output
extract_gpu_kernel_ns_nsys() {
    sed -n "/cuda_gpu_kern_sum/,/Executing/p" "$1" \
        | grep -E "^\s+[0-9]" | head -1 | awk '{print $2}'
}

extract_gpu_kernel_ns_rocprofv3() {
    grep "KERNEL_DISPATCH" "$1" \
        | grep -v "__amd_rocclr\|daxpy" \
        | sort -t'|' -k5 -rn \
        | head -1 \
        | awk -F'|' '{gsub(/^ +| +$/,"",$5); print $5}'
}

extract_gpu_kernel_ns_unitrace() {
    # unitrace Device Timing Summary format:
    # "long kernel name with, commas",  Calls,  Time (ns), ...
    # The kernel name is quoted and may contain commas.
    # Strategy: strip the quoted name first, then parse remaining fields.
    sed -n '/^== L0 Backend ==/,/^=== Kernel Properties ===/p' "$1" \
        | grep -E '^\s*"' \
        | grep -v "zeCommandListAppendMemoryCopy\|DAXPY\|daxpy" \
        | head -1 \
        | sed 's/^[^"]*"[^"]*",//' \
        | awk -F',' '{gsub(/^ +| +$/,"",$2); print $2}'
}

# Run a kernel with vendor GPU profiler.
# Returns "min mean stddev n" in milliseconds, or "" on failure.
# Also appends raw timings to CSV_FILE (if set).
run_rajaperf_kernel_profiled() {
    local bin="$1"
    local kernel="$2"
    local label="$3"  # "original" or "optimized"

    # Warmup: single invocation, discard output
    "$bin" -k "$kernel" -v "$VARIANT" --npasses "$WARMUP" \
        --show-progress > /dev/null 2>&1 || true

    # Measurement: NRUNS invocations, each profiled
    local times_ns=()
    local run_idx=0
    for (( i=1; i<=NRUNS; i++ )); do
        run_idx=$((run_idx + 1))
        local logfile="/tmp/prof_${label}_${run_idx}.log"
        rm -f /tmp/*.nsys-rep /tmp/*.sqlite 2>/dev/null || true

        case "$PROFILER" in
            nsys)
                nsys profile --stats=true -f true -o "/tmp/prof_${label}_${run_idx}" \
                    "$bin" -k "$kernel" -v "$VARIANT" --npasses "$PASSES_PER_RUN" --checkrun 1 \
                    > "$logfile" 2>&1 || true
                ;;
            rocprofv3)
                rocprofv3 --kernel-trace --stats -S -- \
                    "$bin" -k "$kernel" -v "$VARIANT" --npasses "$PASSES_PER_RUN" --checkrun 1 \
                    > "$logfile" 2>&1 || true
                ;;
            unitrace)
                unitrace -d \
                    "$bin" -k "$kernel" -v "$VARIANT" --npasses "$PASSES_PER_RUN" --checkrun 1 \
                    > "$logfile" 2>&1 || true
                ;;
        esac

        local ns
        ns=$(extract_gpu_kernel_ns_${PROFILER} "$logfile")
        if [ -n "$ns" ] && [ "$ns" -gt 0 ] 2>/dev/null; then
            times_ns+=("$ns")
            if [ -n "$CSV_FILE" ]; then
                local ms
                ms=$(awk "BEGIN { printf \"%.6f\", $ns / 1000000.0 }")
                local sec
                sec=$(awk "BEGIN { printf \"%.9f\", $ns / 1000000000.0 }")
                echo "$kernel,$label,$run_idx,1,$sec,$ms" >> "$CSV_FILE"
            fi
        fi
    done

    local n=${#times_ns[@]}
    if [ "$n" -eq 0 ]; then
        echo ""
        return
    fi

    # Compute min, mean, stddev in ms
    printf '%s\n' "${times_ns[@]}" | awk -v n="$n" '
        BEGIN { min = 1e30 }
        {
            x = $1 / 1000000.0
            sum += x; sumsq += x*x
            if (x < min) min = x
        }
        END {
            mean = sum / n
            if (n > 1) { stddev = sqrt((sumsq - sum*sum/n) / (n-1)) }
            else       { stddev = 0 }
            printf "%.6f %.6f %.6f %d", min, mean, stddev, n
        }'
}

# ==================== RAJAPerf built-in timer ====================

# Run a kernel: warmup once, then collect per-pass timings.
# Returns "min mean stddev n" in milliseconds, or "" on failure.
# Also appends raw per-pass timings to CSV_FILE (if set).
#
# With --show-progress, RAJAPerf prints one line per pass:
#   "Running block_64  tuning -- 5.81e-05 sec. x 50 rep. PASSED checksum"
# We collect ALL per-pass timings across NRUNS invocations.
run_rajaperf_kernel() {
    local bin="$1"
    local kernel="$2"
    local label="$3"  # "original" or "optimized"

    # Warmup: single invocation, discard output
    "$bin" -k "$kernel" -v "$VARIANT" --npasses "$WARMUP" \
        --show-progress > /dev/null 2>&1 || true

    # Measurement: NRUNS invocations × PASSES_PER_RUN passes each
    # Collect every per-pass timing value
    local times=()
    local run_idx=0
    for (( i=1; i<=NRUNS; i++ )); do
        local out
        out=$("$bin" -k "$kernel" -v "$VARIANT" --npasses "$PASSES_PER_RUN" \
              --show-progress 2>&1) || true
        run_idx=$((run_idx + 1))
        local pass_idx=0
        # Extract all per-pass timings (one PASSED line per pass)
        while IFS= read -r sec; do
            if [ -n "$sec" ]; then
                times+=("$sec")
                pass_idx=$((pass_idx + 1))
                # Append to CSV: kernel,label,run,pass,time_sec,time_ms
                if [ -n "$CSV_FILE" ]; then
                    local ms
                    ms=$(awk "BEGIN { printf \"%.6f\", $sec * 1000 }")
                    echo "$kernel,$label,$run_idx,$pass_idx,$sec,$ms" >> "$CSV_FILE"
                fi
            fi
        done < <(echo "$out" | grep 'PASSED' | \
                 grep -oP 'tuning -- \K[0-9.eE+-]+(?= sec\.)')
    done

    local n=${#times[@]}
    if [ "$n" -eq 0 ]; then
        echo ""
        return
    fi

    # Compute min, mean, stddev in ms
    printf '%s\n' "${times[@]}" | awk -v n="$n" '
        BEGIN { min = 1e30 }
        {
            x = $1 * 1000
            sum += x; sumsq += x*x
            if (x < min) min = x
        }
        END {
            mean = sum / n
            if (n > 1) { stddev = sqrt((sumsq - sum*sum/n) / (n-1)) }
            else       { stddev = 0 }
            printf "%.6f %.6f %.6f %d", min, mean, stddev, n
        }'
}

# CSV output: raw per-pass timings for plotting
CSV_FILE="/opt/rajaperf-compare-raw.csv"
echo "kernel,version,run,pass,time_sec,time_ms" > "$CSV_FILE"

printf "%-30s %18s %18s %10s %10s\n" "Kernel" "Orig(ms)" "Opt(ms)" "Speedup" "(by mean)"
printf "%-30s %18s %18s %10s %10s\n" "------" "--------" "-------" "-------" "--------"

while IFS= read -r KERNEL; do
    [ -z "$KERNEL" ] && continue
    printf "%-30s " "$KERNEL"

    if [ -n "$PROFILER" ]; then
        ORIG_RESULT=$(run_rajaperf_kernel_profiled "$BIN_ORIGINAL" "$KERNEL" "original")
        OPT_RESULT=$(run_rajaperf_kernel_profiled "$BIN_OPTIMIZED" "$KERNEL" "optimized")
    else
        ORIG_RESULT=$(run_rajaperf_kernel "$BIN_ORIGINAL" "$KERNEL" "original")
        OPT_RESULT=$(run_rajaperf_kernel "$BIN_OPTIMIZED" "$KERNEL" "optimized")
    fi

    ORIG_MIN=$(echo "$ORIG_RESULT" | awk '{print $1}')
    ORIG_MEAN=$(echo "$ORIG_RESULT" | awk '{print $2}')
    ORIG_STD=$(echo "$ORIG_RESULT" | awk '{print $3}')
    ORIG_N=$(echo "$ORIG_RESULT" | awk '{print $4}')
    OPT_MIN=$(echo "$OPT_RESULT" | awk '{print $1}')
    OPT_MEAN=$(echo "$OPT_RESULT" | awk '{print $2}')
    OPT_STD=$(echo "$OPT_RESULT" | awk '{print $3}')
    OPT_N=$(echo "$OPT_RESULT" | awk '{print $4}')

    if [ -n "$ORIG_MIN" ] && [ -n "$OPT_MIN" ]; then
        SPEEDUP_MIN=$(awk "BEGIN { printf \"%.2fx\", $ORIG_MIN / $OPT_MIN }")
        SPEEDUP_MEAN=$(awk "BEGIN { printf \"%.2fx\", $ORIG_MEAN / $OPT_MEAN }")
        printf "%7s±%-9s %7s±%-9s %10s %10s\n" \
            "$(printf '%.4f' "$ORIG_MIN")" "$(printf '%.4f' "$ORIG_STD")" \
            "$(printf '%.4f' "$OPT_MIN")" "$(printf '%.4f' "$OPT_STD")" \
            "$SPEEDUP_MIN" "$SPEEDUP_MEAN"
    elif [ -z "$ORIG_MIN" ]; then
        printf "%18s %18s %10s %10s\n" "FAILED" "-" "-" "-"
    else
        printf "%7s±%-9s %18s %10s %10s\n" \
            "$(printf '%.4f' "$ORIG_MIN")" "$(printf '%.4f' "$ORIG_STD")" "FAILED" "-" "-"
    fi
done <<< "$OPTIMIZED_KERNELS"

# Generate summary CSV from raw data
CSV_SUMMARY="/opt/rajaperf-compare-summary.csv"
echo "kernel,orig_min_ms,orig_mean_ms,orig_stddev_ms,orig_n,opt_min_ms,opt_mean_ms,opt_stddev_ms,opt_n,speedup_min,speedup_mean" > "$CSV_SUMMARY"
awk -F, 'NR>1 {
    key = $1 SUBSEP $2
    n[key]++; sum[key] += $6; sumsq[key] += $6*$6
    if (!(key in min) || $6 < min[key]) min[key] = $6
}
END {
    # Collect unique kernels preserving order
    for (k in n) {
        split(k, parts, SUBSEP)
        kernel = parts[1]; ver = parts[2]
        mean[k] = sum[k] / n[k]
        if (n[k] > 1) stddev[k] = sqrt((sumsq[k] - sum[k]*sum[k]/n[k]) / (n[k]-1))
        else stddev[k] = 0
        kernels[kernel] = 1
    }
    for (kernel in kernels) {
        ok = kernel SUBSEP "original"
        nk = kernel SUBSEP "optimized"
        if ((ok in n) && (nk in n)) {
            sp_min = min[ok] / min[nk]
            sp_mean = mean[ok] / mean[nk]
            printf "%s,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%d,%.4f,%.4f\n", \
                kernel, min[ok], mean[ok], stddev[ok], n[ok], \
                min[nk], mean[nk], stddev[nk], n[nk], sp_min, sp_mean
        }
    }
}' "$CSV_FILE" >> "$CSV_SUMMARY"

echo ""
echo "  Raw timings: $CSV_FILE ($(tail -n+2 "$CSV_FILE" | wc -l) samples)"
echo "  Summary:     $CSV_SUMMARY"

# Print CSVs to stdout (captured by tee on the host side)
echo ""
echo "=== CSV_RAW_BEGIN ==="
cat "$CSV_FILE"
echo "=== CSV_RAW_END ==="
echo ""
echo "=== CSV_SUMMARY_BEGIN ==="
cat "$CSV_SUMMARY"
echo "=== CSV_SUMMARY_END ==="

# Post-process inside the container: drop cold-start passes and compute robust
# statistics. The container's python3 is Ubuntu-22.04 stock (3.10+), which has
# the PEP 585 generics that postprocess.py uses. Doing this in-container means
# the host doesn't need a modern Python.
if [ -f /opt/rajaperf-compare/postprocess.py ] && command -v python3 >/dev/null 2>&1; then
    CLEAN_FILE="/opt/rajaperf-compare-clean.csv"
    if python3 /opt/rajaperf-compare/postprocess.py "$CSV_FILE" --output "$CLEAN_FILE" >/dev/null 2>&1 \
        && [ -s "$CLEAN_FILE" ]; then
        echo ""
        echo "=== CSV_CLEAN_BEGIN ==="
        cat "$CLEAN_FILE"
        echo "=== CSV_CLEAN_END ==="
    fi
fi

echo ""
echo "  Source diff:"
diff -rq "$SRC_ORIGINAL/src/" "$SRC_OPTIMIZED/src/" 2>/dev/null | \
    grep -- "-${SUFFIX}.cpp differ" | sed 's|^|    |' || true

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
    LEO_ARCH[amd]="amd"; LEO_ARCH[nvidia]="nvidia"; LEO_ARCH[intel]="intel"

    EVENT="${HPCRUN_EVENTS[$VENDOR]}"
    ARCH="${LEO_ARCH[$VENDOR]}"
    PROFILE_DIR="/opt/profiles"
    mkdir -p "$PROFILE_DIR"

    echo ""
    echo "============================================"
    echo " HPCToolkit Profiling + Leo Analysis"
    echo "============================================"

    while IFS= read -r KERNEL; do
        [ -z "$KERNEL" ] && continue
        echo ""
        echo "--- $KERNEL ---"

        for label in original optimized; do
            if [ "$label" = "original" ]; then BIN="$BIN_ORIGINAL"; else BIN="$BIN_OPTIMIZED"; fi
            MEAS_DIR="$PROFILE_DIR/${KERNEL}-${label}-measurements"
            DB_DIR="$PROFILE_DIR/${KERNEL}-${label}-database"
            LEO_OUT="$PROFILE_DIR/${KERNEL}-${label}-leo.txt"

            echo "  [$label] hpcrun..."
            rm -rf "$MEAS_DIR" "$DB_DIR"
            hpcrun -o "$MEAS_DIR" -e "$EVENT" \
                "$BIN" -k "$KERNEL" -v "$VARIANT" --npasses 1 2>&1 | tail -1

            echo "  [$label] hpcstruct..."
            hpcstruct --gpucfg yes "$MEAS_DIR" 2>&1 | tail -1 || true

            echo "  [$label] hpcprof..."
            hpcprof -o "$DB_DIR" "$MEAS_DIR" 2>&1 | tail -1

            echo "  [$label] Leo analysis..."
            cd /opt/leo 2>/dev/null || cd "$LEO_ROOT"
            uv run python scripts/analyze_benchmark.py \
                "$MEAS_DIR" --arch "$ARCH" --top-n 1 2>&1 | tee "$LEO_OUT"
        done
    done <<< "$OPTIMIZED_KERNELS"
fi
