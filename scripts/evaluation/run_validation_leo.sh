#!/bin/bash
# Step 2: Run Leo analysis on all validation databases using the universal container.
# Prerequisites: run_validation.sh must have completed (databases in results/validation/).
#
# Usage:
#   ./run_validation_leo.sh                    # Analyze all available databases
#   ./run_validation_leo.sh --kernels Apps_LTIMES Apps_MASS3DEA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$LEO_ROOT/results/validation"
IMAGE="leo-base-universal:latest"

declare -A GPU_ARCH
GPU_ARCH[amd]="mi300"
GPU_ARCH[nvidia]="h100"
GPU_ARCH[intel]="pvc"

# Parse arguments
FILTER_KERNELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --kernels|-k)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FILTER_KERNELS+=("$1")
                shift
            done
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " Leo Validation Analysis (Step 2)"
echo " Container: $IMAGE"
echo "============================================"
echo ""

TOTAL=0
DONE=0
FAIL=0

for kernel_dir in "$RESULTS_BASE"/*/; do
    kernel=$(basename "$kernel_dir")
    [[ "$kernel" == .* ]] && continue

    # Filter if specified
    if [[ ${#FILTER_KERNELS[@]} -gt 0 ]]; then
        match=false
        for fk in "${FILTER_KERNELS[@]}"; do
            [[ "$kernel" == "$fk" ]] && match=true
        done
        $match || continue
    fi

    for vendor_dir in "$kernel_dir"/*/; do
        vendor=$(basename "$vendor_dir")
        arch="${GPU_ARCH[$vendor]}"
        [[ -z "$arch" ]] && continue

        for label in original optimized; do
            data_dir="$vendor_dir/$label"
            meas_dir=$(ls -d "$data_dir"/hpctoolkit-*-measurements 2>/dev/null | head -1)
            [[ -z "$meas_dir" ]] && continue

            output_file="$data_dir/leo_output.txt"
            TOTAL=$((TOTAL + 1))

            # Skip if already analyzed
            if [[ -f "$output_file" && -s "$output_file" ]] && grep -q "STALL ANALYSIS" "$output_file" 2>/dev/null; then
                echo "[skip] $kernel/$vendor/$label"
                DONE=$((DONE + 1))
                continue
            fi

            echo "[run]  $kernel/$vendor/$label (arch=$arch)"

            # Run Leo inside universal container (has all disassemblers)
            docker run --rm \
                -v "$data_dir":/data:ro \
                -v "$LEO_ROOT":/opt/leo-host:ro \
                --entrypoint bash \
                "$IMAGE" \
                -c "
                    cd /opt/leo
                    uv run python scripts/analyze_benchmark.py \
                        /data/$(basename "$meas_dir") \
                        --arch $arch --top-n 2 2>&1
                " > "$output_file" 2>&1

            if grep -q "STALL ANALYSIS" "$output_file" 2>/dev/null; then
                DONE=$((DONE + 1))
            else
                echo "       FAILED (see $output_file)"
                FAIL=$((FAIL + 1))
            fi
        done
    done
done

echo ""
echo "============================================"
echo " Summary: $DONE done, $FAIL failed, $TOTAL total"
echo "============================================"
echo ""
echo "Next step: uv run python scripts/plot_validation_paper.py"
