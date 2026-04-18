#!/bin/bash
# Compare LEO analysis outputs: stored (old) vs current code (new).
# Runs current LEO code inside leo-base-universal Docker container
# against pre-profiled measurement data, then diffs with stored outputs.
#
# Usage:
#   bash scripts/compare_leo_outputs.sh                    # All available results
#   bash scripts/compare_leo_outputs.sh results/amd-xsbench-*  # Specific result dir

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Collect result dirs to compare
if [ $# -gt 0 ]; then
    RESULT_DIRS=("$@")
else
    RESULT_DIRS=()
    for dir in "$LEO_ROOT"/results/*/; do
        if [ -f "$dir/leo_output.txt" ]; then
            # Find measurements dir
            meas=$(find "$dir" -maxdepth 1 -name "hpctoolkit-*-measurements" -type d | head -1)
            if [ -n "$meas" ]; then
                RESULT_DIRS+=("$dir")
            fi
        fi
    done
fi

if [ ${#RESULT_DIRS[@]} -eq 0 ]; then
    echo "No result directories with leo_output.txt + measurements found."
    exit 1
fi

echo "Found ${#RESULT_DIRS[@]} result(s) to compare."
echo ""

# Detect arch from directory name
detect_arch() {
    local dir="$1"
    if echo "$dir" | grep -qi "nvidia\|gilgamesh\|hopper\|athena"; then
        # Check for specific GPU
        if echo "$dir" | grep -qi "hopper\|gh200"; then
            echo "gh200"
        else
            echo "h100"
        fi
    elif echo "$dir" | grep -qi "amd\|odyssey\|mi300"; then
        echo "mi300"
    elif echo "$dir" | grep -qi "intel\|headroom\|pvc"; then
        echo "pvc"
    else
        echo "h100"  # default
    fi
}

PASS=0
DIFF=0
FAIL=0

for result_dir in "${RESULT_DIRS[@]}"; do
    result_dir="${result_dir%/}"
    name="$(basename "$result_dir")"
    old_output="$result_dir/leo_output.txt"
    meas_dir=$(find "$result_dir" -maxdepth 1 -name "hpctoolkit-*-measurements" -type d | head -1)
    arch=$(detect_arch "$result_dir")

    if [ ! -f "$old_output" ] || [ -z "$meas_dir" ]; then
        continue
    fi

    # Get top-n from old output
    top_n=$(grep "Top N:" "$old_output" | awk '{print $NF}')
    [ -z "$top_n" ] && top_n=2

    echo "================================================================"
    echo "  $name (arch=$arch, top-n=$top_n)"
    echo "================================================================"

    # Run new LEO in Docker
    new_output=$(docker run --rm \
        -v "$LEO_ROOT":/opt/leo-host:ro \
        --entrypoint bash leo-base-universal -c "
cd /opt/leo
cp -r /opt/leo-host/src/leo/* src/leo/
cp /opt/leo-host/src/leo/analyzer.py src/leo/
uv sync > /dev/null 2>&1
uv run python scripts/analyze_benchmark.py /opt/leo-host/$meas_dir --arch $arch --top-n $top_n 2>&1
" 2>&1)

    if [ $? -ne 0 ]; then
        echo "  FAILED to run new analysis"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Extract kernel #1 sections for comparison (skip paths and build lines)
    old_k1=$(sed -n '/KERNEL #1/,/KERNEL #2/ p' "$old_output" | head -n -1)
    new_k1=$(echo "$new_output" | sed -n '/KERNEL #1/,/KERNEL #2/ p' | head -n -1)

    # Compare stall chains
    old_chains=$(echo "$old_k1" | grep "<--")
    new_chains=$(echo "$new_k1" | grep "<--")

    if [ "$old_chains" = "$new_chains" ]; then
        echo "  Stall chains: IDENTICAL"
    else
        echo "  Stall chains: DIFFERENT"
        echo "  --- OLD ---"
        echo "$old_chains" | head -5
        echo "  --- NEW ---"
        echo "$new_chains" | head -5
    fi

    # Compare dependency chains
    old_deps=$(echo "$old_k1" | sed -n '/DEPENDENCY CHAINS/,/---/ p' | grep "←")
    new_deps=$(echo "$new_k1" | sed -n '/DEPENDENCY CHAINS/,/---/ p' | grep "←")

    if [ "$old_deps" = "$new_deps" ]; then
        echo "  Dep chains:   IDENTICAL"
    else
        echo "  Dep chains:   DIFFERENT"
        echo "  --- OLD ---"
        echo "$old_deps" | head -3
        echo "  --- NEW ---"
        echo "$new_deps" | head -3
    fi

    # Overall verdict
    if [ "$old_chains" = "$new_chains" ] && [ "$old_deps" = "$new_deps" ]; then
        echo "  Result:       PASS (no regression)"
        PASS=$((PASS + 1))
    else
        echo "  Result:       DIFF (review needed)"
        DIFF=$((DIFF + 1))
    fi
    echo ""
done

echo "================================================================"
echo "  Summary: $PASS pass, $DIFF diff, $FAIL fail"
echo "================================================================"
