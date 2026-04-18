#!/bin/bash
# Run Leo analysis (top-N) on all per-kernel profiling data across all vendors.
# Uses the universal Docker container which has all 3 disassemblers.
#
# Usage:
#   ./run_leo_all_kernels.sh                        # Default: top-1
#   ./run_leo_all_kernels.sh --top-n 2              # Top-2
#   ./run_leo_all_kernels.sh --kernels Basic_DAXPY  # Single kernel
#
# Output: results/leo-all-kernels.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$LEO_ROOT/tests/data/pc/per-kernel"
LOGFILE="$LEO_ROOT/results/leo-all-kernels.log"
IMAGE="leo-base-universal:latest"

TOP_N=1
SELECTED_KERNELS=()
VENDORS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --top-n)
            TOP_N="$2"; shift 2 ;;
        --kernels|-k)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_KERNELS+=("$1")
                shift
            done
            ;;
        --vendors|-v)
            shift
            VENDORS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                VENDORS+=("$1")
                shift
            done
            ;;
        --log)
            LOGFILE="$2"; shift 2 ;;
        --image)
            IMAGE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If no kernels specified, use all
if [[ ${#SELECTED_KERNELS[@]} -eq 0 ]]; then
    for d in "$DATA_DIR"/*/; do
        SELECTED_KERNELS+=("$(basename "$d")")
    done
fi

# Vendor → arch mapping
declare -A VENDOR_ARCH
VENDOR_ARCH[amd]="amd"
VENDOR_ARCH[nvidia-arm]="nvidia"
VENDOR_ARCH[intel]="intel"

# Default vendors if not overridden by --vendors
if [[ ${#VENDORS[@]} -eq 0 ]]; then
    VENDORS=(amd nvidia-arm intel)
fi
TOTAL=$((${#SELECTED_KERNELS[@]} * ${#VENDORS[@]}))

mkdir -p "$(dirname "$LOGFILE")"

echo "============================================" | tee "$LOGFILE"
echo " Leo Analysis: All Kernels x All Vendors"   | tee -a "$LOGFILE"
echo "============================================" | tee -a "$LOGFILE"
echo "Kernels:  ${#SELECTED_KERNELS[@]}"           | tee -a "$LOGFILE"
echo "Vendors:  ${VENDORS[*]}"                     | tee -a "$LOGFILE"
echo "Top-N:    $TOP_N"                             | tee -a "$LOGFILE"
echo "Total:    $TOTAL runs"                        | tee -a "$LOGFILE"
echo "Image:    $IMAGE"                             | tee -a "$LOGFILE"
echo "Log:      $LOGFILE"                           | tee -a "$LOGFILE"
echo ""                                             | tee -a "$LOGFILE"

# Build the analysis commands for all kernels into a single script
# so we only start one container and install packages once.
INNER_SCRIPT=$(mktemp)
trap "rm -f $INNER_SCRIPT" EXIT

cat > "$INNER_SCRIPT" <<'HEADER'
#!/bin/bash
set -e

# Setup Leo once
cp -r /opt/leo/src /tmp/leo-src 2>/dev/null || true
cd /opt/leo
export UV_PROJECT_ENVIRONMENT=/tmp/leo-venv

# Install packages once
uv run python -c "print('Leo environment ready')" 2>&1

COUNT=0
FAIL=0
SKIP=0
HEADER

for KERNEL in "${SELECTED_KERNELS[@]}"; do
    for VENDOR in "${VENDORS[@]}"; do
        ARCH="${VENDOR_ARCH[$VENDOR]}"
        MEAS_NAME=$(ls -td "$DATA_DIR/$KERNEL/$VENDOR"/hpctoolkit-*-measurements 2>/dev/null | head -1 | xargs basename 2>/dev/null)
        if [[ -z "$MEAS_NAME" ]]; then
            MEAS_NAME=$(ls -td "$DATA_DIR/$KERNEL/$VENDOR"/hpctoolkit-*-measurements-* 2>/dev/null | head -1 | xargs basename 2>/dev/null)
        fi

        if [[ -z "$MEAS_NAME" ]]; then
            cat >> "$INNER_SCRIPT" <<EOF
COUNT=\$((COUNT + 1))
echo "[\$COUNT/$TOTAL] SKIP $KERNEL/$VENDOR (no measurements)"
SKIP=\$((SKIP + 1))
EOF
        else
            cat >> "$INNER_SCRIPT" <<EOF
COUNT=\$((COUNT + 1))
echo "[\$COUNT/$TOTAL] $KERNEL / $VENDOR"
echo "----------------------------------------------------------------------------------------------------"
if UV_PROJECT_ENVIRONMENT=/tmp/leo-venv uv run python scripts/analyze_benchmark.py "/data/per-kernel/$KERNEL/$VENDOR/$MEAS_NAME" --arch $ARCH --top-n $TOP_N 2>&1; then
    echo "  OK"
else
    echo "  FAILED"
    FAIL=\$((FAIL + 1))
fi
echo ""
EOF
        fi
    done
done

cat >> "$INNER_SCRIPT" <<'FOOTER'
echo ""
echo "============================================"
echo " Summary: $((COUNT - FAIL - SKIP)) ok, $SKIP skipped, $FAIL failed (of $COUNT)"
echo "============================================"
FOOTER

chmod +x "$INNER_SCRIPT"

# Run everything in a single container
docker run --rm \
    -v "$DATA_DIR:/data/per-kernel:ro" \
    -v "$LEO_ROOT:/opt/leo-host:ro" \
    -v "$INNER_SCRIPT:/tmp/run.sh:ro" \
    --entrypoint='' \
    "$IMAGE" \
    bash -c "
        cd /opt/leo
        export UV_PROJECT_ENVIRONMENT=/tmp/leo-venv
        bash /tmp/run.sh
    " 2>&1 | tee -a "$LOGFILE"

echo ""
echo "Full log: $LOGFILE"
