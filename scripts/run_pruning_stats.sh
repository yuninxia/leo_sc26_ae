#!/bin/bash
# Run Leo with --debug on all per-kernel data across 3 vendors.
# Uses the universal Docker container (nvdisasm + libged.so).
# Extracts pruning statistics to a log file for analysis.
#
# Usage: bash scripts/run_pruning_stats.sh

set -e

LEO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$LEO_ROOT/tests/data/pc/per-kernel"
LOGFILE="$LEO_ROOT/tests/data/pc/per-kernel/pruning-stats.log"
IMAGE="leo-base-universal:latest"

# Vendor → arch mapping (must match run_leo_all_kernels.sh)
declare -A VENDOR_ARCH
VENDOR_ARCH[amd]="amd"
VENDOR_ARCH[nvidia-arm]="nvidia"
VENDOR_ARCH[intel]="intel"

# Discover all kernels and measurements, build inner script
INNER_SCRIPT=$(mktemp)
trap "rm -f $INNER_SCRIPT" EXIT

cat > "$INNER_SCRIPT" <<'HEADER'
#!/bin/bash
set -e
cd /opt/leo
export UV_PROJECT_ENVIRONMENT=/tmp/leo-venv

# Overlay host source changes (latency pruning fix, cpp parser default, etc.)
cp -r /opt/leo-host-src/leo/* src/leo/ 2>/dev/null || true

uv run python -c "print('Leo environment ready')" 2>&1

TOTAL=0
DONE=0
FAIL=0
HEADER

for KERNEL_DIR in "$DATA_DIR"/*/; do
    KERNEL=$(basename "$KERNEL_DIR")
    [[ ! -d "$KERNEL_DIR" ]] && continue

    for VENDOR in amd nvidia-arm intel; do
        ARCH="${VENDOR_ARCH[$VENDOR]}"
        # Find measurements directory
        MEAS_DIR=$(ls -td "$DATA_DIR/$KERNEL/$VENDOR"/hpctoolkit-*-measurements 2>/dev/null | head -1)
        if [[ -z "$MEAS_DIR" ]]; then
            continue
        fi
        MEAS_NAME=$(basename "$MEAS_DIR")

        cat >> "$INNER_SCRIPT" <<EOF
TOTAL=\$((TOTAL + 1))
echo "[\$TOTAL] $KERNEL / $VENDOR ($ARCH)"
OUTPUT=\$(UV_PROJECT_ENVIRONMENT=/tmp/leo-venv uv run python scripts/analyze_benchmark.py "/data/per-kernel/$KERNEL/$VENDOR/$MEAS_NAME" --arch $ARCH --top-n 1 --debug 2>&1) || true
echo "\$OUTPUT" | grep -E "Initial graph|Stage [0-9]|Final graph" || true
if echo "\$OUTPUT" | grep -q "Error\|error\|FAILED\|Traceback"; then
    echo "  ERROR"
    echo "\$OUTPUT" | grep -E "Error|error|FAILED|Traceback" | head -3
    FAIL=\$((FAIL + 1))
else
    DONE=\$((DONE + 1))
fi
echo ""
EOF
    done
done

cat >> "$INNER_SCRIPT" <<'FOOTER'
echo "=== Summary ==="
echo "Total: $TOTAL, Done: $DONE, Failed: $FAIL"
FOOTER

chmod +x "$INNER_SCRIPT"

echo "=== Leo Pruning Statistics (latency pruning DISABLED) ===" | tee "$LOGFILE"
echo "Date: $(date)" | tee -a "$LOGFILE"
echo "Image: $IMAGE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Run everything in a single container
# Mount host src at /opt/leo-host-src (read-only), overlay into image's /opt/leo at runtime
docker run --rm \
    -v "$DATA_DIR:/data/per-kernel:ro" \
    -v "$LEO_ROOT/src:/opt/leo-host-src:ro" \
    -v "$INNER_SCRIPT:/tmp/run.sh:ro" \
    --entrypoint='' \
    "$IMAGE" \
    bash /tmp/run.sh 2>&1 | tee -a "$LOGFILE"

echo ""
echo "Full log: $LOGFILE"
