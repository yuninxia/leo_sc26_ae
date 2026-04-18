#!/bin/bash
# Per-kernel mul_mat_q benchmark for llama.cpp on AMD MI300A
# Measures baseline vs optimized (tile 128→64 + direct store) using rocprof
#
# Usage: bash benchmarks/llamacpp/run_perkernel_amd.sh [--runs N] [--prompt N]
#
# Reports per-kernel mul_mat_q total time (all instantiations + stream_k_fixup)
# Paper value: 1.12 ± 0.04× (pp512, 10 runs)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NRUNS=10
PROMPT=512
MODEL="${MODEL_DIR:-$HOME/models}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
OPT_MMQ="$SCRIPT_DIR/optimized/mmq_amd_optimized.cuh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)   NRUNS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== llama.cpp per-kernel benchmark (pp${PROMPT}, ${NRUNS} runs) ==="
if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$OPT_MMQ" ]; then echo "ERROR: optimized mmq.cuh not found: $OPT_MMQ"; exit 1; fi

# Write container script to temp file (avoids nested quoting)
CONTAINER_SCRIPT=$(mktemp /tmp/llamacpp_perkernel_XXXXX.sh)
cat > "$CONTAINER_SCRIPT" << 'INNEREOF'
#!/bin/bash
set -e
NRUNS=$1
PROMPT=$2

parse_mmq_ns() {
    python3 -c "
import csv, sys
total = 0
with open(sys.argv[1]) as f:
    for row in csv.reader(f):
        if len(row) >= 3 and 'mul_mat_q' in row[0]:
            total += int(row[2])
print(total)
" "$1"
}

echo "PHASE:baseline"
for i in $(seq 1 $NRUNS); do
    rocprof --stats -o /tmp/baseline_${i}.csv \
        llama-bench -m /workspace/model.gguf -p $PROMPT -n 0 -ngl 99 -r 1 \
        > /dev/null 2>&1
    ns=$(parse_mmq_ns /tmp/baseline_${i}.stats.csv)
    echo "DATA:$ns"
done

echo "PHASE:build"
cd /opt/llama.cpp
cp /workspace/mmq_optimized.cuh ggml/src/ggml-cuda/mmq.cuh
cmake --build build -j$(nproc) > /dev/null 2>&1
cp build/bin/llama-bench /opt/llamacpp/bin/
cp build/bin/libggml*.so* /opt/llamacpp/lib/
ldconfig
echo "PHASE:done"

echo "PHASE:optimized"
for i in $(seq 1 $NRUNS); do
    rocprof --stats -o /tmp/optimized_${i}.csv \
        llama-bench -m /workspace/model.gguf -p $PROMPT -n 0 -ngl 99 -r 1 \
        > /dev/null 2>&1
    ns=$(parse_mmq_ns /tmp/optimized_${i}.stats.csv)
    echo "DATA:$ns"
done
INNEREOF
chmod +x "$CONTAINER_SCRIPT"

# Run container, capture only DATA/PHASE lines
RAW=$(docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$MODEL":/workspace/model.gguf:ro \
    -v "$OPT_MMQ":/workspace/mmq_optimized.cuh:ro \
    -v "$CONTAINER_SCRIPT":/workspace/run.sh:ro \
    --entrypoint bash leo-llamacpp-amd /workspace/run.sh "$NRUNS" "$PROMPT" 2>&1 \
    | grep "^PHASE:\|^DATA:")

rm -f "$CONTAINER_SCRIPT"

# Parse and compute summary
python3 << PYEOF
import statistics, math, sys

lines = """$RAW""".strip().split('\n')

baseline_ns = []
optimized_ns = []
current = None
for line in lines:
    if line == 'PHASE:baseline':
        current = 'baseline'
    elif line == 'PHASE:optimized':
        current = 'optimized'
    elif line.startswith('DATA:') and current:
        val = int(line.split(':')[1])
        if current == 'baseline':
            baseline_ns.append(val)
        elif current == 'optimized':
            optimized_ns.append(val)

if not baseline_ns or not optimized_ns:
    print('ERROR: no data collected')
    sys.exit(1)

def fmt(vals, label):
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    ms = [round(v/1e6, 2) for v in sorted(vals)]
    print(f'{label}: {mean/1e6:.2f} ± {std/1e6:.2f} ms  (n={len(vals)})')
    print(f'  runs (ms): {ms}')
    return mean, std

print(f'=== Results (pp$PROMPT, $NRUNS runs) ===')
print()
b_mean, b_std = fmt(baseline_ns, 'Baseline mul_mat_q')
o_mean, o_std = fmt(optimized_ns, 'Optimized mul_mat_q')

speedup = b_mean / o_mean
speedup_std = speedup * math.sqrt((b_std/b_mean)**2 + (o_std/o_mean)**2)
print()
print(f'Per-kernel speedup: {speedup:.2f} ± {speedup_std:.2f}x')
PYEOF
