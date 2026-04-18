#!/bin/bash
# Kripke LTimes optimization benchmark — NVIDIA H100
# Usage: ./run_nvidia.sh [nruns]
# Requires: leo-kripke-nvidia Docker image + NVIDIA GPU
set -e
NRUNS=${1:-5}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== NVIDIA Kripke LTimes Benchmark ($NRUNS runs) ==="

echo ""
echo "--- BASELINE ---"
docker run --rm --gpus all --entrypoint /bin/bash leo-kripke-nvidia -c "
for i in \$(seq 1 $NRUNS); do
  echo \"Run \$i:\"
  /opt/kripke/bin/kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E 'LTimes|LPlusTimes|Population|Solve |Throughput'
  echo ''
done
"

echo ""
echo "--- OPTIMIZED ---"
docker run --rm --gpus all \
  -v "$SCRIPT_DIR/LTimes.h:/opt/LTimes.h:ro" \
  --entrypoint /bin/bash leo-kripke-nvidia -c "
cp /opt/LTimes.h /opt/kripke-src/src/Kripke/Arch/LTimes.h
echo 'Replaced LTimes.h with Zone<->Group swap version'
cd /opt/kripke-build && make -j\$(nproc) 2>/dev/null | tail -1
cp kripke.exe /opt/kripke/bin/
echo ''
for i in \$(seq 1 $NRUNS); do
  echo \"Run \$i:\"
  /opt/kripke/bin/kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E 'LTimes|LPlusTimes|Population|Solve |Throughput'
  echo ''
done
"
