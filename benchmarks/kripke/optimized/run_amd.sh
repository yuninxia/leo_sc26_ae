#!/bin/bash
# Kripke LTimes optimization benchmark — AMD MI300A (CHAI RM mode, no XNACK)
# Usage: ./run_amd.sh [nruns]
# Requires: leo-kripke-amd Docker image + AMD GPU
set -e
NRUNS=${1:-5}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== AMD Kripke LTimes Benchmark, CHAI RM mode ($NRUNS runs) ==="

echo ""
echo "--- BASELINE ---"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --entrypoint /bin/bash leo-kripke-amd -c "
mkdir -p /tmp/kb && cd /tmp/kb
cmake /opt/kripke-src \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH='/opt/rocm;/opt/rocm/lib/cmake' \
  -DENABLE_HIP=On -DENABLE_CHAI=On \
  -DCHAI_DISABLE_RM=OFF -DCHAI_THIN_GPU_ALLOCATE=OFF -DCHAI_ENABLE_UM=OFF \
  -DENABLE_MPI=Off -DENABLE_OPENMP=Off \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DROCM_PATH=/opt/rocm \
  2>/dev/null | tail -1
make -j\$(nproc) 2>/dev/null | tail -1
echo ''
for i in \$(seq 1 $NRUNS); do
  echo \"Run \$i:\"
  ./kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E 'LTimes|LPlusTimes|Population|Solve |Throughput'
  echo ''
done
"

echo ""
echo "--- OPTIMIZED ---"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$SCRIPT_DIR/LTimes.h:/opt/LTimes.h:ro" \
  --entrypoint /bin/bash leo-kripke-amd -c "
mkdir -p /tmp/kb && cd /tmp/kb
cmake /opt/kripke-src \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH='/opt/rocm;/opt/rocm/lib/cmake' \
  -DENABLE_HIP=On -DENABLE_CHAI=On \
  -DCHAI_DISABLE_RM=OFF -DCHAI_THIN_GPU_ALLOCATE=OFF -DCHAI_ENABLE_UM=OFF \
  -DENABLE_MPI=Off -DENABLE_OPENMP=Off \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DROCM_PATH=/opt/rocm \
  2>/dev/null | tail -1
make -j\$(nproc) 2>/dev/null | tail -1

cp /opt/LTimes.h /opt/kripke-src/src/Kripke/Arch/LTimes.h
echo 'Replaced LTimes.h with Zone<->Group swap version'
cd /tmp/kb && make -j\$(nproc) 2>/dev/null | tail -1
echo ''
for i in \$(seq 1 $NRUNS); do
  echo \"Run \$i:\"
  ./kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E 'LTimes|LPlusTimes|Population|Solve |Throughput'
  echo ''
done
"
