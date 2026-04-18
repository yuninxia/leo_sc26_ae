#!/bin/bash
# Time Leo's analysis on ALL workloads from Table IV
# Single Docker container — no per-workload startup overhead
# Usage: bash scripts/time_analysis.sh

# Override via RESULTS_DIR env var to point at pre-collected measurements
R="${RESULTS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/results}"
PK="$R/per-kernel"
IMAGE="leo-base-universal"

docker run --rm \
  -v "$R/nvidia-athena-rajaperf-20260226-094201:/data/nvidia-rajaperf:ro" \
  -v "$PK/Apps_LTIMES/amd:/data/amd-rajaperf:ro" \
  -v "$PK/Apps_LTIMES/intel:/data/intel-rajaperf:ro" \
  -v "$R/nvidia-gilgamesh-minibude-20260222-174228:/data/nvidia-minibude:ro" \
  -v "$R/amd-minibude-20260220-151739:/data/amd-minibude:ro" \
  -v "$R/intel-minibude-20260222-163954:/data/intel-minibude:ro" \
  -v "$R/nvidia-gilgamesh-xsbench-20260222-182940:/data/nvidia-xsbench:ro" \
  -v "$R/amd-xsbench-20260220-175339:/data/amd-xsbench:ro" \
  -v "$R/intel-xsbench-20260218-134304:/data/intel-xsbench:ro" \
  -v "$R/nvidia-gilgamesh-lulesh-20260222-190913:/data/nvidia-lulesh:ro" \
  -v "$R/amd-lulesh-20260220-190435:/data/amd-lulesh:ro" \
  -v "$R/intel-lulesh-20260218-153334:/data/intel-lulesh:ro" \
  -v "$R/nvidia-gilgamesh-quicksilver-20260219-115644:/data/nvidia-quicksilver:ro" \
  -v "$R/amd-quicksilver-20260322-165227:/data/amd-quicksilver:ro" \
  -v "$R/nvidia-gilgamesh-llamacpp-20260325-131430:/data/nvidia-llamacpp:ro" \
  -v "$R/amd-llamacpp-20260325-131416:/data/amd-llamacpp:ro" \
  -v "$R/nvidia-gilgamesh-kripke-20260325-141654:/data/nvidia-kripke:ro" \
  -v "$R/amd-kripke-20260325-141150:/data/amd-kripke:ro" \
  -v "$(cd "$(dirname "$0")" && pwd)/time_analysis.py:/opt/leo/scripts/time_analysis.py:ro" \
  -e PYTHONUNBUFFERED=1 \
  $IMAGE -c "uv run python scripts/time_analysis.py"
