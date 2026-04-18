#!/bin/bash
# Collect per-kernel Single Dependency Coverage for all Table IV workloads
# Usage: bash scripts/collect_sdc.sh

# Paths — override via env vars as needed
LEO="${LEO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
R="${RESULTS_DIR:-$LEO/results}"
PK="${PER_KERNEL_DIR:-$LEO/tests/data/pc/per-kernel}"

docker run --rm \
  -v "$LEO/src/leo:/opt/leo/src/leo:ro" \
  -v "$LEO/scripts/collect_sdc.py:/opt/leo/scripts/collect_sdc.py:ro" \
  -v "$PK:/data/per-kernel:ro" \
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
  -e PYTHONUNBUFFERED=1 \
  leo-base-universal -c "uv run python scripts/collect_sdc.py"
