#!/bin/bash
# Package pre-collected HPCToolkit measurements for distribution via GitHub Release.
# Creates leo-sc26-measurements.tar.gz (~900 MB compressed).
#
# Usage: LEO_ROOT=/path/to/leo bash scripts/package_data.sh [output.tar.gz]
set -euo pipefail

LEO_ROOT="${LEO_ROOT:-$HOME/pg/odyssey/leo}"
OUT="${1:-/tmp/leo-sc26-measurements.tar.gz}"

if [ ! -d "$LEO_ROOT/results" ]; then
  echo "ERROR: $LEO_ROOT/results not found. Set LEO_ROOT env var."
  exit 1
fi

cd "$LEO_ROOT"

# Directories referenced by scripts/collect_sdc.sh and scripts/time_analysis.sh
DIRS=(
  # HPC applications (16 measurements)
  results/nvidia-gilgamesh-minibude-20260222-174228
  results/amd-minibude-20260220-151739
  results/intel-minibude-20260222-163954
  results/nvidia-gilgamesh-xsbench-20260222-182940
  results/amd-xsbench-20260220-175339
  results/intel-xsbench-20260218-134304
  results/nvidia-gilgamesh-lulesh-20260222-190913
  results/amd-lulesh-20260220-190435
  results/intel-lulesh-20260218-153334
  results/nvidia-gilgamesh-quicksilver-20260219-115644
  results/amd-quicksilver-20260322-165227
  results/nvidia-gilgamesh-llamacpp-20260325-131430
  results/amd-llamacpp-20260325-131416
  results/nvidia-gilgamesh-kripke-20260325-141654
  results/amd-kripke-20260325-141150
  results/nvidia-athena-rajaperf-20260226-094201

  # Per-kernel RAJAPerf (15 kernels × 4 vendor variants)
  results/per-kernel
)

# Sanity check: all dirs exist
missing=0
for d in "${DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    echo "MISSING: $d"
    missing=$((missing+1))
  fi
done
if [ "$missing" -gt 0 ]; then
  echo "ERROR: $missing directories missing. Aborting."
  exit 1
fi

echo "Packaging ${#DIRS[@]} directories into $OUT ..."
echo "This takes ~3-5 min for ~900 MB compressed output."

tar -czf "$OUT" "${DIRS[@]}"

echo ""
echo "Done. Output:"
ls -lh "$OUT"
echo ""
echo "Next step: upload to GitHub release"
echo "  gh release create v1.0-sc26-data $OUT \\"
echo "    --title 'SC26 AE: Pre-collected HPCToolkit measurements' \\"
echo "    --notes 'Profiling data for Table IV, Table V, Figure 5, and case studies.'"
