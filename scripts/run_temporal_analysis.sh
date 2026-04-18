#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MEASUREMENTS_DIR="${1:-$ROOT_DIR/tests/data/pc/amd/hpctoolkit-lammps.intelmpi.temporal-measurements}"
OUT_DIR="${2:-$ROOT_DIR/outputs/temporal}"

TIME_BINS="${TIME_BINS:-100}"
HWID_TOP="${HWID_TOP:-32}"

BUILDER_BIN="$ROOT_DIR/builds/temporal_cube_builder"
BUILDER_SRC="$ROOT_DIR/scripts/temporal_cube_builder.cpp"

if [[ ! -d "$MEASUREMENTS_DIR" ]]; then
  echo "Error: measurements dir not found: $MEASUREMENTS_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$ROOT_DIR/builds"

echo "Building temporal cube builder..."
g++ -O3 -fopenmp -std=c++17 -o "$BUILDER_BIN" "$BUILDER_SRC" -lstdc++fs

echo "Running temporal cube builder..."
"$BUILDER_BIN" "$MEASUREMENTS_DIR" \
  --time-bins "$TIME_BINS" \
  --output "$OUT_DIR/cube.bin" \
  --hwid-out "$OUT_DIR/hwid.csv" \
  --hwid-time-out "$OUT_DIR/hwid_time.csv"

echo "Generating plots..."
uv run python "$ROOT_DIR/scripts/temporal_visualize.py" \
  "$OUT_DIR/cube.bin" \
  --hwid "$OUT_DIR/hwid.csv" \
  --hwid-time "$OUT_DIR/hwid_time.csv" \
  --out-dir "$OUT_DIR" \
  --hwid-top "$HWID_TOP"

echo "Done. Outputs in: $OUT_DIR"
