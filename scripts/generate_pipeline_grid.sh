#!/usr/bin/env bash
# Generate per-pipeline CPI stack figures for multiple workloads and combine into a grid.
#
# Usage:
#     ./scripts/generate_pipeline_grid.sh
#
# Reads databases from results/, generates individual PNGs into outputs/,
# then combines them into outputs/pipeline_combined.png.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$OUT_DIR"

# ── Workloads: name=database_path ──
# Edit this list to add/remove workloads.
declare -A WORKLOADS=(
    [minibude]="results/amd-minibude-20260218-114314/hpctoolkit-hip-bude-database"
    [xsbench]="results/amd-xsbench-20260218-123008/hpctoolkit-XSBench-database"
    [lulesh]="results/amd-lulesh-20260218-114506/hpctoolkit-lulesh2.0-database"
    [hipkittens]="results/amd-hipkittens-rmsnorm-20260215-142824/hpctoolkit-hipkittens_rmsnorm_bench-database"
)

# ── Labels (must match order of ORDERED below) ──
declare -A LABELS=(
    [minibude]="MiniBUDE (compute-bound)"
    [xsbench]="XSBench (memory-bound)"
    [lulesh]="LULESH (mixed)"
    [hipkittens]="HipKittens (compute-bound)"
)

# Grid order (top-left → top-right → bottom-left → bottom-right)
ORDERED=(minibude xsbench lulesh hipkittens)

# ── Step 1: Generate individual figures ──
echo "=== Generating individual pipeline figures ==="
for name in "${ORDERED[@]}"; do
    db="${ROOT_DIR}/${WORKLOADS[$name]}"
    out="$OUT_DIR/pipeline_${name}"
    if [[ ! -d "$db" ]]; then
        echo "  SKIP $name: database not found at $db"
        continue
    fi
    echo "  $name → ${out}.png"
    uv run python "$SCRIPT_DIR/pipeline_analysis.py" "$db" --figure-only --show-all -o "$out" >/dev/null
done

# ── Step 2: Combine into grid ──
echo ""
echo "=== Combining into grid ==="
IMAGES=()
LABEL_ARGS=()
for name in "${ORDERED[@]}"; do
    png="$OUT_DIR/pipeline_${name}.png"
    if [[ -f "$png" ]]; then
        IMAGES+=("$png")
        LABEL_ARGS+=("${LABELS[$name]}")
    fi
done

uv run python "$SCRIPT_DIR/combine_pipeline_figures.py" \
    "${IMAGES[@]}" \
    --labels "${LABEL_ARGS[@]}" \
    -o "$OUT_DIR/pipeline_combined.png"

echo ""
echo "Done. Output: $OUT_DIR/pipeline_combined.png"
