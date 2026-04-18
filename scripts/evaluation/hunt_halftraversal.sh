#!/bin/bash
# Repeatedly run ArborX profiling until HalfTraversal gets PC samples.
# Logs each attempt's results and stops when HalfTraversal is captured.
#
# Usage: ./hunt_halftraversal.sh [max_attempts]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAX_ATTEMPTS="${1:-20}"
LOG_FILE="$LEO_ROOT/results/hunt_halftraversal.log"

echo "=== HalfTraversal Hunting Experiment ===" | tee "$LOG_FILE"
echo "Max attempts: $MAX_ATTEMPTS" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Python script to check which kernels got PC samples
# HalfTraversal is always the kernel with the longest execution time (~1.8-2.0s).
# We detect success if any kernel with >1s execution time has non-zero stall_cycles.
CHECK_SCRIPT=$(cat <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ.get("LEO_ROOT", "."), "src"))
from leo.db.reader import DatabaseReader

db_path = sys.argv[1]
reader = DatabaseReader(db_path)
kernels = reader.get_per_kernel_summary(sort_by="stall_cycles")
cct = reader.get_cct()

print("Kernel Summary (sorted by exec time):")
found_hot = False
kernels_sorted = kernels.sort_values("execution_time_s", ascending=False)
for cct_id, row in kernels_sorted.iterrows():
    stall = int(row.get("stall_cycles", 0))
    time_s = float(row.get("execution_time_s", 0))
    # Get kernel name from CCT
    name = cct.loc[cct_id, "name"] if cct_id in cct.index else f"cct#{cct_id}"
    # Truncate long Kokkos names
    short_name = str(name)
    if len(short_name) > 80:
        short_name = short_name[:77] + "..."
    marker = ""
    if time_s > 1.0:
        marker = " <-- HOT KERNEL"
        if stall > 0:
            found_hot = True
            marker += " *** SAMPLED ***"
    if time_s > 0.01 or stall > 0:  # only print significant kernels
        print(f"  {short_name}: time={time_s:.4f}s, stall_cycles={stall:,}{marker}")

sampled_count = len(kernels[kernels["stall_cycles"] > 0])
total_count = len(kernels)
sampled_time = kernels[kernels["stall_cycles"] > 0]["execution_time_s"].sum()
total_time = kernels["execution_time_s"].sum()
print(f"\nSampled: {sampled_count}/{total_count} kernels, {sampled_time:.3f}s/{total_time:.3f}s time coverage")

if found_hot:
    print("\n*** CAPTURED! Hot kernel (>1s) has PC samples! ***")
    sys.exit(0)
else:
    print("\nHot kernel NOT captured this run.")
    sys.exit(1)
PYEOF
)

CAPTURED=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Attempt $attempt / $MAX_ATTEMPTS  ($(date))" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # Run profiling only (skip Leo analysis inside Docker to save time)
    "$SCRIPT_DIR/run_evaluation.sh" amd --workload arborx --skip-leo 2>&1 | tee -a "$LOG_FILE" || true

    # Find the latest results directory
    LATEST=$(ls -dt "$LEO_ROOT"/results/amd-arborx-* 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No results directory found" | tee -a "$LOG_FILE"
        continue
    fi

    # Find the database
    DB=$(ls -d "$LATEST"/hpctoolkit-*-database 2>/dev/null | head -1)
    if [ -z "$DB" ]; then
        echo "ERROR: No database found in $LATEST" | tee -a "$LOG_FILE"
        continue
    fi

    echo "Checking: $DB" | tee -a "$LOG_FILE"

    # Check for hot kernel PC samples (use pipefail to get python's exit code through tee)
    cd "$LEO_ROOT"
    set +e
    LEO_ROOT="$LEO_ROOT" uv run python -c "$CHECK_SCRIPT" "$DB" 2>&1 | tee -a "$LOG_FILE"
    CHECK_EXIT=${PIPESTATUS[0]}
    set -e

    if [ "$CHECK_EXIT" -eq 0 ]; then
        CAPTURED=1
        echo "" | tee -a "$LOG_FILE"
        echo "SUCCESS on attempt $attempt!" | tee -a "$LOG_FILE"
        echo "Results: $LATEST" | tee -a "$LOG_FILE"
        echo "Finished: $(date)" | tee -a "$LOG_FILE"
        break
    fi

    echo "" | tee -a "$LOG_FILE"
done

if [ "$CAPTURED" -eq 0 ]; then
    echo "FAILED: Hot kernel not captured in $MAX_ATTEMPTS attempts." | tee -a "$LOG_FILE"
    echo "Finished: $(date)" | tee -a "$LOG_FILE"
fi

# Print summary
echo "" | tee -a "$LOG_FILE"
echo "=== Summary ===" | tee -a "$LOG_FILE"
echo "Total attempts: $attempt" | tee -a "$LOG_FILE"
echo "Hot kernel captured: $([ $CAPTURED -eq 1 ] && echo 'YES' || echo 'NO')" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
