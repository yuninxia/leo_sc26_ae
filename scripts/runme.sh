#!/bin/bash
# runme.sh — end-to-end reviewer driver for the SC26 AE reproduction.
#
# Runs the Figure-5 path start-to-finish on a fresh host: check build deps,
# install uv, uv sync, download profiling data, build the universal Docker
# image, run collect_sdc.sh, and verify the SHA-256 against the committed
# reference. Every step is idempotent — safe to re-run; already-done steps
# are skipped. Total wall-clock on a clean x86_64 VM: ~35–45 min.
#
# Usage:
#   bash scripts/runme.sh                   # Figure 5 + sha256 verify (CPU-only, ~35–45 min)
#   bash scripts/runme.sh --skip-preflight  # if you already installed python3-dev etc.
#   bash scripts/runme.sh --help
#
# Exit codes: 0 = everything verified, non-zero = a step failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$LEO_ROOT"

DO_PREFLIGHT=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-preflight) DO_PREFLIGHT=false; shift ;;
    --help|-h) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# Per-step log dir under logs/<timestamp>/. Each step also streams live to stdout.
LOG_DIR="$LEO_ROOT/logs/runme-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/runme.log"
exec > >(tee -a "$MASTER_LOG") 2>&1  # duplicate all stdout+stderr to master log
echo "runme.sh log dir: $LOG_DIR"
START_ALL=$(date +%s)

STEP_N=0
run_step() {
  local name="$1"; shift
  STEP_N=$((STEP_N + 1))
  local slug
  slug=$(echo "$name" | tr ' /:' '___' | tr -cd 'A-Za-z0-9_-')
  local log="$LOG_DIR/step${STEP_N}_${slug}.log"
  local t0 t1
  echo ""
  echo "==> [$(date +%H:%M:%S)] Step ${STEP_N}: $name"
  echo "    log: $log"
  t0=$(date +%s)
  if "$@" 2>&1 | tee "$log"; then
    t1=$(date +%s)
    echo "    ok (${name}): $((t1 - t0))s"
  else
    t1=$(date +%s)
    echo "    FAIL (${name}) after $((t1 - t0))s — see $log"
    return 1
  fi
}

ok()   { echo "    ok: $1"; }
skip() { echo "    skip: $1"; }

if [ "$DO_PREFLIGHT" = true ]; then
  run_step "preflight (python3-dev + pkg-config + unzip)" bash "$HERE/preflight.sh" --install
else
  echo ""; echo "==> Step 1 skipped: --skip-preflight"
  STEP_N=$((STEP_N + 1))
fi

run_step "install uv if missing" bash -c '
  if command -v uv >/dev/null 2>&1; then
    echo "uv already present: $(uv --version)"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
'
export PATH="$HOME/.local/bin:$PATH"

run_step "uv sync (Python env + hpcanalysis C++ build)" uv sync

if [ -d "$LEO_ROOT/results/per-kernel" ] && [ -n "$(ls -A "$LEO_ROOT/results/per-kernel" 2>/dev/null)" ]; then
  echo ""; echo "==> Step $((STEP_N + 1)) skipped: results/ already populated ($(du -sh "$LEO_ROOT/results" | cut -f1))"
  STEP_N=$((STEP_N + 1))
else
  run_step "download pre-collected profiling data (~1 GB compressed, ~5.6 GB extracted)" bash "$HERE/download_data.sh"
fi

if docker image inspect leo-base-universal:latest >/dev/null 2>&1; then
  echo ""; echo "==> Step $((STEP_N + 1)) skipped: leo-base-universal:latest already built"
  STEP_N=$((STEP_N + 1))
else
  run_step "build leo-base-universal Docker image (~15-20 min)" bash "$HERE/evaluation/build_containers.sh" universal --base-only
fi

SDC_OUT="$LOG_DIR/sdc_output.txt"
run_step "collect_sdc.sh (Figure 5, ~10-15 min)" bash -c "bash '$HERE/collect_sdc.sh' > '$SDC_OUT' 2>&1 && tail -25 '$SDC_OUT'"

run_step "verify SHA-256 against committed reference" bash -c "cd '$LEO_ROOT' && sha256sum -c sdc_coverage_reference.txt.sha256"

END_ALL=$(date +%s)
echo ""
echo "====================================================================="
echo "  DONE. Figure 5 reproduced and verified (sha256 OK)."
echo "  Total wall-clock: $(( (END_ALL - START_ALL) / 60 )) min $(( (END_ALL - START_ALL) % 60 )) s"
echo "  Per-step logs: $LOG_DIR"
echo ""
echo "  Next (optional, requires GPU):"
echo "    Table IV NVIDIA:  docker build --build-arg GPU_ARCH=<sm> \\"
echo "                        -f scripts/evaluation/docker/Dockerfile.rajaperf-nvidia \\"
echo "                        -t leo-rajaperf-nvidia ."
echo "                      bash scripts/evaluation/run_workload_rajaperf.sh \\"
echo "                        --kernel MASS3DEA,LTIMES,3MM --vendor nvidia"
echo "    Table V LLM:      export OPENROUTER_API_KEY=...; "
echo "                      uv run python -m scripts.validation.main --llm-eval \\"
echo "                        --llm-model google/gemini-3.1-pro-preview"
echo "====================================================================="
