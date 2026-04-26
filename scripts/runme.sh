#!/bin/bash
# runme.sh — end-to-end reviewer driver for the SC26 AE reproduction.
#
# Runs the Figure-5 path start-to-finish on a fresh host: download profiling
# data, build/pull the universal Docker image, run collect_sdc.sh inside that
# container, and verify the SHA-256. The pipeline is fully docker-based:
# no host-side Python environment or compiler is required. Every step is
# idempotent — safe to re-run; already-done steps are skipped. Total
# wall-clock on a clean x86_64 VM: ~20-30 min (Figure 5 only).
#
# Usage:
#   bash scripts/runme.sh                          # Figure 5 + sha256 verify (CPU-only, ~20 min)
#   bash scripts/runme.sh --use-prebuilt           # pull pre-built leo-base-universal from Docker Hub (~3 min vs ~10 min build)
#   bash scripts/runme.sh --with-table-iv          # + build NVIDIA chain, download baselines, run 15 RAJAPerf kernels (~30 min extra, GPU)
#   bash scripts/runme.sh --with-table-iv --use-prebuilt   # everything above, but pull all images from Docker Hub (~10 min vs ~30 min)
#   bash scripts/runme.sh --with-table-iv --gpu-arch 80    # A100 (sm_80) instead of H100/GH200 (sm_90 default)
#   bash scripts/runme.sh --help
#
# Disk requirement: Figure-5-only ≥40 GB free; with --with-table-iv ≥120 GB free
# (the vendor image chain accumulates ~90 GB of intermediate layers).
#
# Exit codes: 0 = everything verified, non-zero = a step failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$LEO_ROOT"

DO_TABLE_IV=false
USE_PREBUILT=false
PREBUILT_TAG="${PREBUILT_TAG:-v0.1.13}"
TABLE_IV_VENDOR="nvidia"
TABLE_IV_GPU_ARCH="${GPU_ARCH:-90}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-table-iv)  DO_TABLE_IV=true; shift ;;
    --use-prebuilt)   USE_PREBUILT=true; shift ;;
    --prebuilt-tag)   PREBUILT_TAG="$2"; shift 2 ;;
    --vendor)         TABLE_IV_VENDOR="$2"; shift 2 ;;
    --gpu-arch)       TABLE_IV_GPU_ARCH="$2"; shift 2 ;;
    --help|-h) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# NOTE: host-side preflight + uv install + uv sync used to live here. They
# were redundant: leo-base-universal ships with python 3.11 + uv + GCC 12 +
# hpcanalysis pre-built (see Dockerfile.base-universal lines 41-110), and
# every downstream step (collect_sdc.sh, run_workload_rajaperf.sh) runs LEO
# inside that container. Reviewers on hosts without GCC >=10 (e.g., RHEL 8.5)
# would hit OpenMP-strict-shared compile errors here that did not exist in
# any container path. Dropped to keep runme.sh purely docker-based.
#
# Reviewers who want host-side LEO (e.g., to use the Python API directly):
#     bash scripts/preflight.sh --install   # python3-dev + pkg-config + unzip
#     curl -LsSf https://astral.sh/uv/install.sh | sh
#     uv sync                                # builds hpcanalysis C++ extension
# This is independent of the AE reproduction flow.

if [ -d "$LEO_ROOT/results/per-kernel" ] && [ -n "$(ls -A "$LEO_ROOT/results/per-kernel" 2>/dev/null)" ]; then
  echo ""; echo "==> Step $((STEP_N + 1)) skipped: results/ already populated ($(du -sh "$LEO_ROOT/results" | cut -f1))"
  STEP_N=$((STEP_N + 1))
else
  run_step "download pre-collected profiling data (~1 GB compressed, ~5.6 GB extracted)" bash "$HERE/download_data.sh"
fi

if docker image inspect leo-base-universal:latest >/dev/null 2>&1; then
  echo ""; echo "==> Step $((STEP_N + 1)) skipped: leo-base-universal:latest already built"
  STEP_N=$((STEP_N + 1))
elif [ "$USE_PREBUILT" = true ]; then
  run_step "pull pre-built leo-base-universal:${PREBUILT_TAG} from Docker Hub (~2 min)" \
    bash "$HERE/evaluation/pull_prebuilt_images.sh" --workloads base-universal --tag "$PREBUILT_TAG"
else
  run_step "build leo-base-universal Docker image (~15-20 min)" bash "$HERE/evaluation/build_containers.sh" universal --base-only
fi

SDC_OUT="$LOG_DIR/sdc_output.txt"
run_step "collect_sdc.sh (Figure 5, ~10-15 min)" bash -c "bash '$HERE/collect_sdc.sh' > '$SDC_OUT' 2>&1 && tail -25 '$SDC_OUT'"

run_step "verify SHA-256 against committed reference" bash -c "cd '$LEO_ROOT' && sha256sum -c sdc_coverage_reference.txt.sha256"

if [ "$DO_TABLE_IV" = true ]; then
  if [ "$USE_PREBUILT" = true ]; then
    run_step "pull pre-built leo-rajaperf-${TABLE_IV_VENDOR}:${PREBUILT_TAG} from Docker Hub (~3-5 min)" \
      bash "$HERE/evaluation/pull_prebuilt_images.sh" --vendor "$TABLE_IV_VENDOR" --workloads base,hpctoolkit,rajaperf --tag "$PREBUILT_TAG"
  else
    run_step "build leo-rajaperf-${TABLE_IV_VENDOR} (3-layer chain)" \
      env GPU_ARCH="$TABLE_IV_GPU_ARCH" \
      bash "$HERE/evaluation/build_workload_image.sh" "$TABLE_IV_VENDOR" rajaperf
  fi

  run_step "download RAJAPerf baselines (original + optimized)" \
    bash "$HERE/evaluation/download_benchmarks.sh"

  run_step "Table IV ${TABLE_IV_VENDOR}: run_compare (all 15 RAJAPerf kernels)" \
    bash "$HERE/evaluation/run_workload_rajaperf.sh" --vendor "$TABLE_IV_VENDOR"

  run_step "compare speedups vs rajaperf-compare-summary.csv" bash -c "
    SUM='$LEO_ROOT/benchmarks/rajaperf-h100/rajaperf-compare-summary.csv'
    if [ -s \"\$SUM\" ]; then
      python3 - <<'PYEOF'
import csv
with open('$LEO_ROOT/benchmarks/rajaperf-h100/rajaperf-compare-summary.csv') as f:
    rows = list(csv.DictReader(f))
if not rows: raise SystemExit('summary CSV empty')
print(f\"{'kernel':<28}{'speedup_mean':>14}\")
print('-' * 42)
for r in rows:
    print(f\"{r['kernel']:<28}{float(r['speedup_mean']):>13.4f}x\")
PYEOF
    else
      echo 'summary CSV not produced — check preceding step'; exit 1
    fi"
fi

END_ALL=$(date +%s)
echo ""
echo "====================================================================="
echo "  DONE. Figure 5 reproduced and verified (sha256 OK)."
if [ "$DO_TABLE_IV" = true ]; then
  echo "  Table IV ${TABLE_IV_VENDOR} RAJAPerf also reproduced."
  echo "  Optional HPC apps (not auto-run): miniBUDE, XSBench, Kripke, LULESH, llama.cpp, QuickSilver."
  echo "    bash scripts/evaluation/build_workload_image.sh ${TABLE_IV_VENDOR} <workload>"
  echo "    bash benchmarks/<workload>/run_compare*.sh"
fi
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
