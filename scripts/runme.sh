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
#   bash scripts/runme.sh --with-table-iv          # + Table IV (15 RAJAPerf kernels) + Table V (6 HPC apps), GPU required
#   bash scripts/runme.sh --with-table-iv --use-prebuilt        # same, but pull all per-app images from Docker Hub (faster)
#   bash scripts/runme.sh --with-table-iv --rajaperf-only       # Table IV only, skip HPC apps
#   bash scripts/runme.sh --with-table-iv --vendor amd          # AMD MI300A path
#   bash scripts/runme.sh --with-table-iv --vendor intel        # Intel PVC path (Kripke/QuickSilver/llama.cpp auto-skipped — no SYCL ports)
#   bash scripts/runme.sh --with-table-iv --gpu-arch 80         # A100 (sm_80) instead of H100/GH200 (sm_90 default)
#   bash scripts/runme.sh --help
#
# Disk requirement: Figure-5-only ≥40 GB free; --with-table-iv (default, with HPC
# apps) ≥250 GB free — six per-app image chains plus RAJAPerf accumulate
# substantial intermediate layers. Use --rajaperf-only for ≥120 GB.
#
# Exit codes: 0 = everything verified, non-zero = a step failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$LEO_ROOT"

DO_TABLE_IV=false
DO_HPC_APPS=true     # under --with-table-iv, also run the 6 HPC apps; opt-out via --rajaperf-only
USE_PREBUILT=false
PREBUILT_TAG="${PREBUILT_TAG:-v0.1.18}"
TABLE_IV_VENDOR="nvidia"
TABLE_IV_GPU_ARCH="${GPU_ARCH:-90}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-table-iv)  DO_TABLE_IV=true; shift ;;
    --rajaperf-only)  DO_HPC_APPS=false; shift ;;
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
  echo "============================================================"
  echo "==> [$(date +%H:%M:%S)] Step ${STEP_N}: $name"
  echo "    log: $log"
  echo "============================================================"
  t0=$(date +%s)
  if "$@" 2>&1 | tee "$log"; then
    t1=$(date +%s)
    echo ""
    echo "    ok (${name}): $((t1 - t0))s"
  else
    t1=$(date +%s)
    echo ""
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
  run_step "profiling data already populated (skipping download)" bash -c \
    "echo 'results/ exists with $(du -sh "$LEO_ROOT/results" 2>/dev/null | cut -f1) — skipping download_data.sh.'"
else
  run_step "download pre-collected profiling data (~1 GB compressed, ~5.6 GB extracted)" bash "$HERE/download_data.sh"
fi

if docker image inspect leo-base-universal:latest >/dev/null 2>&1; then
  run_step "leo-base-universal already present (skipping pull/build)" bash -c \
    "echo 'leo-base-universal:latest is at $(docker image inspect --format \"{{.Id}}\" leo-base-universal:latest) — skipping pull/build.'"
elif [ "$USE_PREBUILT" = true ]; then
  run_step "pull pre-built leo-base-universal:${PREBUILT_TAG} from Docker Hub (~2 min)" \
    bash "$HERE/evaluation/pull_prebuilt_images.sh" --workloads base-universal --tag "$PREBUILT_TAG"
else
  run_step "build leo-base-universal Docker image (~15-20 min)" bash "$HERE/evaluation/build_containers.sh" universal --base-only
fi

SDC_OUT="$LOG_DIR/sdc_output.txt"
run_step "collect_sdc.sh (Figure 5, ~10-15 min)" bash -c "bash '$HERE/collect_sdc.sh' > '$SDC_OUT' 2>&1 && tail -25 '$SDC_OUT'"

run_step "verify SHA-256 against committed reference" bash -c "cd '$LEO_ROOT' && sha256sum -c sdc_coverage_reference.txt.sha256 && echo 'SHA-256 of sdc_coverage_reference.txt matched the committed paper reference.'"

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

  # ---------------- HPC apps (Section V, Table V apps) -----------------------
  # Six HPC apps from the paper's Table IV/V row set, run via the appendix's
  # `bash benchmarks/<name>/run_compare_<vendor>.sh`. Each one drives a
  # vendor-specific Docker image (leo-<workload>-<vendor>) — pulled from
  # Docker Hub under --use-prebuilt, otherwise built locally via
  # build_workload_image.sh. Per appendix § AE Computation Map, Intel SYCL
  # ports for Kripke / QuickSilver / llama.cpp do not exist; those are
  # skipped on --vendor intel.
  if [ "$DO_HPC_APPS" = true ]; then
    declare -A HPC_SCRIPTS=(
      [lulesh]="run_compare_<vendor>.sh"
      [minibude]="run_compare.sh --vendor <vendor>"
      [xsbench]="run_compare_<vendor>.sh"
      [quicksilver]="run_compare_<vendor>.sh"
      [kripke]="run_compare_<vendor>.sh"
      [llamacpp]="run_compare_<vendor>.sh"
    )
    declare -A HPC_VENDORS=(
      [lulesh]="amd nvidia intel"
      [minibude]="amd nvidia intel"
      [xsbench]="amd nvidia intel"
      [quicksilver]="amd nvidia"
      [kripke]="amd nvidia"
      [llamacpp]="amd nvidia"
    )

    run_hpc_app() {
      local app="$1"
      local supported="${HPC_VENDORS[$app]}"
      if ! echo " $supported " | grep -q " $TABLE_IV_VENDOR "; then
        run_step "skip $app (no $TABLE_IV_VENDOR port — see appendix § AE Computation Map)" \
          bash -c "echo 'Skipped: $app has no $TABLE_IV_VENDOR port. Supported vendors: $supported.'"
        return 0
      fi

      # Pull or build the workload image (idempotent — pull/build script
      # short-circuits when the image already exists locally).
      if docker image inspect "leo-${app}-${TABLE_IV_VENDOR}:latest" >/dev/null 2>&1; then
        run_step "leo-${app}-${TABLE_IV_VENDOR} already present (skipping pull/build)" \
          bash -c "echo 'leo-${app}-${TABLE_IV_VENDOR}:latest exists — skipping pull/build.'" || true
      elif [ "$USE_PREBUILT" = true ]; then
        run_step "pull leo-${app}-${TABLE_IV_VENDOR}:${PREBUILT_TAG} from Docker Hub" \
          bash "$HERE/evaluation/pull_prebuilt_images.sh" --vendor "$TABLE_IV_VENDOR" --workloads "base,hpctoolkit,$app" --tag "$PREBUILT_TAG" || true
      else
        run_step "build leo-${app}-${TABLE_IV_VENDOR} (3-layer chain)" \
          env GPU_ARCH="$TABLE_IV_GPU_ARCH" \
          bash "$HERE/evaluation/build_workload_image.sh" "$TABLE_IV_VENDOR" "$app" || true
      fi

      # Resolve the run-compare script path (minibude is the only one that
      # uses a single multi-vendor script with --vendor).
      local cmd_template="${HPC_SCRIPTS[$app]}"
      local cmd="${cmd_template//<vendor>/$TABLE_IV_VENDOR}"
      run_step "Table V ${app} ${TABLE_IV_VENDOR}: bash benchmarks/${app}/${cmd}" \
        bash -c "cd '$LEO_ROOT' && bash benchmarks/${app}/${cmd}" || true
    }

    for app in lulesh minibude xsbench quicksilver kripke llamacpp; do
      run_hpc_app "$app"
    done
  fi
fi

END_ALL=$(date +%s)
echo ""
echo "============================================================"
echo "  RUNME COMPLETE"
echo "============================================================"
echo ""
echo "  Wall-clock: $(( (END_ALL - START_ALL) / 60 )) min $(( (END_ALL - START_ALL) % 60 )) s"
echo ""
echo "  Outputs:"
echo "    Figure 5 SDC table:        $SDC_OUT"
echo "    Figure 5 SHA-256 verified: OK (against committed reference)"
if [ "$DO_TABLE_IV" = true ]; then
  echo "    Table IV CSV (${TABLE_IV_VENDOR}):      $LEO_ROOT/benchmarks/rajaperf-h100/rajaperf-compare-summary.csv"
fi
echo ""
echo "  Logs:"
echo "    Per-step:  $LOG_DIR/step*.log"
echo "    Master:    $MASTER_LOG"
echo ""
if [ "$DO_TABLE_IV" = true ]; then
  if [ "$DO_HPC_APPS" = true ]; then
    echo "  HPC apps (Section V, Table V): auto-run for ${TABLE_IV_VENDOR}."
    echo "    Per-app step logs in $LOG_DIR/step*Table_V*.log"
    echo "    Note: HipKittens (Section VI.D, AMD-only RMSNorm) is not in Table IV/V"
    echo "    and is not auto-run. See benchmarks/hipkittens/ if added in a future release."
  else
    echo "  HPC apps skipped (--rajaperf-only). To run them later:"
    echo "    bash benchmarks/<workload>/run_compare_${TABLE_IV_VENDOR}.sh"
  fi
else
  echo "  Next (optional, requires GPU): add --with-table-iv to also build"
  echo "  the per-vendor RAJAPerf chain and reproduce Table IV (15 RAJAPerf"
  echo "  kernels) plus Table V's 6 HPC apps in one unified flow."
fi
echo "============================================================"
