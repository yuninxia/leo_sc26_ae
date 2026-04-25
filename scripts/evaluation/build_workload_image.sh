#!/bin/bash
# Build a vendor+workload Docker image portably (no SSH, no authors'-side
# conveniences). Handles the 3-layer dependency chain that reviewers would
# otherwise have to intuit from reading multiple Dockerfiles:
#
#   nvidia/cuda:... | rocm/... | intel/oneapi:...     (upstream vendor image)
#       |
#       V
#   leo-base-<vendor>         (Dockerfile.base-<vendor>)
#       |
#       V
#   leo-hpctoolkit-<vendor>   (Dockerfile.hpctoolkit-<vendor>)
#       |
#       V
#   leo-<workload>-<vendor>   (Dockerfile.<workload>-<vendor>)
#
# Two of our workloads take a different path:
#   LULESH  (NVIDIA) — FROM leo-nvhpc-<vendor> (NVIDIA HPC SDK, +nvc++)
#   Other workloads — FROM leo-hpctoolkit-<vendor>
#
# Usage:
#   bash scripts/evaluation/build_workload_image.sh <vendor> <workload>
#
#   vendor:   nvidia | amd | intel    (or nvidia-arm for GH200/Jetson)
#   workload: rajaperf | lulesh | minibude | xsbench | kripke | llamacpp | quicksilver
#
# Environment overrides:
#   GPU_ARCH=90    # CUDA compute capability (default 90 = Hopper; 80 = A100; 86 = A10)
#
# Examples:
#   bash scripts/evaluation/build_workload_image.sh nvidia rajaperf
#   GPU_ARCH=80 bash scripts/evaluation/build_workload_image.sh nvidia rajaperf   # A100
#   bash scripts/evaluation/build_workload_image.sh nvidia-arm rajaperf           # GH200
set -euo pipefail

VENDOR="${1:-}"
WORKLOAD="${2:-}"

if [[ -z "$VENDOR" || -z "$WORKLOAD" ]]; then
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

case "$VENDOR" in
    nvidia|nvidia-arm|amd|intel) ;;
    *) echo "ERROR: vendor must be one of nvidia, nvidia-arm, amd, intel" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"
GPU_ARCH="${GPU_ARCH:-90}"

have_image() { docker image inspect "$1:latest" >/dev/null 2>&1; }

build_layer() {
    local tag="$1" dockerfile="$2" extra_args=("${@:3}")
    if have_image "$tag"; then
        echo "  ok  ${tag}:latest already built (skipping)"
        return 0
    fi
    echo "==> building ${tag}"
    local t0 t1
    t0=$(date +%s)
    docker build "${extra_args[@]}" -f "$dockerfile" -t "${tag}:latest" "$LEO_ROOT"
    t1=$(date +%s)
    echo "  ok  ${tag}:latest built in $((t1 - t0))s"
}

cd "$LEO_ROOT"

# --- Layer 1: base ---
build_layer "leo-base-${VENDOR}" "$DOCKER_DIR/Dockerfile.base-${VENDOR}"

# --- Layer 2: tool layer (hpctoolkit for most, nvhpc only needed for LULESH) ---
# Only build nvhpc if the workload actually needs it.
NEED_NVHPC=false
[[ "$WORKLOAD" == "lulesh" && "$VENDOR" == nvidia* ]] && NEED_NVHPC=true

if $NEED_NVHPC; then
    build_layer "leo-nvhpc-${VENDOR}" "$DOCKER_DIR/Dockerfile.nvhpc-${VENDOR}"
else
    build_layer "leo-hpctoolkit-${VENDOR}" "$DOCKER_DIR/Dockerfile.hpctoolkit-${VENDOR}"
fi

# --- Layer 3: workload ---
WORKLOAD_DOCKERFILE="$DOCKER_DIR/Dockerfile.${WORKLOAD}-${VENDOR}"
if [[ ! -f "$WORKLOAD_DOCKERFILE" ]]; then
    echo "ERROR: $WORKLOAD_DOCKERFILE does not exist" >&2
    exit 1
fi

# rajaperf takes GPU_ARCH build-arg; others don't
BUILD_EXTRA=()
if [[ "$WORKLOAD" == "rajaperf" ]]; then
    BUILD_EXTRA+=(--build-arg "GPU_ARCH=$GPU_ARCH")
fi

build_layer "leo-${WORKLOAD}-${VENDOR}" "$WORKLOAD_DOCKERFILE" "${BUILD_EXTRA[@]}"

echo ""
echo "=== done ==="
docker image inspect "leo-${WORKLOAD}-${VENDOR}:latest" \
    --format '{{printf "image: %s  size: %.2f GB" "leo-'"${WORKLOAD}"'-'"${VENDOR}"'" (div (add .Size 1073741823) 1073741824)}}' 2>/dev/null \
    || docker images "leo-${WORKLOAD}-${VENDOR}" | tail -1
