#!/usr/bin/env bash
# Run Leo tests inside the pre-built universal Docker container.
#
# Usage:
#   ./scripts/run_tests_docker.sh                    # all tests
#   ./scripts/run_tests_docker.sh -m "not slow"      # skip slow tests
#   ./scripts/run_tests_docker.sh tests/test_blame.py # single file
#   ./scripts/run_tests_docker.sh -k "test_nvidia"   # by name pattern
#
# Real-time progress:  tail -f /tmp/leo-test-<timestamp>.log
#
# The container has nvdisasm, llvm-objdump, and libged.so (Intel GED),
# so all vendor-specific tests can run regardless of the host environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${LEO_DOCKER_IMAGE:-leo-base-universal:latest}"
LOG_FILE="/tmp/leo-test-$(date +%Y%m%d-%H%M%S).log"

# Default: run all tests with verbose output and 120s timeout
if [ $# -eq 0 ]; then
    PYTEST_ARGS="-v"
else
    PYTEST_ARGS="$*"
fi

# Check image exists
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Error: Docker image '$IMAGE' not found."
    echo "Build it first:"
    echo "  cd $LEO_ROOT/scripts/evaluation"
    echo "  docker build -f docker/Dockerfile.base-universal -t leo-base-universal:latest ../.."
    exit 1
fi

echo "=== Running Leo tests in $IMAGE ==="
echo "    pytest args: $PYTEST_ARGS"
echo "    log file:    $LOG_FILE"
echo ""
echo "    Track progress:  tail -f $LOG_FILE"
echo ""

# Strategy:
#   The image has a fully-built venv at /opt/leo/.venv with all deps installed
#   (including hpcanalysis C++ extensions). Leo is installed as editable.
#   We mount the host source read-only and use PYTHONPATH to override the
#   stale leo package in the venv with the current source — zero copy needed.
#   pytest runs from a writable /tmp dir for cache files.

docker run --rm \
  -v "$LEO_ROOT":/mnt/leo:ro \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$IMAGE" \
  -c "
set -e

echo '[1/2] Setting up environment...'

# Use the image's pre-built venv directly (no copy needed).
# Override leo's source via PYTHONPATH so the current host code is used
# instead of the stale editable install baked into the image.
export PYTHONPATH=/mnt/leo/src
export GED_LIBRARY_PATH=/opt/gtpin/libged.so

# Work from a writable temp dir (pytest writes .pytest_cache)
mkdir -p /tmp/work
cd /tmp/work

echo '[2/2] Running pytest...'
echo ''

# Run pytest using the image's venv Python directly
/opt/leo/.venv/bin/python -m pytest /mnt/leo/tests/ \
    --rootdir=/mnt/leo \
    -p no:cacheprovider \
    $PYTEST_ARGS
" 2>&1 | tee "$LOG_FILE"
