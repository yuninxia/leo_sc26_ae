#!/bin/bash
# Build all Leo Docker images and push to DockerHub.
#
# Usage:
#   ./build_and_push.sh                    # Build and push all images
#   ./build_and_push.sh --only intel       # Build and push only intel images
#   ./build_and_push.sh --only nvidia-arm  # Build and push only nvidia-arm images
#   ./build_and_push.sh --no-push          # Build only, don't push
#
# Prerequisites:
#   docker login --username jssonxia
#
# Build order (base must be built before workloads):
#   1. leo-base-{vendor}
#   2. leo-rajaperf-{vendor}     (depends on leo-base-{vendor})
#   3. leo-llamacpp-{vendor}     (depends on leo-base-{vendor})
#   4. leo-babelstream-{vendor}  (depends on leo-base-{vendor})
#   5. leo-base-universal        (standalone)
#
# Note: nvidia-arm images must be built on an ARM machine (e.g., hopper1).
#       All other images can be built on x86_64 machines.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCKER_DIR="$SCRIPT_DIR"
REGISTRY="jssonxia"

# Parse arguments
ONLY=""
PUSH=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --only)
            ONLY="$2"; shift 2 ;;
        --no-push)
            PUSH=false; shift ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# All vendors (order matters: base images first)
ALL_VENDORS=(intel amd nvidia nvidia-arm)
STANDALONE_IMAGES=(base-universal)

# Filter vendors if --only specified
if [[ -n "$ONLY" ]]; then
    ALL_VENDORS=("$ONLY")
fi

cd "$LEO_ROOT"

build_and_push() {
    local DOCKERFILE="$1"
    local LOCAL_TAG="$2"
    local REMOTE_TAG="$REGISTRY/$LOCAL_TAG"
    local BUILD_ARGS="${3:-}"

    if [[ ! -f "$DOCKERFILE" ]]; then
        echo "SKIP $LOCAL_TAG (Dockerfile not found: $DOCKERFILE)"
        return
    fi

    echo ""
    echo "============================================"
    echo " Building: $REMOTE_TAG"
    echo " Dockerfile: $DOCKERFILE"
    echo "============================================"

    docker build -f "$DOCKERFILE" \
        $BUILD_ARGS \
        -t "$LOCAL_TAG" \
        -t "$REMOTE_TAG" \
        .

    if [[ "$PUSH" = true ]]; then
        echo "Pushing $REMOTE_TAG..."
        docker push "$REMOTE_TAG"
        echo "Pushed: $REMOTE_TAG"
    fi
}

TOTAL_START=$(date +%s)

for VENDOR in "${ALL_VENDORS[@]}"; do
    echo ""
    echo "####################################################"
    echo " Vendor: $VENDOR"
    echo "####################################################"

    # 1. Base image
    build_and_push \
        "$DOCKER_DIR/Dockerfile.base-${VENDOR}" \
        "leo-base-${VENDOR}:latest"

    # 2. RAJAPerf workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.rajaperf-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.rajaperf-${VENDOR}" \
            "leo-rajaperf-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 3. LlamaCpp workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.llamacpp-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.llamacpp-${VENDOR}" \
            "leo-llamacpp-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 4. BabelStream workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.babelstream-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.babelstream-${VENDOR}" \
            "leo-babelstream-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 5. miniBUDE workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.minibude-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.minibude-${VENDOR}" \
            "leo-minibude-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 6. XSBench workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.xsbench-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.xsbench-${VENDOR}" \
            "leo-xsbench-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 7. ArborX workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.arborx-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.arborx-${VENDOR}" \
            "leo-arborx-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 8. QMCPACK workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.qmcpack-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.qmcpack-${VENDOR}" \
            "leo-qmcpack-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 9. CabanaMD workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.cabanamd-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.cabanamd-${VENDOR}" \
            "leo-cabanamd-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi

    # 10. AMR-Wind workload (depends on base)
    if [[ -f "$DOCKER_DIR/Dockerfile.amrwind-${VENDOR}" ]]; then
        build_and_push \
            "$DOCKER_DIR/Dockerfile.amrwind-${VENDOR}" \
            "leo-amrwind-${VENDOR}:latest" \
            "--build-arg BASE_TAG=latest"
    fi
done

# 4. Standalone images
if [[ -z "$ONLY" ]]; then
    for IMG in "${STANDALONE_IMAGES[@]}"; do
        build_and_push \
            "$DOCKER_DIR/Dockerfile.${IMG}" \
            "leo-${IMG}:latest"
    done
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

echo ""
echo "============================================"
echo " All builds complete (${TOTAL_ELAPSED}s)"
echo "============================================"
echo "Images pushed to: https://hub.docker.com/u/$REGISTRY"
echo ""
echo "To pull on Aurora:"
echo "  module load apptainer"
echo "  apptainer build --fakeroot leo-rajaperf-intel.sif docker://$REGISTRY/leo-rajaperf-intel:latest"
