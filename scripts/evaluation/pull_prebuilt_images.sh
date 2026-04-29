#!/bin/bash
# Pull the pre-built Docker images for this AE release from Docker Hub,
# skipping the ~45 min build step. Alternative to running
# `scripts/evaluation/build_workload_image.sh <vendor> <workload>` from
# source Dockerfiles.
#
# Images are published under the jssonxia/ namespace and tagged with the
# AE release tag (e.g., v0.1.14-sc26-ae). See the appendix for which tag
# pairs with each AE version.
#
# Usage:
#   bash scripts/evaluation/pull_prebuilt_images.sh                  # pull all NVIDIA x86_64 images (default)
#   bash scripts/evaluation/pull_prebuilt_images.sh --vendor nvidia-arm  # GH200 / Jetson ARM64 NVIDIA
#   bash scripts/evaluation/pull_prebuilt_images.sh --workloads rajaperf,minibude,kripke
#   bash scripts/evaluation/pull_prebuilt_images.sh --tag v0.1.14-sc26-ae
#
# Trust note: image Dockerfiles are shipped in the artifact under
# scripts/evaluation/docker/. Reviewers who prefer to rebuild for
# provenance can skip this script entirely and use
# scripts/evaluation/build_workload_image.sh instead.
set -euo pipefail

VENDOR="nvidia"
WORKLOADS="base-universal,base,hpctoolkit,rajaperf,minibude,xsbench,kripke,llamacpp,quicksilver,nvhpc,lulesh"
TAG="${PREBUILT_TAG:-latest}"
REGISTRY="${PREBUILT_REGISTRY:-jssonxia}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vendor)    VENDOR="$2"; shift 2 ;;
        --workloads) WORKLOADS="$2"; shift 2 ;;
        --tag)       TAG="$2"; shift 2 ;;
        --registry)  REGISTRY="$2"; shift 2 ;;
        --help|-h)   sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

echo "=== pulling pre-built images from ${REGISTRY}/ (tag: ${TAG}) ==="
IFS=',' read -ra WL_ARRAY <<< "$WORKLOADS"

for wl in "${WL_ARRAY[@]}"; do
    # base-universal is vendor-agnostic; all others are per-vendor
    if [[ "$wl" == "base-universal" ]]; then
        image="${REGISTRY}/leo-base-universal:${TAG}"
        localtag="leo-base-universal:latest"
    else
        image="${REGISTRY}/leo-${wl}-${VENDOR}:${TAG}"
        localtag="leo-${wl}-${VENDOR}:latest"
    fi

    echo ""
    echo "--> $image"
    # Try the primary tag silently. Vendor images (rajaperf/lulesh/etc.) don't
    # always have a per-LEO-release tag — they're stable across releases since
    # they don't bake the LEO source — so we fall back to :latest cleanly
    # without printing scary "Error response from daemon" lines unless every
    # candidate fails.
    if docker pull "$image" >/dev/null 2>&1; then
        docker tag "$image" "$localtag"
        echo "    pulled $image, tagged as $localtag"
    elif [[ "$TAG" != "latest" ]]; then
        fallback="${REGISTRY}/leo-${wl}-${VENDOR}:latest"
        [[ "$wl" == "base-universal" ]] && fallback="${REGISTRY}/leo-base-universal:latest"
        echo "    :${TAG} not published, falling back to :latest ($fallback)"
        if docker pull "$fallback" >/dev/null 2>&1; then
            docker tag "$fallback" "$localtag"
            echo "    pulled $fallback, tagged as $localtag"
        else
            echo "    ERROR: both :${TAG} and :latest pulls failed; details:"
            docker pull "$fallback" 2>&1 | sed 's/^/      /' | tail -3
            echo "    fall back to build:"
            echo "      bash scripts/evaluation/build_workload_image.sh ${VENDOR} ${wl}"
        fi
    else
        echo "    ERROR: pull failed; details:"
        docker pull "$image" 2>&1 | sed 's/^/      /' | tail -3
        echo "    fall back to build:"
        echo "      bash scripts/evaluation/build_workload_image.sh ${VENDOR} ${wl}"
    fi
done

echo ""
echo "=== done ==="
echo "Local image tags (runme.sh/run_compare.sh will find them as leo-*-${VENDOR}:latest):"
docker images "leo-*" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | head -20
