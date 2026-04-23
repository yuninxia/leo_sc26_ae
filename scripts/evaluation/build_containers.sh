#!/bin/bash
# Build and optionally run evaluation containers.
#
# LOCAL BUILD (no GPU, no SSH): pass vendor "universal" — everything runs on
# localhost against the Dockerfiles in docker/. This is the path AE reviewers
# should take for Figure 5 and Table V:
#     bash build_containers.sh universal --base-only
#
# GPU BUILD (optional, target hardware): vendors amd/nvidia/intel (and variants
# like nvidia-gilgamesh) build locally by default. Adding --run additionally
# SSHes to the authors' Rice cluster (odyssey/gilgamesh/headroom) — only
# relevant if you have access to that cluster; otherwise just drop --run.
#
# Uses a layered architecture:
#   1. Base image (leo-base-*): HPCToolkit + Dyninst + Leo (profiling infrastructure)
#   2. Workload image (leo-{workload}-*): Benchmark built on top of base
#
# Usage:
#   ./build_containers.sh <vendor> [--workload <name>] [--base-only] [--run] [--sif]
#   ./build_containers.sh all [--workload <name>]
#
# Examples:
#   ./build_containers.sh amd                        # Build base image + RAJAPerf workload on top
#   ./build_containers.sh amd --workload llamacpp    # Build base image + llama.cpp workload on top
#   ./build_containers.sh nvidia --base-only         # Build only base image (HPCToolkit + Leo)
#   ./build_containers.sh nvidia --run               # Build and open interactive shell
#   ./build_containers.sh all                        # Build base + RAJAPerf for all vendors
#   ./build_containers.sh all --workload llamacpp    # Build base + llama.cpp for all vendors
#
# The script SSHes to the appropriate target machine, builds the Docker image,
# and optionally converts to Singularity or opens an interactive shell.
# Requires shared filesystem access (NFS) to the Leo project directory.

set -e

# Disable BuildKit to avoid slow cache scans on shared machines with large build caches.
# The legacy builder starts immediately without scanning the BuildKit cache (which can be 600GB+).
export DOCKER_BUILDKIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"

# Valid workloads (add new ones here)
VALID_WORKLOADS="rajaperf llamacpp babelstream minibude xsbench arborx qmcpack minitest cabanamd amrwind quicksilver quicksilver-native lulesh hipkittens thunderkittens cujson flashinfer astra kripke"

# ============================================
# Target machine mapping
# ============================================
declare -A MACHINES ARCH_ARGS SIF_FLAGS DOCKER_GPU_FLAGS
MACHINES[amd]="odyssey"
MACHINES[nvidia]="illyad"
MACHINES[nvidia-gilgamesh]="gilgamesh"
MACHINES[nvidia-athena]="athena"
MACHINES[nvidia-voltar]="voltar"
MACHINES[nvidia-hopper1]="hopper1"
MACHINES[nvidia-hopper2]="hopper2"
MACHINES[intel]="headroom"
MACHINES[universal]=""  # builds locally (no GPU needed)

ARCH_ARGS[amd]="--build-arg GPU_TARGET=gfx942"
ARCH_ARGS[nvidia]="--build-arg GPU_ARCH=90"
ARCH_ARGS[nvidia-gilgamesh]="--build-arg GPU_ARCH=90"
ARCH_ARGS[nvidia-athena]="--build-arg GPU_ARCH=80"
ARCH_ARGS[nvidia-voltar]="--build-arg GPU_ARCH=80"
ARCH_ARGS[nvidia-hopper1]="--build-arg GPU_ARCH=90"
ARCH_ARGS[nvidia-hopper2]="--build-arg GPU_ARCH=90"
ARCH_ARGS[intel]=""
ARCH_ARGS[universal]=""

SIF_FLAGS[amd]="--rocm"
SIF_FLAGS[nvidia]="--nv"
SIF_FLAGS[nvidia-gilgamesh]="--nv"
SIF_FLAGS[nvidia-athena]="--nv"
SIF_FLAGS[nvidia-voltar]="--nv"
SIF_FLAGS[nvidia-hopper1]="--nv"
SIF_FLAGS[nvidia-hopper2]="--nv"
SIF_FLAGS[intel]="--bind /dev/dri"
SIF_FLAGS[universal]=""

DOCKER_GPU_FLAGS[amd]="--device=/dev/kfd --device=/dev/dri"
DOCKER_GPU_FLAGS[nvidia]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-gilgamesh]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-athena]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-voltar]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-hopper1]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-hopper2]="--gpus all"
DOCKER_GPU_FLAGS[intel]="--device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path"
DOCKER_GPU_FLAGS[universal]=""

# ============================================
# Argument parsing
# ============================================
VENDOR=""
WORKLOAD="rajaperf"
ACTION=""
BASE_ONLY=false
REBUILD_BASE=false
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        amd|nvidia|nvidia-gilgamesh|nvidia-athena|nvidia-voltar|nvidia-hopper1|nvidia-hopper2|intel|universal|all)
            VENDOR="$1"; shift ;;
        --workload)
            WORKLOAD="$2"; shift 2 ;;
        --base-only)
            BASE_ONLY=true; shift ;;
        --rebuild-base)
            REBUILD_BASE=true; shift ;;
        --no-cache)
            NO_CACHE="--no-cache"; shift ;;
        --run)
            ACTION="run"; shift ;;
        --sif)
            ACTION="sif"; shift ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate inputs
if [ -z "$VENDOR" ]; then
    echo "Usage: $0 <amd|nvidia|nvidia-gilgamesh|nvidia-athena|nvidia-voltar|nvidia-hopper1|nvidia-hopper2|intel|universal|all> [--workload <name>] [--base-only] [--run] [--sif]"
    echo ""
    echo "Builds layered Docker containers on target GPU machines via SSH."
    echo ""
    echo "Architecture:"
    echo "  Base image (leo-base-*):     HPCToolkit + Dyninst + Leo"
    echo "  Workload image (leo-*-*):    Benchmark built on top of base"
    echo ""
    echo "Target machines:"
    echo "  amd        -> odyssey    (4x MI300A, gfx942)"
    echo "  nvidia           -> illyad     (H100, sm_90)"
    echo "  nvidia-gilgamesh -> gilgamesh  (H100, sm_90)"
    echo "  nvidia-athena    -> athena     (4x A100-SXM4-40GB, sm_80)"
    echo "  nvidia-voltar    -> voltar     (A100-80GB + V100 + P100, sm_80)"
    echo "  nvidia-hopper1   -> hopper1    (GH200, sm_90, ARM)"
    echo "  nvidia-hopper2   -> hopper2    (GH200, sm_90, ARM)"
    echo "  intel      -> headroom   (PVC 1100)"
    echo "  universal  -> local      (all disassemblers, no GPU needed)"
    echo ""
    echo "Options:"
    echo "  --workload <name>  Workload to build (default: rajaperf)"
    echo "                     Available: $VALID_WORKLOADS"
    echo "  --base-only        Build only the base image (skip workload)"
    echo "  --rebuild-base     Force rebuild of the base image"
    echo "  --run              Open interactive shell after build"
    echo "  --sif              Convert to Singularity/Apptainer image"
    echo ""
    echo "GPU architecture overrides (edit ARCH_ARGS in this script):"
    echo "  AMD:    GPU_TARGET=gfx942 (MI300A) or gfx90a (MI210)"
    echo "  NVIDIA: GPU_ARCH=90 (H100) or 80 (A100)"
    exit 1
fi

# Validate workload
if [ "$BASE_ONLY" = false ]; then
    if ! echo "$VALID_WORKLOADS" | grep -qw "$WORKLOAD"; then
        echo "ERROR: Unknown workload '$WORKLOAD'. Available: $VALID_WORKLOADS"
        exit 1
    fi
fi

# ============================================
# Build functions
# ============================================

build_base() {
    local vendor=$1
    local machine="${2:-${MACHINES[$vendor]}}"
    local arch_arg="${3:-${ARCH_ARGS[$vendor]}}"

    # Check if base image already exists (skip unless --rebuild-base)
    if [ "$REBUILD_BASE" = false ]; then
        local base_exists
        if [ -n "$machine" ]; then
            base_exists=$(ssh "$machine" "docker images -q leo-base-$vendor:latest 2>/dev/null" 2>/dev/null || true)
        else
            base_exists=$(docker images -q leo-base-$vendor:latest 2>/dev/null || true)
        fi
        if [ -n "$base_exists" ]; then
            echo "  Base image leo-base-$vendor:latest exists${machine:+ on $machine} (use --rebuild-base to force)"
            return 0
        fi
    fi

    echo "  Building base image leo-base-$vendor:latest${machine:+ on $machine}..."
    if [ -n "$machine" ]; then
        ssh "$machine" "cd $LEO_ROOT && DOCKER_BUILDKIT=0 docker build \
            -f $DOCKER_DIR/Dockerfile.base-$vendor \
            -t leo-base-$vendor:latest \
            $arch_arg \
            ." 2>&1
    else
        cd "$LEO_ROOT" && docker build \
            -f "$DOCKER_DIR/Dockerfile.base-$vendor" \
            -t "leo-base-$vendor:latest" \
            $arch_arg \
            . 2>&1
    fi

    echo "  Base image built: leo-base-$vendor:latest"
}

build_hpctoolkit() {
    local vendor=$1
    local machine="${2:-${MACHINES[$vendor]}}"
    local arch_arg="${3:-${ARCH_ARGS[$vendor]}}"
    local dockerfile="$DOCKER_DIR/Dockerfile.hpctoolkit-$vendor"

    # Skip if no hpctoolkit layer exists for this vendor (falls back to base)
    if [ ! -f "$dockerfile" ]; then
        return 0
    fi

    # Skip if hpctoolkit image already exists (unless --rebuild-base or --no-cache)
    if [ "$REBUILD_BASE" = false ] && [ -z "$NO_CACHE" ]; then
        local hpc_exists
        if [ -n "$machine" ]; then
            hpc_exists=$(ssh "$machine" "docker images -q leo-hpctoolkit-$vendor:latest 2>/dev/null" 2>/dev/null || true)
        else
            hpc_exists=$(docker images -q leo-hpctoolkit-$vendor:latest 2>/dev/null || true)
        fi
        if [ -n "$hpc_exists" ]; then
            echo "  HPCToolkit image leo-hpctoolkit-$vendor:latest exists${machine:+ on $machine} (use --rebuild-base to force)"
            return 0
        fi
    fi

    echo "  Building HPCToolkit image leo-hpctoolkit-$vendor:latest${machine:+ on $machine}..."
    if [ -n "$machine" ]; then
        ssh "$machine" "cd $DOCKER_DIR && DOCKER_BUILDKIT=0 docker build \
            $NO_CACHE \
            -f $DOCKER_DIR/Dockerfile.hpctoolkit-$vendor \
            -t leo-hpctoolkit-$vendor:latest \
            --build-arg BASE_TAG=latest \
            $arch_arg \
            ." 2>&1
    else
        cd "$DOCKER_DIR" && docker build \
            $NO_CACHE \
            -f "$DOCKER_DIR/Dockerfile.hpctoolkit-$vendor" \
            -t "leo-hpctoolkit-$vendor:latest" \
            --build-arg BASE_TAG=latest \
            $arch_arg \
            . 2>&1
    fi

    echo "  HPCToolkit image built: leo-hpctoolkit-$vendor:latest"
}

build_nvhpc() {
    local vendor=$1
    local machine="${2:-${MACHINES[$vendor]}}"
    local arch_arg="${3:-${ARCH_ARGS[$vendor]}}"
    local dockerfile="$DOCKER_DIR/Dockerfile.nvhpc-$vendor"

    # Skip if no nvhpc layer exists for this vendor
    if [ ! -f "$dockerfile" ]; then
        return 0
    fi

    # Check if nvhpc image already exists (skip unless --rebuild-base)
    if [ "$REBUILD_BASE" = false ]; then
        local nvhpc_exists
        if [ -n "$machine" ]; then
            nvhpc_exists=$(ssh "$machine" "docker images -q leo-nvhpc-$vendor:latest 2>/dev/null" 2>/dev/null || true)
        else
            nvhpc_exists=$(docker images -q leo-nvhpc-$vendor:latest 2>/dev/null || true)
        fi
        if [ -n "$nvhpc_exists" ]; then
            echo "  NVHPC image leo-nvhpc-$vendor:latest exists${machine:+ on $machine} (use --rebuild-base to force)"
            return 0
        fi
    fi

    echo "  Building NVHPC image leo-nvhpc-$vendor:latest${machine:+ on $machine}..."
    if [ -n "$machine" ]; then
        ssh "$machine" "cd $DOCKER_DIR && DOCKER_BUILDKIT=0 docker build \
            $NO_CACHE \
            -f $DOCKER_DIR/Dockerfile.nvhpc-$vendor \
            -t leo-nvhpc-$vendor:latest \
            --build-arg BASE_TAG=latest \
            $arch_arg \
            ." 2>&1
    else
        cd "$DOCKER_DIR" && docker build \
            $NO_CACHE \
            -f "$DOCKER_DIR/Dockerfile.nvhpc-$vendor" \
            -t "leo-nvhpc-$vendor:latest" \
            --build-arg BASE_TAG=latest \
            $arch_arg \
            . 2>&1
    fi

    echo "  NVHPC image built: leo-nvhpc-$vendor:latest"
}

build_workload() {
    local vendor=$1
    local workload=$2
    local machine="${3:-${MACHINES[$vendor]}}"
    local arch_arg="${4:-${ARCH_ARGS[$vendor]}}"
    local image_name="leo-${workload}-${vendor}"
    local dockerfile="$DOCKER_DIR/Dockerfile.$workload-$vendor"

    # Workload Dockerfiles don't COPY from build context (they git clone inside).
    # Use the docker/ directory as context (~100KB) instead of $LEO_ROOT (~17GB)
    # to avoid slow NFS context transfer. If a workload Dockerfile does need COPY
    # from the local build context (not COPY --from=<stage>), fall back to full repo.
    local context_dir="$DOCKER_DIR"
    if grep -q '^COPY [^-]' "$dockerfile" 2>/dev/null; then
        context_dir="$LEO_ROOT"
    fi

    echo "  Building workload image $image_name:latest${machine:+ on $machine}..."
    if [ -n "$machine" ]; then
        ssh "$machine" "cd $context_dir && DOCKER_BUILDKIT=0 docker build \
            $NO_CACHE \
            -f $DOCKER_DIR/Dockerfile.$workload-$vendor \
            -t $image_name:latest \
            --build-arg BASE_TAG=latest \
            $arch_arg \
            ." 2>&1
    else
        cd "$context_dir" && docker build \
            $NO_CACHE \
            -f "$DOCKER_DIR/Dockerfile.$workload-$vendor" \
            -t "$image_name:latest" \
            --build-arg BASE_TAG=latest \
            $arch_arg \
            . 2>&1
    fi

    echo "  Workload image built: $image_name:latest"

    # Tag legacy alias for rajaperf (backward compatibility)
    if [ "$workload" = "rajaperf" ]; then
        if [ -n "$machine" ]; then
            ssh "$machine" "docker tag $image_name:latest leo-eval-$vendor:latest" 2>/dev/null || true
        else
            docker tag "$image_name:latest" "leo-eval-$vendor:latest" 2>/dev/null || true
        fi
        echo "  Legacy alias: leo-eval-$vendor:latest -> $image_name:latest"
    fi
}

build_vendor() {
    local vendor=$1
    local machine=${MACHINES[$vendor]}
    local docker_gpu="${DOCKER_GPU_FLAGS[$vendor]}"

    # Map vendor aliases to Docker image/Dockerfile suffix
    # (nvidia-gilgamesh and nvidia-athena share nvidia Dockerfiles and image names)
    local docker_vendor="$vendor"
    if [[ "$vendor" == nvidia-gilgamesh || "$vendor" == nvidia-athena || "$vendor" == nvidia-voltar || "$vendor" == nvidia-hopper1 || "$vendor" == nvidia-hopper2 ]]; then
        docker_vendor="nvidia"
    fi
    local image_name="leo-${WORKLOAD}-${docker_vendor}"

    echo ""
    echo "============================================"
    if [ "$BASE_ONLY" = true ]; then
        echo " Building $vendor base image${machine:+ on $machine}"
    else
        echo " Building $vendor ($WORKLOAD)${machine:+ on $machine}"
    fi
    echo "============================================"
    echo ""

    if [ -n "$machine" ]; then
        # Check SSH connectivity
        if ! ssh -o ConnectTimeout=5 "$machine" "echo 'Connected to \$(hostname)'" 2>/dev/null; then
            echo "ERROR: Cannot SSH to $machine. Skipping $vendor build."
            return 1
        fi

        # Check docker availability on target
        if ! ssh "$machine" "command -v docker &>/dev/null" 2>/dev/null; then
            echo "ERROR: Docker not found on $machine. Skipping $vendor build."
            return 1
        fi
    else
        # Local build — check docker locally
        if ! command -v docker &>/dev/null; then
            echo "ERROR: Docker not found locally. Skipping $vendor build."
            return 1
        fi
    fi

    # Use the original vendor's ARCH_ARGS (e.g., nvidia-athena → GPU_ARCH=80)
    # but docker_vendor's Dockerfiles (e.g., Dockerfile.base-nvidia)
    local arch_arg="${ARCH_ARGS[$vendor]}"

    # Stage 1: Build base image (system deps, Dyninst, Leo — stable)
    build_base "$docker_vendor" "$machine" "$arch_arg"

    # Stage 2: Build HPCToolkit layer (changes frequently)
    build_hpctoolkit "$docker_vendor" "$machine" "$arch_arg"

    # Stage 3: Build NVHPC compiler layer (if needed by workload)
    build_nvhpc "$docker_vendor" "$machine" "$arch_arg"

    # Stage 4: Build workload (unless --base-only)
    if [ "$BASE_ONLY" = true ]; then
        echo ""
        echo "Base-only build complete for $vendor."
        return 0
    fi

    build_workload "$docker_vendor" "$WORKLOAD" "$machine" "$arch_arg"

    # Convert to Singularity .sif if --sif
    if [ "$ACTION" = "sif" ]; then
        local sif_name="leo-${WORKLOAD}-${vendor}.sif"
        local sif_cmd="
            if command -v singularity &>/dev/null; then
                mkdir -p $LEO_ROOT/containers
                echo 'Converting to Singularity...'
                singularity build $LEO_ROOT/containers/$sif_name \
                    docker-daemon://$image_name:latest
                echo 'Singularity image: $LEO_ROOT/containers/$sif_name'
            elif command -v apptainer &>/dev/null; then
                mkdir -p $LEO_ROOT/containers
                echo 'Converting to Apptainer...'
                apptainer build $LEO_ROOT/containers/$sif_name \
                    docker-daemon://$image_name:latest
                echo 'Apptainer image: $LEO_ROOT/containers/$sif_name'
            else
                echo 'Singularity/Apptainer not found'
            fi
        "
        if [ -n "$machine" ]; then
            ssh "$machine" "$sif_cmd" 2>&1
        else
            eval "$sif_cmd" 2>&1
        fi

        # Legacy symlink for rajaperf
        if [ "$WORKLOAD" = "rajaperf" ]; then
            local symlink_cmd="cd $LEO_ROOT/containers && ln -sf $sif_name leo-eval-$vendor.sif 2>/dev/null || true"
            if [ -n "$machine" ]; then
                ssh "$machine" "$symlink_cmd" 2>&1
            else
                eval "$symlink_cmd" 2>&1
            fi
        fi
    fi

    # Open interactive shell if --run
    if [ "$ACTION" = "run" ]; then
        echo ""
        echo "Opening interactive shell${machine:+ on $machine} ($image_name)..."
        if [ -n "$machine" ]; then
            ssh -t "$machine" "docker run --rm -it \
                $docker_gpu \
                -v $LEO_ROOT:/opt/leo \
                $image_name:latest"
        else
            docker run --rm -it \
                -v "$LEO_ROOT:/opt/leo" \
                "$image_name:latest"
        fi
    fi
}

# ============================================
# Main
# ============================================
case "$VENDOR" in
    amd|nvidia|nvidia-gilgamesh|nvidia-athena|nvidia-voltar|nvidia-hopper1|nvidia-hopper2|intel|universal)
        build_vendor "$VENDOR"
        ;;
    all)
        build_vendor "amd" || true
        build_vendor "nvidia" || true
        build_vendor "nvidia-hopper1" || true
        build_vendor "intel" || true
        build_vendor "universal" || true
        ;;
esac

echo ""
echo "============================================"
echo " Done."
echo "============================================"
