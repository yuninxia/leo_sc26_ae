#!/bin/bash
# Check GPU availability on target machines
# Usage: ./check_gpus.sh [node1 node2 ...]
# Default targets: odyssey gilgamesh athena headroom

TARGETS="${@:-odyssey gilgamesh athena headroom}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "============================================"
echo " GPU Availability Check"
echo " $(date)"
echo "============================================"
echo ""

for node in $TARGETS; do
    echo -e "${CYAN}>>> $node${NC}"
    echo "--------------------------------------------"

    # Check if node is reachable
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" true 2>/dev/null; then
        echo -e "  ${RED}[UNREACHABLE] Cannot SSH to $node${NC}"
        echo ""
        continue
    fi

    echo -e "  ${GREEN}[REACHABLE]${NC} SSH OK"

    # Check AMD GPUs (rocm-smi)
    amd_out=$(ssh -o ConnectTimeout=5 "$node" 'which rocm-smi 2>/dev/null && rocm-smi --showid --showproductname 2>/dev/null || echo "NO_ROCM"' 2>/dev/null)
    if [[ "$amd_out" != *"NO_ROCM"* ]]; then
        echo -e "  ${GREEN}[AMD GPU]${NC}"
        echo "$amd_out" | grep -E "GPU|Device|card" | head -8 | sed 's/^/    /'
    fi

    # Check NVIDIA GPUs (nvidia-smi)
    nvidia_out=$(ssh -o ConnectTimeout=5 "$node" 'which nvidia-smi 2>/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "NO_NVIDIA"' 2>/dev/null)
    if [[ "$nvidia_out" != *"NO_NVIDIA"* ]]; then
        echo -e "  ${GREEN}[NVIDIA GPU]${NC}"
        echo "$nvidia_out" | grep -v "^/" | head -8 | sed 's/^/    /'
    fi

    # Check Intel GPUs (clinfo or xpu-smi or sycl-ls)
    intel_out=$(ssh -o ConnectTimeout=5 "$node" 'which xpu-smi 2>/dev/null && xpu-smi discovery 2>/dev/null || (which sycl-ls 2>/dev/null && sycl-ls 2>/dev/null) || echo "NO_INTEL"' 2>/dev/null)
    if [[ "$intel_out" != *"NO_INTEL"* ]]; then
        echo -e "  ${GREEN}[INTEL GPU]${NC}"
        echo "$intel_out" | grep -v "^/" | head -8 | sed 's/^/    /'
    fi

    # Fallback: check /dev for GPU devices
    dev_out=$(ssh -o ConnectTimeout=5 "$node" 'ls /dev/kfd 2>/dev/null && echo "AMD_KFD"; ls /dev/nvidia0 2>/dev/null && echo "NVIDIA_DEV"; ls /dev/dri/renderD* 2>/dev/null' 2>/dev/null)
    if [[ -n "$dev_out" ]]; then
        echo "  [Devices]"
        echo "$dev_out" | sed 's/^/    /'
    fi

    echo ""
done

echo "============================================"
echo " Done."
echo "============================================"
