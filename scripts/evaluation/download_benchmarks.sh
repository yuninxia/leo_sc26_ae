#!/bin/bash
# Clone the out-of-tree benchmark baselines that the evaluation scripts
# bind-mount into their Docker containers. These are gitignored in this
# artifact (they're external projects; shipping them inflates the Zenodo
# zip unnecessarily), so reviewers need to populate them before running
# Table IV or the Section VI case studies.
#
# Currently handled:
#   benchmarks/rajaperf-h100/original   — upstream LLNL/RAJAPerf (develop)
#   benchmarks/rajaperf-h100/optimized  — yuninxia/RAJAPerf fork with Leo opts
#
# TODO: model weights for llama.cpp (Qwen2.5-1.5B Q4_K_M) live in $HOME/models
# and need a separate `huggingface-hub` download; not in scope for this helper.
#
# Usage:
#   bash scripts/evaluation/download_benchmarks.sh          # clone only if missing
#   bash scripts/evaluation/download_benchmarks.sh --force  # re-clone even if present
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FORCE=false
case "${1:-}" in
    --force) FORCE=true ;;
    --help|-h) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) echo "Unknown flag: $1 (try --help)" >&2; exit 2 ;;
esac

RAJA_DIR="$LEO_ROOT/benchmarks/rajaperf-h100"

# Upstream baseline used by the paper. Pinning via commit SHA keeps the
# baseline reproducible even if LLNL/RAJAPerf develop moves ahead.
ORIGINAL_REPO="https://github.com/LLNL/RAJAPerf.git"
ORIGINAL_COMMIT="5daceb0f7aa66d45a01fd43649f6ae848f76caf0"  # master head, 2026-03-26

# Leo-guided optimized fork.
OPTIMIZED_REPO="https://github.com/yuninxia/RAJAPerf.git"
OPTIMIZED_BRANCH="leo-optimized"

clone_pinned() {
    local dir="$1" repo="$2" ref="$3"
    if [[ -d "$dir/.git" ]] && ! $FORCE ; then
        echo "  skip: $(basename "$dir") already cloned at $dir"
        return 0
    fi
    rm -rf "$dir"
    echo "  clone: $repo → $dir  (ref=$ref)"
    git clone --recurse-submodules "$repo" "$dir"
    (cd "$dir" && git checkout "$ref" && git submodule update --init --recursive)
}

echo "=== download_benchmarks.sh — populating external baselines ==="
clone_pinned "$RAJA_DIR/original"  "$ORIGINAL_REPO"  "$ORIGINAL_COMMIT"
clone_pinned "$RAJA_DIR/optimized" "$OPTIMIZED_REPO" "$OPTIMIZED_BRANCH"

echo ""
echo "=== sizes ==="
du -sh "$RAJA_DIR/original" "$RAJA_DIR/optimized" 2>&1 | sed 's/^/  /'
echo ""
echo "done."
