#!/bin/bash
# Download the Qwen2.5-1.5B Q4_K_M GGUF model that benchmarks/llamacpp/run_compare_*.sh
# expects at $HOME/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf.
#
# This file is ~1.1 GB and is *not* bundled in the AE artifact (model
# weights are external and Apache-licensed but bulky). This script pulls
# directly from HuggingFace; reviewers behind a firewall should mirror it
# manually and pass --model to run_compare_nvidia.sh / run_compare_amd.sh
# instead.
#
# Usage:
#   bash benchmarks/llamacpp/download_model.sh           # ~/models/Qwen...gguf
#   bash benchmarks/llamacpp/download_model.sh --force   # re-download even if present
#   MODEL_DIR=/data/models bash benchmarks/llamacpp/download_model.sh
set -euo pipefail

FORCE=false
case "${1:-}" in
    --force) FORCE=true ;;
    --help|-h) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) echo "Unknown flag: $1 (try --help)" >&2; exit 2 ;;
esac

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_NAME="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
TARGET="$MODEL_DIR/$MODEL_NAME"
URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

if [[ -s "$TARGET" ]] && ! $FORCE; then
    SIZE=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET")
    echo "  skip: $TARGET already present (${SIZE} bytes). Use --force to redownload."
    sha256sum "$TARGET" 2>/dev/null || shasum -a 256 "$TARGET"
    exit 0
fi

echo "=== llamacpp download_model.sh ==="
echo "  source: $URL"
echo "  target: $TARGET"
echo ""

# curl with --location follows HF's CDN redirect; --fail aborts on 4xx/5xx.
curl --location --fail --progress-bar --output "$TARGET.partial" "$URL"
mv "$TARGET.partial" "$TARGET"

echo ""
echo "=== verify ==="
SIZE=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET")
echo "  size:   ${SIZE} bytes (~$((SIZE / 1024 / 1024)) MB)"
sha256sum "$TARGET" 2>/dev/null || shasum -a 256 "$TARGET"

# Sanity threshold: file should be > 800 MB. Anything smaller is an HF error
# page or a partial download.
if [[ $SIZE -lt 800000000 ]]; then
    echo "ERROR: downloaded file is suspiciously small (${SIZE} bytes); aborting."
    rm -f "$TARGET"
    exit 1
fi

echo ""
echo "done. Now run:"
echo "  bash benchmarks/llamacpp/run_compare_nvidia.sh"
