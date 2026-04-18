#!/bin/bash
# Explore nvdisasm and cuobjdump capabilities on NVIDIA gpubin files.
#
# Runs inside an NVIDIA CUDA Docker container to access nvdisasm/cuobjdump.
# Outputs results to results/nvidia_tools_explore/ for analysis.
#
# Usage:
#   bash scripts/explore_nvidia_tools.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_DIR/results/nvidia_tools_explore"
mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

DOCKER_IMAGE="nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04"

# Use the single-kernel test gpubin (small, always available)
GPUBIN_REL="tests/data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements/gpubins/67e7ddd42e43d0ca040956d9d9b316fa.gpubin"
GPUBIN_ABS="$REPO_DIR/$GPUBIN_REL"

# Also pick one RAJAPerf gpubin for a more complex kernel
RAJAPERF_DIR="$REPO_DIR/tests/data/pc/nvidia/hpctoolkit-rajaperf.cudaoffload.gcc.cudagpu-measurements/gpubins"
RAJAPERF_GPUBIN="$(ls "$RAJAPERF_DIR"/*.gpubin | head -1)"
RAJAPERF_NAME="$(basename "$RAJAPERF_GPUBIN")"

if [ ! -f "$GPUBIN_ABS" ]; then
    echo "ERROR: Test gpubin not found at $GPUBIN_ABS"
    exit 1
fi

echo "=== NVIDIA Tools Exploration ==="
echo "Docker image: $DOCKER_IMAGE"
echo "Single gpubin: $GPUBIN_REL"
echo "RAJAPerf gpubin: $RAJAPERF_NAME"
echo "Output dir: $OUT_DIR"
echo ""

# Build the exploration script that runs inside the container
cat > "$OUT_DIR/_explore.sh" << 'INNEREOF'
#!/bin/bash
set -euo pipefail
umask 000

GPUBIN="/data/single.gpubin"
RAJAPERF="/data/rajaperf.gpubin"
OUTDIR="/output"

echo "nvdisasm version:"
nvdisasm --version 2>&1 | head -3
echo ""
echo "cuobjdump version:"
cuobjdump --version 2>&1 | head -3
echo ""

# ============================================================
# 1. nvdisasm: Plain text (baseline, what Leo currently uses)
# ============================================================
echo "--- [1/10] nvdisasm plain text ---"
nvdisasm "$GPUBIN" > "$OUTDIR/01_nvdisasm_plain.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/01_nvdisasm_plain.txt") lines"

# ============================================================
# 2. nvdisasm -hex: Instruction encoding bytes
# ============================================================
echo "--- [2/10] nvdisasm -hex (instruction encoding) ---"
nvdisasm -hex "$GPUBIN" > "$OUTDIR/02_nvdisasm_hex.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/02_nvdisasm_hex.txt") lines"

# ============================================================
# 3. nvdisasm -json: JSON structured output
# ============================================================
echo "--- [3/10] nvdisasm -json (structured output) ---"
nvdisasm -json "$GPUBIN" > "$OUTDIR/03_nvdisasm_json.json" 2>&1 || true
echo "  -> $(wc -c < "$OUTDIR/03_nvdisasm_json.json") bytes"

# ============================================================
# 4. nvdisasm -plr: Register life ranges
# ============================================================
echo "--- [4/10] nvdisasm -plr (register life ranges) ---"
nvdisasm -plr "$GPUBIN" > "$OUTDIR/04_nvdisasm_plr.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/04_nvdisasm_plr.txt") lines"

# Also try narrow mode for compact view
nvdisasm -lrm narrow "$GPUBIN" > "$OUTDIR/04b_nvdisasm_lrm_narrow.txt" 2>&1 || true

# Also try count mode (just the number of live registers)
nvdisasm -lrm count "$GPUBIN" > "$OUTDIR/04c_nvdisasm_lrm_count.txt" 2>&1 || true

# ============================================================
# 5. nvdisasm -cfg: Control flow graph (hyperblocks)
# ============================================================
echo "--- [5/10] nvdisasm -cfg (CFG, hyperblocks) ---"
nvdisasm -cfg "$GPUBIN" > "$OUTDIR/05_nvdisasm_cfg.dot" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/05_nvdisasm_cfg.dot") lines"

# ============================================================
# 6. nvdisasm -bbcfg: Basic block CFG
# ============================================================
echo "--- [6/10] nvdisasm -bbcfg (CFG, basic blocks) ---"
nvdisasm -bbcfg "$GPUBIN" > "$OUTDIR/06_nvdisasm_bbcfg.dot" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/06_nvdisasm_bbcfg.dot") lines"

# With instruction offsets
nvdisasm -bbcfg -poff "$GPUBIN" > "$OUTDIR/06b_nvdisasm_bbcfg_poff.dot" 2>&1 || true

# ============================================================
# 7. nvdisasm -g / -gi: Source line info
# ============================================================
echo "--- [7/10] nvdisasm -g (source line info) ---"
nvdisasm -g "$GPUBIN" > "$OUTDIR/07_nvdisasm_lineinfo.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/07_nvdisasm_lineinfo.txt") lines"

nvdisasm -gi "$GPUBIN" > "$OUTDIR/07b_nvdisasm_lineinfo_inline.txt" 2>&1 || true

nvdisasm -gp "$GPUBIN" > "$OUTDIR/07c_nvdisasm_lineinfo_ptx.txt" 2>&1 || true

# ============================================================
# 8. nvdisasm -raw: Raw disassembly (no beautification)
# ============================================================
echo "--- [8/10] nvdisasm -raw (raw disassembly) ---"
nvdisasm -raw "$GPUBIN" > "$OUTDIR/08_nvdisasm_raw.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/08_nvdisasm_raw.txt") lines"

# ============================================================
# 9. cuobjdump: Various modes
# ============================================================
echo "--- [9/10] cuobjdump (various modes) ---"

# Resource usage (registers, shared mem, etc.)
cuobjdump -res-usage "$GPUBIN" > "$OUTDIR/09a_cuobjdump_res_usage.txt" 2>&1 || true
echo "  res-usage: $(wc -l < "$OUTDIR/09a_cuobjdump_res_usage.txt") lines"

# ELF sections
cuobjdump -elf "$GPUBIN" > "$OUTDIR/09b_cuobjdump_elf.txt" 2>&1 || true
echo "  elf: $(wc -l < "$OUTDIR/09b_cuobjdump_elf.txt") lines"

# Symbol names
cuobjdump -symbols "$GPUBIN" > "$OUTDIR/09c_cuobjdump_symbols.txt" 2>&1 || true
echo "  symbols: $(wc -l < "$OUTDIR/09c_cuobjdump_symbols.txt") lines"

# SASS dump
cuobjdump -sass "$GPUBIN" > "$OUTDIR/09d_cuobjdump_sass.txt" 2>&1 || true
echo "  sass: $(wc -l < "$OUTDIR/09d_cuobjdump_sass.txt") lines"

# PTX (if embedded)
cuobjdump -ptx "$GPUBIN" > "$OUTDIR/09e_cuobjdump_ptx.txt" 2>&1 || true
echo "  ptx: $(wc -l < "$OUTDIR/09e_cuobjdump_ptx.txt") lines"

# ============================================================
# 10. RAJAPerf gpubin: Repeat key analyses on a larger binary
# ============================================================
echo "--- [10/10] RAJAPerf gpubin (larger binary) ---"

nvdisasm -json "$RAJAPERF" > "$OUTDIR/10a_rajaperf_json.json" 2>&1 || true
echo "  json: $(wc -c < "$OUTDIR/10a_rajaperf_json.json") bytes"

nvdisasm -plr "$RAJAPERF" > "$OUTDIR/10b_rajaperf_plr.txt" 2>&1 || true
echo "  plr: $(wc -l < "$OUTDIR/10b_rajaperf_plr.txt") lines"

nvdisasm -hex "$RAJAPERF" > "$OUTDIR/10c_rajaperf_hex.txt" 2>&1 || true
echo "  hex: $(wc -l < "$OUTDIR/10c_rajaperf_hex.txt") lines"

cuobjdump -res-usage "$RAJAPERF" > "$OUTDIR/10d_rajaperf_res_usage.txt" 2>&1 || true
echo "  res-usage: $(wc -l < "$OUTDIR/10d_rajaperf_res_usage.txt") lines"

cuobjdump -sass "$RAJAPERF" > "$OUTDIR/10e_rajaperf_sass.txt" 2>&1 || true
echo "  sass: $(wc -l < "$OUTDIR/10e_rajaperf_sass.txt") lines"

echo ""
echo "=== Done ==="
INNEREOF

chmod +x "$OUT_DIR/_explore.sh"

# Run inside Docker
echo "Running exploration inside Docker container..."
docker run --rm \
    -v "$GPUBIN_ABS:/data/single.gpubin:ro" \
    -v "$RAJAPERF_GPUBIN:/data/rajaperf.gpubin:ro" \
    -v "$OUT_DIR:/output" \
    -v "$OUT_DIR/_explore.sh:/explore.sh:ro" \
    "$DOCKER_IMAGE" \
    bash /explore.sh

echo ""
echo "Results saved to: $OUT_DIR/"
echo ""
echo "Key files to inspect:"
echo "  01_nvdisasm_plain.txt      - Baseline (what Leo uses now)"
echo "  02_nvdisasm_hex.txt        - With instruction encoding bytes"
echo "  03_nvdisasm_json.json      - JSON structured output"
echo "  04_nvdisasm_plr.txt        - Register life ranges (wide)"
echo "  04c_nvdisasm_lrm_count.txt - Live register counts"
echo "  05_nvdisasm_cfg.dot        - CFG (hyperblocks)"
echo "  06_nvdisasm_bbcfg.dot      - CFG (basic blocks)"
echo "  09a_cuobjdump_res_usage.txt - Resource usage"
echo "  09d_cuobjdump_sass.txt     - SASS from cuobjdump"
