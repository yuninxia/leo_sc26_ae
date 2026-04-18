#!/bin/bash
# Explore Intel GPU disassembly tools on zebin/gpubin files.
#
# Runs inside the universal Docker container which has GED (from GTPin)
# and standard ELF tools. Explores what information we can extract
# beyond what Leo currently uses.
#
# Usage:
#   bash scripts/explore_intel_tools.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_DIR/results/intel_tools_explore"
mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

DOCKER_IMAGE="leo-base-universal:latest"

# Use the single-kernel test gpubin (always available)
GPUBIN_REL="tests/data/pc/intel/hpctoolkit-single.sycloffload.icpx.intelgpu-measurements/gpubins/73a6cf2fd4844baf239d0e2c13911c34.gpubin"
GPUBIN_ABS="$REPO_DIR/$GPUBIN_REL"

# Also pick a RAJAPerf gpubin for a more complex kernel
RAJAPERF_DIR="$REPO_DIR/tests/data/pc/intel/hpctoolkit-rajaperf.sycloffload.icpx.intelgpu-measurements/gpubins"
if [ -d "$RAJAPERF_DIR" ]; then
    RAJAPERF_GPUBIN="$(ls "$RAJAPERF_DIR"/*.gpubin 2>/dev/null | head -1)"
    RAJAPERF_NAME="$(basename "${RAJAPERF_GPUBIN:-none}")"
else
    RAJAPERF_GPUBIN=""
    RAJAPERF_NAME="(not available)"
fi

if [ ! -f "$GPUBIN_ABS" ]; then
    echo "ERROR: Test gpubin not found at $GPUBIN_ABS"
    exit 1
fi

echo "=== Intel GPU Tools Exploration ==="
echo "Docker image: $DOCKER_IMAGE"
echo "Single gpubin: $GPUBIN_REL"
echo "RAJAPerf gpubin: $RAJAPERF_NAME"
echo "Output dir: $OUT_DIR"
echo ""

# Build the exploration script that runs inside the container
cat > "$OUT_DIR/_explore_intel.sh" << 'INNEREOF'
#!/bin/bash
set -eo pipefail
umask 000

GPUBIN="/data/single.gpubin"
RAJAPERF="/data/rajaperf.gpubin"
OUTDIR="/output"

echo "============================================================"
echo "  PART 1: Tool Discovery"
echo "============================================================"

# 1a. Check available tools
echo "--- [1a] Available tools ---"
for tool in iga64 iga32 readelf objdump llvm-objdump c++filt python3; do
    path=$(which "$tool" 2>/dev/null || true)
    if [ -n "$path" ]; then
        echo "  $tool: $path"
    else
        echo "  $tool: NOT FOUND"
    fi
done

echo ""
echo "--- [1b] GED library search ---"
find / -name "libged*" -type f 2>/dev/null | head -10 || true
echo "  GED_LIBRARY_PATH=${GED_LIBRARY_PATH:-<not set>}"

echo ""
echo "--- [1c] IGA version ---"
if command -v iga64 &>/dev/null; then
    iga64 --help 2>&1 | head -10 || true
else
    echo "  iga64 not available"
fi

echo ""
echo "--- [1d] Intel packages installed ---"
dpkg -l 2>/dev/null | grep -iE "intel|level-zero|igc|iga|libigdfcl" | head -20 || true

echo ""
echo "============================================================"
echo "  PART 2: ELF/zebin Structure"
echo "============================================================"

# 2a. ELF header
echo "--- [2a] readelf -h (ELF header) ---"
readelf -h "$GPUBIN" > "$OUTDIR/02a_elf_header.txt" 2>&1 || true
cat "$OUTDIR/02a_elf_header.txt"

echo ""
echo "--- [2b] readelf -S (section headers) ---"
readelf -S "$GPUBIN" > "$OUTDIR/02b_elf_sections.txt" 2>&1 || true
cat "$OUTDIR/02b_elf_sections.txt"

echo ""
echo "--- [2c] readelf -s (symbol table) ---"
readelf -s "$GPUBIN" > "$OUTDIR/02c_elf_symbols.txt" 2>&1 || true
cat "$OUTDIR/02c_elf_symbols.txt"

echo ""
echo "--- [2d] readelf -n (notes) ---"
readelf -n "$GPUBIN" > "$OUTDIR/02d_elf_notes.txt" 2>&1 || true
cat "$OUTDIR/02d_elf_notes.txt"

echo ""
echo "--- [2e] readelf -p .ze_info (zeinfo metadata) ---"
readelf -p .ze_info "$GPUBIN" > "$OUTDIR/02e_zeinfo.txt" 2>&1 || true
cat "$OUTDIR/02e_zeinfo.txt"

echo ""
echo "--- [2f] readelf --debug-dump=line (DWARF line info) ---"
readelf --debug-dump=line "$GPUBIN" > "$OUTDIR/02f_dwarf_line.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/02f_dwarf_line.txt") lines"

echo ""
echo "--- [2g] readelf --debug-dump=info (DWARF debug info) ---"
readelf --debug-dump=info "$GPUBIN" > "$OUTDIR/02g_dwarf_info.txt" 2>&1 || true
echo "  -> $(wc -l < "$OUTDIR/02g_dwarf_info.txt") lines"

echo ""
echo "============================================================"
echo "  PART 3: IGA Disassembly (if available)"
echo "============================================================"

if command -v iga64 &>/dev/null; then
    # 3a. Default disassembly
    echo "--- [3a] iga64 -d (disassembly) ---"
    # Try different platforms - PVC is XeHPC
    for platform in XeHPC XeHPG XeHP XeHPCXT; do
        echo "  Trying platform: $platform"
        iga64 -d -p "$platform" "$GPUBIN" > "$OUTDIR/03a_iga_disasm_${platform}.txt" 2>&1 || true
        lines=$(wc -l < "$OUTDIR/03a_iga_disasm_${platform}.txt" 2>/dev/null || echo 0)
        echo "    -> $lines lines"
        if [ "$lines" -gt 5 ]; then
            echo "    First 20 lines:"
            head -20 "$OUTDIR/03a_iga_disasm_${platform}.txt" | sed 's/^/      /'
            break
        fi
    done

    # 3b. Try with various flags
    echo ""
    echo "--- [3b] iga64 -d flags exploration ---"
    PLATFORM="XeHPC"

    # Numeric labels (offsets instead of labels)
    iga64 -d -p "$PLATFORM" -n "$GPUBIN" > "$OUTDIR/03b_iga_numeric.txt" 2>&1 || true
    echo "  -n (numeric labels): $(wc -l < "$OUTDIR/03b_iga_numeric.txt") lines"

    # PC offsets
    iga64 -d -p "$PLATFORM" --pc "$GPUBIN" > "$OUTDIR/03c_iga_pc.txt" 2>&1 || true
    echo "  --pc (PC offsets): $(wc -l < "$OUTDIR/03c_iga_pc.txt") lines"

    # With hex bytes
    iga64 -d -p "$PLATFORM" --hex "$GPUBIN" > "$OUTDIR/03d_iga_hex.txt" 2>&1 || true
    echo "  --hex (hex bytes): $(wc -l < "$OUTDIR/03d_iga_hex.txt") lines"

    # List all options
    iga64 --help > "$OUTDIR/03e_iga_help.txt" 2>&1 || true
else
    echo "  iga64 not available - skipping IGA tests"
fi

echo ""
echo "============================================================"
echo "  PART 4: GED Exploration (via Python)"
echo "============================================================"

# 4a. Enumerate GED functions
echo "--- [4a] GED library functions ---"
python3 << 'PYEOF'
import ctypes
import os
import struct

# Find GED library
ged_path = os.environ.get("GED_LIBRARY_PATH", "")
search_paths = [
    ged_path,
    "/opt/hpctoolkit-src/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so",
    "/opt/gtpin/Profilers/Lib/intel64/libged.so",
]

lib = None
for p in search_paths:
    if p and os.path.exists(p):
        try:
            lib = ctypes.CDLL(p)
            print(f"GED loaded from: {p}")
            break
        except Exception as e:
            print(f"Failed to load {p}: {e}")

if lib is None:
    print("ERROR: GED library not found")
    exit(1)

# List all available GED_Get* functions
import subprocess
result = subprocess.run(["nm", "-D", p], capture_output=True, text=True)
ged_funcs = sorted(set(
    line.split()[-1] for line in result.stdout.splitlines()
    if "GED_" in line and " T " in line
))
print(f"\nTotal GED functions: {len(ged_funcs)}")

# Categorize
categories = {
    "Decode/Encode": [],
    "Get (field extraction)": [],
    "Set (field setting)": [],
    "Model/Platform": [],
    "Other": [],
}
for fn in ged_funcs:
    if "Decode" in fn or "Encode" in fn:
        categories["Decode/Encode"].append(fn)
    elif fn.startswith("GED_Get"):
        categories["Get (field extraction)"].append(fn)
    elif fn.startswith("GED_Set"):
        categories["Set (field setting)"].append(fn)
    elif "Model" in fn or "Platform" in fn or "Mapping" in fn:
        categories["Model/Platform"].append(fn)
    else:
        categories["Other"].append(fn)

for cat, funcs in categories.items():
    print(f"\n{cat} ({len(funcs)}):")
    for fn in funcs:
        print(f"  {fn}")
PYEOF

echo ""
echo "--- [4b] GED decode test on actual instructions ---"
python3 << 'PYEOF'
import ctypes
import os
import struct
import sys

# Find and load GED
ged_path = os.environ.get("GED_LIBRARY_PATH", "")
search_paths = [
    ged_path,
    "/opt/hpctoolkit-src/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so",
]
lib = None
for p in search_paths:
    if p and os.path.exists(p):
        try:
            lib = ctypes.CDLL(p)
            break
        except:
            pass
if not lib:
    print("GED not found")
    sys.exit(0)

# Read the gpubin ELF to find .text sections
gpubin = "/data/single.gpubin"
with open(gpubin, "rb") as f:
    data = f.read()

# Parse ELF header (64-bit)
if data[:4] != b'\x7fELF':
    print("Not an ELF file")
    sys.exit(1)

ei_class = data[4]  # 1=32bit, 2=64bit
if ei_class == 2:
    # 64-bit ELF
    e_shoff = struct.unpack_from('<Q', data, 40)[0]
    e_shentsize = struct.unpack_from('<H', data, 58)[0]
    e_shnum = struct.unpack_from('<H', data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', data, 62)[0]
else:
    print("32-bit ELF not supported")
    sys.exit(1)

# Read section headers
sections = []
for i in range(e_shnum):
    off = e_shoff + i * e_shentsize
    sh_name = struct.unpack_from('<I', data, off)[0]
    sh_type = struct.unpack_from('<I', data, off + 4)[0]
    sh_offset = struct.unpack_from('<Q', data, off + 24)[0]
    sh_size = struct.unpack_from('<Q', data, off + 32)[0]
    sections.append((sh_name, sh_type, sh_offset, sh_size))

# Get section name string table
strtab_off = sections[e_shstrndx][2]
strtab_size = sections[e_shstrndx][3]
strtab = data[strtab_off:strtab_off + strtab_size]

def get_name(idx):
    end = strtab.index(b'\x00', idx)
    return strtab[idx:end].decode('ascii')

# Find .text sections
text_sections = []
for i, (name_idx, sh_type, sh_offset, sh_size) in enumerate(sections):
    name = get_name(name_idx)
    if name.startswith('.text') and sh_size > 0:
        text_sections.append((name, sh_offset, sh_size))
        print(f"  Section: {name}  offset=0x{sh_offset:x}  size={sh_size}")

if not text_sections:
    print("No .text sections found")
    sys.exit(0)

# GED constants
GED_MODEL_XE_HPC = 23  # XeHPC model ID (PVC)
GED_RETURN_VALUE_SUCCESS = 0
COMPACT_SIZE = 8
FULL_SIZE = 16

# Setup GED functions
GED_DecodeIns = lib.GED_DecodeIns
GED_GetOpcode = lib.GED_GetOpcode
GED_InsSize = lib.GED_InsSize
GED_GetSWSB = lib.GED_GetSWSB
GED_GetExecSize = lib.GED_GetExecSize

# Try to get additional fields
optional_getters = {}
for fname in ["GED_GetAccessMode", "GED_GetSFID", "GED_GetMathFC",
              "GED_GetSendDesc", "GED_GetSendExDesc",
              "GED_GetCondModifier", "GED_GetPredCtrl",
              "GED_GetFlagRegNum", "GED_GetFlagSubRegNum",
              "GED_GetSaturation", "GED_GetMaskCtrl",
              "GED_GetChannelOffset", "GED_GetCacheOpt",
              "GED_GetDstRegFile", "GED_GetDstRegNum", "GED_GetDstSubRegNum",
              "GED_GetSrc0RegFile", "GED_GetSrc0RegNum", "GED_GetSrc0SubRegNum",
              "GED_GetSrc1RegFile", "GED_GetSrc1RegNum", "GED_GetSrc1SubRegNum",
              "GED_GetSrc2RegFile", "GED_GetSrc2RegNum", "GED_GetSrc2SubRegNum",
              "GED_GetDstDataType", "GED_GetSrc0DataType",
              "GED_GetSrc1DataType", "GED_GetSrc2DataType",
              "GED_GetCmpSrc1Imm",
              "GED_GetAccWrCtrl",
              "GED_GetRawField"]:
    try:
        optional_getters[fname] = getattr(lib, fname)
    except AttributeError:
        pass

print(f"\n  Optional GED getters available: {len(optional_getters)}")
for name in sorted(optional_getters.keys()):
    print(f"    {name}")

# Decode first text section
name, offset, size = text_sections[0]
code = data[offset:offset + size]
print(f"\n  Decoding first 20 instructions from {name}:")

ins_buf = (ctypes.c_ubyte * FULL_SIZE)()
pc = 0
count = 0
while pc < len(code) and count < 20:
    # Try full-size (16 bytes) first
    remaining = len(code) - pc
    if remaining < COMPACT_SIZE:
        break

    # Copy instruction bytes
    ins_size = min(FULL_SIZE, remaining)
    for j in range(ins_size):
        ins_buf[j] = code[pc + j]

    ret = GED_DecodeIns(ctypes.byref(ins_buf), ins_size, GED_MODEL_XE_HPC)
    if ret != GED_RETURN_VALUE_SUCCESS:
        print(f"    PC 0x{pc:04x}: decode failed (ret={ret})")
        pc += COMPACT_SIZE  # skip
        continue

    actual_size = GED_InsSize(ctypes.byref(ins_buf))
    opcode = GED_GetOpcode(ctypes.byref(ins_buf))
    swsb = GED_GetSWSB(ctypes.byref(ins_buf))
    exec_size = GED_GetExecSize(ctypes.byref(ins_buf))

    # Try optional fields
    extras = {}
    for fname, getter in optional_getters.items():
        try:
            val = getter(ctypes.byref(ins_buf))
            if val != 0 and val != 0xFFFFFFFF:  # skip default/invalid
                extras[fname.replace("GED_Get", "")] = val
        except:
            pass

    hex_bytes = " ".join(f"{code[pc+j]:02x}" for j in range(actual_size))
    compact = "(compact)" if actual_size == 8 else ""
    print(f"    PC 0x{pc:04x}: opcode={opcode:3d} swsb=0x{swsb:04x} "
          f"exec_size={exec_size} size={actual_size} {compact}")
    if extras:
        extra_str = ", ".join(f"{k}={v}" for k, v in sorted(extras.items()))
        print(f"             {extra_str}")
    print(f"             bytes: {hex_bytes}")

    pc += actual_size
    count += 1
PYEOF

echo ""
echo "============================================================"
echo "  PART 5: Compare with Leo's current Intel disassembly"
echo "============================================================"

echo "--- [5a] Leo's Intel disassembler output ---"
cd /opt/leo
cp -r /opt/leo-host/src/leo/* src/leo/ 2>/dev/null || true
export UV_PROJECT_ENVIRONMENT=/tmp/leo-venv
uv run python -c "
from leo.binary.disasm.intel import IntelDisassembler
from leo.binary.parser.zebin import ZebinParser
from pathlib import Path
import json

gpubin = Path('/data/single.gpubin')

# Parse zebin
parser = ZebinParser(gpubin)
info = parser.parse()
print(f'Kernel count: {len(info.kernels)}')
for ki in info.kernels:
    print(f'  {ki.name}: {len(ki.code)} bytes at offset 0x{ki.section_offset:x}')

# Disassemble
disasm = IntelDisassembler()
if not disasm.is_available():
    print('ERROR: Intel disassembler not available')
    exit(1)

print(f'Disassembler: {disasm.tool_name}')
results = disasm.disassemble_and_parse_all(gpubin)
print(f'Functions: {len(results)}')

for func in results[:2]:
    print(f'\\nFunction: {func.name} ({len(func.instructions)} instructions)')
    for inst in func.instructions[:15]:
        swsb_str = ''
        if inst.swsb:
            parts = []
            if inst.swsb.has_dist:
                parts.append(f'dist_type={inst.swsb.dist_type} dist_val={inst.swsb.dist_val}')
            if inst.swsb.has_sbid:
                parts.append(f'sbid_type={inst.swsb.sbid_type} sbid_val={inst.swsb.sbid_val}')
            swsb_str = f'  SWSB({", ".join(parts)})'
        dsts_str = f'dsts={inst.dsts}' if inst.dsts else ''
        srcs_str = f'srcs={inst.srcs}' if inst.srcs else ''
        ctrl_str = f'stall={inst.control.stall}' if inst.control.stall else ''
        print(f'  0x{inst.pc:04x}: {inst.op:20s} {dsts_str:15s} {srcs_str:20s} {ctrl_str} {swsb_str}')
" 2>&1 | tee "$OUTDIR/05a_leo_disasm.txt"

echo ""
echo "--- [5b] What Leo is missing (fields we could extract) ---"
python3 << 'PYEOF2'
# Compare GED fields Leo uses vs what's available
leo_uses = {
    "GED_DecodeIns", "GED_GetOpcode", "GED_InsSize", "GED_GetSWSB",
    "GED_GetExecSize",
    "GED_GetDstRegFile", "GED_GetDstRegNum",
    "GED_GetSrc0RegFile", "GED_GetSrc0RegNum",
    "GED_GetSrc1RegFile", "GED_GetSrc1RegNum",
    "GED_GetSrc2RegFile", "GED_GetSrc2RegNum",
}

potentially_useful = {
    "GED_GetSFID": "Send Function ID (memory type: SLM, A64, etc.)",
    "GED_GetSendDesc": "Send descriptor (msg type, data size, cache hints)",
    "GED_GetSendExDesc": "Extended send descriptor (surface type)",
    "GED_GetMathFC": "Math function control (sin, cos, inv, etc.)",
    "GED_GetCondModifier": "Condition modifier (for predicated ops)",
    "GED_GetPredCtrl": "Predicate control",
    "GED_GetFlagRegNum": "Flag register number",
    "GED_GetSaturation": "Saturation mode",
    "GED_GetMaskCtrl": "Mask control (NoMask, etc.)",
    "GED_GetChannelOffset": "SIMD channel offset",
    "GED_GetCacheOpt": "Cache optimization hints (L1/L3 policies)",
    "GED_GetDstSubRegNum": "Destination sub-register number",
    "GED_GetSrc0SubRegNum": "Source 0 sub-register number",
    "GED_GetDstDataType": "Destination data type (F, HF, DF, etc.)",
    "GED_GetSrc0DataType": "Source 0 data type",
    "GED_GetAccWrCtrl": "Accumulator write control (for DPAS chains)",
}

print("Fields Leo currently uses:")
for f in sorted(leo_uses):
    print(f"  {f}")

print(f"\nPotentially useful fields NOT yet used ({len(potentially_useful)}):")
for f, desc in sorted(potentially_useful.items()):
    print(f"  {f:30s} — {desc}")
PYEOF2

echo ""
echo "============================================================"
echo "  PART 6: RAJAPerf gpubin (if available)"
echo "============================================================"

if [ -f "$RAJAPERF" ]; then
    echo "--- [6a] RAJAPerf ELF sections ---"
    readelf -S "$RAJAPERF" > "$OUTDIR/06a_rajaperf_sections.txt" 2>&1 || true
    echo "  -> $(wc -l < "$OUTDIR/06a_rajaperf_sections.txt") lines"

    echo "--- [6b] RAJAPerf symbols ---"
    readelf -s "$RAJAPERF" > "$OUTDIR/06b_rajaperf_symbols.txt" 2>&1 || true
    echo "  -> $(wc -l < "$OUTDIR/06b_rajaperf_symbols.txt") lines"

    echo "--- [6c] RAJAPerf ze_info ---"
    readelf -p .ze_info "$RAJAPERF" > "$OUTDIR/06c_rajaperf_zeinfo.txt" 2>&1 || true
    echo "  -> $(wc -l < "$OUTDIR/06c_rajaperf_zeinfo.txt") lines"

    if command -v iga64 &>/dev/null; then
        echo "--- [6d] RAJAPerf IGA disassembly ---"
        iga64 -d -p XeHPC "$RAJAPERF" > "$OUTDIR/06d_rajaperf_iga.txt" 2>&1 || true
        echo "  -> $(wc -l < "$OUTDIR/06d_rajaperf_iga.txt") lines"
    fi
else
    echo "  RAJAPerf gpubin not available"
fi

echo ""
echo "=== Done ==="
INNEREOF

chmod +x "$OUT_DIR/_explore_intel.sh"

# Build volume mounts
MOUNTS=(
    -v "$GPUBIN_ABS:/data/single.gpubin:ro"
    -v "$OUT_DIR:/output"
    -v "$OUT_DIR/_explore_intel.sh:/explore.sh:ro"
    -v "$REPO_DIR:/opt/leo-host:ro"
)

# Add RAJAPerf gpubin if available
if [ -n "$RAJAPERF_GPUBIN" ] && [ -f "$RAJAPERF_GPUBIN" ]; then
    MOUNTS+=(-v "$RAJAPERF_GPUBIN:/data/rajaperf.gpubin:ro")
fi

# Run inside Docker
echo "Running exploration inside Docker container..."
docker run --rm \
    "${MOUNTS[@]}" \
    --entrypoint "" \
    "$DOCKER_IMAGE" \
    bash /explore.sh 2>&1 | tee "$OUT_DIR/exploration_log.txt"

echo ""
echo "Results saved to: $OUT_DIR/"
echo ""
echo "Key files to inspect:"
echo "  02a_elf_header.txt          - ELF header (e_machine)"
echo "  02b_elf_sections.txt        - Section headers (.text.*, .ze_info, etc.)"
echo "  02e_zeinfo.txt              - zeinfo metadata (kernel attributes)"
echo "  03a_iga_disasm_*.txt        - IGA disassembly (if available)"
echo "  03d_iga_hex.txt             - IGA with hex bytes"
echo "  05a_leo_disasm.txt          - Leo's current Intel disassembly"
echo "  exploration_log.txt         - Full exploration log"
