"""AMD GPU architecture definitions and latency tables.

Provides instruction latency and issue rate information for
AMD Instinct MI300 GPUs.

AMD ISA Reference:
- MI300: CDNA 3 (gfx940/gfx941/gfx942)

Key differences from NVIDIA:
- 4-byte (32-bit) base instruction size (some 64-bit)
- 64-thread wavefronts (vs NVIDIA 32-thread warps)
- Explicit wait counters (s_waitcnt) vs implicit dependencies
- Separate VGPR/SGPR register files

Wait Counter Types (s_waitcnt):
- VM_CNT: Vector memory counter (global/buffer loads)
- LGKM_CNT: LDS, GWS, Constant, Message counter
- VS_CNT: VSKIP counter (less common)

GWS (Global Wave Sync) Instructions:
- All GWS instructions MUST be immediately followed by s_waitcnt 0
- Accessed via ds_gws_* prefix
"""

import re
from typing import Tuple, Optional

from .base import GPUArchitecture
from .opcodes import (
    AMD_ATOMIC_PREFIXES,
    AMD_CONVERT_PREFIXES,
    AMD_CONTROL_PREFIXES,
    AMD_FLOAT_PREFIXES,
    AMD_FLOAT_SUFFIXES,
    AMD_INTEGER_PREFIXES,
    AMD_INTEGER_SUFFIXES,
    AMD_MEMORY_PREFIXES,
    AMD_MISC_PREFIXES,
    AMD_MUFU_FUNCS,
    AMD_PREDICATE_PREFIXES,
    AMD_SYNC_PREFIXES,
    AMD_TENSOR_PREFIXES,
)


__all__ = [
    "AMDArchitecture", "MI300",
    "get_amd_architecture", "WaitcntType"
]


class WaitcntType:
    """Wait counter types for s_waitcnt instruction."""
    VM_CNT = "vmcnt"      # Vector memory counter
    LGKM_CNT = "lgkmcnt"  # LDS, GWS, Constant, Message counter
    VS_CNT = "vscnt"      # VSKIP counter
    EXP_CNT = "expcnt"    # Export counter


class AMDArchitecture(GPUArchitecture):
    """Base class for AMD GPU architectures."""

    @property
    def vendor(self) -> str:
        return "amd"

    @property
    def has_static_stall_field(self) -> bool:
        """AMD uses aggregate stall counters (PC sampling), not per-instruction fields."""
        return False

    @property
    def inst_size(self) -> int:
        # AMD uses 4-byte (32-bit) instructions
        # Some instructions are 8-byte (64-bit) but 4 is the base alignment
        return 4

    @property
    def warp_size(self) -> int:
        # AMD calls this "wave_size" - 64 threads per wavefront
        return 64

    # Alias for AMD terminology
    @property
    def wave_size(self) -> int:
        """Threads per wavefront (AMD terminology for warp_size)."""
        return self.warp_size

    # ==================== Occupancy Limits ====================
    # These define the hardware resource limits per CU for occupancy calculation.
    # Subclasses override where values differ across CDNA generations.

    @property
    def max_vgprs_per_cu(self) -> int:
        """Total VGPRs available per CU (shared across all waves)."""
        return 512  # CDNA 1/2/3: 512 VGPRs per CU (4 SIMDs × 128)

    @property
    def max_sgprs_per_wave(self) -> int:
        """Max SGPRs per wavefront (architectural limit)."""
        return 106  # CDNA: 106 usable SGPRs per wave (108 total, 2 reserved)

    @property
    def max_lds_per_cu(self) -> int:
        """Max LDS (shared memory) per CU in bytes."""
        return 65536  # 64 KB for all CDNA

    @property
    def memory_prefixes(self) -> Tuple[str, ...]:
        return AMD_MEMORY_PREFIXES

    @property
    def sync_prefixes(self) -> Tuple[str, ...]:
        return AMD_SYNC_PREFIXES

    @property
    def atomic_prefixes(self) -> Tuple[str, ...]:
        return AMD_ATOMIC_PREFIXES

    def classify_opcode(self, opcode: str) -> str:
        """Classify AMD opcode into a category.

        AMD instruction naming conventions:
        - v_*: Vector ALU operations
        - s_*: Scalar ALU operations
        - ds_*: Data share (LDS) operations
        - global_*: Global memory operations
        - flat_*: Flat memory operations
        - buffer_*: Buffer operations
        - scratch_*: Scratch (local) memory
        - mfma_*: Matrix operations (Tensor Core equivalent)
        """
        op_lower = opcode.lower()

        # Memory operations (check first)
        if self.is_memory_op(opcode):
            return "MEMORY"

        # Matrix/Tensor operations (MFMA = Matrix Fused Multiply-Add)
        if any(op_lower.startswith(prefix) for prefix in AMD_TENSOR_PREFIXES):
            return "TENSOR"

        # Conversion operations (check BEFORE float to avoid _f32 match)
        if any(op_lower.startswith(conv) for conv in AMD_CONVERT_PREFIXES):
            return "CONVERT"

        # Transcendental / special functions
        if any(op_lower.startswith(f"v_{fn}") for fn in AMD_MUFU_FUNCS):
            return "MUFU"

        # Floating-point operations
        if any(op_lower.startswith(fp) for fp in AMD_FLOAT_PREFIXES):
            return "FLOAT"

        # Also catch scalar float ops and general v_ float patterns
        if any(suffix in op_lower for suffix in AMD_FLOAT_SUFFIXES):
            if op_lower.startswith("v_") or op_lower.startswith("s_"):
                if not self.is_memory_op(opcode):
                    return "FLOAT"

        # Integer operations
        if any(op_lower.startswith(intop) for intop in AMD_INTEGER_PREFIXES):
            return "INTEGER"

        # Also catch patterns with _i32, _u32, _i64, _u64
        if any(suffix in op_lower for suffix in AMD_INTEGER_SUFFIXES):
            if op_lower.startswith("v_") or op_lower.startswith("s_"):
                if not self.is_memory_op(opcode):
                    return "INTEGER"

        # Control flow / synchronization (expanded with PC control instructions)
        if any(op_lower.startswith(ctrl) for ctrl in AMD_CONTROL_PREFIXES):
            return "CONTROL"

        # Move and misc scalar ops
        if any(op_lower.startswith(misc) for misc in AMD_MISC_PREFIXES):
            return "MISC"

        # Predicate/comparison results (EXEC mask manipulation)
        if any(op_lower.startswith(pred) for pred in AMD_PREDICATE_PREFIXES):
            return "PREDICATE"

        return "UNKNOWN"

    def is_gws_op(self, opcode: str) -> bool:
        """Check if opcode is a GWS (Global Wave Sync) operation.

        GWS instructions MUST be immediately followed by s_waitcnt 0.
        This is a critical requirement from the AMD ISA.
        """
        op_lower = opcode.lower()
        return op_lower.startswith("ds_gws_")

    def get_memory_type(self, opcode: str) -> str:
        """Get the memory space type for a memory operation.

        Returns:
            Memory type: "GLOBAL", "SHARED", "LOCAL", "CONSTANT", or "UNKNOWN"
        """
        op_lower = opcode.lower()

        # Global memory operations (including atomics)
        if op_lower.startswith("global_"):
            return "GLOBAL"

        # Buffer operations (typically global)
        if op_lower.startswith("buffer_"):
            return "GLOBAL"

        # Data share = LDS = Shared memory
        if op_lower.startswith("ds_"):
            return "SHARED"

        # Scalar memory loads are from constant cache
        if any(op_lower.startswith(s) for s in ["s_load_", "s_buffer_"]):
            return "CONSTANT"

        # Scratch memory = per-thread local storage
        if op_lower.startswith("scratch_"):
            return "LOCAL"

        # Flat memory can access any space (determined at runtime)
        # Default to GLOBAL for analysis purposes
        if op_lower.startswith("flat_"):
            return "GLOBAL"

        return "UNKNOWN"

    def get_waitcnt_type(self, opcode: str) -> Optional[str]:
        """Get the wait counter type for a s_waitcnt instruction.

        Returns:
            WaitcntType constant or None if not a waitcnt instruction.
        """
        op_lower = opcode.lower()
        if not op_lower.startswith("s_waitcnt"):
            return None

        if "vmcnt" in op_lower or op_lower == "s_waitcnt_vmcnt":
            return WaitcntType.VM_CNT
        elif "lgkmcnt" in op_lower or op_lower == "s_waitcnt_lgkmcnt":
            return WaitcntType.LGKM_CNT
        elif "vscnt" in op_lower or op_lower == "s_waitcnt_vscnt":
            return WaitcntType.VS_CNT
        elif "expcnt" in op_lower or op_lower == "s_waitcnt_expcnt":
            return WaitcntType.EXP_CNT

        # Plain s_waitcnt waits on all counters
        return "all"

    def get_mfma_shape(self, opcode: str) -> Optional[Tuple[int, int, int]]:
        """Extract M×N×K shape from MFMA opcode.

        MFMA opcode format: v_mfma_<dtype>_<M>x<N>x<K>[_<other>]_<src_dtype>

        Returns:
            Tuple of (M, N, K) or None if not an MFMA instruction.
        """
        op_lower = opcode.lower()
        if "mfma" not in op_lower:
            return None

        # Pattern: 32x32x8, 16x16x16, 4x4x4, etc.
        match = re.search(r'(\d+)x(\d+)x(\d+)', op_lower)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return None


class MI300(AMDArchitecture):
    """AMD Instinct MI300A/MI300X (CDNA 3, gfx940/941/942) GPU architecture."""

    @property
    def name(self) -> str:
        return "MI300"

    @property
    def sms(self) -> int:
        # MI300X: 304 CUs total (8 XCDs, 38 CUs each)
        # Report value for MI300X
        return 304

    @property
    def schedulers(self) -> int:
        return 4

    @property
    def warps_per_sm(self) -> int:
        return 40

    @property
    def frequency(self) -> float:
        return 2.10  # GHz (boost clock)

    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get instruction latency range for MI300.

        Min latencies from LLVM SISchedule.td (SIDPGFX942 model).
        Max latencies are engineering estimates for cache miss / contention.
        Key MI300 (gfx942) overrides: WriteDouble=1, WriteIntMul=1 (full-rate).
        MFMA latencies from AMD CDNA 3 ISA docs with FP8 support.
        """
        category = self.classify_opcode(opcode)
        op_lower = opcode.lower()

        if category == "INTEGER":
            # SIDPGFX942: WriteIntMul = 1 (was 4 on MI100/MI250)
            return (1, 1)       # Write32Bit = 1, WriteIntMul = 1

        elif category == "FLOAT":
            if "_f64" in op_lower:
                return (1, 1)   # SIDPGFX942: WriteDouble = 1 (full-rate FP64)
            elif "_f16" in op_lower or "_f8" in op_lower or "_bf8" in op_lower or "pk_" in op_lower:
                return (1, 1)   # Write32Bit = 1
            return (1, 1)       # WriteFloatFMA = 1

        elif category == "MUFU":
            return (4, 4)       # WriteTrans32 = 4

        elif category == "TENSOR":
            # CDNA 3 further improved MFMA with FP8 support
            shape = self.get_mfma_shape(opcode)
            if shape:
                m, n, k = shape
                # FP8 variants have better throughput
                is_fp8 = "_fp8" in op_lower or "_bf8" in op_lower
                if m >= 32 or n >= 32:
                    return (16, 16) if is_fp8 else (32, 32)
                elif m >= 16 or n >= 16:
                    return (8, 8) if is_fp8 else (16, 16)
                else:
                    return (4, 4) if is_fp8 else (8, 8)
            return (8, 16)

        elif category == "MEMORY":
            mem_type = self.get_memory_type(opcode)
            is_atomic = self.is_atomic_op(opcode)
            if mem_type == "SHARED":
                if is_atomic:
                    return (10, 40)   # LDS atomic
                return (5, 20)        # WriteLDS = 5
            elif mem_type == "CONSTANT":
                return (5, 50)        # WriteSMEM = 5
            else:
                if is_atomic:
                    return (100, 400) # Global atomic, HBM3
                return (80, 300)      # WriteVMEM = 80

        elif category == "CONTROL":
            if "barrier" in op_lower:
                return (4, 500)       # WriteBarrier = 500
            elif "waitcnt" in op_lower:
                return (1, 300)       # Variable wait
            return (1, 8)             # WriteBranch = 8

        elif category == "CONVERT":
            if "_fp8" in op_lower or "_bf8" in op_lower:
                return (1, 1)         # Fast FP8 conversion on MI300
            return (4, 4)             # WriteFloatCvt = 4

        elif category == "MISC":
            if "readlane" in op_lower or "writelane" in op_lower:
                return (1, 4)
            return (1, 1)             # WriteSALU = 1

        return (1, 8)

    def issue(self, opcode: str) -> int:
        """Get instruction issue rate for MI300."""
        category = self.classify_opcode(opcode)
        op_lower = opcode.lower()

        if category == "FLOAT" and "_f64" in op_lower:
            return 1  # Full-rate FP64 on MI300

        if category == "TENSOR":
            # CDNA 3 best MFMA throughput
            shape = self.get_mfma_shape(opcode)
            is_fp8 = "_fp8" in op_lower or "_bf8" in op_lower
            if shape:
                m, n, k = shape
                if m >= 32 or n >= 32:
                    return 1 if is_fp8 else 2
                else:
                    return 1
            return 1

        return 1


# Architecture registry
_AMD_ARCHITECTURES = {
    "mi100": MI300,
    "gfx908": MI300,
    "cdna1": MI300,
    "mi250": MI300,
    "mi250x": MI300,
    "gfx90a": MI300,
    "cdna2": MI300,
    "mi300": MI300,
    "mi300a": MI300,
    "mi300x": MI300,
    "gfx940": MI300,
    "gfx941": MI300,
    "gfx942": MI300,
    "cdna3": MI300,
}


def get_amd_architecture(name: str) -> GPUArchitecture:
    """Get AMD architecture by name.

    Args:
        name: Architecture name (e.g., "mi250", "gfx90a", "cdna2")

    Returns:
        GPUArchitecture instance.

    Raises:
        ValueError: If architecture is not supported.
    """
    name_lower = name.lower()
    if name_lower not in _AMD_ARCHITECTURES:
        supported = ", ".join(sorted(set(_AMD_ARCHITECTURES.keys())))
        raise ValueError(f"Unknown AMD architecture '{name}'. Supported: {supported}")
    return _AMD_ARCHITECTURES[name_lower]()
