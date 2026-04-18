"""NVIDIA GPU architecture definitions and latency tables.

Based on GPA's GPUArchitecture.cpp for V100/A100, and Luo et al.
IPDPS 2024 microbenchmarks for H100 (arXiv:2501.12084v2).

Latency: Time from instruction issue until result is available (pipeline depth)
Issue Rate: Cycles between issuing consecutive instructions of same type (throughput)
"""

from typing import Tuple

from .base import GPUArchitecture
from .opcodes import (
    NVIDIA_CONVERT_PREFIXES,
    NVIDIA_CONTROL_PREFIXES,
    NVIDIA_DPX_PREFIXES,
    NVIDIA_FLOAT_PREFIXES,
    NVIDIA_INTEGER_PREFIXES,
    NVIDIA_MEMORY_PREFIXES,
    NVIDIA_MISC_PREFIXES,
    NVIDIA_MUFU_PREFIXES,
    NVIDIA_PREDICATE_PREFIXES,
    NVIDIA_SYNC_PREFIXES,
    NVIDIA_ATOMIC_PREFIXES,
    NVIDIA_TENSOR_PREFIXES,
    NVIDIA_TMA_PREFIXES,
    NVIDIA_WGMMA_PREFIXES,
)


# Re-export GPUArchitecture for backward compatibility
__all__ = ["GPUArchitecture", "NVIDIAArchitecture", "V100", "A100", "H100", "get_architecture"]


class NVIDIAArchitecture(GPUArchitecture):
    """Base class for NVIDIA GPU architectures."""

    @property
    def vendor(self) -> str:
        return "nvidia"

    @property
    def has_static_stall_field(self) -> bool:
        """NVIDIA encodes stall cycles in each instruction's Control word."""
        return True

    @property
    def inst_size(self) -> int:
        return 16  # 16 bytes per instruction for all NVIDIA GPUs

    @property
    def warp_size(self) -> int:
        return 32  # Always 32 threads per warp

    @property
    def memory_prefixes(self) -> Tuple[str, ...]:
        return NVIDIA_MEMORY_PREFIXES

    @property
    def sync_prefixes(self) -> Tuple[str, ...]:
        return NVIDIA_SYNC_PREFIXES

    @property
    def atomic_prefixes(self) -> Tuple[str, ...]:
        return NVIDIA_ATOMIC_PREFIXES

    def classify_opcode(self, opcode: str) -> str:
        """Classify opcode into a category for latency lookup.

        Categories match GPA's classification:
        - INTEGER: IADD, IMAD, IMUL, ISETP, etc.
        - FLOAT: FADD, FMUL, FFMA, etc.
        - MEMORY: LDG, STG, LDS, STS, etc.
        - CONVERT: I2F, F2I, etc.
        - PREDICATE: PSETP, etc.
        - CONTROL: BRA, EXIT, RET, etc.
        - MISC: MOV, S2R, etc.
        """
        op_upper = opcode.upper()

        # Hopper-specific categories (checked first; prefixes don't collide with existing)
        if any(op_upper.startswith(w) for w in NVIDIA_WGMMA_PREFIXES):
            return "WGMMA"
        if any(op_upper.startswith(t) for t in NVIDIA_TMA_PREFIXES):
            return "TMA"
        if any(op_upper.startswith(d) for d in NVIDIA_DPX_PREFIXES):
            return "DPX"

        # Check for memory operations first (most specific)
        if any(op_upper.startswith(mem) for mem in NVIDIA_MEMORY_PREFIXES):
            return "MEMORY"

        # Floating point operations
        if any(op_upper.startswith(fp) for fp in NVIDIA_FLOAT_PREFIXES):
            return "FLOAT"

        # Transcendental / MUFU operations
        if any(op_upper.startswith(mufu) for mufu in NVIDIA_MUFU_PREFIXES):
            return "MUFU"

        # Tensor operations
        if any(op_upper.startswith(tensor) for tensor in NVIDIA_TENSOR_PREFIXES):
            return "TENSOR"

        # Special/misc operations (check before INTEGER due to SH* overlap)
        if any(op_upper.startswith(misc) for misc in NVIDIA_MISC_PREFIXES):
            return "MISC"

        # Integer operations
        if any(op_upper.startswith(intop) for intop in NVIDIA_INTEGER_PREFIXES):
            return "INTEGER"

        # Predicate operations
        if any(op_upper.startswith(pred) for pred in NVIDIA_PREDICATE_PREFIXES):
            return "PREDICATE"

        # Conversion operations
        if any(op_upper.startswith(conv) for conv in NVIDIA_CONVERT_PREFIXES):
            return "CONVERT"

        # Control flow
        if any(op_upper.startswith(ctrl) for ctrl in NVIDIA_CONTROL_PREFIXES):
            return "CONTROL"

        return "UNKNOWN"

    def _get_width(self, opcode: str) -> int:
        """Extract operation width from opcode modifiers."""
        op_upper = opcode.upper()
        if ".128" in op_upper:
            return 128
        elif ".64" in op_upper or op_upper.startswith("D"):
            # D prefix indicates double precision (64-bit)
            return 64
        elif ".16" in op_upper or op_upper.startswith("H"):
            # H prefix indicates half precision (16-bit)
            return 16
        return 32

    def get_memory_type(self, opcode: str) -> str:
        """Get the memory space type for a memory operation.

        Returns:
            Memory type: "GLOBAL", "SHARED", "LOCAL", "CONSTANT", or "UNKNOWN"
        """
        op_upper = opcode.upper()

        # Check for shared memory (LDS, STS, or .S modifier)
        if any(s in op_upper for s in ["LDS", "STS", ".S"]):
            return "SHARED"

        # Check global memory BEFORE constant — LDG.E.64.CONSTANT is a global
        # load with a constant-cache hint, NOT a constant memory access.
        if any(s in op_upper for s in ["LDG", "STG", "ATOM", "RED"]):
            return "GLOBAL"

        # Constant memory: LDC / ULDC (actual constant memory space c[bank][offset])
        if "LDC" in op_upper:
            return "CONSTANT"

        # Check for local memory (LDL, STL)
        if any(s in op_upper for s in ["LDL", "STL"]):
            return "LOCAL"

        # Other LD/ST default to global
        if any(s in op_upper for s in ["LD", "ST"]):
            return "GLOBAL"

        return "UNKNOWN"


class V100(NVIDIAArchitecture):
    """NVIDIA V100 (Volta) GPU architecture."""

    @property
    def name(self) -> str:
        return "V100"

    @property
    def sms(self) -> int:
        return 80

    @property
    def schedulers(self) -> int:
        return 4

    @property
    def warps_per_sm(self) -> int:
        return 64

    @property
    def frequency(self) -> float:
        return 1.38  # GHz (boost clock varies)

    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get instruction latency range for V100.

        Based on GPA's GPUArchitecture.cpp lines 46-83.
        """
        category = self.classify_opcode(opcode)
        width = self._get_width(opcode)
        op_upper = opcode.upper()

        if category == "INTEGER":
            if ".MAD" in op_upper or op_upper.startswith("IMAD"):
                return (5, 5)
            elif "POPC" in op_upper:
                return (10, 10)
            else:
                return (4, 4)

        elif category == "PREDICATE":
            return (5, 5)

        elif category == "CONVERT":
            return (14, 14)

        elif category == "FLOAT":
            if width == 16:
                return (6, 6)
            elif width == 64:
                return (8, 8)
            else:  # 32-bit
                return (4, 4)

        elif category == "MUFU":
            return (14, 14)

        elif category == "TENSOR":
            return (8, 16)

        elif category == "MEMORY":
            if ".S" in op_upper or "LDS" in op_upper or "STS" in op_upper:
                # Shared memory
                return (19, 80)
            elif any(g in op_upper for g in ("LDG", "STG", "ATOM", "RED")):
                # Global memory (includes LDG.E.64.CONSTANT — cache hint,
                # not memory space)
                return (28, 1024)
            elif "LDC" in op_upper:
                # Constant memory (LDC, ULDC)
                return (4, 400)
            else:
                # Other LD/ST default to global/local memory
                return (28, 1024)

        elif category == "CONTROL":
            if "SYNC" in op_upper or "BAR" in op_upper:
                return (4, 25)
            else:
                return (4, 25)

        elif category == "MISC":
            if "SHFL" in op_upper:
                return (4, 25)
            elif "MOV" in op_upper:
                return (4, 4)
            else:
                return (4, 25)

        # Default fallback
        return (4, 25)

    def issue(self, opcode: str) -> int:
        """Get instruction issue rate for V100.

        Based on GPA's GPUArchitecture.cpp lines 86-118.
        """
        category = self.classify_opcode(opcode)
        width = self._get_width(opcode)
        op_upper = opcode.upper()

        if category == "INTEGER":
            return 2

        elif category == "FLOAT":
            if category == "MUFU" or "MUFU" in op_upper:
                return 8
            elif width == 16:
                return 1
            elif width == 64:
                return 4
            else:  # 32-bit
                return 2

        elif category == "MUFU":
            return 8

        elif category == "MEMORY":
            if width == 128:
                return 16
            elif width == 64:
                return 8
            else:
                return 4

        # Default
        return 2


class A100(NVIDIAArchitecture):
    """NVIDIA A100 (Ampere) GPU architecture."""

    @property
    def name(self) -> str:
        return "A100"

    @property
    def sms(self) -> int:
        return 108

    @property
    def schedulers(self) -> int:
        return 4

    @property
    def warps_per_sm(self) -> int:
        return 64

    @property
    def frequency(self) -> float:
        return 1.41  # GHz (boost clock varies)

    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get instruction latency range for A100.

        A100 uses same latency values as V100 according to GPA.
        """
        # A100 has same latencies as V100 in GPA's implementation
        return V100().latency(opcode)

    def issue(self, opcode: str) -> int:
        """Get instruction issue rate for A100.

        A100 uses same issue rates as V100 according to GPA.
        """
        return V100().issue(opcode)


class H100(NVIDIAArchitecture):
    """NVIDIA H100 (Hopper) GPU architecture.

    Also covers GH200 Grace Hopper Superchip (same GPU die).
    """

    @property
    def name(self) -> str:
        return "H100"

    @property
    def sms(self) -> int:
        return 132  # H100 SXM / GH200 full chip

    @property
    def schedulers(self) -> int:
        return 4

    @property
    def warps_per_sm(self) -> int:
        return 64

    @property
    def frequency(self) -> float:
        return 1.83  # GHz (boost clock, SXM variant)

    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get instruction latency range for H100 (Hopper).

        Based on Luo et al. IPDPS 2024 microbenchmark measurements
        (arXiv:2501.12084v2, HPMLL/NVIDIA-Hopper-Benchmark).
        """
        category = self.classify_opcode(opcode)
        width = self._get_width(opcode)
        op_upper = opcode.upper()

        if category == "INTEGER":
            if ".MAD" in op_upper or op_upper.startswith("IMAD"):
                return (4, 5)
            elif "POPC" in op_upper:
                return (10, 10)
            else:
                return (4, 4)

        elif category == "PREDICATE":
            return (5, 5)

        elif category == "CONVERT":
            return (10, 14)

        elif category == "FLOAT":
            if width == 16:
                return (4, 4)   # Hopper improved FP16 pipeline (was 6,6 on V100)
            elif width == 64:
                return (4, 8)   # Hopper doubled FP64 units (was 8,8 on V100)
            else:  # 32-bit
                return (4, 4)

        elif category == "MUFU":
            return (14, 14)

        elif category == "TENSOR":
            return (16, 24)     # Measured mma 16.0-24.5 cycles on H800 (was 8,16)

        elif category == "WGMMA":
            # Hopper warp-group matrix multiply-accumulate (async)
            # Latency depends on N dimension: N=8 RS min=13c to N=256 sparse SS=144c
            return (13, 144)

        elif category == "DPX":
            # Hopper DPX (dynamic programming accelerator, hardware-accelerated)
            # From NVIDIA-Hopper-Benchmark DPX/log/H800.log:
            #   16-bit: vibmax_s16x2=2c, viaddmax_s16x2=4c
            #   32-bit: vimax_s32_relu=2c, vimax3_s32=8c, viaddmax_s32=10c
            if "S16" in op_upper or "16X2" in op_upper:
                return (2, 4)
            return (2, 10)

        elif category == "TMA":
            # Hopper Tensor Memory Accelerator (async bulk copy)
            # Bypasses L1; TMA L2 hit ~369c, TMA global ~762c
            return (369, 762)

        elif category == "MEMORY":
            if ".S" in op_upper or "LDS" in op_upper or "STS" in op_upper:
                if "CLUSTER" in op_upper:
                    # DSM: inter-SM cluster access 181-213c, local DSM 33c
                    return (33, 213)
                # Shared memory: measured 29 cycles on H800 (was 19 min on V100)
                return (29, 80)
            elif any(g in op_upper for g in ("LDG", "STG", "ATOM", "RED")):
                # Global memory (includes LDG.E.64.CONSTANT — cache hint,
                # not memory space)
                return (32, 744)
            elif "LDC" in op_upper:
                # Constant memory (LDC, ULDC)
                return (4, 400)
            else:
                # Other LD/ST default to global/local
                return (32, 744)

        elif category == "CONTROL":
            if "SYNC" in op_upper or "BAR" in op_upper:
                return (4, 25)
            else:
                return (4, 25)

        elif category == "MISC":
            if "SHFL" in op_upper:
                return (4, 25)
            elif "MOV" in op_upper:
                return (4, 4)
            else:
                return (4, 25)

        # Default fallback
        return (4, 25)

    def issue(self, opcode: str) -> int:
        """Get instruction issue rate for H100 (Hopper).

        Based on H100 SM specification: 128 FP32/INT32 cores, 64 FP64 cores,
        128 FP16 (packed half2) per SM.
        """
        category = self.classify_opcode(opcode)
        width = self._get_width(opcode)
        op_upper = opcode.upper()

        if category == "INTEGER":
            return 1            # 128 INT32 cores (was 2 on V100 with 64 cores)

        elif category == "FLOAT":
            if category == "MUFU" or "MUFU" in op_upper:
                return 8
            elif width == 16:
                return 1
            elif width == 64:
                return 2        # 64 FP64 cores (was 4 on V100 with 32 cores)
            else:  # 32-bit
                return 1        # 128 FP32 cores (was 2 on V100 with 64 cores)

        elif category == "MUFU":
            return 8

        elif category == "WGMMA":
            return 4            # Warp-group level op (4 warps = 128 threads)

        elif category == "DPX":
            # 16-bit: ~50 ops/clk/SM → 1 cycle/warp; 32-bit: ~7 ops/clk/SM → 4 cycles/warp
            if "S16" in op_upper or "16X2" in op_upper:
                return 1
            return 4

        elif category == "TMA":
            return 4            # Limited by async TMA unit

        elif category == "MEMORY":
            if width == 128:
                return 16
            elif width == 64:
                return 8
            else:
                return 4

        # Default
        return 1


# Architecture registry
_ARCHITECTURES = {
    "v100": V100,
    "volta": V100,
    "sm_70": V100,
    "a100": A100,
    "ampere": A100,
    "sm_80": A100,
    "h100": H100,
    "hopper": H100,
    "sm_90": H100,
    "gh200": H100,
}


def get_architecture(name: str) -> GPUArchitecture:
    """Get architecture by name.

    Args:
        name: Architecture name (e.g., "v100", "a100", "sm_80")

    Returns:
        GPUArchitecture instance.

    Raises:
        ValueError: If architecture is not supported.
    """
    name_lower = name.lower()
    if name_lower not in _ARCHITECTURES:
        supported = ", ".join(sorted(_ARCHITECTURES.keys()))
        raise ValueError(f"Unknown architecture '{name}'. Supported: {supported}")
    return _ARCHITECTURES[name_lower]()
