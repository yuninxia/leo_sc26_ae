"""Intel GPU architecture definitions and latency tables.

Provides instruction latency and issue rate information for
Intel Data Center GPU Max (Ponte Vecchio).

Intel GEN ISA Reference:
- Ponte Vecchio: Xe HPC architecture
- Uses SIMD-16 execution model (16 lanes per EU thread)
- Primary memory instruction is 'send' with various message types

Key differences from NVIDIA/AMD:
- 16-byte (128-bit) base instruction size (uncompacted)
- 16-thread SIMD width (vs NVIDIA 32 / AMD 64)
- send-based memory model with explicit descriptors

Latency Source:
- Intel Graphics Compiler (IGC) LatencyTable.h XELatencyInfo enum
- https://github.com/intel/intel-graphics-compiler/blob/master/visa/LocalScheduler/LatencyTable.h
- DPAS formula for PVC: DPAS + RepeatCount - 1 (from LatencyTable.cpp)
- FPU/Math latencies scale with ExecSize via DELTA/DELTA_MATH per increment

Opcode Reference:
- Opcodes verified from HPCToolkit's GPUCFG_Intel.cpp (uses Intel IGA library)
- Source: deps/hpctoolkit/src/hpcstruct/intel/GPUCFG_Intel.cpp lines 116-193
- Memory type (SLM vs global) is in SEND message descriptors, not opcodes

Build Environment:
- HPCToolkit built with Intel OneAPI 2025.1
- GTPin 4.5.0 for instruction-level profiling
- IGA (Intel Graphics Assembler) for disassembly
"""

from typing import Tuple

from .base import GPUArchitecture


__all__ = [
    "IntelArchitecture",
    "PonteVecchio",
    "get_intel_architecture",
]


# =============================================================================
# Intel IGA Opcodes - Verified from HPCToolkit's GPUCFG_Intel.cpp
# =============================================================================

# Memory operations (SEND-based message passing and URB)
# Note: Memory type (SLM/global) is in message descriptor, not opcode
INTEL_MEMORY_OPCODES = frozenset({
    "send",    # Send message
    "sendc",   # Send message with EOT
    "sends",   # Split send message
    "sendsc",  # Split send message with EOT
    "urb",     # Unified Return Buffer (verified via GED)
})

# Synchronization operations
INTEL_SYNC_OPCODES = frozenset({
    "sync",    # Synchronization
    "wait",    # Wait for dependencies
})

# Control flow operations
INTEL_CONTROL_OPCODES = frozenset({
    "brc",     # Branch conditional
    "brd",     # Branch divergent
    "break",   # Break from loop
    "call",    # Function call
    "calla",   # Absolute function call
    "cont",    # Continue loop
    "else",    # Else branch
    "endif",   # End if
    "goto",    # Goto
    "halt",    # Halt execution
    "if",      # If branch
    "illegal", # Illegal instruction
    "jmpi",    # Jump immediate
    "join",    # Join threads
    "ret",     # Return
    "while",   # While loop
})

# ALU operations (verified from IGA)
INTEL_ALU_OPCODES = frozenset({
    "add",     # Add
    "add3",    # 3-operand add (verified via GED)
    "addc",    # Add with carry
    "and",     # Bitwise AND
    "asr",     # Arithmetic shift right
    "avg",     # Average
    "bfe",     # Bit field extract
    "bfi1",    # Bit field insert 1
    "bfi2",    # Bit field insert 2
    "bfn",     # Boolean function (3-input, verified via GED)
    "bfrev",   # Bit field reverse
    "cbit",    # Count bits
    "cmp",     # Compare
    "cmpn",    # Compare NaN
    "csel",    # Conditional select
    "dim",     # Dimension
    "dpas",    # Dot Product Accumulate Sparse (matrix operation)
    "dpasw",   # DPAS with wider operands
    "dp2",     # Dot product 2
    "dp3",     # Dot product 3
    "dp4",     # Dot product 4
    "dp4a",    # Dot product 4 accumulate
    "dph",     # Dot product homogeneous
    "f16to32", # Float16 to float32
    "f32to16", # Float32 to float16
    "fbh",     # Find first bit high
    "fbl",     # Find first bit low
    "frc",     # Fraction
    "line",    # Line
    "lrp",     # Linear interpolation
    "lzd",     # Leading zero detect
    "mac",     # Multiply accumulate
    "mach",    # Multiply accumulate high
    "macl",    # Multiply accumulate low (verified via GED)
    "mad",     # Multiply add
    "madm",    # Multiply add for math
    "math",    # Math operation
    "mov",     # Move
    "movi",    # Move immediate
    "mul",     # Multiply
    "nop",     # No operation
    "not",     # Bitwise NOT
    "or",      # Bitwise OR
    "pln",     # Plane
    "rndd",    # Round down
    "rnde",    # Round even
    "rndu",    # Round up
    "rndz",    # Round zero
    "rol",     # Rotate left
    "ror",     # Rotate right
    "sad2",    # Sum of absolute differences 2
    "sada2",   # Sum of absolute differences accumulate 2
    "sel",     # Select
    "shl",     # Shift left
    "shr",     # Shift right
    "smov",    # Scattered move
    "subb",    # Subtract with borrow
    "xor",     # Bitwise XOR
})

# Prefix tuples for base class compatibility
INTEL_MEMORY_PREFIXES = ("send",)  # All memory ops start with "send"
INTEL_SYNC_PREFIXES = ("sync", "wait")
INTEL_ATOMIC_PREFIXES = ()  # Atomics are done via SEND with specific descriptors


class IntelArchitecture(GPUArchitecture):
    """Base class for Intel GPU architectures.

    Opcode classification is based on verified IGA opcodes from HPCToolkit.
    """

    @property
    def vendor(self) -> str:
        return "intel"

    @property
    def has_static_stall_field(self) -> bool:
        """Intel uses aggregate stall counters (PC sampling), not per-instruction fields."""
        return False

    @property
    def inst_size(self) -> int:
        # Intel GEN ISA uses 16-byte (128-bit) uncompacted instructions
        # Some instructions can be compacted to 8 bytes
        return 16

    @property
    def warp_size(self) -> int:
        # Intel calls this "SIMD width" - 16 lanes per EU thread
        return 16

    # Alias for Intel terminology
    @property
    def simd_width(self) -> int:
        """SIMD lanes per EU thread (Intel terminology for warp_size)."""
        return self.warp_size

    @property
    def memory_prefixes(self) -> Tuple[str, ...]:
        return INTEL_MEMORY_PREFIXES

    @property
    def sync_prefixes(self) -> Tuple[str, ...]:
        return INTEL_SYNC_PREFIXES

    @property
    def atomic_prefixes(self) -> Tuple[str, ...]:
        return INTEL_ATOMIC_PREFIXES

    def is_memory_op(self, opcode: str) -> bool:
        """Check if opcode is a memory operation.

        Uses verified IGA opcode set. All Intel memory operations use
        SEND-based message passing.
        """
        op_lower = opcode.lower()
        return op_lower in INTEL_MEMORY_OPCODES

    def is_sync_op(self, opcode: str) -> bool:
        """Check if opcode is a synchronization operation."""
        op_lower = opcode.lower()
        return op_lower in INTEL_SYNC_OPCODES

    def classify_opcode(self, opcode: str) -> str:
        """Classify Intel opcode into a category.

        Categories based on verified IGA opcodes from HPCToolkit:
        - MEMORY: send, sendc, sends, sendsc
        - SYNC: sync, wait
        - CONTROL: jmpi, call, ret, goto, if, else, endif, while, break, etc.
        - ALU: all other operations (add, mul, mov, mad, etc.)
        """
        op_lower = opcode.lower()

        # Memory operations (SEND-based)
        if op_lower in INTEL_MEMORY_OPCODES:
            return "MEMORY"

        # Synchronization operations
        if op_lower in INTEL_SYNC_OPCODES:
            return "SYNC"

        # Control flow
        if op_lower in INTEL_CONTROL_OPCODES:
            return "CONTROL"

        # Default: ALU operations
        # This includes all verified ALU ops plus any unknown opcodes
        return "ALU"

    def get_memory_type(self, opcode: str) -> str:
        """Get the memory space type for a memory operation.

        Note: Intel SEND instructions encode memory type in message descriptors,
        not in the opcode itself. This method returns "global" for all SEND
        operations since the actual memory type requires descriptor analysis.

        For accurate memory type detection, the disassembler output or
        message descriptor would need to be parsed.

        Returns:
            "global" for SEND operations (conservative assumption),
            "none" for non-memory operations
        """
        op_lower = opcode.lower()

        # All SEND-based operations - assume global memory
        # Actual type (SLM vs global) is in message descriptor
        if op_lower in INTEL_MEMORY_OPCODES:
            return "global"

        return "none"


class PonteVecchio(IntelArchitecture):
    """Intel Data Center GPU Max (Ponte Vecchio, Xe HPC) architecture."""

    @property
    def name(self) -> str:
        return "PonteVecchio"

    @property
    def sms(self) -> int:
        # Intel calls these "Xe cores"
        # PVC has 128 Xe cores (8 stacks × 16 Xe cores per stack)
        return 128

    @property
    def schedulers(self) -> int:
        # 8 vector engines per Xe core
        return 8

    @property
    def warps_per_sm(self) -> int:
        # Maximum threads per Xe core
        return 64

    @property
    def frequency(self) -> float:
        return 1.6  # GHz (boost clock ~1.6 GHz)

    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get instruction latency range for Ponte Vecchio.

        Latency values from Intel Graphics Compiler (IGC) LatencyTable.h,
        XELatencyInfo enum for Xe-HPC scheduling model.
        Source: github.com/intel/intel-graphics-compiler/.../LatencyTable.h

        FPU/Math latencies scale with ExecSize (DELTA/DELTA_MATH per increment);
        ranges below cover ExecSize 8 through 32.

        Returns:
            Tuple of (min_latency, max_latency) in cycles.
        """
        category = self.classify_opcode(opcode)
        op_lower = opcode.lower()

        if category == "ALU":
            # Math unit (transcendentals): IGC MATH=17, DELTA_MATH=4, scale 0-3
            if op_lower == "math":
                return (17, 29)

            # DPAS/DPASW (matrix ops): IGC DPAS=21, PVC formula: 21+RepeatCount-1
            if op_lower in ("dpas", "dpasw"):
                return (21, 28)

            # Standard FPU: IGC FPU_ACC=6 (min), FPU=10+DELTA*3=13 (max)
            return (6, 13)

        elif category == "MEMORY":
            # All SEND-based; memory type in descriptor, not opcode.
            # IGC: LSC_UNTYPED_L1=45 (best case), HBM miss ~500 (worst case).
            return (45, 500)

        elif category == "SYNC":
            # IGC: SLM_FENCE=23, BARRIER=30, LSC_TYPED_FENCE=60
            return (23, 100)

        elif category == "CONTROL":
            # IGC: ARF=16, BRANCH=23
            return (16, 23)

        # Default: align with ALU range
        return (6, 16)

    def issue(self, opcode: str) -> int:
        """Get instruction issue rate for Ponte Vecchio.

        Returns:
            Issue latency in cycles (typically 1).
        """
        # Most instructions can issue every cycle
        return 1


# Architecture registry
_INTEL_ARCHITECTURES = {
    "pvc": PonteVecchio,
    "pontevecchio": PonteVecchio,
    "xe_hpc": PonteVecchio,
    "max1100": PonteVecchio,
}


def get_intel_architecture(name: str) -> GPUArchitecture:
    """Get Intel architecture by name.

    Args:
        name: Architecture name (e.g., "pvc", "pontevecchio", "xe_hpc")

    Returns:
        GPUArchitecture instance.

    Raises:
        ValueError: If architecture is not supported.
    """
    name_lower = name.lower()
    if name_lower not in _INTEL_ARCHITECTURES:
        supported = ", ".join(sorted(set(_INTEL_ARCHITECTURES.keys())))
        raise ValueError(f"Unknown Intel architecture '{name}'. Supported: {supported}")
    return _INTEL_ARCHITECTURES[name_lower]()
