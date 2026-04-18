"""Abstract GPU architecture base class.

This module defines the abstract interface for GPU architectures,
enabling support for multiple GPU vendors (NVIDIA, AMD, etc.).
"""

from abc import ABC, abstractmethod
from typing import Tuple


class GPUArchitecture(ABC):
    """Abstract base class for GPU architecture specifications.

    This defines the interface that all GPU architectures must implement,
    including NVIDIA (V100, A100) and AMD (MI100, MI250, MI300).
    """

    # ==================== Core Properties ====================

    @property
    @abstractmethod
    def name(self) -> str:
        """Architecture name (e.g., 'V100', 'MI250')."""
        pass

    @property
    @abstractmethod
    def vendor(self) -> str:
        """GPU vendor ('nvidia' or 'amd')."""
        pass

    @property
    @abstractmethod
    def inst_size(self) -> int:
        """Instruction size in bytes.

        - NVIDIA: 16 bytes (128-bit fixed-width instructions)
        - AMD: 4 bytes (32-bit base, some 64-bit)
        """
        pass

    @property
    @abstractmethod
    def sms(self) -> int:
        """Number of compute units.

        - NVIDIA: Streaming Multiprocessors (SMs)
        - AMD: Compute Units (CUs)
        """
        pass

    @property
    @abstractmethod
    def schedulers(self) -> int:
        """Number of warp/wavefront schedulers per compute unit."""
        pass

    @property
    @abstractmethod
    def warps_per_sm(self) -> int:
        """Maximum warps/wavefronts per compute unit."""
        pass

    @property
    @abstractmethod
    def warp_size(self) -> int:
        """Threads per warp/wavefront.

        - NVIDIA: 32 (warp)
        - AMD: 64 (wavefront/wave64) or 32 (wave32 on RDNA)
        """
        pass

    @property
    @abstractmethod
    def frequency(self) -> float:
        """Clock frequency in GHz."""
        pass

    # ==================== Latency Methods ====================

    @abstractmethod
    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get instruction latency range (min, max) in cycles.

        Args:
            opcode: Instruction opcode string (e.g., "LDG.E.64", "FADD",
                    "global_load_dword", "v_add_f32")

        Returns:
            Tuple of (min_latency, max_latency) in cycles.
        """
        pass

    @abstractmethod
    def issue(self, opcode: str) -> int:
        """Get instruction issue rate (throughput) in cycles.

        This is how many cycles must pass before another instruction
        of the same type can be issued.

        Args:
            opcode: Instruction opcode string

        Returns:
            Issue latency in cycles.
        """
        pass

    # ==================== Opcode Classification ====================

    @abstractmethod
    def classify_opcode(self, opcode: str) -> str:
        """Classify opcode into a category.

        Standard categories (vendor-agnostic):
        - "INTEGER": Integer arithmetic and logic
        - "FLOAT": Floating-point operations
        - "MUFU": Transcendental/special functions (sin, cos, rcp, etc.)
        - "TENSOR": Matrix/tensor core operations
        - "MEMORY": Load/store operations
        - "CONTROL": Branch, call, synchronization
        - "CONVERT": Type conversion operations
        - "PREDICATE": Predicate operations
        - "MISC": Other operations
        - "UNKNOWN": Unrecognized opcodes

        Args:
            opcode: Instruction opcode string

        Returns:
            Category string.
        """
        pass

    # ==================== Opcode Prefix Tables ====================

    @property
    @abstractmethod
    def memory_prefixes(self) -> Tuple[str, ...]:
        """Opcode prefixes that indicate memory operations."""
        pass

    @property
    @abstractmethod
    def sync_prefixes(self) -> Tuple[str, ...]:
        """Opcode prefixes that indicate synchronization operations."""
        pass

    @property
    @abstractmethod
    def atomic_prefixes(self) -> Tuple[str, ...]:
        """Opcode prefixes that indicate atomic operations."""
        pass

    @abstractmethod
    def get_memory_type(self, opcode: str) -> str:
        """Get the memory space type for a memory operation.

        Args:
            opcode: Instruction opcode string

        Returns:
            Memory type: "GLOBAL", "SHARED", "LOCAL", "CONSTANT", or "UNKNOWN"
        """
        pass

    def _matches_prefixes(self, opcode: str, prefixes: Tuple[str, ...]) -> bool:
        """Check if opcode starts with any prefix (case-insensitive)."""
        op_lower = opcode.lower()
        return any(op_lower.startswith(prefix.lower()) for prefix in prefixes)

    def is_memory_op(self, opcode: str) -> bool:
        """Check if opcode is a memory operation."""
        return self._matches_prefixes(opcode, self.memory_prefixes)

    def is_sync_op(self, opcode: str) -> bool:
        """Check if opcode is a synchronization operation."""
        return self._matches_prefixes(opcode, self.sync_prefixes)

    def is_atomic_op(self, opcode: str) -> bool:
        """Check if opcode is an atomic memory operation."""
        return self._matches_prefixes(opcode, self.atomic_prefixes)

    @property
    def has_static_stall_field(self) -> bool:
        """Whether the binary encoding includes a per-instruction stall count.

        NVIDIA SASS encodes a 4-bit stall count (0-15 cycles) in each
        instruction's control word (Control.stall). This static scheduling
        hint tells the warp scheduler how many cycles to wait before issuing
        the next instruction.

        AMD binaries do NOT have this field. However, AMD's s_nop N instruction
        explicitly inserts N+1 idle cycles, which is parsed into Control.stall.
        All other AMD instructions default to stall=1 (1 cycle per issue slot),
        providing instruction-count-based cycle estimation along CFG paths.

        Note: all vendors provide per-instruction stall *breakdown* from
        HPCToolkit PC sampling (gcycles:stl:gmem, gcycles:stl:idep, etc.)
        — this flag is only about the static binary encoding, not profiling data.

        Returns:
            True for NVIDIA, False for AMD and Intel.
        """
        return False
