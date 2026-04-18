"""Perturbed GPU architecture wrapper for latency sensitivity analysis.

Wraps any GPUArchitecture and scales its latency() return values by a
configurable factor. All other properties/methods delegate to the base
architecture unchanged.

Used to prove that Leo's latency pruning results are robust to imprecise
latency table values (reviewer FB1 sensitivity analysis request).
"""

from typing import Tuple

from .base import GPUArchitecture


__all__ = ["PerturbedArchitecture"]


class PerturbedArchitecture(GPUArchitecture):
    """Wrapper that scales latency values by a multiplicative factor.

    Args:
        base: The real GPUArchitecture to wrap.
        latency_scale: Multiplicative factor applied to all latency values.
            1.0 = no change, 0.5 = halve latencies, 2.0 = double latencies.

    Example:
        >>> from leo.arch import get_architecture
        >>> base = get_architecture("a100")
        >>> perturbed = PerturbedArchitecture(base, latency_scale=1.5)
        >>> perturbed.latency("LDG.E")  # scaled up by 1.5x
    """

    def __init__(self, base: GPUArchitecture, latency_scale: float = 1.0):
        self._base = base
        self._latency_scale = latency_scale

    # ==================== Scaled Method ====================

    def latency(self, opcode: str) -> Tuple[int, int]:
        """Get scaled instruction latency range.

        Applies latency_scale to both min and max, with floor at 1 cycle.
        """
        min_lat, max_lat = self._base.latency(opcode)
        return (
            max(1, round(min_lat * self._latency_scale)),
            max(1, round(max_lat * self._latency_scale)),
        )

    # ==================== Delegated Abstract Properties ====================

    @property
    def name(self) -> str:
        return f"{self._base.name}(lat×{self._latency_scale:.2f})"

    @property
    def vendor(self) -> str:
        return self._base.vendor

    @property
    def inst_size(self) -> int:
        return self._base.inst_size

    @property
    def sms(self) -> int:
        return self._base.sms

    @property
    def schedulers(self) -> int:
        return self._base.schedulers

    @property
    def warps_per_sm(self) -> int:
        return self._base.warps_per_sm

    @property
    def warp_size(self) -> int:
        return self._base.warp_size

    @property
    def frequency(self) -> float:
        return self._base.frequency

    @property
    def memory_prefixes(self) -> Tuple[str, ...]:
        return self._base.memory_prefixes

    @property
    def sync_prefixes(self) -> Tuple[str, ...]:
        return self._base.sync_prefixes

    @property
    def atomic_prefixes(self) -> Tuple[str, ...]:
        return self._base.atomic_prefixes

    # ==================== Delegated Abstract Methods ====================

    def issue(self, opcode: str) -> int:
        return self._base.issue(opcode)

    def classify_opcode(self, opcode: str) -> str:
        return self._base.classify_opcode(opcode)

    def get_memory_type(self, opcode: str) -> str:
        return self._base.get_memory_type(opcode)

    # ==================== Concrete Method Overrides ====================
    # These are concrete on GPUArchitecture but overridden by some vendors
    # (e.g., NVIDIA returns True for has_static_stall_field).
    # Must delegate explicitly since inheritance would use the base default.

    @property
    def has_static_stall_field(self) -> bool:
        return self._base.has_static_stall_field

    def is_memory_op(self, opcode: str) -> bool:
        return self._base.is_memory_op(opcode)

    def is_sync_op(self, opcode: str) -> bool:
        return self._base.is_sync_op(opcode)

    def is_atomic_op(self, opcode: str) -> bool:
        return self._base.is_atomic_op(opcode)

    # ==================== Vendor-Specific Fallthrough ====================

    def __getattr__(self, name):
        """Delegate any vendor-specific non-abstract attributes to base."""
        return getattr(self._base, name)
