"""GPU occupancy calculation from kernel resource usage.

Computes theoretical occupancy — the maximum number of wavefronts (waves)
that can run concurrently on a single Compute Unit (CU) — given a kernel's
register and shared memory usage against the hardware's resource limits.

Occupancy is limited by whichever resource is exhausted first:
  max_waves = min(vgpr_limit, sgpr_limit, lds_limit, arch_limit)

This is complementary to the observed hardware utilization from temporal
PC sampling: utilization measures what fraction of execution units were
*actually active*, while occupancy measures the *theoretical upper bound*
on concurrent waves given a kernel's resource footprint.
"""

from dataclasses import dataclass
from math import floor
from typing import Optional

from leo.arch.base import GPUArchitecture
from leo.binary.parser.base import KernelInfo


@dataclass
class OccupancyResult:
    """Theoretical occupancy for a kernel on a specific architecture.

    Attributes:
        max_waves_per_cu: Maximum concurrent waves per CU (the minimum of all limits).
        arch_wave_limit: Architecture max waves per CU.
        vgpr_limited_waves: Max waves allowed by VGPR usage.
        sgpr_limited_waves: Max waves allowed by SGPR usage.
        lds_limited_waves: Max waves allowed by LDS usage.
        vgpr_count: Kernel's VGPR count per work-item.
        sgpr_count: Kernel's SGPR count per wave.
        lds_bytes: Kernel's LDS usage in bytes.
        limiting_factor: Which resource limits occupancy ("vgpr", "sgpr", "lds", "arch").
    """

    max_waves_per_cu: int
    arch_wave_limit: int
    vgpr_limited_waves: int
    sgpr_limited_waves: int
    lds_limited_waves: int
    vgpr_count: int
    sgpr_count: int
    lds_bytes: int
    limiting_factor: str

    @property
    def occupancy_pct(self) -> float:
        """Occupancy as a percentage of architecture max waves."""
        if self.arch_wave_limit <= 0:
            return 0.0
        return min(self.max_waves_per_cu / self.arch_wave_limit * 100.0, 100.0)

    @property
    def has_data(self) -> bool:
        """Whether occupancy was computable (kernel had resource metadata)."""
        return self.vgpr_count > 0


def compute_occupancy(
    kernel_info: KernelInfo,
    arch: GPUArchitecture,
) -> Optional[OccupancyResult]:
    """Compute theoretical occupancy for an AMD kernel.

    Args:
        kernel_info: Kernel metadata with register counts and LDS usage.
        arch: GPU architecture with hardware limits.

    Returns:
        OccupancyResult, or None if architecture doesn't support occupancy
        calculation (non-AMD) or kernel has no register metadata.
    """
    if arch.vendor != "amd":
        return None

    vgpr_count = kernel_info.vgpr_count
    sgpr_count = kernel_info.sgpr_count
    lds_bytes = kernel_info.shared_mem_size

    # Need at least VGPR count to compute occupancy
    if vgpr_count <= 0:
        return None

    # Architecture limits
    max_vgprs_per_cu = arch.max_vgprs_per_cu
    max_sgprs_per_wave = arch.max_sgprs_per_wave
    max_lds_per_cu = arch.max_lds_per_cu
    arch_wave_limit = arch.warps_per_sm  # max waves per CU

    # VGPR limit: total VGPRs per CU / VGPRs per wave
    # AMD allocates VGPRs in granularity of 4 (CDNA) per work-item,
    # and each wave uses vgpr_count * wave_size / simd_width VGPRs.
    # For CDNA with 4 SIMDs per CU, each SIMD has 128 VGPRs (512/4).
    # A wave occupies ceil(vgpr_count/4)*4 VGPRs on one SIMD.
    vgprs_per_simd = max_vgprs_per_cu // 4  # 128 for CDNA
    vgpr_granularity = 4  # CDNA allocates in blocks of 4
    vgprs_allocated = ((vgpr_count + vgpr_granularity - 1) // vgpr_granularity) * vgpr_granularity
    if vgprs_allocated > 0:
        waves_per_simd = floor(vgprs_per_simd / vgprs_allocated)
        vgpr_limited = waves_per_simd * 4  # 4 SIMDs per CU
    else:
        vgpr_limited = arch_wave_limit

    # SGPR limit: SGPRs are per-wave, not per-SIMD
    # CDNA allocates SGPRs in granularity of 16
    sgpr_granularity = 16
    if sgpr_count > 0:
        sgprs_allocated = ((sgpr_count + sgpr_granularity - 1) // sgpr_granularity) * sgpr_granularity
        sgprs_allocated = min(sgprs_allocated, max_sgprs_per_wave)
        # SGPRs don't typically limit occupancy on CDNA since the pool is large
        # But we still compute the limit for completeness
        sgpr_limited = arch_wave_limit  # SGPRs rarely limit on CDNA
    else:
        sgpr_limited = arch_wave_limit

    # LDS limit: total LDS per CU / LDS per workgroup × waves per workgroup
    if lds_bytes > 0:
        # LDS is allocated per workgroup, granularity of 256 bytes
        lds_granularity = 256
        lds_allocated = ((lds_bytes + lds_granularity - 1) // lds_granularity) * lds_granularity
        workgroups_per_cu = floor(max_lds_per_cu / lds_allocated)
        # Each workgroup can have multiple waves
        # We use max_flat_workgroup_size from kernel info if available
        wave_size = getattr(arch, 'wave_size', 64)
        max_workgroup_size = getattr(kernel_info, 'max_flat_workgroup_size', 256)
        waves_per_workgroup = max(1, (max_workgroup_size + wave_size - 1) // wave_size)
        lds_limited = workgroups_per_cu * waves_per_workgroup
    else:
        lds_limited = arch_wave_limit

    # Clamp all limits to arch maximum
    vgpr_limited = min(vgpr_limited, arch_wave_limit)
    sgpr_limited = min(sgpr_limited, arch_wave_limit)
    lds_limited = min(lds_limited, arch_wave_limit)

    # Occupancy = minimum of all limits
    max_waves = min(vgpr_limited, sgpr_limited, lds_limited)

    # Determine limiting factor
    if max_waves == vgpr_limited and vgpr_limited < sgpr_limited and vgpr_limited < lds_limited:
        limiting = "vgpr"
    elif max_waves == lds_limited and lds_limited < vgpr_limited:
        limiting = "lds"
    elif max_waves == sgpr_limited and sgpr_limited < vgpr_limited and sgpr_limited < lds_limited:
        limiting = "sgpr"
    elif max_waves == arch_wave_limit:
        limiting = "arch"
    else:
        limiting = "vgpr"  # Default: VGPR is the most common limiter

    return OccupancyResult(
        max_waves_per_cu=max_waves,
        arch_wave_limit=arch_wave_limit,
        vgpr_limited_waves=vgpr_limited,
        sgpr_limited_waves=sgpr_limited,
        lds_limited_waves=lds_limited,
        vgpr_count=vgpr_count,
        sgpr_count=sgpr_count,
        lds_bytes=lds_bytes,
        limiting_factor=limiting,
    )
