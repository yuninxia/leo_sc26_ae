"""Vendor-aware metric lookup for stall cycle analysis.

This module provides explicit handling of vendor-specific metric names
used by HPCToolkit for GPU stall analysis.

Metric Differences:
    NVIDIA: Uses gcycles:stl:gmem for global memory stalls
    AMD: Uses gcycles:stl:mem for general memory stalls (async model)
    Intel: Uses gcycles:stl:send for SEND unit stalls (memory operations)

The functions here provide a clean abstraction over these differences.
"""

from typing import Dict, Literal, Optional

from leo.utils.vendor import detect_vendor_from_arch_name as _detect_vendor_from_arch_name
from leo.constants.metrics import (
    METRIC_GCYCLES_STL,
    METRIC_GCYCLES_STL_GMEM,
    METRIC_GCYCLES_STL_IDEP,
    METRIC_GCYCLES_STL_LMEM,
    METRIC_GCYCLES_STL_MEM,
    METRIC_GCYCLES_STL_SEND,
    METRIC_GCYCLES_STL_SBID,
    METRIC_GCYCLES_STL_SYNC,
    METRIC_GCYCLES_STL_CONTROL,
    METRIC_GCYCLES_STL_DIST,
    METRIC_GCYCLES_STL_INSFETCH,
    METRIC_GCYCLES_STL_OTHER,
)

Vendor = Literal["nvidia", "amd", "intel"]


# Vendor-specific metric name mappings
NVIDIA_METRICS = {
    "exec_dep": METRIC_GCYCLES_STL_IDEP,   # Execution dependency (RAW hazards)
    "mem_dep": METRIC_GCYCLES_STL_GMEM,    # Global memory stalls
    "local_mem": METRIC_GCYCLES_STL_LMEM,  # Local/shared memory stalls
    "total_stall": METRIC_GCYCLES_STL,     # Total stall cycles
}

AMD_METRICS = {
    "exec_dep": METRIC_GCYCLES_STL_IDEP,     # Execution dependency (same as NVIDIA)
    "mem_dep": METRIC_GCYCLES_STL_MEM,       # General memory stalls (AMD primary)
    "mem_dep_alt": METRIC_GCYCLES_STL_GMEM,  # Fallback (some databases have this)
    "local_mem": METRIC_GCYCLES_STL_LMEM,    # Local/shared memory stalls
    "total_stall": METRIC_GCYCLES_STL,       # Total stall cycles
}

# Intel metrics - uses standard HPCToolkit metric names
# Note: Internal Intel EU stall names (send, sbid, etc.) are mapped to
# standard names (mem, gmem, idep, etc.) before writing to database
INTEL_METRICS = {
    "exec_dep": METRIC_GCYCLES_STL_IDEP,      # Execution dependency stalls
    "mem_dep": METRIC_GCYCLES_STL_MEM,        # Memory stalls (primary)
    "mem_dep_alt": METRIC_GCYCLES_STL_GMEM,   # Global memory stalls (fallback)
    "sync": METRIC_GCYCLES_STL_SYNC,          # Synchronization stalls
    "total_stall": METRIC_GCYCLES_STL,        # Total stall cycles
}


def get_exec_dep_stall(metrics: Dict[str, float], vendor: Optional[Vendor] = None) -> float:
    """Get execution dependency stall cycles.

    Execution dependency stalls occur when an instruction waits for
    a register value from a previous instruction (RAW hazard).

    Args:
        metrics: Dictionary of metric_name -> value from profiling.
        vendor: GPU vendor ("nvidia" or "amd"). If None, tries both.

    Returns:
        Stall cycles for execution dependencies.
    """
    # Both vendors use the same metric name for exec_dep
    return metrics.get(METRIC_GCYCLES_STL_IDEP, 0.0)


def get_mem_dep_stall(metrics: Dict[str, float], vendor: Optional[Vendor] = None) -> float:
    """Get memory dependency stall cycles.

    Memory dependency stalls occur when an instruction waits for
    data from a memory load operation.

    Args:
        metrics: Dictionary of metric_name -> value from profiling.
        vendor: GPU vendor ("nvidia", "amd", or "intel"). If None, tries all.

    Returns:
        Stall cycles for memory dependencies.
    """
    if vendor == "nvidia":
        return metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
    elif vendor == "amd":
        # AMD: try primary metric first, then fallback
        value = metrics.get(METRIC_GCYCLES_STL_MEM, 0.0)
        if value == 0.0:
            value = metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
        return value
    elif vendor == "intel":
        # Intel: uses standard metric names (same as AMD)
        value = metrics.get(METRIC_GCYCLES_STL_MEM, 0.0)
        if value == 0.0:
            value = metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
        return value
    else:
        # Unknown vendor: try all relevant metrics
        return (
            metrics.get(METRIC_GCYCLES_STL_MEM, 0.0)
            + metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
            + metrics.get(METRIC_GCYCLES_STL_SEND, 0.0)
        )


def get_memory_stall_cycles(metrics: Dict[str, float], vendor: Optional[Vendor] = None) -> float:
    """Get total memory-related stall cycles (global + local).

    This aggregates all memory stall types for the vendor.

    Args:
        metrics: Dictionary of metric_name -> value from profiling.
        vendor: GPU vendor ("nvidia", "amd", or "intel"). If None, tries all.

    Returns:
        Total memory stall cycles.
    """
    local_mem = metrics.get(METRIC_GCYCLES_STL_LMEM, 0.0)

    if vendor == "nvidia":
        return metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0) + local_mem
    elif vendor == "amd":
        mem = metrics.get(METRIC_GCYCLES_STL_MEM, 0.0)
        if mem == 0.0:
            mem = metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
        return mem + local_mem
    elif vendor == "intel":
        # Intel: uses standard metric names (same as AMD)
        mem = metrics.get(METRIC_GCYCLES_STL_MEM, 0.0)
        if mem == 0.0:
            mem = metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
        return mem + local_mem
    else:
        # Unknown vendor: try all metrics
        return (
            metrics.get(METRIC_GCYCLES_STL_MEM, 0.0)
            + metrics.get(METRIC_GCYCLES_STL_GMEM, 0.0)
            + metrics.get(METRIC_GCYCLES_STL_SEND, 0.0)
            + local_mem
        )


def get_total_stall(metrics: Dict[str, float]) -> float:
    """Get total stall cycles (all stall types combined).

    Args:
        metrics: Dictionary of metric_name -> value from profiling.

    Returns:
        Total stall cycles.
    """
    return metrics.get(METRIC_GCYCLES_STL, 0.0)


def get_stall_metrics_for_pruning(
    metrics: Dict[str, float],
    vendor: Optional[Vendor] = None,
) -> tuple[float, float, float]:
    """Get all stall metrics needed for opcode pruning.

    This is the main function used by the opcode pruning logic.
    Returns execution dependency, memory dependency, and total stalls.

    For AMD, handles the async memory model where stalls may be
    attributed to s_waitcnt instead of load instructions.

    Args:
        metrics: Dictionary of metric_name -> value from profiling.
        vendor: GPU vendor ("nvidia" or "amd"). If None, tries both.

    Returns:
        Tuple of (exec_dep_lat, mem_dep_lat, total_stall).
    """
    exec_dep = get_exec_dep_stall(metrics, vendor)
    mem_dep = get_mem_dep_stall(metrics, vendor)
    total = get_total_stall(metrics)

    return exec_dep, mem_dep, total


def detect_vendor_from_arch_name(arch_name: str) -> Vendor:
    """Detect vendor from architecture name.

    Defaults to NVIDIA for unknown architecture names.
    """
    return _detect_vendor_from_arch_name(arch_name, default="nvidia")
