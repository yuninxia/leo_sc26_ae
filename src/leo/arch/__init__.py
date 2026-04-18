"""GPU architecture abstractions and latency tables.

This module provides architecture-specific information for GPU performance analysis,
supporting NVIDIA, AMD, and Intel GPUs.

NVIDIA GPUs:
- V100 (Volta, sm_70)
- A100 (Ampere, sm_80)

AMD GPUs:
- MI300 (CDNA 3, gfx940/941/942)

Intel GPUs:
- Ponte Vecchio (Xe HPC, Data Center GPU Max)

Usage:
    # Unified architecture lookup (auto-detects vendor)
    from leo.arch import get_architecture

    arch = get_architecture("a100")   # Returns A100 instance
    arch = get_architecture("mi250")  # Returns MI250 instance
    arch = get_architecture("gfx90a") # Returns MI250 instance
    arch = get_architecture("pvc")    # Returns PonteVecchio instance
"""

from typing import Dict, Tuple, Type

from leo.utils.vendor import detect_vendor_from_arch_name

from .base import GPUArchitecture
from .nvidia import NVIDIAArchitecture, V100, A100, H100
from .nvidia import get_architecture as get_nvidia_architecture
from .amd import AMDArchitecture, MI300
from .amd import get_amd_architecture
from .intel import IntelArchitecture, PonteVecchio
from .intel import get_intel_architecture
from .perturbed import PerturbedArchitecture


# Combined registry mapping architecture names to (class, vendor) tuples
_UNIFIED_ARCHITECTURES: Dict[str, Tuple[Type[GPUArchitecture], str]] = {
    # Vendor-level aliases (convenience defaults → latest architecture)
    "nvidia": (H100, "nvidia"),
    "amd": (MI300, "amd"),
    "intel": (PonteVecchio, "intel"),
    # NVIDIA architectures
    "v100": (V100, "nvidia"),
    "volta": (V100, "nvidia"),
    "sm_70": (V100, "nvidia"),
    "a100": (A100, "nvidia"),
    "ampere": (A100, "nvidia"),
    "sm_80": (A100, "nvidia"),
    "h100": (H100, "nvidia"),
    "hopper": (H100, "nvidia"),
    "sm_90": (H100, "nvidia"),
    "gh200": (H100, "nvidia"),
    # AMD architectures
    "mi100": (MI300, "amd"),
    "gfx908": (MI300, "amd"),
    "cdna1": (MI300, "amd"),
    "mi250": (MI300, "amd"),
    "mi250x": (MI300, "amd"),
    "gfx90a": (MI300, "amd"),
    "cdna2": (MI300, "amd"),
    "mi300": (MI300, "amd"),
    "mi300a": (MI300, "amd"),
    "mi300x": (MI300, "amd"),
    "gfx940": (MI300, "amd"),
    "gfx941": (MI300, "amd"),
    "gfx942": (MI300, "amd"),
    "cdna3": (MI300, "amd"),
    # Intel architectures
    "pvc": (PonteVecchio, "intel"),
    "xe_hpc": (PonteVecchio, "intel"),
    "max1100": (PonteVecchio, "intel"),
    "pontevecchio": (PonteVecchio, "intel"),
}


def _detect_vendor(name: str) -> str:
    """Auto-detect vendor from architecture name.

    Uses pattern matching on the normalized name to determine if this is
    an NVIDIA, AMD, or Intel architecture.

    Args:
        name: Architecture name (e.g., "a100", "gfx90a", "mi250", "pvc")

    Returns:
        Vendor string: "nvidia", "amd", or "intel"

    Raises:
        ValueError: If vendor cannot be determined from the name
    """
    name_lower = name.lower()

    # Intel patterns: pvc, xe_hpc, max1100, pontevecchio
    if name_lower.startswith(("pvc", "xe_", "max1", "pontevecchio")):
        return "intel"

    return detect_vendor_from_arch_name(name, default=None)


def get_architecture(name: str) -> GPUArchitecture:
    """Unified function to get GPU architecture by name with auto-detection.

    This function automatically detects the GPU vendor from the architecture name
    and returns the appropriate GPUArchitecture subclass instance.

    Supported NVIDIA names:
        - Product names: v100, a100
        - Codenames: volta, ampere
        - SM versions: sm_70, sm_80

    Supported AMD names:
        - Product names: mi100, mi250, mi300
        - Product variants: mi250x, mi300a, mi300x
        - GPU codes: gfx908, gfx90a, gfx940, gfx941, gfx942
        - Codenames: cdna1, cdna2, cdna3

    Supported Intel names:
        - Product names: pvc, pontevecchio
        - Product codes: max1100
        - Codenames: xe_hpc

    Args:
        name: Architecture name (case-insensitive).
              Examples: "a100", "v100", "mi250", "gfx90a", "pvc"

    Returns:
        GPUArchitecture instance (NVIDIAArchitecture, AMDArchitecture,
        or IntelArchitecture subclass)

    Raises:
        ValueError: If architecture name is unknown.
                   Error message includes vendor detection info and lists
                   supported architectures by vendor.

    Examples:
        >>> arch = get_architecture("a100")
        >>> arch.vendor
        'nvidia'
        >>> arch = get_architecture("mi250")
        >>> arch.vendor
        'amd'
        >>> arch = get_architecture("gfx90a")
        >>> arch.name
        'MI250'
        >>> arch = get_architecture("pvc")
        >>> arch.vendor
        'intel'
    """
    name_lower = name.lower()

    # Check if architecture is in unified registry
    if name_lower in _UNIFIED_ARCHITECTURES:
        arch_class, _vendor = _UNIFIED_ARCHITECTURES[name_lower]
        return arch_class()

    # Unknown architecture - provide helpful error message
    try:
        detected_vendor = _detect_vendor(name)
        # Vendor detected but name not in registry
        if detected_vendor == "nvidia":
            supported = sorted(
                [k for k, (_, v) in _UNIFIED_ARCHITECTURES.items() if v == "nvidia"]
            )
            raise ValueError(
                f"Unknown NVIDIA architecture '{name}'. "
                f"Supported: {', '.join(supported)}"
            )
        elif detected_vendor == "amd":
            supported = sorted(
                [k for k, (_, v) in _UNIFIED_ARCHITECTURES.items() if v == "amd"]
            )
            raise ValueError(
                f"Unknown AMD architecture '{name}'. "
                f"Supported: {', '.join(supported)}"
            )
        else:  # intel
            supported = sorted(
                [k for k, (_, v) in _UNIFIED_ARCHITECTURES.items() if v == "intel"]
            )
            raise ValueError(
                f"Unknown Intel architecture '{name}'. "
                f"Supported: {', '.join(supported)}"
            )
    except ValueError:
        # Could not detect vendor - provide full list
        nvidia_archs = sorted(
            [k for k, (_, v) in _UNIFIED_ARCHITECTURES.items() if v == "nvidia"]
        )
        amd_archs = sorted(
            [k for k, (_, v) in _UNIFIED_ARCHITECTURES.items() if v == "amd"]
        )
        intel_archs = sorted(
            [k for k, (_, v) in _UNIFIED_ARCHITECTURES.items() if v == "intel"]
        )
        raise ValueError(
            f"Unknown architecture '{name}'.\n"
            f"NVIDIA: {', '.join(nvidia_archs)}\n"
            f"AMD: {', '.join(amd_archs)}\n"
            f"Intel: {', '.join(intel_archs)}"
        )


def get_vendor(name: str) -> str:
    """Get the vendor for an architecture name.

    Args:
        name: Architecture name (case-insensitive).

    Returns:
        Vendor string: "nvidia", "amd", or "intel"

    Raises:
        ValueError: If architecture is unknown or vendor cannot be determined.
    """
    name_lower = name.lower()
    if name_lower in _UNIFIED_ARCHITECTURES:
        _, vendor = _UNIFIED_ARCHITECTURES[name_lower]
        return vendor
    return _detect_vendor(name)


__all__ = [
    # Base class
    "GPUArchitecture",
    # Unified factory (preferred)
    "get_architecture",
    "get_vendor",
    # NVIDIA
    "NVIDIAArchitecture",
    "V100",
    "A100",
    "get_nvidia_architecture",
    # AMD
    "AMDArchitecture",
    "MI300",
    "get_amd_architecture",
    # Intel
    "IntelArchitecture",
    "PonteVecchio",
    "get_intel_architecture",
    # Sensitivity analysis
    "PerturbedArchitecture",
]
