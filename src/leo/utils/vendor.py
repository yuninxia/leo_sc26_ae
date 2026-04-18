"""Vendor detection utilities."""

from typing import Literal, Optional
import re

Vendor = Literal["nvidia", "amd", "intel"]


def detect_vendor_from_arch_name(
    arch_name: str,
    default: Optional[Vendor] = "nvidia",
) -> Vendor:
    """Detect vendor from architecture name.

    Args:
        arch_name: Architecture name (e.g., "a100", "gfx90a", "mi250", "pvc").
        default: Vendor to return when detection fails. Use None to raise.

    Returns:
        Vendor string ("nvidia", "amd", or "intel").

    Raises:
        ValueError: If vendor cannot be determined and default is None.
    """
    name_lower = arch_name.lower()

    # Vendor-level aliases (e.g., --arch nvidia, --arch amd, --arch intel)
    if name_lower == "nvidia":
        return "nvidia"
    if name_lower == "amd":
        return "amd"
    if name_lower == "intel":
        return "intel"

    # NVIDIA patterns: v100, volta, a100, ampere, h100, hopper, gh200, sm_XX
    if name_lower.startswith(("v100", "volta", "a100", "ampere", "h100", "hopper", "gh200")):
        return "nvidia"
    if re.match(r"^sm_\d+$", name_lower):
        return "nvidia"

    # AMD patterns: miXXX, gfxXXX, cdnaX
    if name_lower.startswith(("mi", "gfx", "cdna")):
        return "amd"

    # Intel patterns: pvc, ponte, xe_hpc, max1100, pontevecchio
    if name_lower.startswith(("pvc", "ponte", "xe_", "max1")):
        return "intel"

    if default is not None:
        return default

    raise ValueError(
        f"Cannot auto-detect vendor for architecture '{arch_name}'. "
        f"Name should match NVIDIA (v100, a100, sm_70), "
        f"AMD (mi250, gfx90a, cdna2), or "
        f"Intel (pvc, xe_hpc, max1100) patterns."
    )
