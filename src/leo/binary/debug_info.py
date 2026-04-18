"""Debug info detection for CUDA binaries.

Inspects CUBIN/GPUBIN ELF sections to detect presence of debug info
from -lineinfo or -G compilation flags.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Set

from elftools.elf.elffile import ELFFile


class DebugInfoLevel(Enum):
    """Detected debug info level in CUBIN binary."""

    FULL = "full"  # -G flag: full DWARF debug info
    LINEINFO = "lineinfo"  # -lineinfo flag: minimal line info
    NONE = "none"  # No debug info
    UNKNOWN = "unknown"  # Couldn't determine


@dataclass
class DebugInfoResult:
    """Results of debug info inspection."""

    level: DebugInfoLevel
    has_debug_line: bool
    has_debug_info: bool
    has_nv_debug_line: bool
    debug_sections: Set[str]
    recommendations: List[str]


def inspect_debug_info(cubin_path: str) -> DebugInfoResult:
    """Inspect CUBIN binary for debug info presence.

    Checks for:
    - .debug_line: DWARF line info (from -G or -lineinfo)
    - .debug_info: Full DWARF debug info (from -G)
    - .nv_debug_line: NVIDIA-specific line info

    Args:
        cubin_path: Path to .cubin or .gpubin file

    Returns:
        DebugInfoResult with detection results and recommendations
    """
    path = Path(cubin_path)
    if not path.exists():
        raise FileNotFoundError(f"Binary not found: {cubin_path}")

    debug_sections: Set[str] = set()

    try:
        with open(path, "rb") as f:
            elf = ELFFile(f)
            for section in elf.iter_sections():
                name = section.name
                if name.startswith(".debug") or name.startswith(".nv_debug"):
                    debug_sections.add(name)
    except Exception as e:
        return DebugInfoResult(
            level=DebugInfoLevel.UNKNOWN,
            has_debug_line=False,
            has_debug_info=False,
            has_nv_debug_line=False,
            debug_sections=set(),
            recommendations=[f"Could not inspect binary: {e}"],
        )

    has_debug_info = ".debug_info" in debug_sections
    has_debug_line = ".debug_line" in debug_sections
    has_nv_debug_line = ".nv_debug_line" in debug_sections

    # Determine level
    if has_debug_info and has_debug_line:
        level = DebugInfoLevel.FULL
    elif has_debug_line or has_nv_debug_line:
        level = DebugInfoLevel.LINEINFO
    else:
        level = DebugInfoLevel.NONE

    # Generate recommendations
    recommendations: List[str] = []
    if level == DebugInfoLevel.NONE:
        recommendations = [
            "Binary compiled without debug info (-lineinfo or -G flag)",
            "Instruction-level source mapping will be unavailable",
            "Recompile with -lineinfo for detailed profiling:",
            "  nvcc ... -lineinfo ... -o kernel.cubin",
        ]

    return DebugInfoResult(
        level=level,
        has_debug_line=has_debug_line,
        has_debug_info=has_debug_info,
        has_nv_debug_line=has_nv_debug_line,
        debug_sections=debug_sections,
        recommendations=recommendations,
    )
