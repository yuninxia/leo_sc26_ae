"""CUBIN binary parsing for control field extraction.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from leo.binary.parser instead:

    from leo.binary.parser import CubinParser, CubinSection, CubinFunction

This module re-exports all symbols from leo.binary.parser.cubin.
"""

# Re-export everything from the new location for backward compatibility
from leo.binary.parser.cubin import (
    # Classes
    CubinParser,
    CubinSection,
    CubinFunction,
    # Functions
    extract_control_from_bytes,
    parse_cubin_control_fields,
    # Constants
    MASK_STALL,
    MASK_YIELD,
    MASK_WRITE,
    MASK_READ,
    MASK_WAIT,
    MASK_REUSE,
    SHIFT_STALL,
    SHIFT_YIELD,
    SHIFT_WRITE,
    SHIFT_READ,
    SHIFT_WAIT,
    SHIFT_REUSE,
)

__all__ = [
    "CubinParser",
    "CubinSection",
    "CubinFunction",
    "extract_control_from_bytes",
    "parse_cubin_control_fields",
    "MASK_STALL",
    "MASK_YIELD",
    "MASK_WRITE",
    "MASK_READ",
    "MASK_WAIT",
    "MASK_REUSE",
    "SHIFT_STALL",
    "SHIFT_YIELD",
    "SHIFT_WRITE",
    "SHIFT_READ",
    "SHIFT_WAIT",
    "SHIFT_REUSE",
]
