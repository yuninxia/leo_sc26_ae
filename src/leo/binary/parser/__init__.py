"""GPU binary parser module for NVIDIA CUBIN, AMD Code Object, and Intel zebin files.

This module provides a unified interface for parsing GPU binaries,
extracting sections, functions, and kernel metadata.
"""

from leo.binary.parser.base import (
    BinaryParser,
    BinarySection,
    BinaryFunction,
    KernelInfo,
    ParserError,
    get_parser,
    register_parser,
)
from leo.binary.parser.cubin import (
    CubinParser,
    CubinSection,
    CubinFunction,
    extract_control_from_bytes,
    parse_cubin_control_fields,
    # Control field masks (for direct access if needed)
    MASK_STALL,
    MASK_YIELD,
    MASK_WRITE,
    MASK_READ,
    MASK_WAIT,
    MASK_REUSE,
)
from leo.binary.parser.codeobj import (
    CodeObjectParser,
    CodeObjectSection,
    CodeObjectFunction,
    AMDKernelInfo,
)
from leo.binary.parser.zebin import (
    ZebinParser,
    ZebinSection,
    ZebinFunction,
    IntelKernelInfo,
)

__all__ = [
    # Abstract base
    "BinaryParser",
    "BinarySection",
    "BinaryFunction",
    "KernelInfo",
    "ParserError",
    # Factory
    "get_parser",
    "register_parser",
    # NVIDIA CUBIN
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
    # AMD Code Object
    "CodeObjectParser",
    "CodeObjectSection",
    "CodeObjectFunction",
    "AMDKernelInfo",
    # Intel zebin
    "ZebinParser",
    "ZebinSection",
    "ZebinFunction",
    "IntelKernelInfo",
]
