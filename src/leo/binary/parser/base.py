"""Abstract base classes for GPU binary parsers.

This module defines the interface for parsing GPU binaries (NVIDIA CUBIN, AMD Code Object),
extracting code sections, function symbols, and kernel metadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from leo.utils.validation import require_file_exists


class ParserError(Exception):
    """Error parsing GPU binary."""

    pass


@dataclass
class BinarySection:
    """Abstract representation of a code section within a GPU binary.

    Attributes:
        name: Section name (e.g., ".text._Z10kernelPfi").
        offset: Byte offset in the file.
        size: Section size in bytes.
        vaddr: Virtual address (load memory address).
        data: Raw binary data.
    """

    name: str
    offset: int
    size: int
    vaddr: int
    data: bytes


@dataclass
class BinaryFunction:
    """Abstract representation of a function within a GPU binary.

    Attributes:
        name: Function symbol name.
        offset: Offset within section or file.
        size: Function size in bytes.
        section: Section name containing this function.
    """

    name: str
    offset: int
    size: int
    section: str


@dataclass
class KernelInfo:
    """Abstract kernel metadata (register counts, memory sizes, etc.).

    This is a vendor-neutral representation. AMD and NVIDIA
    have different metadata formats, but common fields are shared.

    Attributes:
        name: Kernel name.
        vgpr_count: Vector GPR count per thread/work-item.
        sgpr_count: Scalar GPR count per warp/wavefront.
        shared_mem_size: Shared/LDS memory size in bytes.
        local_mem_size: Local/scratch memory size per thread.
    """

    name: str
    vgpr_count: int = 0
    sgpr_count: int = 0
    shared_mem_size: int = 0  # NVIDIA shared / AMD LDS
    local_mem_size: int = 0  # NVIDIA local / AMD scratch
    extra: Dict[str, Any] = field(default_factory=dict)  # Vendor-specific fields


class BinaryParser(ABC):
    """Abstract base class for GPU binary parsers.

    Subclasses implement vendor-specific parsing logic for NVIDIA CUBIN
    and AMD Code Object formats.
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize parser with path to GPU binary file.

        Args:
            path: Path to the GPU binary file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ParserError: If file is not a valid GPU binary.
        """
        self.path = require_file_exists(Path(path), "GPU binary")

    @property
    @abstractmethod
    def format(self) -> str:
        """Return the binary format name ('cubin', 'codeobj', etc.)."""
        pass

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Return vendor name ('nvidia' or 'amd')."""
        pass

    @property
    @abstractmethod
    def sections(self) -> Dict[str, BinarySection]:
        """Get all code sections, keyed by section name."""
        pass

    @property
    @abstractmethod
    def functions(self) -> Dict[str, BinaryFunction]:
        """Get all function symbols, keyed by function name."""
        pass

    @abstractmethod
    def get_section_for_function(self, func_name: str) -> Optional[BinarySection]:
        """Get the code section containing a function.

        Args:
            func_name: Function name (mangled or demangled).

        Returns:
            BinarySection or None if not found.
        """
        pass

    def get_kernel_info(self, kernel_name: str) -> Optional[KernelInfo]:
        """Get metadata for a specific kernel.

        Override in subclasses that support kernel metadata extraction.

        Args:
            kernel_name: Kernel function name.

        Returns:
            KernelInfo or None if not available.
        """
        return None

    def get_all_kernel_info(self) -> Dict[str, KernelInfo]:
        """Get metadata for all kernels.

        Override in subclasses that support kernel metadata extraction.

        Returns:
            Dict mapping kernel name to KernelInfo.
        """
        return {}


# Parser registry
_PARSERS: Dict[str, type] = {}
_VENDOR_FORMAT_MAP = {
    "nvidia": "cubin",
    "amd": "codeobj",
    "intel": "zebin",
}


def register_parser(format_name: str):
    """Decorator to register a parser class.

    Args:
        format_name: Format identifier ('cubin', 'codeobj', etc.).

    Returns:
        Class decorator.

    Example:
        @register_parser("cubin")
        class CubinParser(BinaryParser):
            ...
    """

    def decorator(cls):
        _PARSERS[format_name.lower()] = cls
        return cls

    return decorator


def get_parser(format_name: str, path: Union[str, Path]) -> BinaryParser:
    """Get parser instance for a GPU binary file.

    Args:
        format_name: Format identifier ('cubin', 'codeobj', 'nvidia', 'amd').
        path: Path to the GPU binary file.

    Returns:
        BinaryParser instance.

    Raises:
        ValueError: If format is not supported.
    """
    # Map vendor names to format names for convenience
    format_lower = format_name.lower()
    format_lower = _VENDOR_FORMAT_MAP.get(format_lower, format_lower)

    if format_lower not in _PARSERS:
        supported = ", ".join(sorted(_PARSERS.keys()))
        raise ValueError(f"Unknown format '{format_name}'. Supported: {supported}")

    return _PARSERS[format_lower](path)


def get_parser_class(format_name: str) -> type:
    """Get parser class by format name.

    Args:
        format_name: Format identifier ('cubin', 'codeobj', 'nvidia', 'amd').

    Returns:
        Parser class (not instance).

    Raises:
        ValueError: If format is not supported.
    """
    format_lower = format_name.lower()
    format_lower = _VENDOR_FORMAT_MAP.get(format_lower, format_lower)

    if format_lower not in _PARSERS:
        supported = ", ".join(sorted(_PARSERS.keys()))
        raise ValueError(f"Unknown format '{format_name}'. Supported: {supported}")

    return _PARSERS[format_lower]
