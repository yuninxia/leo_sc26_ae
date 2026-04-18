"""Intel GPU zebin (ELF) binary parser.

This module parses Intel GPU zebin format binaries (ELF-based), extracting
code sections and function symbols.

Intel Zebin Format:
    - ELF binary with e_machine = EM_INTELGT (205)
    - Per-kernel .text.<mangled_name> sections (similar to NVIDIA CUBIN)
    - Function symbols in .symtab with STT_FUNC type
    - Virtual addresses in high address space (0x800000000000+)
    - Instruction size: 8 bytes (compacted) or 16 bytes (native)

Based on HPCToolkit's zebinSymbols.c and Level Zero zebin format documentation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from elftools.elf.elffile import ELFFile

from leo.binary.parser.base import (
    BinaryParser,
    BinarySection,
    BinaryFunction,
    KernelInfo,
    ParserError,
    register_parser,
)
from leo.utils.validation import require_file_exists


# Intel Graphics Technology ELF machine type
EM_INTELGT = 205


@dataclass
class ZebinSection(BinarySection):
    """Represents a code section within an Intel zebin file.

    Inherits all fields from BinarySection:
        - name: Section name (e.g., ".text._Z10kernelPfi")
        - offset: Byte offset in the file
        - size: Section size in bytes
        - vaddr: Virtual address (load memory address)
        - data: Raw binary data
    """

    pass


@dataclass
class ZebinFunction(BinaryFunction):
    """Represents a function within an Intel zebin file.

    Inherits all fields from BinaryFunction:
        - name: Function symbol name
        - offset: Virtual address (st_value from symbol)
        - size: Function size in bytes
        - section: Section name containing this function
    """

    pass


@dataclass
class IntelKernelInfo(KernelInfo):
    """Intel-specific kernel metadata.

    Extends KernelInfo with Intel-specific fields.
    Note: Most metadata comes from .ze_info section (YAML format)
    which is not fully parsed in this implementation.
    """

    # Standard fields inherited from KernelInfo:
    # name, vgpr_count, sgpr_count, shared_mem_size, local_mem_size, extra

    # Intel-specific fields
    simd_size: int = 16  # SIMD width (typically 8, 16, or 32)
    grf_count: int = 128  # General Register File count
    barrier_count: int = 0  # Number of barriers used
    slm_size: int = 0  # Shared Local Memory size

    def __post_init__(self):
        # Ensure extra dict exists
        if self.extra is None:
            self.extra = {}


@register_parser("zebin")
class ZebinParser(BinaryParser):
    """Parser for Intel GPU zebin format (ELF-based).

    Extracts code sections and function symbols from Intel GPU binaries.
    The zebin format uses per-kernel .text.<name> sections similar to
    NVIDIA CUBIN format.

    Usage:
        parser = ZebinParser("/path/to/kernel.gpubin")
        for name, section in parser.sections.items():
            print(f"{name}: {section.size} bytes at vaddr 0x{section.vaddr:x}")
        for name, func in parser.functions.items():
            print(f"{name}: size={func.size}")
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize parser with path to zebin file.

        Args:
            path: Path to the Intel zebin file (.gpubin, .bin).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ParserError: If file is not a valid Intel zebin.
        """
        self._path = require_file_exists(Path(path), "Intel zebin file")

        self._sections: Dict[str, ZebinSection] = {}
        self._functions: Dict[str, ZebinFunction] = {}
        self._kernels: Dict[str, IntelKernelInfo] = {}
        self._raw_data: bytes = b""
        self._target: str = ""  # e.g., "xe_hpc" or GPU device name

        self._parse()

    @property
    def format(self) -> str:
        """Return the binary format name."""
        return "zebin"

    @property
    def vendor(self) -> str:
        """Return vendor name."""
        return "intel"

    @property
    def path(self) -> Path:
        """Return path to the binary file."""
        return self._path

    @property
    def target(self) -> str:
        """Get the Intel GPU target (e.g., 'xe_hpc')."""
        return self._target

    @property
    def sections(self) -> Dict[str, ZebinSection]:
        """Get all code sections, keyed by section name."""
        return self._sections

    @property
    def functions(self) -> Dict[str, ZebinFunction]:
        """Get all function symbols, keyed by function name."""
        return self._functions

    @property
    def kernels(self) -> Dict[str, IntelKernelInfo]:
        """Get all kernel metadata."""
        return self._kernels

    def _parse(self) -> None:
        """Parse ELF structure, sections, and symbols."""
        with open(self._path, "rb") as f:
            self._raw_data = f.read()
            f.seek(0)

            try:
                elf = ELFFile(f)
            except Exception as e:
                raise ParserError(f"Invalid ELF file: {e}")

            # Verify this is an Intel GPU binary (optional check)
            machine = elf.header.e_machine
            if machine != "EM_INTELGT" and machine != EM_INTELGT:
                # Log warning but don't fail - might be a valid binary
                # with different machine type
                pass

            # Extract code sections
            self._parse_sections(elf)

            # Extract function symbols
            self._parse_symbols(elf)

            # Build kernel info from sections
            self._build_kernel_info()

    def _parse_sections(self, elf: ELFFile) -> None:
        """Extract code sections from zebin.

        Intel zebin uses per-kernel .text.<kernel_name> sections,
        similar to NVIDIA CUBIN format.
        """
        for section in elf.iter_sections():
            name = section.name
            # Intel kernels are in .text.* sections
            if name.startswith(".text"):
                self._sections[name] = ZebinSection(
                    name=name,
                    offset=section["sh_offset"],
                    size=section["sh_size"],
                    vaddr=section["sh_addr"],
                    data=section.data(),
                )

    def _parse_symbols(self, elf: ELFFile) -> None:
        """Extract function symbols from zebin symbol table.

        Based on HPCToolkit's zebinSymbols.c - filters for STT_FUNC
        symbols and skips _entry points.
        """
        symtab = elf.get_section_by_name(".symtab")
        if not symtab:
            return

        for sym in symtab.iter_symbols():
            if sym["st_info"]["type"] == "STT_FUNC":
                sym_name = sym.name
                # Skip entry point symbols (per HPCToolkit)
                if sym_name == "_entry" or not sym_name:
                    continue

                section_idx = sym["st_shndx"]
                if isinstance(section_idx, int) and section_idx > 0:
                    try:
                        section = elf.get_section(section_idx)
                        if section and section.name.startswith(".text"):
                            self._functions[sym_name] = ZebinFunction(
                                name=sym_name,
                                offset=sym["st_value"],  # Virtual address
                                size=sym["st_size"],
                                section=section.name,
                            )
                    except Exception:
                        # Invalid section index, skip
                        pass

    def _build_kernel_info(self) -> None:
        """Build kernel info from parsed sections and functions.

        Note: Full kernel metadata would require parsing the .ze_info
        section (YAML format), which contains detailed information
        about register usage, SLM size, etc.
        """
        # Create basic kernel info for each function
        for func_name, func in self._functions.items():
            # Extract kernel name from section name if available
            section = self._sections.get(func.section)
            if section:
                self._kernels[func_name] = IntelKernelInfo(
                    name=func_name,
                    # Default values - would need .ze_info parsing for real values
                    vgpr_count=0,
                    sgpr_count=0,
                    shared_mem_size=0,
                    local_mem_size=0,
                    simd_size=16,
                    grf_count=128,
                    barrier_count=0,
                    slm_size=0,
                    extra={
                        "section": func.section,
                        "vaddr": func.offset,
                        "size": func.size,
                    },
                )

    def get_section_for_function(self, func_name: str) -> Optional[ZebinSection]:
        """Get the code section containing a function.

        Args:
            func_name: Function name (mangled or partial).

        Returns:
            ZebinSection or None if not found.
        """
        # Try exact match first
        if func_name in self._functions:
            section_name = self._functions[func_name].section
            return self._sections.get(section_name)

        # Try partial match (function name might be part of section name)
        for section_name, section in self._sections.items():
            if func_name in section_name:
                return section

        return None

    def get_kernel_info(self, kernel_name: str) -> Optional[IntelKernelInfo]:
        """Get metadata for a specific kernel.

        Args:
            kernel_name: Kernel function name.

        Returns:
            IntelKernelInfo or None if not found.
        """
        return self._kernels.get(kernel_name)

    def get_all_kernel_info(self) -> Dict[str, IntelKernelInfo]:
        """Get metadata for all kernels.

        Returns:
            Dict mapping kernel name to IntelKernelInfo.
        """
        return self._kernels

    def get_gpu_arch(self) -> str:
        """Extract GPU architecture from target string.

        Returns:
            GPU architecture (e.g., 'xe_hpc') or empty string.
        """
        return self._target
