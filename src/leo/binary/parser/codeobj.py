"""AMD Code Object parsing for kernel metadata extraction.

This module parses AMD GPU Code Object ELF files, extracting kernel metadata
from the MessagePack-encoded .note.AMDGPU.metadata section.

AMD Code Object Format:
    - ELF binary with e_machine = EM_AMDGPU (0xE0)
    - Kernel code in .text sections
    - MessagePack metadata in .note.AMDGPU.metadata (NT_AMDGPU_METADATA = 32)
    - Kernel descriptors (64 bytes) for each kernel

Key Metadata Fields:
    - .vgpr_count: Vector GPR count per work-item
    - .sgpr_count: Scalar GPR count per wavefront
    - .group_segment_fixed_size: LDS (shared memory) size
    - .private_segment_fixed_size: Scratch memory per work-item
    - .wavefront_size: Threads per wave (32 or 64)

Based on LLVM AMDGPU Backend documentation and HPCToolkit's rocm-binaries.c.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import msgpack

    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

from elftools.elf.elffile import ELFFile
from elftools.elf.notes import iter_notes

from leo.utils.validation import require_file_exists
from leo.binary.parser.base import (
    BinaryParser,
    BinarySection,
    BinaryFunction,
    KernelInfo,
    ParserError,
    register_parser,
)


# AMD Note type for MessagePack metadata
NT_AMDGPU_METADATA = 32

# AMD ELF machine type
EM_AMDGPU = 0xE0


@dataclass
class CodeObjectSection:
    """Represents a code section within an AMD Code Object file."""

    name: str
    offset: int  # Offset in file
    size: int  # Section size in bytes
    vaddr: int  # Virtual address (load memory address)
    data: bytes  # Raw binary data


@dataclass
class CodeObjectFunction:
    """Represents a function within an AMD Code Object file."""

    name: str
    offset: int  # Offset within section or file
    size: int  # Function size in bytes
    section: str  # Section name containing this function


@dataclass
class AMDKernelInfo(KernelInfo):
    """AMD-specific kernel metadata.

    Extends KernelInfo with AMD-specific fields from MessagePack metadata.
    """

    # Standard fields inherited from KernelInfo:
    # name, vgpr_count, sgpr_count, shared_mem_size, local_mem_size, extra

    # AMD-specific fields
    symbol: str = ""  # Kernel descriptor symbol (e.g., "kernel.kd")
    agpr_count: int = 0  # Accumulator VGPRs (CDNA only)
    vgpr_spill_count: int = 0  # Spilled VGPRs
    sgpr_spill_count: int = 0  # Spilled SGPRs
    wavefront_size: int = 64  # Wave size (32 or 64)
    max_flat_workgroup_size: int = 256  # Max work-items per workgroup
    kernarg_segment_size: int = 0  # Kernel argument buffer size
    kernarg_segment_align: int = 0  # Argument alignment
    uses_dynamic_stack: bool = False  # Dynamic call stack usage

    def __post_init__(self):
        # Ensure extra dict exists
        if self.extra is None:
            self.extra = {}


@register_parser("codeobj")
class CodeObjectParser(BinaryParser):
    """Parser for AMD GPU Code Object ELF files.

    Extracts kernel metadata from MessagePack-encoded .note.AMDGPU.metadata
    section, along with code sections and function symbols.

    Usage:
        parser = CodeObjectParser("/path/to/kernel.co")
        kernels = parser.get_all_kernel_info()
        for name, info in kernels.items():
            print(f"{name}: VGPRs={info.vgpr_count}, LDS={info.shared_mem_size}")
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize parser with path to Code Object file.

        Args:
            path: Path to the AMD Code Object file (.co, .hsaco, .gpubin).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ParserError: If file is not a valid AMD Code Object.
            ImportError: If msgpack is not installed.
        """
        if not HAS_MSGPACK:
            raise ImportError(
                "msgpack is required for AMD Code Object parsing. "
                "Install with: pip install msgpack"
            )

        self._path = require_file_exists(Path(path), "Code Object file")

        self._sections: Dict[str, CodeObjectSection] = {}
        self._functions: Dict[str, CodeObjectFunction] = {}
        self._kernels: Dict[str, AMDKernelInfo] = {}
        self._metadata: Dict[str, Any] = {}
        self._raw_data: bytes = b""
        self._target: str = ""  # e.g., "amdgcn-amd-amdhsa--gfx90a"

        self._parse()

    @property
    def format(self) -> str:
        return "codeobj"

    @property
    def vendor(self) -> str:
        return "amd"

    @property
    def path(self) -> Path:
        return self._path

    @property
    def target(self) -> str:
        """Get the AMD GPU target (e.g., 'gfx90a', 'gfx942')."""
        return self._target

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the raw MessagePack metadata dict."""
        return self._metadata

    def _parse(self) -> None:
        """Parse ELF structure, metadata, and symbols."""
        with open(self._path, "rb") as f:
            self._raw_data = f.read()
            f.seek(0)

            try:
                elf = ELFFile(f)
            except Exception as e:
                raise ParserError(f"Invalid ELF file: {e}")

            # Verify this is an AMD GPU binary
            machine = elf.header.e_machine
            if machine != "EM_AMDGPU" and machine != EM_AMDGPU:
                # Allow non-AMD binaries but warn (might be a CUDA binary)
                pass  # We'll still try to parse what we can

            # Extract code sections
            self._parse_sections(elf)

            # Extract function symbols
            self._parse_symbols(elf)

            # Extract MessagePack metadata
            self._parse_metadata(elf)

            # Build kernel info from metadata
            self._build_kernel_info()

    def _parse_sections(self, elf: ELFFile) -> None:
        """Extract code sections from ELF."""
        for section in elf.iter_sections():
            name = section.name
            # AMD kernels are in .text sections
            if name.startswith(".text"):
                self._sections[name] = CodeObjectSection(
                    name=name,
                    offset=section["sh_offset"],
                    size=section["sh_size"],
                    vaddr=section["sh_addr"],
                    data=section.data(),
                )

    def _parse_symbols(self, elf: ELFFile) -> None:
        """Extract function symbols from ELF symbol table."""
        symtab = elf.get_section_by_name(".symtab")
        if not symtab:
            return

        for sym in symtab.iter_symbols():
            if sym["st_info"]["type"] == "STT_FUNC":
                section_idx = sym["st_shndx"]
                if isinstance(section_idx, int) and section_idx > 0:
                    try:
                        section = elf.get_section(section_idx)
                        if section and section.name.startswith(".text"):
                            self._functions[sym.name] = CodeObjectFunction(
                                name=sym.name,
                                offset=sym["st_value"],
                                size=sym["st_size"],
                                section=section.name,
                            )
                    except Exception:
                        # Invalid section index
                        pass

    def _parse_metadata(self, elf: ELFFile) -> None:
        """Extract and parse MessagePack metadata from .note.AMDGPU.metadata."""
        note_section = elf.get_section_by_name(".note.AMDGPU.metadata")
        if not note_section:
            # Try alternative section name
            note_section = elf.get_section_by_name(".note")
            if not note_section:
                return

        try:
            # Use pyelftools iter_notes to handle alignment correctly
            for note in iter_notes(
                elf, note_section["sh_offset"], note_section["sh_size"]
            ):
                # Check for AMD GPU metadata note
                note_type = note["n_type"]
                note_name = note["n_name"]

                # NT_AMDGPU_METADATA = 32, name = "AMDGPU"
                if (note_type == NT_AMDGPU_METADATA or note_type == 32) and (
                    note_name == "AMDGPU" or note_name.startswith("AMDGPU")
                ):
                    # Extract MessagePack data
                    msgpack_data = note["n_descdata"]
                    self._metadata = self._decode_msgpack(msgpack_data)
                    break

        except Exception as e:
            # If iter_notes fails, try manual parsing
            self._parse_metadata_manual(note_section.data())

    def _parse_metadata_manual(self, note_data: bytes) -> None:
        """Manually parse note section if iter_notes fails.

        Note header format:
            - namesz (4 bytes): Name size including null
            - descsz (4 bytes): Description size
            - type (4 bytes): Note type
            - name (namesz bytes, padded to 4-byte alignment)
            - desc (descsz bytes): MessagePack data
        """
        if len(note_data) < 12:
            return

        offset = 0
        while offset + 12 <= len(note_data):
            namesz = int.from_bytes(note_data[offset : offset + 4], "little")
            descsz = int.from_bytes(note_data[offset + 4 : offset + 8], "little")
            note_type = int.from_bytes(note_data[offset + 8 : offset + 12], "little")

            # Skip header
            offset += 12

            # Read name (with alignment padding)
            name_end = offset + namesz
            if name_end > len(note_data):
                break
            name = note_data[offset:name_end].rstrip(b"\x00").decode("utf-8", "replace")

            # Align to 4 bytes
            aligned_namesz = (namesz + 3) & ~3
            offset += aligned_namesz

            # Read description
            desc_end = offset + descsz
            if desc_end > len(note_data):
                break

            if note_type == NT_AMDGPU_METADATA and name == "AMDGPU":
                msgpack_data = note_data[offset:desc_end]
                self._metadata = self._decode_msgpack(msgpack_data)
                break

            # Align to 4 bytes and continue
            aligned_descsz = (descsz + 3) & ~3
            offset += aligned_descsz

    def _decode_msgpack(self, data: bytes) -> Dict[str, Any]:
        """Safely decode MessagePack data.

        Args:
            data: Raw MessagePack binary data.

        Returns:
            Decoded dict, or empty dict on error.
        """
        try:
            return msgpack.unpackb(
                data,
                raw=False,  # Decode strings as str
                unicode_errors="replace",
                max_str_len=10000000,
                max_array_len=100000,
                max_map_len=50000,
                strict_map_key=True,
            )
        except Exception as e:
            # Log warning but don't fail
            return {}

    def _build_kernel_info(self) -> None:
        """Build kernel info from parsed MessagePack metadata."""
        # Extract target
        self._target = self._metadata.get("amdhsa.target", "")

        # Extract kernel information
        kernels = self._metadata.get("amdhsa.kernels", [])
        for kernel in kernels:
            name = kernel.get(".name", "")
            if not name:
                continue

            info = AMDKernelInfo(
                name=name,
                symbol=kernel.get(".symbol", ""),
                vgpr_count=kernel.get(".vgpr_count", 0),
                sgpr_count=kernel.get(".sgpr_count", 0),
                agpr_count=kernel.get(".agpr_count", 0),
                vgpr_spill_count=kernel.get(".vgpr_spill_count", 0),
                sgpr_spill_count=kernel.get(".sgpr_spill_count", 0),
                shared_mem_size=kernel.get(".group_segment_fixed_size", 0),  # LDS
                local_mem_size=kernel.get(".private_segment_fixed_size", 0),  # Scratch
                wavefront_size=kernel.get(".wavefront_size", 64),
                max_flat_workgroup_size=kernel.get(".max_flat_workgroup_size", 256),
                kernarg_segment_size=kernel.get(".kernarg_segment_size", 0),
                kernarg_segment_align=kernel.get(".kernarg_segment_align", 0),
                uses_dynamic_stack=kernel.get(".uses_dynamic_stack", False),
                extra={
                    k: v for k, v in kernel.items() if not k.startswith(".")
                },  # Non-standard fields
            )
            self._kernels[name] = info

    @property
    def sections(self) -> Dict[str, CodeObjectSection]:
        """Get all code sections."""
        return self._sections

    @property
    def functions(self) -> Dict[str, CodeObjectFunction]:
        """Get all function symbols."""
        return self._functions

    @property
    def kernels(self) -> Dict[str, AMDKernelInfo]:
        """Get all kernel metadata."""
        return self._kernels

    def get_section_for_function(
        self, func_name: str
    ) -> Optional[CodeObjectSection]:
        """Get the code section containing a function.

        Args:
            func_name: Function name (mangled or demangled).

        Returns:
            CodeObjectSection or None if not found.
        """
        # Try exact match first
        if func_name in self._functions:
            section_name = self._functions[func_name].section
            return self._sections.get(section_name)

        # Try partial match
        for section_name, section in self._sections.items():
            if func_name in section_name:
                return section

        return None

    def get_kernel_info(self, kernel_name: str) -> Optional[AMDKernelInfo]:
        """Get metadata for a specific kernel.

        Args:
            kernel_name: Kernel function name.

        Returns:
            AMDKernelInfo or None if not found.
        """
        return self._kernels.get(kernel_name)

    def get_all_kernel_info(self) -> Dict[str, AMDKernelInfo]:
        """Get metadata for all kernels.

        Returns:
            Dict mapping kernel name to AMDKernelInfo.
        """
        return self._kernels

    def get_gpu_arch(self) -> str:
        """Extract GPU architecture from target string.

        Returns:
            GPU architecture (e.g., 'gfx90a', 'gfx942') or empty string.
        """
        # Target format: "amdgcn-amd-amdhsa--gfx90a"
        if self._target:
            parts = self._target.split("--")
            if len(parts) >= 2:
                return parts[-1]  # e.g., "gfx90a"
        return ""
