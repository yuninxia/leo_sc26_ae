"""CUBIN binary parsing for control field extraction.

This module extracts scheduling control fields directly from CUBIN binaries,
providing accurate stall counts, barrier information, and other scheduling hints
that cannot be obtained from nvdisasm output alone.

Based on GPA-artifact's AnalyzeInstruction.cpp (lines 640-660).

Control Field Bit Layout (bits 41-63 of instruction word):
    bits [44:41] - stall: Pipeline stall count (0-15 cycles)
    bit  [45]    - yield: Warp scheduler yield hint
    bits [48:46] - write: Write barrier ID (0-6, 7 = none)
    bits [51:49] - read: Read barrier ID (0-6, 7 = none)
    bits [57:52] - wait: Barrier wait mask (bits for B1-B6)
    bits [61:58] - reuse: Register reuse cache flags

CUBIN Format:
    CUBIN files are ELF binaries with .text sections containing GPU code.
    For Volta+ architectures, instructions are 16 bytes (128 bits).
    Control fields are in the first 8 bytes (lower 64 bits) of each instruction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from elftools.elf.elffile import ELFFile

from leo.binary.instruction import Control, InstructionStat
from leo.utils.validation import require_file_exists
from leo.binary.parser.base import (
    BinaryParser,
    BinarySection,
    BinaryFunction,
    KernelInfo,
    register_parser,
)


# Control field bit masks for NVIDIA SASS instructions
# From GPA-artifact AnalyzeInstruction.cpp:646-651
MASK_STALL = 0x00001E0000000000  # bits [44:41]
MASK_YIELD = 0x0000200000000000  # bit [45]
MASK_WRITE = 0x0001C00000000000  # bits [48:46]
MASK_READ = 0x000E000000000000  # bits [51:49]
MASK_WAIT = 0x03F0000000000000  # bits [57:52]
MASK_REUSE = 0x3C00000000000000  # bits [61:58]

SHIFT_STALL = 41
SHIFT_YIELD = 45
SHIFT_WRITE = 46
SHIFT_READ = 49
SHIFT_WAIT = 52
SHIFT_REUSE = 58


@dataclass
class CubinSection:
    """Represents a code section within a CUBIN file."""

    name: str
    offset: int  # Offset in file
    size: int  # Section size in bytes
    vaddr: int  # Virtual address (load memory address)
    data: bytes  # Raw binary data


@dataclass
class CubinFunction:
    """Represents a function within a CUBIN file."""

    name: str
    offset: int  # Offset within section
    size: int  # Function size in bytes
    section: str  # Section name containing this function


@register_parser("cubin")
class CubinParser(BinaryParser):
    """Parser for NVIDIA CUBIN binary files.

    Extracts control fields from instruction encodings using ELF parsing
    and bit manipulation based on GPA-artifact's extraction logic.

    Usage:
        parser = CubinParser("/path/to/kernel.cubin")
        control = parser.extract_control_fields(pc=0x100, section=".text._Z...")

        # Or populate control fields for existing instructions:
        parser.populate_instruction_controls(instructions, section=".text._Z...")
    """

    def __init__(self, cubin_path: Union[str, Path]):
        """Initialize parser with path to CUBIN file.

        Args:
            cubin_path: Path to the CUBIN binary file.

        Raises:
            FileNotFoundError: If CUBIN file doesn't exist.
            ValueError: If file is not a valid ELF/CUBIN.
        """
        # Note: We don't call super().__init__() to maintain backward compatibility
        # with existing code that expects the specific error messages
        self.cubin_path = require_file_exists(Path(cubin_path), "CUBIN file")

        self._sections: Dict[str, CubinSection] = {}
        self._functions: Dict[str, CubinFunction] = {}
        self._raw_data: bytes = b""

        self._parse_elf()

    # BinaryParser abstract property implementations
    @property
    def format(self) -> str:
        return "cubin"

    @property
    def vendor(self) -> str:
        return "nvidia"

    @property
    def path(self) -> Path:
        return self.cubin_path

    def _parse_elf(self) -> None:
        """Parse ELF structure and extract code sections."""
        with open(self.cubin_path, "rb") as f:
            self._raw_data = f.read()
            f.seek(0)

            try:
                elf = ELFFile(f)
            except Exception as e:
                raise ValueError(f"Invalid ELF/CUBIN file: {e}")

            # Extract code sections (.text.*)
            for section in elf.iter_sections():
                name = section.name
                if name.startswith(".text"):
                    self._sections[name] = CubinSection(
                        name=name,
                        offset=section["sh_offset"],
                        size=section["sh_size"],
                        vaddr=section["sh_addr"],
                        data=section.data(),
                    )

            # Extract function symbols
            symtab = elf.get_section_by_name(".symtab")
            if symtab:
                for sym in symtab.iter_symbols():
                    if sym["st_info"]["type"] == "STT_FUNC":
                        # Find which section this function belongs to
                        section_idx = sym["st_shndx"]
                        if isinstance(section_idx, int) and section_idx > 0:
                            section = elf.get_section(section_idx)
                            if section and section.name.startswith(".text"):
                                self._functions[sym.name] = CubinFunction(
                                    name=sym.name,
                                    offset=sym["st_value"],
                                    size=sym["st_size"],
                                    section=section.name,
                                )

    @property
    def sections(self) -> Dict[str, CubinSection]:
        """Get all code sections."""
        return self._sections

    @property
    def functions(self) -> Dict[str, CubinFunction]:
        """Get all function symbols."""
        return self._functions

    def get_section_for_function(self, func_name: str) -> Optional[CubinSection]:
        """Get the code section containing a function.

        Args:
            func_name: Function name (mangled or demangled).

        Returns:
            CubinSection or None if not found.
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

    def extract_control_at_offset(
        self, section_offset: int, section_data: bytes
    ) -> Control:
        """Extract control fields from instruction at given section offset.

        Based on GPA-artifact AnalyzeInstruction.cpp:640-660.

        Args:
            section_offset: Byte offset within section (the PC).
            section_data: Raw section binary data.

        Returns:
            Control object with extracted fields.
        """
        if section_offset + 8 > len(section_data):
            return Control()  # Default if out of bounds

        # Read 8 bytes (64 bits) at the instruction offset
        # Control fields are in bits 41-63 of this 64-bit word
        bits = int.from_bytes(
            section_data[section_offset : section_offset + 8], byteorder="little"
        )

        # Extract each field using masks and shifts
        stall = (bits & MASK_STALL) >> SHIFT_STALL
        yield_flag = (bits & MASK_YIELD) >> SHIFT_YIELD
        write = (bits & MASK_WRITE) >> SHIFT_WRITE
        read = (bits & MASK_READ) >> SHIFT_READ
        wait = (bits & MASK_WAIT) >> SHIFT_WAIT
        reuse = (bits & MASK_REUSE) >> SHIFT_REUSE

        return Control(
            reuse=reuse,
            wait=wait,
            read=read,
            write=write,
            yield_flag=yield_flag,
            stall=stall,
        )

    def extract_control_fields(
        self,
        pc: int,
        section_name: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> Optional[Control]:
        """Extract control fields for instruction at given PC.

        Args:
            pc: Program counter (offset within function/section).
            section_name: Explicit section name (e.g., ".text._Z10kernelPfi").
            function_name: Function name to find section automatically.

        Returns:
            Control object or None if section not found.
        """
        section = None

        if section_name:
            section = self._sections.get(section_name)
        elif function_name:
            section = self.get_section_for_function(function_name)
        else:
            # Try first .text section
            for name, sec in self._sections.items():
                if name.startswith(".text"):
                    section = sec
                    break

        if not section:
            return None

        return self.extract_control_at_offset(pc, section.data)

    def populate_instruction_controls(
        self,
        instructions: List[InstructionStat],
        section_name: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> int:
        """Populate control fields for a list of instructions.

        Modifies instructions in place.

        Args:
            instructions: List of InstructionStat objects to update.
            section_name: Section containing the instructions.
            function_name: Function name to find section.

        Returns:
            Number of instructions successfully updated.
        """
        section = None

        if section_name:
            section = self._sections.get(section_name)
        elif function_name:
            section = self.get_section_for_function(function_name)
        else:
            # Try first .text section
            for name, sec in self._sections.items():
                if name.startswith(".text"):
                    section = sec
                    break

        if not section:
            return 0

        count = 0
        for inst in instructions:
            if inst.pc + 8 <= section.size:
                inst.control = self.extract_control_at_offset(inst.pc, section.data)
                count += 1

        return count

    def extract_all_controls(
        self,
        section_name: str,
        inst_size: int = 16,
    ) -> Dict[int, Control]:
        """Extract control fields for all instructions in a section.

        Args:
            section_name: Section to extract from.
            inst_size: Instruction size in bytes (16 for Volta+).

        Returns:
            Dict mapping PC -> Control.
        """
        section = self._sections.get(section_name)
        if not section:
            return {}

        controls = {}
        for pc in range(0, section.size, inst_size):
            controls[pc] = self.extract_control_at_offset(pc, section.data)

        return controls


def extract_control_from_bytes(instruction_bytes: bytes) -> Control:
    """Extract control fields from raw instruction bytes.

    Utility function for when you have instruction bytes directly.

    Args:
        instruction_bytes: At least 8 bytes of instruction data.

    Returns:
        Control object with extracted fields.
    """
    if len(instruction_bytes) < 8:
        return Control()

    bits = int.from_bytes(instruction_bytes[:8], byteorder="little")

    return Control(
        reuse=(bits & MASK_REUSE) >> SHIFT_REUSE,
        wait=(bits & MASK_WAIT) >> SHIFT_WAIT,
        read=(bits & MASK_READ) >> SHIFT_READ,
        write=(bits & MASK_WRITE) >> SHIFT_WRITE,
        yield_flag=(bits & MASK_YIELD) >> SHIFT_YIELD,
        stall=(bits & MASK_STALL) >> SHIFT_STALL,
    )


def parse_cubin_control_fields(
    cubin_path: Union[str, Path],
    instructions: List[InstructionStat],
    function_name: Optional[str] = None,
    section_name: Optional[str] = None,
) -> int:
    """Convenience function to parse CUBIN and populate instruction controls.

    Args:
        cubin_path: Path to CUBIN file.
        instructions: Instructions to update.
        function_name: Function name for section lookup.
        section_name: Explicit section name.

    Returns:
        Number of instructions updated.
    """
    parser = CubinParser(cubin_path)
    return parser.populate_instruction_controls(
        instructions,
        section_name=section_name,
        function_name=function_name,
    )
