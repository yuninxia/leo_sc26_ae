"""Abstract base class for GPU disassemblers.

This module defines the interface for GPU binary disassemblers,
supporting both NVIDIA (nvdisasm) and AMD (llvm-objdump).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import subprocess
from typing import List, Optional

from leo.binary.instruction import InstructionStat


class DisassemblerError(Exception):
    """Error invoking or parsing disassembler output."""

    pass


@dataclass
class ParsedInstruction:
    """Raw parsed instruction before full analysis.

    This is a vendor-neutral representation of a parsed instruction
    that can be converted to InstructionStat.
    """

    offset: int  # PC/offset within function
    opcode: str  # Full opcode with modifiers (e.g., "LDG.E.64", "global_load_dword")
    operands: List[str]  # Raw operand strings
    raw_line: str  # Original line from disassembler
    predicate: Optional[str] = None  # Predicate guard (NVIDIA: "@P0", AMD: None typically)


@dataclass
class ParsedFunction:
    """Parsed GPU function/kernel from disassembler output.

    This is a flat list of instructions. Use build_cfg_from_instructions()
    to convert to a CFG with basic blocks.
    """

    name: str
    instructions: List[InstructionStat] = field(default_factory=list)
    labels: dict = field(default_factory=dict)  # label_name -> offset


class Disassembler(ABC):
    """Abstract base class for GPU disassemblers."""

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Return vendor name ('nvidia' or 'amd')."""
        pass

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Return the name of the disassembler tool."""
        pass

    @abstractmethod
    def check_available(self) -> bool:
        """Check if the disassembler tool is available.

        Returns:
            True if the tool can be invoked.
        """
        pass

    @abstractmethod
    def get_version(self) -> Optional[str]:
        """Get the disassembler version string.

        Returns:
            Version string or None if unavailable.
        """
        pass

    @abstractmethod
    def disassemble(
        self,
        binary_path: str,
        function_index: Optional[int] = None,
        timeout: int = 60,
    ) -> str:
        """Invoke the disassembler on a GPU binary file.

        Args:
            binary_path: Path to GPU binary file (cubin, gpubin, .co, .hsaco).
            function_index: Optional function/kernel index to disassemble.
            timeout: Timeout in seconds.

        Returns:
            Raw disassembler output as string.

        Raises:
            DisassemblerError: If disassembly fails or times out.
        """
        pass

    @abstractmethod
    def parse_instruction_line(self, line: str) -> Optional[ParsedInstruction]:
        """Parse a single instruction line from disassembler output.

        Args:
            line: Single line from disassembler output.

        Returns:
            ParsedInstruction or None if not an instruction line.
        """
        pass

    @abstractmethod
    def parsed_to_instruction_stat(self, parsed: ParsedInstruction) -> InstructionStat:
        """Convert ParsedInstruction to InstructionStat with register analysis.

        Args:
            parsed: Raw parsed instruction.

        Returns:
            InstructionStat with register operands extracted.
        """
        pass

    @abstractmethod
    def parse_function(
        self, output: str, function_name: Optional[str] = None
    ) -> Optional[ParsedFunction]:
        """Parse disassembler output to extract a function's instructions.

        Args:
            output: Raw disassembler output.
            function_name: Optional function name to extract. If None, extracts first.

        Returns:
            ParsedFunction with instructions, or None if not found.
        """
        pass

    @abstractmethod
    def parse_all_functions(self, output: str) -> List[ParsedFunction]:
        """Parse disassembler output to extract all functions.

        Args:
            output: Raw disassembler output.

        Returns:
            List of ParsedFunctions with instructions.
        """
        pass

    def disassemble_and_parse(
        self,
        binary_path: str,
        function_name: Optional[str] = None,
    ) -> Optional[ParsedFunction]:
        """Convenience: disassemble and parse in one step.

        Args:
            binary_path: Path to GPU binary file.
            function_name: Optional function name to extract.

        Returns:
            ParsedFunction or None.
        """
        output = self.disassemble(binary_path)
        return self.parse_function(output, function_name)

    def disassemble_and_parse_all(self, binary_path: str) -> List[ParsedFunction]:
        """Disassemble and parse all functions from a GPU binary.

        Args:
            binary_path: Path to GPU binary file.

        Returns:
            List of ParsedFunction, one per function in the binary.
        """
        output = self.disassemble(binary_path)
        return self.parse_all_functions(output)

    def _split_operands(self, operand_str: str) -> List[str]:
        """Split operand string by commas, handling brackets.

        Handles cases like:
        - NVIDIA: "R1, [R2+0x10], c[0x0][0x28]"
        - AMD: "v0, v[1:2], s[0:1] offset:0"

        Args:
            operand_str: Comma-separated operand string.

        Returns:
            List of individual operand strings.
        """
        operands = []
        current = []
        bracket_depth = 0

        for char in operand_str:
            if char in "([{":
                bracket_depth += 1
                current.append(char)
            elif char in ")]}":
                bracket_depth -= 1
                current.append(char)
            elif char == "," and bracket_depth == 0:
                op = "".join(current).strip()
                if op:
                    operands.append(op)
                current = []
            else:
                current.append(char)

        # Don't forget last operand
        op = "".join(current).strip()
        if op:
            operands.append(op)

        return operands

    def _run_command(
        self,
        cmd: List[str],
        timeout: int,
        tool_label: str,
        tool_path: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a disassembler command with consistent error handling."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise DisassemblerError(f"{tool_label} timed out after {timeout}s")
        except FileNotFoundError:
            if tool_path:
                raise DisassemblerError(f"{tool_label} not found at {tool_path}")
            raise DisassemblerError(f"{tool_label} not found in PATH")


# Disassembler registry
_DISASSEMBLERS = {}


def register_disassembler(vendor: str):
    """Decorator to register a disassembler class."""

    def decorator(cls):
        _DISASSEMBLERS[vendor.lower()] = cls
        return cls

    return decorator


def get_disassembler(vendor: str) -> Disassembler:
    """Get disassembler instance by vendor name.

    Args:
        vendor: Vendor name ('nvidia' or 'amd').

    Returns:
        Disassembler instance.

    Raises:
        ValueError: If vendor is not supported.
    """
    vendor_lower = vendor.lower()
    if vendor_lower not in _DISASSEMBLERS:
        supported = ", ".join(sorted(_DISASSEMBLERS.keys()))
        raise ValueError(f"Unknown vendor '{vendor}'. Supported: {supported}")
    return _DISASSEMBLERS[vendor_lower]()
