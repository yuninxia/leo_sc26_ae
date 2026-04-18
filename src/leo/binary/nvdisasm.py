"""nvdisasm invocation and parsing for CUDA binaries.

DEPRECATED: This module is maintained for backward compatibility.
New code should use leo.binary.disasm instead.

Example migration:
    # Old:
    from leo.binary.nvdisasm import disassemble, parse_function

    # New:
    from leo.binary.disasm import get_disassembler
    disasm = get_disassembler("nvidia")
    output = disasm.disassemble(path)
    func = disasm.parse_function(output)
"""

from typing import List, Optional

# Re-export from new location for backward compatibility
from leo.binary.disasm.base import (
    DisassemblerError,
    ParsedFunction,
    ParsedInstruction,
)
from leo.binary.disasm.nvidia import (
    NVIDIADisassembler,
    extract_branch_target_label,
    extract_call_target,
    get_branch_target_from_instruction,
)

# Backward-compatible alias
NvdisasmError = DisassemblerError

# Module-level disassembler instance for backward compatibility
_disasm = NVIDIADisassembler()


def check_nvdisasm_available() -> bool:
    """Check if nvdisasm is available in PATH.

    DEPRECATED: Use NVIDIADisassembler().check_available() instead.
    """
    return _disasm.check_available()


def get_nvdisasm_version() -> Optional[str]:
    """Get nvdisasm version string.

    DEPRECATED: Use NVIDIADisassembler().get_version() instead.
    """
    return _disasm.get_version()


def disassemble(
    cubin_path: str,
    function_index: Optional[int] = None,
    timeout: int = 60,
) -> str:
    """Invoke nvdisasm on a CUDA binary file.

    DEPRECATED: Use NVIDIADisassembler().disassemble() instead.

    Args:
        cubin_path: Path to cubin/gpubin file.
        function_index: Optional function symbol index to disassemble.
        timeout: Timeout in seconds.

    Returns:
        Raw nvdisasm output as string.

    Raises:
        NvdisasmError: If nvdisasm fails or times out.
    """
    return _disasm.disassemble(cubin_path, function_index, timeout)


def parse_instruction_line(line: str) -> Optional[ParsedInstruction]:
    """Parse a single instruction line from nvdisasm output.

    DEPRECATED: Use NVIDIADisassembler().parse_instruction_line() instead.
    """
    return _disasm.parse_instruction_line(line)


def parsed_to_instruction_stat(parsed: ParsedInstruction):
    """Convert ParsedInstruction to InstructionStat with register analysis.

    DEPRECATED: Use NVIDIADisassembler().parsed_to_instruction_stat() instead.
    """
    return _disasm.parsed_to_instruction_stat(parsed)


def parse_function(
    output: str, function_name: Optional[str] = None
) -> Optional[ParsedFunction]:
    """Parse nvdisasm output to extract a function's instructions.

    DEPRECATED: Use NVIDIADisassembler().parse_function() instead.

    Args:
        output: Raw nvdisasm output.
        function_name: Optional function name to extract. If None, extracts first.

    Returns:
        ParsedFunction with instructions, or None if not found.
    """
    return _disasm.parse_function(output, function_name)


def parse_all_functions(output: str) -> List[ParsedFunction]:
    """Parse nvdisasm output to extract all functions.

    DEPRECATED: Use NVIDIADisassembler().parse_all_functions() instead.

    Args:
        output: Raw nvdisasm output.

    Returns:
        List of ParsedFunctions with instructions.
    """
    return _disasm.parse_all_functions(output)


def disassemble_and_parse(
    cubin_path: str,
    function_name: Optional[str] = None,
) -> Optional[ParsedFunction]:
    """Convenience function to disassemble and parse in one step.

    DEPRECATED: Use NVIDIADisassembler().disassemble_and_parse() instead.

    Args:
        cubin_path: Path to cubin/gpubin file.
        function_name: Optional function name to extract.

    Returns:
        ParsedFunction or None.
    """
    return _disasm.disassemble_and_parse(cubin_path, function_name)


def disassemble_and_parse_all(cubin_path: str) -> List[ParsedFunction]:
    """Disassemble and parse all functions from a CUBIN binary.

    DEPRECATED: Use NVIDIADisassembler().disassemble_and_parse_all() instead.

    Args:
        cubin_path: Path to cubin/gpubin file.

    Returns:
        List of ParsedFunction, one per function in the binary.
    """
    return _disasm.disassemble_and_parse_all(cubin_path)


# Re-export everything for backward compatibility
__all__ = [
    "NvdisasmError",
    "ParsedFunction",
    "ParsedInstruction",
    "check_nvdisasm_available",
    "get_nvdisasm_version",
    "disassemble",
    "disassemble_and_parse",
    "disassemble_and_parse_all",
    "extract_branch_target_label",
    "extract_call_target",
    "get_branch_target_from_instruction",
    "parse_all_functions",
    "parse_function",
    "parse_instruction_line",
    "parsed_to_instruction_stat",
]
