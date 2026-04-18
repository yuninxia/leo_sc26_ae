"""Binary analysis module for GPU binaries (NVIDIA and AMD)."""

from leo.binary.cfg import (
    CFG,
    Block,
    EdgeType,
    Function,
    Target,
    build_cfg_from_instructions,
)

# DWARF line information parser
from leo.binary.dwarf_line import (
    DWARFLineParser,
    SourceLocation,
    get_dwarf_line_parser,
)

# New parser module (preferred for new code)
from leo.binary.parser import (
    # Abstract base classes
    BinaryParser,
    BinarySection,
    BinaryFunction,
    KernelInfo,
    ParserError,
    # Factory functions
    get_parser,
    register_parser,
    # NVIDIA CUBIN parser
    CubinParser,
    CubinSection,
    CubinFunction,
    extract_control_from_bytes,
    parse_cubin_control_fields,
    # AMD Code Object parser
    CodeObjectParser,
    CodeObjectSection,
    CodeObjectFunction,
    AMDKernelInfo,
)
from leo.binary.dependency import (
    backward_slice,
    backward_slice_for_register,
    build_assign_pcs,
    get_all_dependencies,
    get_dependency_stats,
    print_dependencies,
    validate_assign_pcs,
)
from leo.binary.instruction import (
    BARRIER_NONE,
    Control,
    InstructionStat,
    NO_BARRIER_VALUES,
    PredicateFlag,
    is_no_barrier,
)

# New disassembler module (preferred for new code)
from leo.binary.disasm import (
    AMDDisassembler,
    Disassembler,
    DisassemblerError,
    NVIDIADisassembler,
    get_disassembler,
)

# Backward compatibility imports from nvdisasm (deprecated)
from leo.binary.nvdisasm import (
    NvdisasmError,
    ParsedFunction,
    check_nvdisasm_available,
    disassemble,
    disassemble_and_parse,
    disassemble_and_parse_all,
    extract_branch_target_label,
    extract_call_target,
    get_branch_target_from_instruction,
    parse_all_functions,
    parse_function,
)

__all__ = [
    # Core types
    "BARRIER_NONE",
    "Block",
    "CFG",
    "Control",
    "EdgeType",
    "Function",
    "InstructionStat",
    "NO_BARRIER_VALUES",
    "ParsedFunction",
    "PredicateFlag",
    "Target",
    # DWARF line info
    "DWARFLineParser",
    "SourceLocation",
    "get_dwarf_line_parser",
    # New parser API (preferred)
    "BinaryParser",
    "BinarySection",
    "BinaryFunction",
    "KernelInfo",
    "ParserError",
    "get_parser",
    "register_parser",
    # NVIDIA CUBIN parser
    "CubinFunction",
    "CubinParser",
    "CubinSection",
    "extract_control_from_bytes",
    "parse_cubin_control_fields",
    # AMD Code Object parser
    "CodeObjectParser",
    "CodeObjectSection",
    "CodeObjectFunction",
    "AMDKernelInfo",
    # New disassembler API (preferred)
    "AMDDisassembler",
    "Disassembler",
    "DisassemblerError",
    "NVIDIADisassembler",
    "get_disassembler",
    # Legacy nvdisasm API (deprecated but maintained)
    "NvdisasmError",
    "check_nvdisasm_available",
    "disassemble",
    "disassemble_and_parse",
    "disassemble_and_parse_all",
    "extract_branch_target_label",
    "extract_call_target",
    "get_branch_target_from_instruction",
    "parse_all_functions",
    "parse_function",
    # Dependency analysis
    "backward_slice",
    "backward_slice_for_register",
    "build_assign_pcs",
    "build_cfg_from_instructions",
    "get_all_dependencies",
    "get_dependency_stats",
    "is_no_barrier",
    "print_dependencies",
    "validate_assign_pcs",
]
