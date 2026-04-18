"""Intel GPU disassembler using GED (Graphics Encoder/Decoder) library.

Wraps Intel's GED library (part of GTPin) to disassemble Intel GPU binaries
(zebin format) and parses the output into structured instruction data.

GED is more reliable than standalone IGA for newer zebin format 1.50 binaries.
This is the same library HPCToolkit uses internally for Intel GPU support.

GED Library Reference:
- Location: deps/hpctoolkit/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so
- GED_MODEL_XE_HPC = 12 (Ponte Vecchio)
- Instruction sizes: 8 bytes (compact) or 16 bytes (native)
"""

import ctypes
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from leo.binary.instruction import InstructionStat, PredicateFlag
from leo.binary.parser.zebin import ZebinParser
from leo.utils.validation import require_file_exists

from .base import (
    Disassembler,
    DisassemblerError,
    ParsedFunction,
    ParsedInstruction,
    register_disassembler,
)


# =============================================================================
# GED Library Constants
# =============================================================================

# GED Model IDs
GED_MODEL_XE_HPC = 12  # Ponte Vecchio

# GED Return Values
GED_RETURN_VALUE_SUCCESS = 0

# GED Register File IDs
GED_REG_FILE_ARF = 0       # Architecture Register File
GED_REG_FILE_GRF = 1       # General Register File (r0-r127)
GED_REG_FILE_IMM = 2       # Immediate value
GED_REG_FILE_INVALID = 3   # Invalid

# GED Instruction buffer size (opaque structure)
GED_INS_SIZE = 256  # bytes

# GED Opcode names (indexed by GED_OPCODE enum value)
# From GED library header and verified against actual binaries
GED_OPCODE_NAMES = [
    "illegal", "mov", "sel", "movi", "not", "and", "or", "xor",
    "shr", "shl", "smov", "asr", "ror", "rol", "cmp", "cmpn",
    "csel", "bfrev", "bfe", "bfi1", "bfi2", "jmpi", "brd", "if",
    "brc", "else", "endif", "while", "break", "cont", "halt", "calla",
    "call", "ret", "goto", "join", "wait", "send", "sendc", "sends",
    "sendsc", "math", "add", "mul", "avg", "frc", "rndu", "rndd",
    "rnde", "rndz", "mac", "mach", "lzd", "fbh", "fbl", "cbit",
    "addc", "subb", "sad2", "sada2", "dp4", "dph", "dp3", "dp2",
    "line", "pln", "mad", "lrp", "madm", "nop", "rndr", "rndrd",
    "urb", "sync", "reserved", "dpas", "add3", "bfn", "macl", "srnd",
    "fcvt", "reserved2", "reserved3", "reserved4", "f32to16", "f16to32", "dim", "dp4a",
    "dpasw",
]


def _find_ged_library() -> Optional[str]:
    """Find libged.so in common locations.

    Returns:
        Path to libged.so or None if not found.
    """
    # Check environment variable first
    ged_path = os.environ.get("GED_LIBRARY_PATH")
    if ged_path and Path(ged_path).exists():
        return ged_path

    # Common search paths (newest/most likely first)
    search_paths = [
        # Relative to project root (HPCToolkit build directory)
        "deps/hpctoolkit/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so",
        "3rdparty/hpctoolkit/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so",
        # Common GTPin installations
        "/opt/intel/oneapi/gtpin/latest/Profilers/Lib/intel64/libged.so",
        "/opt/gtpin/Profilers/Lib/intel64/libged.so",
    ]

    for path in search_paths:
        if Path(path).exists():
            return path

    return None


@register_disassembler("intel")
class IntelDisassembler(Disassembler):
    """Intel GPU disassembler using GED library.

    Uses the Graphics Encoder/Decoder (GED) library from GTPin to
    disassemble Intel GPU binaries in zebin format.

    Example:
        disasm = IntelDisassembler()
        if disasm.check_available():
            funcs = disasm.disassemble_and_parse_all("kernel.gpubin")
            for func in funcs:
                print(f"{func.name}: {len(func.instructions)} instructions")
    """

    def __init__(
        self,
        ged_library_path: Optional[str] = None,
        gpu_model: int = GED_MODEL_XE_HPC,
    ):
        """Initialize Intel disassembler.

        Args:
            ged_library_path: Path to libged.so. If None, searches common locations.
            gpu_model: GED model ID (default: GED_MODEL_XE_HPC for Ponte Vecchio).
        """
        self._ged_path = ged_library_path or _find_ged_library()
        self._gpu_model = gpu_model
        self._ged: Optional[ctypes.CDLL] = None
        self._ged_available: Optional[bool] = None

    @property
    def vendor(self) -> str:
        return "intel"

    @property
    def tool_name(self) -> str:
        return "ged"

    def _load_ged(self) -> Optional[ctypes.CDLL]:
        """Load the GED library via ctypes.

        Returns:
            ctypes.CDLL handle or None if loading fails.
        """
        if self._ged is not None:
            return self._ged

        if self._ged_path is None:
            return None

        try:
            ged = ctypes.CDLL(self._ged_path)

            # Set up function signatures
            # GED_DecodeIns(model, raw_bytes, raw_bytes_size, ins) -> int
            ged.GED_DecodeIns.argtypes = [
                ctypes.c_int,      # GED_MODEL
                ctypes.c_void_p,   # const uint8_t* raw_bytes
                ctypes.c_uint32,   # uint32_t raw_bytes_size
                ctypes.c_void_p,   # GED_INS* ins
            ]
            ged.GED_DecodeIns.restype = ctypes.c_int

            # GED_GetOpcode(ins) -> int (GED_OPCODE enum)
            ged.GED_GetOpcode.argtypes = [ctypes.c_void_p]
            ged.GED_GetOpcode.restype = ctypes.c_int

            # GED_InsSize(ins) -> uint32_t
            ged.GED_InsSize.argtypes = [ctypes.c_void_p]
            ged.GED_InsSize.restype = ctypes.c_uint32

            # GED_IsCompact(ins) -> int (0 = native, 1 = compact)
            ged.GED_IsCompact.argtypes = [ctypes.c_void_p]
            ged.GED_IsCompact.restype = ctypes.c_int

            # Destination register extraction
            ged.GED_GetDstRegFile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetDstRegFile.restype = ctypes.c_int
            ged.GED_GetDstRegNum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetDstRegNum.restype = ctypes.c_uint32

            # Source 0 register extraction
            ged.GED_GetSrc0RegFile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc0RegFile.restype = ctypes.c_int
            ged.GED_GetSrc0RegNum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc0RegNum.restype = ctypes.c_uint32
            ged.GED_GetSrc0IsImm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc0IsImm.restype = ctypes.c_uint32

            # Source 1 register extraction
            ged.GED_GetSrc1RegFile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc1RegFile.restype = ctypes.c_int
            ged.GED_GetSrc1RegNum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc1RegNum.restype = ctypes.c_uint32
            ged.GED_GetSrc1IsImm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc1IsImm.restype = ctypes.c_uint32

            # Source 2 register extraction (for 3-operand instructions)
            ged.GED_GetSrc2RegFile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc2RegFile.restype = ctypes.c_int
            ged.GED_GetSrc2RegNum.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSrc2RegNum.restype = ctypes.c_uint32

            # SWSB (Software Scoreboarding) field extraction
            ged.GED_GetSWSB.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            ged.GED_GetSWSB.restype = ctypes.c_uint32

            self._ged = ged
            return ged

        except (OSError, AttributeError) as e:
            # Library not found or missing functions
            return None

    def check_available(self) -> bool:
        """Check if the GED library is available.

        Returns:
            True if libged.so can be loaded and has required functions.
        """
        if self._ged_available is not None:
            return self._ged_available

        ged = self._load_ged()
        self._ged_available = ged is not None
        return self._ged_available

    def get_version(self) -> Optional[str]:
        """Get GED library version.

        Returns:
            Version string or None if unavailable.
            Note: GED doesn't expose a version function, so we return
            the GTPin version from the library path if available.
        """
        if not self.check_available():
            return None

        # Extract version from path if possible (e.g., gtpin-4.5.0)
        if self._ged_path:
            import re
            match = re.search(r"gtpin-(\d+\.\d+\.\d+)", self._ged_path)
            if match:
                return match.group(1)

        return "unknown"

    def _opcode_to_name(self, opcode_id: int) -> str:
        """Convert GED opcode enum value to opcode name.

        Args:
            opcode_id: GED_OPCODE enum value.

        Returns:
            Opcode name string (e.g., "mov", "add", "send").
        """
        if 0 <= opcode_id < len(GED_OPCODE_NAMES):
            return GED_OPCODE_NAMES[opcode_id]
        return f"unknown_{opcode_id}"

    def _extract_operands_from_ged(
        self,
        ins_ptr: ctypes.c_void_p,
        ged: ctypes.CDLL,
    ) -> Tuple[List[int], List[int]]:
        """Extract destination and source registers from a GED instruction.

        Args:
            ins_ptr: Pointer to decoded GED instruction.
            ged: Loaded GED library.

        Returns:
            Tuple of (dsts, srcs) lists containing GRF register numbers.
        """
        dsts: List[int] = []
        srcs: List[int] = []
        result = ctypes.c_uint32()

        # Extract destination register
        dst_reg_file = ged.GED_GetDstRegFile(ins_ptr, ctypes.byref(result))
        if result.value == GED_RETURN_VALUE_SUCCESS and dst_reg_file == GED_REG_FILE_GRF:
            dst_reg_num = ged.GED_GetDstRegNum(ins_ptr, ctypes.byref(result))
            if result.value == GED_RETURN_VALUE_SUCCESS:
                dsts.append(dst_reg_num)

        # Extract source 0 register
        src0_reg_file = ged.GED_GetSrc0RegFile(ins_ptr, ctypes.byref(result))
        if result.value == GED_RETURN_VALUE_SUCCESS and src0_reg_file == GED_REG_FILE_GRF:
            # Check if it's not an immediate
            src0_is_imm = ged.GED_GetSrc0IsImm(ins_ptr, ctypes.byref(result))
            if result.value != GED_RETURN_VALUE_SUCCESS or src0_is_imm == 0:
                src0_reg_num = ged.GED_GetSrc0RegNum(ins_ptr, ctypes.byref(result))
                if result.value == GED_RETURN_VALUE_SUCCESS:
                    srcs.append(src0_reg_num)

        # Extract source 1 register
        src1_reg_file = ged.GED_GetSrc1RegFile(ins_ptr, ctypes.byref(result))
        if result.value == GED_RETURN_VALUE_SUCCESS and src1_reg_file == GED_REG_FILE_GRF:
            # Check if it's not an immediate
            src1_is_imm = ged.GED_GetSrc1IsImm(ins_ptr, ctypes.byref(result))
            if result.value != GED_RETURN_VALUE_SUCCESS or src1_is_imm == 0:
                src1_reg_num = ged.GED_GetSrc1RegNum(ins_ptr, ctypes.byref(result))
                if result.value == GED_RETURN_VALUE_SUCCESS:
                    srcs.append(src1_reg_num)

        # Extract source 2 register (for 3-operand instructions like mad, add3)
        src2_reg_file = ged.GED_GetSrc2RegFile(ins_ptr, ctypes.byref(result))
        if result.value == GED_RETURN_VALUE_SUCCESS and src2_reg_file == GED_REG_FILE_GRF:
            src2_reg_num = ged.GED_GetSrc2RegNum(ins_ptr, ctypes.byref(result))
            if result.value == GED_RETURN_VALUE_SUCCESS:
                srcs.append(src2_reg_num)

        return dsts, srcs

    def _decode_section(
        self,
        data: bytes,
        base_offset: int = 0,
    ) -> List[Tuple[int, str, int, List[int], List[int]]]:
        """Decode all instructions in a code section.

        Args:
            data: Raw instruction bytes.
            base_offset: Base offset to add to instruction PCs.

        Returns:
            List of (offset, opcode_name, size, dsts, srcs) tuples.
        """
        ged = self._load_ged()
        if ged is None:
            raise DisassemblerError("GED library not available")

        instructions = []
        offset = 0
        data_len = len(data)

        # Create GED instruction buffer
        ins_buffer = ctypes.create_string_buffer(GED_INS_SIZE)
        ins_ptr = ctypes.cast(ins_buffer, ctypes.c_void_p)

        while offset < data_len:
            # Ensure we have at least 8 bytes (minimum instruction size)
            remaining = data_len - offset
            if remaining < 8:
                break

            # Get pointer to instruction bytes
            chunk_size = min(16, remaining)  # Max instruction size is 16 bytes
            chunk = data[offset:offset + chunk_size]
            chunk_buffer = ctypes.create_string_buffer(chunk)
            chunk_ptr = ctypes.cast(chunk_buffer, ctypes.c_void_p)

            # Decode instruction
            result = ged.GED_DecodeIns(
                self._gpu_model,
                chunk_ptr,
                ctypes.c_uint32(chunk_size),
                ins_ptr,
            )

            if result != GED_RETURN_VALUE_SUCCESS:
                # Skip 8 bytes on decode failure (might be data or padding)
                offset += 8
                continue

            # Get instruction info
            opcode_id = ged.GED_GetOpcode(ins_ptr)
            ins_size = ged.GED_InsSize(ins_ptr)

            opcode_name = self._opcode_to_name(opcode_id)

            # Extract operands
            dsts, srcs = self._extract_operands_from_ged(ins_ptr, ged)

            # Extract SWSB field
            swsb_result = ctypes.c_uint32()
            swsb_raw = ged.GED_GetSWSB(ins_ptr, ctypes.byref(swsb_result))
            if swsb_result.value != GED_RETURN_VALUE_SUCCESS:
                swsb_raw = 0

            # Record instruction
            instructions.append((base_offset + offset, opcode_name, ins_size, dsts, srcs, swsb_raw))

            # Move to next instruction
            offset += ins_size

        return instructions

    def disassemble(
        self,
        binary_path: str,
        function_index: Optional[int] = None,
        timeout: int = 60,
    ) -> str:
        """Disassemble an Intel GPU binary file.

        Note: Unlike NVIDIA/AMD disassemblers that return text output,
        this returns a structured string representation since GED
        doesn't produce text output.

        Args:
            binary_path: Path to Intel zebin file (.gpubin).
            function_index: Optional function index (not used - GED decodes all).
            timeout: Timeout in seconds (not used - GED is fast).

        Returns:
            String representation of disassembly (for interface compatibility).

        Raises:
            DisassemblerError: If disassembly fails.
        """
        try:
            path = require_file_exists(Path(binary_path), "File")
        except FileNotFoundError as exc:
            raise DisassemblerError(str(exc))

        if not self.check_available():
            raise DisassemblerError(
                f"GED library not available. Set GED_LIBRARY_PATH or install GTPin."
            )

        # Parse zebin to get sections
        try:
            parser = ZebinParser(path)
        except Exception as e:
            raise DisassemblerError(f"Failed to parse zebin: {e}")

        # Build text representation
        lines = []
        for section_name, section in parser.sections.items():
            lines.append(f"\n.section {section_name}")

            # Decode instructions in this section
            try:
                instrs = self._decode_section(section.data, base_offset=0)
            except DisassemblerError as e:
                lines.append(f"  ; Error decoding: {e}")
                continue

            for offset, opcode, size, dsts, srcs, swsb_raw in instrs:
                # Format: /*offset*/ opcode dsts=r0,r1 srcs=r2,r3
                # Include GRF register operands so parse_instruction_line()
                # can recover them for the text round-trip path.
                parts = [f"  /*{offset:04x}*/ {opcode}"]
                if dsts:
                    parts.append(f"dsts={','.join(str(r) for r in dsts)}")
                if srcs:
                    parts.append(f"srcs={','.join(str(r) for r in srcs)}")
                lines.append(" ".join(parts))

        return "\n".join(lines)

    def parse_instruction_line(self, line: str) -> Optional[ParsedInstruction]:
        """Parse a single instruction line.

        Parses the text format produced by disassemble(), which encodes
        GRF register operands as ``dsts=N,M srcs=N,M`` tokens.

        Args:
            line: Line from disassemble() output.

        Returns:
            ParsedInstruction or None.
        """
        import re

        # Match format: /*offset*/ opcode [dsts=... srcs=...]
        match = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s+(\S+)(.*)", line)
        if not match:
            return None

        offset = int(match.group(1), 16)
        opcode = match.group(2)
        rest = match.group(3).strip()

        # Parse register operands from dsts=N,M srcs=N,M tokens
        operands: List[str] = []
        if rest:
            for token in rest.split():
                operands.append(token)

        return ParsedInstruction(
            offset=offset,
            opcode=opcode,
            operands=operands,
            raw_line=line.strip(),
            predicate=None,
        )

    def _classify_intel_opcode(self, opcode: str) -> str:
        """Classify Intel opcode into a category.

        Uses the same categories as intel.py architecture module.

        Returns:
            Category string: "MEMORY", "SYNC", "CONTROL", or "ALU"
        """
        from leo.arch.intel import (
            INTEL_MEMORY_OPCODES,
            INTEL_SYNC_OPCODES,
            INTEL_CONTROL_OPCODES,
        )

        op_lower = opcode.lower()

        if op_lower in INTEL_MEMORY_OPCODES:
            return "MEMORY"
        if op_lower in INTEL_SYNC_OPCODES:
            return "SYNC"
        if op_lower in INTEL_CONTROL_OPCODES:
            return "CONTROL"
        return "ALU"

    def _is_store_op(self, opcode: str) -> bool:
        """Check if opcode is a store operation.

        Intel uses SEND-based memory operations; actual store vs load
        is determined by message descriptor, not opcode.
        """
        # For Intel, we can't determine from opcode alone
        # All SEND operations could be stores
        return False

    def _is_branch_op(self, opcode: str) -> bool:
        """Check if opcode is a branch/control flow operation."""
        from leo.arch.intel import INTEL_CONTROL_OPCODES
        return opcode.lower() in INTEL_CONTROL_OPCODES

    def parsed_to_instruction_stat(self, parsed: ParsedInstruction) -> InstructionStat:
        """Convert ParsedInstruction to InstructionStat.

        Extracts GRF register operands from ``dsts=N,M`` / ``srcs=N,M``
        tokens stored in parsed.operands by parse_instruction_line().

        Args:
            parsed: Raw parsed instruction.

        Returns:
            InstructionStat with opcode, PC, and GRF register operands.
        """
        # Intel doesn't have explicit predicate guards in the same way
        # Predication is done via execution mask
        pred_reg = -1
        pred_flag = PredicateFlag.PREDICATE_NONE

        # Extract GRF register numbers from operand tokens
        dsts: List[int] = []
        srcs: List[int] = []
        for operand in parsed.operands:
            if operand.startswith("dsts="):
                dsts = [int(r) for r in operand[5:].split(",") if r]
            elif operand.startswith("srcs="):
                srcs = [int(r) for r in operand[5:].split(",") if r]

        return InstructionStat(
            op=parsed.opcode,
            pc=parsed.offset,
            predicate=pred_reg,
            predicate_flag=pred_flag,
            dsts=dsts,
            srcs=srcs,
            pdsts=[],
            psrcs=[],
            bdsts=[],
            bsrcs=[],
            udsts=[],
            usrcs=[],
            indirect=False,
            branch_target=None,
        )

    def parse_function(
        self, output: str, function_name: Optional[str] = None
    ) -> Optional[ParsedFunction]:
        """Parse disassembler output to extract a function.

        Args:
            output: Raw disassembler output.
            function_name: Optional function name to extract.

        Returns:
            ParsedFunction or None.
        """
        functions = self.parse_all_functions(output)

        if function_name:
            for func in functions:
                if func.name == function_name or function_name in func.name:
                    return func
            return None
        elif functions:
            return functions[0]
        return None

    def parse_all_functions(self, output: str) -> List[ParsedFunction]:
        """Parse disassembler output to extract all functions.

        Args:
            output: Raw disassembler output.

        Returns:
            List of ParsedFunctions.
        """
        import re

        lines = output.split("\n")
        functions: List[ParsedFunction] = []
        current_func: Optional[ParsedFunction] = None

        for line in lines:
            # Detect section start: .section .text.kernel_name
            section_match = re.match(r"\.section\s+(\S+)", line)
            if section_match:
                if current_func and current_func.instructions:
                    functions.append(current_func)

                section_name = section_match.group(1)
                # Extract function name from section (e.g., .text._Z10kernelPfi -> _Z10kernelPfi)
                if section_name.startswith(".text."):
                    func_name = section_name[6:]  # Remove ".text."
                else:
                    func_name = section_name

                current_func = ParsedFunction(name=func_name)
                continue

            # Parse instructions
            if current_func:
                parsed = self.parse_instruction_line(line)
                if parsed:
                    inst = self.parsed_to_instruction_stat(parsed)
                    current_func.instructions.append(inst)

        # Don't forget last function
        if current_func and current_func.instructions:
            functions.append(current_func)

        return functions

    def disassemble_and_parse_all(self, binary_path: str) -> List[ParsedFunction]:
        """Disassemble and parse all functions from an Intel GPU binary.

        This is a more efficient method that directly uses GED to decode
        instructions without going through the text representation.

        Args:
            binary_path: Path to Intel zebin file (.gpubin).

        Returns:
            List of ParsedFunction, one per function in the binary.

        Raises:
            DisassemblerError: If disassembly fails.
        """
        try:
            path = require_file_exists(Path(binary_path), "File")
        except FileNotFoundError as exc:
            raise DisassemblerError(str(exc))

        if not self.check_available():
            raise DisassemblerError(
                f"GED library not available. Set GED_LIBRARY_PATH or install GTPin."
            )

        # Parse zebin to get sections and functions
        try:
            parser = ZebinParser(path)
        except Exception as e:
            raise DisassemblerError(f"Failed to parse zebin: {e}")

        functions: List[ParsedFunction] = []

        # Process each text section
        for section_name, section in parser.sections.items():
            # Extract function name from section name
            if section_name.startswith(".text."):
                func_name = section_name[6:]  # Remove ".text."
            elif section_name == ".text":
                func_name = "_text"
            else:
                func_name = section_name

            # Decode instructions
            try:
                instrs = self._decode_section(section.data, base_offset=0)
            except DisassemblerError:
                continue

            # Build ParsedFunction
            func = ParsedFunction(name=func_name)

            from leo.binary.swsb import decode_swsb_xehpc

            for offset, opcode, size, dsts, srcs, swsb_raw in instrs:
                swsb = decode_swsb_xehpc(swsb_raw)
                inst = InstructionStat(
                    op=opcode,
                    pc=offset,
                    predicate=-1,
                    predicate_flag=PredicateFlag.PREDICATE_NONE,
                    dsts=dsts,
                    srcs=srcs,
                    swsb=swsb,
                )
                func.instructions.append(inst)

            if func.instructions:
                functions.append(func)

        return functions


# =============================================================================
# Intel-specific helper functions
# =============================================================================


def get_instruction_size(is_compact: bool) -> int:
    """Get Intel instruction size based on compaction.

    Args:
        is_compact: True if instruction is compacted.

    Returns:
        Instruction size in bytes (8 or 16).
    """
    return 8 if is_compact else 16
