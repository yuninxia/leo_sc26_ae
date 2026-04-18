"""NVIDIA GPU disassembler using nvdisasm.

Wraps NVIDIA's nvdisasm tool to disassemble CUDA binaries (cubin/gpubin)
and parses the output into structured instruction data.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from leo.binary.instruction import (
    InstructionStat,
    PredicateFlag,
    detect_indirect_addressing,
    get_operation_width,
    parse_barrier,
    parse_predicate,
    parse_registers,
    parse_uniform_register,
)
from leo.utils.validation import require_file_exists

from .base import (
    Disassembler,
    DisassemblerError,
    ParsedFunction,
    ParsedInstruction,
    register_disassembler,
)


@register_disassembler("nvidia")
class NVIDIADisassembler(Disassembler):
    """NVIDIA GPU disassembler using nvdisasm."""

    @property
    def vendor(self) -> str:
        return "nvidia"

    @property
    def tool_name(self) -> str:
        return "nvdisasm"

    def check_available(self) -> bool:
        """Check if nvdisasm is available in PATH."""
        try:
            result = subprocess.run(
                ["nvdisasm", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_version(self) -> Optional[str]:
        """Get nvdisasm version string."""
        try:
            result = subprocess.run(
                ["nvdisasm", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse version from output like "Cuda compilation tools, release 12.9, V12.9.88"
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
            return None
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def disassemble(
        self,
        binary_path: str,
        function_index: Optional[int] = None,
        timeout: int = 60,
    ) -> str:
        """Invoke nvdisasm on a CUDA binary file.

        Args:
            binary_path: Path to cubin/gpubin file.
            function_index: Optional function symbol index to disassemble.
            timeout: Timeout in seconds.

        Returns:
            Raw nvdisasm output as string.

        Raises:
            DisassemblerError: If nvdisasm fails or times out.
        """
        try:
            path = require_file_exists(Path(binary_path), "File")
        except FileNotFoundError as exc:
            raise DisassemblerError(str(exc))

        cmd = ["nvdisasm"]
        if function_index is not None:
            cmd.extend(["-fun", str(function_index)])
        cmd.append(str(path))

        result = self._run_command(cmd, timeout, "nvdisasm")
        # nvdisasm may print warnings to stderr but still succeed
        if result.returncode != 0 and "fatal" in result.stderr.lower():
            raise DisassemblerError(f"nvdisasm failed: {result.stderr}")
        return result.stdout

    def parse_instruction_line(self, line: str) -> Optional[ParsedInstruction]:
        """Parse a single instruction line from nvdisasm output.

        Expected format: "        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;"

        Args:
            line: Single line from nvdisasm output.

        Returns:
            ParsedInstruction or None if not an instruction line.
        """
        # Skip non-instruction lines
        if "/*" not in line or "*/" not in line:
            return None

        # Extract offset from /*XXXX*/
        offset_match = re.search(r"/\*([0-9a-fA-F]+)\*/", line)
        if not offset_match:
            return None

        try:
            offset = int(offset_match.group(1), 16)
        except ValueError:
            return None

        # Get instruction part after the offset comment
        instr_part = line[offset_match.end() :].strip()

        # Skip data directives (.byte, .dword, etc.)
        if instr_part.startswith("."):
            return None

        # Remove trailing semicolon and comments
        instr_part = re.sub(r"\s*;.*$", "", instr_part).strip()
        if not instr_part:
            return None

        # Check for predicate (@P0, @!P1)
        predicate = None
        pred_match = re.match(r"@(!?P\d)\s+", instr_part)
        if pred_match:
            predicate = pred_match.group(1)
            instr_part = instr_part[pred_match.end() :]

        # Split opcode and operands
        parts = instr_part.split(None, 1)
        if not parts:
            return None

        opcode = parts[0]
        operands = []

        if len(parts) > 1:
            # Parse operands - split by comma but handle brackets
            operand_str = parts[1]
            operands = self._split_operands(operand_str)

        return ParsedInstruction(
            offset=offset,
            predicate=predicate,
            opcode=opcode,
            operands=operands,
            raw_line=line.strip(),
        )

    def _parse_predicate_string(
        self, pred_str: Optional[str]
    ) -> Tuple[int, PredicateFlag]:
        """Parse predicate string like "P0" or "!P1".

        Returns:
            Tuple of (predicate_reg, predicate_flag).
        """
        if pred_str is None:
            return (-1, PredicateFlag.PREDICATE_NONE)

        negated = pred_str.startswith("!")
        pred_str = pred_str.lstrip("!")

        match = re.match(r"P(\d)", pred_str)
        if match:
            pred_reg = int(match.group(1))
            flag = (
                PredicateFlag.PREDICATE_FALSE
                if negated
                else PredicateFlag.PREDICATE_TRUE
            )
            return (pred_reg, flag)

        return (-1, PredicateFlag.PREDICATE_NONE)

    def _extract_branch_target_label(self, operand: str) -> Optional[str]:
        """Extract label from branch operand.

        Matches patterns like:
        - `(.L_x_0)
        - `(.L_x_123)
        """
        match = re.search(r"`\(?(\.[Ll]_[a-zA-Z0-9_]+)\)?", operand)
        return match.group(1) if match else None

    def parsed_to_instruction_stat(self, parsed: ParsedInstruction) -> InstructionStat:
        """Convert ParsedInstruction to InstructionStat with register analysis.

        Args:
            parsed: Raw parsed instruction.

        Returns:
            InstructionStat with register operands extracted.
        """
        pred_reg, pred_flag = self._parse_predicate_string(parsed.predicate)

        # Determine operation width from opcode modifiers
        width = get_operation_width(parsed.opcode)

        # Extract registers from operands
        dsts: List[int] = []
        srcs: List[int] = []
        pdsts: List[int] = []
        psrcs: List[int] = []
        bdsts: List[int] = []
        bsrcs: List[int] = []
        udsts: List[int] = []
        usrcs: List[int] = []

        # Check for indirect addressing
        indirect = any(detect_indirect_addressing(op) for op in parsed.operands)

        # Determine which operands are destinations vs sources
        # Most instructions: first operand is destination
        # Exceptions: stores (STG, STS) where memory is "destination"
        base_op = parsed.opcode.split(".")[0]
        is_store = base_op in {"ST", "STG", "STS", "STL", "ATOM", "RED"}

        for i, operand in enumerate(parsed.operands):
            # Parse different register types
            regs = parse_registers(operand, width)
            pred = parse_predicate(operand)
            barrier = parse_barrier(operand)
            ureg = parse_uniform_register(operand)

            # Determine if this is a destination or source
            if is_store:
                # For stores like STG [R1], R2: R1 is src (address), R2 is src (data)
                if regs:
                    srcs.extend(regs)
                if ureg is not None:
                    usrcs.append(ureg)
            else:
                # Standard: first operand is destination
                if i == 0:
                    if regs:
                        dsts.extend(regs)
                    if pred is not None:
                        pdsts.append(pred)
                    if barrier is not None:
                        bdsts.append(barrier)
                    if ureg is not None:
                        udsts.append(ureg)
                else:
                    if regs:
                        srcs.extend(regs)
                    if pred is not None:
                        psrcs.append(pred)
                    if barrier is not None:
                        bsrcs.append(barrier)
                    if ureg is not None:
                        usrcs.append(ureg)

        # Handle special cases
        # SETP instructions write to predicates
        if base_op in {"SETP", "ISETP", "FSETP", "DSETP"}:
            # First operand(s) are predicate destinations
            for operand in parsed.operands[:2]:  # Can have two predicate dests
                pred = parse_predicate(operand)
                if pred is not None and pred not in pdsts:
                    pdsts.append(pred)

        # Barrier instructions - ensure we capture barrier operands (avoid duplicates)
        if base_op in {"BAR", "BSSY", "BSYNC", "DEPBAR"}:
            for operand in parsed.operands:
                barrier = parse_barrier(operand)
                if barrier is not None and barrier not in bdsts:
                    bdsts.append(barrier)

        # If predicated, the predicate register is a source
        if pred_reg >= 0:
            psrcs.append(pred_reg)

        # Extract branch target label for branch/jump instructions
        branch_target: Optional[str] = None
        if base_op in {"BRA", "BRX", "JMP", "JMX", "CALL"}:
            for operand in parsed.operands:
                label = self._extract_branch_target_label(operand)
                if label:
                    branch_target = label
                    break

        return InstructionStat(
            op=parsed.opcode,
            pc=parsed.offset,
            predicate=pred_reg,
            predicate_flag=pred_flag,
            dsts=dsts,
            srcs=srcs,
            pdsts=pdsts,
            psrcs=psrcs,
            bdsts=bdsts,
            bsrcs=bsrcs,
            udsts=udsts,
            usrcs=usrcs,
            indirect=indirect,
            branch_target=branch_target,
        )

    def parse_function(
        self, output: str, function_name: Optional[str] = None
    ) -> Optional[ParsedFunction]:
        """Parse nvdisasm output to extract a function's instructions.

        Args:
            output: Raw nvdisasm output.
            function_name: Optional function name to extract. If None, extracts first.

        Returns:
            ParsedFunction with instructions, or None if not found.
        """
        lines = output.split("\n")
        functions: List[ParsedFunction] = []
        current_func: Optional[ParsedFunction] = None
        in_text_section = False
        pending_labels: List[str] = []

        for line in lines:
            # Detect function start
            # Format: ".section .text._Z8xcomputePKdS0_Pdii,"
            section_match = re.search(r'\.section\s+\.text\.([^,\s"]+)', line)
            if section_match:
                if current_func and current_func.instructions:
                    functions.append(current_func)
                func_name = section_match.group(1)
                current_func = ParsedFunction(name=func_name)
                in_text_section = True
                pending_labels = []
                continue

            # Detect labels within function
            # Format: ".L_x_0:" or ".L_index_123:"
            label_match = re.match(r"^(\.[Ll]_[a-zA-Z0-9_]+):\s*$", line.strip())
            if label_match and current_func:
                pending_labels.append(label_match.group(1))
                continue

            # Parse instructions
            if in_text_section and current_func:
                parsed = self.parse_instruction_line(line)
                if parsed:
                    inst = self.parsed_to_instruction_stat(parsed)
                    current_func.instructions.append(inst)

                    # Associate pending labels with this instruction's PC
                    for label in pending_labels:
                        current_func.labels[label] = parsed.offset
                    pending_labels = []

            # Detect end of text section
            stripped = line.strip()
            if re.match(r"\.section\s", stripped) and ".text." not in line:
                in_text_section = False

        # Don't forget last function
        if current_func and current_func.instructions:
            functions.append(current_func)

        # Return requested function or first one
        if function_name:
            for func in functions:
                if func.name == function_name or function_name in func.name:
                    return func
            return None
        elif functions:
            return functions[0]
        return None

    def parse_all_functions(self, output: str) -> List[ParsedFunction]:
        """Parse nvdisasm output to extract all functions.

        Args:
            output: Raw nvdisasm output.

        Returns:
            List of ParsedFunctions with instructions.
        """
        lines = output.split("\n")
        functions: List[ParsedFunction] = []
        current_func: Optional[ParsedFunction] = None
        in_text_section = False
        pending_labels: List[str] = []

        for line in lines:
            section_match = re.search(r'\.section\s+\.text\.([^,\s"]+)', line)
            if section_match:
                if current_func and current_func.instructions:
                    functions.append(current_func)
                func_name = section_match.group(1)
                current_func = ParsedFunction(name=func_name)
                in_text_section = True
                pending_labels = []
                continue

            # Detect labels within function
            label_match = re.match(r"^(\.[Ll]_[a-zA-Z0-9_]+):\s*$", line.strip())
            if label_match and current_func:
                pending_labels.append(label_match.group(1))
                continue

            if in_text_section and current_func:
                parsed = self.parse_instruction_line(line)
                if parsed:
                    inst = self.parsed_to_instruction_stat(parsed)
                    current_func.instructions.append(inst)

                    # Associate pending labels with this instruction's PC
                    for label in pending_labels:
                        current_func.labels[label] = parsed.offset
                    pending_labels = []

            # Detect end of text section
            stripped = line.strip()
            if re.match(r"\.section\s", stripped) and ".text." not in line:
                in_text_section = False

        if current_func and current_func.instructions:
            functions.append(current_func)

        return functions


# =============================================================================
# Branch Target and Edge Type Helpers (backward compatibility)
# =============================================================================


def extract_branch_target_label(operand: str) -> Optional[str]:
    """Extract label from branch operand.

    Matches patterns like:
    - `(.L_x_0)
    - `(.L_x_123)
    - `(.L_index_456)

    Args:
        operand: Operand string from nvdisasm, e.g., "`(.L_x_0)"

    Returns:
        Label name like ".L_x_0" or None if not a label.
    """
    match = re.search(r"`\(?(\.[Ll]_[a-zA-Z0-9_]+)\)?", operand)
    return match.group(1) if match else None


def extract_call_target(operand: str) -> Optional[str]:
    """Extract function name from CALL operand.

    Matches patterns like:
    - `(_Z10helperFunc)
    - `(_Z3foo)

    Args:
        operand: Operand string from CALL instruction.

    Returns:
        Function name or None if not found.
    """
    match = re.search(r"`\(?([_a-zA-Z][_a-zA-Z0-9]*)\)?", operand)
    return match.group(1) if match else None


def get_branch_target_from_instruction(
    inst: InstructionStat, operands: List[str]
) -> Optional[str]:
    """Get branch target label from an instruction's operands.

    Args:
        inst: The instruction to check.
        operands: Original operand strings (not parsed registers).

    Returns:
        Label name if this is a branch with a label target, None otherwise.
    """
    if not inst.is_branch():
        return None

    for operand in operands:
        label = extract_branch_target_label(operand)
        if label:
            return label
    return None
