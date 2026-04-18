"""AMD GPU disassembler using llvm-objdump.

Wraps LLVM's llvm-objdump tool to disassemble AMD GPU binaries
(Code Objects, .co, .hsaco) and parses the output into structured
instruction data.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from leo.binary.instruction import Control, InstructionStat, PredicateFlag
from leo.utils.validation import require_file_exists

from .base import (
    Disassembler,
    DisassemblerError,
    ParsedFunction,
    ParsedInstruction,
    register_disassembler,
)


# Find llvm-objdump in common ROCm locations or PATH
def _find_llvm_objdump() -> str:
    """Find llvm-objdump in common locations."""
    import shutil
    import os

    # First check if it's in PATH
    path_result = shutil.which("llvm-objdump")
    if path_result:
        return path_result

    # Check ROCM_PATH environment variable
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        candidate = Path(rocm_path) / "llvm" / "bin" / "llvm-objdump"
        if candidate.exists():
            return str(candidate)

    # Check common ROCm installation paths (newest first)
    common_paths = [
        "/opt/rocm/llvm/bin/llvm-objdump",  # symlink to current
        "/opt/rocm-7.2.0/llvm/bin/llvm-objdump",
        "/opt/rocm-7.1.1/llvm/bin/llvm-objdump",
        "/opt/rocm-7.0.1/llvm/bin/llvm-objdump",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    # Fallback - let it fail later with a clear error
    return "llvm-objdump"


DEFAULT_LLVM_OBJDUMP = _find_llvm_objdump()


@register_disassembler("amd")
class AMDDisassembler(Disassembler):
    """AMD GPU disassembler using llvm-objdump."""

    def __init__(self, llvm_objdump_path: Optional[str] = None, gpu_arch: str = "gfx90a"):
        """Initialize AMD disassembler.

        Args:
            llvm_objdump_path: Path to llvm-objdump. If None, uses default ROCm path.
            gpu_arch: GPU architecture (e.g., "gfx90a" for MI250, "gfx942" for MI300).
        """
        self._llvm_objdump = llvm_objdump_path or DEFAULT_LLVM_OBJDUMP
        self._gpu_arch = gpu_arch

    @property
    def vendor(self) -> str:
        return "amd"

    @property
    def tool_name(self) -> str:
        return "llvm-objdump"

    def check_available(self) -> bool:
        """Check if llvm-objdump is available."""
        try:
            result = subprocess.run(
                [self._llvm_objdump, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_version(self) -> Optional[str]:
        """Get llvm-objdump version string."""
        try:
            result = subprocess.run(
                [self._llvm_objdump, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse version from output like "LLVM version 14.0.0"
                match = re.search(r"LLVM version (\d+\.\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
                # Alternative format: "llvm-objdump, LLVM 14.0.0"
                match = re.search(r"LLVM (\d+\.\d+\.\d+)", result.stdout)
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
        """Invoke llvm-objdump on an AMD GPU binary file.

        Args:
            binary_path: Path to code object file (.co, .hsaco).
            function_index: Optional (not used for llvm-objdump).
            timeout: Timeout in seconds.

        Returns:
            Raw llvm-objdump output as string.

        Raises:
            DisassemblerError: If llvm-objdump fails or times out.
        """
        try:
            path = require_file_exists(Path(binary_path), "File")
        except FileNotFoundError as exc:
            raise DisassemblerError(str(exc))

        cmd = [
            self._llvm_objdump,
            "-d",  # Disassemble
            "--arch-name=amdgcn",
            f"--mcpu={self._gpu_arch}",
            str(path),
        ]

        result = self._run_command(
            cmd,
            timeout,
            "llvm-objdump",
            tool_path=self._llvm_objdump,
        )
        if result.returncode != 0:
            raise DisassemblerError(f"llvm-objdump failed: {result.stderr}")
        return result.stdout

    def parse_instruction_line(self, line: str) -> Optional[ParsedInstruction]:
        """Parse a single instruction line from llvm-objdump output.

        Expected formats:
        - "s_mov_b32 m0, 0x10000 // 000000000100: BEFC00FF 00010000"
        - "       4:	7e000280        	v_mov_b32_e32 v0, 0"

        Args:
            line: Single line from llvm-objdump output.

        Returns:
            ParsedInstruction or None if not an instruction line.
        """
        # Try format with comment: "instr // addr: bytes"
        comment_match = re.search(
            r"^(.+?)\s+//\s+([0-9a-fA-F]+):\s+([0-9a-fA-F\s]+)$", line.strip()
        )
        if comment_match:
            instr_part = comment_match.group(1).strip()
            offset = int(comment_match.group(2), 16)
            return self._parse_instruction_part(instr_part, offset, line)

        # Try format: "addr: bytes opcode operands"
        # Example: "       4:	7e000280        	v_mov_b32_e32 v0, 0"
        addr_match = re.match(
            r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F\s]+)\s+(\S+.*?)$", line
        )
        if addr_match:
            offset = int(addr_match.group(1), 16)
            instr_part = addr_match.group(3).strip()
            return self._parse_instruction_part(instr_part, offset, line)

        return None

    def _parse_instruction_part(
        self, instr_part: str, offset: int, raw_line: str
    ) -> Optional[ParsedInstruction]:
        """Parse the instruction mnemonic and operands.

        Args:
            instr_part: Instruction string like "s_mov_b32 m0, 0x10000"
            offset: Instruction offset/PC
            raw_line: Original line for debugging

        Returns:
            ParsedInstruction or None if invalid.
        """
        if not instr_part:
            return None

        # Split opcode and operands
        parts = instr_part.split(None, 1)
        if not parts:
            return None

        opcode = parts[0]

        # Skip data/metadata directives
        if opcode.startswith(".") or opcode.startswith("0x"):
            return None

        operands = []
        if len(parts) > 1:
            operand_str = parts[1]
            operands = self._split_operands(operand_str)

        return ParsedInstruction(
            offset=offset,
            predicate=None,  # AMD uses EXEC mask, not explicit predicates
            opcode=opcode,
            operands=operands,
            raw_line=raw_line.strip(),
        )

    def _parse_vgpr(self, operand: str) -> List[int]:
        """Parse VGPR register(s) from operand.

        Handles: v0, v[0:1], v[0:3]

        Returns:
            List of VGPR register IDs.
        """
        # Single VGPR: v0, v255
        single_match = re.match(r"^v(\d+)$", operand, re.IGNORECASE)
        if single_match:
            return [int(single_match.group(1))]

        # VGPR range: v[0:1], v[0:3]
        range_match = re.match(r"^v\[(\d+):(\d+)\]$", operand, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return list(range(start, end + 1))

        return []

    def _parse_sgpr(self, operand: str) -> List[int]:
        """Parse SGPR register(s) from operand.

        Handles: s0, s[0:1], s[0:3]

        Returns:
            List of SGPR register IDs.
        """
        # Single SGPR: s0, s103
        single_match = re.match(r"^s(\d+)$", operand, re.IGNORECASE)
        if single_match:
            return [int(single_match.group(1))]

        # SGPR range: s[0:1], s[0:3]
        range_match = re.match(r"^s\[(\d+):(\d+)\]$", operand, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return list(range(start, end + 1))

        return []

    def _parse_accvgpr(self, operand: str) -> List[int]:
        """Parse AccVGPR register(s) from operand (CDNA).

        Handles: a0, a255, acc0

        Returns:
            List of AccVGPR register IDs.
        """
        # Single AccVGPR: a0, a255
        match = re.match(r"^a(\d+)$", operand, re.IGNORECASE)
        if match:
            return [int(match.group(1))]

        # Alternative format: acc0
        match = re.match(r"^acc(\d+)$", operand, re.IGNORECASE)
        if match:
            return [int(match.group(1))]

        # Range: a[0:3]
        range_match = re.match(r"^a\[(\d+):(\d+)\]$", operand, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return list(range(start, end + 1))

        return []

    def _is_special_register(self, operand: str) -> bool:
        """Check if operand is a special register."""
        special_regs = {
            "exec",
            "exec_lo",
            "exec_hi",
            "vcc",
            "vcc_lo",
            "vcc_hi",
            "scc",
            "m0",
            "vccz",
            "execz",
            "null",
            "flat_scratch",
            "flat_scratch_lo",
            "flat_scratch_hi",
        }
        return operand.lower() in special_regs

    def _classify_amd_opcode(self, opcode: str) -> str:
        """Classify AMD opcode into a category.

        Returns category like "MEMORY", "FLOAT", "INTEGER", etc.
        """
        op_lower = opcode.lower()

        # Memory operations
        if any(
            op_lower.startswith(mem)
            for mem in [
                "global_load",
                "global_store",
                "global_atomic",
                "flat_load",
                "flat_store",
                "flat_atomic",
                "buffer_load",
                "buffer_store",
                "buffer_atomic",
                "ds_read",
                "ds_write",
                "ds_add",
                "ds_sub",
                "ds_min",
                "ds_max",
                "ds_and",
                "ds_or",
                "ds_xor",
                "ds_cmpst",
                "ds_swizzle",
                "scratch_load",
                "scratch_store",
                "s_load",
                "s_store",
                "s_buffer_load",
            ]
        ):
            return "MEMORY"

        # Synchronization
        if any(
            op_lower.startswith(sync)
            for sync in [
                "s_barrier",
                "s_waitcnt",
                "s_sleep",
                "s_memtime",
                "s_memrealtime",
            ]
        ):
            return "CONTROL"

        # Matrix operations (MFMA)
        if op_lower.startswith("v_mfma") or op_lower.startswith("mfma"):
            return "TENSOR"

        # Control flow
        if any(
            op_lower.startswith(ctrl)
            for ctrl in [
                "s_branch",
                "s_cbranch",
                "s_call",
                "s_return",
                "s_endpgm",
                "s_setpc",
                "s_getpc",
                "s_swappc",
            ]
        ):
            return "CONTROL"

        # Floating point (VALU)
        if any(
            op_lower.startswith(fp)
            for fp in [
                "v_add_f",
                "v_sub_f",
                "v_mul_f",
                "v_fma_f",
                "v_mac_f",
                "v_mad_f",
                "v_div_f",
                "v_sqrt_f",
                "v_rsq_f",
                "v_rcp_f",
                "v_exp_f",
                "v_log_f",
                "v_sin_f",
                "v_cos_f",
                "v_floor_f",
                "v_ceil_f",
                "v_trunc_f",
                "v_fract_f",
                "v_min_f",
                "v_max_f",
                "v_cmp_f",
                "v_cmpx_f",
                "v_cndmask",
            ]
        ):
            return "FLOAT"

        # Integer (VALU)
        if any(
            op_lower.startswith(intop)
            for intop in [
                "v_add_i",
                "v_add_u",
                "v_sub_i",
                "v_sub_u",
                "v_mul_i",
                "v_mul_u",
                "v_mad_i",
                "v_mad_u",
                "v_and_b",
                "v_or_b",
                "v_xor_b",
                "v_not_b",
                "v_lshl",
                "v_lshr",
                "v_ashr",
                "v_bfe",
                "v_bfi",
                "v_min_i",
                "v_min_u",
                "v_max_i",
                "v_max_u",
                "v_cmp_i",
                "v_cmp_u",
                "v_cmpx_i",
                "v_cmpx_u",
            ]
        ):
            return "INTEGER"

        # Scalar ALU (SALU)
        if op_lower.startswith("s_") and not op_lower.startswith("s_load") and not op_lower.startswith("s_store"):
            if any(
                salu in op_lower
                for salu in [
                    "_add",
                    "_sub",
                    "_mul",
                    "_and",
                    "_or",
                    "_xor",
                    "_not",
                    "_lshl",
                    "_lshr",
                    "_ashr",
                    "_bfe",
                    "_cmp",
                    "_mov",
                    "_cselect",
                ]
            ):
                return "INTEGER"

        # Conversion
        if "v_cvt_" in op_lower:
            return "CONVERT"

        # Default vector move
        if op_lower.startswith("v_mov"):
            return "MISC"

        # Default scalar move
        if op_lower.startswith("s_mov"):
            return "MISC"

        return "UNKNOWN"

    def _is_store_op(self, opcode: str) -> bool:
        """Check if opcode is a store operation."""
        op_lower = opcode.lower()
        return any(
            store in op_lower
            for store in [
                "global_store",
                "flat_store",
                "buffer_store",
                "ds_write",
                "scratch_store",
                "s_store",
            ]
        )

    def _is_branch_op(self, opcode: str) -> bool:
        """Check if opcode is a branch/jump operation."""
        op_lower = opcode.lower()
        return any(
            branch in op_lower
            for branch in [
                "s_branch",
                "s_cbranch",
                "s_call",
                "s_return",
                "s_setpc",
                "s_swappc",
            ]
        )

    def parsed_to_instruction_stat(self, parsed: ParsedInstruction) -> InstructionStat:
        """Convert ParsedInstruction to InstructionStat with register analysis.

        For AMD:
        - VGPRs map to dsts/srcs (like NVIDIA general registers)
        - SGPRs map to udsts/usrcs (uniform registers)
        - AccVGPRs are tracked separately in dsts/srcs with offset

        Args:
            parsed: Raw parsed instruction.

        Returns:
            InstructionStat with register operands extracted.
        """
        # AMD doesn't have explicit predicate guards like NVIDIA
        # Uses EXEC mask implicitly
        pred_reg = -1
        pred_flag = PredicateFlag.PREDICATE_NONE

        # Extract registers from operands
        dsts: List[int] = []
        srcs: List[int] = []
        pdsts: List[int] = []
        psrcs: List[int] = []
        bdsts: List[int] = []
        bsrcs: List[int] = []
        udsts: List[int] = []  # SGPRs go here
        usrcs: List[int] = []

        is_store = self._is_store_op(parsed.opcode)

        for i, operand in enumerate(parsed.operands):
            # Skip modifiers like "offset:0", "glc", "slc"
            if ":" in operand and not operand.startswith("v[") and not operand.startswith("s["):
                continue

            # Parse different register types
            vgprs = self._parse_vgpr(operand)
            sgprs = self._parse_sgpr(operand)
            accvgprs = self._parse_accvgpr(operand)

            # Determine if this is a destination or source
            if is_store:
                # For stores: all registers are sources
                if vgprs:
                    srcs.extend(vgprs)
                if sgprs:
                    usrcs.extend(sgprs)
                if accvgprs:
                    srcs.extend(accvgprs)  # AccVGPRs treated like VGPRs
            else:
                # Standard: first operand is destination
                if i == 0:
                    if vgprs:
                        dsts.extend(vgprs)
                    if sgprs:
                        udsts.extend(sgprs)
                    if accvgprs:
                        dsts.extend(accvgprs)
                else:
                    if vgprs:
                        srcs.extend(vgprs)
                    if sgprs:
                        usrcs.extend(sgprs)
                    if accvgprs:
                        srcs.extend(accvgprs)

        # Extract branch target for control flow instructions
        branch_target: Optional[str] = None
        if self._is_branch_op(parsed.opcode):
            for operand in parsed.operands:
                # AMD branch targets are usually labels like "BB0_1" or ".LBB0_1"
                if operand.startswith(".") or operand.startswith("BB"):
                    branch_target = operand
                    break

        # Check for indirect addressing (memory through registers)
        indirect = any(
            "[" in op and ("v" in op.lower() or "s" in op.lower())
            for op in parsed.operands
        )

        # Capture operand details for enhanced analysis
        operands_raw = " ".join(parsed.operands) if parsed.operands else None
        operand_details = None

        # Parse s_waitcnt counters
        if parsed.opcode.lower().startswith("s_waitcnt"):
            operand_details = parse_waitcnt(operands_raw or "")

        # AMD ISA: s_nop N inserts N+1 idle cycles.
        # Parse into Control.stall for CFG-based latency pruning.
        control = Control(stall=1)  # default: 1 cycle per issue slot
        if parsed.opcode.lower() == "s_nop" and parsed.operands:
            try:
                nop_count = int(parsed.operands[0], 0)
                control = Control(stall=nop_count + 1)
            except (ValueError, IndexError):
                pass

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
            operands_raw=operands_raw,
            operand_details=operand_details,
            control=control,
        )

    def parse_function(
        self, output: str, function_name: Optional[str] = None
    ) -> Optional[ParsedFunction]:
        """Parse llvm-objdump output to extract a function's instructions.

        Args:
            output: Raw llvm-objdump output.
            function_name: Optional function name to extract. If None, extracts first.

        Returns:
            ParsedFunction with instructions, or None if not found.
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
        """Parse llvm-objdump output to extract all functions.

        Args:
            output: Raw llvm-objdump output.

        Returns:
            List of ParsedFunctions with instructions.
        """
        lines = output.split("\n")
        functions: List[ParsedFunction] = []
        current_func: Optional[ParsedFunction] = None
        pending_labels: List[str] = []

        for line in lines:
            # Detect function start
            # Format: "0000000000001000 <_Z10testKernelPfS_i>:"
            # or: "kernel_name:"
            func_match = re.match(r"^([0-9a-fA-F]+)\s+<([^>]+)>:\s*$", line)
            if func_match:
                if current_func and current_func.instructions:
                    functions.append(current_func)
                func_name = func_match.group(2)
                current_func = ParsedFunction(name=func_name)
                pending_labels = []
                continue

            # Alternative function format: "kernel_name:" at start of line
            simple_func_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*$", line)
            if simple_func_match and not line.strip().startswith("."):
                if current_func and current_func.instructions:
                    functions.append(current_func)
                func_name = simple_func_match.group(1)
                current_func = ParsedFunction(name=func_name)
                pending_labels = []
                continue

            # Detect labels within function
            # Format: ".LBB0_1:" or "BB0_1:"
            label_match = re.match(r"^(\.[Ll]BB[0-9_]+|BB[0-9_]+):\s*$", line.strip())
            if label_match and current_func:
                pending_labels.append(label_match.group(1))
                continue

            # Parse instructions
            if current_func:
                parsed = self.parse_instruction_line(line)
                if parsed:
                    inst = self.parsed_to_instruction_stat(parsed)
                    current_func.instructions.append(inst)

                    # Associate pending labels with this instruction's PC
                    for label in pending_labels:
                        current_func.labels[label] = parsed.offset
                    pending_labels = []

        # Don't forget last function
        if current_func and current_func.instructions:
            functions.append(current_func)

        return functions


# =============================================================================
# AMD-specific helper functions
# =============================================================================


def parse_waitcnt(operand_str: str) -> dict:
    """Parse s_waitcnt operand into counter values.

    Args:
        operand_str: Operand like "vmcnt(0) lgkmcnt(0)" or "0"

    Returns:
        Dict with counter names and values, e.g., {"vmcnt": 0, "lgkmcnt": 0}
    """
    result = {}

    # Pattern for named counters: vmcnt(N), lgkmcnt(N), expcnt(N), vscnt(N)
    for match in re.finditer(r"(vmcnt|lgkmcnt|expcnt|vscnt)\((\d+)\)", operand_str):
        counter_name = match.group(1)
        counter_value = int(match.group(2))
        result[counter_name] = counter_value

    # If just a number, it's a combined wait mask (legacy format)
    if not result:
        try:
            val = int(operand_str, 0)  # Handle hex or decimal
            # Decode combined mask (architecture-dependent bit layout)
            result["combined"] = val
        except ValueError:
            pass

    return result
