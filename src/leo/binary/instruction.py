"""Instruction data structures for CUDA binary analysis.

Based on GPA's AnalyzeInstruction.hpp - defines the core data structures
for representing parsed NVIDIA GPU instructions and their register dependencies.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional
import re


class PredicateFlag(IntEnum):
    """Predicate execution mode for an instruction."""

    PREDICATE_NONE = 0  # Unconditional execution
    PREDICATE_TRUE = 1  # @P0 - execute if predicate is true
    PREDICATE_FALSE = 2  # @!P0 - execute if predicate is false


# Constants for register counts
MAX_GENERAL_REGS = 256  # R0-R255
MAX_PREDICATE_REGS = 7  # P0-P6
MAX_BARRIER_REGS = 6  # B1-B6 (B0 reserved)
MAX_UNIFORM_REGS = 64  # UR0-UR63 (SM_75+)
BARRIER_NONE = 7  # Special value indicating no barrier

# In NVIDIA's actual encoding, barrier 0 also means "no barrier"
# GPA uses BARRIER_NONE (7) as the default, but CUBIN encodes 0 for "no barrier"
NO_BARRIER_VALUES = frozenset({0, BARRIER_NONE})


def is_no_barrier(barrier_id: int) -> bool:
    """Check if a barrier value means 'no barrier'.

    In NVIDIA's encoding:
    - 0 = no barrier (actual CUBIN encoding)
    - 7 = BARRIER_NONE (GPA's constant/default value)
    - 1-6 = valid barriers B1-B6
    """
    return barrier_id in NO_BARRIER_VALUES


def build_pc_to_inst_map(
    instructions: List["InstructionStat"],
) -> Dict[int, "InstructionStat"]:
    """Build a PC -> InstructionStat map from a list of instructions."""
    return {inst.pc: inst for inst in instructions}


@dataclass
class Control:
    """Control bits extracted from NVIDIA CUBIN instruction encoding.

    These bits control instruction scheduling and synchronization.
    Extracted from bits [41:63] of the 64-bit instruction word.

    Attributes:
        reuse: Register reuse cache hint (1 bit)
        wait: 6-bit barrier wait mask - which barriers to wait for (B1-B6)
        read: 3-bit barrier read ID (7 = BARRIER_NONE)
        write: 3-bit barrier write ID (7 = BARRIER_NONE)
        yield_flag: Warp scheduler yield hint
        stall: 4-bit pipeline stall count (cycles to wait before issue)
    """

    reuse: int = 0
    wait: int = 0  # 6-bit mask for barriers B1-B6
    read: int = BARRIER_NONE  # 3-bit, 7 means no barrier
    write: int = BARRIER_NONE  # 3-bit, 7 means no barrier
    yield_flag: int = 0
    stall: int = 1  # Default minimum stall

    def waits_on_barrier(self, barrier_id: int) -> bool:
        """Check if this instruction waits on a specific barrier (1-6)."""
        if barrier_id < 1 or barrier_id > 6:
            return False
        return bool(self.wait & (1 << (barrier_id - 1)))

    def get_wait_barriers(self) -> List[int]:
        """Get list of barrier IDs this instruction waits on."""
        barriers = []
        for i in range(6):
            if self.wait & (1 << i):
                barriers.append(i + 1)  # Barriers are B1-B6
        return barriers


@dataclass
class IntelSWSB:
    """Intel SWSB (Software Scoreboarding) annotation for Xe HPC.

    Decoded from GED_GetSWSB() using FourDistPipe encoding mode.

    RegDist: in-order pipeline dependency distance (1-7 instructions).
    SBID: out-of-order scoreboard token for long-latency ops (send).
    """

    raw: int = 0
    has_reg_dist: bool = False
    reg_dist_pipe: str = ""  # "generic", "all", "float", "int", "long", "math"
    reg_dist_distance: int = 0  # 1-7
    has_sbid: bool = False
    sbid_type: str = ""  # "set", "dst_wait", "src_wait"
    sbid: int = -1  # 0-31


@dataclass
class InstructionStat:
    """Parsed instruction with register operands and dependency information.

    This is the core data structure for back-slicing analysis. It tracks:
    - What registers an instruction reads (sources) and writes (destinations)
    - Which previous instructions wrote each source register (assign_pcs)
    - Predicate conditions and barrier synchronization

    The assign_pcs maps are critical for back-slicing: they answer
    "for each source register, which instruction(s) could have written it?"

    Attributes:
        op: Normalized opcode (e.g., "LDG", "FADD", "IADD3")
        pc: Program counter (offset within function)
        predicate: Predicate register ID (0-6), -1 if unconditional
        predicate_flag: Whether predicate is true/false/none
        predicate_assign_pcs: PCs that wrote the predicate register
        dsts: Destination general registers (R0-R255)
        srcs: Source general registers (R0-R255)
        pdsts: Destination predicate registers (P0-P6)
        psrcs: Source predicate registers (P0-P6)
        bdsts: Destination barrier registers (B1-B6)
        bsrcs: Source barrier registers (B1-B6)
        udsts: Destination uniform registers (UR0-UR63, SM_75+)
        usrcs: Source uniform registers (UR0-UR63, SM_75+)
        updsts: Destination uniform predicates
        upsrcs: Source uniform predicates
        assign_pcs: Map of general reg -> PCs that wrote it
        passign_pcs: Map of predicate reg -> PCs that wrote it
        bassign_pcs: Map of barrier -> PCs that wrote it
        uassign_pcs: Map of uniform reg -> PCs that wrote it
        upassign_pcs: Map of uniform predicate -> PCs that wrote it
        control: Scheduling control bits
        barrier_threshold: For LDGDEPBAR/LDGSTS instructions
        indirect: True if uses indirect memory addressing
    """

    # Basic identification
    op: str
    pc: int
    function_name: str = ""  # Name of function this instruction belongs to

    # Predicate information
    predicate: int = -1  # -1 means no predicate (unconditional)
    predicate_flag: PredicateFlag = PredicateFlag.PREDICATE_NONE
    predicate_assign_pcs: List[int] = field(default_factory=list)

    # General register operands (R0-R255)
    dsts: List[int] = field(default_factory=list)
    srcs: List[int] = field(default_factory=list)

    # Predicate register operands (P0-P6)
    pdsts: List[int] = field(default_factory=list)
    psrcs: List[int] = field(default_factory=list)

    # Barrier operands (B1-B6)
    bdsts: List[int] = field(default_factory=list)
    bsrcs: List[int] = field(default_factory=list)

    # Uniform registers (SM_75+, UR0-UR63)
    udsts: List[int] = field(default_factory=list)
    usrcs: List[int] = field(default_factory=list)

    # Uniform predicates
    updsts: List[int] = field(default_factory=list)
    upsrcs: List[int] = field(default_factory=list)

    # CRITICAL: Register assignment maps for back-slicing
    # Key: register ID, Value: list of PCs that could have written this register
    assign_pcs: Dict[int, List[int]] = field(default_factory=dict)
    passign_pcs: Dict[int, List[int]] = field(default_factory=dict)
    bassign_pcs: Dict[int, List[int]] = field(default_factory=dict)
    uassign_pcs: Dict[int, List[int]] = field(default_factory=dict)
    upassign_pcs: Dict[int, List[int]] = field(default_factory=dict)

    # Control bits
    control: Control = field(default_factory=Control)

    # Intel SWSB annotation (None for non-Intel)
    swsb: Optional["IntelSWSB"] = None

    # Additional fields
    barrier_threshold: int = -1
    indirect: bool = False
    branch_target: Optional[str] = None  # Label like ".L_x_0" for branches

    # Operand details for enhanced analysis (e.g., s_waitcnt counters)
    operands_raw: Optional[str] = None  # Raw operand string (e.g., "vmcnt(0) lgkmcnt(0)")
    operand_details: Optional[Dict[str, Any]] = None  # Parsed operand details

    def __lt__(self, other: "InstructionStat") -> bool:
        """Support sorting by PC."""
        return self.pc < other.pc

    def __hash__(self) -> int:
        """Hash by PC for use in sets/dicts."""
        return hash(self.pc)

    def __eq__(self, other: object) -> bool:
        """Equality by PC."""
        if not isinstance(other, InstructionStat):
            return NotImplemented
        return self.pc == other.pc

    # Register lookup methods
    def has_src_reg(self, reg: int) -> bool:
        """Check if register is in source list."""
        return reg in self.srcs

    def has_dst_reg(self, reg: int) -> bool:
        """Check if register is in destination list."""
        return reg in self.dsts

    def has_src_pred(self, pred: int) -> bool:
        """Check if predicate is in source list."""
        return pred in self.psrcs

    def has_src_barrier(self, barrier: int) -> bool:
        """Check if barrier is in source list."""
        return barrier in self.bsrcs

    def has_src_ureg(self, ureg: int) -> bool:
        """Check if uniform register is in source list."""
        return ureg in self.usrcs

    # Dependency information
    def get_assign_pcs_for_reg(self, reg: int) -> List[int]:
        """Get PCs that wrote to a general register used by this instruction."""
        return self.assign_pcs.get(reg, [])

    def get_assign_pcs_for_pred(self, pred: int) -> List[int]:
        """Get PCs that wrote to a predicate register used by this instruction."""
        return self.passign_pcs.get(pred, [])

    def get_all_dependencies(self) -> List[int]:
        """Get all PCs this instruction depends on (via any register type)."""
        deps = set()
        for pcs in self.assign_pcs.values():
            deps.update(pcs)
        for pcs in self.passign_pcs.values():
            deps.update(pcs)
        for pcs in self.bassign_pcs.values():
            deps.update(pcs)
        for pcs in self.uassign_pcs.values():
            deps.update(pcs)
        for pcs in self.upassign_pcs.values():
            deps.update(pcs)
        deps.update(self.predicate_assign_pcs)
        return sorted(deps)

    # Instruction classification
    def is_memory_op(self) -> bool:
        """Check if this is a memory operation (NVIDIA, AMD, or Intel)."""
        # NVIDIA patterns
        nvidia_mem_ops = {"LD", "LDG", "LDS", "LDL", "LDC", "ST", "STG", "STS", "STL", "ATOM", "RED", "UTMALDG", "USTMG"}
        base_op = self.op.split(".")[0]
        if base_op in nvidia_mem_ops:
            return True
        # AMD patterns (opcodes are lowercase)
        op_lower = self.op.lower()
        if any(p in op_lower for p in [
            "global_load", "global_store",
            "buffer_load", "buffer_store",
            "flat_load", "flat_store",
            "ds_read", "ds_write",
            "scratch_load", "scratch_store",
            "global_atomic", "flat_atomic", "buffer_atomic",
        ]):
            return True
        # Intel patterns: all memory ops use send/sends/sendc/sendsc
        if base_op.lower() in ("send", "sends", "sendc", "sendsc"):
            return True
        return False

    def is_load(self) -> bool:
        """Check if this is a load operation (NVIDIA, AMD, or Intel)."""
        # NVIDIA patterns
        nvidia_load_ops = {"LD", "LDG", "LDS", "LDL", "LDC"}
        base_op = self.op.split(".")[0]
        if base_op in nvidia_load_ops:
            return True
        # AMD patterns
        op_lower = self.op.lower()
        if any(p in op_lower for p in [
            "global_load", "buffer_load", "flat_load", "ds_read", "scratch_load",
        ]):
            return True
        # Intel: send opcodes are used for both loads and stores;
        # without parsing message descriptors we conservatively treat all sends as loads
        if base_op.lower() in ("send", "sends", "sendc", "sendsc"):
            return True
        return False

    def is_store(self) -> bool:
        """Check if this is a store operation (NVIDIA, AMD, or Intel)."""
        # NVIDIA patterns
        nvidia_store_ops = {"ST", "STG", "STS", "STL"}
        base_op = self.op.split(".")[0]
        if base_op in nvidia_store_ops:
            return True
        # AMD patterns
        op_lower = self.op.lower()
        if any(p in op_lower for p in [
            "global_store", "buffer_store", "flat_store", "ds_write", "scratch_store",
        ]):
            return True
        # Intel: send opcodes are used for both loads and stores;
        # without parsing message descriptors we conservatively treat all sends as stores
        if base_op.lower() in ("send", "sends", "sendc", "sendsc"):
            return True
        return False

    def is_shared_memory_op(self) -> bool:
        """Check if this is a shared/LDS memory operation (NVIDIA or AMD).

        NVIDIA shared memory uses LDS (load) and STS (store) opcodes,
        or opcodes with .S modifier for shared memory.
        AMD shared memory uses ds_read and ds_write opcodes (Data Share = LDS).
        """
        return is_shared_memory_opcode(self.op)

    def is_sync(self) -> bool:
        """Check if this is a synchronization instruction."""
        # NVIDIA sync instructions
        sync_ops = {"BAR", "MEMBAR", "DEPBAR", "WARPSYNC", "BSSY", "BSYNC"}
        base_op = self.op.split(".")[0]
        if base_op in sync_ops:
            return True
        # AMD sync instructions
        op_lower = self.op.lower()
        if op_lower in ("s_barrier", "s_waitcnt", "s_wait_idle"):
            return True
        # Intel sync instruction
        if base_op.lower() == "sync":
            return True
        return False

    def is_branch(self) -> bool:
        """Check if this is a branch instruction."""
        branch_ops = {"BRA", "BRX", "JMP", "JMX", "RET", "EXIT"}
        base_op = self.op.split(".")[0]
        return base_op in branch_ops

    def is_call(self) -> bool:
        """Check if this is a call instruction."""
        return self.op.split(".")[0] == "CALL"

    def is_predicated(self) -> bool:
        """Check if this instruction has a predicate guard."""
        return self.predicate >= 0 and self.predicate_flag != PredicateFlag.PREDICATE_NONE


def parse_register(operand: str) -> Optional[int]:
    """Parse a general register from an operand string.

    Args:
        operand: Operand string like "R0", "R255", "[R2]"

    Returns:
        Register number or None if not a register.
    """
    match = re.search(r"\bR(\d+)\b", operand)
    if match:
        reg = int(match.group(1))
        if 0 <= reg < MAX_GENERAL_REGS:
            return reg
    return None


def parse_registers(operand: str, width: int = 32) -> List[int]:
    """Parse general registers from an operand, handling multi-register ops.

    For 64-bit operations, returns consecutive register pair (R0, R1).
    For 128-bit operations, returns 4 consecutive registers.

    Args:
        operand: Operand string
        width: Operation width in bits (32, 64, or 128)

    Returns:
        List of register numbers.
    """
    regs = []
    for match in re.finditer(r"\bR(\d+)\b", operand):
        reg = int(match.group(1))
        if 0 <= reg < MAX_GENERAL_REGS:
            if width == 64:
                regs.extend([reg, reg + 1])
            elif width == 128:
                regs.extend([reg, reg + 1, reg + 2, reg + 3])
            else:
                regs.append(reg)
    return regs


def parse_predicate(operand: str) -> Optional[int]:
    """Parse a predicate register from an operand string.

    Args:
        operand: Operand string like "P0", "P6", "@P1"

    Returns:
        Predicate register number (0-6) or None.
    """
    match = re.search(r"\bP(\d)\b", operand)
    if match:
        pred = int(match.group(1))
        if 0 <= pred < MAX_PREDICATE_REGS:
            return pred
    return None


def parse_barrier(operand: str) -> Optional[int]:
    """Parse a barrier register from an operand string.

    Args:
        operand: Operand string like "SB1", "B5", "B0"

    Returns:
        Barrier register number (0-6) or None.
    """
    # Skip labels like `(.L_x_1) which might contain B0, B1, etc. as substrings
    if operand.startswith("`") or operand.startswith(".L"):
        return None

    # Match standalone barrier registers at start of operand
    match = re.match(r"^(?:SB|B)(\d)$", operand.strip())
    if match:
        barrier = int(match.group(1))
        if 0 <= barrier <= MAX_BARRIER_REGS:
            return barrier
    return None


def parse_uniform_register(operand: str) -> Optional[int]:
    """Parse a uniform register from an operand string.

    Args:
        operand: Operand string like "UR0", "UR63"

    Returns:
        Uniform register number (0-63) or None.
    """
    match = re.search(r"\bUR(\d+)\b", operand)
    if match:
        ureg = int(match.group(1))
        if 0 <= ureg < MAX_UNIFORM_REGS:
            return ureg
    return None


def parse_instruction_predicate(inst_str: str) -> tuple[int, PredicateFlag]:
    """Parse predicate guard from instruction string.

    Args:
        inst_str: Full instruction string like "@P1 FADD R0, R1, R2"

    Returns:
        Tuple of (predicate_reg, predicate_flag).
        Returns (-1, PREDICATE_NONE) if no predicate.
    """
    # Match @P1 or @!P1 at start of instruction
    match = re.match(r"@(!)?P(\d)\s", inst_str.strip())
    if match:
        negated = match.group(1) is not None
        pred_reg = int(match.group(2))
        flag = PredicateFlag.PREDICATE_FALSE if negated else PredicateFlag.PREDICATE_TRUE
        return (pred_reg, flag)
    return (-1, PredicateFlag.PREDICATE_NONE)


def detect_indirect_addressing(operand: str) -> bool:
    """Check if operand uses indirect memory addressing.

    Args:
        operand: Operand string like "[R2]", "[R2 + 0x10]"

    Returns:
        True if indirect addressing is used.
    """
    return bool(re.search(r"\[R\d+", operand))


def get_operation_width(opcode: str) -> int:
    """Get operation width from opcode modifiers.

    Args:
        opcode: Opcode string like "LDG.64", "FADD.32"

    Returns:
        Width in bits (32, 64, or 128).
    """
    if ".128" in opcode:
        return 128
    elif ".64" in opcode:
        return 64
    return 32


def is_constant_memory_opcode(opcode: str) -> bool:
    """Check if opcode is a constant memory operation (NVIDIA).

    NVIDIA constant memory uses:
    - LDC (load constant) opcodes
    - CONSTANT in the opcode name

    Args:
        opcode: The instruction opcode string.

    Returns:
        True if this is a constant memory operation.
    """
    op_upper = opcode.upper()
    # Only match actual constant load opcodes (LDC, ULDC), not cache hints
    # like LDG.E.64.CONSTANT which is a global load through the constant cache.
    return "LDC" in op_upper


def is_local_memory_opcode(opcode: str) -> bool:
    """Check if opcode is a local memory operation (NVIDIA).

    NVIDIA local memory uses:
    - LDL (load local), STL (store local) opcodes
    - LOCAL in the opcode name

    Note: This is NVIDIA local memory (per-thread), not AMD LDS (shared).

    Args:
        opcode: The instruction opcode string.

    Returns:
        True if this is a local memory operation.
    """
    op_upper = opcode.upper()
    return "LOCAL" in op_upper or "LDL" in op_upper or "STL" in op_upper


def is_shared_memory_opcode(opcode: str) -> bool:
    """Check if opcode is a shared/LDS memory operation (NVIDIA or AMD).

    NVIDIA shared memory uses:
    - LDS (load shared), STS (store shared) opcodes
    - Opcodes with .S modifier (e.g., LD.S, ST.S)
    - SHARED in the opcode name

    AMD shared memory uses:
    - ds_read, ds_write opcodes (Data Share = LDS)
    - Any opcode starting with ds_ prefix

    Args:
        opcode: The instruction opcode string.

    Returns:
        True if this is a shared memory operation.
    """
    # NVIDIA patterns (uppercase)
    op_upper = opcode.upper()
    if any(x in op_upper for x in ["SHARED", "LDS", "STS"]):
        return True
    # Check for .S modifier (shared memory access)
    if ".S" in op_upper:
        return True

    # AMD patterns (lowercase)
    op_lower = opcode.lower()
    if op_lower.startswith("ds_"):
        return True

    return False
