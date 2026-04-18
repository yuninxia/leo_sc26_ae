"""Control flow graph structures for CUDA binary analysis.

Based on GPA's DotCFG.hpp - defines structures for representing
control flow graphs parsed from nvdisasm output.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from leo.binary.instruction import InstructionStat


class EdgeType(IntEnum):
    """Control flow edge types."""

    DIRECT = 0  # Unconditional fall-through
    COND_TAKEN = 1  # Conditional branch taken
    COND_NOT_TAKEN = 2  # Conditional branch not taken
    CALL = 3  # Function call
    CALL_FT = 4  # Function call fall-through (return point)
    RETURN = 5  # Return from function


@dataclass
class Target:
    """Represents a control flow edge from an instruction to a block.

    Attributes:
        from_pc: PC of the instruction that creates this edge.
        to_block_id: ID of the target block.
        edge_type: Type of control flow edge.
    """

    from_pc: int
    to_block_id: int
    edge_type: EdgeType = EdgeType.DIRECT

    def __lt__(self, other: "Target") -> bool:
        """Sort by source PC."""
        return self.from_pc < other.from_pc


@dataclass
class Block:
    """Basic block - straight-line sequence of instructions.

    A basic block has one entry point (first instruction) and one or more
    exit points (last instruction may branch to multiple targets).

    Attributes:
        id: Unique block identifier.
        name: Block label (e.g., ".L_x_1").
        instructions: List of instructions in program order.
        targets: Control flow edges to successor blocks.
        begin_offset: Start offset (may skip control code).
    """

    id: int
    name: str = ""
    instructions: List[InstructionStat] = field(default_factory=list)
    targets: List[Target] = field(default_factory=list)
    begin_offset: int = 0

    def __lt__(self, other: "Block") -> bool:
        """Sort blocks by first instruction PC."""
        if not self.instructions:
            return True
        if not other.instructions:
            return False
        return self.instructions[0].pc < other.instructions[0].pc

    def __hash__(self) -> int:
        """Hash by block ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by block ID."""
        if not isinstance(other, Block):
            return NotImplemented
        return self.id == other.id

    @property
    def start_pc(self) -> Optional[int]:
        """First instruction PC, or None if empty."""
        return self.instructions[0].pc if self.instructions else None

    @property
    def end_pc(self) -> Optional[int]:
        """Last instruction PC, or None if empty."""
        return self.instructions[-1].pc if self.instructions else None

    @property
    def size(self) -> int:
        """Number of instructions in block."""
        return len(self.instructions)

    def get_instruction_at(self, pc: int) -> Optional[InstructionStat]:
        """Find instruction by PC."""
        for inst in self.instructions:
            if inst.pc == pc:
                return inst
        return None

    def contains_pc(self, pc: int) -> bool:
        """Check if PC is within this block."""
        if not self.instructions:
            return False
        return self.start_pc <= pc <= self.end_pc

    def get_successor_block_ids(self) -> List[int]:
        """Get IDs of all successor blocks."""
        return [t.to_block_id for t in self.targets]

    def is_exit_block(self) -> bool:
        """Check if this block has no successors (function exit)."""
        return len(self.targets) == 0

    def has_branch(self) -> bool:
        """Check if last instruction is a branch."""
        if not self.instructions:
            return False
        return self.instructions[-1].is_branch()

    def has_call(self) -> bool:
        """Check if last instruction is a call."""
        if not self.instructions:
            return False
        return self.instructions[-1].is_call()


@dataclass
class Function:
    """CUDA kernel or device function.

    Attributes:
        name: Function name (mangled).
        blocks: List of basic blocks.
        entry_block_id: ID of the entry block.
    """

    name: str
    blocks: List[Block] = field(default_factory=list)
    entry_block_id: int = 0

    @property
    def entry_block(self) -> Optional[Block]:
        """Get entry block."""
        for block in self.blocks:
            if block.id == self.entry_block_id:
                return block
        return self.blocks[0] if self.blocks else None

    @property
    def num_instructions(self) -> int:
        """Total number of instructions across all blocks."""
        return sum(len(b.instructions) for b in self.blocks)

    def get_block_by_id(self, block_id: int) -> Optional[Block]:
        """Find block by ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None

    def get_block_by_name(self, name: str) -> Optional[Block]:
        """Find block by label name."""
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def get_block_containing_pc(self, pc: int) -> Optional[Block]:
        """Find block containing a given PC."""
        for block in self.blocks:
            if block.contains_pc(pc):
                return block
        return None

    def get_instruction_at(self, pc: int) -> Optional[InstructionStat]:
        """Find instruction by PC across all blocks."""
        for block in self.blocks:
            inst = block.get_instruction_at(pc)
            if inst:
                return inst
        return None

    def get_all_instructions(self) -> List[InstructionStat]:
        """Get all instructions in PC order."""
        all_insts = []
        for block in self.blocks:
            all_insts.extend(block.instructions)
        return sorted(all_insts, key=lambda i: i.pc)


class CFG:
    """Control flow graph for a function.

    Provides navigation methods for traversing the CFG and
    analyzing control flow relationships.
    """

    def __init__(self, function: Function):
        """Initialize CFG from a function.

        Args:
            function: Function containing blocks and edges.
        """
        self.function = function
        self._blocks_by_id: Dict[int, Block] = {b.id: b for b in function.blocks}
        self._predecessors: Dict[int, List[int]] = {}
        self._build_predecessor_map()

    def _build_predecessor_map(self) -> None:
        """Build mapping from block ID to predecessor block IDs."""
        for block in self.function.blocks:
            for target in block.targets:
                if target.to_block_id not in self._predecessors:
                    self._predecessors[target.to_block_id] = []
                if block.id not in self._predecessors[target.to_block_id]:
                    self._predecessors[target.to_block_id].append(block.id)

    def get_block(self, block_id: int) -> Optional[Block]:
        """Get block by ID."""
        return self._blocks_by_id.get(block_id)

    def predecessors(self, block: Block) -> List[Block]:
        """Get all predecessor blocks (blocks that branch to this one)."""
        pred_ids = self._predecessors.get(block.id, [])
        return [self._blocks_by_id[pid] for pid in pred_ids if pid in self._blocks_by_id]

    def successors(self, block: Block) -> List[Block]:
        """Get all successor blocks (blocks reachable from this one)."""
        succ_ids = block.get_successor_block_ids()
        return [self._blocks_by_id[sid] for sid in succ_ids if sid in self._blocks_by_id]

    def get_entry_block(self) -> Optional[Block]:
        """Get the function entry block."""
        return self.function.entry_block

    def get_exit_blocks(self) -> List[Block]:
        """Get all blocks with no successors (function exits)."""
        return [b for b in self.function.blocks if b.is_exit_block()]

    def is_reachable(self, from_block: Block, to_block: Block) -> bool:
        """Check if to_block is reachable from from_block."""
        visited: Set[int] = set()

        def dfs(block_id: int) -> bool:
            if block_id == to_block.id:
                return True
            if block_id in visited:
                return False
            visited.add(block_id)

            block = self._blocks_by_id.get(block_id)
            if not block:
                return False

            for succ_id in block.get_successor_block_ids():
                if dfs(succ_id):
                    return True
            return False

        return dfs(from_block.id)

    def is_loop_header(self, block: Block) -> bool:
        """Check if block is a loop header (has back edge from a dominated block).

        A back edge is an edge from a block that is dominated by its target.
        A loop header receives at least one back edge.
        """
        for pred in self.predecessors(block):
            # If this block dominates a predecessor, there's a back edge
            if self.dominates(block, pred):
                return True
        return False

    def get_loop_blocks(self, header: Block) -> Set[Block]:
        """Get all blocks in a loop given its header.

        Uses reverse DFS from back edge sources to find loop body.
        A back edge is an edge from a dominated block back to the header.
        """
        loop_blocks: Set[Block] = {header}

        # Find predecessors that are dominated by header (back edge sources)
        for pred in self.predecessors(header):
            if self.dominates(header, pred):
                # This pred has a back edge to header, collect loop body
                self._collect_loop_blocks(pred, header, loop_blocks)

        return loop_blocks

    def _collect_loop_blocks(
        self, block: Block, header: Block, loop_blocks: Set[Block]
    ) -> None:
        """Collect blocks in loop via reverse traversal."""
        if block in loop_blocks:
            return
        loop_blocks.add(block)

        for pred in self.predecessors(block):
            if pred != header:
                self._collect_loop_blocks(pred, header, loop_blocks)

    def get_dominator_tree(self) -> Dict[int, Optional[int]]:
        """Compute immediate dominators for all blocks.

        Returns:
            Dict mapping block_id -> immediate_dominator_id (None for entry).
        """
        entry = self.get_entry_block()
        if not entry:
            return {}

        # Initialize dominators
        dom: Dict[int, Set[int]] = {}
        all_block_ids = set(self._blocks_by_id.keys())

        for block_id in all_block_ids:
            if block_id == entry.id:
                dom[block_id] = {entry.id}
            else:
                dom[block_id] = all_block_ids.copy()

        # Iterate until fixpoint
        changed = True
        while changed:
            changed = False
            for block_id in all_block_ids:
                if block_id == entry.id:
                    continue

                pred_ids = self._predecessors.get(block_id, [])
                if not pred_ids:
                    continue

                # New dominators = intersection of predecessor dominators + self
                new_dom = all_block_ids.copy()
                for pred_id in pred_ids:
                    new_dom &= dom.get(pred_id, all_block_ids)
                new_dom.add(block_id)

                if new_dom != dom[block_id]:
                    dom[block_id] = new_dom
                    changed = True

        # Convert to immediate dominators
        idom: Dict[int, Optional[int]] = {entry.id: None}
        for block_id in all_block_ids:
            if block_id == entry.id:
                continue

            # Immediate dominator is the dominator closest to block
            # (largest dominator set size excluding self)
            block_doms = dom[block_id] - {block_id}
            if not block_doms:
                idom[block_id] = None
                continue

            # Find dominator with largest dom set (closest to block)
            best_idom = None
            best_size = -1
            for d in block_doms:
                d_size = len(dom.get(d, set()))
                if d_size > best_size:
                    best_size = d_size
                    best_idom = d

            idom[block_id] = best_idom

        return idom

    def dominates(self, dom_block: Block, block: Block) -> bool:
        """Check if dom_block dominates block."""
        idom = self.get_dominator_tree()
        current = block.id

        while current is not None:
            if current == dom_block.id:
                return True
            current = idom.get(current)

        return False

    def get_block_containing_pc(self, pc: int) -> Optional[Block]:
        """Find block containing a given PC."""
        return self.function.get_block_containing_pc(pc)

    def get_instruction_at(self, pc: int) -> Optional[InstructionStat]:
        """Find instruction by PC across all blocks."""
        return self.function.get_instruction_at(pc)

    def get_all_instructions(self) -> List[InstructionStat]:
        """Get all instructions in PC order."""
        return self.function.get_all_instructions()


def compute_liveness(
    cfg: CFG,
) -> Tuple[Dict[int, Set], Dict[int, Set]]:
    """Compute per-block register liveness via backward dataflow analysis.

    Standard fixpoint iteration:
        GEN[B]  = registers used before being defined in B
        KILL[B] = registers defined in B
        live_in[B]  = GEN[B] | (live_out[B] - KILL[B])
        live_out[B] = union of live_in[succ] for all successors of B

    Register identifiers are tagged tuples to distinguish spaces:
        ("r", id) for general registers R0-R255
        ("p", id) for predicate registers P0-P6
        ("b", id) for barrier registers B0-B6
        ("u", id) for uniform registers UR0-UR63

    Args:
        cfg: Control flow graph with instructions in blocks.

    Returns:
        Tuple of (live_in, live_out) dicts mapping block_id -> set of
        (tag, reg_id) tuples.
    """
    blocks = cfg.function.blocks
    if not blocks:
        return {}, {}

    # Step 1: Compute GEN and KILL sets per block
    gen: Dict[int, Set] = {}
    kill: Dict[int, Set] = {}

    for block in blocks:
        block_gen: Set = set()
        block_kill: Set = set()

        for inst in block.instructions:
            # Uses before defs -> GEN (only if not already killed in this block)
            for src in inst.srcs:
                r = ("r", src)
                if r not in block_kill:
                    block_gen.add(r)
            for psrc in inst.psrcs:
                r = ("p", psrc)
                if r not in block_kill:
                    block_gen.add(r)
            for bsrc in inst.bsrcs:
                r = ("b", bsrc)
                if r not in block_kill:
                    block_gen.add(r)
            for usrc in inst.usrcs:
                r = ("u", usrc)
                if r not in block_kill:
                    block_gen.add(r)
            if inst.predicate >= 0:
                r = ("p", inst.predicate)
                if r not in block_kill:
                    block_gen.add(r)

            # Defs -> KILL
            for dst in inst.dsts:
                block_kill.add(("r", dst))
            for pdst in inst.pdsts:
                block_kill.add(("p", pdst))
            for bdst in inst.bdsts:
                block_kill.add(("b", bdst))
            for udst in inst.udsts:
                block_kill.add(("u", udst))

        gen[block.id] = block_gen
        kill[block.id] = block_kill

    # Step 2: Backward dataflow fixpoint iteration
    live_in: Dict[int, Set] = {b.id: set() for b in blocks}
    live_out: Dict[int, Set] = {b.id: set() for b in blocks}

    changed = True
    while changed:
        changed = False
        for block in blocks:
            # live_out[B] = union of live_in[succ] for all successors
            new_out: Set = set()
            for succ in cfg.successors(block):
                new_out |= live_in[succ.id]

            # live_in[B] = GEN[B] | (live_out[B] - KILL[B])
            new_in = gen[block.id] | (new_out - kill[block.id])

            if new_in != live_in[block.id] or new_out != live_out[block.id]:
                live_in[block.id] = new_in
                live_out[block.id] = new_out
                changed = True

    return live_in, live_out


def compute_reaching_defs(
    cfg: CFG,
) -> Tuple[Dict[int, Set], Dict[int, Set]]:
    """Compute per-block reaching definitions via forward dataflow analysis.

    Standard fixpoint iteration:
        GEN[B]  = definitions generated in B (3-tuples: tag, reg_id, def_pc)
        KILL[B] = registers defined in B (2-tuples: tag, reg_id)
        reach_out[B] = GEN[B] | (reach_in[B] - KILL[B])
        reach_in[B]  = union of reach_out[pred] for all predecessors of B

    A reaching def (tag, reg_id, def_pc) is "killed" if (tag, reg_id) is in
    KILL[B], meaning the register is redefined in B and the old definition
    no longer reaches the block exit.

    Args:
        cfg: Control flow graph with instructions in blocks.

    Returns:
        Tuple of (reach_in, reach_out) dicts mapping block_id -> set of
        (tag, reg_id, def_pc) 3-tuples.
    """
    blocks = cfg.function.blocks
    if not blocks:
        return {}, {}

    # Step 1: Compute GEN and KILL sets per block
    gen: Dict[int, Set] = {}
    kill: Dict[int, Set] = {}

    for block in blocks:
        block_gen: Set = set()
        block_kill: Set = set()

        for inst in block.instructions:
            # Defs -> GEN (definition generated at this PC)
            # Also add to KILL (kills any prior reaching def of same register)
            for dst in inst.dsts:
                # Kill earlier defs of same register within this block
                block_gen = {d for d in block_gen if not (d[0] == "r" and d[1] == dst)}
                block_gen.add(("r", dst, inst.pc))
                block_kill.add(("r", dst))
            for pdst in inst.pdsts:
                block_gen = {d for d in block_gen if not (d[0] == "p" and d[1] == pdst)}
                block_gen.add(("p", pdst, inst.pc))
                block_kill.add(("p", pdst))
            for bdst in inst.bdsts:
                block_gen = {d for d in block_gen if not (d[0] == "b" and d[1] == bdst)}
                block_gen.add(("b", bdst, inst.pc))
                block_kill.add(("b", bdst))
            for udst in inst.udsts:
                block_gen = {d for d in block_gen if not (d[0] == "u" and d[1] == udst)}
                block_gen.add(("u", udst, inst.pc))
                block_kill.add(("u", udst))

        gen[block.id] = block_gen
        kill[block.id] = block_kill

    # Step 2: Forward dataflow fixpoint iteration
    reach_in: Dict[int, Set] = {b.id: set() for b in blocks}
    reach_out: Dict[int, Set] = {b.id: set() for b in blocks}

    changed = True
    while changed:
        changed = False
        for block in blocks:
            # reach_in[B] = union of reach_out[pred] for all predecessors
            new_in: Set = set()
            for pred in cfg.predecessors(block):
                new_in |= reach_out[pred.id]

            # reach_out[B] = GEN[B] | (reach_in[B] - KILL[B])
            # Filter: remove reaching defs whose register is killed in this block
            surviving = {d for d in new_in if (d[0], d[1]) not in kill[block.id]}
            new_out = gen[block.id] | surviving

            if new_in != reach_in[block.id] or new_out != reach_out[block.id]:
                reach_in[block.id] = new_in
                reach_out[block.id] = new_out
                changed = True

    return reach_in, reach_out


def build_cfg_from_instructions(
    instructions: List[InstructionStat],
    function_name: str = "unknown",
    label_to_pc: Optional[Dict[str, int]] = None,
) -> CFG:
    """Build a CFG from a list of instructions.

    Creates blocks at branch points and branch targets, with proper edge types
    for conditional/unconditional branches and calls.

    Args:
        instructions: List of instructions in PC order.
        function_name: Name for the function.
        label_to_pc: Mapping of label names to PCs (from ParsedFunction.labels).

    Returns:
        CFG object.
    """
    if not instructions:
        func = Function(name=function_name)
        return CFG(func)

    if label_to_pc is None:
        label_to_pc = {}

    # Sort instructions by PC
    instructions = sorted(instructions, key=lambda i: i.pc)

    # Build PC to instruction index mapping
    pc_to_inst_idx: Dict[int, int] = {inst.pc: i for i, inst in enumerate(instructions)}

    # Find block boundaries (after branches and at branch targets)
    block_starts: Set[int] = {instructions[0].pc}  # First instruction starts a block

    # Find all branch targets and instructions after branches
    for i, inst in enumerate(instructions):
        if inst.is_branch() or inst.is_call():
            # Instruction after branch/call starts a new block
            if i + 1 < len(instructions):
                block_starts.add(instructions[i + 1].pc)

            # Add branch target PC as block start
            if inst.branch_target and inst.branch_target in label_to_pc:
                target_pc = label_to_pc[inst.branch_target]
                if target_pc in pc_to_inst_idx:
                    block_starts.add(target_pc)

    # Create blocks
    block_starts_sorted = sorted(block_starts)
    blocks: List[Block] = []
    pc_to_block_id: Dict[int, int] = {}  # Map start PC to block ID

    for i, start_pc in enumerate(block_starts_sorted):
        # Find end of this block
        if i + 1 < len(block_starts_sorted):
            end_pc = block_starts_sorted[i + 1]
        else:
            end_pc = instructions[-1].pc + 1  # Past last instruction

        # Collect instructions in this block
        block_insts = [inst for inst in instructions if start_pc <= inst.pc < end_pc]

        block = Block(
            id=i,
            name=f".L_{i}",
            instructions=block_insts,
            begin_offset=start_pc,
        )
        blocks.append(block)
        pc_to_block_id[start_pc] = i

    # Add edges between blocks
    for block in blocks:
        if not block.instructions:
            continue

        last_inst = block.instructions[-1]
        base_op = last_inst.op.split(".")[0]

        # Determine if this is a branch/call that has a target
        has_explicit_target = (
            last_inst.branch_target is not None
            and last_inst.branch_target in label_to_pc
        )

        # Add branch target edge (if applicable)
        if has_explicit_target:
            target_pc = label_to_pc[last_inst.branch_target]
            if target_pc in pc_to_block_id:
                target_block_id = pc_to_block_id[target_pc]

                # Determine edge type for branch target
                if last_inst.is_call():
                    edge_type = EdgeType.CALL
                elif last_inst.is_predicated():
                    edge_type = EdgeType.COND_TAKEN
                else:
                    edge_type = EdgeType.DIRECT

                block.targets.append(
                    Target(
                        from_pc=last_inst.pc,
                        to_block_id=target_block_id,
                        edge_type=edge_type,
                    )
                )

        # Determine if we need a fall-through edge
        # Fall-through happens for:
        # - Non-branch instructions
        # - CALL instructions (CALL_FT - return point)
        # - Predicated branches (when predicate is false)
        # No fall-through for:
        # - Unconditional branches (BRA, JMP without predicate)
        # - EXIT, RET instructions

        needs_fallthrough = False
        fallthrough_edge_type = EdgeType.DIRECT

        if base_op in {"EXIT", "RET"}:
            # These never fall through
            needs_fallthrough = False
        elif last_inst.is_call():
            # Calls always have a return point (fall-through)
            needs_fallthrough = True
            fallthrough_edge_type = EdgeType.CALL_FT
        elif last_inst.is_branch():
            if last_inst.is_predicated():
                # Predicated branch may not be taken
                needs_fallthrough = True
                fallthrough_edge_type = EdgeType.COND_NOT_TAKEN
            else:
                # Unconditional branch doesn't fall through
                needs_fallthrough = False
        else:
            # Regular instruction falls through
            needs_fallthrough = True
            fallthrough_edge_type = EdgeType.DIRECT

        # Add fall-through edge to next block
        if needs_fallthrough:
            # Find next block by PC
            next_block_idx = block.id + 1
            if next_block_idx < len(blocks):
                next_block = blocks[next_block_idx]
                block.targets.append(
                    Target(
                        from_pc=last_inst.pc,
                        to_block_id=next_block.id,
                        edge_type=fallthrough_edge_type,
                    )
                )

    # Create function and CFG
    func = Function(name=function_name, blocks=blocks, entry_block_id=0)
    return CFG(func)
