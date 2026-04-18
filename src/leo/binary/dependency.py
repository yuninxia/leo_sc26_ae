"""Register dependency analysis for GPU binary back-slicing.

This module implements a two-phase dependency analysis:

Phase 1 (build_assign_pcs): A simple forward-pass linear scan that tracks
the most recent writer per register. This is a simplification compared to
GPA, which uses Dyninst's backward slicing to find ALL reaching definitions.
LEO's linear scan may miss reaching definitions at merge points where
different branches write the same register, but this is rare in GPU code
because compilers heavily use predication instead of branching.

Phase 2 (prune_dead_dependencies): A CFG-aware liveness analysis pass that
removes false cross-block dependencies where the definition does not reach
the consumer. This is an improvement over GPA, which has no liveness pruning.

The two phases together provide a practical approximation of reaching
definitions that works across NVIDIA, AMD, and Intel GPU binaries without
requiring Dyninst (which only supports NVIDIA CUDA).
"""

from typing import TYPE_CHECKING, Dict, List, Set

from leo.binary.instruction import InstructionStat

if TYPE_CHECKING:
    from leo.binary.cfg import CFG


def build_assign_pcs(instructions: List[InstructionStat], cfg: "CFG | None" = None) -> None:
    """Build assign_pcs maps: register dependency from source to defining PC.

    When a CFG is provided, uses reaching definitions (forward dataflow
    fixpoint) to find ALL possible writers per register at each use point.
    This correctly handles merge points where different branches define
    the same register.

    Without a CFG, falls back to a simple linear forward scan that tracks
    only the most recent writer per register (may miss reaching definitions
    at merge points, but works when no CFG is available).

    Args:
        instructions: List of InstructionStat. Modified in-place to populate
                     assign_pcs, passign_pcs, bassign_pcs, uassign_pcs,
                     and predicate_assign_pcs maps.
        cfg: Optional control flow graph. When provided, enables CFG-aware
             reaching definitions analysis.
    """
    if not instructions:
        return

    if cfg is not None:
        _build_assign_pcs_reaching_defs(instructions, cfg)
    else:
        _build_assign_pcs_linear(instructions)


def _build_assign_pcs_reaching_defs(instructions: List[InstructionStat], cfg: "CFG") -> None:
    """Build assign_pcs using CFG-aware reaching definitions analysis."""
    from leo.binary.cfg import compute_reaching_defs

    reach_in, _ = compute_reaching_defs(cfg)

    # Build PC -> block_id map
    pc_to_block_id: Dict[int, int] = {}
    for block in cfg.function.blocks:
        for inst in block.instructions:
            pc_to_block_id[inst.pc] = block.id

    # For reaching defs within a block, we also need intra-block precision.
    # reach_in gives defs reaching the block ENTRY. For instructions in the
    # middle of a block, we need to account for earlier defs in the same block.
    # Strategy: walk each block forward, maintaining a per-instruction
    # reaching set that starts from reach_in and updates with each def.

    for block in cfg.function.blocks:
        # Start with defs reaching block entry
        current_reach: Set = set(reach_in.get(block.id, set()))

        for inst in block.instructions:
            # Initialize maps
            inst.assign_pcs = {}
            inst.passign_pcs = {}
            inst.bassign_pcs = {}
            inst.uassign_pcs = {}
            inst.upassign_pcs = {}
            inst.predicate_assign_pcs = []

            # Link source registers to ALL reaching definitions
            for src in inst.srcs:
                defs = [d[2] for d in current_reach if d[0] == "r" and d[1] == src]
                if defs:
                    inst.assign_pcs[src] = defs

            for psrc in inst.psrcs:
                defs = [d[2] for d in current_reach if d[0] == "p" and d[1] == psrc]
                if defs:
                    inst.passign_pcs[psrc] = defs

            for bsrc in inst.bsrcs:
                defs = [d[2] for d in current_reach if d[0] == "b" and d[1] == bsrc]
                if defs:
                    inst.bassign_pcs[bsrc] = defs

            for usrc in inst.usrcs:
                defs = [d[2] for d in current_reach if d[0] == "u" and d[1] == usrc]
                if defs:
                    inst.uassign_pcs[usrc] = defs

            # Predicate guard
            if inst.predicate >= 0:
                defs = [d[2] for d in current_reach if d[0] == "p" and d[1] == inst.predicate]
                if defs:
                    inst.predicate_assign_pcs = defs

            # Update current_reach: kill old defs, add new def
            for dst in inst.dsts:
                current_reach = {d for d in current_reach if not (d[0] == "r" and d[1] == dst)}
                current_reach.add(("r", dst, inst.pc))
            for pdst in inst.pdsts:
                current_reach = {d for d in current_reach if not (d[0] == "p" and d[1] == pdst)}
                current_reach.add(("p", pdst, inst.pc))
            for bdst in inst.bdsts:
                current_reach = {d for d in current_reach if not (d[0] == "b" and d[1] == bdst)}
                current_reach.add(("b", bdst, inst.pc))
            for udst in inst.udsts:
                current_reach = {d for d in current_reach if not (d[0] == "u" and d[1] == udst)}
                current_reach.add(("u", udst, inst.pc))


def _build_assign_pcs_linear(instructions: List[InstructionStat]) -> None:
    """Build assign_pcs using simple linear forward scan (fallback).

    Tracks only the most recent writer per register. May miss reaching
    definitions at merge points where different branches define the same
    register. Use when no CFG is available.
    """
    instructions = sorted(instructions, key=lambda i: i.pc)

    reg_last: Dict[int, int] = {}      # R0-R255 -> PC
    pred_last: Dict[int, int] = {}     # P0-P6 -> PC
    barrier_last: Dict[int, int] = {}  # B0-B6 -> PC
    ureg_last: Dict[int, int] = {}     # UR0-UR63 -> PC

    for inst in instructions:
        pc = inst.pc

        inst.assign_pcs = {}
        inst.passign_pcs = {}
        inst.bassign_pcs = {}
        inst.uassign_pcs = {}
        inst.upassign_pcs = {}
        inst.predicate_assign_pcs = []

        for src in inst.srcs:
            if src in reg_last:
                inst.assign_pcs[src] = [reg_last[src]]

        for psrc in inst.psrcs:
            if psrc in pred_last:
                inst.passign_pcs[psrc] = [pred_last[psrc]]

        for bsrc in inst.bsrcs:
            if bsrc in barrier_last:
                inst.bassign_pcs[bsrc] = [barrier_last[bsrc]]

        for usrc in inst.usrcs:
            if usrc in ureg_last:
                inst.uassign_pcs[usrc] = [ureg_last[usrc]]

        if inst.predicate >= 0:
            if inst.predicate in pred_last:
                inst.predicate_assign_pcs = [pred_last[inst.predicate]]

        for dst in inst.dsts:
            reg_last[dst] = pc
        for pdst in inst.pdsts:
            pred_last[pdst] = pc
        for bdst in inst.bdsts:
            barrier_last[bdst] = pc
        for udst in inst.udsts:
            ureg_last[udst] = pc


def prune_dead_dependencies(
    instructions: List[InstructionStat],
    cfg: "CFG",
) -> int:
    """Remove false register dependencies using liveness analysis.

    For each cross-block dependency (def_pc -> use_pc on register R),
    removes the dependency if R is not live on exit from the definition's
    block — meaning the definition cannot reach the use on any path.

    Same-block dependencies are always kept (the forward scan correctly
    tracks the most recent writer within a block).

    Args:
        instructions: Instructions with populated assign_pcs maps.
        cfg: Control flow graph for the function.

    Returns:
        Number of dependency edges removed.
    """
    from leo.binary.cfg import compute_liveness

    _, live_out = compute_liveness(cfg)

    # Build PC -> block_id map for fast lookup
    pc_to_block_id: Dict[int, int] = {}
    for block in cfg.function.blocks:
        for inst in block.instructions:
            pc_to_block_id[inst.pc] = block.id

    removed = 0

    for inst in instructions:
        use_bid = pc_to_block_id.get(inst.pc)
        if use_bid is None:
            continue

        # Filter one assign_pcs map for a given register space tag
        def _filter_map(amap: Dict[int, List[int]], tag: str) -> int:
            n = 0
            for reg, def_pcs in list(amap.items()):
                kept = []
                for dpc in def_pcs:
                    dbid = pc_to_block_id.get(dpc)
                    if dbid is None or dbid == use_bid:
                        kept.append(dpc)  # same block or unknown -> keep
                    elif (tag, reg) in live_out.get(dbid, set()):
                        kept.append(dpc)  # live on exit -> keep
                    else:
                        n += 1  # dead -> remove
                amap[reg] = kept
            return n

        removed += _filter_map(inst.assign_pcs, "r")
        removed += _filter_map(inst.passign_pcs, "p")
        removed += _filter_map(inst.bassign_pcs, "b")
        removed += _filter_map(inst.uassign_pcs, "u")

        # Predicate guard
        if inst.predicate >= 0 and inst.predicate_assign_pcs:
            kept = []
            for dpc in inst.predicate_assign_pcs:
                dbid = pc_to_block_id.get(dpc)
                if dbid is None or dbid == use_bid:
                    kept.append(dpc)
                elif ("p", inst.predicate) in live_out.get(dbid, set()):
                    kept.append(dpc)
                else:
                    removed += 1
            inst.predicate_assign_pcs = kept

    return removed


# =============================================================================
# Query Functions
# =============================================================================


def get_all_dependencies(inst: InstructionStat) -> Set[int]:
    """Get all PCs this instruction depends on via any register type.

    Args:
        inst: Instruction with populated assign_pcs maps.

    Returns:
        Set of instruction PCs that this instruction depends on.
    """
    deps: Set[int] = set()

    # General registers
    for pcs in inst.assign_pcs.values():
        deps.update(pcs)

    # Predicate registers
    for pcs in inst.passign_pcs.values():
        deps.update(pcs)

    # Barrier registers
    for pcs in inst.bassign_pcs.values():
        deps.update(pcs)

    # Uniform registers
    for pcs in inst.uassign_pcs.values():
        deps.update(pcs)

    # Uniform predicates
    for pcs in inst.upassign_pcs.values():
        deps.update(pcs)

    # Predicate guard
    deps.update(inst.predicate_assign_pcs)

    return deps


def backward_slice(
    target_pc: int,
    pc_to_inst: Dict[int, InstructionStat],
    max_depth: int = 100,
) -> Set[int]:
    """Compute transitive closure of dependencies from target instruction.

    Finds all instructions that could affect the target instruction
    through data dependencies.

    Args:
        target_pc: Starting instruction PC.
        pc_to_inst: Mapping from PC to InstructionStat.
        max_depth: Maximum trace depth to prevent infinite loops.

    Returns:
        Set of all PCs in the backward slice (including target_pc).
    """
    visited: Set[int] = set()
    worklist: List[tuple] = [(target_pc, 0)]  # (pc, depth)

    while worklist:
        pc, depth = worklist.pop()

        if pc in visited:
            continue
        if depth > max_depth:
            continue

        visited.add(pc)

        inst = pc_to_inst.get(pc)
        if not inst:
            continue

        # Get all dependencies and add to worklist
        deps = get_all_dependencies(inst)
        for dep_pc in deps:
            if dep_pc not in visited:
                worklist.append((dep_pc, depth + 1))

    return visited


def backward_slice_for_register(
    target_pc: int,
    register: int,
    pc_to_inst: Dict[int, InstructionStat],
    max_depth: int = 100,
) -> Set[int]:
    """Compute backward slice starting from a specific register use.

    Args:
        target_pc: Starting instruction PC.
        register: General register ID (0-255) to trace.
        pc_to_inst: Mapping from PC to InstructionStat.
        max_depth: Maximum trace depth.

    Returns:
        Set of all PCs in the backward slice for this register.
    """
    visited: Set[int] = set()

    inst = pc_to_inst.get(target_pc)
    if not inst:
        return visited

    # Get initial dependencies for the specific register
    initial_deps = inst.assign_pcs.get(register, [])

    worklist: List[tuple] = [(pc, 0) for pc in initial_deps]

    while worklist:
        pc, depth = worklist.pop()

        if pc in visited:
            continue
        if depth > max_depth:
            continue

        visited.add(pc)

        dep_inst = pc_to_inst.get(pc)
        if not dep_inst:
            continue

        # Follow all dependencies from this instruction
        deps = get_all_dependencies(dep_inst)
        for dep_pc in deps:
            if dep_pc not in visited:
                worklist.append((dep_pc, depth + 1))

    return visited


# =============================================================================
# Validation and Debugging
# =============================================================================


def validate_assign_pcs(instructions: List[InstructionStat]) -> List[str]:
    """Validate assign_pcs maps for consistency.

    Checks:
    - All referenced PCs exist in instruction list
    - No self-references (instruction depending on itself)
    - Source registers have corresponding assign_pcs entries

    Args:
        instructions: List of instructions with populated assign_pcs.

    Returns:
        List of warning/error messages (empty if valid).
    """
    errors: List[str] = []
    valid_pcs = {inst.pc for inst in instructions}

    for inst in instructions:
        # Check general register dependencies
        for reg, pcs in inst.assign_pcs.items():
            if reg not in inst.srcs:
                errors.append(
                    f"PC 0x{inst.pc:x}: assign_pcs has R{reg} but not in srcs"
                )
            for pc in pcs:
                if pc not in valid_pcs:
                    errors.append(
                        f"PC 0x{inst.pc:x}: assign_pcs references invalid PC 0x{pc:x}"
                    )
                if pc == inst.pc:
                    errors.append(
                        f"PC 0x{inst.pc:x}: self-reference in assign_pcs"
                    )

        # Check predicate dependencies
        for _, pcs in inst.passign_pcs.items():
            for pc in pcs:
                if pc not in valid_pcs:
                    errors.append(
                        f"PC 0x{inst.pc:x}: passign_pcs references invalid PC 0x{pc:x}"
                    )

        # Check predicate guard
        for pc in inst.predicate_assign_pcs:
            if pc not in valid_pcs:
                errors.append(
                    f"PC 0x{inst.pc:x}: predicate_assign_pcs references invalid PC 0x{pc:x}"
                )

    return errors


def print_dependencies(instructions: List[InstructionStat]) -> None:
    """Print dependency information for debugging.

    Args:
        instructions: List of instructions with populated assign_pcs.
    """
    for inst in instructions:
        print(f"PC 0x{inst.pc:04x}: {inst.op}")

        if inst.assign_pcs:
            for reg, pcs in sorted(inst.assign_pcs.items()):
                pcs_str = ", ".join(f"0x{pc:04x}" for pc in pcs)
                print(f"  R{reg} <- [{pcs_str}]")

        if inst.passign_pcs:
            for pred, pcs in sorted(inst.passign_pcs.items()):
                pcs_str = ", ".join(f"0x{pc:04x}" for pc in pcs)
                print(f"  P{pred} <- [{pcs_str}]")

        if inst.predicate_assign_pcs:
            pcs_str = ", ".join(f"0x{pc:04x}" for pc in inst.predicate_assign_pcs)
            print(f"  @P{inst.predicate} guard <- [{pcs_str}]")

        if inst.bassign_pcs:
            for barrier, pcs in sorted(inst.bassign_pcs.items()):
                pcs_str = ", ".join(f"0x{pc:04x}" for pc in pcs)
                print(f"  B{barrier} <- [{pcs_str}]")


# =============================================================================
# Statistics
# =============================================================================


def get_dependency_stats(instructions: List[InstructionStat]) -> Dict[str, int]:
    """Get statistics about assign_pcs maps.

    Returns:
        Dictionary with counts:
        - total_instructions: Number of instructions
        - instructions_with_deps: Instructions with at least one dependency
        - total_dependencies: Total number of dependency edges
        - avg_deps_per_inst: Average dependencies per instruction
    """
    total_deps = 0
    insts_with_deps = 0

    for inst in instructions:
        deps = get_all_dependencies(inst)
        if deps:
            insts_with_deps += 1
            total_deps += len(deps)

    return {
        "total_instructions": len(instructions),
        "instructions_with_deps": insts_with_deps,
        "total_dependencies": total_deps,
        "avg_deps_per_inst": total_deps / len(instructions) if instructions else 0,
    }
