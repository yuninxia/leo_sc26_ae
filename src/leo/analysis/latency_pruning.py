"""Latency-based dependency pruning.

Based on GPA's GPUAdvisor-Blame.cpp trackDep function (lines 713-817).

Latency pruning removes false dependencies where enough instruction cycles
pass to "hide" the latency through pipelining.

Problem it solves:
    0x00: LDG R1, [R2]        ; Global load, latency 100-400 cycles
    0x10: IADD R3, R4, R5     ; 4 stall cycles
    0x20: IADD R6, R7, R8     ; 4 stall cycles
    ... (many instructions) ...
    0x80: FADD R9, R1, R10    ; Uses R1 - is 0x00 really a bottleneck?

If 100+ stall cycles accumulated between 0x00 and 0x80, the dependency is
"hidden" by the pipeline and shouldn't be blamed for stalls.

Algorithm:
1. Start from the defining instruction (source)
2. Walk forward through instructions, accumulating control.stall cycles
3. If we reach the latency threshold before finding the use, dependency is hidden
4. If we find the use before threshold, dependency is real (critical)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable

from leo.binary.instruction import InstructionStat, build_pc_to_inst_map
from leo.binary.cfg import CFG, Block
from leo.arch import GPUArchitecture


@dataclass
class LatencyCheckResult:
    """Result of latency hidden check."""
    is_hidden: bool           # True if dependency is hidden by latency
    accumulated_cycles: int   # Total stall cycles accumulated
    latency_threshold: int    # Architecture latency for the defining instruction
    found_use: bool           # True if we found the register use


def check_latency_hidden_linear(
    from_pc: int,
    to_pc: int,
    target_reg: int,
    instructions: List[InstructionStat],
    arch: GPUArchitecture,
    use_min_latency: bool = True,
) -> LatencyCheckResult:
    """Check if dependency is hidden by latency (linear scan version).

    This is for simple basic-block-level analysis without CFG traversal.

    Args:
        from_pc: PC of instruction that defines the register.
        to_pc: PC of instruction that uses the register.
        target_reg: Register ID being tracked.
        instructions: List of instructions in PC order.
        arch: GPU architecture for latency lookup.
        use_min_latency: If True, use min latency (best case).
                        If False, use max latency (worst case).

    Returns:
        LatencyCheckResult with hidden status and details.
    """
    # Build PC to instruction map
    pc_to_inst = build_pc_to_inst_map(instructions)

    # Get the defining instruction's latency
    from_inst = pc_to_inst.get(from_pc)
    if not from_inst:
        return LatencyCheckResult(
            is_hidden=False,
            accumulated_cycles=0,
            latency_threshold=0,
            found_use=False,
        )

    lat_min, lat_max = arch.latency(from_inst.op)
    latency_threshold = lat_min if use_min_latency else lat_max

    # Sort instructions by PC and find range
    sorted_insts = sorted(instructions, key=lambda i: i.pc)

    accumulated_cycles = 0
    found_use = False
    current_pc = from_pc + arch.inst_size

    for inst in sorted_insts:
        if inst.pc <= from_pc:
            continue  # Skip instructions before source
        if inst.pc > to_pc:
            break  # Past target

        # Accumulate stall cycles
        accumulated_cycles += inst.control.stall

        # Check if this instruction uses the target register
        if target_reg in inst.srcs:
            found_use = True
            break

        # Check if latency is hidden
        if accumulated_cycles >= latency_threshold:
            return LatencyCheckResult(
                is_hidden=True,
                accumulated_cycles=accumulated_cycles,
                latency_threshold=latency_threshold,
                found_use=False,
            )

    return LatencyCheckResult(
        is_hidden=False,
        accumulated_cycles=accumulated_cycles,
        latency_threshold=latency_threshold,
        found_use=found_use,
    )


@dataclass
class PathResult:
    """Result of CFG path traversal."""
    blocks: List[Block]       # Blocks in this path
    is_hidden: bool           # True if dependency is hidden on this path
    found_use: bool           # True if register use was found
    accumulated_cycles: int   # Total stall cycles on this path


def track_dep_with_latency_cfg(
    from_pc: int,
    to_pc: int,
    target_reg: int,
    cfg: CFG,
    pc_to_inst: Dict[int, InstructionStat],
    arch: GPUArchitecture,
    use_min_latency: bool = True,
) -> List[PathResult]:
    """Track dependency through CFG with latency pruning.

    This is the full CFG-aware version that handles control flow properly.

    Args:
        from_pc: PC of instruction that defines the register.
        to_pc: PC of instruction that uses the register.
        target_reg: Register ID being tracked.
        cfg: Control flow graph.
        pc_to_inst: Map from PC to InstructionStat.
        arch: GPU architecture for latency lookup.
        use_min_latency: If True, use min latency.

    Returns:
        List of PathResult for all valid paths (not hidden, use found).
    """
    # Get the defining instruction's latency
    from_inst = pc_to_inst.get(from_pc)
    if not from_inst:
        return []

    lat_min, lat_max = arch.latency(from_inst.op)
    latency_threshold = lat_min if use_min_latency else lat_max

    # Find blocks containing from_pc and to_pc
    from_block = cfg.get_block_containing_pc(from_pc)
    to_block = cfg.get_block_containing_pc(to_pc)

    if not from_block or not to_block:
        return []

    # DFS traversal
    valid_paths: List[PathResult] = []
    visited_blocks: Set[int] = set()
    current_path: List[Block] = []

    _track_dep_dfs(
        from_pc=from_pc,
        to_pc=to_pc,
        target_reg=target_reg,
        current_block=from_block,
        to_block=to_block,
        accumulated_cycles=0,
        latency_threshold=latency_threshold,
        cfg=cfg,
        pc_to_inst=pc_to_inst,
        arch=arch,
        visited_blocks=visited_blocks,
        current_path=current_path,
        valid_paths=valid_paths,
    )

    return valid_paths


def _track_dep_dfs(
    from_pc: int,
    to_pc: int,
    target_reg: int,
    current_block: Block,
    to_block: Block,
    accumulated_cycles: int,
    latency_threshold: int,
    cfg: CFG,
    pc_to_inst: Dict[int, InstructionStat],
    arch: GPUArchitecture,
    visited_blocks: Set[int],
    current_path: List[Block],
    valid_paths: List[PathResult],
) -> None:
    """DFS traversal for latency-aware dependency tracking.

    Based on GPA's trackDep (GPUAdvisor-Blame.cpp:714-817).
    """
    # Prevent infinite loops
    if current_block.id in visited_blocks:
        return
    visited_blocks.add(current_block.id)
    current_path.append(current_block)

    # Determine instruction range within this block
    block_start_pc = current_block.start_pc
    block_end_pc = current_block.end_pc

    # Adjust start if from_pc is in this block
    start_pc = block_start_pc
    if from_pc >= block_start_pc and from_pc <= block_end_pc:
        start_pc = from_pc + arch.inst_size  # Skip the defining instruction

    # Adjust end if to_pc is in this block
    end_pc = block_end_pc
    if to_pc >= block_start_pc and to_pc <= block_end_pc:
        end_pc = to_pc - arch.inst_size  # Stop before the using instruction

    # Check for loop (from_pc >= to_pc in same block)
    loop_block = False
    if (from_pc >= block_start_pc and from_pc <= block_end_pc and
        to_pc >= block_start_pc and to_pc <= block_end_pc and
        from_pc >= to_pc):
        # Loop case: reset and search entire block
        visited_blocks.discard(current_block.id)
        end_pc = block_end_pc
        loop_block = True

    # Iterate through instructions in this block
    found_use = False
    is_hidden = False

    for inst in current_block.instructions:
        if inst.pc < start_pc:
            continue
        if inst.pc > end_pc:
            break

        # Accumulate stall cycles
        accumulated_cycles += inst.control.stall

        # Check if instruction uses target register
        if target_reg in inst.srcs:
            found_use = True
            break

        # Check latency threshold
        if accumulated_cycles >= latency_threshold:
            is_hidden = True
            break

    # Record result if we found use or finished
    if found_use:
        valid_paths.append(PathResult(
            blocks=list(current_path),
            is_hidden=False,
            found_use=True,
            accumulated_cycles=accumulated_cycles,
        ))
    elif is_hidden:
        # Dependency is hidden, don't add to valid paths
        pass
    elif current_block == to_block and not loop_block:
        # Reached target block without finding use or hitting threshold
        valid_paths.append(PathResult(
            blocks=list(current_path),
            is_hidden=False,
            found_use=False,
            accumulated_cycles=accumulated_cycles,
        ))
    else:
        # Continue to successor blocks
        for target in current_block.targets:
            successor = cfg.get_block(target.to_block_id)
            if successor:
                _track_dep_dfs(
                    from_pc=0,  # Start from beginning of successor
                    to_pc=to_pc,
                    target_reg=target_reg,
                    current_block=successor,
                    to_block=to_block,
                    accumulated_cycles=accumulated_cycles,
                    latency_threshold=latency_threshold,
                    cfg=cfg,
                    pc_to_inst=pc_to_inst,
                    arch=arch,
                    visited_blocks=set(visited_blocks),  # Copy for branching
                    current_path=list(current_path),      # Copy for branching
                    valid_paths=valid_paths,
                )

    # Backtrack
    visited_blocks.discard(current_block.id)
    if current_path and current_path[-1] == current_block:
        current_path.pop()


def prune_assign_pcs_by_latency(
    instructions: List[InstructionStat],
    arch: GPUArchitecture,
    use_min_latency: bool = True,
) -> Dict[int, Dict[int, List[int]]]:
    """Prune assign_pcs maps based on latency hiding.

    For each instruction, check if its dependencies are hidden by
    pipeline latency. Hidden dependencies are removed from assign_pcs.

    Args:
        instructions: Instructions with populated assign_pcs.
        arch: GPU architecture for latency lookup.
        use_min_latency: Use min latency (optimistic) or max (pessimistic).

    Returns:
        Dictionary mapping PC -> reg -> pruned list of assignment PCs.
    """
    result: Dict[int, Dict[int, List[int]]] = {}

    for inst in instructions:
        result[inst.pc] = {}

        for reg, assign_pcs in inst.assign_pcs.items():
            valid_pcs: List[int] = []

            for from_pc in assign_pcs:
                check_result = check_latency_hidden_linear(
                    from_pc=from_pc,
                    to_pc=inst.pc,
                    target_reg=reg,
                    instructions=instructions,
                    arch=arch,
                    use_min_latency=use_min_latency,
                )

                if not check_result.is_hidden:
                    valid_pcs.append(from_pc)

            if valid_pcs:
                result[inst.pc][reg] = valid_pcs

    return result


def apply_latency_pruning(
    instructions: List[InstructionStat],
    arch: GPUArchitecture,
    use_min_latency: bool = True,
) -> None:
    """Apply latency pruning to instruction assign_pcs in place.

    Args:
        instructions: Instructions with populated assign_pcs.
        arch: GPU architecture for latency lookup.
        use_min_latency: Use min latency (optimistic) or max (pessimistic).

    Modifies:
        Each instruction's assign_pcs in place.
    """
    pruned = prune_assign_pcs_by_latency(instructions, arch, use_min_latency)

    for inst in instructions:
        if inst.pc in pruned:
            # Replace assign_pcs with pruned version
            inst.assign_pcs = pruned[inst.pc]
