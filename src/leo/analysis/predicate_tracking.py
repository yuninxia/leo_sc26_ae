"""Predicate tracking for accurate dependency analysis.

Based on GPA's AnalyzeInstruction.cpp trackDependency function (lines 677-815).

Predicate tracking avoids over-conservative dependencies by tracking predicate
conditions (@P0 vs @!P0) during dependency traversal.

Problem it solves:
    0x00: @P0  MOV R5, R1      ; Writes R5 if P0 is true
    0x10: @!P0 MOV R5, R2      ; Writes R5 if P0 is false
    0x20:      ADD R6, R5, R3  ; Uses R5 - which definition?

Without predicate tracking, we'd conservatively say R5 comes from both 0x00 and 0x10.
With predicate tracking, we know exactly one will execute based on P0's value.

Three Stopping Conditions:
1. Unconditional instruction - No predicate guard, always executes
2. Exact predicate match - Same predicate register AND same flag (TRUE/FALSE)
3. Contradictory predicates - Found both @P_i and @!P_i on the current path
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Callable

from leo.binary.instruction import InstructionStat, PredicateFlag


# Default maximum recursion depth (matches GPA's TRACK_LIMIT)
DEFAULT_TRACK_LIMIT = 8


class StopReason(IntEnum):
    """Reason for stopping predicate tracking."""
    CONTINUE = 0           # Don't stop, continue traversal
    DEPTH_LIMIT = 1        # Reached maximum depth
    UNCONDITIONAL = 2      # Found unconditional instruction
    EXACT_MATCH = 3        # Found exact predicate match
    CONTRADICTION = 4      # Found contradictory predicates (@P and @!P)


@dataclass
class PredicateState:
    """State for predicate tracking during DFS traversal.

    The predicate_map tracks which predicates have been encountered
    on the current path. Uses signed integer keys:
    - Positive key (pred_id + 1): PREDICATE_TRUE seen
    - Negative key (-(pred_id + 1)): PREDICATE_FALSE seen

    The +1 offset avoids ambiguity with P0 (which would otherwise map to 0).

    Example:
        {1: 2, -2: 1} means:
        - @P0 (TRUE) seen twice on path
        - @!P1 (FALSE) seen once on path
    """
    predicate_map: Dict[int, int] = field(default_factory=dict)
    depth: int = 0
    track_limit: int = DEFAULT_TRACK_LIMIT

    def _get_key(self, pred_id: int, flag: PredicateFlag) -> int:
        """Convert predicate register and flag to map key."""
        offset = pred_id + 1  # +1 to avoid P0 = 0 issue
        if flag == PredicateFlag.PREDICATE_TRUE:
            return offset
        else:  # PREDICATE_FALSE
            return -offset

    def push_predicate(self, pred_id: int, flag: PredicateFlag) -> None:
        """Record a predicate encounter on the current path."""
        if flag == PredicateFlag.PREDICATE_NONE:
            return
        key = self._get_key(pred_id, flag)
        self.predicate_map[key] = self.predicate_map.get(key, 0) + 1

    def pop_predicate(self, pred_id: int, flag: PredicateFlag) -> None:
        """Remove a predicate from the current path (backtracking)."""
        if flag == PredicateFlag.PREDICATE_NONE:
            return
        key = self._get_key(pred_id, flag)
        if key in self.predicate_map:
            self.predicate_map[key] -= 1
            if self.predicate_map[key] <= 0:
                del self.predicate_map[key]

    def has_contradiction(self, pred_id: int, flag: PredicateFlag) -> bool:
        """Check if adding this predicate creates a contradiction.

        A contradiction occurs when we've seen @P_i on the path and
        now encounter @!P_i (or vice versa).
        """
        if flag == PredicateFlag.PREDICATE_NONE:
            return False

        offset = pred_id + 1

        if flag == PredicateFlag.PREDICATE_TRUE:
            # Checking if @P_i, look for @!P_i in history
            return self.predicate_map.get(-offset, 0) > 0
        else:
            # Checking if @!P_i, look for @P_i in history
            return self.predicate_map.get(offset, 0) > 0

    def clone(self) -> "PredicateState":
        """Create a copy of this state for branching."""
        return PredicateState(
            predicate_map=dict(self.predicate_map),
            depth=self.depth,
            track_limit=self.track_limit,
        )


def should_stop_predicate_tracking(
    target_inst: InstructionStat,
    candidate_inst: InstructionStat,
    state: PredicateState,
) -> StopReason:
    """Determine if we should stop tracking at the candidate instruction.

    Based on GPA's three stopping conditions (AnalyzeInstruction.cpp:783-794).

    Args:
        target_inst: The instruction whose dependencies we're tracing.
        candidate_inst: The candidate dependency we're evaluating.
        state: Current predicate tracking state.

    Returns:
        StopReason indicating why to stop, or CONTINUE if we should proceed.
    """
    # Condition 1: Depth limit reached
    if state.depth >= state.track_limit:
        return StopReason.DEPTH_LIMIT

    # Condition 2a: Candidate has no predicate guard (unconditional)
    # This instruction always executes, so it's definitely the source
    if candidate_inst.predicate_flag == PredicateFlag.PREDICATE_NONE:
        return StopReason.UNCONDITIONAL

    # Condition 2b: Exact predicate match
    # Same predicate register AND same flag means same execution condition
    if (target_inst.predicate == candidate_inst.predicate and
        target_inst.predicate_flag == candidate_inst.predicate_flag):
        return StopReason.EXACT_MATCH

    # Condition 3: Contradictory predicates
    # If we've seen @P_i on path and candidate is @!P_i (or vice versa),
    # these are mutually exclusive - stop tracking
    if state.has_contradiction(candidate_inst.predicate, candidate_inst.predicate_flag):
        return StopReason.CONTRADICTION

    return StopReason.CONTINUE


def track_dependency_with_predicates(
    target_inst: InstructionStat,
    pc_to_inst: Dict[int, InstructionStat],
    get_predecessors: Callable[[int], List[int]],
    track_limit: int = DEFAULT_TRACK_LIMIT,
) -> Dict[int, List[int]]:
    """Build assign_pcs with predicate-aware dependency tracking.

    This performs DFS through the dependency graph while tracking predicate
    conditions to prune impossible paths.

    Args:
        target_inst: Instruction to find dependencies for.
        pc_to_inst: Map from PC to InstructionStat.
        get_predecessors: Function that returns predecessor PCs for a given PC.
        track_limit: Maximum recursion depth.

    Returns:
        Dictionary mapping register ID to list of valid assignment PCs.
        This is the refined assign_pcs after predicate pruning.
    """
    result: Dict[int, List[int]] = {}

    # Initialize predicate state with target's predicate (if any)
    initial_state = PredicateState(track_limit=track_limit)
    if target_inst.predicate >= 0:
        initial_state.push_predicate(target_inst.predicate, target_inst.predicate_flag)

    # Process each source register
    for src_reg in target_inst.srcs:
        valid_pcs: Set[int] = set()

        # Get initial candidates from assign_pcs
        initial_candidates = target_inst.assign_pcs.get(src_reg, [])

        for candidate_pc in initial_candidates:
            candidate = pc_to_inst.get(candidate_pc)
            if not candidate:
                continue

            # Run predicate-aware DFS from this candidate
            _track_register_dfs(
                target_inst=target_inst,
                current_inst=candidate,
                src_reg=src_reg,
                state=initial_state.clone(),
                pc_to_inst=pc_to_inst,
                get_predecessors=get_predecessors,
                valid_pcs=valid_pcs,
                visited=set(),
            )

        if valid_pcs:
            result[src_reg] = sorted(valid_pcs)

    return result


def _track_register_dfs(
    target_inst: InstructionStat,
    current_inst: InstructionStat,
    src_reg: int,
    state: PredicateState,
    pc_to_inst: Dict[int, InstructionStat],
    get_predecessors: Callable[[int], List[int]],
    valid_pcs: Set[int],
    visited: Set[int],
) -> None:
    """DFS traversal with predicate state tracking.

    Args:
        target_inst: Original target instruction.
        current_inst: Current instruction being evaluated.
        src_reg: Register we're tracking.
        state: Current predicate state.
        pc_to_inst: PC to instruction map.
        get_predecessors: Function to get predecessor PCs.
        valid_pcs: Set to collect valid assignment PCs (output).
        visited: Set of already-visited PCs to prevent cycles.
    """
    if current_inst.pc in visited:
        return
    visited.add(current_inst.pc)

    # Check stopping conditions
    stop_reason = should_stop_predicate_tracking(target_inst, current_inst, state)

    if stop_reason == StopReason.DEPTH_LIMIT:
        # At depth limit, conservatively include this PC
        valid_pcs.add(current_inst.pc)
        return

    if stop_reason == StopReason.UNCONDITIONAL:
        # Unconditional instruction - definitely a valid source
        valid_pcs.add(current_inst.pc)
        return

    if stop_reason == StopReason.EXACT_MATCH:
        # Exact predicate match - definitely a valid source
        valid_pcs.add(current_inst.pc)
        return

    if stop_reason == StopReason.CONTRADICTION:
        # Contradictory predicates - this path is impossible, don't include
        return

    # CONTINUE: Push predicate state and recurse to predecessors
    if current_inst.predicate >= 0:
        state.push_predicate(current_inst.predicate, current_inst.predicate_flag)

    state.depth += 1

    # Does current_inst write to src_reg?
    if src_reg in current_inst.dsts:
        # This instruction writes the register
        # Check if we should continue looking further back
        # For now, record this as a valid source
        valid_pcs.add(current_inst.pc)

    # Continue to predecessors if we haven't found a definitive source
    # or if there might be other paths
    pred_pcs = get_predecessors(current_inst.pc)
    for pred_pc in pred_pcs:
        pred_inst = pc_to_inst.get(pred_pc)
        if pred_inst:
            _track_register_dfs(
                target_inst=target_inst,
                current_inst=pred_inst,
                src_reg=src_reg,
                state=state.clone(),  # Clone for each branch
                pc_to_inst=pc_to_inst,
                get_predecessors=get_predecessors,
                valid_pcs=valid_pcs,
                visited=set(visited),  # Copy visited for each branch
            )

    # Pop predicate state (backtrack)
    if current_inst.predicate >= 0:
        state.pop_predicate(current_inst.predicate, current_inst.predicate_flag)

    state.depth -= 1


def refine_assign_pcs_with_predicates(
    instructions: List[InstructionStat],
    pc_to_inst: Dict[int, InstructionStat],
    get_predecessors: Callable[[int], List[int]],
    track_limit: int = DEFAULT_TRACK_LIMIT,
) -> None:
    """Refine assign_pcs maps using predicate tracking.

    This post-processes assign_pcs to remove impossible dependencies
    based on predicate conditions.

    Args:
        instructions: List of instructions with populated assign_pcs.
        pc_to_inst: Map from PC to InstructionStat.
        get_predecessors: Function that returns predecessor PCs.
        track_limit: Maximum tracking depth.

    Modifies:
        Each instruction's assign_pcs in place.
    """
    for inst in instructions:
        if not inst.is_predicated():
            continue  # No predicate to track

        # Only refine if instruction has a predicate guard
        refined = track_dependency_with_predicates(
            target_inst=inst,
            pc_to_inst=pc_to_inst,
            get_predecessors=get_predecessors,
            track_limit=track_limit,
        )

        # Update assign_pcs with refined results
        for reg, pcs in refined.items():
            if reg in inst.assign_pcs:
                # Keep only PCs that passed predicate tracking
                inst.assign_pcs[reg] = pcs
