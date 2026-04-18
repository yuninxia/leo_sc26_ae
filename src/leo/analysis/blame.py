"""GPU Instruction Blame Attribution Engine.

Based on GPA's GPUAdvisor-Blame.cpp algorithm (lines 1408-1713).

The blame attribution engine traces performance stalls backward through
instruction dependencies to identify root causes. It uses inverse-distance
weighting: closer instructions get MORE blame because they had more opportunity
to hide latency through pipelining.

Algorithm Overview:
1. Build dependency graph from assign_pcs maps
2. For each stalled instruction, identify incoming dependencies
3. Compute distance (instruction count) along CFG paths
4. Apply three-factor weighting: distance × efficiency × issue_count
5. Distribute blame proportionally to dependencies

Key Insight: blame_i = total_stall × (pivot_dist / dist_i) × (1/eff_i) × (issue_i / total_issue)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from leo.analysis.vma_property import VMAPropertyMap
from leo.binary.instruction import (
    InstructionStat,
    build_pc_to_inst_map,
    is_constant_memory_opcode,
    is_local_memory_opcode,
    is_shared_memory_opcode,
)
from leo.constants.metrics import METRIC_GCYCLES_ISU, METRIC_GCYCLES_LAT
from leo.binary.cfg import CFG
from leo.arch import GPUArchitecture

logger = logging.getLogger(__name__)


class BlameCategory(Enum):
    """High-level blame categories."""
    SELF = "self"           # No dependencies found (scheduler/indirect)
    EXEC_DEP = "exec_dep"   # Execution dependency (register RAW)
    MEM_DEP = "mem_dep"     # Memory dependency (load latency)
    SYNC = "sync"           # Synchronization (barrier)


@dataclass
class ChainNode:
    """Single node in a multi-hop blame chain."""
    pc: int
    opcode: str
    blame: float = 0.0  # Blame attributed at this hop


@dataclass
class BlameChain:
    """Multi-hop dependency path from stall to ultimate root cause.

    Example: s_waitcnt(0x100) <- global_load(0x0f0) <- v_mad_i64(0x0e0)
    The chain reveals the full causal story, not just the immediate dependency.
    """
    stall_pc: int
    total_blame: float
    nodes: List["ChainNode"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """Number of hops (edges) in the chain."""
        return max(0, len(self.nodes) - 1)

    @property
    def root_pc(self) -> int:
        """PC of the ultimate root cause (last node in chain)."""
        return self.nodes[-1].pc if self.nodes else self.stall_pc


@dataclass
class BlameEdge:
    """Represents blame flowing from source to destination instruction.

    Attributes:
        src_pc: PC of instruction causing blame (source of dependency)
        dst_pc: PC of instruction experiencing stall (uses the result)
        distance: Instruction count between them (0=same, -1=unknown, >0=exact)
        stall_blame: Stall cycles attributed to src_pc
        lat_blame: Latency cycles attributed to src_pc
        efficiency: Source instruction's hardware efficiency (0-1)
        pred_true: Predicate execution probability (-1 if unknown)
        issue_count: How many times src instruction issued
        blame_type: Specific type (e.g., "exec_dep_dep", "mem_dep_gmem")
        blame_category: High-level category (self, exec_dep, mem_dep, sync)
        src_opcode: Source instruction opcode
        dst_opcode: Destination instruction opcode
        src_operand_details: Parsed operand details for source instruction
        dst_operand_details: Parsed operand details for destination instruction
    """
    src_pc: int
    dst_pc: int
    distance: float
    stall_blame: float
    lat_blame: float
    efficiency: float = 1.0
    pred_true: float = -1.0
    issue_count: int = 0
    blame_type: str = "unknown"
    blame_category: str = "unknown"
    src_opcode: str = ""
    dst_opcode: str = ""
    src_operand_details: Optional[Dict[str, Any]] = None
    dst_operand_details: Optional[Dict[str, Any]] = None

    def total_blame(self) -> float:
        """Sum of stall and latency blame."""
        return self.stall_blame + self.lat_blame

    def is_self_blame(self) -> bool:
        """Check if this is self-blame (source == destination)."""
        return self.src_pc == self.dst_pc


@dataclass
class KernelBlameResult:
    """Aggregated blame analysis for a kernel.

    Contains all blame edges plus summary statistics for analysis.
    """
    blames: List[BlameEdge] = field(default_factory=list)
    blame_chains: List[BlameChain] = field(default_factory=list)

    # Totals
    total_stall_blame: float = 0.0
    total_lat_blame: float = 0.0

    # Aggregations
    blame_by_type: Dict[str, float] = field(default_factory=dict)
    blame_by_category: Dict[str, float] = field(default_factory=dict)

    # Rankings
    top_sources: List[Tuple[int, float]] = field(default_factory=list)
    top_sinks: List[Tuple[int, float]] = field(default_factory=list)

    # Statistics
    num_blame_edges: int = 0
    num_self_blame: int = 0
    num_single_source: int = 0
    num_multi_source: int = 0


    def get_blame_for_pc(self, pc: int) -> float:
        """Sum all blame attributed to a PC as source."""
        return sum(b.total_blame() for b in self.blames if b.src_pc == pc)

    def get_stalls_for_pc(self, pc: int) -> float:
        """Sum all stalls experienced by a PC as destination."""
        return sum(b.stall_blame for b in self.blames if b.dst_pc == pc)

    def get_top_blame_sources(self, n: int = 10) -> List[Tuple[int, float, str]]:
        """Get top N instructions by total outgoing blame.

        Returns:
            List of (pc, blame, opcode) tuples.
        """
        src_blames: Dict[int, Tuple[float, str]] = {}
        for b in self.blames:
            if b.src_pc not in src_blames:
                src_blames[b.src_pc] = (0.0, b.src_opcode)
            old_blame, opcode = src_blames[b.src_pc]
            src_blames[b.src_pc] = (old_blame + b.total_blame(), opcode)

        sorted_items = sorted(src_blames.items(), key=lambda x: x[1][0], reverse=True)
        return [(pc, blame, opcode) for pc, (blame, opcode) in sorted_items[:n]]

    def get_blame_type_breakdown(self) -> Dict[str, float]:
        """Get blame breakdown by type, sorted by value."""
        return dict(sorted(self.blame_by_type.items(), key=lambda x: x[1], reverse=True))


def reverse_ratio(distances: Dict[int, float]) -> Dict[int, float]:
    """Compute inverse-distance weighting for blame distribution.

    Closer instructions get MORE blame because they had more opportunity
    to hide latency through pipelining.

    Args:
        distances: Map from node_id to distance (instruction count)

    Returns:
        Normalized weights summing to 1.0

    Example:
        distances = {A: 2, B: 4, C: 10}
        pivot_distance = 2
        ratios = {A: 1.0, B: 0.5, C: 0.2}
        ratio_sum = 1.7
        weights = {A: 0.588, B: 0.294, C: 0.118}
    """
    if not distances:
        return {}

    nodes = list(distances.keys())

    # Step 1: Find pivot (minimum distance = closest dependency)
    # Using min ensures deterministic behavior regardless of dict iteration order
    pivot_distance = max(min(distances.values()), 1.0)

    # Step 2: Compute inverse ratios
    ratios: Dict[int, float] = {}
    ratio_sum = 0.0

    for node, dist in distances.items():
        dist = max(dist, 1.0)  # Avoid division by zero
        ratio = pivot_distance / dist  # Inverse: closer = larger
        ratios[node] = ratio
        ratio_sum += ratio

    # Step 3: Normalize to sum to 1.0
    if ratio_sum == 0:
        return {n: 1.0 / len(nodes) for n in nodes}

    return {node: ratio / ratio_sum for node, ratio in ratios.items()}


def distribute_blame(
    total_stall: float,
    total_lat: float,
    distances: Dict[int, float],
    efficiencies: Dict[int, float],
    issue_counts: Dict[int, int],
    stall_match_weights: Optional[Dict[int, float]] = None,
) -> Dict[int, Tuple[float, float]]:
    """Distribute blame across multiple dependencies using weighted factors.

    The factors are:
    1. Inverse distance: closer instructions get more blame
    2. Inverse efficiency: poorly-optimized instructions get more blame
    3. Issue count: frequently-executed instructions get more blame
    4. Stall-category match: edges whose type matches the destination's
       hardware-reported stall reason get more blame (optional)

    Args:
        total_stall: Total stall cycles to distribute
        total_lat: Total latency cycles to distribute
        distances: Distance from each source to destination
        efficiencies: Efficiency of each source (0-1, higher is better)
        issue_counts: Issue count for each source
        stall_match_weights: Optional weight per source (0-1) indicating how
            well the edge's blame category matches the destination's stall type.

    Returns:
        Dict mapping source_id to (stall_blame, lat_blame)
    """
    if not distances:
        return {}

    nodes = list(distances.keys())

    # Factor 1: Inverse distance weighting
    inst_ratios = reverse_ratio(distances)

    # Factor 2: Inverse efficiency weighting
    eff_ratios: Dict[int, float] = {}
    # Use minimum efficiency as pivot for deterministic behavior
    # (min efficiency = worst performer gets ratio 1.0, better ones get < 1.0)
    pivot_eff = min(efficiencies.values()) if efficiencies else 1.0

    if pivot_eff > 0:
        eff_sum = 0.0
        for node, eff in efficiencies.items():
            # Inverted: poor efficiency → higher weight → more blame
            ratio = pivot_eff / max(eff, 0.01)
            eff_ratios[node] = ratio
            eff_sum += ratio

        if eff_sum > 0:
            for node in eff_ratios:
                eff_ratios[node] /= eff_sum
    else:
        # Uniform efficiency
        for node in efficiencies:
            eff_ratios[node] = 1.0 / len(efficiencies)

    # Factor 3: Issue count normalization
    total_issues = sum(issue_counts.values())
    issue_ratios: Dict[int, float] = {}

    if total_issues > 0:
        for node, count in issue_counts.items():
            issue_ratios[node] = count / total_issues
    else:
        for node in issue_counts:
            issue_ratios[node] = 1.0 / len(issue_counts)

    # Combine factors multiplicatively
    combined: Dict[int, float] = {}
    combined_sum = 0.0

    for node in nodes:
        factor = (
            inst_ratios.get(node, 0) *
            eff_ratios.get(node, 1.0) *
            issue_ratios.get(node, 1.0)
        )
        if stall_match_weights is not None:
            factor *= stall_match_weights.get(node, 1.0)
        combined[node] = factor
        combined_sum += factor

    # Normalize and apply to totals
    result: Dict[int, Tuple[float, float]] = {}

    for node, factor in combined.items():
        if combined_sum > 0:
            normalized = factor / combined_sum
        else:
            normalized = 1.0 / len(nodes)

        result[node] = (
            total_stall * normalized,
            total_lat * normalized
        )

    return result


def compute_distance(
    from_vma: int,
    to_vma: int,
    inst_size: int = 16,
) -> float:
    """Compute instruction distance from from_vma to to_vma.

    Uses absolute distance for both forward and backward dependencies.
    This provides accurate blame attribution for loops.

    GPA Reference: Distance computation aligns with GPUAdvisor-Blame.cpp

    Args:
        from_vma: Source VMA (program counter).
        to_vma: Destination VMA.
        inst_size: Size of each instruction in bytes (default 16 for CUDA).

    Returns:
        Instruction count distance (always non-negative).
    """
    if from_vma == to_vma:
        return 0.0

    # Use absolute distance for both forward and backward dependencies
    # GPA-aligned: backward deps (loops) get real distance, not placeholder
    return abs(to_vma - from_vma) / inst_size


def compute_efficiency(
    inst: InstructionStat,
    prof_metrics: Dict[str, float],
) -> float:
    """Compute instruction efficiency (actual / theoretical bandwidth).

    For memory operations: coalescing efficiency
    For branches: 1 - divergence_ratio
    For others: 1.0 (no efficiency penalty)

    Args:
        inst: Instruction to evaluate
        prof_metrics: Profile metrics for this instruction

    Returns:
        Efficiency ratio (0-1, higher is better)
    """
    op = inst.op.upper()

    # Branch divergence efficiency
    if "BRA" in op or "BRANCH" in op:
        div = prof_metrics.get("branch_div", 0)
        exe = prof_metrics.get("branch_exe", 1)
        if exe > 0:
            return max(0.01, 1.0 - (div / exe))
        return 1.0

    # Memory coalescing efficiency
    if inst.is_memory_op():
        if is_shared_memory_opcode(op):
            # Shared memory efficiency
            actual = (
                prof_metrics.get("smem_load_trans", 0) +
                prof_metrics.get("smem_store_trans", 0)
            )
            theoretical = (
                prof_metrics.get("smem_load_trans_theor", 0) +
                prof_metrics.get("smem_store_trans_theor", 0)
            )
        else:
            # Global memory efficiency
            actual = (
                prof_metrics.get("gmem_cache_load_trans", 0) +
                prof_metrics.get("gmem_uncache_load_trans", 0) +
                prof_metrics.get("gmem_cache_store_trans", 0)
            )
            theoretical = (
                prof_metrics.get("gmem_cache_load_trans_theor", 0) +
                prof_metrics.get("gmem_uncache_load_trans_theor", 0) +
                prof_metrics.get("gmem_cache_store_trans_theor", 0)
            )

        if actual > 0 and theoretical > 0:
            # Efficiency = theoretical / actual (higher is better)
            return min(1.0, theoretical / actual)

    # Default: no efficiency penalty
    return 1.0


def detailize_blame_type(
    from_inst: InstructionStat,
    to_inst: InstructionStat,
    category: BlameCategory,
) -> str:
    """Classify blame type based on instruction properties.

    Args:
        from_inst: Source instruction (dependency provider)
        to_inst: Destination instruction (dependency consumer)
        category: High-level blame category

    Returns:
        Specific blame type string (e.g., "exec_dep_dep", "mem_dep_gmem")
    """
    if category == BlameCategory.SELF:
        if from_inst.indirect:
            return "self_indirect"
        return "self_scheduler"

    if category == BlameCategory.SYNC:
        return "sync_barrier"

    from_op = from_inst.op.upper()

    if category == BlameCategory.MEM_DEP:
        # Classify by memory type
        if is_local_memory_opcode(from_op):
            return "mem_dep_lmem"
        elif is_constant_memory_opcode(from_op):
            return "mem_dep_cmem"
        elif "TEX" in from_op:
            return "mem_dep_tmem"
        elif is_shared_memory_opcode(from_op):
            return "mem_dep_smem"
        else:
            return "mem_dep_gmem"

    if category == BlameCategory.EXEC_DEP:
        # Check for register dependency
        has_reg_dep = False

        # General registers
        for dst in from_inst.dsts:
            if dst in to_inst.srcs:
                has_reg_dep = True
                break

        # Uniform registers
        if not has_reg_dep:
            for dst in from_inst.udsts:
                if dst in to_inst.usrcs:
                    has_reg_dep = True
                    break

        # Predicate registers
        if not has_reg_dep:
            for dst in from_inst.pdsts:
                if dst in to_inst.psrcs or to_inst.predicate == dst:
                    has_reg_dep = True
                    break

        if has_reg_dep:
            # RAW dependency - classify by memory type if applicable
            if is_shared_memory_opcode(from_op):
                return "exec_dep_smem"
            elif is_constant_memory_opcode(from_op):
                return "exec_dep_cmem"
            else:
                return "exec_dep_dep"  # General RAW dependency

        # Check for barrier dependency
        has_barrier_dep = False
        for bdst in from_inst.bdsts:
            if bdst in to_inst.bsrcs:
                has_barrier_dep = True
                break

        if has_barrier_dep:
            return "exec_dep_war"  # Write-after-read via barrier

        return "exec_dep_sche"  # Scheduler dependency (no direct dep found)

    return "unknown"


def _determine_blame_category(
    from_inst: InstructionStat,
    to_inst: InstructionStat,
) -> BlameCategory:
    """Determine the high-level blame category for a dependency.

    Args:
        from_inst: Source instruction
        to_inst: Destination instruction

    Returns:
        BlameCategory enum value
    """
    # Check for synchronization
    if from_inst.is_sync() or to_inst.is_sync():
        return BlameCategory.SYNC

    # Check for memory operation
    if from_inst.is_memory_op():
        # Memory load causes latency
        if from_inst.is_load():
            return BlameCategory.MEM_DEP

    # Default to execution dependency
    return BlameCategory.EXEC_DEP


def get_all_dependencies(inst: InstructionStat) -> Set[int]:
    """Get all dependency PCs for an instruction.

    Collects dependencies from all assign_pcs maps.

    Args:
        inst: Instruction to get dependencies for

    Returns:
        Set of PCs that this instruction depends on
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

    # Predicate assignment
    deps.update(inst.predicate_assign_pcs)

    return deps


def blame_instructions(
    vma_map: VMAPropertyMap,
    instructions: List[InstructionStat],
    cfg: Optional[CFG] = None,
    arch: Optional[GPUArchitecture] = None,
) -> KernelBlameResult:
    """Main blame attribution engine.

    Attributes performance stalls to their root causes by tracing
    dependencies backward and distributing blame using inverse-distance
    weighting.

    Args:
        vma_map: VMA property map with profile data
        instructions: Parsed instructions with assign_pcs populated
        cfg: Optional control flow graph for accurate distance calculation
        arch: Optional GPU architecture for instruction size

    Returns:
        KernelBlameResult with all blame edges and aggregates

    Example:
        >>> result = blame_instructions(vma_map, instructions)
        >>> for src_pc, blame, opcode in result.get_top_blame_sources(5):
        ...     print(f"PC {src_pc:#x} ({opcode}): {blame:.0f} cycles")
    """
    logger.info(f"Starting blame attribution for {len(instructions)} instructions")

    # Build PC to instruction map
    pc_to_inst: Dict[int, InstructionStat] = build_pc_to_inst_map(instructions)

    # Get instruction size from architecture or use default
    inst_size = arch.inst_size if arch else 16

    # Collect all blame edges
    blame_edges: List[BlameEdge] = []

    # Track statistics
    single_source_count = 0
    multi_source_count = 0

    # Process each instruction with profile data
    for to_pc, prop in vma_map.items():
        if not prop.has_profile_data:
            continue

        if to_pc not in pc_to_inst:
            continue

        to_inst = pc_to_inst[to_pc]

        # Get stall metrics for this instruction
        stall_cycles = prop.stall_cycles
        lat_cycles = prop.prof_metrics.get(METRIC_GCYCLES_LAT, 0.0)
        issue_count = int(prop.prof_metrics.get(METRIC_GCYCLES_ISU, 0))

        # Skip if no stalls to attribute
        if stall_cycles <= 0 and lat_cycles <= 0:
            continue

        # Get all dependencies
        dep_pcs = get_all_dependencies(to_inst)

        # Filter to only PCs that exist in the binary
        # Note: Dependencies don't need profile data to receive blame.
        # The stalling instruction has the samples; root cause instructions
        # (like address calculations) may not stall themselves but still
        # cause the stall.
        valid_deps = {
            pc for pc in dep_pcs
            if pc in pc_to_inst
        }

        if not valid_deps:
            # Self-blame: no valid dependencies found
            blame = BlameEdge(
                src_pc=to_pc,
                dst_pc=to_pc,
                distance=0.0,
                stall_blame=stall_cycles,
                lat_blame=lat_cycles,
                efficiency=1.0,
                pred_true=-1.0,
                issue_count=issue_count,
                blame_type="self_scheduler" if not to_inst.indirect else "self_indirect",
                blame_category=BlameCategory.SELF.value,
                src_opcode=to_inst.op,
                dst_opcode=to_inst.op,
                src_operand_details=to_inst.operand_details,
                dst_operand_details=to_inst.operand_details,
            )
            blame_edges.append(blame)
        else:
            # Distribute blame to dependencies
            if len(valid_deps) == 1:
                single_source_count += 1
            else:
                multi_source_count += 1

            # Compute distances
            distances: Dict[int, float] = {}
            for from_pc in valid_deps:
                dist = compute_distance(from_pc, to_pc, inst_size)
                distances[from_pc] = max(dist, 1.0)  # Ensure positive distance

            # Compute efficiencies
            efficiencies: Dict[int, float] = {}
            for from_pc in valid_deps:
                from_inst = pc_to_inst[from_pc]
                from_prop = vma_map.get(from_pc)
                if from_prop:
                    eff = compute_efficiency(from_inst, from_prop.prof_metrics)
                else:
                    eff = 1.0
                efficiencies[from_pc] = eff

            # Compute issue counts
            issue_counts: Dict[int, int] = {}
            for from_pc in valid_deps:
                from_prop = vma_map.get(from_pc)
                if from_prop:
                    issue_counts[from_pc] = int(from_prop.prof_metrics.get(METRIC_GCYCLES_ISU, 1))
                else:
                    issue_counts[from_pc] = 1

            # Distribute blame using three-factor weighting
            blame_dist = distribute_blame(
                total_stall=stall_cycles,
                total_lat=lat_cycles,
                distances=distances,
                efficiencies=efficiencies,
                issue_counts=issue_counts,
            )

            # Create BlameEdge for each dependency
            for from_pc, (stall, lat) in blame_dist.items():
                from_inst = pc_to_inst[from_pc]

                # Determine blame category
                category = _determine_blame_category(from_inst, to_inst)

                # Get specific blame type
                blame_type = detailize_blame_type(from_inst, to_inst, category)

                blame = BlameEdge(
                    src_pc=from_pc,
                    dst_pc=to_pc,
                    distance=distances[from_pc],
                    stall_blame=stall,
                    lat_blame=lat,
                    efficiency=efficiencies[from_pc],
                    pred_true=-1.0,  # TODO: compute from predicate metrics
                    issue_count=issue_counts[from_pc],
                    blame_type=blame_type,
                    blame_category=category.value,
                    src_opcode=from_inst.op,
                    dst_opcode=to_inst.op,
                    src_operand_details=from_inst.operand_details,
                    dst_operand_details=to_inst.operand_details,
                )
                blame_edges.append(blame)

    logger.info(f"Created {len(blame_edges)} blame edges")

    # Aggregate results
    return _aggregate_blames(
        blame_edges,
        single_source_count,
        multi_source_count
    )


def _aggregate_blames(
    edges: List[BlameEdge],
    single_source_count: int,
    multi_source_count: int,
) -> KernelBlameResult:
    """Aggregate blame edges into result structure.

    Args:
        edges: List of BlameEdge objects
        single_source_count: Count of nodes with single dependency
        multi_source_count: Count of nodes with multiple dependencies

    Returns:
        KernelBlameResult with aggregated statistics
    """
    # Compute totals
    total_stall = sum(e.stall_blame for e in edges)
    total_lat = sum(e.lat_blame for e in edges)

    # Aggregate by type and category
    blame_by_type: Dict[str, float] = {}
    blame_by_category: Dict[str, float] = {}

    for edge in edges:
        total = edge.total_blame()

        blame_by_type[edge.blame_type] = (
            blame_by_type.get(edge.blame_type, 0) + total
        )
        blame_by_category[edge.blame_category] = (
            blame_by_category.get(edge.blame_category, 0) + total
        )

    # Top sources (instructions causing most blame)
    src_blames: Dict[int, float] = {}
    for edge in edges:
        src_blames[edge.src_pc] = src_blames.get(edge.src_pc, 0) + edge.total_blame()

    top_sources = sorted(src_blames.items(), key=lambda x: x[1], reverse=True)[:10]

    # Top sinks (instructions experiencing most stalls)
    sink_blames: Dict[int, float] = {}
    for edge in edges:
        sink_blames[edge.dst_pc] = sink_blames.get(edge.dst_pc, 0) + edge.stall_blame

    top_sinks = sorted(sink_blames.items(), key=lambda x: x[1], reverse=True)[:10]

    # Count self-blame edges
    num_self = sum(1 for e in edges if e.is_self_blame())

    return KernelBlameResult(
        blames=edges,
        total_stall_blame=total_stall,
        total_lat_blame=total_lat,
        blame_by_type=blame_by_type,
        blame_by_category=blame_by_category,
        top_sources=top_sources,
        top_sinks=top_sinks,
        num_blame_edges=len(edges),
        num_self_blame=num_self,
        num_single_source=single_source_count,
        num_multi_source=multi_source_count,
    )


def print_blame_report(result: KernelBlameResult, top_n: int = 10) -> None:
    """Print a human-readable blame report.

    Args:
        result: KernelBlameResult to print
        top_n: Number of top items to show in each section
    """
    logger.info("=" * 60)
    logger.info("BLAME ATTRIBUTION REPORT")
    logger.info("=" * 60)

    logger.info(f"\nTotal Stall Blame: {result.total_stall_blame:,.0f} cycles")
    logger.info(f"Total Latency Blame: {result.total_lat_blame:,.0f} cycles")
    logger.info(f"Total Blame Edges: {result.num_blame_edges}")
    logger.info(f"  Self-blame: {result.num_self_blame}")
    logger.info(f"  Single-source: {result.num_single_source}")
    logger.info(f"  Multi-source: {result.num_multi_source}")

    logger.info(f"\n--- Blame by Category ---")
    for category, blame in sorted(result.blame_by_category.items(),
                                   key=lambda x: x[1], reverse=True):
        pct = blame / (result.total_stall_blame + result.total_lat_blame) * 100
        logger.info(f"  {category}: {blame:,.0f} ({pct:.1f}%)")

    logger.info(f"\n--- Blame by Type ---")
    for btype, blame in sorted(result.blame_by_type.items(),
                                key=lambda x: x[1], reverse=True)[:top_n]:
        pct = blame / (result.total_stall_blame + result.total_lat_blame) * 100
        logger.info(f"  {btype}: {blame:,.0f} ({pct:.1f}%)")

    logger.info(f"\n--- Top Blame Sources (Instructions Causing Stalls) ---")
    for src_pc, blame, opcode in result.get_top_blame_sources(top_n):
        logger.info(f"  PC {src_pc:#x} ({opcode}): {blame:,.0f} cycles")

    logger.info(f"\n--- Top Stalled Instructions ---")
    for dst_pc, stall in result.top_sinks[:top_n]:
        logger.info(f"  PC {dst_pc:#x}: {stall:,.0f} stall cycles")

    logger.info("=" * 60)
