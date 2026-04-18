"""Unified back-slicing analysis engine for GPU performance diagnosis.

Based on GPA's GPUAdvisor::blame() function from GPUAdvisor-Blame.cpp.

This module is the main orchestrator that coordinates the complete back-slicing
pipeline from binary+profile data through blame attribution:

1. Initialize: Parse binary, build CFG, populate assign_pcs
2. Build VMA Property Map: Join binary + profile data
3. Build CCT Dependency Graph: Create nodes and edges from dependencies
4. 4-Stage Pruning Pipeline:
   - Stage 1: Opcode constraints (exec_dep vs mem_dep)
   - Stage 2: Barrier constraints (synchronization matching)
   - Stage 3: Latency constraints (pipeline hiding)
   - Stage 4: Branch constraints (execution path filtering)
5. Blame Attribution: Inverse-distance weighted distribution

Usage:
    engine = BackSliceEngine(
        db_path="/path/to/database",
        instructions=parsed_instructions,
        arch_name="a100"
    )
    result = engine.analyze()

    # Access results
    print(f"Total blame: {result.total_stall_blame:.0f} cycles")
    for src_pc, blame in result.get_top_blame_sources(5):
        print(f"  PC {src_pc:#x}: {blame:.0f} cycles")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from leo.binary.instruction import (
    InstructionStat,
    build_pc_to_inst_map,
)
from leo.binary.cfg import CFG
from leo.arch import get_architecture
from leo.arch.base import GPUArchitecture

from leo.analysis.vma_property import VMAPropertyMap
from leo.constants.metrics import (
    METRIC_GCYCLES_STL_GMEM,
    METRIC_GCYCLES_STL_IDEP,
    METRIC_GCYCLES_STL_IFET,
    METRIC_GCYCLES_STL_LMEM,
    METRIC_GCYCLES_STL_MEM,
    METRIC_GCYCLES_STL_PIPE,
    METRIC_GCYCLES_STL_SYNC,
    METRIC_GCYCLES_STL_TMEM,
)
from leo.analysis.graph import (
    CCTDepGraph,
    CCTDepNode,
    build_cct_dep_graph,
    prune_opcode_constraints,
    prune_latency_constraints,
    prune_barrier_constraints,
    prune_execution_constraints,
    get_stalling_nodes,
)
from leo.analysis.blame import (
    BlameCategory,
    BlameChain,
    BlameEdge,
    ChainNode,
    KernelBlameResult,
    _aggregate_blames,
    _determine_blame_category,
    distribute_blame,
    compute_distance,
    compute_efficiency,
    detailize_blame_type,
)

logger = logging.getLogger(__name__)


@dataclass
class BackSliceConfig:
    """Configuration for back-slicing analysis.

    Attributes:
        arch_name: GPU architecture name ("a100", "v100", etc.)
        enable_predicate_tracking: If True, refine dependencies with predicate analysis.
        predicate_track_limit: Maximum DFS depth for predicate tracking.
        use_min_latency: If True, use minimum latency (optimistic).
        apply_opcode_pruning: If True, apply opcode-based edge pruning.
        apply_barrier_pruning: If True, apply barrier-based edge pruning.
        apply_graph_latency_pruning: If True, apply latency-based graph pruning.
        enable_execution_pruning: If True, prune edges where source was never executed.
            Disabled by default (matches GPA's default behavior).
        stall_threshold: Minimum stall cycles to consider for blame.
        debug: If True, print debug information during analysis.
    """

    arch_name: str = "a100"
    enable_predicate_tracking: bool = True
    predicate_track_limit: int = 8
    use_min_latency: bool = True
    apply_opcode_pruning: bool = True
    apply_barrier_pruning: bool = True
    apply_graph_latency_pruning: bool = True
    enable_execution_pruning: bool = False
    stall_threshold: float = 0.0
    debug: bool = False


@dataclass
class BackSliceStats:
    """Statistics from back-slicing analysis."""

    # Input counts
    num_instructions: int = 0
    num_instructions_with_profile: int = 0

    # Graph stats
    initial_nodes: int = 0
    initial_edges: int = 0
    final_nodes: int = 0
    final_edges: int = 0

    # Pruning stats
    edges_pruned_opcode: int = 0
    edges_pruned_barrier: int = 0
    edges_pruned_latency: int = 0
    edges_pruned_execution: int = 0

    # Single dependency coverage (GPA-style graph quality metric)
    sdc_before_pruning: float = 0.0  # after register edges + type classification, before sync tracing
    sdc_after_pruning: float = 0.0   # after sync tracing + pruning

    # Blame stats
    num_stalling_nodes: int = 0
    num_blame_edges: int = 0
    num_self_blame: int = 0


class BackSliceEngine:
    """Unified back-slicing analysis engine.

    Orchestrates the complete pipeline from binary+profile data through
    blame attribution. This corresponds to GPUAdvisor::blame() in the
    GPA C++ implementation.

    Example:
        # With database path
        engine = BackSliceEngine.from_database(
            db_path="/path/to/database",
            binary_path="/path/to/kernel.cubin",
        )
        result = engine.analyze()

        # With pre-built VMAPropertyMap
        engine = BackSliceEngine(
            vma_map=vma_map,
            instructions=instructions,
        )
        result = engine.analyze()
    """

    def __init__(
        self,
        vma_map: VMAPropertyMap,
        instructions: List[InstructionStat],
        cfg: Optional[CFG] = None,
        config: Optional[BackSliceConfig] = None,
        arch_override: Optional[GPUArchitecture] = None,
    ):
        """Initialize the back-slicing engine.

        Args:
            vma_map: VMAPropertyMap with joined binary + profile data.
            instructions: List of parsed instructions with assign_pcs populated.
            cfg: Optional CFG for path-based analysis.
            config: Optional configuration (uses defaults if not provided).
            arch_override: Optional pre-constructed GPUArchitecture instance.
                If provided, used instead of creating one from config.arch_name.
                Useful for sensitivity analysis with PerturbedArchitecture.
        """
        self.vma_map = vma_map
        self.instructions = instructions
        self.cfg = cfg
        self.config = config or BackSliceConfig()
        self.arch = arch_override or get_architecture(self.config.arch_name)

        # Build PC -> instruction map
        self._pc_to_inst: Dict[int, InstructionStat] = build_pc_to_inst_map(instructions)

        # Results (populated by analyze())
        self.graph: Optional[CCTDepGraph] = None
        self.stats = BackSliceStats()
        self._result: Optional[KernelBlameResult] = None

    @classmethod
    def from_database(
        cls,
        db_path: str,
        instructions: List[InstructionStat],
        cfg: Optional[CFG] = None,
        config: Optional[BackSliceConfig] = None,
    ) -> "BackSliceEngine":
        """Create engine from database path.

        This method builds the VMAPropertyMap from the database and
        instructions, then initializes the engine.

        Args:
            db_path: Path to HPCToolkit database.
            instructions: List of parsed instructions.
            cfg: Optional CFG.
            config: Optional configuration.

        Returns:
            Initialized BackSliceEngine.
        """
        config = config or BackSliceConfig()

        # Build VMA property map
        vma_map = VMAPropertyMap.build(
            db_path=db_path,
            instructions=instructions,
            arch_name=config.arch_name,
        )

        return cls(
            vma_map=vma_map,
            instructions=instructions,
            cfg=cfg,
            config=config,
        )

    @classmethod
    def from_instructions(
        cls,
        instructions: List[InstructionStat],
        cfg: Optional[CFG] = None,
        config: Optional[BackSliceConfig] = None,
    ) -> "BackSliceEngine":
        """Create engine from instructions only (no database).

        Useful for testing or when profile data is simulated.

        Args:
            instructions: List of parsed instructions.
            cfg: Optional CFG.
            config: Optional configuration.

        Returns:
            Initialized BackSliceEngine.
        """
        config = config or BackSliceConfig()

        # Build VMA property map without database
        vma_map = VMAPropertyMap.build_from_instructions(
            instructions=instructions,
            arch_name=config.arch_name,
        )

        return cls(
            vma_map=vma_map,
            instructions=instructions,
            cfg=cfg,
            config=config,
        )

    def analyze(self) -> KernelBlameResult:
        """Execute the complete back-slicing analysis pipeline.

        This is the main entry point that coordinates all analysis steps:
        1. Build dependency graph
        2. Apply pruning stages
        3. Compute blame attribution

        Returns:
            KernelBlameResult with blame attribution results.
        """
        # Initialize stats
        self.stats = BackSliceStats()
        self.stats.num_instructions = len(self.instructions)
        self.stats.num_instructions_with_profile = self.vma_map.get_sampled_count()

        # Step 1: Build CCT dependency graph
        self._build_graph()

        # Step 2: Apply pruning pipeline
        self._apply_pruning_pipeline()

        # Step 3: Compute blame attribution
        self._compute_blame()

        # Step 4: Extract multi-hop blame chains (post-processing)
        self._extract_blame_chains()

        return self._result

    def _build_graph(self) -> None:
        """Step 1: Build the CCT dependency graph.

        Creates nodes from VMAPropertyMap entries with profile data,
        and edges from instruction dependencies (assign_pcs).
        """
        if self.config.debug:
            logger.debug("Building CCT dependency graph...")

        # Use the existing build_cct_dep_graph function but without auto-pruning
        # We'll apply pruning stages manually for better control
        self.graph = build_cct_dep_graph(
            vma_map=self.vma_map,
            instructions=self.instructions,
            arch=self.arch,
            apply_opcode_pruning=False,  # We'll do this manually
            apply_latency_pruning=False,  # We'll do this manually
        )

        self.stats.initial_nodes = self.graph.size()
        self.stats.initial_edges = self.graph.edge_count()
        self.stats.sdc_before_pruning = getattr(self.graph, '_sdc_before_sync', self.graph.compute_single_dependency_coverage()[2])

        if self.config.debug:
            logger.debug(f"  Initial graph: {self.stats.initial_nodes} nodes, "
                         f"{self.stats.initial_edges} edges")

    def _apply_pruning_pipeline(self) -> None:
        """Step 2: Apply the 4-stage pruning pipeline.

        Stages:
        1. Opcode constraints - Remove edges violating exec/mem dependency rules
        2. Barrier constraints - Remove edges with mismatched barriers
        3. Latency constraints - Remove edges hidden by pipeline latency
        4. (Branch constraints - not implemented yet)
        """
        if self.config.debug:
            logger.debug("Applying pruning pipeline...")

        # Helper to count mem_waitcnt edges (debug only)
        def _count_mwc():
            if not self.config.debug:
                return 0
            return sum(1 for _e in self.graph.edges()
                       if self.graph.get_edge_type(_e.from_cct_id, _e.to_cct_id) == "mem_waitcnt")

        # Stage 1: Opcode constraints
        if self.config.apply_opcode_pruning:
            self.stats.edges_pruned_opcode = prune_opcode_constraints(
                self.graph, self.vma_map, debug=self.config.debug
            )
            if self.config.debug:
                logger.debug(f"  Stage 1 (opcode): removed {self.stats.edges_pruned_opcode} edges "
                             f"({_count_mwc()} mem_waitcnt remain)")

        # Stage 2: Barrier constraints
        if self.config.apply_barrier_pruning:
            self.stats.edges_pruned_barrier = prune_barrier_constraints(self.graph)
            if self.config.debug:
                logger.debug(f"  Stage 2 (barrier): removed {self.stats.edges_pruned_barrier} edges "
                             f"({_count_mwc()} mem_waitcnt remain)")

        # Stage 3: Latency constraints (requires CFG for path-based analysis)
        if self.config.apply_graph_latency_pruning and self.cfg is not None:
            self.stats.edges_pruned_latency = prune_latency_constraints(
                self.graph, self.arch,
                cfg=self.cfg,
                pc_to_inst=self._pc_to_inst,
            )
            if self.config.debug:
                logger.debug(f"  Stage 3 (latency): removed {self.stats.edges_pruned_latency} edges "
                             f"({_count_mwc()} mem_waitcnt remain)")

        # Stage 4: Execution constraints (disabled by default, matches GPA)
        # Prunes edges where source instruction was never executed
        if self.config.enable_execution_pruning:
            self.stats.edges_pruned_execution = prune_execution_constraints(
                self.graph, self.vma_map
            )
            if self.config.debug:
                logger.debug(f"  Stage 4 (execution): removed {self.stats.edges_pruned_execution} edges")

        self.stats.final_nodes = self.graph.size()
        self.stats.final_edges = self.graph.edge_count()
        self.stats.sdc_after_pruning = self.graph.compute_single_dependency_coverage()[2]

        if self.config.debug:
            logger.debug(f"  Final graph: {self.stats.final_nodes} nodes, "
                  f"{self.stats.final_edges} edges "
                  f"({_count_mwc()} mem_waitcnt)")

    def _compute_blame(self) -> None:
        """Step 3: Compute blame attribution.

        For each stalled instruction:
        1. Get incoming dependencies from graph
        2. Compute distances and efficiencies
        3. Distribute blame using inverse-distance weighting
        """
        if self.config.debug:
            logger.debug("Computing blame attribution...")

        blame_edges: List[BlameEdge] = []

        # Get nodes with stalls
        stalling_nodes = get_stalling_nodes(self.graph, self.config.stall_threshold)
        self.stats.num_stalling_nodes = len(stalling_nodes)

        for to_node in stalling_nodes:
            total_stall = to_node.stall_cycles
            total_lat = to_node.lat_cycles

            if total_stall <= 0 and total_lat <= 0:
                continue

            # Get incoming dependencies
            from_nodes = list(self.graph.incoming_nodes(to_node))

            if not from_nodes:
                # Self-blame: no dependencies found
                edge = self._create_self_blame_edge(to_node)
                blame_edges.append(edge)
                self.stats.num_self_blame += 1
                # Log s_waitcnt self-blame for debugging trace-through issues
                if (self.config.debug and to_node.instruction and
                        to_node.instruction.op.lower().startswith("s_waitcnt")):
                    logger.debug(
                        "Self-blame s_waitcnt: PC=0x%x operands=%s stall=%.0f",
                        to_node.vma, to_node.instruction.operands_raw,
                        total_stall,
                    )
            else:
                # Distribute blame to dependencies
                edges = self._distribute_blame_to_deps(to_node, from_nodes)
                blame_edges.extend(edges)

        self.stats.num_blame_edges = len(blame_edges)

        # Aggregate results (count sources per destination)
        dst_sources: Dict[int, int] = {}
        for edge in blame_edges:
            dst_sources[edge.dst_pc] = dst_sources.get(edge.dst_pc, 0) + 1

        single_source_count = sum(1 for count in dst_sources.values() if count == 1)
        multi_source_count = sum(1 for count in dst_sources.values() if count > 1)

        self._result = _aggregate_blames(
            blame_edges,
            single_source_count,
            multi_source_count,
        )

        if self.config.debug:
            logger.debug(f"  Blame edges: {self.stats.num_blame_edges}")
            logger.debug(f"  Self-blame: {self.stats.num_self_blame}")
            logger.debug(f"  Total stall blame: {self._result.total_stall_blame:.0f}")
            logger.debug(f"  Total lat blame: {self._result.total_lat_blame:.0f}")

    def _extract_blame_chains(self, max_depth: int = 5) -> None:
        """Step 4: Extract multi-hop blame chains from the graph.

        For each top blame edge, walks backward through the dependency graph
        following the highest-blame predecessor at each hop. This reveals the
        full causal path (e.g., s_waitcnt <- global_load <- v_mad_i64).

        This is a post-processing step — it doesn't change blame distribution,
        just provides a deeper view of dependency paths.
        """
        if self._result is None or self.graph is None:
            return

        # Build blame-by-src lookup: src_pc -> total blame as source
        src_blame: Dict[int, float] = {}
        for edge in self._result.blames:
            src_blame[edge.src_pc] = src_blame.get(edge.src_pc, 0.0) + edge.total_blame()

        # Build pc -> node lookup for the graph
        pc_to_node: Dict[int, CCTDepNode] = {}
        for node in self.graph.nodes():
            pc_to_node[node.vma] = node

        # Build pc -> opcode lookup
        pc_to_opcode: Dict[int, str] = {}
        for inst in self.instructions:
            pc_to_opcode[inst.pc] = inst.op

        # Extract chains from top blame edges (deduplicate by dst_pc)
        seen_stall_pcs: set = set()
        chains: List[BlameChain] = []

        # Sort edges by blame descending, take unique stall PCs
        sorted_edges = sorted(self._result.blames, key=lambda e: e.total_blame(), reverse=True)

        for edge in sorted_edges:
            if edge.is_self_blame():
                continue
            if edge.dst_pc in seen_stall_pcs:
                continue
            seen_stall_pcs.add(edge.dst_pc)

            # Build chain starting from the stall node
            chain_nodes = [ChainNode(
                pc=edge.dst_pc,
                opcode=edge.dst_opcode,
                blame=edge.stall_blame + edge.lat_blame,
            )]

            # Walk backward: start from the immediate root cause
            current_pc = edge.src_pc
            visited: set = {edge.dst_pc}

            for _ in range(max_depth):
                if current_pc in visited:
                    break
                visited.add(current_pc)

                chain_nodes.append(ChainNode(
                    pc=current_pc,
                    opcode=pc_to_opcode.get(current_pc, "?"),
                    blame=src_blame.get(current_pc, 0.0),
                ))

                # Find highest-blame predecessor in the graph
                node = pc_to_node.get(current_pc)
                if node is None:
                    break

                best_pred_pc = None
                best_pred_blame = -1.0
                for pred in self.graph.incoming_nodes(node):
                    pred_blame = src_blame.get(pred.vma, 0.0)
                    if pred.vma not in visited and pred_blame > best_pred_blame:
                        best_pred_blame = pred_blame
                        best_pred_pc = pred.vma

                if best_pred_pc is None:
                    break
                current_pc = best_pred_pc

            if len(chain_nodes) > 1:
                chains.append(BlameChain(
                    stall_pc=edge.dst_pc,
                    total_blame=edge.total_blame(),
                    nodes=chain_nodes,
                ))

        self._result.blame_chains = chains

        if self.config.debug:
            multi_hop = sum(1 for c in chains if c.depth > 1)
            logger.debug(f"  Blame chains: {len(chains)} total, {multi_hop} multi-hop (depth > 1)")

    def _create_self_blame_edge(self, node: CCTDepNode) -> BlameEdge:
        """Create a self-blame edge for a node with no dependencies.

        Uses hardware-reported stall breakdown from HPCToolkit to classify
        self-blame into meaningful categories instead of a generic label.
        """
        inst = node.instruction
        opcode = inst.op if inst else "unknown"
        blame_type = self._classify_self_blame(node, inst)

        return BlameEdge(
            src_pc=node.vma,
            dst_pc=node.vma,
            distance=0.0,
            stall_blame=node.stall_cycles,
            lat_blame=node.lat_cycles,
            efficiency=1.0,
            blame_type=blame_type,
            blame_category="self",
            src_opcode=opcode,
            dst_opcode=opcode,
            issue_count=node.issue_count,
            src_operand_details=inst.operand_details if inst else None,
            dst_operand_details=inst.operand_details if inst else None,
        )

    def _classify_self_blame(
        self, node: CCTDepNode, inst: Optional[InstructionStat]
    ) -> str:
        """Classify self-blame using hardware-reported stall breakdown.

        Uses the dominant stall category from HPCToolkit PC sampling metrics
        to produce actionable labels instead of generic "self_scheduler".

        Categories:
        - self_memory_latency: dominant stall is memory (gmem/mem/lmem/tmem)
        - self_compute_saturated: dominant stall is execution dependency (idep)
        - self_sync_overhead: dominant stall is synchronization (sync)
        - self_pipe_busy: dominant stall is pipeline contention (pipe)
        - self_ifetch: dominant stall is instruction fetch (ifet)
        - self_indirect: instruction uses indirect memory addressing
        - self_scheduler: fallback when no stall breakdown available
        """
        # Check indirect flag first (preserved from original logic)
        if inst and inst.indirect:
            return "self_indirect"

        # Look up hardware-reported stall breakdown
        prop = self.vma_map.get(node.vma)
        if not prop:
            return "self_scheduler"

        breakdown = prop.get_stall_breakdown()
        if not breakdown:
            return "self_scheduler"

        # Memory stall categories
        memory_stall = sum(
            breakdown.get(m, 0.0)
            for m in (
                METRIC_GCYCLES_STL_MEM,
                METRIC_GCYCLES_STL_GMEM,
                METRIC_GCYCLES_STL_LMEM,
                METRIC_GCYCLES_STL_TMEM,
            )
        )
        idep_stall = breakdown.get(METRIC_GCYCLES_STL_IDEP, 0.0)
        sync_stall = breakdown.get(METRIC_GCYCLES_STL_SYNC, 0.0)
        pipe_stall = breakdown.get(METRIC_GCYCLES_STL_PIPE, 0.0)
        ifet_stall = breakdown.get(METRIC_GCYCLES_STL_IFET, 0.0)

        # Find the dominant stall category
        categories = {
            "self_memory_latency": memory_stall,
            "self_compute_saturated": idep_stall,
            "self_sync_overhead": sync_stall,
            "self_pipe_busy": pipe_stall,
            "self_ifetch": ifet_stall,
        }

        dominant = max(categories, key=categories.get)  # type: ignore[arg-type]
        if categories[dominant] > 0:
            return dominant

        return "self_scheduler"

    def _compute_stall_match_weights(
        self, to_node: CCTDepNode, from_nodes: List[CCTDepNode]
    ) -> Optional[Dict[int, float]]:
        """Compute stall-category match weights for each incoming edge.

        For each dependency edge, computes how well its blame category
        (MEM_DEP, EXEC_DEP, SYNC) matches the destination instruction's
        hardware-reported stall breakdown. Returns None if no stall
        breakdown is available (disabling this factor).

        Args:
            to_node: The stalled instruction node.
            from_nodes: List of dependency nodes.

        Returns:
            Dict mapping cct_id → match weight (0-1), or None if breakdown
            is unavailable.
        """
        to_prop = self.vma_map.get(to_node.vma)
        if not to_prop:
            return None

        breakdown = to_prop.get_stall_breakdown()
        if not breakdown:
            return None

        total_stalls = sum(breakdown.values())
        if total_stalls <= 0:
            return None

        # Pre-compute stall totals per category
        memory_stalls = sum(
            breakdown.get(m, 0.0)
            for m in (
                METRIC_GCYCLES_STL_MEM,
                METRIC_GCYCLES_STL_GMEM,
                METRIC_GCYCLES_STL_LMEM,
                METRIC_GCYCLES_STL_TMEM,
            )
        )
        exec_stalls = breakdown.get(METRIC_GCYCLES_STL_IDEP, 0.0)
        sync_stalls = breakdown.get(METRIC_GCYCLES_STL_SYNC, 0.0)

        to_inst = to_node.instruction
        weights: Dict[int, float] = {}

        for from_node in from_nodes:
            from_inst = from_node.instruction
            if not from_inst or not to_inst:
                weights[from_node.cct_id] = 1.0
                continue

            category = _determine_blame_category(from_inst, to_inst)

            if category == BlameCategory.MEM_DEP:
                match_stalls = memory_stalls
            elif category == BlameCategory.EXEC_DEP:
                match_stalls = exec_stalls
            elif category == BlameCategory.SYNC:
                match_stalls = sync_stalls
            else:
                weights[from_node.cct_id] = 1.0
                continue

            weights[from_node.cct_id] = match_stalls / total_stalls

        return weights

    def _distribute_blame_to_deps(
        self, to_node: CCTDepNode, from_nodes: List[CCTDepNode]
    ) -> List[BlameEdge]:
        """Distribute blame from a stalled node to its dependencies.

        Uses three-factor weighting:
        - Inverse distance: closer instructions get more blame
        - Inverse efficiency: poorly-optimized instructions get more blame
        - Issue count: frequently-executed instructions get proportional blame

        GPA Reference: blameCCTDepGraph() at GPUAdvisor-Blame.cpp:1185-1256
        Uses CFG path-based distance averaging when available.

        Args:
            to_node: The stalled instruction node.
            from_nodes: List of dependency nodes.

        Returns:
            List of BlameEdge for each dependency.
        """
        edges = []

        # Compute distances using CFG paths when available (GPA-aligned)
        # GPA Reference: GPUAdvisor-Blame.cpp:1193-1198
        distances: Dict[int, float] = {}
        for from_node in from_nodes:
            # Try to get edge with recorded path data from latency pruning
            edge = self.graph.get_edge(from_node.cct_id, to_node.cct_id)
            if edge and edge.has_valid_paths():
                # Use average distance across CFG paths (GPA-aligned)
                dist = edge.average_path_distance()
            else:
                # Fallback to simple VMA-based distance
                dist = compute_distance(from_node.vma, to_node.vma, self.arch.inst_size)
            distances[from_node.cct_id] = max(dist, 1.0)

        # Compute efficiencies
        efficiencies: Dict[int, float] = {}
        for from_node in from_nodes:
            prop = self.vma_map.get(from_node.vma)
            inst = from_node.instruction
            if prop and inst:
                eff = compute_efficiency(inst, prop.prof_metrics)
            else:
                eff = 1.0
            efficiencies[from_node.cct_id] = max(eff, 0.01)

        # Get issue counts
        issue_counts: Dict[int, int] = {}
        for from_node in from_nodes:
            issue_counts[from_node.cct_id] = max(from_node.issue_count, 1)

        # Compute stall-category match weights (4th factor)
        # Each edge gets a weight based on how well its blame category
        # (MEM_DEP/EXEC_DEP/SYNC) matches the destination's stall breakdown.
        stall_match_weights = self._compute_stall_match_weights(
            to_node, from_nodes
        )

        # Distribute blame
        blame_dist = distribute_blame(
            total_stall=to_node.stall_cycles,
            total_lat=to_node.lat_cycles,
            distances=distances,
            efficiencies=efficiencies,
            issue_counts=issue_counts,
            stall_match_weights=stall_match_weights,
        )

        # Create blame edges
        to_inst = to_node.instruction
        to_opcode = to_inst.op if to_inst else "unknown"

        for from_node in from_nodes:
            from_inst = from_node.instruction
            from_opcode = from_inst.op if from_inst else "unknown"

            stall_blame, lat_blame = blame_dist.get(from_node.cct_id, (0.0, 0.0))

            # Determine blame type
            if from_inst and to_inst:
                blame_type, blame_category = self._detailize_blame_type(
                    from_inst, to_inst
                )
            else:
                blame_type = "unknown"
                blame_category = "unknown"

            edge = BlameEdge(
                src_pc=from_node.vma,
                dst_pc=to_node.vma,
                distance=distances[from_node.cct_id],
                stall_blame=stall_blame,
                lat_blame=lat_blame,
                efficiency=efficiencies[from_node.cct_id],
                blame_type=blame_type,
                blame_category=blame_category,
                src_opcode=from_opcode,
                dst_opcode=to_opcode,
                issue_count=issue_counts[from_node.cct_id],
                src_operand_details=from_inst.operand_details if from_inst else None,
                dst_operand_details=to_inst.operand_details if to_inst else None,
            )
            edges.append(edge)

        return edges

    def get_graph(self) -> Optional[CCTDepGraph]:
        """Get the dependency graph (available after analyze())."""
        return self.graph

    def get_stats(self) -> BackSliceStats:
        """Get analysis statistics."""
        return self.stats

    def get_result(self) -> Optional[KernelBlameResult]:
        """Get blame attribution result (available after analyze())."""
        return self._result

    def _detailize_blame_type(
        self, from_inst: InstructionStat, to_inst: InstructionStat
    ) -> Tuple[str, str]:
        """Backward-compatible wrapper for blame type classification."""
        has_reg_dep = any(dst in to_inst.srcs for dst in from_inst.dsts)

        if has_reg_dep:
            category = BlameCategory.EXEC_DEP
        elif from_inst.is_memory_op():
            category = BlameCategory.MEM_DEP
        elif from_inst.is_sync() or to_inst.is_sync():
            category = BlameCategory.SYNC
        else:
            category = BlameCategory.EXEC_DEP

        blame_type = detailize_blame_type(from_inst, to_inst, category)
        return blame_type, category.value


def analyze_kernel(
    db_path: str,
    instructions: List[InstructionStat],
    cfg: Optional[CFG] = None,
    arch_name: str = "a100",
    debug: bool = False,
) -> KernelBlameResult:
    """Convenience function to analyze a kernel.

    This is a simplified entry point for common use cases.

    Args:
        db_path: Path to HPCToolkit database.
        instructions: List of parsed instructions.
        cfg: Optional CFG.
        arch_name: GPU architecture name.
        debug: If True, print debug information.

    Returns:
        KernelBlameResult with blame attribution.
    """
    config = BackSliceConfig(arch_name=arch_name, debug=debug)
    engine = BackSliceEngine.from_database(
        db_path=db_path,
        instructions=instructions,
        cfg=cfg,
        config=config,
    )
    return engine.analyze()


def analyze_instructions(
    instructions: List[InstructionStat],
    vma_map: Optional[VMAPropertyMap] = None,
    cfg: Optional[CFG] = None,
    arch_name: str = "a100",
    debug: bool = False,
) -> KernelBlameResult:
    """Convenience function to analyze instructions without a database.

    Uses VMAPropertyMap if provided, otherwise builds one from instructions.
    Useful for testing or when profile data is simulated.

    Args:
        instructions: List of parsed instructions.
        vma_map: Optional VMAPropertyMap (built from instructions if not provided).
        cfg: Optional CFG.
        arch_name: GPU architecture name.
        debug: If True, print debug information.

    Returns:
        KernelBlameResult with blame attribution.
    """
    config = BackSliceConfig(arch_name=arch_name, debug=debug)

    if vma_map is not None:
        engine = BackSliceEngine(
            vma_map=vma_map,
            instructions=instructions,
            cfg=cfg,
            config=config,
        )
    else:
        engine = BackSliceEngine.from_instructions(
            instructions=instructions,
            cfg=cfg,
            config=config,
        )

    return engine.analyze()


def print_analysis_summary(result: KernelBlameResult, stats: BackSliceStats) -> None:
    """Print a summary of the analysis results.

    Args:
        result: KernelBlameResult from analysis.
        stats: BackSliceStats from analysis.
    """
    logger.info("=" * 60)
    logger.info("BACK-SLICING ANALYSIS SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nInput:")
    logger.info(f"  Instructions: {stats.num_instructions}")
    logger.info(f"  With profile data: {stats.num_instructions_with_profile}")

    logger.info(f"\nGraph:")
    logger.info(f"  Initial: {stats.initial_nodes} nodes, {stats.initial_edges} edges")
    logger.info(f"  Final: {stats.final_nodes} nodes, {stats.final_edges} edges")

    logger.info(f"  Single Dependency Coverage: {stats.sdc_after_pruning:.1%} (before: {stats.sdc_before_pruning:.1%})")

    logger.info(f"\nPruning:")
    logger.info(f"  Opcode constraints: {stats.edges_pruned_opcode} edges removed")
    logger.info(f"  Barrier constraints: {stats.edges_pruned_barrier} edges removed")
    logger.info(f"  Latency constraints: {stats.edges_pruned_latency} edges removed")
    logger.info(f"  Execution constraints: {stats.edges_pruned_execution} edges removed")

    logger.info(f"\nBlame Attribution:")
    logger.info(f"  Stalling nodes: {stats.num_stalling_nodes}")
    logger.info(f"  Blame edges: {stats.num_blame_edges}")
    logger.info(f"  Self-blame: {stats.num_self_blame}")

    total_blame = result.total_stall_blame + result.total_lat_blame
    logger.info(f"\nBlame Totals:")
    logger.info(f"  Stall blame: {result.total_stall_blame:.0f} cycles")
    logger.info(f"  Latency blame: {result.total_lat_blame:.0f} cycles")
    logger.info(f"  Total: {total_blame:.0f} cycles")

    if result.blame_by_category:
        logger.info(f"\nBlame by Category:")
        for cat, blame in sorted(result.blame_by_category.items(),
                                  key=lambda x: -x[1]):
            pct = blame / total_blame * 100 if total_blame > 0 else 0
            logger.info(f"  {cat}: {blame:.0f} cycles ({pct:.1f}%)")

    top_sources = result.get_top_blame_sources(5)
    if top_sources:
        logger.info(f"\nTop Blame Sources:")
        for src_pc, blame in top_sources:
            pct = blame / total_blame * 100 if total_blame > 0 else 0
            logger.info(f"  PC {src_pc:#x}: {blame:.0f} cycles ({pct:.1f}%)")

    logger.info("=" * 60)
