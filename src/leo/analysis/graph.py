"""CCT Dependency Graph for back-slicing analysis.

Based on GPA's CCTGraph<T> and CCTDepGraph from:
- GPUAdvisor-Blame.cpp (graph construction and pruning)
- CCTGraph.hpp (template graph structure)

The CCTDepGraph represents dependencies between CCT (Calling Context Tree)
nodes for blame attribution. It bridges instruction-level dependencies
with profiling data via CCT nodes.

Key concepts:
- Nodes represent instruction instances (from CCT)
- Edges represent data/control dependencies
- Edge direction is backward (from effect to cause)
- Used for computing blame attribution weights
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Iterator, Any, TYPE_CHECKING
from collections import defaultdict

from leo.binary.instruction import (
    InstructionStat,
    BARRIER_NONE,
    build_pc_to_inst_map,
    is_no_barrier,
    is_shared_memory_opcode,
)
from leo.constants.metrics import METRIC_GCYCLES_ISU, METRIC_GCYCLES_LAT, EXECUTION_METRICS
from leo.arch import GPUArchitecture
from leo.analysis.metrics import get_stall_metrics_for_pruning

if TYPE_CHECKING:
    from leo.binary.cfg import CFG

logger = logging.getLogger(__name__)


@dataclass
class CCTDepEdge:
    """Represents a directed dependency edge in the CCT dependency graph.

    An edge from node A to node B means: A's performance affects B's performance.
    This is the inverse of execution flow - we trace backward from stalled
    instructions to their root causes.

    GPA Reference: CCTEdgePathMap stores paths per edge for distance computation.

    Attributes:
        from_cct_id: Source node CCT ID (dependency provider)
        to_cct_id: Destination node CCT ID (stalled instruction)
        edge_type: Type of dependency - "exec_dep", "mem_dep", "sync", "control"
        valid_paths: Paths that are NOT latency-hidden (used for blame)
    """

    from_cct_id: int
    to_cct_id: int
    edge_type: str = "exec_dep"

    # Path storage (GPA-aligned) - populated during latency pruning
    # Note: We use a list here but check for duplicates when adding
    valid_paths: List["EdgePathInfo"] = field(default_factory=list)

    def __hash__(self) -> int:
        """Edges are hashable by their endpoint IDs."""
        return hash((self.from_cct_id, self.to_cct_id))

    def __eq__(self, other: object) -> bool:
        """Edges are equal if they connect same nodes."""
        if not isinstance(other, CCTDepEdge):
            return NotImplemented
        return (
            self.from_cct_id == other.from_cct_id
            and self.to_cct_id == other.to_cct_id
        )

    def has_valid_paths(self) -> bool:
        """Check if this edge has any non-latency-hidden paths."""
        return len(self.valid_paths) > 0

    def average_path_distance(self) -> float:
        """Compute average instruction distance across valid paths.

        GPA Reference: blameCCTDepGraph() at GPUAdvisor-Blame.cpp:1554-1557

        Returns:
            Average instruction count, or -1.0 if no valid paths.
        """
        if not self.valid_paths:
            return -1.0
        total = sum(p.instruction_count for p in self.valid_paths)
        return total / len(self.valid_paths)

    def add_path(self, path: "EdgePathInfo") -> bool:
        """Add a path with GPA-aligned deduplication.

        Args:
            path: EdgePathInfo to add.

        Returns:
            True if path was added, False if duplicate.
        """
        # Check for duplicate (same block sequence)
        for existing in self.valid_paths:
            if existing.block_ids == path.block_ids:
                return False
        self.valid_paths.append(path)
        return True


@dataclass
class CCTDepNode:
    """Represents a CCT node in the dependency graph.

    Wraps a CCT node from the database and caches commonly-accessed
    dependency information to avoid repeated lookups.

    Attributes:
        cct_id: Unique CCT node identifier
        vma: Virtual Memory Address (PC offset within function)
        instruction: Parsed instruction from nvdisasm (may be None)
        stall_cycles: Total stall cycles at this node
        lat_cycles: Latency cycles (execution dependency)
        mem_lat_cycles: Memory latency cycles
        issue_count: How many times the instruction was issued
    """

    cct_id: int
    vma: int
    instruction: Optional[InstructionStat] = None

    # Cached metrics (populated from VMAProperty)
    stall_cycles: float = 0.0
    lat_cycles: float = 0.0
    mem_lat_cycles: float = 0.0
    issue_count: int = 1

    def __hash__(self) -> int:
        """Nodes are hashable by CCT ID."""
        return hash(self.cct_id)

    def __eq__(self, other: object) -> bool:
        """Nodes are equal if they have the same CCT ID."""
        if not isinstance(other, CCTDepNode):
            return NotImplemented
        return self.cct_id == other.cct_id

    def has_stalls(self) -> bool:
        """Check if this node has detectable stall cycles."""
        return self.stall_cycles > 0.0 or self.lat_cycles > 0.0

    def get_total_stalls(self) -> float:
        """Get total stall + latency cycles."""
        return self.stall_cycles + self.lat_cycles

    def is_memory_op(self) -> bool:
        """Check if this node's instruction is a memory operation."""
        if self.instruction:
            return self.instruction.is_memory_op()
        return False

    def is_load(self) -> bool:
        """Check if this node's instruction is a load operation."""
        if self.instruction:
            return self.instruction.is_load()
        return False

    def is_sync(self) -> bool:
        """Check if this node's instruction is a synchronization."""
        if self.instruction:
            return self.instruction.is_sync()
        return False


@dataclass
class DependencyPath:
    """Represents a single path through dependencies for distance calculation."""

    nodes: List[CCTDepNode] = field(default_factory=list)
    distance: float = 0.0
    latency_hidden: bool = False

    def length(self) -> int:
        """Number of nodes in path."""
        return len(self.nodes)


@dataclass
class EdgePathInfo:
    """Stores path information for an edge (GPA-aligned).

    GPA Reference: CCTEdgePathMap in GPUAdvisor.hpp:142

    Represents a single CFG path between two instructions. Multiple
    EdgePathInfo objects can exist per edge (different control flow paths).

    Attributes:
        block_ids: IDs of CFG blocks traversed (for comparison/deduplication)
        accumulated_stalls: Total stall cycles along this path
        instruction_count: Number of instructions in path (for distance)
        is_hidden: True if latency is hidden on this path
    """

    block_ids: Tuple[int, ...] = field(default_factory=tuple)
    accumulated_stalls: int = 0
    instruction_count: float = 0.0
    is_hidden: bool = False

    def __eq__(self, other: object) -> bool:
        """Paths are equal if they have same block sequence (for deduplication)."""
        if not isinstance(other, EdgePathInfo):
            return NotImplemented
        return self.block_ids == other.block_ids

    def __hash__(self) -> int:
        """Hash by block sequence for set operations."""
        return hash(self.block_ids)


@dataclass
class GraphStatistics:
    """Aggregated statistics about the dependency graph."""

    num_nodes: int = 0
    num_edges: int = 0
    num_nodes_with_stalls: int = 0
    num_source_nodes: int = 0  # Nodes with no incoming edges
    num_sink_nodes: int = 0  # Nodes with no outgoing edges
    total_stall_cycles: float = 0.0
    total_lat_cycles: float = 0.0
    avg_fan_in: float = 0.0
    avg_fan_out: float = 0.0
    max_fan_in: int = 0
    max_fan_out: int = 0
    edge_types: Dict[str, int] = field(default_factory=dict)


class CCTDepGraph:
    """Represents the CCT dependency graph for back-slicing analysis.

    This is a directed graph where:
    - Nodes represent instruction instances (from CCT)
    - Edges represent data/control dependencies
    - Edge direction is backward (from effect to cause)

    Key operations:
    - Add nodes and edges during graph construction
    - Query incoming/outgoing nodes
    - Traverse for blame attribution
    - Compute statistics

    Example:
        graph = CCTDepGraph()
        node1 = CCTDepNode(cct_id=1, vma=0x100)
        node2 = CCTDepNode(cct_id=2, vma=0x110)
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(node1, node2, "exec_dep")

        # Query dependencies
        for pred in graph.incoming_nodes(node2):
            print(f"Node {node2.cct_id} depends on {pred.cct_id}")
    """

    def __init__(self):
        """Initialize an empty graph."""
        self._nodes: Dict[int, CCTDepNode] = {}  # cct_id -> node
        self._edges: Set[CCTDepEdge] = set()
        self._incoming: Dict[int, Set[int]] = defaultdict(set)  # cct_id -> predecessors
        self._outgoing: Dict[int, Set[int]] = defaultdict(set)  # cct_id -> successors
        self._edge_types: Dict[Tuple[int, int], str] = {}  # (from, to) -> type

    def add_node(self, node: CCTDepNode) -> None:
        """Add a node to the graph.

        If the node already exists (by cct_id), it will be updated.

        Args:
            node: CCTDepNode to add.
        """
        self._nodes[node.cct_id] = node
        # Ensure adjacency entries exist
        if node.cct_id not in self._incoming:
            self._incoming[node.cct_id] = set()
        if node.cct_id not in self._outgoing:
            self._outgoing[node.cct_id] = set()

    def add_edge(
        self, from_node: CCTDepNode, to_node: CCTDepNode, edge_type: str = "exec_dep"
    ) -> None:
        """Add a directed edge from from_node to to_node.

        Both nodes are added to the graph if not already present.
        Duplicate edges are ignored.

        Args:
            from_node: Source node (dependency provider).
            to_node: Destination node (depends on from_node).
            edge_type: Type of dependency ("exec_dep", "mem_dep", "sync", "control").
        """
        # Ensure both nodes are in graph
        self.add_node(from_node)
        self.add_node(to_node)

        # Skip self-loops
        if from_node.cct_id == to_node.cct_id:
            return

        edge = CCTDepEdge(from_node.cct_id, to_node.cct_id, edge_type)
        if edge not in self._edges:
            self._edges.add(edge)
            self._incoming[to_node.cct_id].add(from_node.cct_id)
            self._outgoing[from_node.cct_id].add(to_node.cct_id)
            self._edge_types[(from_node.cct_id, to_node.cct_id)] = edge_type

    def add_edge_by_id(
        self, from_cct_id: int, to_cct_id: int, edge_type: str = "exec_dep"
    ) -> bool:
        """Add an edge by CCT IDs.

        Args:
            from_cct_id: Source node CCT ID.
            to_cct_id: Destination node CCT ID.
            edge_type: Type of dependency.

        Returns:
            True if edge was added, False if nodes don't exist.
        """
        from_node = self._nodes.get(from_cct_id)
        to_node = self._nodes.get(to_cct_id)
        if from_node is None or to_node is None:
            return False
        self.add_edge(from_node, to_node, edge_type)
        return True

    def remove_edge(self, from_node: CCTDepNode, to_node: CCTDepNode) -> bool:
        """Remove an edge between two nodes.

        Args:
            from_node: Source node.
            to_node: Destination node.

        Returns:
            True if edge existed and was removed.
        """
        return self.remove_edge_by_id(from_node.cct_id, to_node.cct_id)

    def remove_edge_by_id(self, from_cct_id: int, to_cct_id: int) -> bool:
        """Remove an edge by CCT IDs.

        Args:
            from_cct_id: Source node CCT ID.
            to_cct_id: Destination node CCT ID.

        Returns:
            True if edge existed and was removed.
        """
        edge = CCTDepEdge(from_cct_id, to_cct_id)
        if edge in self._edges:
            self._edges.discard(edge)
            self._incoming[to_cct_id].discard(from_cct_id)
            self._outgoing[from_cct_id].discard(to_cct_id)
            self._edge_types.pop((from_cct_id, to_cct_id), None)
            return True
        return False

    def has_edge(self, from_cct_id: int, to_cct_id: int) -> bool:
        """Check if an edge exists between two nodes."""
        return (from_cct_id, to_cct_id) in self._edge_types

    def get_edge_type(self, from_cct_id: int, to_cct_id: int) -> Optional[str]:
        """Get the type of an edge."""
        return self._edge_types.get((from_cct_id, to_cct_id))

    def get_edge(self, from_cct_id: int, to_cct_id: int) -> Optional[CCTDepEdge]:
        """Get the edge object between two nodes.

        GPA Reference: Used for accessing edge.valid_paths for distance computation.

        Args:
            from_cct_id: Source node CCT ID.
            to_cct_id: Destination node CCT ID.

        Returns:
            CCTDepEdge if exists, None otherwise.
        """
        if not self.has_edge(from_cct_id, to_cct_id):
            return None

        # Find the actual edge object (with path data)
        for edge in self._edges:
            if edge.from_cct_id == from_cct_id and edge.to_cct_id == to_cct_id:
                return edge
        return None

    def get_node(self, cct_id: int) -> Optional[CCTDepNode]:
        """Get a node by CCT ID.

        Args:
            cct_id: CCT node identifier.

        Returns:
            CCTDepNode or None if not found.
        """
        return self._nodes.get(cct_id)

    def get_node_by_vma(self, vma: int) -> Optional[CCTDepNode]:
        """Get a node by VMA (program counter).

        Note: This is O(n). Use get_node(cct_id) if you have the CCT ID.

        Args:
            vma: Virtual memory address.

        Returns:
            First matching CCTDepNode or None.
        """
        for node in self._nodes.values():
            if node.vma == vma:
                return node
        return None

    def incoming_nodes(self, node: CCTDepNode) -> Iterator[CCTDepNode]:
        """Get all nodes with edges pointing to this node (predecessors).

        Args:
            node: Target node.

        Yields:
            CCTDepNode instances that have edges to the target.
        """
        for cct_id in self._incoming.get(node.cct_id, set()):
            pred = self._nodes.get(cct_id)
            if pred is not None:
                yield pred

    def outgoing_nodes(self, node: CCTDepNode) -> Iterator[CCTDepNode]:
        """Get all nodes this node points to (successors).

        Args:
            node: Source node.

        Yields:
            CCTDepNode instances that the source has edges to.
        """
        for cct_id in self._outgoing.get(node.cct_id, set()):
            succ = self._nodes.get(cct_id)
            if succ is not None:
                yield succ

    def incoming_node_ids(self, cct_id: int) -> Set[int]:
        """Get CCT IDs of nodes with edges pointing to this node."""
        return self._incoming.get(cct_id, set()).copy()

    def outgoing_node_ids(self, cct_id: int) -> Set[int]:
        """Get CCT IDs of nodes this node points to."""
        return self._outgoing.get(cct_id, set()).copy()

    def in_degree(self, cct_id: int) -> int:
        """Get number of incoming edges for a node."""
        return len(self._incoming.get(cct_id, set()))

    def out_degree(self, cct_id: int) -> int:
        """Get number of outgoing edges for a node."""
        return len(self._outgoing.get(cct_id, set()))

    def nodes(self) -> Iterator[CCTDepNode]:
        """Iterate over all nodes in the graph."""
        return iter(self._nodes.values())

    def edges(self) -> Iterator[CCTDepEdge]:
        """Iterate over all edges in the graph."""
        return iter(self._edges)

    def node_ids(self) -> Set[int]:
        """Get set of all CCT IDs in the graph."""
        return set(self._nodes.keys())

    def size(self) -> int:
        """Number of nodes in graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Number of edges in graph."""
        return len(self._edges)

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._incoming.clear()
        self._outgoing.clear()
        self._edge_types.clear()

    def get_statistics(self) -> GraphStatistics:
        """Compute summary statistics about the graph.

        Returns:
            GraphStatistics with node/edge counts, stall totals, etc.
        """
        stats = GraphStatistics()
        stats.num_nodes = len(self._nodes)
        stats.num_edges = len(self._edges)

        if stats.num_nodes == 0:
            return stats

        # Edge type counts
        for edge in self._edges:
            edge_type = edge.edge_type
            stats.edge_types[edge_type] = stats.edge_types.get(edge_type, 0) + 1

        # Node statistics
        total_in = 0
        total_out = 0
        for node in self._nodes.values():
            if node.has_stalls():
                stats.num_nodes_with_stalls += 1
            stats.total_stall_cycles += node.stall_cycles
            stats.total_lat_cycles += node.lat_cycles

            in_deg = len(self._incoming.get(node.cct_id, set()))
            out_deg = len(self._outgoing.get(node.cct_id, set()))

            if in_deg == 0:
                stats.num_source_nodes += 1
            if out_deg == 0:
                stats.num_sink_nodes += 1

            stats.max_fan_in = max(stats.max_fan_in, in_deg)
            stats.max_fan_out = max(stats.max_fan_out, out_deg)
            total_in += in_deg
            total_out += out_deg

        stats.avg_fan_in = total_in / stats.num_nodes
        stats.avg_fan_out = total_out / stats.num_nodes

        return stats

    def compute_single_dependency_coverage(self) -> tuple[int, int, float]:
        """Compute GPA-style single dependency coverage.

        A node is "single dependency" if it has at most one incoming edge
        per dependency category (memory vs execution). This metric validates
        that pruning produces clean graphs suitable for unambiguous blame.

        Memory category: mem_dep, mem_waitcnt, mem_barrier, mem_swsb
        Execution category: exec_dep, sync, control

        Returns:
            (single_dep_count, total_count, coverage_ratio)
        """
        MEM_TYPES = {"mem_dep", "mem_waitcnt", "mem_barrier", "mem_swsb"}

        single_count = 0
        total = len(self._nodes)

        if total == 0:
            return (0, 0, 0.0)

        for cct_id in self._nodes:
            pred_ids = self._incoming.get(cct_id, set())
            mem_count = 0
            exec_count = 0
            for pred_id in pred_ids:
                edge_type = self._edge_types.get((pred_id, cct_id), "exec_dep")
                if edge_type in MEM_TYPES:
                    mem_count += 1
                else:
                    exec_count += 1
            if mem_count <= 1 and exec_count <= 1:
                single_count += 1

        return (single_count, total, single_count / total)

def compute_backward_slice(
    root_node: CCTDepNode, graph: CCTDepGraph, max_depth: int = 100
) -> Set[CCTDepNode]:
    """Find all nodes that contribute to a given node via backward slice.

    Uses BFS to explore nearest contributors first.

    Args:
        root_node: Starting node (stalled instruction).
        graph: CCTDepGraph to traverse.
        max_depth: Maximum traversal depth to prevent infinite loops.

    Returns:
        Set of all nodes in the backward slice (including root).
    """
    visited: Set[int] = set()
    result: Set[CCTDepNode] = set()
    queue: List[Tuple[CCTDepNode, int]] = [(root_node, 0)]

    while queue:
        node, depth = queue.pop(0)

        if node.cct_id in visited or depth > max_depth:
            continue

        visited.add(node.cct_id)
        result.add(node)

        # Add predecessors
        for pred in graph.incoming_nodes(node):
            if pred.cct_id not in visited:
                queue.append((pred, depth + 1))

    return result


def compute_forward_slice(
    root_node: CCTDepNode, graph: CCTDepGraph, max_depth: int = 100
) -> Set[CCTDepNode]:
    """Find all nodes affected by a given node via forward slice.

    Args:
        root_node: Starting node.
        graph: CCTDepGraph to traverse.
        max_depth: Maximum traversal depth.

    Returns:
        Set of all nodes in the forward slice (including root).
    """
    visited: Set[int] = set()
    result: Set[CCTDepNode] = set()
    queue: List[Tuple[CCTDepNode, int]] = [(root_node, 0)]

    while queue:
        node, depth = queue.pop(0)

        if node.cct_id in visited or depth > max_depth:
            continue

        visited.add(node.cct_id)
        result.add(node)

        # Add successors
        for succ in graph.outgoing_nodes(node):
            if succ.cct_id not in visited:
                queue.append((succ, depth + 1))

    return result


def compute_distance(
    from_vma: int,
    to_vma: int,
    inst_size: int = 16,
) -> float:
    """Compute instruction distance from from_vma to to_vma.

    Delegates to the canonical implementation in leo.analysis.blame.
    """
    from leo.analysis.blame import compute_distance as _compute_distance

    return _compute_distance(from_vma, to_vma, inst_size)


# Counter-to-opcode mapping for s_waitcnt memory dependency tracing
# vmcnt: Vector memory operations (global, flat, buffer, scratch)
VMCNT_OPCODE_PREFIXES = ("global_", "flat_", "buffer_", "scratch_")
# lgkmcnt: LDS/GDS/Constant/Message operations
LGKMCNT_OPCODE_PREFIXES = ("ds_", "s_load_", "s_buffer_")


def find_waitcnt_memory_deps(
    waitcnt_inst: InstructionStat,
    instructions: List[InstructionStat],
    waitcnt_index: int,
) -> List[int]:
    """Find memory operations that a s_waitcnt instruction is waiting for.

    This enables tracing through s_waitcnt to find the actual memory operations
    causing stalls, instead of self-blaming the s_waitcnt instruction.

    Args:
        waitcnt_inst: The s_waitcnt instruction with operand_details.
        instructions: All instructions in the function (sorted by PC).
        waitcnt_index: Index of waitcnt_inst in instructions list.

    Returns:
        List of PCs for memory operations this s_waitcnt depends on.

    Example:
        s_waitcnt vmcnt(0) → returns PCs of all pending global_load/buffer_load ops
        s_waitcnt vmcnt(1) → returns PCs of all but 1 most recent memory ops
    """
    if not waitcnt_inst.operand_details:
        return []

    dep_pcs: List[int] = []

    # Process each counter type in operand_details
    for counter_name, counter_value in waitcnt_inst.operand_details.items():
        if counter_name == "vmcnt":
            prefixes = VMCNT_OPCODE_PREFIXES
        elif counter_name == "lgkmcnt":
            prefixes = LGKMCNT_OPCODE_PREFIXES
        elif counter_name == "combined":
            # Legacy s_waitcnt 0 format - wait on all counters
            prefixes = VMCNT_OPCODE_PREFIXES + LGKMCNT_OPCODE_PREFIXES
            counter_value = 0
        else:
            # expcnt, vscnt - skip for now
            continue

        # Scan backward from s_waitcnt to find matching memory ops
        # counter_value=0 means wait for ALL ops, so we find all of them
        # counter_value=N means wait until N or fewer are pending,
        # so we find the (total - N) oldest ops that must complete
        #
        # Epoch-aware: stop scanning at a previous s_waitcnt that drains
        # the same counter to 0 (or to a value <= current counter_value),
        # since those older ops were already waited on.
        matching_ops: List[int] = []
        for i in range(waitcnt_index - 1, -1, -1):
            inst = instructions[i]
            op_lower = inst.op.lower()

            # Stop at a previous s_waitcnt that already drained this counter
            if op_lower.startswith("s_waitcnt") and inst.operand_details:
                prev_counter = inst.operand_details.get(counter_name)
                if prev_counter is not None and prev_counter <= counter_value:
                    break
                # Also stop at combined=0 (legacy s_waitcnt 0)
                if "combined" in inst.operand_details and inst.operand_details["combined"] == 0:
                    break

            # Check if this instruction matches the counter type
            if any(op_lower.startswith(prefix) for prefix in prefixes):
                matching_ops.append(inst.pc)

        # For vmcnt(N), the s_waitcnt waits until only N ops remain pending
        # So we need to include all ops except the N most recent ones
        # If we have M ops total, we wait for (M - N) oldest ops to complete
        if counter_value == 0:
            # vmcnt(0) = wait for ALL pending ops
            dep_pcs.extend(matching_ops)
        elif len(matching_ops) > counter_value:
            # vmcnt(N) with N > 0: only the oldest (M - N) ops must complete
            # The N most recent ops (matching_ops[:N]) can remain pending
            dep_pcs.extend(matching_ops[counter_value:])

    return dep_pcs


def find_barrier_memory_deps(
    inst: InstructionStat,
    instructions: List[InstructionStat],
    inst_index: int,
) -> List[int]:
    """Find instructions that set barriers this instruction waits on (NVIDIA).

    NVIDIA uses hardware scoreboards (B1-B6) for synchronization:
    - Memory operations (LDG, STG, etc.) SET a barrier via Control.write
    - Consuming instructions WAIT on barriers via Control.wait bitmask
    - DEPBAR instructions explicitly wait on scoreboards

    This traces through barrier synchronization to find the actual memory
    operations causing stalls, analogous to AMD's find_waitcnt_memory_deps().

    Args:
        inst: Instruction with Control.wait bits or DEPBAR operands.
        instructions: All instructions in the function (sorted by PC).
        inst_index: Index of inst in instructions list.

    Returns:
        List of PCs for instructions that set matching barriers.
    """
    # Collect barrier IDs this instruction waits on
    wait_barriers: List[int] = []

    # From Control.wait bitmask (set by hardware scheduler)
    if inst.control and inst.control.wait:
        wait_barriers.extend(inst.control.get_wait_barriers())

    # From DEPBAR operand (explicit scoreboard wait)
    base_op = inst.op.split(".")[0].upper()
    if base_op == "DEPBAR" and inst.bdsts:
        for barrier_id in inst.bdsts:
            if barrier_id not in wait_barriers and 1 <= barrier_id <= 6:
                wait_barriers.append(barrier_id)

    if not wait_barriers:
        return []

    # Scan backward to find instructions that WROTE to these barriers
    dep_pcs: List[int] = []
    found_barriers: set = set()

    for i in range(inst_index - 1, -1, -1):
        prev_inst = instructions[i]
        if not prev_inst.control:
            continue

        write_barrier = prev_inst.control.write
        if is_no_barrier(write_barrier):
            continue

        if write_barrier in wait_barriers and write_barrier not in found_barriers:
            dep_pcs.append(prev_inst.pc)
            found_barriers.add(write_barrier)

            # Stop when all waited barriers are found
            if len(found_barriers) == len(wait_barriers):
                break

    return dep_pcs


def find_swsb_memory_deps(
    inst: InstructionStat,
    instructions: List[InstructionStat],
    inst_index: int,
) -> List[int]:
    """Find send instructions that an SWSB wait is waiting for (Intel).

    Intel uses SWSB tokens (SBID 0-31) for out-of-order synchronization:
    - send instructions with SBID 'set $T' are producers
    - Instructions with SBID 'dst_wait $T' / 'src_wait $T' are consumers
    - The explicit 'wait' opcode also serves as a synchronization point

    This traces through SWSB tokens to find actual memory operations
    causing stalls, analogous to AMD's find_waitcnt_memory_deps().

    Args:
        inst: Instruction with SWSB wait token or explicit wait opcode.
        instructions: All instructions in the function (sorted by PC).
        inst_index: Index of inst in instructions list.

    Returns:
        List of PCs for send instructions this wait depends on.
    """
    dep_pcs: List[int] = []

    # Case 1: Instruction has explicit SBID wait (dst_wait or src_wait)
    if inst.swsb and inst.swsb.has_sbid and inst.swsb.sbid_type in ("dst_wait", "src_wait"):
        target_token = inst.swsb.sbid
        # Scan backward for the nearest 'set $T' with matching token
        for i in range(inst_index - 1, -1, -1):
            prev = instructions[i]
            if (prev.swsb and prev.swsb.has_sbid
                    and prev.swsb.sbid_type == "set"
                    and prev.swsb.sbid == target_token):
                dep_pcs.append(prev.pc)
                break  # Only the most recent set for this token

    # Case 2: Explicit 'wait' opcode — drain all pending SBID tokens
    elif inst.op.lower() == "wait":
        # Track which tokens have been consumed by intervening waits
        consumed_tokens: set = set()
        for i in range(inst_index - 1, -1, -1):
            prev = instructions[i]
            if prev.swsb and prev.swsb.has_sbid:
                if prev.swsb.sbid_type in ("dst_wait", "src_wait"):
                    # This token was already consumed by an earlier wait
                    consumed_tokens.add(prev.swsb.sbid)
                elif prev.swsb.sbid_type == "set" and prev.swsb.sbid not in consumed_tokens:
                    dep_pcs.append(prev.pc)
                    consumed_tokens.add(prev.swsb.sbid)
            # Stop at previous explicit wait (epoch boundary)
            if prev.op.lower() == "wait" and i != inst_index:
                break

    return dep_pcs


def classify_edge_type(
    from_inst: InstructionStat, to_inst: InstructionStat
) -> str:
    """Classify the type of dependency edge between two instructions.

    Args:
        from_inst: Source instruction (dependency provider).
        to_inst: Destination instruction (depends on source).

    Returns:
        Edge type string: "exec_dep", "mem_dep", "sync", or "control".
    """
    # Check for synchronization
    if from_inst.is_sync() or to_inst.is_sync():
        return "sync"

    # Check for memory operation
    if from_inst.is_memory_op() and from_inst.is_load():
        return "mem_dep"

    # Check for control flow (branch)
    if from_inst.is_branch():
        return "control"

    # Default: execution dependency (register RAW)
    return "exec_dep"


def build_cct_dep_graph(
    vma_map: Any,  # VMAPropertyMap
    instructions: List[InstructionStat],
    arch: Optional[GPUArchitecture] = None,
    apply_opcode_pruning: bool = True,
    apply_latency_pruning: bool = True,
    cfg: Optional["CFG"] = None,
) -> CCTDepGraph:
    """Build CCT dependency graph from VMA property map and instructions.

    This is the main factory function that:
    1. Creates nodes from VMAPropertyMap entries with profile data
    2. Builds edges from instruction dependencies (assign_pcs)
    3. Optionally prunes edges based on constraints

    Args:
        vma_map: VMAPropertyMap with joined binary + profile data.
        instructions: List of InstructionStat with populated assign_pcs.
        arch: Optional GPU architecture for latency-based pruning.
        apply_opcode_pruning: If True, remove edges violating opcode constraints.
        apply_latency_pruning: If True, remove edges hidden by pipeline latency.

    Returns:
        Populated CCTDepGraph.
    """
    graph = CCTDepGraph()

    # Index instructions by PC for O(1) lookup
    pc_to_inst: Dict[int, InstructionStat] = build_pc_to_inst_map(instructions)

    # Step 1: Create nodes from vma_map
    pc_to_node: Dict[int, CCTDepNode] = {}
    for pc, prop in vma_map.items():
        if prop.has_profile_data and prop.cct_id >= 0:
            node = CCTDepNode(
                cct_id=prop.cct_id,
                vma=pc,
                instruction=prop.instruction,
                stall_cycles=prop.stall_cycles,
                lat_cycles=prop.prof_metrics.get(METRIC_GCYCLES_LAT, 0.0),
                mem_lat_cycles=prop.memory_stall_cycles,
                issue_count=max(1, int(prop.prof_metrics.get(METRIC_GCYCLES_ISU, 1))),
            )
            graph.add_node(node)
            pc_to_node[pc] = node

    # Step 2: Build edges from instruction dependencies
    # Track next available CCT ID for nodes without profile data
    next_virtual_cct_id = -1000

    for to_inst in instructions:
        to_node = pc_to_node.get(to_inst.pc)
        if to_node is None:
            continue

        # Get all dependency PCs from assign_pcs maps
        dep_pcs = to_inst.get_all_dependencies()
        for from_pc in dep_pcs:
            from_inst = pc_to_inst.get(from_pc)
            if from_inst is None:
                continue

            from_node = pc_to_node.get(from_pc)
            if from_node is None:
                # Create a node for dependency instruction without profile data
                # This allows root cause instructions (like address calculations)
                # to receive blame even if they don't stall themselves
                from_node = CCTDepNode(
                    cct_id=next_virtual_cct_id,
                    vma=from_pc,
                    instruction=from_inst,
                    stall_cycles=0,  # No profile data
                    lat_cycles=0,
                    mem_lat_cycles=0,
                    issue_count=1,
                )
                graph.add_node(from_node)
                pc_to_node[from_pc] = from_node
                next_virtual_cct_id -= 1

            # Classify edge type
            edge_type = classify_edge_type(from_inst, to_inst)
            graph.add_edge(from_node, to_node, edge_type)

    # Compute SDC baseline before vendor-specific sync tracing
    graph._sdc_before_sync = graph.compute_single_dependency_coverage()[2]

    # Step 2b: Build memory counter edges for s_waitcnt instructions
    # This traces s_waitcnt → actual memory operations (global_load, ds_read, etc.)
    # instead of leaving s_waitcnt as self-blame dead ends
    _dbg_waitcnt_total = 0
    _dbg_waitcnt_no_node = 0
    _dbg_waitcnt_no_deps = 0
    _dbg_waitcnt_edges = 0
    for i, to_inst in enumerate(instructions):
        # Only process s_waitcnt instructions
        if not to_inst.op.lower().startswith("s_waitcnt"):
            continue
        _dbg_waitcnt_total += 1

        to_node = pc_to_node.get(to_inst.pc)
        if to_node is None:
            _dbg_waitcnt_no_node += 1
            continue

        # Find memory operations this s_waitcnt is waiting for
        mem_dep_pcs = find_waitcnt_memory_deps(to_inst, instructions, i)
        if not mem_dep_pcs:
            _dbg_waitcnt_no_deps += 1
            logger.debug(
                "s_waitcnt at PC 0x%x (%s) found NO memory deps "
                "(operand_details=%s, index=%d/%d)",
                to_inst.pc, to_inst.operands_raw, to_inst.operand_details, i, len(instructions)
            )

        for from_pc in mem_dep_pcs:
            from_inst = pc_to_inst.get(from_pc)
            if from_inst is None:
                continue

            from_node = pc_to_node.get(from_pc)
            if from_node is None:
                # Create virtual node for memory op without profile data
                from_node = CCTDepNode(
                    cct_id=next_virtual_cct_id,
                    vma=from_pc,
                    instruction=from_inst,
                    stall_cycles=0,
                    lat_cycles=0,
                    mem_lat_cycles=0,
                    issue_count=1,
                )
                graph.add_node(from_node)
                pc_to_node[from_pc] = from_node
                next_virtual_cct_id -= 1

            # Create memory counter dependency edge
            graph.add_edge(from_node, to_node, "mem_waitcnt")
            _dbg_waitcnt_edges += 1

    if _dbg_waitcnt_total > 0:
        logger.debug(
            "s_waitcnt trace: %d total, %d no_node, %d no_deps, %d edges created",
            _dbg_waitcnt_total, _dbg_waitcnt_no_node, _dbg_waitcnt_no_deps, _dbg_waitcnt_edges
        )

    # Step 2c: Build barrier-based dependency edges for NVIDIA instructions
    # NVIDIA uses hardware scoreboards (B1-B6) encoded in Control bits:
    # - Memory ops SET a barrier via Control.write (e.g., LDG sets B1)
    # - Consuming instructions WAIT on barriers via Control.wait bitmask
    # - DEPBAR instructions explicitly wait on scoreboards
    # This traces through barrier synchronization to find actual memory ops,
    # analogous to AMD's s_waitcnt → memory op linking above.
    vendor = getattr(arch, 'vendor', None) if arch else None
    if vendor == 'nvidia':
        for i, to_inst in enumerate(instructions):
            # Check if instruction waits on any barrier
            has_wait = (to_inst.control and to_inst.control.wait)
            is_depbar = to_inst.op.split(".")[0].upper() == "DEPBAR"
            if not has_wait and not is_depbar:
                continue

            to_node = pc_to_node.get(to_inst.pc)
            if to_node is None:
                continue

            barrier_dep_pcs = find_barrier_memory_deps(to_inst, instructions, i)

            for from_pc in barrier_dep_pcs:
                from_inst = pc_to_inst.get(from_pc)
                if from_inst is None:
                    continue

                from_node = pc_to_node.get(from_pc)
                if from_node is None:
                    from_node = CCTDepNode(
                        cct_id=next_virtual_cct_id,
                        vma=from_pc,
                        instruction=from_inst,
                        stall_cycles=0,
                        lat_cycles=0,
                        mem_lat_cycles=0,
                        issue_count=1,
                    )
                    graph.add_node(from_node)
                    pc_to_node[from_pc] = from_node
                    next_virtual_cct_id -= 1

                graph.add_edge(from_node, to_node, "mem_barrier")

    # Step 2d: Build SWSB-based dependency edges for Intel instructions
    # Intel uses SWSB tokens (SBID 0-31) for synchronization:
    # - send set $T → allocates token (producer)
    # - dst_wait $T / src_wait $T → waits for token (consumer)
    # - explicit 'wait' opcode → drains all pending tokens
    # This traces through SWSB tokens to find actual memory ops,
    # analogous to AMD's s_waitcnt and NVIDIA's barrier linking above.
    if vendor == 'intel':
        _dbg_swsb_total = 0
        _dbg_swsb_edges = 0
        for i, to_inst in enumerate(instructions):
            # Check if instruction has SWSB wait or is explicit wait opcode
            has_swsb_wait = (to_inst.swsb and to_inst.swsb.has_sbid
                             and to_inst.swsb.sbid_type in ("dst_wait", "src_wait"))
            is_wait_op = to_inst.op.lower() == "wait"
            if not has_swsb_wait and not is_wait_op:
                continue
            _dbg_swsb_total += 1

            to_node = pc_to_node.get(to_inst.pc)
            if to_node is None:
                continue

            swsb_dep_pcs = find_swsb_memory_deps(to_inst, instructions, i)

            for from_pc in swsb_dep_pcs:
                from_inst = pc_to_inst.get(from_pc)
                if from_inst is None:
                    continue

                from_node = pc_to_node.get(from_pc)
                if from_node is None:
                    from_node = CCTDepNode(
                        cct_id=next_virtual_cct_id,
                        vma=from_pc,
                        instruction=from_inst,
                        stall_cycles=0,
                        lat_cycles=0,
                        mem_lat_cycles=0,
                        issue_count=1,
                    )
                    graph.add_node(from_node)
                    pc_to_node[from_pc] = from_node
                    next_virtual_cct_id -= 1

                graph.add_edge(from_node, to_node, "mem_swsb")
                _dbg_swsb_edges += 1

        if _dbg_swsb_total > 0:
            logger.debug(
                "Intel SWSB trace: %d wait instructions, %d mem_swsb edges created",
                _dbg_swsb_total, _dbg_swsb_edges
            )

    # Step 3: Apply pruning constraints
    if apply_opcode_pruning:
        prune_opcode_constraints(graph, vma_map)

    if apply_latency_pruning and arch is not None and cfg is not None:
        prune_latency_constraints(graph, arch, cfg, pc_to_inst)

    return graph


def prune_opcode_constraints(graph: CCTDepGraph, vma_map: Any,
                             debug: bool = False) -> int:
    """Remove edges that violate opcode constraints.

    Based on GPA's pruneCCTDepGraphOpcode:
    - Shared memory loads only cause execution dependencies
    - Global memory loads cause memory dependencies
    - Other instructions cause execution dependencies

    Args:
        graph: CCTDepGraph to prune (modified in-place).
        vma_map: VMAPropertyMap for metric lookup.
        debug: If True, print detailed pruning diagnostics.

    Returns:
        Number of edges removed.
    """
    to_remove: List[Tuple[int, int]] = []

    # Debug counters
    dbg_total = 0
    dbg_no_node = 0
    dbg_no_inst = 0
    dbg_no_prop = 0
    dbg_no_samples = 0
    dbg_fallback = 0
    dbg_is_mem = 0
    dbg_is_shared = 0
    dbg_is_compute = 0
    dbg_opcode_freq: Dict[str, int] = {}
    dbg_examples: List[str] = []

    for edge in graph.edges():
        dbg_total += 1

        # Skip opcode pruning for explicitly-constructed memory dependency
        # edges — these are traced from hardware synchronization primitives
        # and represent real dependencies. Opcode pruning's metric-based
        # heuristics (exec_dep vs mem_dep) don't apply to them:
        # - mem_waitcnt: AMD s_waitcnt → memory ops (LDS stalls classified
        #   under gcycles:stl:mem, not gcycles:stl:idep)
        # - mem_barrier: NVIDIA scoreboard barrier → memory ops (Control.wait
        #   → Control.write linkage)
        # - mem_swsb: Intel SWSB token → send ops (SBID set/wait linkage)
        edge_type = graph.get_edge_type(edge.from_cct_id, edge.to_cct_id)
        if edge_type in ("mem_waitcnt", "mem_barrier", "mem_swsb"):
            continue

        from_node = graph.get_node(edge.from_cct_id)
        to_node = graph.get_node(edge.to_cct_id)

        if from_node is None or to_node is None:
            dbg_no_node += 1
            continue

        from_inst = from_node.instruction
        if from_inst is None:
            dbg_no_inst += 1
            continue

        to_prop = vma_map.get(to_node.vma)
        if to_prop is None:
            dbg_no_prop += 1
            continue

        op = from_inst.op
        base_op = op.split(".")[0].lower()
        dbg_opcode_freq[base_op] = dbg_opcode_freq.get(base_op, 0) + 1

        # Get stall metrics using vendor-aware lookup
        # This handles NVIDIA vs AMD metric naming differences:
        # - NVIDIA: gcycles:stl:gmem for global memory stalls
        # - AMD: gcycles:stl:mem for memory stalls (async model)
        vendor = getattr(to_prop, 'vendor', None)
        exec_dep_lat, mem_dep_lat, total_stall = get_stall_metrics_for_pruning(
            to_prop.prof_metrics, vendor
        )

        # Fallback for async memory models (AMD, Intel):
        # Stalls may be attributed to wait/sync instructions rather than
        # the load/store instruction itself. Use total stalls when specific
        # breakdown is zero at this instruction.
        has_breakdown = (exec_dep_lat > 0 or mem_dep_lat > 0)
        if not has_breakdown and total_stall > 0:
            exec_dep_lat = total_stall
            mem_dep_lat = total_stall
            dbg_fallback += 1
            if dbg_fallback <= 5:
                logger.warning(
                    "Stall breakdown unavailable for PC %#x, using conservative "
                    "pruning (total_stall fallback)", to_node.vma
                )

        # Skip pruning when destination has no profile samples at all.
        # Zero total_stall means no PC samples landed on this instruction,
        # not that the instruction never stalls.
        if total_stall == 0 and not has_breakdown:
            dbg_no_samples += 1
            continue

        # Apply opcode constraint logic
        is_mem = from_inst.is_memory_op()
        if is_mem:
            # Check for shared memory (NVIDIA: LDS/STS, AMD: ds_read/ds_write)
            is_shared = is_shared_memory_opcode(op)
            if is_shared:
                dbg_is_shared += 1
                # Shared memory only causes exec_dep
                if exec_dep_lat == 0:
                    to_remove.append((edge.from_cct_id, edge.to_cct_id))
                    if debug and len(dbg_examples) < 5:
                        dbg_examples.append(
                            f"    PRUNE shared-mem edge: {op} -> to_vma=0x{to_node.vma:x} "
                            f"(exec_dep={exec_dep_lat}, mem_dep={mem_dep_lat})")
            else:
                dbg_is_mem += 1
                # Global/local/constant memory causes mem_dep.
                # NVIDIA exception: memory loads can cause exec_dep stalls at
                # consuming compute instructions because register dependencies
                # from memory show as idep (not gmem) at the consumer. Keep
                # the edge if the destination has ANY stalls (exec or mem).
                if vendor == "nvidia":
                    if mem_dep_lat == 0 and exec_dep_lat == 0:
                        to_remove.append((edge.from_cct_id, edge.to_cct_id))
                        if debug and len(dbg_examples) < 5:
                            dbg_examples.append(
                                f"    PRUNE global-mem edge: {op} -> to_vma=0x{to_node.vma:x} "
                                f"(exec_dep={exec_dep_lat}, mem_dep={mem_dep_lat})")
                else:
                    if mem_dep_lat == 0:
                        to_remove.append((edge.from_cct_id, edge.to_cct_id))
                        if debug and len(dbg_examples) < 5:
                            dbg_examples.append(
                                f"    PRUNE global-mem edge: {op} -> to_vma=0x{to_node.vma:x} "
                                f"(exec_dep={exec_dep_lat}, mem_dep={mem_dep_lat})")
        else:
            dbg_is_compute += 1
            # Other instructions cause exec_dep
            if exec_dep_lat == 0:
                to_remove.append((edge.from_cct_id, edge.to_cct_id))
                if debug and len(dbg_examples) < 5:
                    dbg_examples.append(
                        f"    PRUNE compute edge: {op} -> to_vma=0x{to_node.vma:x} "
                        f"(exec_dep={exec_dep_lat}, mem_dep={mem_dep_lat})")

    for from_id, to_id in to_remove:
        graph.remove_edge_by_id(from_id, to_id)

    if dbg_fallback > 0:
        logger.warning(
            "Opcode pruning: stall breakdown unavailable for %d / %d edges, "
            "used total_stall fallback (conservative pruning)",
            dbg_fallback, dbg_total
        )

    if debug:
        logger.debug("    [opcode pruning debug]")
        logger.debug(f"      Edges examined: {dbg_total}")
        logger.debug(f"      Skipped: no_node={dbg_no_node}, no_inst={dbg_no_inst}, "
              f"no_prop={dbg_no_prop}, no_samples={dbg_no_samples}")
        logger.debug(f"      Classification: memory={dbg_is_mem}, shared={dbg_is_shared}, "
              f"compute={dbg_is_compute}")
        logger.debug(f"      Fallback (no breakdown, used total_stall): {dbg_fallback}")
        # Top 10 opcodes
        sorted_ops = sorted(dbg_opcode_freq.items(), key=lambda x: -x[1])[:10]
        logger.debug(f"      Top opcodes: {', '.join(f'{op}={n}' for op, n in sorted_ops)}")
        # Show raw metric keys from a few sample TO nodes
        sample_count = 0
        for edge in graph.edges():
            to_node = graph.get_node(edge.to_cct_id)
            if to_node is None:
                continue
            to_prop = vma_map.get(to_node.vma)
            if to_prop is None:
                continue
            m = to_prop.prof_metrics
            stl = m.get('gcycles:stl', 0)
            if stl > 0 and sample_count < 3:
                stl_keys = {k: v for k, v in m.items() if 'stl' in k and v != 0}
                from_node = graph.get_node(edge.from_cct_id)
                from_op = from_node.instruction.op if from_node and from_node.instruction else "?"
                logger.debug(f"      Sample edge: {from_op} -> 0x{to_node.vma:x}  "
                      f"vendor={getattr(to_prop, 'vendor', '?')}  "
                      f"nonzero_stl_metrics={stl_keys}")
                sample_count += 1
        if dbg_examples:
            logger.debug("      Example prunes:")
            for ex in dbg_examples:
                logger.debug(ex)

    return len(to_remove)


def _build_sbid_lifecycle_map(
    pc_to_inst: Dict[int, InstructionStat],
) -> Dict[int, Tuple[int, Set[int], int]]:
    """Build SBID lifecycle map for Intel SWSB-based pruning.

    Walks instructions in PC order. For each SBID 'set $T', tracks
    subsequent 'wait $T' instructions until the next 'set $T' reuses
    the token (which implicitly closes the lifecycle).

    Args:
        pc_to_inst: Map from PC to InstructionStat (all instructions).

    Returns:
        Dict mapping set_pc -> (sbid_token, set_of_wait_pcs, end_pc).
        end_pc is the PC of the next 'set $T' (implicit barrier) or -1.
    """
    sorted_pcs = sorted(pc_to_inst.keys())

    # Track currently active set for each SBID token (0-31)
    # token -> [set_pc, sbid, wait_pcs]
    active: Dict[int, List] = {}
    result: Dict[int, Tuple[int, Set[int], int]] = {}

    for pc in sorted_pcs:
        inst = pc_to_inst[pc]
        if inst.swsb is None or not inst.swsb.has_sbid:
            continue

        token = inst.swsb.sbid

        if inst.swsb.sbid_type == "set":
            # Close previous lifecycle for this token
            if token in active:
                prev = active[token]
                result[prev[0]] = (prev[1], prev[2], pc)  # end_pc = this set
            # Start new lifecycle
            active[token] = [pc, token, set()]

        elif inst.swsb.sbid_type in ("dst_wait", "src_wait"):
            if token in active:
                active[token][2].add(pc)

    # Close remaining active lifecycles (end of function)
    for token, entry in active.items():
        if entry[0] not in result:
            result[entry[0]] = (entry[1], entry[2], -1)

    return result


def prune_latency_constraints(
    graph: CCTDepGraph,
    arch: GPUArchitecture,
    cfg: "CFG",
    pc_to_inst: Dict[int, InstructionStat],
) -> int:
    """Remove edges where dependency is hidden by pipeline latency.

    GPA Reference: pruneCCTDepGraphLatency() at GPUAdvisor-Blame.cpp:868-971

    Uses DFS traversal through CFG to find valid paths, accumulating
    control.stall cycles along all paths (GPA-aligned). An edge is
    removed if no valid paths exist (all paths are latency-hidden).

    Args:
        graph: CCTDepGraph to prune (modified in-place).
        arch: GPU architecture for latency lookup.
        cfg: CFG for path-based analysis (GPA-aligned).
        pc_to_inst: Map from PC to InstructionStat.

    Returns:
        Number of edges removed.
    """
    to_remove: List[Tuple[int, int]] = []
    # CFG-based latency tracking accumulates control.stall cycles along all
    # CFG paths from producer to consumer.
    # - NVIDIA: stall from binary encoding (0-15 cycles per instruction)
    # - AMD: stall defaults to 1 (instruction count as cycle lower bound),
    #   except s_nop N which contributes N+1 cycles
    # CFG-based handles branches correctly (GPA-aligned).

    # Intel-specific: build SBID lifecycle map for SWSB-based memory pruning
    is_intel = hasattr(arch, 'vendor') and arch.vendor == 'intel'
    sbid_map: Optional[Dict[int, Tuple[int, Set[int], int]]] = None
    if is_intel:
        sbid_map = _build_sbid_lifecycle_map(pc_to_inst)

    for edge in graph.edges():
        # Skip latency pruning for hardware synchronization edges.
        # These represent compiler/hardware-verified dependencies
        # (AMD s_waitcnt → load, NVIDIA Control.wait → Control.write,
        # Intel SWSB token → send) that cannot be hidden by pipelining.
        edge_type = graph.get_edge_type(edge.from_cct_id, edge.to_cct_id)
        if edge_type in ("mem_waitcnt", "mem_barrier", "mem_swsb"):
            continue

        from_node = graph.get_node(edge.from_cct_id)
        to_node = graph.get_node(edge.to_cct_id)

        if from_node is None or to_node is None:
            continue

        from_inst = from_node.instruction
        to_inst = to_node.instruction
        if from_inst is None:
            continue

        # Intel-specific: skip ALU latency pruning entirely.
        # ALU latency (6-13 cycles) is too small relative to typical instruction
        # distances — pruning ALU edges leaves 0 edges for compute-bound kernels.
        if is_intel and not from_inst.is_memory_op():
            abs_distance = abs(to_node.vma - from_node.vma) / arch.inst_size
            edge.add_path(EdgePathInfo(
                block_ids=(),
                accumulated_stalls=0,
                instruction_count=abs_distance,
                is_hidden=False,
            ))
            continue

        # Intel SWSB SBID-aware pruning for send (memory) edges.
        # If the send has SBID set $T, check for matching wait $T between
        # producer and consumer. If wait found → keep (compiler enforces).
        # If no wait → prune (compiler says latency is hidden).
        if is_intel and from_inst.is_memory_op() and sbid_map is not None:
            from_swsb = from_inst.swsb
            if (from_swsb is not None and from_swsb.has_sbid
                    and from_swsb.sbid_type == "set"):
                lifecycle = sbid_map.get(from_node.vma)
                if lifecycle is not None and to_node.vma > from_node.vma:
                    _sbid, wait_pcs, end_pc = lifecycle
                    # Check for explicit wait between producer and consumer
                    has_wait = any(
                        from_node.vma < wp <= to_node.vma
                        for wp in wait_pcs
                    )
                    # Check for implicit wait (token reuse before consumer)
                    if not has_wait and end_pc != -1:
                        has_wait = from_node.vma < end_pc <= to_node.vma

                    abs_distance = abs(to_node.vma - from_node.vma) / arch.inst_size
                    if has_wait:
                        # Compiler enforces dependency → KEEP
                        edge.add_path(EdgePathInfo(
                            block_ids=(),
                            accumulated_stalls=0,
                            instruction_count=abs_distance,
                            is_hidden=False,
                        ))
                        continue
                    else:
                        # No wait → compiler says latency hidden → PRUNE
                        to_remove.append((edge.from_cct_id, edge.to_cct_id))
                        continue

            # Send without SBID set, or backward dep → fall through to
            # standard latency check below

        # CFG-based latency tracking (GPA-aligned)
        # Returns both pruning decision AND valid paths for blame computation
        should_prune, valid_paths = _check_latency_hidden_cfg(
            from_node.vma, to_node.vma,
            from_inst, to_inst,
            cfg, pc_to_inst, arch
        )

        # Store valid paths on edge for later use in blame computation
        # GPA Reference: cct_edge_path_map storage at GPUAdvisor-Blame.cpp:895-941
        for path_info in valid_paths:
            edge.add_path(path_info)

        if should_prune:
            to_remove.append((edge.from_cct_id, edge.to_cct_id))

    # Log removed edges at debug level
    if to_remove:
        logger.debug(f"  Latency pruning removing {len(to_remove)} edges")

    for from_id, to_id in to_remove:
        graph.remove_edge_by_id(from_id, to_id)

    return len(to_remove)


def _check_latency_hidden_cfg(
    from_vma: int,
    to_vma: int,
    from_inst: InstructionStat,
    to_inst: Optional[InstructionStat],
    cfg: "CFG",
    pc_to_inst: Dict[int, InstructionStat],
    arch: GPUArchitecture,
) -> Tuple[bool, List[EdgePathInfo]]:
    """Check if all paths from from_vma to to_vma are latency-hidden.

    Uses CFG DFS to find valid paths where the dependency is NOT hidden.
    Returns (should_prune, valid_paths) tuple.

    GPA Reference: trackDep() at GPUAdvisor-Blame.cpp:714-817
                   Path storage at GPUAdvisor-Blame.cpp:895-941

    Returns:
        Tuple of (should_prune, valid_paths):
        - should_prune: True if NO valid paths exist
        - valid_paths: List of EdgePathInfo for paths that are NOT latency-hidden
    """
    from leo.analysis.latency_pruning import track_dep_with_latency_cfg

    if to_inst is None:
        return False, []  # Can't determine, keep edge

    # Find common registers between from_inst.dsts and to_inst.srcs
    common_regs = set(from_inst.dsts) & set(to_inst.srcs)

    if not common_regs:
        # No register dependency found - might be control/barrier dep
        # Fall back to simple distance check for non-register deps
        _, max_latency = arch.latency(from_inst.op)
        if to_vma > from_vma:
            distance = (to_vma - from_vma) / arch.inst_size
            if distance >= max_latency:
                return True, []  # Hidden by distance
            # Not hidden - create a simple path info
            simple_path = EdgePathInfo(
                block_ids=(),  # No block info for simple case
                accumulated_stalls=0,
                instruction_count=distance,
                is_hidden=False,
            )
            return False, [simple_path]
        return False, []  # Keep backward dependencies

    # Check each register dependency and collect valid paths
    all_valid_paths: List[EdgePathInfo] = []
    seen_block_ids: Set[Tuple[int, ...]] = set()

    for reg in common_regs:
        # Use GPA-aligned CFG tracking
        path_results = track_dep_with_latency_cfg(
            from_pc=from_vma,
            to_pc=to_vma,
            target_reg=reg,
            cfg=cfg,
            pc_to_inst=pc_to_inst,
            arch=arch,
            use_min_latency=False,  # Use max latency (conservative)
        )

        # Convert PathResult to EdgePathInfo and deduplicate
        for pr in path_results:
            if pr.is_hidden:
                continue  # Skip hidden paths

            # Create block ID tuple for deduplication
            block_ids = tuple(b.id for b in pr.blocks)

            # GPA-aligned deduplication: skip if same block sequence
            if block_ids in seen_block_ids:
                continue
            seen_block_ids.add(block_ids)

            # Compute instruction count (GPA's computePathInsts logic)
            inst_count = _compute_path_instruction_count(
                pr.blocks, from_vma, to_vma, arch.inst_size
            )

            path_info = EdgePathInfo(
                block_ids=block_ids,
                accumulated_stalls=pr.accumulated_cycles,
                instruction_count=inst_count,
                is_hidden=False,
            )
            all_valid_paths.append(path_info)

    # Should prune if NO valid paths exist
    should_prune = len(all_valid_paths) == 0
    return should_prune, all_valid_paths


def _compute_path_instruction_count(
    blocks: List[Any],  # List[Block]
    from_vma: int,
    to_vma: int,
    inst_size: int,
) -> float:
    """Compute instruction count along a CFG path.

    GPA Reference: computePathInsts() at GPUAdvisor-Blame.cpp:1177-1207

    Args:
        blocks: List of CFG blocks in the path.
        from_vma: Source instruction VMA.
        to_vma: Destination instruction VMA.
        inst_size: Instruction size in bytes.

    Returns:
        Number of instructions in path (excluding source, including destination).
    """
    if not blocks:
        # Fallback for empty path
        if to_vma > from_vma:
            return (to_vma - from_vma) / inst_size
        return 1.0

    total_insts = 0.0

    for i, block in enumerate(blocks):
        # Get block boundaries
        if not block.instructions:
            continue

        front_vma = block.instructions[0].pc
        back_vma = block.instructions[-1].pc

        # Determine range to count
        start_vma = front_vma
        end_vma = back_vma

        if i == 0:
            # First block: start after source instruction
            start_vma = from_vma + inst_size

        if i == len(blocks) - 1:
            # Last block: end at destination
            end_vma = to_vma

        # Count instructions: (end - start) / inst_size + 1
        if end_vma >= start_vma:
            total_insts += (end_vma - start_vma) / inst_size + 1

    return max(total_insts, 1.0)  # At least 1 instruction


def prune_barrier_constraints(graph: CCTDepGraph) -> int:
    """Remove edges that violate barrier synchronization constraints.

    GPA Reference: pruneCCTDepGraphBarrier() at GPUAdvisor-Blame.cpp:437-486

    An edge is REMOVED if:
    - Source instruction has at least one valid barrier set (1-6), AND
    - Destination instruction does NOT wait on any of the source's active barriers

    Barrier encoding in NVIDIA CUBIN:
    - 0 = no barrier (actual CUBIN encoding for compute instructions)
    - 1-6 = valid synchronization barriers (B1-B6)
    - 7 = BARRIER_NONE (GPA's default constant)

    Both 0 and 7 are treated as "no barrier", so typical compute kernels
    (without explicit __syncthreads or barriers) will have minimal pruning.

    Args:
        graph: CCTDepGraph to prune (modified in-place).

    Returns:
        Number of edges removed.
    """
    to_remove: List[Tuple[int, int]] = []

    for edge in graph.edges():
        from_node = graph.get_node(edge.from_cct_id)
        to_node = graph.get_node(edge.to_cct_id)

        if from_node is None or to_node is None:
            continue

        from_inst = from_node.instruction
        to_inst = to_node.instruction

        if from_inst is None or to_inst is None:
            continue

        from_read_barrier = from_inst.control.read
        from_write_barrier = from_inst.control.write

        # Skip if BOTH barriers indicate "no barrier"
        # In NVIDIA's encoding: 0 = no barrier (actual CUBIN), 7 = BARRIER_NONE (GPA default)
        # Only barriers 1-6 are valid synchronization barriers (B1-B6)
        if is_no_barrier(from_read_barrier) and is_no_barrier(from_write_barrier):
            continue  # No barrier constraint to check

        # Check if destination waits on either of the source's barriers
        # waits_on_barrier() returns False for BARRIER_NONE (out of range 1-6)
        wait_on_read = to_inst.control.waits_on_barrier(from_read_barrier)
        wait_on_write = to_inst.control.waits_on_barrier(from_write_barrier)

        # Remove edge if destination doesn't wait on ANY of the source's active barriers
        if not wait_on_read and not wait_on_write:
            to_remove.append((edge.from_cct_id, edge.to_cct_id))

    for from_id, to_id in to_remove:
        graph.remove_edge_by_id(from_id, to_id)

    return len(to_remove)


def prune_execution_constraints(
    graph: CCTDepGraph,
    vma_map: Any,  # VMAPropertyMap
) -> int:
    """Remove edges where source instruction was never executed.

    GPA Reference: pruneCCTDepGraphExecution() at GPUAdvisor-Blame.cpp
    (disabled by default in GPA, so Leo also disables it by default)

    An edge is removed if the source instruction has an execution count of 0,
    meaning the path through that instruction was never taken.

    Execution metrics checked (vendor-specific):
    - NVIDIA: GINS:EXE (instruction execution count from PC sampling)
    - AMD/Intel: gcycles:isu (issue cycles as proxy for execution)

    Args:
        graph: CCTDepGraph to prune (modified in-place).
        vma_map: VMAPropertyMap for metric lookup.

    Returns:
        Number of edges removed.
    """
    to_remove: List[Tuple[int, int]] = []

    for edge in graph.edges():
        from_node = graph.get_node(edge.from_cct_id)
        if from_node is None:
            continue

        # Get profile data for source instruction
        from_prop = vma_map.get(from_node.vma)
        if from_prop is None:
            # No profile data available - keep edge (conservative)
            continue

        # Check execution count from multiple vendor-specific metrics
        exec_count = -1.0
        for metric_name in EXECUTION_METRICS:
            if metric_name in from_prop.prof_metrics:
                exec_count = from_prop.prof_metrics[metric_name]
                break

        if exec_count == 0:
            # Explicit zero execution count - prune edge
            to_remove.append((edge.from_cct_id, edge.to_cct_id))
        elif exec_count < 0:
            # No explicit execution metric found
            # Fall back to checking stall_cycles - if instruction stalled, it executed
            if from_prop.stall_cycles == 0 and from_prop.issue_cycles == 0:
                # No evidence of execution, but be conservative - keep edge
                # Only prune if we have definitive zero execution count
                pass

    for from_id, to_id in to_remove:
        graph.remove_edge_by_id(from_id, to_id)

    return len(to_remove)


def get_stalling_nodes(
    graph: CCTDepGraph, threshold: float = 0.0
) -> List[CCTDepNode]:
    """Get all nodes with stall cycles above threshold, sorted by stalls.

    Args:
        graph: CCTDepGraph to query.
        threshold: Minimum stall cycles to include.

    Returns:
        List of CCTDepNode sorted by stall cycles descending.
    """
    stalling = [
        node for node in graph.nodes()
        if node.get_total_stalls() > threshold
    ]
    return sorted(stalling, key=lambda n: n.get_total_stalls(), reverse=True)


def print_graph_summary(graph: CCTDepGraph) -> None:
    """Print a summary of the graph structure."""
    stats = graph.get_statistics()
    logger.info(f"CCTDepGraph Summary:")
    logger.info(f"  Nodes: {stats.num_nodes}")
    logger.info(f"  Edges: {stats.num_edges}")
    logger.info(f"  Nodes with stalls: {stats.num_nodes_with_stalls}")
    logger.info(f"  Source nodes (no deps): {stats.num_source_nodes}")
    logger.info(f"  Sink nodes (no dependents): {stats.num_sink_nodes}")
    logger.info(f"  Total stall cycles: {stats.total_stall_cycles:.2f}")
    logger.info(f"  Avg fan-in: {stats.avg_fan_in:.2f}")
    logger.info(f"  Avg fan-out: {stats.avg_fan_out:.2f}")
    logger.info(f"  Edge types: {stats.edge_types}")
