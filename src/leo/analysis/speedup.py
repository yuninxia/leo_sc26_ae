"""Speedup Estimation based on Amdahl's Law.

Based on GPA's speedup estimation methodology (CGO 2021).

The speedup formula uses Amdahl's Law:
    speedup = total_cycles / (total_cycles - blamed_cycles)

This estimates the maximum achievable speedup if the blamed bottleneck
is completely eliminated. The estimation has been validated to achieve
4.1% average error across 33 optimization scenarios.

References:
    - GPA: A GPU Performance Advisor Based on Instruction Sampling (CGO 2021)
    - Amdahl, G. M. "Validity of the single processor approach" (1967)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from leo.binary.instruction import is_shared_memory_opcode
from leo.utils.location import format_source_location


class OptimizationType(Enum):
    """Types of optimization with their typical reducibility factors."""
    MEMORY_COALESCING = "memory_coalescing"
    MEMORY_LOCALITY = "memory_locality"
    EXECUTION_REORDER = "execution_reorder"
    LOOP_UNROLL = "loop_unroll"
    BARRIER_REDUCTION = "barrier_reduction"
    UNKNOWN = "unknown"


# Reducibility factors based on GPA validation results
# Higher factor = more reducible through optimization
REDUCIBILITY_FACTORS = {
    OptimizationType.MEMORY_COALESCING: 0.85,   # High - data layout change
    OptimizationType.MEMORY_LOCALITY: 0.60,     # Medium - caching/tiling
    OptimizationType.EXECUTION_REORDER: 0.70,   # Medium - ILP improvement
    OptimizationType.LOOP_UNROLL: 0.65,         # Medium - latency hiding
    OptimizationType.BARRIER_REDUCTION: 0.30,   # Low - algorithmic change
    OptimizationType.UNKNOWN: 0.50,             # Conservative default
}

# Optimization suggestions by type
OPTIMIZATION_SUGGESTIONS = {
    OptimizationType.MEMORY_COALESCING: (
        "Improve memory coalescing by reorganizing data layout or access patterns"
    ),
    OptimizationType.MEMORY_LOCALITY: (
        "Improve cache locality through tiling, blocking, or data reuse"
    ),
    OptimizationType.EXECUTION_REORDER: (
        "Increase instruction-level parallelism through code reordering"
    ),
    OptimizationType.LOOP_UNROLL: (
        "Unroll loops to hide memory latency and expose more parallelism"
    ),
    OptimizationType.BARRIER_REDUCTION: (
        "Reduce synchronization overhead by restructuring algorithm"
    ),
    OptimizationType.UNKNOWN: (
        "Analyze instruction dependencies and optimize based on bottleneck type"
    ),
}


@dataclass
class SpeedupEstimate:
    """Estimated speedup from optimizing a specific root cause.

    Attributes:
        root_cause_pc: Program counter of the root cause instruction.
        root_cause_opcode: Opcode of the root cause instruction.
        source_location: Optional (filename, line) tuple for source mapping.
        blame_cycles: Total cycles blamed to this instruction.
        blame_ratio: Fraction of total cycles (0.0 to 1.0).
        estimated_speedup: Estimated speedup using Amdahl's Law.
        optimization_type: Classified optimization type.
        optimization_suggestion: Human-readable optimization advice.
        reducibility: Estimated fraction of blame that can be eliminated (0.0 to 1.0).
        confidence: Confidence level of the estimate ("high", "medium", "low").
    """
    root_cause_pc: int
    root_cause_opcode: str
    source_location: Optional[Tuple[str, int]]
    blame_cycles: float
    blame_ratio: float
    estimated_speedup: float
    optimization_type: OptimizationType
    optimization_suggestion: str
    reducibility: float
    confidence: str

    def format_location(self) -> str:
        """Format source location as 'file:line' or hex PC."""
        if self.source_location:
            return format_source_location(
                self.source_location[0],
                self.source_location[1],
                col=0,
                short=True,
            )
        return f"0x{self.root_cause_pc:x}"


def classify_optimization_type(opcode: str, blame_type: str) -> OptimizationType:
    """Classify the optimization type based on instruction and blame type.

    Args:
        opcode: Instruction opcode (e.g., "LDG.E.64", "global_load_dwordx2").
        blame_type: Blame type from analysis (e.g., "mem_dep_gmem", "exec_dep_dep").

    Returns:
        OptimizationType enum value.
    """
    opcode_upper = opcode.upper()

    # Memory operations
    if "mem_dep" in blame_type or "MEM" in blame_type:
        # Global memory - check for coalescing opportunities
        if any(x in opcode_upper for x in ["LDG", "STG", "GLOBAL", "BUFFER", "FLAT"]):
            return OptimizationType.MEMORY_COALESCING
        # Shared/local memory - locality optimization
        if is_shared_memory_opcode(opcode):
            return OptimizationType.MEMORY_LOCALITY

    # Execution dependencies
    if "exec_dep" in blame_type:
        return OptimizationType.EXECUTION_REORDER

    # Synchronization
    if "sync" in blame_type or any(x in opcode_upper for x in ["BAR", "WAITCNT", "BARRIER"]):
        return OptimizationType.BARRIER_REDUCTION

    # Self-blame on memory operations
    if any(x in opcode_upper for x in ["LDG", "STG", "GLOBAL", "BUFFER", "FLAT"]):
        return OptimizationType.MEMORY_COALESCING

    return OptimizationType.UNKNOWN


def compute_confidence(
    blame_ratio: float,
    optimization_type: OptimizationType,
    is_self_blame: bool,
) -> str:
    """Compute confidence level for a speedup estimate.

    Args:
        blame_ratio: Fraction of total cycles blamed to this instruction.
        optimization_type: Classified optimization type.
        is_self_blame: Whether this is self-blame (src_pc == dst_pc).

    Returns:
        Confidence level: "high", "medium", or "low".
    """
    # High confidence for significant bottlenecks with known optimization paths
    if blame_ratio >= 0.15 and optimization_type != OptimizationType.UNKNOWN:
        return "high"

    # Medium confidence for moderate bottlenecks or unknown types
    if blame_ratio >= 0.05:
        if optimization_type == OptimizationType.UNKNOWN:
            return "medium"
        return "high" if not is_self_blame else "medium"

    # Low confidence for small bottlenecks
    return "low"


def estimate_speedup_amdahl(
    total_cycles: float,
    blamed_cycles: float,
) -> float:
    """Calculate speedup using Amdahl's Law.

    Formula: speedup = total / (total - blamed)

    This represents the maximum achievable speedup if the blamed
    portion is completely eliminated.

    Args:
        total_cycles: Total execution cycles.
        blamed_cycles: Cycles attributed to the bottleneck.

    Returns:
        Estimated speedup factor (>= 1.0).
    """
    if total_cycles <= 0:
        return 1.0
    if blamed_cycles >= total_cycles:
        return float('inf')
    if blamed_cycles <= 0:
        return 1.0

    return total_cycles / (total_cycles - blamed_cycles)


def estimate_speedup_conservative(
    total_cycles: float,
    blamed_cycles: float,
    reducibility: float,
) -> float:
    """Calculate conservative speedup estimate.

    Uses reducibility factor to account for the fact that not all
    blamed cycles can be eliminated through optimization.

    Formula: speedup = total / (total - blamed × reducibility)

    Args:
        total_cycles: Total execution cycles.
        blamed_cycles: Cycles attributed to the bottleneck.
        reducibility: Fraction of blamed cycles that can be eliminated (0-1).

    Returns:
        Conservative speedup estimate (>= 1.0).
    """
    reducible = blamed_cycles * reducibility
    return estimate_speedup_amdahl(total_cycles, reducible)


def compute_speedup_estimates(
    blame_by_pc: Dict[int, Tuple[float, str, str]],  # pc -> (blame, opcode, blame_type)
    total_cycles: float,
    source_mapping: Optional[Dict[int, Tuple[str, int]]] = None,
    top_n: int = 10,
) -> List[SpeedupEstimate]:
    """Compute speedup estimates for top blamed instructions.

    Args:
        blame_by_pc: Map from PC to (blame_cycles, opcode, blame_type).
        total_cycles: Total kernel execution cycles.
        source_mapping: Optional PC to (filename, line) mapping.
        top_n: Number of top estimates to return.

    Returns:
        List of SpeedupEstimate objects sorted by estimated speedup.
    """
    if total_cycles <= 0:
        return []

    estimates = []

    for pc, (blame, opcode, blame_type) in blame_by_pc.items():
        if blame <= 0:
            continue

        # Calculate blame ratio
        blame_ratio = blame / total_cycles

        # Classify optimization type
        opt_type = classify_optimization_type(opcode, blame_type)

        # Get reducibility factor
        reducibility = REDUCIBILITY_FACTORS[opt_type]

        # Calculate speedup (conservative estimate)
        speedup = estimate_speedup_conservative(total_cycles, blame, reducibility)

        # Get source location if available
        source_loc = source_mapping.get(pc) if source_mapping else None

        # Determine if self-blame
        is_self_blame = "self" in blame_type.lower()

        # Compute confidence
        confidence = compute_confidence(blame_ratio, opt_type, is_self_blame)

        estimates.append(SpeedupEstimate(
            root_cause_pc=pc,
            root_cause_opcode=opcode,
            source_location=source_loc,
            blame_cycles=blame,
            blame_ratio=blame_ratio,
            estimated_speedup=speedup,
            optimization_type=opt_type,
            optimization_suggestion=OPTIMIZATION_SUGGESTIONS[opt_type],
            reducibility=reducibility,
            confidence=confidence,
        ))

    # Sort by estimated speedup (descending)
    estimates.sort(key=lambda e: e.estimated_speedup, reverse=True)

    return estimates[:top_n]


def format_speedup_report(
    estimates: List[SpeedupEstimate],
    total_cycles: float,
) -> str:
    """Format speedup estimates as a human-readable report.

    Args:
        estimates: List of SpeedupEstimate objects.
        total_cycles: Total kernel execution cycles.

    Returns:
        Formatted string report.
    """
    if not estimates:
        return "  (no optimization opportunities identified)"

    lines = [
        "  Rank  Location                      Blame        Speedup  Confidence",
        "  " + "-" * 70,
    ]

    for i, est in enumerate(estimates, 1):
        loc_str = est.format_location()
        blame_pct = est.blame_ratio * 100
        speedup_str = f"{est.estimated_speedup:.2f}x"

        lines.append(
            f"  #{i:<3}  {loc_str:<28} {blame_pct:>5.1f}%  →  {speedup_str:<7}  {est.confidence}"
        )

    # Add total potential speedup
    if estimates:
        # Calculate combined speedup if all top issues fixed
        # Use Amdahl's Law: combined_speedup = total / (total - sum(reducible))
        total_reducible = sum(
            e.blame_cycles * e.reducibility
            for e in estimates
        )
        combined_speedup = estimate_speedup_amdahl(total_cycles, total_reducible)

        lines.append("")
        lines.append(f"  Combined potential speedup (all above): {combined_speedup:.2f}x")

    return "\n".join(lines)
