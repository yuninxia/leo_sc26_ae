"""Analysis algorithms for GPU performance optimization."""

from .predicate_tracking import (
    track_dependency_with_predicates,
    should_stop_predicate_tracking,
    refine_assign_pcs_with_predicates,
    PredicateState,
    StopReason,
)

from .latency_pruning import (
    check_latency_hidden_linear,
    track_dep_with_latency_cfg,
    prune_assign_pcs_by_latency,
    apply_latency_pruning,
    LatencyCheckResult,
    PathResult,
)

from .vma_property import (
    VMAProperty,
    VMAPropertyMap,
    build_vma_property_map,
)

from .blame import (
    BlameCategory,
    BlameEdge,
    KernelBlameResult,
    blame_instructions,
    reverse_ratio,
    distribute_blame,
    compute_distance,
    print_blame_report,
)

from .graph import (
    CCTDepNode,
    CCTDepEdge,
    CCTDepGraph,
    DependencyPath,
    EdgePathInfo,
    GraphStatistics,
    build_cct_dep_graph,
    compute_backward_slice,
    compute_forward_slice,
    classify_edge_type,
    find_barrier_memory_deps,
    prune_opcode_constraints,
    prune_latency_constraints,
    prune_barrier_constraints,
    get_stalling_nodes,
    print_graph_summary,
)

from .backslice import (
    BackSliceConfig,
    BackSliceStats,
    BackSliceEngine,
    analyze_kernel,
    analyze_instructions,
    print_analysis_summary,
)

from .topdown import (
    TopDownAnalyzer,
    TopDownResult,
    MetricValue,
    topdown_analysis,
    print_topdown_summary,
)

from .pipeline import (
    PipelineAnalyzer,
    PipelineResult,
    PipelineMetricValue,
    pipeline_analysis,
    print_pipeline_summary,
)

__all__ = [
    # Predicate tracking
    "track_dependency_with_predicates",
    "should_stop_predicate_tracking",
    "refine_assign_pcs_with_predicates",
    "PredicateState",
    "StopReason",
    # Latency pruning
    "check_latency_hidden_linear",
    "track_dep_with_latency_cfg",
    "prune_assign_pcs_by_latency",
    "apply_latency_pruning",
    "LatencyCheckResult",
    "PathResult",
    # VMA property map
    "VMAProperty",
    "VMAPropertyMap",
    "build_vma_property_map",
    # Blame attribution
    "BlameCategory",
    "BlameEdge",
    "KernelBlameResult",
    "blame_instructions",
    "reverse_ratio",
    "distribute_blame",
    "compute_distance",
    "print_blame_report",
    # CCT Dependency Graph
    "CCTDepNode",
    "CCTDepEdge",
    "CCTDepGraph",
    "DependencyPath",
    "EdgePathInfo",
    "GraphStatistics",
    "build_cct_dep_graph",
    "compute_backward_slice",
    "compute_forward_slice",
    "classify_edge_type",
    "find_barrier_memory_deps",
    "prune_opcode_constraints",
    "prune_latency_constraints",
    "prune_barrier_constraints",
    "get_stalling_nodes",
    "print_graph_summary",
    # Back-slicing Engine
    "BackSliceConfig",
    "BackSliceStats",
    "BackSliceEngine",
    "analyze_kernel",
    "analyze_instructions",
    "print_analysis_summary",
    # Top-down analysis
    "TopDownAnalyzer",
    "TopDownResult",
    "MetricValue",
    "topdown_analysis",
    "print_topdown_summary",
    # Pipeline analysis
    "PipelineAnalyzer",
    "PipelineResult",
    "PipelineMetricValue",
    "pipeline_analysis",
    "print_pipeline_summary",
]
