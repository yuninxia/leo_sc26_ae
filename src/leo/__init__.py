"""Leo: GPU Performance Root Cause Analyzer.

A Python implementation of GPA (GPU Performance Advisor) that performs
back-slicing dataflow analysis to identify root causes of GPU performance
bottlenecks.

Example:
    from leo import KernelAnalyzer, AnalysisConfig
    import json

    config = AnalysisConfig(
        db_path="/path/to/hpctoolkit-database",
        gpubin_path="/path/to/kernel.gpubin",
    )
    analyzer = KernelAnalyzer(config)
    result = analyzer.analyze()

    # Export as JSON (canonical raw data format)
    json_data = result.to_json()
    print(json.dumps(json_data, indent=2))

    # Or use helper scripts to derive presentation formats:
    #   scripts/format_summary.py leo.json      -> Human-readable table
    #   scripts/annotate_assembly.py asm leo.json -> Annotated assembly
    #   scripts/build_graph.py leo.json         -> Dependency graph
"""

__version__ = "0.1.0"

from leo.analyzer import AnalysisConfig, AnalysisResult, KernelAnalyzer, analyze_kernel
from leo.analysis.speedup import SpeedupEstimate
from leo.analysis.topdown import (
    TopDownAnalyzer,
    TopDownResult,
    topdown_analysis,
)
from leo.analysis.pipeline import (
    PipelineAnalyzer,
    PipelineResult,
    pipeline_analysis,
)
from leo.program_analysis import (
    PerKernelAnalysis,
    ProgramAnalysisResult,
    analyze_program,
)
from leo.output.json_output import to_json_dict
from leo.db.discovery import (
    find_database,
    find_all_gpubins,
    find_hpcstruct,
    discover_analysis_inputs,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "KernelAnalyzer",
    "SpeedupEstimate",
    "analyze_kernel",
    "to_json_dict",
    # Top-down analysis
    "TopDownAnalyzer",
    "TopDownResult",
    "topdown_analysis",
    # Pipeline analysis
    "PipelineAnalyzer",
    "PipelineResult",
    "pipeline_analysis",
    # Whole-program analysis
    "PerKernelAnalysis",
    "ProgramAnalysisResult",
    "analyze_program",
    # Discovery utilities
    "find_database",
    "find_all_gpubins",
    "find_hpcstruct",
    "discover_analysis_inputs",
]
