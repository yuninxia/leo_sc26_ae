"""Per-Pipeline CPI Stack Analysis for GPU Profiling.

This module provides analysis for generating stacked bar visualizations showing:
- Per-pipeline state breakdown: Issue%, Stall%, Idle%
- 11 GPU pipelines: VEC, VEC2, SCLR, MATR, LDS, LDSD, TEX, BMSG, FLAT, MISC, XPRT

The analysis is vendor-aware and automatically adapts to available metrics
for NVIDIA, AMD, and Intel GPUs.

Example:
    from leo.db import DatabaseReader
    from leo.analysis.pipeline import PipelineAnalyzer

    reader = DatabaseReader("/path/to/hpctoolkit-database")
    analyzer = PipelineAnalyzer(reader)
    result = analyzer.analyze()

    # Print summary
    print(result.format_summary())

    # Get data for visualization
    stacked_bar_data = result.stacked_bar_data
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from leo.db import DatabaseReader


# =============================================================================
# Pipeline Definitions
# =============================================================================

# Pipeline definitions: (id, short_name, description)
PIPELINES: List[Tuple[str, str, str]] = [
    ("vec", "VEC", "Vector ALU"),
    ("vec2", "VEC2", "Vector ALU 2"),
    ("sclr", "SCLR", "Scalar"),
    ("matr", "MATR", "Matrix"),
    ("lds", "LDS", "Local Data Share"),
    ("ldsd", "LDSD", "LDS Direct"),
    ("tex", "TEX", "Texture"),
    ("bmsg", "BMSG", "Branch Message"),
    ("flat", "FLAT", "Flat Memory"),
    ("misc", "MISC", "Miscellaneous"),
    ("xprt", "XPRT", "Export"),
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PipelineMetricValue:
    """Metrics for a single pipeline showing Issue/Stall/Idle breakdown.

    Attributes:
        pipeline_id: Short identifier for the pipeline (e.g., "vec").
        pipeline_name: Display name for the pipeline (e.g., "VEC").
        description: Full description of the pipeline.
        issue_cycles: Number of cycles where the pipeline issued instructions.
        stall_cycles: Number of cycles where the pipeline was stalled.
        idle_cycles: Number of cycles where the pipeline was idle.
        total_cycles: Total GPU cycles for normalization.
        issue_pct: Issue cycles as percentage of total.
        stall_pct: Stall cycles as percentage of total.
        idle_pct: Idle cycles as percentage of total.
    """

    pipeline_id: str
    pipeline_name: str
    description: str
    issue_cycles: float = 0.0
    stall_cycles: float = 0.0
    idle_cycles: float = 0.0
    total_cycles: float = 0.0
    issue_pct: float = 0.0
    stall_pct: float = 0.0
    idle_pct: float = 0.0

    @property
    def active_cycles(self) -> float:
        """Get active cycles (issue + stall)."""
        return self.issue_cycles + self.stall_cycles

    @property
    def active_pct(self) -> float:
        """Get active percentage (issue + stall)."""
        return self.issue_pct + self.stall_pct

    @property
    def utilization_pct(self) -> float:
        """Get pipeline utilization as percentage (issue / total)."""
        return self.issue_pct


@dataclass
class PipelineResult:
    """Result of per-pipeline CPI stack analysis.

    Attributes:
        database_path: Path to the HPCToolkit database.
        scope: Metric scope used (e.g., "i" for inclusive, "e" for exclusive).
        total_cycles: Total GPU cycles (GCYCLES).
        pipelines: Dictionary mapping pipeline_id to PipelineMetricValue.
        stacked_bar_data: Data structure for stacked bar visualization.
        available_metrics: List of metric names found in the database.
    """

    database_path: str
    scope: str
    total_cycles: float
    pipelines: Dict[str, PipelineMetricValue] = field(default_factory=dict)
    stacked_bar_data: Dict[str, Any] = field(default_factory=dict)
    available_metrics: List[str] = field(default_factory=list)

    def get_active_pipelines(self) -> List[PipelineMetricValue]:
        """Get pipelines with non-zero activity, sorted by utilization.

        Returns:
            List of PipelineMetricValue sorted by issue_pct descending.
        """
        active = [p for p in self.pipelines.values() if p.active_pct > 0]
        return sorted(active, key=lambda x: -x.issue_pct)

    def get_top_pipelines(self, n: int = 5) -> List[PipelineMetricValue]:
        """Get top N pipelines sorted by issue percentage.

        Args:
            n: Maximum number of pipelines to return.

        Returns:
            List of PipelineMetricValue sorted by issue_pct descending.
        """
        return self.get_active_pipelines()[:n]

    def get_most_stalled_pipelines(self, n: int = 5) -> List[PipelineMetricValue]:
        """Get top N pipelines sorted by stall percentage.

        Args:
            n: Maximum number of pipelines to return.

        Returns:
            List of PipelineMetricValue sorted by stall_pct descending.
        """
        stalled = [p for p in self.pipelines.values() if p.stall_pct > 0]
        return sorted(stalled, key=lambda x: -x.stall_pct)[:n]

    def format_summary(self, show_zero: bool = False) -> str:
        """Format the analysis as a human-readable summary table.

        Args:
            show_zero: Include pipelines with zero values.

        Returns:
            Formatted string summary.
        """
        lines = [
            "=" * 70,
            "GPU Per-Pipeline CPI Stack Analysis",
            "=" * 70,
            f"\nDatabase: {self.database_path}",
            f"Total GPU Cycles: {self.total_cycles:,.0f}",
            "",
            "--- Pipeline State Breakdown (% of Total Cycles) ---",
            "",
            f"{'Pipeline':<10} {'Description':<20} {'Issue%':>10} {'Stall%':>10} {'Idle%':>10}",
            "-" * 70,
        ]

        # Sort pipelines by their defined order
        pipeline_order = [p[0] for p in PIPELINES]
        for pipe_id in pipeline_order:
            if pipe_id in self.pipelines:
                p = self.pipelines[pipe_id]
                if p.active_pct > 0 or show_zero:
                    lines.append(
                        f"{p.pipeline_name:<10} {p.description:<20} "
                        f"{p.issue_pct:>9.2f}% {p.stall_pct:>9.2f}% {p.idle_pct:>9.2f}%"
                    )

        lines.append("-" * 70)

        # Summary statistics
        total_issue = sum(p.issue_pct for p in self.pipelines.values())
        total_stall = sum(p.stall_pct for p in self.pipelines.values())
        active_pipes = len([p for p in self.pipelines.values() if p.active_pct > 0])

        lines.extend([
            "",
            "--- Summary ---",
            f"Active Pipelines: {active_pipes}",
            f"Total Issue%: {total_issue:.2f}%",
            f"Total Stall%: {total_stall:.2f}%",
        ])

        # Top pipelines
        top = self.get_top_pipelines(3)
        if top:
            lines.append("")
            lines.append("Top Utilized Pipelines:")
            for p in top:
                lines.append(f"  {p.pipeline_name}: {p.issue_pct:.2f}% issue, {p.stall_pct:.2f}% stall")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary suitable for json.dumps().
        """
        return {
            "metadata": {
                "database": self.database_path,
                "scope": self.scope,
                "description": "GPU Per-Pipeline CPI Stack Analysis",
            },
            "total_cycles": self.total_cycles,
            "pipelines": {
                pipe_id: {
                    "name": p.pipeline_name,
                    "description": p.description,
                    "issue_cycles": p.issue_cycles,
                    "stall_cycles": p.stall_cycles,
                    "idle_cycles": p.idle_cycles,
                    "issue_pct": p.issue_pct,
                    "stall_pct": p.stall_pct,
                    "idle_pct": p.idle_pct,
                }
                for pipe_id, p in self.pipelines.items()
            },
            "stacked_bar_data": self.stacked_bar_data,
            "available_metrics": self.available_metrics,
        }

    def to_csv(self, include_header: bool = True) -> str:
        """Convert to CSV format for data export.

        Args:
            include_header: Include CSV header row.

        Returns:
            CSV formatted string.
        """
        lines = []
        if include_header:
            lines.append(
                "pipeline_id,pipeline_name,description,"
                "issue_cycles,stall_cycles,idle_cycles,"
                "issue_pct,stall_pct,idle_pct"
            )

        # Maintain pipeline order
        pipeline_order = [p[0] for p in PIPELINES]
        for pipe_id in pipeline_order:
            if pipe_id in self.pipelines:
                p = self.pipelines[pipe_id]
                lines.append(
                    f"{p.pipeline_id},{p.pipeline_name},{p.description},"
                    f"{p.issue_cycles:.0f},{p.stall_cycles:.0f},{p.idle_cycles:.0f},"
                    f"{p.issue_pct:.4f},{p.stall_pct:.4f},{p.idle_pct:.4f}"
                )

        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        """Save result as JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_figure(self, path: str, label_threshold: float = 5.0, show_all: bool = False) -> None:
        """Save result as stacked bar chart PNG.

        Each pipeline bar totals 100%:
        - Stall (red) = back-pressure during issue
        - Issue (green) = successful issue
        - Idle (light hatched) = remaining cycles (100% - issue%)

        Args:
            path: Output file path.
            label_threshold: Minimum percentage to show label on bar (default 5%).
            show_all: Show all 11 pipelines even if 100% idle (default False).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if show_all:
            # Show all pipelines in canonical order
            pipeline_order = [p[0] for p in PIPELINES]
            active_pipes = [self.pipelines[pid] for pid in pipeline_order if pid in self.pipelines]
        else:
            # Get active pipelines (those with non-zero issue or stall)
            active_pipes = [p for p in self.pipelines.values() if p.issue_pct > 0 or p.stall_pct > 0]
        if not active_pipes:
            # No active pipelines, create empty figure
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No active pipelines found\n(Per-pipeline metrics are AMD-specific)",
                   ha='center', va='center', fontsize=14)
            ax.set_title("GPU Per-Pipeline CPI Stack")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return

        # Sort by issue percentage (most utilized first), unless show_all
        # which preserves canonical pipeline order for cross-workload comparison
        if not show_all:
            active_pipes = sorted(active_pipes, key=lambda x: -x.issue_pct)

        # Prepare data - each bar totals 100%
        # Stall (red) = back-pressure, subset of issue attempts
        # Success (green) = issue - stall
        # Idle (hatched) = 100% - issue
        names = [p.pipeline_name for p in active_pipes]
        issue_vals = [p.issue_pct for p in active_pipes]
        stall_vals = [min(p.stall_pct, p.issue_pct) for p in active_pipes]
        success_vals = [issue - stall for issue, stall in zip(issue_vals, stall_vals)]
        idle_vals = [max(0, 100 - issue) for issue in issue_vals]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(names))
        width = 0.6

        # Stacked bars: Stall (red) -> Success (green) -> Idle (light hatched)
        bars_stall = ax.bar(x, stall_vals, width, label='Stall (back-pressure)', color='#e74c3c')
        bars_success = ax.bar(x, success_vals, width, bottom=stall_vals,
                             label='Issue (successful)', color='#2ecc71')
        bars_idle = ax.bar(x, idle_vals, width, bottom=issue_vals,
                          label='Idle', color='#d5d5d5', hatch='///', edgecolor='#a0a0a0')

        # Add labels on bars above threshold
        for i, (stall, success, idle, issue) in enumerate(zip(stall_vals, success_vals, idle_vals, issue_vals)):
            bar_x = x[i]
            # Stall label
            if stall >= label_threshold:
                ax.text(bar_x, stall / 2,
                       f'{stall:.0f}%', ha='center', va='center', fontsize=9,
                       fontweight='bold', color='white')
            # Success label
            if success >= label_threshold:
                ax.text(bar_x, stall + success / 2,
                       f'{success:.0f}%', ha='center', va='center', fontsize=9,
                       fontweight='bold', color='white')
            # Idle label (only if significant)
            if idle >= label_threshold:
                ax.text(bar_x, issue + idle / 2,
                       f'{idle:.0f}%', ha='center', va='center', fontsize=9,
                       fontweight='bold', color='#555555')

        ax.set_ylabel('Percentage of Total Cycles', fontsize=12)
        ax.set_xlabel('GPU Pipeline', fontsize=12)
        ax.set_title('GPU Per-Pipeline CPI Stack\n(Red=Stall, Green=Issue, Hatched=Idle)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.legend(loc='upper right', fontsize=10)

        # Y-axis fixed at 100%
        ax.set_ylim(0, 105)

        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# Analyzer Class
# =============================================================================


class PipelineAnalyzer:
    """Analyzer for GPU per-pipeline CPI stack analysis.

    This class provides vendor-aware analysis of GPU pipeline utilization
    for NVIDIA, AMD, and Intel GPUs.

    Example:
        reader = DatabaseReader("/path/to/database")
        analyzer = PipelineAnalyzer(reader)
        result = analyzer.analyze()
        print(result.format_summary())
    """

    def __init__(self, reader: DatabaseReader):
        """Initialize the analyzer.

        Args:
            reader: DatabaseReader instance for the HPCToolkit database.
        """
        self._reader = reader
        self._available_metrics: Optional[List[str]] = None

    @property
    def available_metrics(self) -> List[str]:
        """Get list of available metric names in the database."""
        if self._available_metrics is None:
            try:
                metrics_df = self._reader.get_metrics("*")
                self._available_metrics = metrics_df["name"].tolist()
            except (ValueError, KeyError):
                self._available_metrics = []
        return self._available_metrics

    def _get_metric_value(self, metric_name: str, scope: str = "i") -> float:
        """Get the aggregated value of a metric.

        Args:
            metric_name: Name of the metric (e.g., "gpipe:isu:vec").
            scope: Metric scope ("i" for inclusive, "e" for exclusive).

        Returns:
            Aggregated metric value, or 0.0 if not found.
        """
        try:
            result = self._reader.get_profile_slices(
                cct_exp="application",
                profiles_exp="summary",
                metrics_exp=f"{metric_name}:sum ({scope})",
            )
            if len(result) > 0:
                return float(result["value"].sum())
        except (ValueError, KeyError, Exception):
            pass
        return 0.0

    def _build_stacked_bar_data(
        self,
        total_cycles: float,
        pipelines: Dict[str, PipelineMetricValue],
    ) -> Dict[str, Any]:
        """Build data structure for stacked bar visualization.

        Args:
            total_cycles: Total GPU cycles.
            pipelines: Dictionary of pipeline metrics.

        Returns:
            Dictionary suitable for D3.js/Chart.js stacked bar charts.
        """
        # Maintain pipeline order from PIPELINES
        pipeline_order = [p[0] for p in PIPELINES]

        categories = []
        issue_data = []
        stall_data = []
        idle_data = []

        for pipe_id in pipeline_order:
            if pipe_id in pipelines:
                p = pipelines[pipe_id]
                categories.append(p.pipeline_name)
                issue_data.append(p.issue_pct)
                stall_data.append(p.stall_pct)
                idle_data.append(p.idle_pct)

        return {
            "chart_type": "stacked_bar",
            "title": "Per-Pipeline CPI Stack",
            "total_cycles": total_cycles,
            "categories": categories,
            "series": [
                {
                    "name": "Issue",
                    "data": issue_data,
                    "color": "#4CAF50",  # Green
                },
                {
                    "name": "Stall",
                    "data": stall_data,
                    "color": "#F44336",  # Red
                },
                {
                    "name": "Idle",
                    "data": idle_data,
                    "color": "#9E9E9E",  # Gray
                },
            ],
        }

    def analyze(self, scope: str = "i") -> PipelineResult:
        """Perform per-pipeline CPI stack analysis.

        Args:
            scope: Metric scope to use ("i" for inclusive, "e" for exclusive).
                   Default is "i" (inclusive) which aggregates across all CCT nodes.

        Returns:
            PipelineResult with the complete pipeline analysis.
        """
        db_path = self._reader.db_path

        # Get total cycles for normalization
        total_cycles = self._get_metric_value("gcycles", scope)

        # Analyze each pipeline
        pipelines: Dict[str, PipelineMetricValue] = {}
        for pipe_id, pipe_name, description in PIPELINES:
            # Query issue and stall metrics for this pipeline
            issue_cycles = self._get_metric_value(f"gpipe:isu:{pipe_id}", scope)
            stall_cycles = self._get_metric_value(f"gpipe:stl:{pipe_id}", scope)

            # Calculate idle cycles (remaining cycles)
            active_cycles = issue_cycles + stall_cycles
            idle_cycles = max(0.0, total_cycles - active_cycles)

            # Calculate percentages
            if total_cycles > 0:
                issue_pct = (issue_cycles / total_cycles) * 100.0
                stall_pct = (stall_cycles / total_cycles) * 100.0
                idle_pct = (idle_cycles / total_cycles) * 100.0
            else:
                issue_pct = stall_pct = idle_pct = 0.0

            pipelines[pipe_id] = PipelineMetricValue(
                pipeline_id=pipe_id,
                pipeline_name=pipe_name,
                description=description,
                issue_cycles=issue_cycles,
                stall_cycles=stall_cycles,
                idle_cycles=idle_cycles,
                total_cycles=total_cycles,
                issue_pct=issue_pct,
                stall_pct=stall_pct,
                idle_pct=idle_pct,
            )

        # Build stacked bar data structure
        stacked_bar_data = self._build_stacked_bar_data(total_cycles, pipelines)

        return PipelineResult(
            database_path=db_path,
            scope=scope,
            total_cycles=total_cycles,
            pipelines=pipelines,
            stacked_bar_data=stacked_bar_data,
            available_metrics=self.available_metrics,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def pipeline_analysis(
    db_path: str,
    scope: str = "i",
    use_cpp_parser: bool = True,
) -> PipelineResult:
    """Convenience function for per-pipeline CPI stack analysis.

    Args:
        db_path: Path to HPCToolkit database directory.
        scope: Metric scope ("i" for inclusive, "e" for exclusive).
        use_cpp_parser: Use C++ parser for better performance.

    Returns:
        PipelineResult with the complete pipeline analysis.

    Example:
        from leo.analysis.pipeline import pipeline_analysis

        result = pipeline_analysis("/path/to/database")
        print(result.format_summary())
    """
    reader = DatabaseReader(db_path, use_cpp_parser=use_cpp_parser)
    analyzer = PipelineAnalyzer(reader)
    return analyzer.analyze(scope=scope)


def print_pipeline_summary(db_path: str, scope: str = "i") -> None:
    """Print per-pipeline CPI stack summary to stdout.

    Args:
        db_path: Path to HPCToolkit database directory.
        scope: Metric scope ("i" for inclusive, "e" for exclusive).
    """
    result = pipeline_analysis(db_path, scope=scope)
    print(result.format_summary())
