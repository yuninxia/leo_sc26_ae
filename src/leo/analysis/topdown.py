"""Top-Down Hierarchical GPU Cycle Breakdown Analysis.

Provides analysis for generating 2-ring donut/sunburst visualizations showing:
- Level 0 (center): Total GPU cycles
- Level 1 (inner ring): Issue | Hidden | Exposed
- Level 2 (outer ring): Instruction types (Issue) and Stall reasons (Exposed)

Example:
    from leo.analysis.topdown import topdown_analysis

    result = topdown_analysis("/path/to/hpctoolkit-database")
    print(result.format_summary())
    result.save_json("output.json")
    result.save_figure("output.png")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from leo.db import DatabaseReader


# =============================================================================
# Metric Definitions
# =============================================================================

LEVEL1_METRICS = {
    "gcycles:isu": "Issue",
    "gcycles:stl": "Stall",
    "gcycles:stl:hid": "Hidden",
}

LEVEL2_ISSUE_METRICS = {
    "gcycles:isu:vec": "VEC (Vector ALU)",
    "gcycles:isu:vec2": "VEC2 (Vector ALU 2)",
    "gcycles:isu:sclr": "SCLR (Scalar)",
    "gcycles:isu:matr": "MATR (Matrix)",
    "gcycles:isu:lds": "LDS (Local Data Share)",
    "gcycles:isu:ldsd": "LDSD (LDS Direct)",
    "gcycles:isu:tex": "TEX (Texture)",
    "gcycles:isu:flat": "FLAT (Flat Memory)",
    "gcycles:isu:mesg": "MESG (Message)",
    "gcycles:isu:bmsg": "BMSG (Branch Message)",
    "gcycles:isu:bar": "BAR (Barrier)",
    "gcycles:isu:xprt": "XPRT (Export)",
    "gcycles:isu:brnt": "BRNT (Branch Not Taken)",
    "gcycles:isu:brt": "BRT (Branch Taken)",
    "gcycles:isu:jmp": "JMP (Jump)",
    "gcycles:isu:misc": "MISC (Miscellaneous)",
    "gcycles:isu:othr": "OTHR (Other)",
    "gcycles:isu:unk": "UNK (Unknown)",
}

LEVEL2_STALL_METRICS = {
    "gcycles:stl:mem": "MEM (Memory)",
    "gcycles:stl:gmem": "GMEM (Global Memory)",
    "gcycles:stl:tmem": "TMEM (Texture Memory)",
    "gcycles:stl:cmem": "CMEM (Constant Memory)",
    "gcycles:stl:lmem": "LMEM (Local/Shared Memory)",
    "gcycles:stl:sync": "SYNC (Synchronization)",
    "gcycles:stl:idep": "IDEP (Instruction Dependency)",
    "gcycles:stl:pipe": "PIPE (Pipeline)",
    "gcycles:stl:ifet": "IFET (Instruction Fetch)",
    "gcycles:stl:slp": "SLP (Sleep)",
    "gcycles:stl:inv": "INV (Invalid)",
    "gcycles:stl:mthr": "MTHR (Multi-thread)",
    "gcycles:stl:othr": "OTHR (Other)",
    # Intel-specific
    "gcycles:stl:active": "Active (Not Stalled)",
    "gcycles:stl:control": "Control Flow",
    "gcycles:stl:send": "SEND (Memory Unit)",
    "gcycles:stl:dist": "DIST (Distance/ARF)",
    "gcycles:stl:sbid": "SBID (Scoreboard)",
    "gcycles:stl:insfetch": "InsFetch (Instruction Fetch)",
    "gcycles:stl:other": "Other",
}

# Colors for visualization
_COLORS = {
    "Issue": "#2ecc71",
    "Hidden": "#1abc9c",
    "Exposed": "#e74c3c",
    "issue_shades": ["#27ae60", "#2ecc71", "#58d68d", "#82e0aa", "#abebc6", "#d5f5e3"],
    "exposed_shades": ["#c0392b", "#e74c3c", "#ec7063", "#f1948a", "#f5b7b1", "#fadbd8"],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetricValue:
    """A metric with its value and percentages."""
    metric_name: str
    label: str
    value: float
    percentage_of_total: float = 0.0
    percentage_of_parent: float = 0.0


@dataclass
class TopDownResult:
    """Result of top-down hierarchical cycle breakdown analysis."""

    database_path: str
    scope: str
    total_cycles: float
    issue_cycles: float
    hidden_cycles: float
    exposed_cycles: float
    level2_issue: Dict[str, MetricValue] = field(default_factory=dict)
    level2_exposed: Dict[str, MetricValue] = field(default_factory=dict)
    sunburst_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def issue_percentage(self) -> float:
        return (self.issue_cycles / self.total_cycles * 100) if self.total_cycles > 0 else 0.0

    @property
    def hidden_percentage(self) -> float:
        return (self.hidden_cycles / self.total_cycles * 100) if self.total_cycles > 0 else 0.0

    @property
    def exposed_percentage(self) -> float:
        return (self.exposed_cycles / self.total_cycles * 100) if self.total_cycles > 0 else 0.0

    def get_top_issue_types(self, n: int = 5) -> List[MetricValue]:
        """Get top N issue types sorted by value."""
        items = [mv for mv in self.level2_issue.values() if mv.value > 0]
        return sorted(items, key=lambda x: -x.value)[:n]

    def get_top_stall_reasons(self, n: int = 5) -> List[MetricValue]:
        """Get top N exposed stall reasons sorted by value."""
        items = [mv for mv in self.level2_exposed.values() if mv.value > 0]
        return sorted(items, key=lambda x: -x.value)[:n]

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        lines = [
            "=" * 70,
            "GPU Top-Down Cycle Breakdown",
            "=" * 70,
            f"\nDatabase: {self.database_path}",
            f"Total GPU Cycles: {self.total_cycles:,.0f}",
            "\n--- Level 1: Issue / Hidden / Exposed ---",
            f"  Issue:   {self.issue_cycles:>18,.0f} ({self.issue_percentage:>6.2f}%)",
            f"  Hidden:  {self.hidden_cycles:>18,.0f} ({self.hidden_percentage:>6.2f}%)",
            f"  Exposed: {self.exposed_cycles:>18,.0f} ({self.exposed_percentage:>6.2f}%)",
        ]

        lines.append("\n--- Level 2: Issue Breakdown ---")
        for mv in self.get_top_issue_types(10):
            lines.append(f"  {mv.label}: {mv.value:,.0f} ({mv.percentage_of_parent:.1f}% of issue)")

        lines.append("\n--- Level 2: Exposed Stall Breakdown ---")
        for mv in self.get_top_stall_reasons(10):
            lines.append(f"  {mv.label}: {mv.value:,.0f} ({mv.percentage_of_parent:.1f}% of exposed)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cycles": self.total_cycles,
            "level1": {
                "issue": {"value": self.issue_cycles, "percentage": self.issue_percentage},
                "hidden": {"value": self.hidden_cycles, "percentage": self.hidden_percentage},
                "exposed": {"value": self.exposed_cycles, "percentage": self.exposed_percentage},
            },
            "level2": {
                "issue": {k: {"label": v.label, "value": v.value, "pct": v.percentage_of_parent}
                         for k, v in self.level2_issue.items()},
                "exposed": {k: {"label": v.label, "value": v.value, "pct": v.percentage_of_parent}
                           for k, v in self.level2_exposed.items()},
            },
            "sunburst_data": self.sunburst_data,
        }

    def save_json(self, path: str) -> None:
        """Save result as JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_figure(self, path: str, label_threshold: float = 5.0) -> None:
        """Save result as 2-ring donut chart PNG.

        Args:
            path: Output file path
            label_threshold: Minimum percentage to show label on segment (default 5%)
        """
        import math
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(12, 10))

        # Level 1 (inner ring) - build data with labels
        l1_data = []  # (value, color, label, pct)
        if self.issue_cycles > 0:
            l1_data.append((self.issue_cycles, _COLORS["Issue"], "Issue", self.issue_percentage))
        if self.hidden_cycles > 0:
            l1_data.append((self.hidden_cycles, _COLORS["Hidden"], "Hidden", self.hidden_percentage))
        if self.exposed_cycles > 0:
            l1_data.append((self.exposed_cycles, _COLORS["Exposed"], "Exposed", self.exposed_percentage))

        l1_values = [d[0] for d in l1_data]
        l1_colors = [d[1] for d in l1_data]

        # Level 2 (outer ring) - build data with labels
        l2_data = []  # (value, color, label, pct_of_total)
        issue_items = sorted(self.level2_issue.items(), key=lambda x: -x[1].value)
        for i, (_, mv) in enumerate(issue_items):
            if mv.value > 0:
                # Extract short name (e.g., "VEC" from "VEC (Vector ALU)")
                short_label = mv.label.split(" ")[0]
                pct_of_total = mv.percentage_of_total
                l2_data.append((mv.value, _COLORS["issue_shades"][i % len(_COLORS["issue_shades"])],
                               short_label, pct_of_total))

        if self.hidden_cycles > 0:
            l2_data.append((self.hidden_cycles, _COLORS["Hidden"], "Hidden", self.hidden_percentage))

        exposed_items = sorted(self.level2_exposed.items(), key=lambda x: -x[1].value)
        for i, (_, mv) in enumerate(exposed_items):
            if mv.value > 0:
                short_label = mv.label.split(" ")[0]
                pct_of_total = mv.percentage_of_total
                l2_data.append((mv.value, _COLORS["exposed_shades"][i % len(_COLORS["exposed_shades"])],
                               short_label, pct_of_total))

        l2_values = [d[0] for d in l2_data]
        l2_colors = [d[1] for d in l2_data]

        # Draw rings
        if l2_values:
            ax.pie(l2_values, colors=l2_colors, radius=1.0,
                   wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2),
                   startangle=90, counterclock=False)
        if l1_values:
            ax.pie(l1_values, colors=l1_colors, radius=0.7,
                   wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2),
                   startangle=90, counterclock=False)

        # Add labels on segments above threshold
        def add_labels_on_ring(data_list, radius, threshold):
            """Add text labels on ring segments above threshold."""
            total = sum(d[0] for d in data_list)
            if total == 0:
                return
            angle = 90  # start angle in degrees
            for value, color, label, pct in data_list:
                sweep = 360 * value / total
                mid_angle = angle - sweep / 2
                if pct >= threshold:
                    # Position label at middle of the ring width
                    rad = math.radians(mid_angle)
                    x = radius * math.cos(rad)
                    y = radius * math.sin(rad)
                    # Use white text on dark backgrounds, black on light
                    text_color = 'white' if pct >= 20 else 'black'
                    ax.text(x, y, f"{label}\n{pct:.0f}%", ha='center', va='center',
                           fontsize=8, fontweight='bold', color=text_color)
                angle -= sweep

        # Labels on inner ring (Level 1) at radius 0.55 (middle of ring 0.4-0.7)
        add_labels_on_ring(l1_data, 0.55, threshold=label_threshold)
        # Labels on outer ring (Level 2) at radius 0.85 (middle of ring 0.7-1.0)
        add_labels_on_ring(l2_data, 0.85, threshold=label_threshold)

        # Center text
        ax.text(0, 0, f"Total\n{self.total_cycles:,.0f}\ncycles",
                ha='center', va='center', fontsize=14, fontweight='bold')

        # Legend
        legend = [mpatches.Patch(color='none', label='─── Level 1 ───')]
        if self.issue_cycles > 0:
            legend.append(mpatches.Patch(facecolor=_COLORS["Issue"],
                          label=f'Issue: {self.issue_percentage:.1f}%'))
        if self.hidden_cycles > 0:
            legend.append(mpatches.Patch(facecolor=_COLORS["Hidden"],
                          label=f'Hidden: {self.hidden_percentage:.1f}% (good)'))
        if self.exposed_cycles > 0:
            legend.append(mpatches.Patch(facecolor=_COLORS["Exposed"],
                          label=f'Exposed: {self.exposed_percentage:.1f}% (bad)'))

        if issue_items:
            legend.append(mpatches.Patch(color='none', label='─ Issue Types ─'))
            for i, (_, mv) in enumerate(issue_items[:5]):
                if mv.value > 0:
                    legend.append(mpatches.Patch(
                        facecolor=_COLORS["issue_shades"][i % len(_COLORS["issue_shades"])],
                        label=f'  {mv.label}: {mv.percentage_of_parent:.1f}%'))

        if exposed_items:
            legend.append(mpatches.Patch(color='none', label='─ Stall Reasons ─'))
            for i, (_, mv) in enumerate(exposed_items[:5]):
                if mv.value > 0:
                    legend.append(mpatches.Patch(
                        facecolor=_COLORS["exposed_shades"][i % len(_COLORS["exposed_shades"])],
                        label=f'  {mv.label}: {mv.percentage_of_parent:.1f}%'))

        ax.legend(handles=legend, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
        ax.set_title('GPU Top-Down Cycle Breakdown\n(Issue / Hidden / Exposed)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# Analyzer
# =============================================================================


class TopDownAnalyzer:
    """Analyzer for GPU top-down hierarchical cycle breakdown."""

    def __init__(self, reader: DatabaseReader):
        self._reader = reader

    def _get_metric(self, name: str, scope: str = "i") -> float:
        """Get aggregated metric value."""
        try:
            result = self._reader.get_profile_slices(
                cct_exp="application",
                profiles_exp="summary",
                metrics_exp=f"{name}:sum ({scope})",
            )
            if len(result) > 0:
                return float(result["value"].sum())
        except Exception:
            pass
        return 0.0

    def analyze(self, scope: str = "i") -> TopDownResult:
        """Perform top-down analysis."""
        db_path = self._reader.db_path

        # Level 0 & 1
        total = self._get_metric("gcycles", scope)
        issue = self._get_metric("gcycles:isu", scope)
        stall = self._get_metric("gcycles:stl", scope)
        hidden = self._get_metric("gcycles:stl:hid", scope)
        exposed = stall - hidden

        # Level 2: Issue breakdown
        level2_issue = {}
        for name, label in LEVEL2_ISSUE_METRICS.items():
            val = self._get_metric(name, scope)
            if val > 0:
                level2_issue[name] = MetricValue(
                    metric_name=name, label=label, value=val,
                    percentage_of_total=(val / total * 100) if total > 0 else 0,
                    percentage_of_parent=(val / issue * 100) if issue > 0 else 0,
                )

        # Level 2: Exposed stall breakdown
        level2_exposed = {}
        for name, label in LEVEL2_STALL_METRICS.items():
            val = self._get_metric(name, scope)
            if val > 0:
                level2_exposed[name] = MetricValue(
                    metric_name=name, label=label, value=val,
                    percentage_of_total=(val / total * 100) if total > 0 else 0,
                    percentage_of_parent=(val / exposed * 100) if exposed > 0 else 0,
                )

        # Build sunburst structure
        sunburst = self._build_sunburst(total, issue, hidden, exposed, level2_issue, level2_exposed)

        return TopDownResult(
            database_path=db_path, scope=scope, total_cycles=total,
            issue_cycles=issue, hidden_cycles=hidden, exposed_cycles=exposed,
            level2_issue=level2_issue, level2_exposed=level2_exposed,
            sunburst_data=sunburst,
        )

    def _build_sunburst(self, total, issue, hidden, exposed, l2_issue, l2_exposed) -> Dict:
        """Build sunburst data structure for D3.js visualization."""
        children = []
        if issue > 0:
            children.append({
                "name": "Issue", "value": issue,
                "percentage": (issue / total * 100) if total > 0 else 0,
                "color": "#2ecc71",
                "children": [{"name": mv.label, "value": mv.value, "percentage": mv.percentage_of_parent}
                             for mv in sorted(l2_issue.values(), key=lambda x: -x.value)],
            })
        if hidden > 0:
            children.append({
                "name": "Hidden", "value": hidden,
                "percentage": (hidden / total * 100) if total > 0 else 0,
                "color": "#1abc9c",
            })
        if exposed > 0:
            children.append({
                "name": "Exposed", "value": exposed,
                "percentage": (exposed / total * 100) if total > 0 else 0,
                "color": "#e74c3c",
                "children": [{"name": mv.label, "value": mv.value, "percentage": mv.percentage_of_parent}
                             for mv in sorted(l2_exposed.values(), key=lambda x: -x.value)],
            })
        return {"name": "GPU Cycles", "value": total, "children": children}


# =============================================================================
# Convenience Functions
# =============================================================================


def topdown_analysis(db_path: str, scope: str = "i") -> TopDownResult:
    """Perform top-down analysis on an HPCToolkit database."""
    reader = DatabaseReader(db_path)
    analyzer = TopDownAnalyzer(reader)
    return analyzer.analyze(scope)


def print_topdown_summary(db_path: str, scope: str = "i") -> None:
    """Print top-down analysis summary to stdout."""
    result = topdown_analysis(db_path, scope)
    print(result.format_summary())
