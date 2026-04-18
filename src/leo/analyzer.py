"""High-level API for GPU kernel performance analysis.

This module provides a simplified interface for running Leo's analysis pipeline.
Instead of manually orchestrating 10+ steps, users can run analysis in ~10 lines.

Example:
    from leo import KernelAnalyzer, AnalysisConfig

    config = AnalysisConfig(
        db_path="/path/to/hpctoolkit-database",
        gpubin_path="/path/to/kernel.gpubin",
    )
    analyzer = KernelAnalyzer(config)
    result = analyzer.analyze()

    for pc, blame, opcode in result.get_top_blame_sources(10):
        print(f"PC {pc:#x} ({opcode}): {blame:,.0f} cycles")
"""

import bisect
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from leo.analysis.backslice import BackSliceConfig, BackSliceEngine
from leo.analysis.blame import KernelBlameResult
from leo.analysis.debug_quality import QualityAssessment, assess_quality
from leo.analysis.speedup import (
    SpeedupEstimate,
    compute_speedup_estimates,
    format_speedup_report,
)
from leo.analysis.vma_property import VMAPropertyMap
from leo.arch import get_architecture
from leo.binary.debug_info import inspect_debug_info
from leo.binary.cfg import CFG, Function, build_cfg_from_instructions
from leo.binary.dependency import build_assign_pcs, prune_dead_dependencies, get_dependency_stats
from leo.binary.disasm import get_disassembler, Disassembler
from leo.binary.instruction import InstructionStat
from leo.binary.parser import get_parser, BinaryParser
from leo.output.table_formatter import TableBuilder
from leo.utils.location import extract_filename, format_source_location
from leo.utils.vendor import detect_vendor_from_arch_name


def format_opcode_with_details(op: str, details: Optional[Dict[str, Any]]) -> str:
    """Format opcode with operand details if available.

    For s_waitcnt instructions, appends counter values to the opcode string.

    Args:
        op: Base opcode string (e.g., "s_waitcnt")
        details: Parsed operand details dict (e.g., {"vmcnt": 0, "lgkmcnt": 0})

    Returns:
        Formatted opcode string (e.g., "s_waitcnt vmcnt(0) lgkmcnt(0)")

    Example:
        >>> format_opcode_with_details("s_waitcnt", {"vmcnt": 0, "lgkmcnt": 0})
        's_waitcnt vmcnt(0) lgkmcnt(0)'
    """
    if not details:
        return op

    # Handle s_waitcnt counters
    if op.lower().startswith("s_waitcnt"):
        parts = []
        for counter in ["vmcnt", "lgkmcnt", "expcnt", "vscnt"]:
            if counter in details:
                parts.append(f"{counter}({details[counter]})")
        if parts:
            return f"{op} {' '.join(parts)}"

    return op


# Intel opcodes are semantically opaque (e.g., "urb", "math", "send").
# Unlike AMD's "global_load_dwordx2" or NVIDIA's "LDG.E.64", Intel opcode
# names don't convey what kind of operation they represent.  This mapping
# adds a parenthetical annotation so that LLM evaluators (and humans) can
# understand the bottleneck type.
_INTEL_OPCODE_ANNOTATIONS: Dict[str, str] = {
    # Memory operations — URB is the primary memory path on Xe HPC
    "urb":    "global_mem",
    "send":   "mem",
    "sendc":  "mem",
    "sends":  "mem",
    "sendsc": "mem",
    # Synchronization
    "wait":   "mem_dep",
    "sync":   "sync",
    # Compute — disambiguate ALU subtypes
    "math":   "transcendental",
    "dpas":   "matrix",
    "dpasw":  "matrix",
}


def annotate_intel_opcode(op: str) -> str:
    """Add semantic annotation to an Intel opcode.

    Returns the opcode with a parenthetical hint, e.g. ``urb (global_mem)``.
    Non-Intel or unrecognised opcodes are returned unchanged.
    """
    label = _INTEL_OPCODE_ANNOTATIONS.get(op.lower())
    if label:
        return f"{op} ({label})"
    return op


@dataclass
class AnalysisConfig:
    """Configuration for kernel analysis.

    Attributes:
        db_path: Path to HPCToolkit database directory.
        gpubin_path: Path to GPUBIN/CUBIN binary file.
        arch: GPU architecture name ("a100", "v100", "mi250", "gfx90a", etc.).
              Default: "a100". The vendor is auto-detected from the arch name.
        vendor: Optional vendor override ("nvidia", "amd", or "intel"). If None, detected from arch.
        hpcstruct_path: Optional path to hpcstruct XML for source line mapping.
        function_name: Optional function name to analyze (auto-detected if None).
        enable_predicate_tracking: Enable predicate-aware analysis. Default: True.
        predicate_track_limit: Max DFS depth for predicate tracking. Default: 8.
        use_min_latency: Use minimum latency (optimistic). Default: True.
        apply_opcode_pruning: Apply opcode constraint pruning. Default: True.
        apply_barrier_pruning: Apply barrier constraint pruning. Default: True.
        apply_graph_latency_pruning: Apply latency-based graph pruning. Default: True.
        stall_threshold: Minimum stall cycles to consider. Default: 0.0.
        debug: Enable debug output. Default: False.
        module_id: HPCToolkit load-module numeric ID (int) for filtering CCT nodes.
            When specified, only instruction nodes from this module are included
            in the VMA property map, preventing stall cycles from other kernels
            being incorrectly attributed. This is the integer key from
            HPCToolkit's load_modules table (not a file path string).
            Default: None (no filtering).
    """

    db_path: str
    gpubin_path: str
    arch: str = "a100"
    vendor: Optional[str] = None
    hpcstruct_path: Optional[str] = None
    function_name: Optional[str] = None
    enable_predicate_tracking: bool = True
    predicate_track_limit: int = 8
    use_min_latency: bool = True
    apply_opcode_pruning: bool = True
    apply_barrier_pruning: bool = True
    apply_graph_latency_pruning: bool = True
    enable_execution_pruning: bool = False  # Disabled by default (matches GPA)
    stall_threshold: float = 0.0
    debug: bool = False
    module_id: Optional[int] = None  # HPCToolkit load-module numeric ID for filtering CCT nodes

    def __post_init__(self):
        """Auto-detect vendor from architecture name if not specified."""
        if self.vendor is None:
            # Try to detect vendor from arch name
            try:
                self.vendor = detect_vendor_from_arch_name(self.arch, default=None)
            except ValueError:
                # Fall back to file extension heuristic
                gpubin_lower = self.gpubin_path.lower()
                if gpubin_lower.endswith((".co", ".hsaco")):
                    self.vendor = "amd"
                elif gpubin_lower.endswith(".zebin"):
                    self.vendor = "intel"
                else:
                    self.vendor = "nvidia"
        self.vendor = self.vendor.lower()


@dataclass
class AnalysisResult:
    """Results from kernel analysis.

    Attributes:
        backslice_result: The raw KernelBlameResult with blame attribution.
        vma_map: VMA property map with profile data.
        instructions: Parsed instructions.
        cfg: Control flow graph.
        config: Analysis configuration used.
        source_mapping: PC to (filename, line) mapping (if hpcstruct provided).
        stats: Analysis statistics.
        quality: Quality assessment with warnings (if debug info missing, etc.)
    """

    backslice_result: KernelBlameResult
    vma_map: VMAPropertyMap
    instructions: List[InstructionStat]
    cfg: CFG
    config: "AnalysisConfig"
    source_mapping: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    _source_pcs_sorted: List[int] = field(default_factory=list)  # For nearest lookup
    stats: dict = field(default_factory=dict)
    quality: Optional[QualityAssessment] = None
    resolved_kernel_name: Optional[str] = None  # Set by program_analysis after offset-based resolution

    def __post_init__(self):
        """Build sorted PC list for nearest-neighbor source lookup."""
        if self.source_mapping and not self._source_pcs_sorted:
            self._source_pcs_sorted = sorted(self.source_mapping.keys())

    def to_json(self) -> dict:
        """Export analysis results as JSON (canonical raw data format).

        Returns a dict suitable for json.dumps(). This is the source of truth
        for all Leo output - other formats (summary table, graphs) are derived
        from this JSON representation.

        Returns:
            Dict with edges, chains, and metadata.
        """
        from leo.output.json_output import to_json_dict

        return to_json_dict(self, self.config)

    @property
    def total_stall_blame(self) -> float:
        """Total blame cycles distributed."""
        return self.backslice_result.total_stall_blame

    @property
    def total_stall_cycles(self) -> float:
        """Total stall cycles in profile."""
        return self.vma_map.get_total_stall_cycles()

    def get_top_blame_sources(self, n: int = 10) -> List[Tuple[int, float, str]]:
        """Get top N instructions receiving the most blame.

        Returns:
            List of (pc, blame_cycles, opcode) tuples sorted by blame descending.
        """
        return self.backslice_result.get_top_blame_sources(n)

    def get_top_stalling(self, n: int = 10) -> List[Tuple[int, float, str]]:
        """Get top N instructions with the most stall cycles.

        Returns:
            List of (pc, stall_cycles, opcode) tuples.
        """
        results = []
        for pc, prop in self.vma_map.get_top_stalling(n):
            opcode = prop.instruction.op if prop.instruction else "?"
            results.append((pc, prop.stall_cycles, opcode))
        return results

    def get_blame_edges(self, min_blame: float = 0) -> List[dict]:
        """Get blame edges with optional minimum threshold.

        Args:
            min_blame: Minimum blame cycles to include.

        Returns:
            List of dicts with src_pc, dst_pc, blame, src_opcode, dst_opcode.
        """
        pc_to_inst = {i.pc: i for i in self.instructions}
        edges = []
        for blame in self.backslice_result.blames:
            total = blame.total_blame()
            if total >= min_blame:
                src_inst = pc_to_inst.get(blame.src_pc)
                dst_inst = pc_to_inst.get(blame.dst_pc)
                src_op = format_opcode_with_details(
                    src_inst.op if src_inst else "?",
                    src_inst.operand_details if src_inst else None,
                )
                dst_op = format_opcode_with_details(
                    dst_inst.op if dst_inst else "?",
                    dst_inst.operand_details if dst_inst else None,
                )
                if self.config.vendor == "intel":
                    src_op = annotate_intel_opcode(src_op)
                    dst_op = annotate_intel_opcode(dst_op)
                edges.append({
                    "src_pc": blame.src_pc,
                    "dst_pc": blame.dst_pc,
                    "blame": total,
                    "src_opcode": src_op,
                    "dst_opcode": dst_op,
                    "src_source": self.get_source_location(blame.src_pc),
                    "dst_source": self.get_source_location(blame.dst_pc),
                })
        return sorted(edges, key=lambda e: -e["blame"])

    def get_top_blame_pairs(self, n: int = 10, min_blame: float = 0) -> List[dict]:
        """Get top N blame pairs showing stall location → root cause.

        Each pair shows where a stall was observed and what instruction caused it.

        Args:
            n: Maximum number of pairs to return.
            min_blame: Minimum blame cycles to include.

        Returns:
            List of dicts with:
                - stall_pc, stall_opcode, stall_source: Where stall was observed
                - cause_pc, cause_opcode, cause_source: Root cause instruction
                - blame: Cycles attributed to this edge
                - same_location: True if stall and cause are at same source line
        """
        edges = self.get_blame_edges(min_blame=min_blame)
        pairs = []
        for edge in edges[:n]:
            stall_src = edge["dst_source"]
            cause_src = edge["src_source"]
            same_location = (edge["src_pc"] == edge["dst_pc"]) or (stall_src == cause_src)
            pairs.append({
                "stall_pc": edge["dst_pc"],
                "stall_opcode": edge["dst_opcode"],
                "stall_source": stall_src,
                "cause_pc": edge["src_pc"],
                "cause_opcode": edge["src_opcode"],
                "cause_source": cause_src,
                "blame": edge["blame"],
                "same_location": same_location,
            })
        return pairs

    def format_blame_pairs(
        self, n: int = 10, min_blame: float = 0, only_different: bool = False
    ) -> str:
        """Format top blame pairs as a readable string.

        Args:
            n: Maximum number of pairs to show.
            min_blame: Minimum blame cycles to include.
            only_different: Only show pairs where stall and root cause differ.

        Returns:
            Formatted string showing stall → root cause pairs.
        """
        pairs = self.get_top_blame_pairs(n=n * 3 if only_different else n, min_blame=min_blame)
        if only_different:
            pairs = [p for p in pairs if not p["same_location"]][:n]

        if not pairs:
            return "  (none found)"

        lines = [
            "  Stall Location                    Root Cause                          Blame",
            "  " + "-" * 80,
        ]

        def format_loc(opcode: str, source: Optional[Tuple[str, int]]) -> str:
            if source:
                loc = format_source_location(source[0], source[1], short=True)
                return f"{opcode} @ {loc}"
            return opcode

        for pair in pairs:
            stall_str = format_loc(pair["stall_opcode"], pair["stall_source"])
            cause_str = format_loc(pair["cause_opcode"], pair["cause_source"])
            blame_str = f"{pair['blame']:,.0f}"

            if pair["same_location"]:
                # Self-blame: stall location is the root cause
                lines.append(f"  {stall_str:<33} (self)                              {blame_str:>10}")
            else:
                # Different: show the dependency
                lines.append(f"  {stall_str:<33} ← {cause_str:<28} {blame_str:>10}")

        return "\n".join(lines)

    def get_source_location(
        self,
        pc: int,
        nearest: bool = True,
        include_column: bool = False,
    ) -> Optional[Tuple[str, int] | Tuple[str, int, int]]:
        """Get source file and line (and optionally column) for a PC.

        Args:
            pc: Program counter.
            nearest: If True, fall back to nearest mapped PC when exact match fails.
                This handles gaps in hpcstruct VMA ranges (e.g., CALL instructions).
            include_column: If True, return (file, line, column) with column 0
                when not available.

        Returns:
            (filename, line_number) or (filename, line_number, column) tuple, or None.
        """
        if include_column:
            # Try VMA map (DWARF debug info) - exact match
            vma_prop = self.vma_map.get(pc)
            if vma_prop and vma_prop.has_source_info:
                return (vma_prop.source_file, vma_prop.source_line, vma_prop.source_column)

            # Try exact match from source_mapping (hpcstruct)
            loc = self.source_mapping.get(pc)
            if loc is not None and loc[1] > 0:
                return (loc[0], loc[1], 0)
        else:
            # Try exact match from source_mapping first (hpcstruct)
            loc = self.source_mapping.get(pc)
            if loc is not None and loc[1] > 0:
                return loc

            # Try VMA map (DWARF debug info) - exact match
            vma_prop = self.vma_map.get(pc)
            if vma_prop and vma_prop.has_source_info:
                return (vma_prop.source_file, vma_prop.source_line)

        # Fall back to nearest PC if enabled
        if nearest:
            # Try source_mapping nearest lookup first
            if self._source_pcs_sorted:
                idx = bisect.bisect_left(self._source_pcs_sorted, pc)

                # Find closest among left and right neighbors
                candidates = []
                if idx > 0:
                    candidates.append(self._source_pcs_sorted[idx - 1])
                if idx < len(self._source_pcs_sorted):
                    candidates.append(self._source_pcs_sorted[idx])

                if candidates:
                    # Pick the nearest PC (within reasonable distance, e.g., 64 bytes)
                    nearest_pc = min(candidates, key=lambda p: abs(p - pc))
                    if abs(nearest_pc - pc) <= 64:
                        loc = self.source_mapping[nearest_pc]
                        if loc[1] > 0:
                            if include_column:
                                return (loc[0], loc[1], 0)
                            return loc

            # Try VMA map nearest lookup (for DWARF info)
            # Find PCs with source info near the target PC
            vma_pcs_with_source = [
                p for p in self.vma_map.keys()
                if self.vma_map.get(p) and self.vma_map.get(p).has_source_info
                and abs(p - pc) <= 64
            ]
            if vma_pcs_with_source:
                nearest_pc = min(vma_pcs_with_source, key=lambda p: abs(p - pc))
                vma = self.vma_map.get(nearest_pc)
                if include_column:
                    return (vma.source_file, vma.source_line, vma.source_column)
                return (vma.source_file, vma.source_line)

        return None

    def get_speedup_estimates(self, top_n: int = 10) -> List[SpeedupEstimate]:
        """Get speedup estimates for top blamed instructions using Amdahl's Law.

        Based on GPA's validated speedup estimation methodology (4.1% avg error).

        Formula: speedup = total_cycles / (total_cycles - blamed_cycles × reducibility)

        Args:
            top_n: Number of top estimates to return.

        Returns:
            List of SpeedupEstimate objects sorted by estimated speedup.
        """
        # Aggregate blame by PC with opcode and blame_type
        blame_by_pc: Dict[int, Tuple[float, str, str]] = {}

        for blame in self.backslice_result.blames:
            pc = blame.src_pc
            total = blame.total_blame()

            if pc not in blame_by_pc:
                blame_by_pc[pc] = (0.0, blame.src_opcode, blame.blame_type)

            old_blame, opcode, blame_type = blame_by_pc[pc]
            blame_by_pc[pc] = (old_blame + total, opcode, blame_type)

        # Get total cycles from VMA map
        total_cycles = self.vma_map.get_total_cycles()

        return compute_speedup_estimates(
            blame_by_pc=blame_by_pc,
            total_cycles=total_cycles,
            source_mapping=self.source_mapping if self.source_mapping else None,
            top_n=top_n,
        )

    def format_speedup_estimates(self, top_n: int = 5) -> str:
        """Format speedup estimates as a human-readable string.

        Args:
            top_n: Number of top estimates to show.

        Returns:
            Formatted string showing optimization opportunities.
        """
        estimates = self.get_speedup_estimates(top_n=top_n)
        total_cycles = self.vma_map.get_total_cycles()
        return format_speedup_report(estimates, total_cycles)

    def summary(self, top_n: int = 10, aggregate: bool = True) -> str:
        """Get a text summary of the analysis results.

        Generates a unified table showing stall analysis with source locations,
        root causes, cycle counts, percentages, and speedup estimates.

        Args:
            top_n: Maximum number of rows to show in the table.
            aggregate: If True, aggregate rows by (stall_location, stall_opcode,
                cause_location, cause_opcode) to combine loop-unrolled instructions.

        Returns:
            Formatted summary string.
        """
        # Get kernel name and architecture
        # Prefer resolved_kernel_name (set by program_analysis via offset-based resolution,
        # critical for AMD where function_names[0] is an OCKL library function)
        kernel_name = self.resolved_kernel_name or self.config.function_name
        if not kernel_name and self.stats.get("function_names"):
            kernel_name = self.stats["function_names"][0]
        kernel_name = kernel_name or "unknown"

        # Demangle C++ kernel name for readability
        from leo.utils.demangle import demangle
        kernel_name = demangle(kernel_name)

        try:
            arch = get_architecture(self.config.arch)
            arch_display = f"{self.config.vendor.upper()} {arch.name}"
        except ValueError:
            arch_display = self.config.arch

        total_stall = self.total_stall_cycles
        total_execution = self.vma_map.get_total_cycles()  # For speedup calculation

        # Get more pairs than needed for aggregation, then trim after
        raw_pairs = self.get_top_blame_pairs(n=top_n * 10 if aggregate else top_n)

        # Aggregate pairs by (stall_location, stall_opcode, cause_location, cause_opcode)
        if aggregate:
            aggregated: Dict[tuple, dict] = {}
            for pair in raw_pairs:
                # Create key from source locations and opcodes
                stall_src = pair["stall_source"]
                cause_src = pair["cause_source"]
                key = (
                    stall_src,  # (file, line) or None
                    pair["stall_opcode"],
                    cause_src,  # (file, line) or None
                    pair["cause_opcode"],
                )
                if key not in aggregated:
                    aggregated[key] = {
                        "stall_pc": pair["stall_pc"],  # Keep first PC for fallback
                        "stall_opcode": pair["stall_opcode"],
                        "stall_source": stall_src,
                        "cause_pc": pair["cause_pc"],
                        "cause_opcode": pair["cause_opcode"],
                        "cause_source": cause_src,
                        "blame": 0.0,
                        "same_location": pair["same_location"],
                    }
                aggregated[key]["blame"] += pair["blame"]

            # Sort by blame descending and take top_n
            pairs = sorted(aggregated.values(), key=lambda p: -p["blame"])[:top_n]
        else:
            pairs = raw_pairs

        def format_source_full(source: Optional[Tuple[str, int, int]], pc: int) -> str:
            """Format source location without truncation, or fall back to PC."""
            if source:
                filename = extract_filename(source[0])
                return format_source_location(filename, source[1], source[2], short=False)
            return f"0x{pc:x}"

        # Pre-compute all source locations to determine column widths
        source_data = []
        for pair in pairs:
            stall_src = self.get_source_location(
                pair["stall_pc"], nearest=True, include_column=True
            )
            cause_src = self.get_source_location(
                pair["cause_pc"], nearest=True, include_column=True
            )
            stall_loc = format_source_full(stall_src, pair["stall_pc"])
            cause_loc = format_source_full(cause_src, pair["cause_pc"])
            source_data.append((stall_loc, cause_loc, pair))

        # Calculate dynamic column widths based on actual data
        W_CYCLES = 18    # Cycles column
        W_PCT = 8        # % Total column
        W_SPD = 8        # Speedup column

        # Find max location lengths (with minimum for headers)
        max_stall_loc = max((len(s[0]) for s in source_data), default=14)
        max_cause_loc = max((len(s[1]) for s in source_data), default=19)
        W_STALL_LOC = max(max_stall_loc + 1, len("Stall Location") + 1)
        W_CAUSE_LOC = max(max_cause_loc + 1, len("Root Cause Location") + 1)

        # Find max opcode lengths
        max_stall_op = max((len(p["stall_opcode"]) for p in pairs), default=10)
        max_cause_op = max((len(p["cause_opcode"]) for p in pairs), default=10)

        # Ensure minimum widths for headers
        W_STALL_OP = max(max_stall_op + 1, len("Stall Opcode"))  # +1 for padding
        W_ARROW = 5  # " <-- " is 5 chars
        W_CAUSE_OP = max(max_cause_op + 1, len("Root Opcode"))

        LINE_WIDTH = W_STALL_LOC + W_STALL_OP + W_ARROW + W_CAUSE_LOC + W_CAUSE_OP + W_CYCLES + W_PCT + W_SPD

        # Build table using TableBuilder
        table = TableBuilder(title="Leo GPU Performance Analysis")
        table.set_width(LINE_WIDTH)
        table.add_header()

        # Show quality warnings if any
        if self.quality and self.quality.warnings:
            for warning in self.quality.warnings:
                table.add_line(f"WARNING: {warning}")

        table.add_line(f"Kernel: {kernel_name:<50} Architecture: {arch_display}")
        table.add_line(f"Total Stall Cycles: {total_stall:,.0f}")
        sdc_after = self.stats.get("sdc_after_pruning", 0)
        sdc_before = self.stats.get("sdc_before_pruning", 0)
        if sdc_after > 0:
            table.add_line(f"Single Dependency Coverage: {sdc_after:.1%} (before: {sdc_before:.1%})")
        table.add_footer()
        table.add_blank_line()

        table.add_section_header("STALL ANALYSIS (PC Sampling \u2192 Back-slicing \u2192 Root Cause)")
        table.add_separator("-")

        # Column headers (custom format due to arrow column)
        header = (
            f"{'Stall Location':<{W_STALL_LOC}}"
            f"{'Stall Opcode':<{W_STALL_OP}}"
            f"{'':^{W_ARROW}}"
            f"{'Root Cause Location':<{W_CAUSE_LOC}}"
            f"{'Root Opcode':<{W_CAUSE_OP}}"
            f"{'Cycles':>{W_CYCLES}}"
            f"{'% Total':>{W_PCT}}"
            f"{'Speedup':>{W_SPD}}"
        )
        table.add_line(header)
        table.add_separator("-")

        # Data rows
        for stall_loc, cause_loc, pair in source_data:
            stall_op = pair["stall_opcode"]
            cause_op = pair["cause_opcode"]

            blame = pair["blame"]
            pct = (blame / total_stall * 100) if total_stall > 0 else 0
            # Amdahl's Law: speedup = total / (total - reducible)
            speedup = total_execution / (total_execution - blame) if total_execution > blame else 1.0

            row = (
                f"{stall_loc:<{W_STALL_LOC}}"
                f"{stall_op:<{W_STALL_OP}}"
                f"<-- "
                f"{cause_loc:<{W_CAUSE_LOC}}"
                f"{cause_op:<{W_CAUSE_OP}}"
                f"{blame:>{W_CYCLES},.0f}"
                f"{pct:>{W_PCT - 1}.1f}%"
                f"{speedup:>{W_SPD - 1}.2f}x"
            )
            table.add_line(row)

        table.add_separator("-")

        # Multi-hop blame chains (show chains with depth > 1)
        multi_hop_chains = [
            c for c in self.backslice_result.blame_chains if c.depth > 1
        ]
        if multi_hop_chains:
            table.add_blank_line()
            table.add_section_header("DEPENDENCY CHAINS (multi-hop root cause paths)")
            table.add_separator("-")

            for chain in multi_hop_chains[:top_n]:
                # Format: stall_op <- intermediate_op <- ... <- root_cause_op   cycles
                parts = []
                for node in chain.nodes:
                    op_display = node.opcode
                    if self.config.vendor == "intel":
                        op_display = annotate_intel_opcode(op_display)
                    loc = self.get_source_location(node.pc, nearest=True, include_column=True)
                    if loc:
                        filename = extract_filename(loc[0])
                        loc_str = format_source_location(filename, loc[1], loc[2] if len(loc) > 2 else 0, short=False)
                        parts.append(f"{op_display} @ {loc_str}")
                    else:
                        parts.append(f"{op_display}")
                chain_str = " \u2190 ".join(parts)
                blame_str = f"{chain.total_blame:,.0f}"
                table.add_line(f"  {chain_str}  {blame_str:>10}")

            table.add_separator("-")

        table.add_footer()

        return table.build()


def _collect_all_instructions(
    functions: list,  # List of ParsedFunction-like objects
    binary_parser: BinaryParser,
    debug: bool = False,
) -> Tuple[List[InstructionStat], Dict[str, int]]:
    """Collect all instructions from all functions, normalized to file offsets.

    Args:
        functions: Parsed functions from disassembler.
        binary_parser: Binary parser with section info.
        debug: Print debug information.

    Returns:
        Tuple of:
        - All instructions with PCs normalized to file offsets
        - Dict mapping function_name -> section_offset
    """
    all_instructions = []
    function_offsets = {}
    vendor = binary_parser.vendor

    for func in functions:
        # NVIDIA uses per-function sections: .text.<function_name>
        # AMD uses a single .text section with vaddr-based offsets
        # Intel uses per-function sections like NVIDIA: .text.<function_name>
        if vendor == "nvidia":
            section_name = f".text.{func.name}"
            section = binary_parser.sections.get(section_name)
            if section is None:
                if debug:
                    print(f"  WARNING: Section not found for {func.name}")
                continue
            section_offset = section.offset
        elif vendor == "intel":
            # Intel: Uses per-function sections like NVIDIA
            # Try both .text.<name> and bare .text
            section_name = f".text.{func.name}"
            section = binary_parser.sections.get(section_name)
            if section is None:
                section = binary_parser.sections.get(".text")
            if section is None:
                if debug:
                    print(f"  WARNING: Section not found for {func.name}")
                continue
            # Intel: Uses virtual addresses like AMD (section_offset = 0)
            # HPCToolkit CCT stores vaddrs for Intel, so they match directly
            section_offset = 0
        else:
            # AMD: Find the .text section
            section = binary_parser.sections.get(".text")
            if section is None:
                if debug:
                    print(f"  WARNING: .text section not found")
                continue
            # AMD: Keep PCs as vaddrs (from llvm-objdump output)
            # HPCToolkit CCT also stores vaddrs for AMD, so they match directly
            section_offset = 0  # Not used for AMD - PCs stay as vaddrs

        function_offsets[func.name] = section_offset if vendor == "nvidia" else section.offset

        if debug:
            print(f"  Function: {func.name}")
            print(f"    Instructions: {len(func.instructions)}")
            if vendor == "nvidia":
                print(f"    Section offset: 0x{section_offset:x}")
            else:
                print(f"    .text offset: 0x{section.offset:x}, vaddr: 0x{section.vaddr:x}")

        # Get function vaddr for Intel (only need to look up once per function)
        intel_func_vaddr = 0
        if vendor == "intel":
            func_info = binary_parser.functions.get(func.name)
            if func_info:
                intel_func_vaddr = func_info.offset  # func_info.offset is vaddr for Intel

        # Normalize PCs and set function_name
        for inst in func.instructions:
            if vendor == "nvidia":
                inst.pc += section_offset  # Convert to file offset
            elif vendor == "intel":
                # Intel GED disassembler returns section-relative offsets (0, 0x10, ...)
                # but HPCToolkit CCT stores virtual addresses (0x8000001007c0, ...)
                inst.pc += intel_func_vaddr
            # AMD: Keep PC as vaddr - llvm-objdump already outputs vaddrs
            # No conversion needed, they will match directly
            inst.function_name = func.name
            all_instructions.append(inst)

    return all_instructions, function_offsets


class KernelAnalyzer:
    """Unified interface for GPU kernel performance analysis.

    Orchestrates the complete Leo analysis pipeline:
    1. Parse binary (disassembly + control fields)
    2. Build CFG and dependency graph
    3. Join with profile data (VMA property map)
    4. Run back-slicing analysis
    5. Attribute blame to root cause instructions

    Supports both NVIDIA and AMD GPUs:
    - NVIDIA: Uses nvdisasm for disassembly, CubinParser for binary parsing
    - AMD: Uses llvm-objdump for disassembly, CodeObjectParser for binary parsing

    Example:
        config = AnalysisConfig(
            db_path="/path/to/database",
            gpubin_path="/path/to/kernel.gpubin",
            arch="a100",  # or "mi250" for AMD
        )
        analyzer = KernelAnalyzer(config)
        result = analyzer.analyze()

        print(result.summary())
        for pc, blame, opcode in result.get_top_blame_sources(10):
            print(f"PC {pc:#x}: {blame:,.0f} cycles")
    """

    def __init__(self, config: AnalysisConfig):
        """Initialize the analyzer.

        Args:
            config: Analysis configuration.

        Raises:
            FileNotFoundError: If required paths don't exist.
            RuntimeError: If required disassembler is not available.
        """
        self.config = config
        self._db_path = Path(config.db_path)
        self._gpubin_path = Path(config.gpubin_path)
        self._hpcstruct_path = Path(config.hpcstruct_path) if config.hpcstruct_path else None

        # Get vendor-appropriate disassembler
        self._disassembler: Disassembler = get_disassembler(config.vendor)

        self._validate()

        # Lazily populated
        self._parsed = None  # First parsed function
        self._binary_parser: Optional[BinaryParser] = None
        self._cfg: Optional[CFG] = None
        self._function_cfgs: Dict[str, CFG] = {}
        self._vma_map: Optional[VMAPropertyMap] = None
        self._source_mapping: Dict[int, Tuple[str, int]] = {}

    def _validate(self) -> None:
        """Validate configuration."""
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path}")
        if not self._gpubin_path.exists():
            raise FileNotFoundError(f"GPUBIN not found: {self._gpubin_path}")
        if self._hpcstruct_path and not self._hpcstruct_path.exists():
            raise FileNotFoundError(f"hpcstruct not found: {self._hpcstruct_path}")

        # Check that the disassembler is available
        if not self._disassembler.check_available():
            tool_name = self._disassembler.tool_name
            vendor = self.config.vendor
            if vendor == "nvidia":
                raise RuntimeError(f"{tool_name} is not available. Please install CUDA toolkit.")
            elif vendor == "intel":
                raise RuntimeError(
                    f"{tool_name} is not available. Please install GTPin or set GED_LIBRARY_PATH."
                )
            else:
                raise RuntimeError(f"{tool_name} is not available. Please install ROCm.")

    def analyze(self) -> AnalysisResult:
        """Run the complete analysis pipeline.

        Returns:
            AnalysisResult with blame attribution and analysis data.

        Raises:
            ValueError: If binary parsing fails.
            RuntimeError: If analysis fails.
        """
        vendor = self.config.vendor

        # Step 1: Disassemble and parse all functions
        # Intel uses disassemble_and_parse_all() for direct GED-based operand extraction
        # NVIDIA/AMD use the two-step disassemble() + parse_all_functions() flow
        if hasattr(self._disassembler, 'disassemble_and_parse_all'):
            all_functions = self._disassembler.disassemble_and_parse_all(str(self._gpubin_path))
        else:
            disasm_output = self._disassembler.disassemble(str(self._gpubin_path))
            all_functions = self._disassembler.parse_all_functions(disasm_output)
        if not all_functions:
            raise ValueError(f"No functions found in binary: {self._gpubin_path}")

        # Check for debug info in binary
        debug_info = inspect_debug_info(str(self._gpubin_path))

        if self.config.debug:
            print(f"[Analyzer Debug] Step 1: Binary parsed ({vendor})")
            print(f"  Total functions: {len(all_functions)}")
            for func in all_functions:
                print(f"    {func.name}: {len(func.instructions)} instructions")
            print(f"  Debug info level: {debug_info.level.value}")
            if debug_info.debug_sections:
                print(f"  Debug sections: {debug_info.debug_sections}")

        # Step 2: Parse binary for section info
        self._binary_parser = get_parser(vendor, str(self._gpubin_path))

        # Step 3: Populate control fields for each function (NVIDIA only)
        # Must happen BEFORE _collect_all_instructions() which modifies inst.pc
        # from section-relative offsets to file offsets. populate_instruction_controls()
        # uses inst.pc as a section-relative offset to index into raw section data.
        if vendor == "nvidia" and hasattr(self._binary_parser, "populate_instruction_controls"):
            for func in all_functions:
                section_name = f".text.{func.name}"
                if section_name in self._binary_parser.sections:
                    self._binary_parser.populate_instruction_controls(
                        func.instructions, section_name=section_name
                    )

        # Step 4: Collect all instructions
        # NVIDIA: normalized to file offsets
        # AMD: kept as vaddrs (matches HPCToolkit CCT offsets)
        all_instructions, _function_offsets = _collect_all_instructions(
            all_functions, self._binary_parser, debug=self.config.debug
        )

        if self.config.debug:
            print(f"[Analyzer Debug] Step 2: Instructions normalized")
            print(f"  Total instructions: {len(all_instructions)}")

        # Step 5: Build per-function CFGs
        self._function_cfgs: Dict[str, CFG] = {}
        for func in all_functions:
            cfg = build_cfg_from_instructions(func.instructions, func.labels)
            self._function_cfgs[func.name] = cfg

        if self.config.debug:
            total_blocks = sum(len(cfg.function.blocks) for cfg in self._function_cfgs.values())
            print(f"[Analyzer Debug] Step 3: CFGs built")
            print(f"  Total blocks: {total_blocks}")

        # Build a merged CFG containing all functions' blocks for latency pruning.
        # The BackSliceEngine needs to look up blocks by VMA across all functions.
        self._cfg = self._build_merged_cfg(all_functions)

        # Store the first parsed function for backward compatibility
        self._parsed = all_functions[0]

        # Step 6: Build dependency graph (assign_pcs)
        # Use CFG-aware reaching definitions when CFGs are available
        if self._function_cfgs:
            for func_name, func_cfg in self._function_cfgs.items():
                func_insts = func_cfg.get_all_instructions()
                build_assign_pcs(func_insts, cfg=func_cfg)
            # Handle any instructions not covered by a CFG
            cfg_pcs = set()
            for func_cfg in self._function_cfgs.values():
                for inst in func_cfg.get_all_instructions():
                    cfg_pcs.add(inst.pc)
            uncovered = [i for i in all_instructions if i.pc not in cfg_pcs]
            if uncovered:
                build_assign_pcs(uncovered)  # linear fallback
        else:
            build_assign_pcs(all_instructions)

        # Step 6b: Prune false dependencies using liveness analysis (safety check)
        total_pruned = 0
        for func_name, func_cfg in self._function_cfgs.items():
            func_insts = func_cfg.get_all_instructions()
            pruned = prune_dead_dependencies(func_insts, func_cfg)
            total_pruned += pruned

        if self.config.debug:
            deps_count = sum(len(i.assign_pcs) for i in all_instructions)
            print(f"[Analyzer Debug] Step 4: Dependencies computed")
            print(f"  Total dependency edges: {deps_count}")
            if total_pruned > 0:
                print(f"  Liveness pruning: removed {total_pruned} false deps")

        # Step 7: Build VMA property map - PCs are already file offsets, so offset=0
        self._vma_map = VMAPropertyMap.build(
            db_path=str(self._db_path),
            instructions=all_instructions,
            arch_name=self.config.arch,
            section_file_offset=0,  # PCs are already file offsets!
            debug=self.config.debug,
            binary_path=str(self._gpubin_path),  # For DWARF source locations
            module_id=self.config.module_id,  # Filter CCT nodes by module
        )

        # Step 8: Load source mapping (optional)
        # With normalized PCs (file offsets), we don't need section_offset adjustment
        if self._hpcstruct_path:
            self._source_mapping = self._parse_hpcstruct(
                str(self._hpcstruct_path),
                section_offset=0,  # PCs are already file offsets
            )

        # Step 9: Create backslice config (pass all parameters to match exactly)
        bs_config = BackSliceConfig(
            arch_name=self.config.arch,
            enable_predicate_tracking=self.config.enable_predicate_tracking,
            predicate_track_limit=self.config.predicate_track_limit,
            use_min_latency=self.config.use_min_latency,
            apply_opcode_pruning=self.config.apply_opcode_pruning,
            apply_barrier_pruning=self.config.apply_barrier_pruning,
            apply_graph_latency_pruning=self.config.apply_graph_latency_pruning,
            enable_execution_pruning=self.config.enable_execution_pruning,
            stall_threshold=self.config.stall_threshold,
            debug=self.config.debug,
        )

        # Step 10: Run back-slicing analysis
        engine = BackSliceEngine(
            vma_map=self._vma_map,
            instructions=all_instructions,
            cfg=self._cfg,
            config=bs_config,
        )
        backslice_result = engine.analyze()

        # Step 11: Collect stats
        dep_stats = get_dependency_stats(all_instructions)
        total_blocks = sum(len(cfg.function.blocks) for cfg in self._function_cfgs.values())
        stats = {
            "function_names": [f.name for f in all_functions],
            "num_functions": len(all_functions),
            "num_instructions": len(all_instructions),
            "num_blocks": total_blocks,
            "dependencies": dep_stats,
            "initial_nodes": engine.stats.initial_nodes,
            "initial_edges": engine.stats.initial_edges,
            "final_edges": engine.stats.final_edges,
            "edges_pruned_opcode": engine.stats.edges_pruned_opcode,
            "edges_pruned_latency": engine.stats.edges_pruned_latency,
            "edges_pruned_barrier": engine.stats.edges_pruned_barrier,
            "sdc_before_pruning": engine.stats.sdc_before_pruning,
            "sdc_after_pruning": engine.stats.sdc_after_pruning,
        }

        # Step 12: Assess analysis quality
        quality = assess_quality(
            self._vma_map,
            debug_info=debug_info,
            source_mapping=self._source_mapping if self._source_mapping else None,
        )

        if self.config.debug and quality.warnings:
            print(f"[Analyzer Debug] Quality warnings:")
            for warning in quality.warnings:
                print(f"  {warning}")

        return AnalysisResult(
            backslice_result=backslice_result,
            vma_map=self._vma_map,
            instructions=all_instructions,
            cfg=self._cfg,
            config=self.config,
            source_mapping=self._source_mapping,
            stats=stats,
            quality=quality,
        )

    def _build_merged_cfg(self, all_functions: list) -> CFG:
        """Build a merged CFG containing blocks from all functions.

        The latency pruning needs to find blocks by VMA across the entire
        binary, not just the first function. This merges all per-function
        blocks into a single CFG for block lookup.
        """
        all_blocks = []
        for func in all_functions:
            cfg = self._function_cfgs.get(func.name)
            if cfg:
                all_blocks.extend(cfg.function.blocks)

        merged_func = Function(
            name="__merged__",
            blocks=all_blocks,
            entry_block_id=all_blocks[0].id if all_blocks else 0,
        )
        return CFG(merged_func)

    @staticmethod
    def _parse_hpcstruct(hpcstruct_path: str, section_offset: int) -> Dict[int, Tuple[str, int]]:
        """Parse hpcstruct XML to build PC -> (file, line) mapping."""
        tree = ET.parse(hpcstruct_path)
        root = tree.getroot()
        pc_to_source: Dict[int, Tuple[str, int]] = {}

        def parse_vma_ranges(v_str: str) -> List[Tuple[int, int]]:
            ranges = []
            for match in re.finditer(r'\[0x([0-9a-fA-F]+)-0x([0-9a-fA-F]+)\)', v_str):
                start = int(match.group(1), 16)
                end = int(match.group(2), 16)
                ranges.append((start, end))
            return ranges

        def process_element(elem, current_file: str) -> None:
            nonlocal pc_to_source
            if elem.tag == 'F':
                current_file = elem.get('n', current_file)
            elif elem.tag == 'A':
                f = elem.get('f')
                if f:
                    current_file = f

            if elem.tag == 'S':
                line = elem.get('l')
                v = elem.get('v', '')
                if line and v:
                    for start, end in parse_vma_ranges(v):
                        for vma in range(start, end, 16):
                            pc = vma - section_offset
                            if pc >= 0:
                                pc_to_source[pc] = (current_file, int(line))

            for child in elem:
                process_element(child, current_file)

        for lm in root.findall('.//LM'):
            for child in lm:
                process_element(child, '')

        return pc_to_source

    # Convenience properties for accessing intermediate results
    @property
    def parsed_function(self):
        """Get the first parsed function (available after analyze())."""
        return self._parsed

    @property
    def cfg(self) -> Optional[CFG]:
        """Get the control flow graph (available after analyze())."""
        return self._cfg

    @property
    def vma_map(self) -> Optional[VMAPropertyMap]:
        """Get the VMA property map (available after analyze())."""
        return self._vma_map

    @property
    def source_mapping(self) -> Dict[int, Tuple[str, int]]:
        """Get the PC to source location mapping."""
        return self._source_mapping


def analyze_kernel(
    db_path: str,
    gpubin_path: str,
    arch: str = "a100",
    hpcstruct_path: Optional[str] = None,
    **kwargs,
) -> AnalysisResult:
    """Convenience function for one-line analysis.

    Args:
        db_path: Path to HPCToolkit database.
        gpubin_path: Path to GPUBIN file.
        arch: GPU architecture name.
        hpcstruct_path: Optional path to hpcstruct for source mapping.
        **kwargs: Additional options passed to AnalysisConfig.

    Returns:
        AnalysisResult with blame attribution.

    Example:
        result = analyze_kernel(
            db_path="/path/to/database",
            gpubin_path="/path/to/kernel.gpubin",
        )
        print(result.summary())
    """
    config = AnalysisConfig(
        db_path=db_path,
        gpubin_path=gpubin_path,
        arch=arch,
        hpcstruct_path=hpcstruct_path,
        **kwargs,
    )
    analyzer = KernelAnalyzer(config)
    return analyzer.analyze()
