"""VMA Property Map - joins binary analysis with profile data.

Based on GPA's VMAProperty structure from GPUAdvisor.hpp (lines 144-163)
and initialization from GPUAdvisor-Init.cpp (lines 292-412).

The VMA (Virtual Memory Address) Property Map is the critical data structure
that joins:
1. Static binary data (from nvdisasm): InstructionStat with assign_pcs
2. Dynamic profile data (from HPCToolkit): CCT nodes with metrics
3. Architecture info: latency tables

Join Key: InstructionStat.pc == CCT.offset (both are offsets within function)

Example:
    Binary:    LDG R0, [R2] at pc=0x320
    CCT:       instruction node at offset=0x320, cct_id=42
    Metrics:   gcycles:stl:mem=50000 for cct_id=42
    Result:    VMAProperty linking all three

This enables back-slicing to attribute blame correctly based on actual
runtime performance metrics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

from leo.binary.instruction import InstructionStat
from leo.db.reader import DatabaseReader
from leo.arch import GPUArchitecture, get_architecture
from leo.analysis.metrics import get_memory_stall_cycles, detect_vendor_from_arch_name, Vendor
from leo.utils.location import format_source_location
from leo.constants.metrics import (
    METRIC_GCYCLES,
    METRIC_GCYCLES_ISU,
    METRIC_GCYCLES_STL,
    METRIC_GCYCLES_STL_IDEP,
    METRIC_GCYCLES_STL_IFET,
    METRIC_GCYCLES_STL_GMEM,
    METRIC_GCYCLES_STL_LMEM,
    METRIC_GCYCLES_STL_MEM,
    METRIC_GCYCLES_STL_PIPE,
    METRIC_GCYCLES_STL_SYNC,
    METRIC_GCYCLES_STL_TMEM,
)

logger = logging.getLogger(__name__)


@dataclass
class VMAProperty:
    """Property entry for one instruction VMA, joining binary and profile data.

    This is the core data structure for blame analysis. It contains:
    - Static info: parsed instruction with registers and dependencies
    - Dynamic info: runtime metrics from profiling
    - Architecture info: latency estimates
    - Source info: file, line, and column from DWARF debug info

    Attributes:
        vma: Virtual Memory Address (PC offset within function)
        instruction: Parsed instruction from nvdisasm (may be None if not found)
        cct_id: CCT node ID for this instruction (-1 if no profile data)
        prof_metrics: Dictionary of metric_name -> value from profiling
        function_name: Name of containing function
        block_id: Basic block ID (if CFG available)
        latency_lower: Minimum latency from architecture (best case)
        latency_upper: Maximum latency from architecture (worst case)
        latency_issue: Issue cycles (throughput)
        has_profile_data: True if this instruction was sampled
        source_file: Source filename from DWARF debug info
        source_line: Source line number (1-indexed, 0 = unknown)
        source_column: Source column number (0 = unknown)
    """

    vma: int = 0
    instruction: Optional[InstructionStat] = None
    cct_id: int = -1
    prof_metrics: Dict[str, float] = field(default_factory=dict)
    function_name: str = ""
    block_id: int = -1
    latency_lower: int = 0
    latency_upper: int = 0
    latency_issue: int = 0
    has_profile_data: bool = False
    vendor: Vendor = "nvidia"
    # Source location from DWARF debug info
    source_file: str = ""
    source_line: int = 0
    source_column: int = 0

    # Convenience accessors for common metrics
    @property
    def total_cycles(self) -> float:
        """Total GPU cycles at this instruction."""
        return self.prof_metrics.get(METRIC_GCYCLES, 0.0)

    @property
    def stall_cycles(self) -> float:
        """Total stall cycles (sum of all stall types)."""
        return self.prof_metrics.get(METRIC_GCYCLES_STL, 0.0)

    @property
    def issue_cycles(self) -> float:
        """Issue cycles (not stalled)."""
        return self.prof_metrics.get(METRIC_GCYCLES_ISU, 0.0)

    @property
    def memory_stall_cycles(self) -> float:
        """Memory-related stall cycles (vendor-aware)."""
        return get_memory_stall_cycles(self.prof_metrics, self.vendor)

    @property
    def sync_stall_cycles(self) -> float:
        """Synchronization stall cycles."""
        return self.prof_metrics.get(METRIC_GCYCLES_STL_SYNC, 0.0)

    @property
    def dependency_stall_cycles(self) -> float:
        """Execution dependency stall cycles (RAW hazards)."""
        return self.prof_metrics.get(METRIC_GCYCLES_STL_IDEP, 0.0)

    def get_stall_breakdown(self) -> Dict[str, float]:
        """Get breakdown of stall types.

        Returns vendor-appropriate stall metrics.
        All vendors use standard HPCToolkit metric names:
        - mem, gmem, lmem, tmem, sync, idep, ifet, pipe
        """
        breakdown = {}

        # All vendors use the same standard HPCToolkit stall metrics
        stall_prefixes = [
            METRIC_GCYCLES_STL_MEM,
            METRIC_GCYCLES_STL_GMEM,
            METRIC_GCYCLES_STL_LMEM,
            METRIC_GCYCLES_STL_TMEM,
            METRIC_GCYCLES_STL_SYNC,
            METRIC_GCYCLES_STL_IDEP,
            METRIC_GCYCLES_STL_IFET,
            METRIC_GCYCLES_STL_PIPE,
        ]
        for prefix in stall_prefixes:
            if prefix in self.prof_metrics:
                breakdown[prefix] = self.prof_metrics[prefix]
        return breakdown

    def is_stalling(self, threshold: float = 0.0) -> bool:
        """Check if this instruction has significant stalls."""
        return self.stall_cycles > threshold

    # Source location convenience methods
    @property
    def has_source_info(self) -> bool:
        """Check if source location info is available."""
        return bool(self.source_file and self.source_line > 0)

    @property
    def has_column_info(self) -> bool:
        """Check if column info is available (more precise than line)."""
        return self.has_source_info and self.source_column > 0

    @property
    def source_location(self) -> str:
        """Format source location as file:line:column string."""
        if not self.has_source_info:
            return ""
        return format_source_location(
            self.source_file,
            self.source_line,
            self.source_column,
            short=False,
        )

    @property
    def short_source_location(self) -> str:
        """Format source location with just filename (no path)."""
        if not self.has_source_info:
            return ""
        return format_source_location(
            self.source_file,
            self.source_line,
            self.source_column,
            short=True,
        )


class VMAPropertyMap:
    """Map of VMA -> VMAProperty for a kernel/function.

    This class manages the joined view of binary and profile data.
    It is built from:
    1. Instructions parsed from nvdisasm
    2. CCT data from HPCToolkit database
    3. Architecture specification for latencies

    Usage:
        # Build from database and instructions
        vma_map = VMAPropertyMap.build(
            db_path="/path/to/database",
            instructions=parsed_instructions,
            arch_name="a100"
        )

        # Query properties
        prop = vma_map.get(0x320)
        if prop and prop.has_profile_data:
            print(f"Stall cycles: {prop.stall_cycles}")

        # Find hotspots
        for pc, prop in vma_map.get_top_stalling(n=10):
            print(f"PC {pc:#x}: {prop.stall_cycles} stall cycles")
    """

    def __init__(self):
        self._map: Dict[int, VMAProperty] = {}
        self._arch: Optional[GPUArchitecture] = None
        self._function_name: str = ""
        self._vendor: Vendor = "nvidia"

    def __getitem__(self, pc: int) -> VMAProperty:
        """Get property by PC, raises KeyError if not found."""
        return self._map[pc]

    def __contains__(self, pc: int) -> bool:
        """Check if PC exists in map."""
        return pc in self._map

    def __iter__(self):
        """Iterate over PCs."""
        return iter(self._map)

    def __len__(self) -> int:
        """Number of entries."""
        return len(self._map)

    def get(self, pc: int, default: Optional[VMAProperty] = None) -> Optional[VMAProperty]:
        """Get property by PC with optional default."""
        return self._map.get(pc, default)

    def items(self):
        """Iterate over (pc, property) pairs."""
        return self._map.items()

    def values(self):
        """Iterate over properties."""
        return self._map.values()

    def keys(self):
        """Iterate over PCs."""
        return self._map.keys()

    @property
    def arch(self) -> Optional[GPUArchitecture]:
        """Architecture used for latency lookup."""
        return self._arch

    @property
    def function_name(self) -> str:
        """Name of the function this map covers."""
        return self._function_name

    @property
    def vendor(self) -> Vendor:
        """GPU vendor (nvidia or amd)."""
        return self._vendor

    # Statistics methods
    def get_total_cycles(self) -> float:
        """Sum of all cycles across instructions."""
        return sum(p.total_cycles for p in self._map.values())

    def get_total_stall_cycles(self) -> float:
        """Sum of all stall cycles across instructions."""
        return sum(p.stall_cycles for p in self._map.values())

    def get_sampled_count(self) -> int:
        """Number of instructions with profile data."""
        return sum(1 for p in self._map.values() if p.has_profile_data)

    def get_top_stalling(self, n: int = 10) -> List[Tuple[int, VMAProperty]]:
        """Get top N stalling instructions by stall cycles.

        Args:
            n: Number of instructions to return.

        Returns:
            List of (pc, property) tuples sorted by stall cycles descending.
        """
        sorted_props = sorted(
            self._map.items(), key=lambda x: x[1].stall_cycles, reverse=True
        )
        return sorted_props[:n]

    def get_stalling_instructions(
        self, threshold: float = 0.0
    ) -> List[Tuple[int, VMAProperty]]:
        """Get all instructions with stall cycles above threshold.

        Args:
            threshold: Minimum stall cycles to include.

        Returns:
            List of (pc, property) tuples sorted by stall cycles descending.
        """
        stalling = [(pc, p) for pc, p in self._map.items() if p.stall_cycles > threshold]
        return sorted(stalling, key=lambda x: x[1].stall_cycles, reverse=True)

    def get_stall_type_breakdown(self) -> Dict[str, float]:
        """Get aggregate stall breakdown across all instructions."""
        breakdown: Dict[str, float] = {}
        for prop in self._map.values():
            for metric, value in prop.get_stall_breakdown().items():
                breakdown[metric] = breakdown.get(metric, 0.0) + value
        return breakdown

    @classmethod
    def build(
        cls,
        db_path: str,
        instructions: List[InstructionStat],
        arch_name: str = "a100",
        function_filter: Optional[str] = None,
        section_file_offset: int = 0,
        debug: bool = False,
        binary_path: Optional[str] = None,
        module_id: Optional[int] = None,
    ) -> "VMAPropertyMap":
        """Build VMA property map by joining binary and profile data.

        This is the main factory method that:
        1. Opens the HPCToolkit database
        2. Queries CCT for instruction nodes
        3. Queries profile metrics
        4. Joins with binary instructions by PC offset
        5. Adds architecture latency information
        6. (Optional) Adds DWARF source location info with column

        Args:
            db_path: Path to HPCToolkit database directory.
            instructions: List of InstructionStat from nvdisasm parsing.
            arch_name: Architecture name for latency lookup ("a100", "v100").
            function_filter: Optional function name to filter CCT nodes.
            section_file_offset: File offset of the .text section in CUBIN.
                CCT offsets in HPCToolkit are file offsets, not section-relative.
                Pass the section's file offset (from CubinSection.offset) to
                compute section-relative PCs: pc = cct_offset - section_file_offset.
            debug: If True, print diagnostic information at key checkpoints.
            binary_path: Optional path to GPU binary for DWARF source info.
                If provided, source file/line/column will be extracted from
                DWARF debug info and added to VMAProperty entries.
            module_id: Optional module ID to filter CCT nodes. When specified,
                only instruction nodes from this module are included. This prevents
                stall cycles from other kernels being incorrectly attributed.

        Returns:
            Populated VMAPropertyMap instance.
        """
        result = cls()
        result._arch = get_architecture(arch_name)
        result._vendor = detect_vendor_from_arch_name(arch_name)

        # Step 1: Index instructions by PC for O(1) lookup
        inst_by_pc: Dict[int, InstructionStat] = {inst.pc: inst for inst in instructions}

        if debug:
            min_pc = min(inst_by_pc.keys()) if inst_by_pc else 0
            max_pc = max(inst_by_pc.keys()) if inst_by_pc else 0
            logger.debug(f"[VMA Debug] Step 1: Binary instructions indexed")
            logger.debug(f"  Instructions: {len(inst_by_pc)}")
            logger.debug(f"  PC range: 0x{min_pc:x} - 0x{max_pc:x}")
            logger.debug(f"  Section file offset: 0x{section_file_offset:x}")

        # Step 1.5: Parse DWARF line info for source locations (optional)
        dwarf_locations: Dict[int, "SourceLocation"] = {}
        if binary_path:
            try:
                from leo.binary.dwarf_line import DWARFLineParser

                dwarf_parser = DWARFLineParser(binary_path)
                if dwarf_parser.has_dwarf_info:
                    dwarf_locations = dwarf_parser.get_all_locations()
                    if debug:
                        stats = dwarf_parser.get_column_statistics()
                        logger.debug(f"[VMA Debug] Step 1.5: DWARF line info loaded")
                        logger.debug(f"  Source locations: {stats['total']}")
                        logger.debug(f"  With column info: {stats['with_column']} ({stats['column_coverage_pct']:.1f}%)")
            except Exception as e:
                if debug:
                    logger.debug(f"[VMA Debug] Step 1.5: DWARF parsing failed: {e}")

        # Step 2: Open database
        db = DatabaseReader(db_path)

        # Step 3: Get metric ID to name mapping
        all_metrics = db.get_metrics("*")
        metric_id_to_name: Dict[int, str] = dict(
            zip(all_metrics["id"], all_metrics["name"])
        )

        if debug:
            stall_metrics = [n for n in metric_id_to_name.values() if "stl" in n]
            logger.debug(f"[VMA Debug] Step 3: Metrics loaded")
            logger.debug(f"  Total metrics: {len(metric_id_to_name)}")
            logger.debug(f"  Stall metrics: {len(stall_metrics)}")

        # Step 4: Query CCT for instruction nodes
        cct = db.get_cct("*")
        instruction_nodes = cct[cct["type"] == "instruction"]

        if debug:
            logger.debug(f"[VMA Debug] Step 4: CCT queried")
            logger.debug(f"  Total CCT nodes: {len(cct)}")
            logger.debug(f"  Instruction nodes: {len(instruction_nodes)}")
            if not instruction_nodes.empty:
                offsets = instruction_nodes["offset"].dropna()
                if len(offsets) > 0:
                    logger.debug(f"  CCT offset range: 0x{int(offsets.min()):x} - 0x{int(offsets.max()):x}")
                    # Show sample offsets and their converted PCs
                    sample_offsets = offsets.head(5).tolist()
                    logger.debug(f"  Sample CCT offsets -> PCs:")
                    for off in sample_offsets:
                        pc = int(off) - section_file_offset
                        match = "MATCH" if pc in inst_by_pc else "NO MATCH"
                        logger.debug(f"    0x{int(off):x} -> 0x{pc:x} ({match})")

        if instruction_nodes.empty:
            # No instruction-level nodes (standard HPCToolkit database)
            # Fall back to building from binary only
            if debug:
                logger.warning("[VMA Debug] WARNING: No instruction nodes, using binary-only mode")
            return cls._build_from_binary_only(instructions, result._arch, result._vendor)

        # Step 5: Query profile data (summary profile)
        try:
            profile_slices = db.get_profile_slices("*", "summary", "*")
        except Exception:
            profile_slices = pd.DataFrame()

        if debug:
            logger.debug(f"[VMA Debug] Step 5: Profile data queried")
            logger.debug(f"  Profile slices: {len(profile_slices)}")

        # Step 6: Build the join - with diagnostic tracking
        matched_count = 0
        skipped_offset_mismatch = 0
        skipped_module_mismatch = 0
        skipped_function_filter = 0
        total_stall_cycles = 0.0
        sample_skipped: List[Tuple[int, int, int]] = []  # (cct_id, cct_offset, computed_pc)

        for cct_id, cct_row in instruction_nodes.iterrows():
            cct_offset = cct_row["offset"]

            # Convert CCT offset (file offset) to section-relative PC
            # CCT stores file offsets, nvdisasm gives section-relative PCs
            pc = cct_offset - section_file_offset

            # Skip if instruction not in our binary (wrong section or out of range)
            if pc not in inst_by_pc:
                skipped_offset_mismatch += 1
                if len(sample_skipped) < 5:
                    sample_skipped.append((cct_id, int(cct_offset), pc))
                continue

            # Skip if instruction is from wrong module (when module_id specified)
            if module_id is not None:
                cct_module_id = cct_row.get("module_path")
                if cct_module_id is not None and int(cct_module_id) != module_id:
                    skipped_module_mismatch += 1
                    continue

            inst = inst_by_pc[pc]

            # Get function name - prefer from instruction, fallback to CCT
            if inst.function_name:
                function_name = inst.function_name
            else:
                function_name = _get_function_name(cct, cct_id)

            # Apply function filter if specified
            if function_filter and function_name != function_filter:
                skipped_function_filter += 1
                continue

            if not result._function_name:
                result._function_name = function_name

            # Look up profile metrics for this instruction
            prof_metrics = _get_metrics_for_cct_node(
                profile_slices, cct_id, metric_id_to_name
            )

            # Track stall cycles for debugging
            stall = prof_metrics.get(METRIC_GCYCLES_STL, 0.0)
            total_stall_cycles += stall

            # Build VMAProperty with source info if available
            source_file, source_line, source_column = "", 0, 0
            # For AMD, DWARF addresses are vaddrs which match instruction PCs directly
            # For NVIDIA, we may need to convert (CCT offset == file offset)
            dwarf_key = pc if result._vendor == "amd" else cct_offset
            if dwarf_key in dwarf_locations:
                loc = dwarf_locations[dwarf_key]
                source_file = loc.file
                source_line = loc.line
                source_column = loc.column

            prop = VMAProperty(
                vma=pc,
                instruction=inst,
                cct_id=cct_id,
                prof_metrics=prof_metrics,
                function_name=function_name,
                block_id=-1,  # Would need CFG to populate
                latency_lower=result._arch.latency(inst.op)[0],
                latency_upper=result._arch.latency(inst.op)[1],
                latency_issue=result._arch.issue(inst.op),
                has_profile_data=len(prof_metrics) > 0,
                vendor=result._vendor,
                source_file=source_file,
                source_line=source_line,
                source_column=source_column,
            )

            result._map[pc] = prop
            matched_count += 1

        if debug:
            logger.debug(f"[VMA Debug] Step 6: Join completed")
            logger.debug(f"  Matched: {matched_count}")
            logger.debug(f"  Skipped (offset mismatch): {skipped_offset_mismatch}")
            logger.debug(f"  Skipped (module mismatch): {skipped_module_mismatch}")
            logger.debug(f"  Skipped (function filter): {skipped_function_filter}")
            logger.debug(f"  Total stall cycles ({METRIC_GCYCLES_STL}): {total_stall_cycles:.0f}")
            if sample_skipped:
                logger.debug(f"  Sample skipped offsets:")
                for cct_id, cct_off, pc in sample_skipped:
                    logger.debug(f"    CCT {cct_id}: 0x{cct_off:x} -> PC 0x{pc:x}")
            if matched_count == 0 and skipped_offset_mismatch > 0:
                logger.warning("  WARNING: All CCT nodes skipped due to offset mismatch!")
                logger.warning("  This usually means section_file_offset is incorrect.")
                logger.warning("  Expected: CCT offsets should be in range [section_offset, section_offset + section_size]")

        # Step 7: Add instructions without profile data
        for pc, inst in inst_by_pc.items():
            if pc not in result._map:
                # Instruction not sampled - add with empty metrics
                # Get source info if available
                source_file, source_line, source_column = "", 0, 0
                if pc in dwarf_locations:
                    loc = dwarf_locations[pc]
                    source_file = loc.file
                    source_line = loc.line
                    source_column = loc.column

                prop = VMAProperty(
                    vma=pc,
                    instruction=inst,
                    cct_id=-1,
                    prof_metrics={},
                    function_name=result._function_name,
                    block_id=-1,
                    latency_lower=result._arch.latency(inst.op)[0],
                    latency_upper=result._arch.latency(inst.op)[1],
                    latency_issue=result._arch.issue(inst.op),
                    has_profile_data=False,
                    vendor=result._vendor,
                    source_file=source_file,
                    source_line=source_line,
                    source_column=source_column,
                )
                result._map[pc] = prop

        if debug:
            sampled = sum(1 for p in result._map.values() if p.has_profile_data)
            logger.debug(f"[VMA Debug] Step 7: Final map built")
            logger.debug(f"  Total entries: {len(result._map)}")
            logger.debug(f"  With profile data: {sampled}")
            logger.debug(f"  Function name: {result._function_name}")

        return result

    @classmethod
    def _build_from_binary_only(
        cls,
        instructions: List[InstructionStat],
        arch: GPUArchitecture,
        vendor: Vendor = "nvidia",
    ) -> "VMAPropertyMap":
        """Build map from binary instructions only (no profile data).

        Used when HPCToolkit database doesn't have instruction-level nodes.
        """
        result = cls()
        result._arch = arch
        result._vendor = vendor

        for inst in instructions:
            prop = VMAProperty(
                vma=inst.pc,
                instruction=inst,
                cct_id=-1,
                prof_metrics={},
                function_name="",
                block_id=-1,
                latency_lower=arch.latency(inst.op)[0],
                latency_upper=arch.latency(inst.op)[1],
                latency_issue=arch.issue(inst.op),
                has_profile_data=False,
                vendor=vendor,
            )
            result._map[inst.pc] = prop

        return result

    @classmethod
    def build_from_instructions(
        cls,
        instructions: List[InstructionStat],
        arch_name: str = "a100",
    ) -> "VMAPropertyMap":
        """Build map from instructions only without database.

        This is useful for testing or when profile data is unavailable.

        Args:
            instructions: List of InstructionStat from nvdisasm parsing.
            arch_name: Architecture name for latency lookup.

        Returns:
            VMAPropertyMap with no profile data (has_profile_data=False for all).
        """
        arch = get_architecture(arch_name)
        vendor = detect_vendor_from_arch_name(arch_name)
        return cls._build_from_binary_only(instructions, arch, vendor)


def _get_function_name(cct: pd.DataFrame, cct_id: int) -> str:
    """Traverse CCT upward to find containing function name.

    Args:
        cct: CCT DataFrame from hpcanalysis.
        cct_id: Starting node ID.

    Returns:
        Function name or "unknown" if not found.
    """
    current_id = cct_id
    max_depth = 100  # Prevent infinite loops

    for _ in range(max_depth):
        if current_id not in cct.index:
            break

        node = cct.loc[current_id]
        if node["type"] == "function":
            return str(node["name"])

        parent_id = node["parent"]
        if pd.isna(parent_id):
            break
        current_id = int(parent_id)

    return "unknown"


def _get_metrics_for_cct_node(
    profile_slices: pd.DataFrame,
    cct_id: int,
    metric_id_to_name: Dict[int, str],
) -> Dict[str, float]:
    """Extract metrics for a specific CCT node from profile slices.

    Args:
        profile_slices: Profile data with MultiIndex (profile_id, cct_id, metric_id).
        cct_id: CCT node ID to query.
        metric_id_to_name: Mapping from metric ID to name.

    Returns:
        Dictionary mapping metric name to value.
    """
    if profile_slices.empty:
        return {}

    prof_metrics: Dict[str, float] = {}

    try:
        # Profile slices has MultiIndex: (profile_id, cct_id, metric_id)
        # Query all metrics for this instruction in summary profile (profile_id=0)
        # Using xs() for cross-section lookup
        inst_metrics = profile_slices.xs((0, cct_id), level=(0, 1))

        for metric_id, row in inst_metrics.iterrows():
            metric_name = metric_id_to_name.get(metric_id, f"metric_{metric_id}")
            # Handle both DataFrame and Series return types
            if isinstance(row, pd.Series):
                value = row["value"] if "value" in row.index else row.iloc[0]
            else:
                value = row
            prof_metrics[metric_name] = float(value)
    except KeyError:
        # No profile data for this instruction (not sampled)
        pass
    except Exception:
        # Handle any other errors gracefully
        pass

    return prof_metrics


def build_vma_property_map(
    db_path: str,
    instructions: List[InstructionStat],
    arch_name: str = "a100",
    section_file_offset: int = 0,
) -> VMAPropertyMap:
    """Convenience function to build VMA property map.

    This is a shortcut for VMAPropertyMap.build().

    Args:
        db_path: Path to HPCToolkit database directory.
        instructions: List of InstructionStat from nvdisasm parsing.
        arch_name: Architecture name for latency lookup.
        section_file_offset: File offset of the .text section in CUBIN.
            See VMAPropertyMap.build() for details.

    Returns:
        Populated VMAPropertyMap instance.
    """
    return VMAPropertyMap.build(
        db_path, instructions, arch_name, section_file_offset=section_file_offset
    )
