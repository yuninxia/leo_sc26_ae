"""Whole-program GPU performance analysis across all kernels.

This module provides functions to analyze all GPU kernels in an HPCToolkit
measurement directory, ranking them by stall cycles or execution time.

Example:
    from leo import analyze_program

    result = analyze_program(
        measurements_dir="/path/to/hpctoolkit-measurements",
        arch="a100",
        top_n_kernels=10,
    )
    print(result.summary())
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from leo.analyzer import AnalysisConfig, AnalysisResult, KernelAnalyzer
from leo.analysis.occupancy import OccupancyResult, compute_occupancy
from leo.analysis.speedup import SpeedupEstimate
from leo.arch import get_architecture
from leo.db.discovery import discover_analysis_inputs, find_all_gpubins
from leo.db.reader import DatabaseReader
from leo.output.table_formatter import TableBuilder
from leo.temporal.utilization import HardwareUtilization, ProgramUtilization, compute_utilization


def _format_rate(sps: float) -> str:
    """Format samples per second as human-readable string.

    Args:
        sps: Samples per second.

    Returns:
        Formatted string (e.g., "1.2M/s", "345.6K/s").
    """
    if sps >= 1e9:
        return f"{sps/1e9:.1f}G/s"
    elif sps >= 1e6:
        return f"{sps/1e6:.1f}M/s"
    elif sps >= 1e3:
        return f"{sps/1e3:.1f}K/s"
    return f"{sps:.0f}/s"


def _demangle_name(mangled: str) -> str:
    """Demangle a C++ symbol name using c++filt.

    Delegates to the shared cached utility in leo.utils.demangle.

    Args:
        mangled: Mangled C++ name (e.g., _ZN6Kokkos...).

    Returns:
        Demangled name, or original if demangling fails.
    """
    from leo.utils.demangle import demangle
    return demangle(mangled)


def _extract_short_name(kernel_name: str) -> str:
    """Extract a short, readable name from a kernel name.

    Handles both mangled and demangled C++ names, extracting the
    most relevant part (usually the functor or kernel class name).

    Args:
        kernel_name: Full kernel name (mangled or demangled).

    Returns:
        Short, human-readable name.
    """
    # Try to demangle first
    name = _demangle_name(kernel_name)

    # For LAMMPS/Kokkos patterns, extract the functor name
    # Pattern: ...ParallelFor<LAMMPS_NS::SomeFunctor<...>>...
    # or: ...ParallelReduce<LAMMPS_NS::SomeFunctor<...>>...
    # RAJAPerf: rajaperf::apps::edge3d → EDGE3D
    rajaperf_match = re.search(r"rajaperf::(\w+)::(\w+)", name)
    if rajaperf_match:
        return rajaperf_match.group(2).upper()

    patterns = [
        r"LAMMPS_NS::(\w+)<",  # LAMMPS functor
        r"ParallelFor<(?:[\w]+::)*(\w+)<",  # Kokkos ParallelFor (deep namespaces)
        r"ParallelReduce<(?:[\w]+::)*(\w+)<",  # Kokkos ParallelReduce (deep namespaces)
        r"::(\w+Functor)\w*<",  # Any *Functor class
        r"::(\w+Kernel)\w*<",  # Any *Kernel class
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1)

    # Fallback: extract last meaningful identifier
    # Remove template args and get the last component
    name_no_templates = re.sub(r"<[^>]*>", "", name)
    parts = re.findall(r"\b([A-Z][a-zA-Z0-9_]+)\b", name_no_templates)
    if parts:
        # Return the last meaningful name (usually the actual kernel)
        return parts[-1]

    # Last resort: truncate the original
    if len(kernel_name) > 50:
        return kernel_name[:47] + "..."
    return kernel_name


def _populate_occupancy(
    kernel_analysis: "PerKernelAnalysis",
    kernels_meta: dict,
    gpu_arch,
) -> None:
    """Compute occupancy from kernel metadata and attach to analysis."""
    if not kernels_meta:
        return
    # Match by resolved kernel name
    resolved_name = kernel_analysis._resolve_kernel_name_by_offset()
    matched_kinfo = None
    if resolved_name:
        for kname, kinfo in kernels_meta.items():
            if resolved_name in kname or kname in resolved_name:
                matched_kinfo = kinfo
                break
    # Try analysis_result function names
    if not matched_kinfo and kernel_analysis.analysis_result:
        func_names = kernel_analysis.analysis_result.stats.get("function_names", [])
        for fname in func_names:
            for kname, kinfo in kernels_meta.items():
                if fname in kname or kname in fname:
                    matched_kinfo = kinfo
                    break
            if matched_kinfo:
                break
    # Fallback: use kernel with highest VGPR count
    if not matched_kinfo:
        matched_kinfo = max(kernels_meta.values(), key=lambda k: k.vgpr_count)
    if matched_kinfo:
        occ = compute_occupancy(matched_kinfo, gpu_arch)
        if occ and occ.has_data:
            kernel_analysis.occupancy = occ


@dataclass
class PerKernelAnalysis:
    """Single kernel's analysis results within whole-program context."""

    cct_id: int
    gpubin_path: str
    execution_time_s: float
    stall_cycles: float
    total_cycles: float
    stall_ratio: float
    launch_count: float
    analysis_result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    utilization: Optional[HardwareUtilization] = None
    occupancy: Optional[OccupancyResult] = None
    # GCYCLES / GKER(s) — aggregate cycle rate across all active compute units.
    # Higher = more parallelism. Unit depends on vendor (cycles, ns, or instructions).
    samples_per_second: float = 0.0
    # Vendor-normalized GPU utilization (0-100%). Measures the fraction of compute
    # units actively participating in the kernel. See Step 4c for methodology.
    gpu_utilization_pct: float = 0.0
    # Kernel offset (vaddr) from HPCToolkit CCT, used to resolve the correct
    # function name on AMD where all functions share a single .text section.
    kernel_offset: Optional[int] = None
    # Function address ranges from the binary parser: list of (name, start_vaddr, end_vaddr).
    # Populated after analysis to enable offset-based function name resolution.
    _function_ranges: List[Tuple[str, int, int]] = field(default_factory=list)

    @property
    def analyzed(self) -> bool:
        """Whether full Leo analysis was performed."""
        return self.analysis_result is not None

    @property
    def top_blame_sources(self) -> List[Tuple[int, float, str]]:
        """Get top blamed instructions (if analyzed)."""
        if self.analysis_result:
            return self.analysis_result.get_top_blame_sources(10)
        return []

    def _resolve_kernel_name_by_offset(self) -> Optional[str]:
        """Resolve the correct kernel function name using the CCT offset.

        On AMD GPUs, all device code is compiled into a single .text section,
        so function_names[0] is typically an OCKL runtime library function
        rather than the actual application kernel. This method uses the
        kernel's offset (vaddr from HPCToolkit CCT) to find the matching
        function by address range.

        Returns:
            The matching function name, or None if no match found.
        """
        if self.kernel_offset is None or not self._function_ranges:
            return None

        offset = self.kernel_offset

        # First try exact match on function start address
        for name, start, end in self._function_ranges:
            if offset == start:
                return name

        # Then try range match (offset falls within function's address range)
        for name, start, end in self._function_ranges:
            if start <= offset < end:
                return name

        return None

    @property
    def kernel_name(self) -> str:
        """Get kernel name from analysis result or gpubin path.

        On AMD, uses the CCT offset to resolve the correct function name
        from the binary's symbol table, since all functions share one .text
        section and the first function is typically an OCKL library function.
        """
        # Try offset-based resolution first (works even without analysis_result,
        # using cached function ranges from a sibling kernel's analysis)
        resolved = self._resolve_kernel_name_by_offset()
        if resolved:
            return resolved
        if self.analysis_result and self.analysis_result.stats.get("function_names"):
            return self.analysis_result.stats["function_names"][0]
        return Path(self.gpubin_path).stem

    @property
    def short_name(self) -> str:
        """Get a short, human-readable kernel name."""
        return _extract_short_name(self.kernel_name)

    def get_speedup_estimates(self, top_n: int = 5) -> List[SpeedupEstimate]:
        """Get speedup estimates (if analyzed)."""
        if self.analysis_result:
            return self.analysis_result.get_speedup_estimates(top_n)
        return []


@dataclass
class ProgramAnalysisResult:
    """Whole-program GPU performance analysis results.

    Contains per-kernel analysis results ranked by stall cycles or execution time,
    along with program-level aggregates and cross-kernel summaries.
    """

    database_path: str
    measurements_dir: str
    program_totals: dict
    per_kernel_results: List[PerKernelAnalysis]
    kernel_sort_metric: str = "stall_cycles"
    kernels_analyzed: int = 0
    kernels_skipped: int = 0
    errors: Dict[int, str] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    program_utilization: Optional[ProgramUtilization] = None
    max_sample_rate: float = 0.0  # Theoretical max: num_CUs × frequency_Hz

    @property
    def total_execution_time_s(self) -> float:
        """Total program execution time in seconds."""
        return self.program_totals.get("total_execution_time_s", 0.0)

    @property
    def total_stall_cycles(self) -> float:
        """Total stall cycles across all kernels."""
        return self.program_totals.get("total_stall_cycles", 0.0)

    @property
    def total_cycles(self) -> float:
        """Total GPU cycles."""
        return self.program_totals.get("total_cycles", 0.0)

    @property
    def stall_ratio(self) -> float:
        """Overall stall ratio (0-1)."""
        return self.program_totals.get("stall_ratio", 0.0)

    def get_top_kernels(self, n: int = 5) -> List[PerKernelAnalysis]:
        """Get top N kernels (already sorted by sort metric)."""
        return self.per_kernel_results[:n]

    def get_analyzed_kernels(self) -> List[PerKernelAnalysis]:
        """Get only kernels that were successfully analyzed."""
        return [k for k in self.per_kernel_results if k.analyzed]

    def get_top_blame_sources_overall(
        self, n: int = 10
    ) -> List[Tuple[str, int, float, str]]:
        """Get top blamed instructions across all analyzed kernels.

        Returns:
            List of (kernel_name, pc, blame_cycles, opcode) tuples.
        """
        all_blamed = []
        for kernel in self.get_analyzed_kernels():
            for pc, blame, opcode in kernel.top_blame_sources:
                all_blamed.append((kernel.kernel_name, pc, blame, opcode))

        # Sort by blame descending
        all_blamed.sort(key=lambda x: -x[2])
        return all_blamed[:n]

    def summary(self, top_n: int = 10, top_kernels: int = 5, show_detailed: bool = True) -> str:
        """Format as human-readable program summary.

        Args:
            top_n: Number of top blamed instructions to show per kernel.
            top_kernels: Number of top kernels to show in detail.
            show_detailed: If True, show detailed per-kernel analysis tables.

        Returns:
            Formatted summary string.
        """
        LINE_WIDTH = 100

        # Build main table
        table = TableBuilder(title="Leo Whole-Program GPU Performance Analysis", width=LINE_WIDTH)
        table.add_header()
        table.add_blank_line()

        table.add_line(f"Database: {self.database_path}")
        table.add_line(f"Measurements: {self.measurements_dir}")
        table.add_blank_line()

        table.add_section_header("PROGRAM TOTALS")
        table.add_separator("-")
        table.add_key_value("Total Execution Time:", f"{self.total_execution_time_s:.4f}s", indent=2, key_width=20)
        table.add_key_value("Total Stall Cycles:", f"{self.total_stall_cycles:,.0f}", indent=2, key_width=20)
        table.add_key_value("Total GPU Cycles:", f"{self.total_cycles:,.0f}", indent=2, key_width=20)
        table.add_key_value("Overall Stall Ratio:", f"{self.stall_ratio:.1%}", indent=2, key_width=20)
        table.add_key_value("Kernels Analyzed:", str(self.kernels_analyzed), indent=2, key_width=20)
        table.add_key_value("Kernels Skipped:", str(self.kernels_skipped), indent=2, key_width=20)
        if self.program_utilization and self.program_utilization.has_data:
            pu = self.program_utilization.program
            table.add_key_value(
                "GPU Utilization:",
                f"{pu.utilization_pct:.0f}% ({pu.unique_execution_units}/{pu.max_execution_units} units, "
                f"{pu.unique_waves} waves, {pu.total_samples:,} samples)",
                indent=2,
                key_width=20,
            )
        table.add_blank_line()

        # Per-kernel summary table
        displayed_kernels = min(top_kernels, len(self.per_kernel_results))
        table.add_section_header(f"TOP {displayed_kernels} KERNELS BY {self.kernel_sort_metric.upper().replace('_', ' ')}")
        table.add_separator("-")
        has_occ = any(k.occupancy is not None for k in self.per_kernel_results[:top_kernels])
        has_sps = any(k.samples_per_second > 0 for k in self.per_kernel_results[:top_kernels])
        columns = [
            ("#", 3, "<"),
            ("Kernel", 40, "<"),
            ("Time (s)", 12, ">"),
            ("Stall Cycles", 18, ">"),
            ("Stall %", 10, ">"),
        ]
        if has_occ:
            columns.append(("Occupancy", 18, ">"))
        if has_sps:
            columns.append(("Sample Rate", 18, ">"))
        table.add_columns(columns)
        table.add_separator("-")

        for i, kernel in enumerate(self.per_kernel_results[:top_kernels], 1):
            # Use short_name for readability in the table
            name = kernel.short_name
            if len(name) > 40:
                name = name[:37] + "..."
            row_values = [
                str(i),
                name,
                f"{kernel.execution_time_s:.4f}",
                f"{kernel.stall_cycles:,.0f}",
                f"{kernel.stall_ratio:.1%}",
            ]
            if has_occ:
                if kernel.occupancy and kernel.occupancy.has_data:
                    occ = kernel.occupancy
                    row_values.append(f"{occ.occupancy_pct:>3.0f}% ({occ.limiting_factor})")
                else:
                    row_values.append("-")
            if has_sps:
                if kernel.samples_per_second > 0:
                    if kernel.gpu_utilization_pct > 0:
                        row_values.append(f"{_format_rate(kernel.samples_per_second)} / {kernel.gpu_utilization_pct:.0f}%")
                    else:
                        row_values.append(_format_rate(kernel.samples_per_second))
                else:
                    row_values.append("-")
            table.add_row(*row_values)

        table.add_separator("-")

        # Detailed per-kernel analysis (if available)
        if show_detailed:
            analyzed_kernels = [k for k in self.per_kernel_results[:top_kernels] if k.analyzed]
            for i, kernel in enumerate(analyzed_kernels, 1):
                # Show short name as header, full name and gpubin below
                gpubin_name = Path(kernel.gpubin_path).name
                table.add_blank_line()
                table.add_section_header(f"KERNEL #{i}: {kernel.short_name}")
                table.add_line(f"  GPU Binary: {gpubin_name}")
                if kernel.occupancy and kernel.occupancy.has_data:
                    occ = kernel.occupancy
                    table.add_line(
                        f"  Occupancy: {occ.occupancy_pct:.0f}% "
                        f"({occ.max_waves_per_cu}/{occ.arch_wave_limit} waves/CU, "
                        f"limited by {occ.limiting_factor}) "
                        f"[VGPRs={occ.vgpr_count}, SGPRs={occ.sgpr_count}, LDS={occ.lds_bytes}B]"
                    )
                # Include the detailed analysis from AnalysisResult
                if kernel.analysis_result:
                    table.add_line(kernel.analysis_result.summary(top_n=top_n))

        table.add_footer()
        return table.build()

    def to_json(self) -> dict:
        """Export as JSON-serializable dictionary."""
        return {
            "database_path": self.database_path,
            "measurements_dir": self.measurements_dir,
            "program_totals": self.program_totals,
            "kernel_sort_metric": self.kernel_sort_metric,
            "kernels_analyzed": self.kernels_analyzed,
            "kernels_skipped": self.kernels_skipped,
            "per_kernel_results": [
                {
                    "cct_id": k.cct_id,
                    "kernel_name": k.kernel_name,
                    "gpubin_path": k.gpubin_path,
                    "execution_time_s": k.execution_time_s,
                    "stall_cycles": k.stall_cycles,
                    "total_cycles": k.total_cycles,
                    "stall_ratio": k.stall_ratio,
                    "launch_count": k.launch_count,
                    "samples_per_second": k.samples_per_second,
                    "gpu_utilization_pct": k.gpu_utilization_pct,
                    "analyzed": k.analyzed,
                    "error": k.error,
                    "top_blame_sources": [
                        {"pc": pc, "blame": blame, "opcode": opcode}
                        for pc, blame, opcode in k.top_blame_sources
                    ] if k.analyzed else [],
                }
                for k in self.per_kernel_results
            ],
            "top_blame_sources_overall": [
                {"kernel": name, "pc": pc, "blame": blame, "opcode": opcode}
                for name, pc, blame, opcode in self.get_top_blame_sources_overall(20)
            ],
            "errors": self.errors,
            "stats": self.stats,
        }


def _build_module_id_to_gpubin_map(
    db_reader: "DatabaseReader",
    gpubins: List[Tuple[Path, Optional[Path]]],
    debug: bool = False,
) -> Dict[int, Tuple[Path, Optional[Path]]]:
    """Build mapping from module_id to gpubin path.

    Uses load_modules table from database to match kernel module IDs
    to their corresponding gpubin files.

    Args:
        db_reader: DatabaseReader instance with access to load_modules table.
        gpubins: List of (gpubin_path, hpcstruct_path) tuples.
        debug: Print matching decisions.

    Returns:
        Dict mapping module_id -> (gpubin_path, hpcstruct_path or None).
    """
    # Get load_modules table which has module_id -> relative_path mapping
    load_modules = db_reader._query._load_modules

    # Build filename -> gpubin path mapping for quick lookup
    gpubin_by_filename: Dict[str, Tuple[Path, Optional[Path]]] = {}
    for gpubin_path, hpcstruct_path in gpubins:
        gpubin_by_filename[gpubin_path.name] = (gpubin_path, hpcstruct_path)

    # Build module_id -> gpubin mapping
    module_to_gpubin: Dict[int, Tuple[Path, Optional[Path]]] = {}

    for module_id, row in load_modules.iterrows():
        module_path = str(row["module_path"])

        # Extract filename from path (e.g., "gpubins/hash.gpubin" -> "hash.gpubin")
        filename = Path(module_path).name

        if filename in gpubin_by_filename:
            module_to_gpubin[int(module_id)] = gpubin_by_filename[filename]
            if debug:
                print(f"  Module {module_id}: {filename}")

    if debug:
        print(f"  Mapped {len(module_to_gpubin)} modules to gpubins")

    return module_to_gpubin


def analyze_program(
    measurements_dir: str,
    arch: str = "a100",
    vendor: Optional[str] = None,
    top_n_kernels: Optional[int] = None,
    sort_by: str = "stall_cycles",
    run_full_analysis: bool = True,
    skip_failed_kernels: bool = True,
    debug: bool = False,
    **kwargs,
) -> ProgramAnalysisResult:
    """Analyze all GPU kernels in a whole-program measurement.

    Discovers all GPU binaries in the measurements directory, ranks kernels
    by stall cycles or execution time, and optionally runs full Leo analysis
    on each kernel.

    Args:
        measurements_dir: Path to HPCToolkit measurements directory.
        arch: GPU architecture (e.g., "a100", "mi250", "gfx90a").
        vendor: Optional vendor override ("nvidia", "amd", "intel").
        top_n_kernels: Analyze only top N kernels by metric (None = all).
        sort_by: Metric to rank kernels ("stall_cycles" or "execution_time").
        run_full_analysis: If True, run full Leo back-slicing analysis per kernel.
            If False, only collect metrics from database (faster).
        skip_failed_kernels: If True, continue if individual kernels fail analysis.
        debug: Enable debug output.
        **kwargs: Additional AnalysisConfig parameters.

    Returns:
        ProgramAnalysisResult with per-kernel analyses and aggregates.

    Raises:
        FileNotFoundError: If measurements directory, database, or gpubins not found.
        ValueError: If no kernels found or all analyses fail.

    Example:
        result = analyze_program(
            measurements_dir="/path/to/hpctoolkit-measurements",
            arch="a100",
            top_n_kernels=10,
        )
        print(result.summary())

        # Access individual kernel results
        for kernel in result.get_top_kernels(5):
            print(f"{kernel.kernel_name}: {kernel.stall_cycles:,.0f} stall cycles")
    """
    measurements_path = Path(measurements_dir)
    if not measurements_path.exists():
        raise FileNotFoundError(f"Measurements directory not found: {measurements_dir}")

    # Step 1: Discover inputs
    if debug:
        print(f"[Program Analysis] Discovering inputs from {measurements_dir}")

    inputs = discover_analysis_inputs(measurements_path)
    database_path = inputs["database"]
    gpubins = inputs["gpubins"]

    if debug:
        print(f"  Database: {database_path}")
        print(f"  GPU binaries: {len(gpubins)}")

    # Step 2: Read kernel metrics from database
    db_reader = DatabaseReader(str(database_path))
    program_totals = db_reader.get_program_totals()
    kernel_df = db_reader.get_per_kernel_summary(sort_by=sort_by, top_n=top_n_kernels)

    if len(kernel_df) == 0:
        raise ValueError("No kernels with metrics found in database")

    if debug:
        print(f"  Found {len(kernel_df)} kernels with metrics")
        print(f"  Program totals: {program_totals['total_stall_cycles']:,.0f} stall cycles")

    # Step 2b: Build module_id -> gpubin mapping
    module_to_gpubin = _build_module_id_to_gpubin_map(db_reader, gpubins, debug=debug)

    # Step 2c: Compute hardware utilization from temporal PC sampling (if available)
    program_util: Optional[ProgramUtilization] = None
    try:
        cct = db_reader.get_cct()
        kernel_cct_ids = set(kernel_df.index)
        program_util = compute_utilization(
            measurements_dir,
            cct=cct,
            kernel_cct_ids=kernel_cct_ids,
            debug=debug,
        )
        if program_util.has_data and debug:
            print(f"  Hardware utilization: {program_util.program.utilization_pct:.0f}%")
    except Exception as e:
        if debug:
            print(f"  [Utilization] Skipped: {e}")

    # Step 3: Analyze each kernel
    gpu_arch = get_architecture(arch)
    per_kernel_results: List[PerKernelAnalysis] = []
    errors: Dict[int, str] = {}
    kernels_analyzed = 0
    kernels_skipped = 0

    # Cache binary parser data by gpubin path to avoid re-parsing for
    # kernels that share the same binary but skip full analysis.
    # Stores: (function_ranges, kernels_meta) per gpubin path.
    _gpubin_cache: Dict[str, Tuple[List[Tuple[str, int, int]], dict]] = {}

    for cct_id, row in kernel_df.iterrows():
        if debug:
            print(f"\n[Kernel {cct_id}] stall_cycles={row['stall_cycles']:,.0f}, time={row['execution_time_s']:.4f}s")

        # Find matching gpubin using module_id
        module_id = int(row["module_id"])
        if module_id in module_to_gpubin:
            gpubin_path, hpcstruct_path = module_to_gpubin[module_id]
            if debug:
                print(f"  Matched module {module_id} -> {gpubin_path.name}")
        elif len(gpubins) == 1:
            # Fallback for single gpubin case
            gpubin_path, hpcstruct_path = gpubins[0]
            if debug:
                print(f"  Module {module_id} not mapped, using single gpubin: {gpubin_path.name}")
        else:
            error_msg = f"No gpubin found for module_id {module_id}"
            if skip_failed_kernels:
                errors[cct_id] = error_msg
                kernels_skipped += 1
                if debug:
                    print(f"  ERROR: {error_msg}")
                continue
            raise FileNotFoundError(error_msg)

        # Create per-kernel analysis object
        # Store the CCT offset for kernel name resolution (critical for AMD,
        # where all functions share a single .text section)
        kernel_offset = row.get("offset")
        if kernel_offset is not None:
            kernel_offset = int(kernel_offset)

        kernel_analysis = PerKernelAnalysis(
            cct_id=int(cct_id),
            gpubin_path=str(gpubin_path),
            execution_time_s=float(row["execution_time_s"]),
            stall_cycles=float(row["stall_cycles"]),
            total_cycles=float(row["total_cycles"]),
            stall_ratio=float(row["stall_ratio"]),
            launch_count=float(row["launch_count"]),
            kernel_offset=kernel_offset,
        )

        # Attach per-kernel utilization if available
        if program_util and program_util.has_data:
            kernel_analysis.utilization = program_util.per_kernel.get(int(cct_id))

        # Run full Leo analysis if requested
        # Skip full analysis for kernels with no stall data —
        # avoids showing misleading results when multiple kernels
        # share the same .gpubin (common on AMD)
        if run_full_analysis and not (kernel_analysis.stall_cycles == 0 and kernel_analysis.total_cycles == 0):
            try:
                config = AnalysisConfig(
                    db_path=str(database_path),
                    gpubin_path=str(gpubin_path),
                    arch=arch,
                    vendor=vendor,
                    hpcstruct_path=str(hpcstruct_path) if hpcstruct_path else None,
                    debug=debug,
                    module_id=module_id,  # Filter CCT nodes to this module only
                    **kwargs,
                )
                analyzer = KernelAnalyzer(config)
                kernel_analysis.analysis_result = analyzer.analyze()
                kernels_analyzed += 1

                # Populate function address ranges for kernel name resolution.
                # On AMD, all functions share a single .text section, so we need
                # offset-based lookup to find the correct function name.
                func_ranges: List[Tuple[str, int, int]] = []
                kernels_meta: dict = {}
                if analyzer._binary_parser and analyzer._binary_parser.functions:
                    func_ranges = [
                        (func.name, func.offset, func.offset + func.size)
                        for func in analyzer._binary_parser.functions.values()
                        if func.size > 0
                    ]
                    kernel_analysis._function_ranges = func_ranges
                if analyzer._binary_parser and hasattr(analyzer._binary_parser, 'kernels'):
                    kernels_meta = analyzer._binary_parser.kernels

                # Cache for kernels that share this gpubin but skip analysis
                _gpubin_cache[str(gpubin_path)] = (func_ranges, kernels_meta)

                # Propagate resolved kernel name to the AnalysisResult so that
                # summary() and to_json() show the correct name instead of
                # function_names[0] (which is wrong on AMD).
                resolved = kernel_analysis._resolve_kernel_name_by_offset()
                if resolved:
                    kernel_analysis.analysis_result.resolved_kernel_name = resolved

                # Compute occupancy from kernel metadata (AMD only)
                _populate_occupancy(kernel_analysis, kernels_meta, gpu_arch)

                if debug:
                    top_blame = kernel_analysis.analysis_result.get_top_blame_sources(3)
                    print(f"  Analyzed: {len(top_blame)} top blame sources")
                    if kernel_analysis.occupancy:
                        occ = kernel_analysis.occupancy
                        print(f"  Occupancy: {occ.occupancy_pct:.0f}% ({occ.max_waves_per_cu}/{occ.arch_wave_limit} waves, limited by {occ.limiting_factor})")

            except Exception as e:
                error_msg = f"Analysis failed: {e}"
                kernel_analysis.error = error_msg
                errors[cct_id] = error_msg
                kernels_skipped += 1

                if debug:
                    print(f"  ERROR: {error_msg}")

                if not skip_failed_kernels:
                    raise
        else:
            kernels_analyzed += 1
            # For kernels that skip full analysis (no stall data), still
            # resolve kernel name and occupancy from cached binary parser data.
            cached = _gpubin_cache.get(str(gpubin_path))
            if not cached:
                # No sibling kernel was analyzed for this gpubin yet.
                # Parse just the binary (cheap) for function names and metadata.
                try:
                    from leo.binary.parser.base import get_parser
                    bp = get_parser(vendor or gpu_arch.vendor, str(gpubin_path))
                    func_ranges = [
                        (func.name, func.offset, func.offset + func.size)
                        for func in bp.functions.values()
                        if func.size > 0
                    ] if bp.functions else []
                    kernels_meta = bp.kernels if hasattr(bp, 'kernels') else {}
                    _gpubin_cache[str(gpubin_path)] = (func_ranges, kernels_meta)
                    cached = (func_ranges, kernels_meta)
                except Exception:
                    pass  # Binary parsing failed; names will show gpubin hash
            if cached:
                func_ranges, kernels_meta = cached
                if func_ranges:
                    kernel_analysis._function_ranges = func_ranges
                _populate_occupancy(kernel_analysis, kernels_meta, gpu_arch)

        per_kernel_results.append(kernel_analysis)

    if kernels_analyzed == 0 and run_full_analysis:
        raise ValueError("All kernel analyses failed")

    # Step 4b: Compute GPU parallelism metric (GCYCLES / GKER ratio)
    # A higher sample rate means more CUs were actively executing during the
    # kernel's lifetime. Low ratio signals lack of parallelism — the kernel
    # is not utilizing all available compute units.
    for ka in per_kernel_results:
        if ka.execution_time_s > 0:
            ka.samples_per_second = ka.total_cycles / ka.execution_time_s

    # Step 4c: Compute GPU utilization percentage (vendor-specific normalization)
    #
    # HPCToolkit's GCYCLES = (issues + non_issues) × sampling_period, but the
    # unit of sampling_period differs by vendor (see hpcrun/gpu/gpu-metrics.c and
    # each vendor's sample source):
    #
    #   AMD (rocm-ss.c):    period = 2^20 GPU cycles   → GCYCLES is in cycles
    #   Intel (level0.c):   period = 2^20 nanoseconds  → GCYCLES is in nanoseconds
    #   NVIDIA (nvidia.c):  period = 2^20 instructions → GCYCLES is in instruction units
    #
    # Because idle compute units generate no samples, GCYCLES only accumulates
    # from units that have threads dispatched. This gives us:
    #
    #   utilization = GCYCLES / GKER / theoretical_max
    #
    # where theoretical_max = (all units active) × (unit contribution per second).
    #
    # AMD:  Each CU contributes freq_Hz cycles/sec when active.
    #       max = CUs × freq_Hz.  util = rate / max = active CUs / total CUs.
    #       Verified: MI300A LAMMPS kernel shows 23% (≈70/304 CUs), consistent
    #       with 10% occupancy and partial CU dispatch.
    #
    # Intel: Each XVE (vector engine) contributes 1e9 ns/sec of attributed time
    #        when active. max = total_XVEs × 1e9.  The result measures the fraction
    #        of XVEs with threads loaded (non-idle), equivalent to Intel VTune's
    #        (EU Active + EU Stalled) / (EU Active + EU Stalled + EU Idle).
    #        PVC has 1024 XVEs (128 Xe-cores × 8), so the same workload that
    #        fills 94% of MI100's 120 CUs may only fill 49% of PVC's 1024 XVEs.
    #        Ref: Intel oneAPI Optimization Guide, "Hardware-assisted Stall Sampling".
    #
    # NVIDIA: The instruction-based period makes direct normalization unreliable
    #         (multi-warp round-robin inflates GCYCLES beyond SM×freq). Instead,
    #         we use HPCToolkit's pre-computed SM efficiency: gsamp:tot / gsamp:exp,
    #         where gsamp:exp = (clock_rate × kernel_time / period_in_cycles) × num_SMs
    #         (see hpcrun/gpu/api/nvidia/cupti-analysis.c:cupti_sm_efficiency_analyze).
    #
    if gpu_arch.vendor == "amd":
        max_rate = gpu_arch.sms * gpu_arch.frequency * 1e9
        for ka in per_kernel_results:
            if ka.samples_per_second > 0 and max_rate > 0:
                pct = ka.samples_per_second / max_rate * 100
                if pct <= 100:
                    ka.gpu_utilization_pct = pct
    elif gpu_arch.vendor == "intel":
        total_eus = gpu_arch.sms * gpu_arch.schedulers  # Xe-cores × vector-engines
        max_rate = total_eus * 1e9
        for ka in per_kernel_results:
            if ka.samples_per_second > 0 and max_rate > 0:
                pct = ka.samples_per_second / max_rate * 100
                if pct <= 100:
                    ka.gpu_utilization_pct = pct
    elif gpu_arch.vendor == "nvidia":
        # NVIDIA: read gsamp:tot / gsamp:exp per kernel from database
        # These metrics are at the host-side kernel launch CCT node (parent of
        # the GPU function node), so we check both the kernel's CCT ID and its parent.
        from leo.constants.metrics import METRIC_GSAMP_EXP, METRIC_GSAMP_TOT
        try:
            all_metrics = db_reader.get_metrics("*")
            gsamp_tot_m = all_metrics[
                (all_metrics["name"] == METRIC_GSAMP_TOT)
                & (all_metrics["scope"] == "e")
                & (all_metrics["aggregation"] == "sum")
            ]
            gsamp_exp_m = all_metrics[
                (all_metrics["name"] == METRIC_GSAMP_EXP)
                & (all_metrics["scope"] == "e")
                & (all_metrics["aggregation"] == "sum")
            ]
            if len(gsamp_tot_m) > 0 and len(gsamp_exp_m) > 0:
                tot_id = int(gsamp_tot_m.iloc[0]["id"])
                exp_id = int(gsamp_exp_m.iloc[0]["id"])
                slices = db_reader.get_profile_slices("*", "summary", "*").reset_index()
                cct_to_tot = dict(
                    slices[slices["metric_id"] == tot_id][["cct_id", "value"]].values
                )
                cct_to_exp = dict(
                    slices[slices["metric_id"] == exp_id][["cct_id", "value"]].values
                )
                # Build child -> parent map from CCT
                cct = db_reader.get_cct()
                child_to_parent = {}
                for nid, node in cct.iterrows():
                    children = node.get("children")
                    if isinstance(children, list):
                        for child in children:
                            child_to_parent[child] = nid

                for ka in per_kernel_results:
                    # Check kernel node first, then parent (host-side launch node)
                    for cid in [ka.cct_id, child_to_parent.get(ka.cct_id)]:
                        if cid is None:
                            continue
                        tot = cct_to_tot.get(cid, 0)
                        exp = cct_to_exp.get(cid, 0)
                        if exp > 0:
                            ka.gpu_utilization_pct = tot / exp * 100
                            break
        except Exception:
            pass  # gsamp metrics not available

    # Step 4d: Build result
    stats = {
        "total_kernels_in_database": len(kernel_df),
        "kernels_processed": len(per_kernel_results),
        "kernels_analyzed": kernels_analyzed,
        "kernels_skipped": kernels_skipped,
    }

    max_sample_rate = 0  # No longer used for percentage (computed per-kernel above)

    return ProgramAnalysisResult(
        database_path=str(database_path),
        measurements_dir=str(measurements_dir),
        program_totals=program_totals,
        per_kernel_results=per_kernel_results,
        kernel_sort_metric=sort_by,
        kernels_analyzed=kernels_analyzed,
        kernels_skipped=kernels_skipped,
        errors=errors,
        stats=stats,
        program_utilization=program_util,
        max_sample_rate=max_sample_rate,
    )
