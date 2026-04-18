"""GPU hardware utilization analysis from temporal PC sampling data.

Computes per-kernel and whole-program GPU utilization metrics by analyzing
hardware ID distributions (chiplet, CU, SIMD) from temporal PC samples.

Utilization is measured as the fraction of hardware execution units
(chiplet × CU × SIMD combinations) that were observed active during
kernel execution.

Per-kernel attribution uses module-based matching: each temporal sample's
CCT node has a module_path (GPU binary), which identifies the kernel it
belongs to. This is more robust than CCT ancestor walking because HPCToolkit
often attributes temporal samples to CPU-side calling contexts rather than
GPU kernel CCT subtrees.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from leo.temporal.reader import (
    TemporalSample,
    find_temporal_files,
    read_temporal_samples,
)


@dataclass
class HardwareUtilization:
    """GPU hardware utilization metrics for a kernel or whole program.

    Utilization is computed from the number of unique (chiplet, CU, SIMD)
    tuples observed in PC samples, relative to the total available.
    """

    total_samples: int = 0
    unique_chiplets: int = 0
    unique_cus: int = 0
    unique_simds: int = 0
    unique_execution_units: int = 0  # unique (chiplet, CU, SIMD) combos
    max_execution_units: int = 0  # total available (from arch or observed)
    unique_waves: int = 0  # unique (chiplet, CU, SIMD, wave) combos

    @property
    def utilization_pct(self) -> float:
        """Execution unit utilization as a percentage (0-100)."""
        if self.max_execution_units <= 0:
            return 0.0
        return min(self.unique_execution_units / self.max_execution_units * 100.0, 100.0)

    @property
    def has_data(self) -> bool:
        """Whether any temporal samples were available."""
        return self.total_samples > 0


@dataclass
class ProgramUtilization:
    """Whole-program GPU utilization with per-kernel breakdown."""

    program: HardwareUtilization = field(default_factory=HardwareUtilization)
    per_kernel: Dict[int, HardwareUtilization] = field(default_factory=dict)
    has_hw_ids: bool = False

    @property
    def has_data(self) -> bool:
        """Whether temporal data was available."""
        return self.program.has_data


def _build_module_sample_map(
    cct: pd.DataFrame,
    all_samples: List[TemporalSample],
    kernel_modules: Dict[int, List[int]],
) -> Dict[int, List[TemporalSample]]:
    """Group temporal samples by GPU module, matching to pipeline kernels.

    Each pipeline kernel runs in a specific GPU binary (module). Temporal
    samples whose CCT nodes belong to that same module are attributed to
    the kernel.

    Args:
        cct: CCT DataFrame with 'module_path' column.
        all_samples: All temporal PC samples.
        kernel_modules: Mapping of module_id → list of kernel CCT IDs.

    Returns:
        Dict mapping kernel CCT ID → list of temporal samples.
    """
    # Build CCT node → module_path lookup for all sample CCT IDs
    gpu_modules = set(kernel_modules.keys())
    cct_to_module: Dict[int, Optional[int]] = {}

    for s in all_samples:
        cid = s.cct_node_id
        if cid not in cct_to_module:
            if cid in cct.index:
                mod = cct.loc[cid]["module_path"]
                if not pd.isna(mod) and int(mod) in gpu_modules:
                    cct_to_module[cid] = int(mod)
                else:
                    cct_to_module[cid] = None
            else:
                cct_to_module[cid] = None

    # Group samples by module, then assign to kernels
    module_samples: Dict[int, List[TemporalSample]] = defaultdict(list)
    for s in all_samples:
        mod = cct_to_module.get(s.cct_node_id)
        if mod is not None:
            module_samples[mod].append(s)

    # Map module samples to kernel CCT IDs
    kernel_samples: Dict[int, List[TemporalSample]] = {}
    for mod_id, kernel_ids in kernel_modules.items():
        samples = module_samples.get(mod_id, [])
        if samples:
            for kid in kernel_ids:
                kernel_samples[kid] = samples

    return kernel_samples


def _compute_utilization(
    samples: List[TemporalSample],
    max_execution_units: int = 0,
) -> HardwareUtilization:
    """Compute hardware utilization from a list of temporal samples.

    Args:
        samples: Temporal PC samples (must have hw_id fields).
        max_execution_units: Total available execution units. If 0,
            auto-detected from observed maximum values.

    Returns:
        HardwareUtilization with computed metrics.
    """
    if not samples:
        return HardwareUtilization()

    chiplets: Set[int] = set()
    cus: Set[Tuple[int, int]] = set()  # (chiplet, CU)
    simds: Set[Tuple[int, int, int]] = set()  # (chiplet, CU, SIMD)
    waves: Set[Tuple[int, int, int, int]] = set()  # (chiplet, CU, SIMD, wave)

    for s in samples:
        chiplets.add(s.chiplet)
        cus.add((s.chiplet, s.cu_or_wgp_id))
        simds.add((s.chiplet, s.cu_or_wgp_id, s.simd_id))
        waves.add((s.chiplet, s.cu_or_wgp_id, s.simd_id, s.wave_id))

    if max_execution_units <= 0:
        # Auto-detect from observed ranges
        max_chiplets = max(s.chiplet for s in samples) + 1
        max_cus = max(s.cu_or_wgp_id for s in samples) + 1
        max_simds = max(s.simd_id for s in samples) + 1
        max_execution_units = max_chiplets * max_cus * max_simds

    return HardwareUtilization(
        total_samples=len(samples),
        unique_chiplets=len(chiplets),
        unique_cus=len(cus),
        unique_simds=len(simds),
        unique_execution_units=len(simds),
        max_execution_units=max_execution_units,
        unique_waves=len(waves),
    )


def compute_utilization(
    measurements_dir: str,
    cct: Optional[pd.DataFrame] = None,
    kernel_cct_ids: Optional[Set[int]] = None,
    max_execution_units: int = 0,
    debug: bool = False,
) -> ProgramUtilization:
    """Compute GPU utilization from temporal PC sampling data.

    Reads temporal-*.bin files from the measurements directory and computes
    hardware utilization metrics at both program and per-kernel granularity.

    Args:
        measurements_dir: Path to HPCToolkit measurements directory.
        cct: Optional CCT DataFrame for per-kernel mapping. If None,
            only whole-program utilization is computed.
        kernel_cct_ids: Optional set of kernel CCT IDs to report. If None
            and cct is provided, all function ancestors are reported.
        max_execution_units: Total available execution units (e.g., 912 for
            MI300A with 6 XCDs × 38 CUs × 4 SIMDs = 228 CUs, 912 SIMDs).
            If 0, auto-detected from observed maximum values.
        debug: Print debug information.

    Returns:
        ProgramUtilization with program-level and per-kernel metrics.
    """
    result = ProgramUtilization()

    # Find temporal files
    temporal_files = find_temporal_files(measurements_dir)
    if not temporal_files:
        if debug:
            print("  [Utilization] No temporal-*.bin files found")
        return result

    # Read all samples
    all_samples: List[TemporalSample] = []
    has_hw_ids = False

    for tf in temporal_files:
        samples, hw = read_temporal_samples(str(tf))
        all_samples.extend(samples)
        has_hw_ids = has_hw_ids or hw

    if not all_samples or not has_hw_ids:
        if debug:
            print(f"  [Utilization] {len(all_samples)} samples, hw_ids={has_hw_ids}")
        return result

    result.has_hw_ids = True

    if debug:
        print(f"  [Utilization] Read {len(all_samples)} temporal samples from {len(temporal_files)} files")

    # Whole-program utilization
    result.program = _compute_utilization(all_samples, max_execution_units)

    if debug:
        u = result.program
        print(f"  [Utilization] Program: {u.unique_execution_units}/{u.max_execution_units} "
              f"execution units ({u.utilization_pct:.0f}%), {u.unique_waves} waves")

    # Per-kernel utilization via module-based matching
    # Each pipeline kernel runs in a GPU binary (module). We match temporal
    # samples to kernels by their CCT node's module_path.
    if cct is not None and kernel_cct_ids:
        # Build module_id → kernel CCT IDs mapping
        kernel_modules: Dict[int, List[int]] = {}
        for kid in kernel_cct_ids:
            if kid in cct.index:
                mod = cct.loc[kid]["module_path"]
                if not pd.isna(mod):
                    kernel_modules.setdefault(int(mod), []).append(kid)

        if kernel_modules:
            kernel_samples = _build_module_sample_map(
                cct, all_samples, kernel_modules,
            )

            matched_samples = sum(len(v) for v in kernel_samples.values())
            if debug:
                print(f"  [Utilization] Module matching: {matched_samples} samples "
                      f"across {len(kernel_samples)} kernels")

            for kernel_id, samples in kernel_samples.items():
                result.per_kernel[kernel_id] = _compute_utilization(
                    samples, max_execution_units
                )

    return result
