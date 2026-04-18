"""Database reader wrapper around hpcanalysis library."""

from typing import Optional

import hpcanalysis
import pandas as pd

from leo.constants.metrics import (
    METRIC_GCYCLES,
    METRIC_GCYCLES_ISU,
    METRIC_GCYCLES_STL,
    METRIC_GCYCLES_STL_GMEM,
    METRIC_GCYCLES_STL_LMEM,
    METRIC_GCYCLES_STL_MEM,
    METRIC_GCYCLES_STL_SYNC,
    METRIC_GKER,
    METRIC_GKER_COUNT,
)


class DatabaseReader:
    """Wrapper around hpcanalysis for reading HPCToolkit databases.

    Provides a simplified interface for accessing metrics, CCT (Calling Context Tree),
    and profile data from HPCToolkit performance databases.

    Example:
        >>> reader = DatabaseReader("/path/to/database")
        >>> metrics = reader.get_metrics("gcycles:*")
        >>> cct = reader.get_cct()
        >>> slices = reader.get_profile_slices()
    """

    def __init__(
        self,
        db_path: str,
        use_cpp_parser: bool = False,
        dataframe_type: str = "pandas",
    ):
        """Initialize the database reader.

        Args:
            db_path: Path to the HPCToolkit database directory.
            use_cpp_parser: Use C++ parser for better performance.
            dataframe_type: DataFrame backend ("pandas" or "cudf").
        """
        self._db_path = db_path
        self._api = hpcanalysis.open_db(
            db_path,
            use_cpp_parser=use_cpp_parser,
            hpc_dataframe_type=dataframe_type,
            exclude_empty_profiles=True,
        )
        self._query = self._api._query_api

    @property
    def db_path(self) -> str:
        """Return the database path."""
        return self._db_path

    def get_metrics(self, pattern: str = "*") -> pd.DataFrame:
        """Get metric descriptions matching the pattern.

        Args:
            pattern: Glob pattern for metric names (e.g., "gcycles:*", "gcycles:stl:*").

        Returns:
            DataFrame with columns: id, name, aggregation, scope, unit.
        """
        return self._query.query_metric_descriptions(pattern)

    def get_cct(self, pattern: str = "*") -> pd.DataFrame:
        """Get the Calling Context Tree.

        Args:
            pattern: Filter expression for CCT nodes (e.g., "function(kernel_name)").

        Returns:
            DataFrame with columns: type, parent, children, depth, name,
            file_path, line, module_path, offset.
            Index: id (cct_id).
        """
        return self._query.query_cct(pattern)

    def get_profile_slices(
        self,
        cct_exp: str = "*",
        profiles_exp: str = "summary",
        metrics_exp: str = "*",
    ) -> pd.DataFrame:
        """Get profile data slices.

        Args:
            cct_exp: CCT filter expression.
            profiles_exp: Profile filter ("summary" for aggregated, or specific).
            metrics_exp: Metric filter expression.

        Returns:
            DataFrame with MultiIndex (profile_id, cct_id, metric_id) and value column.
        """
        return self._query.query_profile_slices(cct_exp, profiles_exp, metrics_exp)

    def get_profile_descriptions(self, pattern: str = "*") -> pd.DataFrame:
        """Get profile descriptions (threads, ranks, etc.).

        Args:
            pattern: Filter pattern for profiles.

        Returns:
            DataFrame with profile metadata including rank, thread info.
        """
        return self._query.query_profile_descriptions(pattern)

    def get_gpu_metrics_summary(self) -> dict:
        """Get a summary of GPU metrics from the database.

        Uses propagated scope ('p') which only has values at leaf instruction
        nodes, avoiding the triple-counting that occurs with exclusive scope
        ('e') where HPCToolkit stores identical values at function, line, and
        instruction levels of the CCT hierarchy.

        Returns:
            Dictionary mapping metric names to their total values.
        """
        all_metrics = self.get_metrics("*")
        slices = self.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()

        metric_totals = slices_flat.groupby("metric_id")["value"].sum()

        # Use propagated scope ('p') to avoid triple-counting. HPCToolkit
        # stores exclusive ('e') values at every CCT level (function, line,
        # instruction), but propagated ('p') values only at leaf nodes.
        # Fall back to exclusive if propagated is not available (older databases).
        results = {}
        for _, row in all_metrics.iterrows():
            if row["aggregation"] == "sum" and row["scope"] == "p":
                val = metric_totals.get(row["id"], 0)
                if val > 0:
                    results[row["name"]] = val

        # Fallback: if no propagated metrics found, use exclusive scope
        if not results:
            for _, row in all_metrics.iterrows():
                if row["aggregation"] == "sum" and row["scope"] == "e":
                    val = metric_totals.get(row["id"], 0)
                    if val > 0:
                        results[row["name"]] = val

        return results

    def classify_workload(self) -> str:
        """Classify the workload based on GPU metrics.

        Returns:
            Classification string: "Compute-bound", "Memory-bound",
            "Sync-bound", or "Mixed".
        """
        results = self.get_gpu_metrics_summary()
        gcycles = results.get(METRIC_GCYCLES, 1)

        if gcycles == 0:
            return "Unknown"

        isu_pct = results.get(METRIC_GCYCLES_ISU, 0) / gcycles * 100
        mem_stall = sum(
            results.get(m, 0)
            for m in [METRIC_GCYCLES_STL_MEM, METRIC_GCYCLES_STL_GMEM, METRIC_GCYCLES_STL_LMEM]
        )
        mem_pct = mem_stall / gcycles * 100
        sync_pct = results.get(METRIC_GCYCLES_STL_SYNC, 0) / gcycles * 100

        if isu_pct > 70:
            return "Compute-bound"
        elif mem_pct > 40:
            return "Memory-bound"
        elif sync_pct > 5:
            return "Sync-bound"
        else:
            return "Mixed"

    def get_per_kernel_summary(
        self,
        sort_by: str = "stall_cycles",
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get per-kernel metrics summary for whole-program analysis.

        Returns a DataFrame with one row per GPU kernel, including execution time,
        stall cycles, and launch count. Kernels are identified by their CCT node ID.

        This enables GPA-style whole-program analysis where kernels can be ranked
        by their impact (execution time or stall cycles).

        Args:
            sort_by: Metric to sort by. Options:
                - "stall_cycles": Total stall cycles (default, Leo-style)
                - "execution_time": Kernel execution time in seconds (GPA-style)
                - "launch_count": Number of kernel invocations
            top_n: If specified, return only the top N kernels.

        Returns:
            DataFrame with columns:
                - cct_id: CCT node identifier (index)
                - execution_time_s: Kernel execution time in seconds (from gker)
                - stall_cycles: Total stall cycles inclusive (from gcycles:stl)
                - total_cycles: Total GPU cycles inclusive (from gcycles)
                - issue_cycles: Issue cycles inclusive (from gcycles:isu)
                - launch_count: Number of kernel invocations (from gker:count)
                - stall_ratio: stall_cycles / total_cycles (0-1)
                - module_id: Module path ID (for grouping by gpubin)
                - offset: Function offset in module

        Example:
            >>> reader = DatabaseReader("/path/to/database")
            >>> kernels = reader.get_per_kernel_summary(sort_by="stall_cycles", top_n=10)
            >>> for _, row in kernels.iterrows():
            ...     print(f"Kernel {row.name}: {row['stall_cycles']:,.0f} stall cycles")
        """
        # Get all metrics metadata
        all_metrics = self.get_metrics("*")

        # Build metric name -> ID mapping
        # Use exclusive (scope='e') for gker metrics (kernel-level, no children)
        # Use inclusive (scope='i') for gcycles metrics (aggregates children)
        def get_metric_id(name: str, scope: str = "e") -> Optional[int]:
            matches = all_metrics[
                (all_metrics["name"] == name)
                & (all_metrics["scope"] == scope)
                & (all_metrics["aggregation"] == "sum")
            ]
            return int(matches["id"].iloc[0]) if len(matches) > 0 else None

        # Kernel-level metrics (exclusive scope - no children to aggregate)
        gker_id = get_metric_id(METRIC_GKER, scope="e")
        gker_count_id = get_metric_id(METRIC_GKER_COUNT, scope="e")

        # Cycle metrics use INCLUSIVE scope to aggregate from instruction children
        gcycles_id = get_metric_id(METRIC_GCYCLES, scope="i")
        gcycles_isu_id = get_metric_id(METRIC_GCYCLES_ISU, scope="i")
        gcycles_stl_id = get_metric_id(METRIC_GCYCLES_STL, scope="i")

        # Get CCT and filter to function nodes (kernels)
        cct = self.get_cct()
        functions = cct[cct["type"] == "function"]

        if len(functions) == 0:
            return pd.DataFrame()

        # Get profile slices
        slices = self.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()

        # Helper to extract a metric value from a slice subset
        def get_slice_value(
            node_slices: pd.DataFrame, metric_id: Optional[int]
        ) -> float:
            if metric_id is None:
                return 0.0
            matches = node_slices[node_slices["metric_id"] == metric_id]
            return float(matches["value"].iloc[0]) if len(matches) > 0 else 0.0

        # --- Phase 1: Identify host-side kernel nodes (have gker > 0) ---
        # These nodes come from host-side instrumentation and carry execution
        # time (gker) and launch count (gker:count) but NO cycle data.
        host_kernel_ids: set = set()
        if gker_id is not None:
            gker_slices = slices_flat[
                (slices_flat["metric_id"] == gker_id)
                & (slices_flat["value"] > 0)
            ]
            host_kernel_ids = set(gker_slices["cct_id"]) & set(functions.index)

        # --- Phase 2: Identify device-side GPU function nodes ---
        # These nodes come from PC sampling on the GPU binary and carry cycle
        # metrics (gcycles, gcycles:stl, gcycles:isu) but NO gker data.
        # In HPCToolkit's CCT, host-side and device-side nodes are in disjoint
        # subtrees. Host-side kernel launches go through the application call
        # chain, while device-side PC samples go through the GPU runtime
        # (HSA/ROCr/CUDA driver) call chain. They share the same GPU binary
        # module_path and kernel function offset, which we use to join them.
        device_func_ids: set = set()
        if gcycles_id is not None:
            gcycles_slices = slices_flat[
                (slices_flat["metric_id"] == gcycles_id)
                & (slices_flat["value"] > 0)
            ]
            device_func_ids = set(gcycles_slices["cct_id"]) & set(functions.index)

        # --- Phase 3: Build device-side cycle data keyed by (module_path, offset) ---
        # Find the kernel entry point for each device-side function by walking up
        # the CCT parent chain to find the topmost function node on the GPU binary
        # module that has a non-zero offset. This is the kernel function entry point.
        # Then collect inclusive cycle metrics at that entry point.
        #
        # For kernels where all device functions have offset=0 (e.g., synthetic
        # function nodes), we look for the topmost ancestor function with gcycles
        # data that has a non-zero offset on any module — this is the kernel
        # entry as identified by hpcstruct.

        # Build mapping: (module_path, offset) -> {gcycles, gcycles:stl, gcycles:isu}
        # Use the TOPMOST device function with a meaningful offset as the entry.
        device_cycle_data: dict = {}  # (module_path, offset) -> cycle metrics dict

        # Identify GPU binary modules: modules that host-side kernels reference
        # (these are the .gpubin / .co files). We restrict to host-side modules
        # only because device-side CCT paths may include inlined host-side
        # functions (e.g., Kokkos template specializations) on the application
        # binary module, which is NOT a GPU binary.
        gpu_module_ids: set = set()
        for hid in host_kernel_ids:
            mp = cct.loc[hid, "module_path"]
            if pd.notna(mp):
                gpu_module_ids.add(int(mp))

        # For each device function with gcycles, walk up the parent chain to
        # find the kernel entry (topmost function on a GPU module with non-zero
        # offset). Build cycle data keyed by (module, offset) of that entry.
        processed_entries: set = set()  # avoid double-counting entries
        for did in device_func_ids:
            # Walk parent chain to find the kernel entry function
            kernel_entry_id = None
            current = did
            for _ in range(200):
                node = cct.loc[current]
                tp = node["type"]
                mp = node["module_path"]
                off = node["offset"]
                if (
                    tp == "function"
                    and pd.notna(mp)
                    and int(mp) in gpu_module_ids
                    and pd.notna(off)
                    and int(off) > 0
                ):
                    kernel_entry_id = current
                    # Keep going up — we want the TOPMOST kernel entry
                parent_val = node["parent"]
                if pd.isna(parent_val):
                    break
                current = int(parent_val)

            if kernel_entry_id is None:
                continue
            if kernel_entry_id in processed_entries:
                continue
            processed_entries.add(kernel_entry_id)

            entry_node = cct.loc[kernel_entry_id]
            entry_mp = int(entry_node["module_path"])
            entry_off = int(entry_node["offset"])
            key = (entry_mp, entry_off)

            if key not in device_cycle_data:
                # Read inclusive cycle metrics at the kernel entry node
                entry_slices = slices_flat[
                    slices_flat["cct_id"] == kernel_entry_id
                ]
                device_cycle_data[key] = {
                    "total_cycles": get_slice_value(entry_slices, gcycles_id),
                    "stall_cycles": get_slice_value(
                        entry_slices, gcycles_stl_id
                    ),
                    "issue_cycles": get_slice_value(
                        entry_slices, gcycles_isu_id
                    ),
                    "device_cct_id": kernel_entry_id,
                    "device_module_path": entry_mp,
                    "device_offset": entry_off,
                }

        # --- Phase 4: Build per-kernel data by joining host + device ---
        kernel_data = []

        # Case A: We have host-side kernel nodes — join with device-side
        # by matching (module_path, offset)
        if host_kernel_ids:
            for host_id in host_kernel_ids:
                host_slices = slices_flat[slices_flat["cct_id"] == host_id]
                exec_time = get_slice_value(host_slices, gker_id)
                launch_count = get_slice_value(host_slices, gker_count_id)

                host_node = cct.loc[host_id]
                host_mp = host_node["module_path"]
                host_off = host_node["offset"]

                total_cycles = 0.0
                issue_cycles = 0.0
                stall_cycles = 0.0
                module_id = host_mp
                offset = host_off

                # Try to match with device-side cycle data by (module, offset)
                if pd.notna(host_mp) and pd.notna(host_off):
                    key = (int(host_mp), int(host_off))
                    if key in device_cycle_data:
                        cycle_info = device_cycle_data[key]
                        total_cycles = cycle_info["total_cycles"]
                        stall_cycles = cycle_info["stall_cycles"]
                        issue_cycles = cycle_info["issue_cycles"]
                        # Use device-side module/offset for gpubin matching
                        module_id = cycle_info["device_module_path"]
                        offset = cycle_info["device_offset"]

                # If no match by offset, check if the host node itself has
                # cycle data (single-node case, e.g., NVIDIA where both host
                # and device data may be on the same function node)
                if total_cycles == 0 and stall_cycles == 0:
                    total_cycles = get_slice_value(host_slices, gcycles_id)
                    issue_cycles = get_slice_value(host_slices, gcycles_isu_id)
                    stall_cycles = get_slice_value(host_slices, gcycles_stl_id)

                # Skip kernels with no meaningful data
                if exec_time == 0 and stall_cycles == 0 and total_cycles == 0:
                    continue

                stall_ratio = (
                    stall_cycles / total_cycles if total_cycles > 0 else 0.0
                )

                kernel_data.append({
                    "cct_id": host_id,
                    "execution_time_s": exec_time,
                    "stall_cycles": stall_cycles,
                    "total_cycles": total_cycles,
                    "issue_cycles": issue_cycles,
                    "launch_count": launch_count,
                    "stall_ratio": stall_ratio,
                    "module_id": module_id,
                    "offset": offset,
                })

        # Case B: No host-side kernel nodes found — fall back to device-side
        # only (handles databases that only have PC sampling data without gker)
        if not kernel_data and device_cycle_data:
            for key, cycle_info in device_cycle_data.items():
                dev_cct_id = cycle_info["device_cct_id"]
                dev_slices = slices_flat[slices_flat["cct_id"] == dev_cct_id]

                kernel_data.append({
                    "cct_id": dev_cct_id,
                    "execution_time_s": get_slice_value(dev_slices, gker_id),
                    "stall_cycles": cycle_info["stall_cycles"],
                    "total_cycles": cycle_info["total_cycles"],
                    "issue_cycles": cycle_info["issue_cycles"],
                    "launch_count": get_slice_value(dev_slices, gker_count_id),
                    "stall_ratio": (
                        cycle_info["stall_cycles"] / cycle_info["total_cycles"]
                        if cycle_info["total_cycles"] > 0
                        else 0.0
                    ),
                    "module_id": cycle_info["device_module_path"],
                    "offset": cycle_info["device_offset"],
                })

        # Case C: No gker metric exists at all — original fallback using all
        # function nodes (preserves behavior for non-GPU databases)
        if not kernel_data and gker_id is None:
            for cct_id in set(functions.index):
                node_slices = slices_flat[slices_flat["cct_id"] == cct_id]
                if len(node_slices) == 0:
                    continue

                exec_time = get_slice_value(node_slices, gker_id)
                launch_count = get_slice_value(node_slices, gker_count_id)
                total_cycles = get_slice_value(node_slices, gcycles_id)
                issue_cycles = get_slice_value(node_slices, gcycles_isu_id)
                stall_cycles = get_slice_value(node_slices, gcycles_stl_id)

                if exec_time == 0 and stall_cycles == 0 and total_cycles == 0:
                    continue

                node = cct.loc[cct_id]
                stall_ratio = (
                    stall_cycles / total_cycles if total_cycles > 0 else 0.0
                )

                kernel_data.append({
                    "cct_id": cct_id,
                    "execution_time_s": exec_time,
                    "stall_cycles": stall_cycles,
                    "total_cycles": total_cycles,
                    "issue_cycles": issue_cycles,
                    "launch_count": launch_count,
                    "stall_ratio": stall_ratio,
                    "module_id": node.get("module_path"),
                    "offset": node.get("offset"),
                })

        if not kernel_data:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(kernel_data)
        df = df.set_index("cct_id")

        # Sort by requested metric
        sort_map = {
            "stall_cycles": "stall_cycles",
            "execution_time": "execution_time_s",
            "launch_count": "launch_count",
            "total_cycles": "total_cycles",
        }
        sort_col = sort_map.get(sort_by, "stall_cycles")
        df = df.sort_values(sort_col, ascending=False)

        # Limit to top N if requested
        if top_n is not None:
            df = df.head(top_n)

        return df

    def get_program_totals(self) -> dict:
        """Get program-level totals for key GPU metrics.

        Returns:
            Dictionary with:
                - total_execution_time_s: Total GPU kernel time in seconds
                - total_stall_cycles: Total stall cycles across all kernels
                - total_cycles: Total GPU cycles
                - total_issue_cycles: Total issue cycles
                - total_kernel_launches: Total number of kernel invocations
                - stall_ratio: Overall stall ratio (0-1)
        """
        # Use get_gpu_metrics_summary which returns exclusive scope metrics
        # For program totals, exclusive is correct (no double-counting)
        summary = self.get_gpu_metrics_summary()

        total_cycles = summary.get(METRIC_GCYCLES, 0)
        stall_cycles = summary.get(METRIC_GCYCLES_STL, 0)

        return {
            "total_execution_time_s": summary.get(METRIC_GKER, 0),
            "total_stall_cycles": stall_cycles,
            "total_cycles": total_cycles,
            "total_issue_cycles": summary.get(METRIC_GCYCLES_ISU, 0),
            "total_kernel_launches": summary.get(METRIC_GKER_COUNT, 0),
            "stall_ratio": stall_cycles / total_cycles if total_cycles > 0 else 0.0,
        }
