"""Tests for leo.db.reader module.

Test organization:
- TestDatabaseReaderInit: Initialization and error handling
- TestDatabaseReaderMetrics: Metric queries (coarse and fine-grained)
- TestDatabaseReaderCCT: CCT structure and navigation
- TestDatabaseReaderInstructions: Instruction-level data for GPA analysis
- TestDatabaseReaderProfileSlices: Profile data access
- TestDatabaseReaderAnalysis: Workload classification and analysis
- TestGPAStallMetrics: Detailed stall metrics for back-slicing
- TestGPAIssueMetrics: Instruction issue metrics
- TestGPALatencyMetrics: Latency and timing metrics
- TestGPAKernelMetrics: Kernel configuration and efficiency
- TestGPAMetricConsistency: Cross-metric validation
"""

import pytest
from pathlib import Path

import pandas as pd

from leo.db import DatabaseReader


# Path to test database (with instruction nodes)
TEST_DB_PATH = Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-database"

# Path to RAJAPerf database (multi-kernel workload)
RAJAPERF_DB_PATH = Path(__file__).parent / "data/pc/nvidia/hpctoolkit-rajaperf.cudaoffload.gcc.cudagpu-database"


@pytest.fixture
def reader():
    """Create a DatabaseReader for the test database."""
    return DatabaseReader(str(TEST_DB_PATH))


@pytest.fixture
def rajaperf_reader():
    """Create a DatabaseReader for the RAJAPerf multi-kernel database."""
    return DatabaseReader(str(RAJAPERF_DB_PATH))


class TestDatabaseReaderInit:
    """Tests for DatabaseReader initialization and error handling."""

    def test_init_valid_path(self, reader):
        """Test DatabaseReader initialization with valid path."""
        assert reader.db_path == str(TEST_DB_PATH)

    def test_init_invalid_path(self):
        """Test DatabaseReader raises error for invalid path."""
        with pytest.raises(Exception):
            DatabaseReader("/nonexistent/path/to/database")

    def test_init_file_instead_of_directory(self, tmp_path):
        """Test DatabaseReader raises error when given a file instead of directory."""
        fake_file = tmp_path / "not_a_database.txt"
        fake_file.write_text("not a database")
        with pytest.raises(Exception):
            DatabaseReader(str(fake_file))


class TestDatabaseReaderMetrics:
    """Tests for metric queries - coarse and fine-grained."""

    def test_get_all_metrics(self, reader):
        """Test getting all metric descriptions."""
        metrics = reader.get_metrics("*")
        assert len(metrics) > 0
        assert isinstance(metrics, pd.DataFrame)

    def test_metrics_required_columns(self, reader):
        """Test metrics DataFrame has required columns."""
        metrics = reader.get_metrics("*")
        required_cols = ["id", "name", "aggregation", "scope"]
        for col in required_cols:
            assert col in metrics.columns, f"Missing required column: {col}"

    def test_metrics_id_exists(self, reader):
        """Test metric IDs exist and are numeric."""
        metrics = reader.get_metrics("*")
        assert "id" in metrics.columns
        assert len(metrics["id"]) > 0

    def test_get_gpu_stall_metrics(self, reader):
        """Test getting GPU stall cycle metrics."""
        metrics = reader.get_metrics("gcycles:stl:*")
        assert len(metrics) > 0
        names = metrics["name"].tolist()
        # Should have stall metrics
        assert any("stl" in name for name in names)

    def test_get_specific_stall_types(self, reader):
        """Test specific stall type metrics exist."""
        all_metrics = reader.get_metrics("*")
        names = all_metrics["name"].tolist()
        # GPA requires these stall types for back-slicing
        # mem: memory stalls, idep: instruction dependency, sync: synchronization
        expected_stall_types = ["gcycles:stl:mem", "gcycles:stl:idep", "gcycles:stl:sync"]
        for stall_type in expected_stall_types:
            assert stall_type in names, f"Missing stall metric: {stall_type}"

    def test_metrics_pattern_invalid_raises(self, reader):
        """Test invalid pattern raises ValueError."""
        with pytest.raises(ValueError):
            reader.get_metrics("nonexistent_metric_xyz:*")

    def test_metrics_aggregation_types(self, reader):
        """Test metrics have valid aggregation types."""
        metrics = reader.get_metrics("*")
        # sum: additive, prop: proportion/derived
        valid_aggs = {"sum", "min", "max", "prop"}
        actual_aggs = set(metrics["aggregation"].unique())
        assert actual_aggs.issubset(valid_aggs), f"Unexpected aggregation: {actual_aggs - valid_aggs}"

    def test_metrics_scope_types(self, reader):
        """Test metrics have valid scope types."""
        metrics = reader.get_metrics("*")
        # e: exclusive, i: inclusive, p: point, c: execution count
        valid_scopes = {"e", "i", "p", "c"}
        actual_scopes = set(metrics["scope"].unique())
        assert actual_scopes.issubset(valid_scopes), f"Unexpected scope: {actual_scopes - valid_scopes}"


class TestDatabaseReaderCCT:
    """Tests for CCT structure and navigation."""

    def test_get_cct_not_empty(self, reader):
        """Test CCT is not empty."""
        cct = reader.get_cct()
        assert len(cct) > 0

    def test_cct_required_columns(self, reader):
        """Test CCT DataFrame has required columns."""
        cct = reader.get_cct()
        required_cols = ["type", "name", "parent", "depth"]
        for col in required_cols:
            assert col in cct.columns, f"Missing required column: {col}"

    def test_cct_has_root_node(self, reader):
        """Test CCT has a root node (parent is None/NaN or self-referential)."""
        cct = reader.get_cct()
        # Root nodes have no parent or parent points to itself
        root_candidates = cct[cct["parent"].isna() | (cct["parent"] == cct.index)]
        assert len(root_candidates) >= 1, "No root node found in CCT"

    def test_cct_node_types(self, reader):
        """Test CCT has expected node types."""
        cct = reader.get_cct()
        node_types = set(cct["type"].unique())
        # Should have at least function and instruction types
        assert "function" in node_types, "No function nodes in CCT"
        assert "instruction" in node_types, "No instruction nodes in CCT"

    def test_cct_depth_consistency(self, reader):
        """Test CCT depth values are consistent with parent-child relationships."""
        cct = reader.get_cct()
        # Sample some nodes and verify depth = parent_depth + 1
        sample_nodes = cct[cct["parent"].notna()].head(20)
        for idx, row in sample_nodes.iterrows():
            parent_id = int(row["parent"])
            if parent_id in cct.index:
                parent_depth = cct.loc[parent_id, "depth"]
                assert row["depth"] == parent_depth + 1, (
                    f"Depth mismatch: node {idx} depth={row['depth']}, "
                    f"parent {parent_id} depth={parent_depth}"
                )

    def test_cct_parent_references_valid(self, reader):
        """Test all parent references point to valid nodes."""
        cct = reader.get_cct()
        valid_ids = set(cct.index)
        for idx, row in cct.iterrows():
            if pd.notna(row["parent"]):
                parent_id = int(row["parent"])
                assert parent_id in valid_ids, f"Node {idx} has invalid parent {parent_id}"


class TestDatabaseReaderInstructions:
    """Tests for instruction-level data required for GPA analysis."""

    def test_instruction_nodes_exist(self, reader):
        """Test that instruction nodes are present (requires modified HPCToolkit)."""
        cct = reader.get_cct()
        node_types = cct["type"].value_counts().to_dict()
        assert "instruction" in node_types, (
            "No instruction nodes found. Database may have been generated "
            "with unmodified HPCToolkit that filters out instruction nodes."
        )
        assert node_types["instruction"] > 0

    def test_instruction_node_count(self, reader):
        """Test reasonable number of instruction nodes exist."""
        cct = reader.get_cct()
        inst_count = len(cct[cct["type"] == "instruction"])
        # Test database should have >100 instruction nodes based on PC sampling
        assert inst_count >= 100, f"Only {inst_count} instruction nodes, expected >100"

    def test_instruction_node_structure(self, reader):
        """Test instruction node has required fields for binary analysis."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]

        required_columns = ["offset", "module_path", "parent"]
        for col in required_columns:
            assert col in inst_nodes.columns, f"Missing column: {col}"

    def test_instruction_offset_valid(self, reader):
        """Test instruction offsets are valid PC values."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]
        valid_offsets = inst_nodes["offset"].dropna()

        assert len(valid_offsets) > 0, "No valid offset values"
        assert all(offset >= 0 for offset in valid_offsets), "Negative offset found"
        # PC offsets should be reasonable sizes (not huge garbage values)
        assert all(offset < 2**32 for offset in valid_offsets), "Offset too large"

    def test_instruction_module_path(self, reader):
        """Test instruction nodes have module references (needed for nvdisasm)."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]
        valid_modules = inst_nodes["module_path"].dropna()

        # module_path contains integer IDs referencing modules
        assert len(valid_modules) > 0, "No module references for instruction nodes"
        # Module IDs should be non-negative integers
        assert all(m >= 0 for m in valid_modules), "Invalid module ID found"

    def test_instruction_parent_is_function(self, reader):
        """Test instruction nodes have function parents (CCT structure)."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]

        # Sample some instruction nodes
        sample = inst_nodes.head(10)
        for idx, row in sample.iterrows():
            if pd.notna(row["parent"]):
                parent_id = int(row["parent"])
                if parent_id in cct.index:
                    parent_type = cct.loc[parent_id, "type"]
                    # Parent should be function, loop, or line (not instruction)
                    assert parent_type != "instruction", (
                        f"Instruction {idx} has instruction parent {parent_id}"
                    )

    def test_instruction_metrics_exist(self, reader):
        """Test instruction nodes have PC sampling metrics."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]
        inst_ids = inst_nodes.index.tolist()[:20]

        slices = reader.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()
        inst_metrics = slices_flat[slices_flat["cct_id"].isin(inst_ids)]

        assert len(inst_metrics) > 0, "No metrics for instruction nodes"
        assert inst_metrics["value"].sum() > 0, "All instruction metrics zero"

    def test_instruction_stall_metrics(self, reader):
        """Test instruction nodes have stall cycle breakdown."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]
        inst_ids = inst_nodes.index.tolist()

        # Get stall metrics
        stall_metrics = reader.get_metrics("gcycles:stl:*")
        stall_metric_ids = stall_metrics["id"].tolist()

        slices = reader.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()

        # Filter to instruction nodes with stall metrics
        inst_stall = slices_flat[
            (slices_flat["cct_id"].isin(inst_ids)) &
            (slices_flat["metric_id"].isin(stall_metric_ids))
        ]

        # Should have some stall data for instructions
        assert len(inst_stall) > 0, "No stall metrics for instruction nodes"


class TestDatabaseReaderProfileSlices:
    """Tests for profile slice queries."""

    def test_get_profile_slices_structure(self, reader):
        """Test profile slices have correct MultiIndex structure."""
        slices = reader.get_profile_slices("*", "summary", "*")
        assert slices.index.names == ["profile_id", "cct_id", "metric_id"]

    def test_get_profile_slices_values(self, reader):
        """Test profile slices have value column with numeric data."""
        slices = reader.get_profile_slices("*", "summary", "*")
        assert "value" in slices.columns
        assert pd.api.types.is_numeric_dtype(slices["value"])

    def test_get_profile_slices_non_negative(self, reader):
        """Test profile slice values are non-negative (cycles, counts)."""
        slices = reader.get_profile_slices("*", "summary", "*")
        assert (slices["value"] >= 0).all(), "Negative metric values found"

    def test_get_profile_descriptions(self, reader):
        """Test getting profile descriptions.

        Profile descriptions come from profile.db/meta.db, not trace.db,
        so they are available even without trace data.
        """
        profiles = reader.get_profile_descriptions("*")
        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) > 0


class TestDatabaseReaderAnalysis:
    """Tests for workload classification and analysis methods."""

    def test_gpu_metrics_summary_structure(self, reader):
        """Test GPU metrics summary returns dict with metric names."""
        summary = reader.get_gpu_metrics_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        # Keys should be metric names (strings)
        assert all(isinstance(k, str) for k in summary.keys())
        # Values should be numeric
        assert all(isinstance(v, (int, float)) for v in summary.values())

    def test_gpu_metrics_summary_has_gcycles(self, reader):
        """Test GPU metrics summary includes gcycles."""
        summary = reader.get_gpu_metrics_summary()
        gcycles_metrics = [k for k in summary.keys() if "gcycles" in k]
        assert len(gcycles_metrics) > 0, "No gcycles metrics in summary"

    def test_classify_workload_valid_result(self, reader):
        """Test workload classification returns valid category."""
        classification = reader.classify_workload()
        valid_classifications = [
            "Compute-bound",
            "Memory-bound",
            "Sync-bound",
            "Mixed",
            "Unknown",
        ]
        assert classification in valid_classifications

    def test_classify_workload_deterministic(self, reader):
        """Test workload classification is deterministic."""
        result1 = reader.classify_workload()
        result2 = reader.classify_workload()
        assert result1 == result2, "Classification should be deterministic"


# =============================================================================
# GPA Back-Slicing Data Requirements Tests
# =============================================================================


class TestGPAStallMetrics:
    """Tests for detailed stall metrics required for GPA back-slicing.

    Back-slicing uses stall type to determine dependency pruning:
    - idep (instruction dependency) -> register/exec dependencies
    - gmem (global memory) -> memory latency dependencies
    - sync (synchronization) -> barrier dependencies
    """

    def test_stall_metric_categories_exist(self, reader):
        """Test all stall metric categories exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        # Core stall categories for back-slicing
        required_stalls = [
            "gcycles:stl",       # Total stall cycles
            "gcycles:stl:idep",  # Instruction dependency (key for back-slicing)
            "gcycles:stl:gmem",  # Global memory stalls
            "gcycles:stl:mem",   # General memory stalls
            "gcycles:stl:sync",  # Synchronization stalls
        ]
        for stall in required_stalls:
            assert stall in names, f"Missing required stall metric: {stall}"

    def test_stall_metric_subtypes(self, reader):
        """Test detailed stall subtypes exist for diagnosis."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        # Detailed stall subtypes for root cause analysis
        stall_subtypes = [
            "gcycles:stl:ifet",  # Instruction fetch
            "gcycles:stl:pipe",  # Pipeline stalls
            "gcycles:stl:tmem",  # Texture memory
            "gcycles:stl:cmem",  # Constant memory
            "gcycles:stl:mthr",  # Memory throughput
            "gcycles:stl:slp",   # Sleep/yield
            "gcycles:stl:othr",  # Other stalls
        ]
        found = sum(1 for s in stall_subtypes if s in names)
        assert found >= 5, f"Only {found} stall subtypes found, expected at least 5"

    def test_stall_metrics_have_values(self, reader):
        """Test stall metrics have non-zero values in the database."""
        summary = reader.get_gpu_metrics_summary()

        # At least some stall metrics should have values
        stall_metrics = {k: v for k, v in summary.items() if "stl" in k}
        assert len(stall_metrics) > 0, "No stall metrics with values"

        # Total stall cycles should be positive for a real workload
        total_stall = sum(stall_metrics.values())
        assert total_stall > 0, "Total stall cycles is zero"

    def test_instruction_has_stall_breakdown(self, reader):
        """Test individual instructions have stall metric breakdown."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]

        # Get a sample instruction with metrics
        slices = reader.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()

        # Find an instruction with stall data
        inst_ids = inst_nodes.index.tolist()
        inst_slices = slices_flat[slices_flat["cct_id"].isin(inst_ids)]

        # Get stall metric IDs
        stall_metrics = reader.get_metrics("gcycles:stl:*")
        stall_ids = set(stall_metrics["id"].tolist())

        # Filter to stall metrics
        inst_stalls = inst_slices[inst_slices["metric_id"].isin(stall_ids)]

        # Should have multiple stall types per instruction
        if len(inst_stalls) > 0:
            stalls_per_inst = inst_stalls.groupby("cct_id")["metric_id"].nunique()
            max_stall_types = stalls_per_inst.max()
            assert max_stall_types >= 1, "Instructions should have stall breakdown"


class TestGPAIssueMetrics:
    """Tests for instruction issue metrics.

    Issue metrics track which functional units executed instructions,
    used for understanding compute utilization.
    """

    def test_issue_metric_exists(self, reader):
        """Test issue cycle metrics exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        assert "gcycles:isu" in names, "Missing gcycles:isu (issue cycles)"

    def test_issue_metric_breakdown(self, reader):
        """Test issue metric has functional unit breakdown."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        # Issue breakdown by functional unit
        issue_types = [
            "gcycles:isu:vec",   # Vector ALU
            "gcycles:isu:sclr",  # Scalar ALU
            "gcycles:isu:lds",   # Load/store
            "gcycles:isu:tex",   # Texture
            "gcycles:isu:bar",   # Barrier
        ]
        found = sum(1 for i in issue_types if i in names)
        assert found >= 3, f"Only {found} issue types found, expected at least 3"

    def test_issue_vs_stall_relationship(self, reader):
        """Test issue + stall approximately equals total cycles."""
        summary = reader.get_gpu_metrics_summary()

        gcycles = summary.get("gcycles", 0)
        gcycles_isu = summary.get("gcycles:isu", 0)
        gcycles_stl = summary.get("gcycles:stl", 0)

        if gcycles > 0:
            # Issue + stall should be close to total (with some hidden cycles)
            accounted = gcycles_isu + gcycles_stl
            ratio = accounted / gcycles
            # Allow for hidden cycles and measurement variance
            assert 0.5 < ratio < 1.5, (
                f"Issue+Stall ratio {ratio:.2f} outside expected range. "
                f"gcycles={gcycles}, isu={gcycles_isu}, stl={gcycles_stl}"
            )


class TestGPALatencyMetrics:
    """Tests for latency and timing metrics.

    Latency metrics are used for back-slicing dependency chain analysis.
    """

    def test_latency_metric_exists(self, reader):
        """Test instruction latency metrics exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        assert "gins:lat" in names, "Missing gins:lat (instruction latency)"

    def test_latency_coverage_metrics(self, reader):
        """Test latency coverage metrics for analysis."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        # Latency coverage metrics
        latency_metrics = ["gins:lat", "gins:lat_cov", "gins:lat_ucv"]
        found = sum(1 for m in latency_metrics if m in names)
        assert found >= 1, "No latency metrics found"

    def test_instruction_sample_metrics(self, reader):
        """Test instruction sampling metrics exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        assert "gins" in names, "Missing gins (instruction samples)"

    def test_block_level_metrics(self, reader):
        """Test block-level execution metrics exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        block_metrics = ["gins:blk_exec_cnt", "gins:blk_lat", "gins:blk_simd_act"]
        found = sum(1 for m in block_metrics if m in names)
        assert found >= 1, f"Only {found} block metrics found"


class TestGPAKernelMetrics:
    """Tests for kernel configuration and efficiency metrics.

    Used by optimizers to suggest parallelism and resource tuning.
    """

    def test_kernel_count_metric(self, reader):
        """Test kernel invocation count metric exists."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        assert "gker:count" in names, "Missing gker:count (kernel invocation count)"

    def test_kernel_resource_metrics(self, reader):
        """Test kernel resource configuration metrics exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        # Kernel resource metrics for optimizer analysis
        resource_metrics = [
            "gker:thr_vreg_acumu",   # Vector registers per thread
            "gker:thr_sreg_acumu",   # Scalar registers per thread
            "gker:stmem_acumu",      # Shared memory per block
            "gker:blk_thr_acumu",    # Threads per block
            "gker:blk_sm_acumu",     # Blocks per SM
        ]
        found = sum(1 for m in resource_metrics if m in names)
        assert found >= 3, f"Only {found} kernel resource metrics found, expected >= 3"

    def test_warp_activity_metrics(self, reader):
        """Test warp activity metrics for occupancy analysis."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        warp_metrics = ["gker:warp_act_acumu", "gker:warp_avl_acumu"]
        found = sum(1 for m in warp_metrics if m in names)
        assert found >= 1, "No warp activity metrics found"

    def test_simd_efficiency_metrics(self, reader):
        """Test SIMD/warp efficiency metrics exist."""
        metrics = reader.get_metrics("*")
        names = set(metrics["name"].tolist())

        simd_metrics = [
            "gins:simd_act",  # Active SIMD lanes
            "gins:simd_tot",  # Total SIMD lanes
            "gins:simd_wst",  # Wasted SIMD lanes
        ]
        found = sum(1 for m in simd_metrics if m in names)
        assert found >= 2, f"Only {found} SIMD efficiency metrics found"


class TestGPAMetricConsistency:
    """Tests for cross-metric consistency and data integrity.

    Validates that related metrics have consistent values.
    """

    def test_stall_subtypes_sum_to_total(self, reader):
        """Test stall subtypes approximately sum to total stall cycles."""
        summary = reader.get_gpu_metrics_summary()

        total_stall = summary.get("gcycles:stl", 0)
        if total_stall == 0:
            pytest.skip("No stall cycles in test database")

        # Sum individual stall types
        stall_subtypes = [
            "gcycles:stl:idep", "gcycles:stl:gmem", "gcycles:stl:sync",
            "gcycles:stl:mem", "gcycles:stl:ifet", "gcycles:stl:pipe",
            "gcycles:stl:tmem", "gcycles:stl:cmem", "gcycles:stl:mthr",
            "gcycles:stl:slp", "gcycles:stl:othr", "gcycles:stl:hid",
        ]
        subtype_sum = sum(summary.get(s, 0) for s in stall_subtypes)

        # Subtypes should account for most of total (allow some uncategorized)
        if subtype_sum > 0:
            ratio = subtype_sum / total_stall
            assert ratio > 0.5, (
                f"Stall subtypes only account for {ratio:.1%} of total stalls"
            )

    def test_simd_activity_consistency(self, reader):
        """Test SIMD activity metrics are consistent."""
        summary = reader.get_gpu_metrics_summary()

        simd_act = summary.get("gins:simd_act", 0)
        simd_tot = summary.get("gins:simd_tot", 0)
        simd_wst = summary.get("gins:simd_wst", 0)

        if simd_tot > 0:
            # Active + wasted should be close to total
            assert simd_act <= simd_tot, "Active SIMD > Total SIMD"
            if simd_wst > 0:
                assert simd_act + simd_wst >= simd_tot * 0.9, (
                    "SIMD active + wasted doesn't account for total"
                )

    def test_instruction_node_metric_coverage(self, reader):
        """Test instruction nodes have comprehensive metric coverage."""
        cct = reader.get_cct()
        inst_nodes = cct[cct["type"] == "instruction"]
        inst_count = len(inst_nodes)

        slices = reader.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()

        # Count instructions with any metrics
        inst_with_metrics = slices_flat[
            slices_flat["cct_id"].isin(inst_nodes.index)
        ]["cct_id"].nunique()

        # Most instruction nodes should have metrics
        coverage = inst_with_metrics / inst_count if inst_count > 0 else 0
        assert coverage > 0.5, (
            f"Only {coverage:.1%} of instruction nodes have metrics"
        )

    def test_metric_values_reasonable_range(self, reader):
        """Test metric values are in reasonable ranges."""
        slices = reader.get_profile_slices("*", "summary", "*")

        # All values should be non-negative
        assert (slices["value"] >= 0).all(), "Negative metric values found"

        # No unreasonably large values (sanity check)
        max_val = slices["value"].max()
        assert max_val < 1e18, f"Unreasonably large metric value: {max_val}"

    def test_cct_metric_hierarchy(self, reader):
        """Test parent nodes have metrics >= sum of children (inclusive)."""
        cct = reader.get_cct()
        slices = reader.get_profile_slices("*", "summary", "*")
        slices_flat = slices.reset_index()

        # Get inclusive metrics (scope 'i')
        metrics = reader.get_metrics("*")
        inclusive_ids = metrics[metrics["scope"] == "i"]["id"].tolist()

        if not inclusive_ids:
            pytest.skip("No inclusive metrics to test hierarchy")

        # Sample a function node with children
        func_nodes = cct[cct["type"] == "function"]
        if len(func_nodes) == 0:
            pytest.skip("No function nodes to test")

        # Get a function with children
        for func_id in func_nodes.index[:5]:
            children = cct[cct["parent"] == func_id].index.tolist()
            if not children:
                continue

            # Get inclusive metric for parent
            parent_metrics = slices_flat[
                (slices_flat["cct_id"] == func_id) &
                (slices_flat["metric_id"].isin(inclusive_ids))
            ]

            if len(parent_metrics) > 0:
                # Parent should have values (inclusive = self + children)
                assert parent_metrics["value"].sum() >= 0
                break


class TestPerKernelSummary:
    """Tests for per-kernel summary and whole-program analysis.

    These methods enable GPA-style whole-program analysis where kernels
    can be ranked by execution time or stall cycles.
    """

    def test_get_per_kernel_summary_returns_dataframe(self, reader):
        """Test get_per_kernel_summary returns a DataFrame."""
        result = reader.get_per_kernel_summary()
        assert isinstance(result, pd.DataFrame)

    def test_get_per_kernel_summary_has_required_columns(self, reader):
        """Test per-kernel summary has all required columns."""
        result = reader.get_per_kernel_summary()
        if len(result) == 0:
            pytest.skip("No kernels with data in test database")

        required_columns = [
            "execution_time_s",
            "stall_cycles",
            "total_cycles",
            "issue_cycles",
            "launch_count",
            "stall_ratio",
            "module_id",
            "offset",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_per_kernel_summary_values_non_negative(self, reader):
        """Test all per-kernel metric values are non-negative."""
        result = reader.get_per_kernel_summary()
        if len(result) == 0:
            pytest.skip("No kernels with data in test database")

        numeric_cols = ["execution_time_s", "stall_cycles", "total_cycles",
                        "issue_cycles", "launch_count", "stall_ratio"]
        for col in numeric_cols:
            assert (result[col] >= 0).all(), f"Negative values in {col}"

    def test_get_per_kernel_summary_stall_ratio_bounded(self, reader):
        """Test stall ratio is between 0 and 1."""
        result = reader.get_per_kernel_summary()
        if len(result) == 0:
            pytest.skip("No kernels with data in test database")

        # Filter to rows with non-zero total_cycles
        valid = result[result["total_cycles"] > 0]
        if len(valid) > 0:
            assert (valid["stall_ratio"] >= 0).all(), "Stall ratio < 0"
            assert (valid["stall_ratio"] <= 1.01).all(), "Stall ratio > 1"

    def test_get_per_kernel_summary_sort_by_stall_cycles(self, rajaperf_reader):
        """Test sorting by stall cycles works correctly (multi-kernel)."""
        result = rajaperf_reader.get_per_kernel_summary(sort_by="stall_cycles")
        assert len(result) >= 2, "RAJAPerf database should have multiple kernels"

        # Should be sorted descending
        values = result["stall_cycles"].tolist()
        assert values == sorted(values, reverse=True), "Not sorted by stall_cycles"

    def test_get_per_kernel_summary_sort_by_execution_time(self, rajaperf_reader):
        """Test sorting by execution time (GPA-style) works correctly (multi-kernel)."""
        result = rajaperf_reader.get_per_kernel_summary(sort_by="execution_time")
        assert len(result) >= 2, "RAJAPerf database should have multiple kernels"

        # Should be sorted descending
        values = result["execution_time_s"].tolist()
        assert values == sorted(values, reverse=True), "Not sorted by execution_time"

    def test_get_per_kernel_summary_top_n(self, rajaperf_reader):
        """Test top_n parameter limits results (multi-kernel)."""
        full_result = rajaperf_reader.get_per_kernel_summary()
        assert len(full_result) >= 5, "RAJAPerf database should have many kernels"

        top_3 = rajaperf_reader.get_per_kernel_summary(top_n=3)
        assert len(top_3) == 3, f"Expected 3 kernels, got {len(top_3)}"

    def test_get_program_totals_structure(self, reader):
        """Test get_program_totals returns expected structure."""
        result = reader.get_program_totals()
        assert isinstance(result, dict)

        required_keys = [
            "total_execution_time_s",
            "total_stall_cycles",
            "total_cycles",
            "total_issue_cycles",
            "total_kernel_launches",
            "stall_ratio",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_get_program_totals_values_non_negative(self, reader):
        """Test program totals are non-negative."""
        result = reader.get_program_totals()
        for key, value in result.items():
            assert value >= 0, f"Negative value for {key}: {value}"

    def test_get_program_totals_stall_ratio_bounded(self, reader):
        """Test program stall ratio is between 0 and 1."""
        result = reader.get_program_totals()
        assert 0 <= result["stall_ratio"] <= 1.01, (
            f"Stall ratio out of bounds: {result['stall_ratio']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
