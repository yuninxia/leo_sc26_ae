"""Tests for vendor-aware metric lookup."""

import pytest
from leo.analysis.metrics import (
    get_exec_dep_stall,
    get_mem_dep_stall,
    get_memory_stall_cycles,
    get_total_stall,
    get_stall_metrics_for_pruning,
    detect_vendor_from_arch_name,
)


class TestGetExecDepStall:
    """Tests for get_exec_dep_stall function."""

    def test_returns_idep_metric(self):
        """Should return gcycles:stl:idep value."""
        metrics = {"gcycles:stl:idep": 5000.0}
        assert get_exec_dep_stall(metrics) == 5000.0

    def test_returns_zero_if_missing(self):
        """Should return 0 if metric not present."""
        metrics = {}
        assert get_exec_dep_stall(metrics) == 0.0

    def test_vendor_agnostic(self):
        """Both vendors use the same metric name."""
        metrics = {"gcycles:stl:idep": 3000.0}
        assert get_exec_dep_stall(metrics, "nvidia") == 3000.0
        assert get_exec_dep_stall(metrics, "amd") == 3000.0


class TestGetMemDepStall:
    """Tests for get_mem_dep_stall function."""

    def test_nvidia_uses_gmem(self):
        """NVIDIA should use gcycles:stl:gmem."""
        metrics = {
            "gcycles:stl:mem": 1000.0,
            "gcycles:stl:gmem": 2000.0,
        }
        assert get_mem_dep_stall(metrics, "nvidia") == 2000.0

    def test_amd_uses_mem(self):
        """AMD should use gcycles:stl:mem as primary."""
        metrics = {
            "gcycles:stl:mem": 1000.0,
            "gcycles:stl:gmem": 2000.0,
        }
        assert get_mem_dep_stall(metrics, "amd") == 1000.0

    def test_amd_fallback_to_gmem(self):
        """AMD should fallback to gmem if mem is zero."""
        metrics = {
            "gcycles:stl:mem": 0.0,
            "gcycles:stl:gmem": 2000.0,
        }
        assert get_mem_dep_stall(metrics, "amd") == 2000.0

    def test_unknown_vendor_sums_both(self):
        """Unknown vendor should sum both metrics."""
        metrics = {
            "gcycles:stl:mem": 1000.0,
            "gcycles:stl:gmem": 2000.0,
        }
        assert get_mem_dep_stall(metrics, None) == 3000.0


class TestGetMemoryStallCycles:
    """Tests for get_memory_stall_cycles function."""

    def test_nvidia_gmem_plus_lmem(self):
        """NVIDIA should sum gmem + lmem."""
        metrics = {
            "gcycles:stl:gmem": 2000.0,
            "gcycles:stl:lmem": 500.0,
            "gcycles:stl:mem": 1000.0,  # Should be ignored
        }
        assert get_memory_stall_cycles(metrics, "nvidia") == 2500.0

    def test_amd_mem_plus_lmem(self):
        """AMD should sum mem + lmem."""
        metrics = {
            "gcycles:stl:mem": 1000.0,
            "gcycles:stl:lmem": 500.0,
            "gcycles:stl:gmem": 2000.0,  # Should be ignored
        }
        assert get_memory_stall_cycles(metrics, "amd") == 1500.0


class TestGetTotalStall:
    """Tests for get_total_stall function."""

    def test_returns_total(self):
        """Should return gcycles:stl value."""
        metrics = {"gcycles:stl": 10000.0}
        assert get_total_stall(metrics) == 10000.0

    def test_returns_zero_if_missing(self):
        """Should return 0 if metric not present."""
        metrics = {}
        assert get_total_stall(metrics) == 0.0


class TestGetStallMetricsForPruning:
    """Tests for get_stall_metrics_for_pruning function."""

    def test_returns_tuple(self):
        """Should return (exec_dep, mem_dep, total) tuple."""
        metrics = {
            "gcycles:stl:idep": 1000.0,
            "gcycles:stl:gmem": 2000.0,
            "gcycles:stl": 5000.0,
        }
        exec_dep, mem_dep, total = get_stall_metrics_for_pruning(metrics, "nvidia")
        assert exec_dep == 1000.0
        assert mem_dep == 2000.0
        assert total == 5000.0


class TestDetectVendorFromArchName:
    """Tests for detect_vendor_from_arch_name function."""

    def test_nvidia_architectures(self):
        """Should detect NVIDIA architectures."""
        assert detect_vendor_from_arch_name("a100") == "nvidia"
        assert detect_vendor_from_arch_name("v100") == "nvidia"
        assert detect_vendor_from_arch_name("A100") == "nvidia"
        assert detect_vendor_from_arch_name("h100") == "nvidia"

    def test_amd_architectures(self):
        """Should detect AMD architectures."""
        assert detect_vendor_from_arch_name("mi100") == "amd"
        assert detect_vendor_from_arch_name("mi200") == "amd"
        assert detect_vendor_from_arch_name("mi250") == "amd"
        assert detect_vendor_from_arch_name("mi300") == "amd"
        assert detect_vendor_from_arch_name("MI300") == "amd"
        assert detect_vendor_from_arch_name("gfx942") == "amd"

    def test_unknown_defaults_to_nvidia(self):
        """Unknown architectures should default to nvidia."""
        assert detect_vendor_from_arch_name("unknown") == "nvidia"
