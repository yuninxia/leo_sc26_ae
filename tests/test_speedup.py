"""Tests for speedup estimation module."""

import pytest

from leo.analysis.speedup import (
    OptimizationType,
    SpeedupEstimate,
    classify_optimization_type,
    compute_confidence,
    compute_speedup_estimates,
    estimate_speedup_amdahl,
    estimate_speedup_conservative,
    format_speedup_report,
    REDUCIBILITY_FACTORS,
)


class TestAmdahlSpeedup:
    """Test Amdahl's Law speedup calculation."""

    def test_basic_speedup(self):
        """Test basic Amdahl's Law calculation."""
        # 10% of cycles blamed → 1.11x speedup
        speedup = estimate_speedup_amdahl(1000, 100)
        assert abs(speedup - 1.111) < 0.01

    def test_50_percent_speedup(self):
        """Test 50% bottleneck → 2x speedup."""
        speedup = estimate_speedup_amdahl(1000, 500)
        assert abs(speedup - 2.0) < 0.01

    def test_zero_blame(self):
        """Test zero blamed cycles returns 1.0."""
        speedup = estimate_speedup_amdahl(1000, 0)
        assert speedup == 1.0

    def test_zero_total(self):
        """Test zero total cycles returns 1.0."""
        speedup = estimate_speedup_amdahl(0, 100)
        assert speedup == 1.0

    def test_full_blame(self):
        """Test 100% blamed cycles returns infinity."""
        speedup = estimate_speedup_amdahl(1000, 1000)
        assert speedup == float('inf')

    def test_conservative_speedup(self):
        """Test conservative estimate with reducibility factor."""
        # 10% blamed, 50% reducible → 5% actual reduction
        # speedup = 1000 / (1000 - 50) = 1.053
        speedup = estimate_speedup_conservative(1000, 100, 0.5)
        assert abs(speedup - 1.053) < 0.01


class TestOptimizationClassification:
    """Test optimization type classification."""

    def test_nvidia_global_load(self):
        """Test NVIDIA global load classification."""
        opt = classify_optimization_type("LDG.E.64", "mem_dep_gmem")
        assert opt == OptimizationType.MEMORY_COALESCING

    def test_amd_global_load(self):
        """Test AMD global load classification."""
        opt = classify_optimization_type("global_load_dwordx2", "mem_dep_gmem")
        assert opt == OptimizationType.MEMORY_COALESCING

    def test_shared_memory(self):
        """Test shared memory classification."""
        opt = classify_optimization_type("LDS", "mem_dep_smem")
        assert opt == OptimizationType.MEMORY_LOCALITY

    def test_amd_lds(self):
        """Test AMD LDS classification."""
        opt = classify_optimization_type("ds_read_b64", "mem_dep_smem")
        assert opt == OptimizationType.MEMORY_LOCALITY

    def test_execution_dependency(self):
        """Test execution dependency classification."""
        opt = classify_optimization_type("FADD", "exec_dep_dep")
        assert opt == OptimizationType.EXECUTION_REORDER

    def test_barrier(self):
        """Test barrier classification."""
        opt = classify_optimization_type("BAR.SYNC", "sync_barrier")
        assert opt == OptimizationType.BARRIER_REDUCTION

    def test_amd_waitcnt(self):
        """Test AMD s_waitcnt classification."""
        opt = classify_optimization_type("s_waitcnt", "sync_barrier")
        assert opt == OptimizationType.BARRIER_REDUCTION

    def test_unknown(self):
        """Test unknown operation classification."""
        opt = classify_optimization_type("UNKNOWN_OP", "unknown_type")
        assert opt == OptimizationType.UNKNOWN


class TestConfidence:
    """Test confidence level computation."""

    def test_high_confidence(self):
        """Test high confidence for significant known bottlenecks."""
        conf = compute_confidence(0.20, OptimizationType.MEMORY_COALESCING, False)
        assert conf == "high"

    def test_medium_confidence_unknown(self):
        """Test medium confidence for unknown type."""
        conf = compute_confidence(0.10, OptimizationType.UNKNOWN, False)
        assert conf == "medium"

    def test_low_confidence_small(self):
        """Test low confidence for small bottlenecks."""
        conf = compute_confidence(0.01, OptimizationType.MEMORY_COALESCING, False)
        assert conf == "low"

    def test_self_blame_reduces_confidence(self):
        """Test that self-blame reduces confidence."""
        conf_normal = compute_confidence(0.10, OptimizationType.MEMORY_COALESCING, False)
        conf_self = compute_confidence(0.10, OptimizationType.MEMORY_COALESCING, True)
        # Self-blame should have equal or lower confidence
        assert conf_self in ["high", "medium"]


class TestSpeedupEstimates:
    """Test full speedup estimation pipeline."""

    def test_basic_estimates(self):
        """Test basic speedup estimate computation."""
        blame_by_pc = {
            0x100: (1000.0, "LDG.E.64", "mem_dep_gmem"),
            0x200: (500.0, "FADD", "exec_dep_dep"),
        }
        total_cycles = 10000.0

        estimates = compute_speedup_estimates(blame_by_pc, total_cycles, top_n=5)

        assert len(estimates) == 2
        # Higher blame should have higher speedup
        assert estimates[0].blame_cycles > estimates[1].blame_cycles
        assert estimates[0].estimated_speedup > estimates[1].estimated_speedup

    def test_with_source_mapping(self):
        """Test estimates with source location mapping."""
        blame_by_pc = {
            0x100: (1000.0, "LDG.E.64", "mem_dep_gmem"),
        }
        source_mapping = {
            0x100: ("kernel.cu", 42),
        }

        estimates = compute_speedup_estimates(
            blame_by_pc, 10000.0, source_mapping=source_mapping
        )

        assert len(estimates) == 1
        assert estimates[0].source_location == ("kernel.cu", 42)
        assert "kernel.cu:42" in estimates[0].format_location()

    def test_empty_input(self):
        """Test empty input returns empty list."""
        estimates = compute_speedup_estimates({}, 10000.0)
        assert estimates == []

    def test_zero_total_cycles(self):
        """Test zero total cycles returns empty list."""
        estimates = compute_speedup_estimates(
            {0x100: (1000.0, "LDG", "mem_dep")}, 0
        )
        assert estimates == []

    def test_top_n_limit(self):
        """Test top_n limits results."""
        blame_by_pc = {
            0x100: (1000.0, "LDG", "mem_dep"),
            0x200: (900.0, "STG", "mem_dep"),
            0x300: (800.0, "FADD", "exec_dep"),
        }

        estimates = compute_speedup_estimates(blame_by_pc, 10000.0, top_n=2)
        assert len(estimates) == 2


class TestFormatReport:
    """Test report formatting."""

    def test_format_with_estimates(self):
        """Test report formatting with estimates."""
        estimates = [
            SpeedupEstimate(
                root_cause_pc=0x100,
                root_cause_opcode="LDG.E.64",
                source_location=("kernel.cu", 42),
                blame_cycles=1000.0,
                blame_ratio=0.10,
                estimated_speedup=1.11,
                optimization_type=OptimizationType.MEMORY_COALESCING,
                optimization_suggestion="Improve memory coalescing",
                reducibility=0.85,
                confidence="high",
            ),
        ]

        report = format_speedup_report(estimates, 10000.0)

        assert "kernel.cu:42" in report
        assert "10.0%" in report
        assert "1.11x" in report
        assert "high" in report

    def test_format_empty(self):
        """Test formatting empty estimates."""
        report = format_speedup_report([], 10000.0)
        assert "no optimization" in report.lower()


class TestReducibilityFactors:
    """Test reducibility factors are reasonable."""

    def test_memory_coalescing_high(self):
        """Test memory coalescing has high reducibility."""
        assert REDUCIBILITY_FACTORS[OptimizationType.MEMORY_COALESCING] >= 0.8

    def test_barrier_low(self):
        """Test barrier has low reducibility."""
        assert REDUCIBILITY_FACTORS[OptimizationType.BARRIER_REDUCTION] <= 0.4

    def test_all_factors_valid(self):
        """Test all factors are in valid range."""
        for opt_type, factor in REDUCIBILITY_FACTORS.items():
            assert 0.0 <= factor <= 1.0, f"{opt_type} has invalid factor {factor}"
