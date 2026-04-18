"""Tests for leo.program_analysis module.

Tests whole-program analysis functionality including kernel discovery,
ranking, and aggregation.
"""

import pytest
from pathlib import Path

from leo.program_analysis import (
    PerKernelAnalysis,
    ProgramAnalysisResult,
    analyze_program,
)


# Path to test measurements directory (AMD LAMMPS)
TEST_MEASUREMENTS_DIR = Path(__file__).parent / "data/pc/amd/hpctoolkit-lammps.rocm7.2.0.kokkos4.6.2-measurements"


@pytest.fixture
def measurements_dir():
    """Return test measurements directory path."""
    if not TEST_MEASUREMENTS_DIR.exists():
        pytest.skip(f"Test measurements not found: {TEST_MEASUREMENTS_DIR}")
    return str(TEST_MEASUREMENTS_DIR)


class TestPerKernelAnalysis:
    """Tests for PerKernelAnalysis dataclass."""

    def test_per_kernel_analysis_creation(self):
        """Test PerKernelAnalysis can be created with required fields."""
        analysis = PerKernelAnalysis(
            cct_id=123,
            gpubin_path="/path/to/kernel.gpubin",
            execution_time_s=1.5,
            stall_cycles=1000000,
            total_cycles=1200000,
            stall_ratio=0.833,
            launch_count=10,
        )
        assert analysis.cct_id == 123
        assert analysis.execution_time_s == 1.5
        assert analysis.stall_cycles == 1000000
        assert not analysis.analyzed

    def test_per_kernel_analysis_not_analyzed(self):
        """Test PerKernelAnalysis without full analysis."""
        analysis = PerKernelAnalysis(
            cct_id=1,
            gpubin_path="/path/to/kernel.gpubin",
            execution_time_s=0.5,
            stall_cycles=500000,
            total_cycles=600000,
            stall_ratio=0.833,
            launch_count=5,
        )
        assert not analysis.analyzed
        assert analysis.top_blame_sources == []
        assert analysis.get_speedup_estimates() == []

    def test_kernel_name_from_gpubin(self):
        """Test kernel_name falls back to gpubin stem."""
        analysis = PerKernelAnalysis(
            cct_id=1,
            gpubin_path="/path/to/abc123.gpubin",
            execution_time_s=0.5,
            stall_cycles=500000,
            total_cycles=600000,
            stall_ratio=0.833,
            launch_count=5,
        )
        assert analysis.kernel_name == "abc123"

    def test_short_name_extraction(self):
        """Test short_name extracts readable name from gpubin."""
        # Simple gpubin name should stay as-is
        analysis = PerKernelAnalysis(
            cct_id=1,
            gpubin_path="/path/to/abc123.gpubin",
            execution_time_s=0.5,
            stall_cycles=500000,
            total_cycles=600000,
            stall_ratio=0.833,
            launch_count=5,
        )
        # short_name should be derived from kernel_name
        assert analysis.short_name is not None
        assert len(analysis.short_name) > 0


class TestProgramAnalysisResult:
    """Tests for ProgramAnalysisResult dataclass."""

    def test_result_creation(self):
        """Test ProgramAnalysisResult can be created."""
        result = ProgramAnalysisResult(
            database_path="/path/to/database",
            measurements_dir="/path/to/measurements",
            program_totals={
                "total_execution_time_s": 10.0,
                "total_stall_cycles": 1000000000,
                "total_cycles": 1200000000,
                "stall_ratio": 0.833,
            },
            per_kernel_results=[],
        )
        assert result.database_path == "/path/to/database"
        assert result.total_execution_time_s == 10.0
        assert result.total_stall_cycles == 1000000000

    def test_get_top_kernels(self):
        """Test get_top_kernels returns correct number."""
        kernels = [
            PerKernelAnalysis(
                cct_id=i,
                gpubin_path=f"/path/{i}.gpubin",
                execution_time_s=float(i),
                stall_cycles=i * 1000,
                total_cycles=i * 1200,
                stall_ratio=0.833,
                launch_count=1,
            )
            for i in range(10)
        ]
        result = ProgramAnalysisResult(
            database_path="/path/to/db",
            measurements_dir="/path/to/meas",
            program_totals={"total_stall_cycles": 0, "stall_ratio": 0},
            per_kernel_results=kernels,
        )
        assert len(result.get_top_kernels(5)) == 5
        assert len(result.get_top_kernels(15)) == 10  # Only 10 available

    def test_to_json(self):
        """Test JSON export includes required fields."""
        result = ProgramAnalysisResult(
            database_path="/path/to/db",
            measurements_dir="/path/to/meas",
            program_totals={"total_stall_cycles": 1000, "stall_ratio": 0.5},
            per_kernel_results=[],
        )
        json_data = result.to_json()
        assert "database_path" in json_data
        assert "program_totals" in json_data
        assert "per_kernel_results" in json_data

    def test_summary_format(self):
        """Test summary returns non-empty string."""
        result = ProgramAnalysisResult(
            database_path="/path/to/db",
            measurements_dir="/path/to/meas",
            program_totals={
                "total_execution_time_s": 1.0,
                "total_stall_cycles": 1000,
                "total_cycles": 1200,
                "stall_ratio": 0.833,
            },
            per_kernel_results=[],
        )
        summary = result.summary()
        assert len(summary) > 0
        assert "PROGRAM TOTALS" in summary


class TestAnalyzeProgram:
    """Tests for analyze_program function."""

    def test_analyze_program_metrics_only(self, measurements_dir):
        """Test analyze_program with metrics only (no full analysis)."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=3,
            run_full_analysis=False,
        )
        assert isinstance(result, ProgramAnalysisResult)
        assert len(result.per_kernel_results) <= 3
        assert result.kernels_analyzed >= 0

    def test_analyze_program_returns_result(self, measurements_dir):
        """Test analyze_program returns ProgramAnalysisResult."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=1,
            run_full_analysis=False,
        )
        assert isinstance(result, ProgramAnalysisResult)
        assert result.database_path is not None
        assert result.measurements_dir is not None

    def test_analyze_program_sort_by_stall_cycles(self, measurements_dir):
        """Test kernels are sorted by stall_cycles."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=5,
            sort_by="stall_cycles",
            run_full_analysis=False,
        )
        if len(result.per_kernel_results) >= 2:
            stall_values = [k.stall_cycles for k in result.per_kernel_results]
            assert stall_values == sorted(stall_values, reverse=True)

    def test_analyze_program_sort_by_execution_time(self, measurements_dir):
        """Test kernels are sorted by execution_time."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=5,
            sort_by="execution_time",
            run_full_analysis=False,
        )
        if len(result.per_kernel_results) >= 2:
            time_values = [k.execution_time_s for k in result.per_kernel_results]
            assert time_values == sorted(time_values, reverse=True)

    def test_analyze_program_invalid_measurements_dir(self):
        """Test analyze_program raises error for invalid directory."""
        with pytest.raises(FileNotFoundError):
            analyze_program(
                measurements_dir="/nonexistent/path",
                arch="mi250",
            )

    def test_analyze_program_program_totals(self, measurements_dir):
        """Test program_totals contains expected keys."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=1,
            run_full_analysis=False,
        )
        assert result.total_stall_cycles >= 0
        assert result.total_execution_time_s >= 0
        assert 0 <= result.stall_ratio <= 1.01


class TestAnalyzeProgramWithFullAnalysis:
    """Tests for analyze_program with full Leo analysis.

    These tests are slower as they run the full back-slicing pipeline.
    """

    @pytest.mark.slow
    def test_analyze_program_full_analysis(self, measurements_dir):
        """Test analyze_program with full Leo analysis on top kernel."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=1,
            run_full_analysis=True,
            skip_failed_kernels=True,
        )
        assert result.kernels_analyzed >= 0

        # If analysis succeeded, check results
        analyzed = result.get_analyzed_kernels()
        if analyzed:
            kernel = analyzed[0]
            assert kernel.analyzed
            assert kernel.analysis_result is not None
            # Should have blame sources
            assert len(kernel.top_blame_sources) > 0

    @pytest.mark.slow
    def test_analyze_program_top_blame_overall(self, measurements_dir):
        """Test cross-kernel blame aggregation."""
        result = analyze_program(
            measurements_dir=measurements_dir,
            arch="mi250",
            top_n_kernels=1,
            run_full_analysis=True,
            skip_failed_kernels=True,
        )

        if result.kernels_analyzed > 0:
            top_blamed = result.get_top_blame_sources_overall(5)
            # Should have format (kernel_name, pc, blame, opcode)
            for item in top_blamed:
                assert len(item) == 4
                kernel_name, pc, blame, opcode = item
                assert isinstance(kernel_name, str)
                assert isinstance(pc, int)
                assert blame >= 0
                assert isinstance(opcode, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
