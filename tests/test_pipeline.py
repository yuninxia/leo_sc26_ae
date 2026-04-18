"""Tests for leo.analysis.pipeline module - GPU per-pipeline CPI stack analysis.

Test organization:
- TestPipelineMetricValue: PipelineMetricValue dataclass tests
- TestPipelineResult: PipelineResult dataclass tests
- TestPipelineAnalyzer: PipelineAnalyzer class tests
- TestPipelineAnalyzerNvidia: NVIDIA-specific integration tests
- TestPipelineAnalyzerAmd: AMD-specific integration tests
- TestPipelineAnalyzerIntel: Intel-specific integration tests
- TestPipelineConvenienceFunctions: Convenience function tests
"""

import json
import pytest
from pathlib import Path

from leo.db import DatabaseReader
from leo.analysis.pipeline import (
    PipelineMetricValue,
    PipelineResult,
    PipelineAnalyzer,
    pipeline_analysis,
    print_pipeline_summary,
    PIPELINES,
)


# Test database paths
NVIDIA_DB_PATH = Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-database"
AMD_DB_PATH = Path(__file__).parent / "data/pc/amd/hpctoolkit-single.hipoffload.amdclang.rocmgpu-database"
INTEL_DB_PATH = Path(__file__).parent / "data/pc/intel/hpctoolkit-single.sycloffload.icpx.intelgpu-database"


# =============================================================================
# PipelineMetricValue Tests
# =============================================================================


class TestPipelineMetricValue:
    """Tests for PipelineMetricValue dataclass."""

    def test_create_basic(self):
        """Test creating PipelineMetricValue with basic values."""
        pmv = PipelineMetricValue(
            pipeline_id="vec",
            pipeline_name="VEC",
            description="Vector ALU",
        )
        assert pmv.pipeline_id == "vec"
        assert pmv.pipeline_name == "VEC"
        assert pmv.description == "Vector ALU"
        assert pmv.issue_cycles == 0.0
        assert pmv.stall_cycles == 0.0
        assert pmv.idle_cycles == 0.0

    def test_create_with_values(self):
        """Test creating PipelineMetricValue with cycle values."""
        pmv = PipelineMetricValue(
            pipeline_id="vec",
            pipeline_name="VEC",
            description="Vector ALU",
            issue_cycles=50000.0,
            stall_cycles=20000.0,
            idle_cycles=30000.0,
            total_cycles=100000.0,
            issue_pct=50.0,
            stall_pct=20.0,
            idle_pct=30.0,
        )
        assert pmv.issue_cycles == 50000.0
        assert pmv.stall_cycles == 20000.0
        assert pmv.idle_cycles == 30000.0
        assert pmv.issue_pct == 50.0
        assert pmv.stall_pct == 20.0
        assert pmv.idle_pct == 30.0

    def test_active_cycles_property(self):
        """Test active_cycles property."""
        pmv = PipelineMetricValue(
            pipeline_id="vec",
            pipeline_name="VEC",
            description="Vector ALU",
            issue_cycles=50000.0,
            stall_cycles=20000.0,
        )
        assert pmv.active_cycles == 70000.0

    def test_active_pct_property(self):
        """Test active_pct property."""
        pmv = PipelineMetricValue(
            pipeline_id="vec",
            pipeline_name="VEC",
            description="Vector ALU",
            issue_pct=50.0,
            stall_pct=20.0,
        )
        assert pmv.active_pct == 70.0

    def test_utilization_pct_property(self):
        """Test utilization_pct property."""
        pmv = PipelineMetricValue(
            pipeline_id="vec",
            pipeline_name="VEC",
            description="Vector ALU",
            issue_pct=50.0,
        )
        assert pmv.utilization_pct == 50.0


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample PipelineResult for testing."""
        pipelines = {
            "vec": PipelineMetricValue(
                pipeline_id="vec",
                pipeline_name="VEC",
                description="Vector ALU",
                issue_cycles=40000.0,
                stall_cycles=10000.0,
                idle_cycles=50000.0,
                total_cycles=100000.0,
                issue_pct=40.0,
                stall_pct=10.0,
                idle_pct=50.0,
            ),
            "lds": PipelineMetricValue(
                pipeline_id="lds",
                pipeline_name="LDS",
                description="Local Data Share",
                issue_cycles=20000.0,
                stall_cycles=15000.0,
                idle_cycles=65000.0,
                total_cycles=100000.0,
                issue_pct=20.0,
                stall_pct=15.0,
                idle_pct=65.0,
            ),
            "tex": PipelineMetricValue(
                pipeline_id="tex",
                pipeline_name="TEX",
                description="Texture",
                issue_cycles=0.0,
                stall_cycles=0.0,
                idle_cycles=100000.0,
                total_cycles=100000.0,
                issue_pct=0.0,
                stall_pct=0.0,
                idle_pct=100.0,
            ),
        }
        return PipelineResult(
            database_path="/test/path",
            scope="i",
            total_cycles=100000.0,
            pipelines=pipelines,
        )

    def test_get_active_pipelines(self, sample_result):
        """Test get_active_pipelines method."""
        active = sample_result.get_active_pipelines()
        assert len(active) == 2  # vec and lds
        # Should be sorted by issue_pct descending
        assert active[0].pipeline_id == "vec"
        assert active[1].pipeline_id == "lds"

    def test_get_top_pipelines(self, sample_result):
        """Test get_top_pipelines method."""
        top = sample_result.get_top_pipelines(n=1)
        assert len(top) == 1
        assert top[0].pipeline_id == "vec"

    def test_get_most_stalled_pipelines(self, sample_result):
        """Test get_most_stalled_pipelines method."""
        stalled = sample_result.get_most_stalled_pipelines(n=5)
        assert len(stalled) == 2
        # Should be sorted by stall_pct descending
        assert stalled[0].pipeline_id == "lds"  # 15% stall
        assert stalled[1].pipeline_id == "vec"  # 10% stall

    def test_format_summary(self, sample_result):
        """Test format_summary method."""
        summary = sample_result.format_summary()
        assert "GPU Per-Pipeline CPI Stack Analysis" in summary
        assert "Total GPU Cycles: 100,000" in summary
        assert "VEC" in summary
        assert "LDS" in summary
        assert "Issue%" in summary
        assert "Stall%" in summary
        assert "Idle%" in summary

    def test_format_summary_with_show_zero(self, sample_result):
        """Test format_summary with show_zero=True."""
        summary = sample_result.format_summary(show_zero=True)
        assert "TEX" in summary

    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        data = sample_result.to_dict()
        assert "metadata" in data
        assert data["metadata"]["database"] == "/test/path"
        assert "total_cycles" in data
        assert data["total_cycles"] == 100000.0
        assert "pipelines" in data
        assert "vec" in data["pipelines"]
        assert data["pipelines"]["vec"]["issue_pct"] == 40.0
        assert "stacked_bar_data" in data
        assert "available_metrics" in data

    def test_to_dict_json_serializable(self, sample_result):
        """Test to_dict output is JSON serializable."""
        data = sample_result.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_to_csv(self, sample_result):
        """Test to_csv method."""
        csv = sample_result.to_csv()
        lines = csv.split("\n")
        # Header + data lines
        assert len(lines) >= 2
        assert "pipeline_id" in lines[0]
        assert "issue_pct" in lines[0]

    def test_to_csv_without_header(self, sample_result):
        """Test to_csv without header."""
        csv = sample_result.to_csv(include_header=False)
        lines = csv.split("\n")
        assert "pipeline_id" not in lines[0]


# =============================================================================
# PipelineAnalyzer Tests
# =============================================================================


class TestPipelineAnalyzer:
    """Tests for PipelineAnalyzer class."""

    @pytest.fixture
    def nvidia_reader(self):
        """Create a DatabaseReader for NVIDIA test database."""
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        return DatabaseReader(str(NVIDIA_DB_PATH))

    def test_init(self, nvidia_reader):
        """Test PipelineAnalyzer initialization."""
        analyzer = PipelineAnalyzer(nvidia_reader)
        assert analyzer._reader is nvidia_reader

    def test_available_metrics(self, nvidia_reader):
        """Test available_metrics property."""
        analyzer = PipelineAnalyzer(nvidia_reader)
        metrics = analyzer.available_metrics
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_analyze_returns_result(self, nvidia_reader):
        """Test analyze method returns PipelineResult."""
        analyzer = PipelineAnalyzer(nvidia_reader)
        result = analyzer.analyze()
        assert isinstance(result, PipelineResult)

    def test_analyze_has_total_cycles(self, nvidia_reader):
        """Test analyze result has total cycles."""
        analyzer = PipelineAnalyzer(nvidia_reader)
        result = analyzer.analyze()
        assert result.total_cycles >= 0

    def test_analyze_has_pipelines(self, nvidia_reader):
        """Test analyze result has pipelines."""
        analyzer = PipelineAnalyzer(nvidia_reader)
        result = analyzer.analyze()
        assert len(result.pipelines) == len(PIPELINES)

    def test_analyze_stacked_bar_data(self, nvidia_reader):
        """Test analyze result has stacked_bar_data."""
        analyzer = PipelineAnalyzer(nvidia_reader)
        result = analyzer.analyze()
        assert "chart_type" in result.stacked_bar_data
        assert result.stacked_bar_data["chart_type"] == "stacked_bar"
        assert "categories" in result.stacked_bar_data
        assert "series" in result.stacked_bar_data


# =============================================================================
# NVIDIA-specific Tests
# =============================================================================


class TestPipelineAnalyzerNvidia:
    """NVIDIA-specific integration tests."""

    @pytest.fixture
    def result(self):
        """Run analysis on NVIDIA test database."""
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        return pipeline_analysis(str(NVIDIA_DB_PATH))

    def test_total_cycles_positive(self, result):
        """Test total cycles is positive."""
        assert result.total_cycles > 0

    def test_all_pipelines_present(self, result):
        """Test all pipelines are present in result."""
        expected_ids = [p[0] for p in PIPELINES]
        for pipe_id in expected_ids:
            assert pipe_id in result.pipelines

    def test_percentages_valid(self, result):
        """Test percentages are within valid range."""
        for p in result.pipelines.values():
            assert 0 <= p.issue_pct <= 100
            assert 0 <= p.stall_pct <= 100
            assert 0 <= p.idle_pct <= 100

    def test_stacked_bar_structure(self, result):
        """Test stacked bar data has correct structure."""
        sbd = result.stacked_bar_data
        assert "categories" in sbd
        assert "series" in sbd
        assert len(sbd["series"]) == 3  # Issue, Stall, Idle
        assert sbd["series"][0]["name"] == "Issue"
        assert sbd["series"][1]["name"] == "Stall"
        assert sbd["series"][2]["name"] == "Idle"

    def test_format_summary_output(self, result):
        """Test format_summary produces readable output."""
        summary = result.format_summary()
        assert len(summary) > 100
        assert "=" * 70 in summary


# =============================================================================
# AMD-specific Tests
# =============================================================================


class TestPipelineAnalyzerAmd:
    """AMD-specific integration tests."""

    @pytest.fixture
    def result(self):
        """Run analysis on AMD test database."""
        if not AMD_DB_PATH.exists():
            pytest.skip(f"AMD test database not found: {AMD_DB_PATH}")
        return pipeline_analysis(str(AMD_DB_PATH))

    def test_total_cycles_positive(self, result):
        """Test total cycles is positive."""
        assert result.total_cycles > 0

    def test_database_path_recorded(self, result):
        """Test database path is recorded."""
        assert str(AMD_DB_PATH) in result.database_path

    def test_pipelines_have_descriptions(self, result):
        """Test all pipelines have descriptions."""
        for p in result.pipelines.values():
            assert len(p.description) > 0

    def test_csv_export(self, result):
        """Test CSV export works."""
        csv = result.to_csv()
        lines = csv.split("\n")
        assert len(lines) > 1
        # Check header
        assert "pipeline_id" in lines[0]
        assert "issue_pct" in lines[0]


# =============================================================================
# Intel-specific Tests
# =============================================================================


class TestPipelineAnalyzerIntel:
    """Intel-specific integration tests."""

    @pytest.fixture
    def result(self):
        """Run analysis on Intel test database."""
        if not INTEL_DB_PATH.exists():
            pytest.skip(f"Intel test database not found: {INTEL_DB_PATH}")
        return pipeline_analysis(str(INTEL_DB_PATH))

    def test_total_cycles_non_negative(self, result):
        """Test total cycles is non-negative."""
        assert result.total_cycles >= 0

    def test_scope_recorded(self, result):
        """Test scope is recorded."""
        assert result.scope == "i"

    def test_to_dict_complete(self, result):
        """Test to_dict contains all expected keys."""
        data = result.to_dict()
        assert "metadata" in data
        assert "total_cycles" in data
        assert "pipelines" in data
        assert "stacked_bar_data" in data
        assert "available_metrics" in data


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestPipelineConvenienceFunctions:
    """Tests for convenience functions."""

    def test_pipeline_analysis_returns_result(self):
        """Test pipeline_analysis returns PipelineResult."""
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        result = pipeline_analysis(str(NVIDIA_DB_PATH))
        assert isinstance(result, PipelineResult)

    def test_pipeline_analysis_with_exclusive_scope(self):
        """Test pipeline_analysis with exclusive scope."""
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        result = pipeline_analysis(str(NVIDIA_DB_PATH), scope="e")
        assert result.scope == "e"

    def test_print_pipeline_summary_no_error(self, capsys):
        """Test print_pipeline_summary runs without error."""
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        # Should not raise
        print_pipeline_summary(str(NVIDIA_DB_PATH))
        captured = capsys.readouterr()
        assert "GPU Per-Pipeline CPI Stack Analysis" in captured.out


# =============================================================================
# Pipeline Definition Tests
# =============================================================================


class TestPipelineDefinitions:
    """Tests for pipeline definition constants."""

    def test_pipelines_defined(self):
        """Test PIPELINES constant is defined."""
        assert len(PIPELINES) == 11

    def test_pipelines_have_all_fields(self):
        """Test each pipeline has id, name, description."""
        for pipe_id, pipe_name, description in PIPELINES:
            assert len(pipe_id) > 0
            assert len(pipe_name) > 0
            assert len(description) > 0

    def test_expected_pipelines_present(self):
        """Test expected pipelines are present."""
        ids = [p[0] for p in PIPELINES]
        assert "vec" in ids
        assert "lds" in ids
        assert "tex" in ids
        assert "flat" in ids
        assert "xprt" in ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
