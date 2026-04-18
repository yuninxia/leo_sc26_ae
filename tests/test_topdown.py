"""Tests for leo.analysis.topdown module - GPU top-down cycle breakdown analysis."""

import json
import pytest
from pathlib import Path

from leo.db import DatabaseReader
from leo.analysis.topdown import (
    MetricValue,
    TopDownResult,
    TopDownAnalyzer,
    topdown_analysis,
    print_topdown_summary,
    LEVEL1_METRICS,
    LEVEL2_ISSUE_METRICS,
    LEVEL2_STALL_METRICS,
)


# Test database paths
NVIDIA_DB_PATH = Path(__file__).parent / "data/pc/nvidia/hpctoolkit-single.cudaoffload.gcc.cudagpu-database"
AMD_DB_PATH = Path(__file__).parent / "data/pc/amd/hpctoolkit-single.hipoffload.amdclang.rocmgpu-database"
INTEL_DB_PATH = Path(__file__).parent / "data/pc/intel/hpctoolkit-single.sycloffload.icpx.intelgpu-database"


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_create_basic(self):
        mv = MetricValue(metric_name="gcycles:stl:mem", label="MEM", value=50000.0)
        assert mv.metric_name == "gcycles:stl:mem"
        assert mv.value == 50000.0
        assert mv.percentage_of_total == 0.0

    def test_create_with_percentages(self):
        mv = MetricValue(
            metric_name="gcycles:stl",
            label="Stall",
            value=60000.0,
            percentage_of_total=60.0,
            percentage_of_parent=60.0,
        )
        assert mv.percentage_of_total == 60.0
        assert mv.percentage_of_parent == 60.0


class TestTopDownResult:
    """Tests for TopDownResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        level2_issue = {
            "gcycles:isu:vec": MetricValue(
                metric_name="gcycles:isu:vec",
                label="VEC",
                value=30000.0,
                percentage_of_total=30.0,
                percentage_of_parent=75.0,
            ),
        }
        level2_exposed = {
            "gcycles:stl:mem": MetricValue(
                metric_name="gcycles:stl:mem",
                label="MEM",
                value=35000.0,
                percentage_of_total=35.0,
                percentage_of_parent=87.5,
            ),
            "gcycles:stl:idep": MetricValue(
                metric_name="gcycles:stl:idep",
                label="IDEP",
                value=5000.0,
                percentage_of_total=5.0,
                percentage_of_parent=12.5,
            ),
        }
        return TopDownResult(
            database_path="/test/path",
            scope="i",
            total_cycles=100000.0,
            issue_cycles=40000.0,
            hidden_cycles=20000.0,
            exposed_cycles=40000.0,
            level2_issue=level2_issue,
            level2_exposed=level2_exposed,
        )

    def test_issue_percentage(self, sample_result):
        assert sample_result.issue_percentage == 40.0

    def test_hidden_percentage(self, sample_result):
        assert sample_result.hidden_percentage == 20.0

    def test_exposed_percentage(self, sample_result):
        assert sample_result.exposed_percentage == 40.0

    def test_get_top_stall_reasons(self, sample_result):
        top_stalls = sample_result.get_top_stall_reasons(n=5)
        assert len(top_stalls) == 2
        assert top_stalls[0].metric_name == "gcycles:stl:mem"
        assert top_stalls[1].metric_name == "gcycles:stl:idep"

    def test_get_top_issue_types(self, sample_result):
        top_issues = sample_result.get_top_issue_types(n=5)
        assert len(top_issues) == 1
        assert top_issues[0].metric_name == "gcycles:isu:vec"

    def test_format_summary(self, sample_result):
        summary = sample_result.format_summary()
        assert "GPU Top-Down Cycle Breakdown" in summary
        assert "100,000" in summary

    def test_to_dict(self, sample_result):
        data = sample_result.to_dict()
        assert "total_cycles" in data
        assert data["total_cycles"] == 100000.0
        assert "level1" in data
        assert "level2" in data
        assert "sunburst_data" in data

    def test_to_dict_json_serializable(self, sample_result):
        data = sample_result.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 0


class TestTopDownAnalyzer:
    """Tests for TopDownAnalyzer class."""

    @pytest.fixture
    def nvidia_reader(self):
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        return DatabaseReader(str(NVIDIA_DB_PATH))

    def test_init(self, nvidia_reader):
        analyzer = TopDownAnalyzer(nvidia_reader)
        assert analyzer._reader is nvidia_reader

    def test_analyze_returns_result(self, nvidia_reader):
        analyzer = TopDownAnalyzer(nvidia_reader)
        result = analyzer.analyze()
        assert isinstance(result, TopDownResult)

    def test_analyze_has_total_cycles(self, nvidia_reader):
        analyzer = TopDownAnalyzer(nvidia_reader)
        result = analyzer.analyze()
        assert result.total_cycles >= 0


class TestTopDownAnalyzerNvidia:
    """NVIDIA-specific integration tests."""

    @pytest.fixture
    def result(self):
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        return topdown_analysis(str(NVIDIA_DB_PATH))

    def test_total_cycles_positive(self, result):
        assert result.total_cycles > 0

    def test_has_stall_breakdown(self, result):
        if result.exposed_cycles > 0:
            assert len(result.level2_exposed) > 0

    def test_percentages_valid(self, result):
        assert 0 <= result.issue_percentage <= 100
        assert 0 <= result.hidden_percentage <= 100
        assert 0 <= result.exposed_percentage <= 100

    def test_sunburst_data_structure(self, result):
        sunburst = result.sunburst_data
        assert "name" in sunburst
        assert "value" in sunburst
        assert sunburst["name"] == "GPU Cycles"

    def test_format_summary_output(self, result):
        summary = result.format_summary()
        assert len(summary) > 100


class TestTopDownAnalyzerAmd:
    """AMD-specific integration tests."""

    @pytest.fixture
    def result(self):
        if not AMD_DB_PATH.exists():
            pytest.skip(f"AMD test database not found: {AMD_DB_PATH}")
        return topdown_analysis(str(AMD_DB_PATH))

    def test_total_cycles_positive(self, result):
        assert result.total_cycles > 0

    def test_has_breakdown(self, result):
        has_breakdown = len(result.level2_issue) > 0 or len(result.level2_exposed) > 0
        if result.total_cycles > 0:
            assert has_breakdown or (result.issue_cycles == 0 and result.exposed_cycles == 0)

    def test_database_path_recorded(self, result):
        assert str(AMD_DB_PATH) in result.database_path


class TestTopDownAnalyzerIntel:
    """Intel-specific integration tests."""

    @pytest.fixture
    def result(self):
        if not INTEL_DB_PATH.exists():
            pytest.skip(f"Intel test database not found: {INTEL_DB_PATH}")
        return topdown_analysis(str(INTEL_DB_PATH))

    def test_total_cycles_non_negative(self, result):
        assert result.total_cycles >= 0

    def test_scope_recorded(self, result):
        assert result.scope == "i"

    def test_to_dict_complete(self, result):
        data = result.to_dict()
        assert "total_cycles" in data
        assert "level1" in data
        assert "level2" in data
        assert "sunburst_data" in data


class TestTopdownConvenienceFunctions:
    """Tests for convenience functions."""

    def test_topdown_analysis_returns_result(self):
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        result = topdown_analysis(str(NVIDIA_DB_PATH))
        assert isinstance(result, TopDownResult)

    def test_topdown_analysis_with_exclusive_scope(self):
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        result = topdown_analysis(str(NVIDIA_DB_PATH), scope="e")
        assert result.scope == "e"

    def test_print_topdown_summary_no_error(self, capsys):
        if not NVIDIA_DB_PATH.exists():
            pytest.skip(f"NVIDIA test database not found: {NVIDIA_DB_PATH}")
        print_topdown_summary(str(NVIDIA_DB_PATH))
        captured = capsys.readouterr()
        assert "GPU Top-Down Cycle Breakdown" in captured.out


class TestMetricDefinitions:
    """Tests for metric definition constants."""

    def test_level1_metrics_defined(self):
        assert "gcycles:isu" in LEVEL1_METRICS
        assert "gcycles:stl" in LEVEL1_METRICS

    def test_level2_issue_metrics_defined(self):
        assert len(LEVEL2_ISSUE_METRICS) > 0
        assert "gcycles:isu:vec" in LEVEL2_ISSUE_METRICS

    def test_level2_stall_metrics_defined(self):
        assert len(LEVEL2_STALL_METRICS) > 0
        assert "gcycles:stl:mem" in LEVEL2_STALL_METRICS
        assert "gcycles:stl:idep" in LEVEL2_STALL_METRICS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
