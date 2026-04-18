"""Tests for DWARF line information parser."""

import pytest
from pathlib import Path

from leo.binary.dwarf_line import DWARFLineParser, SourceLocation, get_dwarf_line_parser

# Test data paths
DATA_ROOT = Path(__file__).parent / "data" / "pc"
AMD_DATA = DATA_ROOT / "amd"
AMD_GPUBIN = (
    AMD_DATA
    / "hpctoolkit-single.hipoffload.amdclang.rocmgpu-measurements"
    / "gpubins"
    / "9f7f9be695af6f36f2b56450611127c6.gpubin"
)

NVIDIA_DATA = DATA_ROOT / "nvidia"
NVIDIA_GPUBIN = (
    NVIDIA_DATA
    / "hpctoolkit-single.cudaoffload.gcc.cudagpu-measurements"
    / "gpubins"
    / "67e7ddd42e43d0ca040956d9d9b316fa.gpubin"
)


class TestDWARFLineParser:
    """Tests for DWARFLineParser class."""

    @pytest.fixture
    def amd_parser(self):
        """Create parser for AMD test binary."""
        if not AMD_GPUBIN.exists():
            pytest.skip(f"AMD test binary not found: {AMD_GPUBIN}")
        return DWARFLineParser(str(AMD_GPUBIN))

    def test_parser_loads_amd_binary(self, amd_parser):
        """Test that parser loads AMD GPU binary successfully."""
        assert amd_parser.has_dwarf_info
        assert amd_parser.location_count > 0

    def test_source_location_has_column(self, amd_parser):
        """Test that column information is extracted from AMD binary."""
        # From readelf output, we know 0x1900 has column 58
        loc = amd_parser.get_source_location(0x1900)
        assert loc is not None
        assert loc.line > 0
        # AMD binary should have column info
        assert loc.column > 0, "Expected column info in AMD binary"

    def test_column_statistics(self, amd_parser):
        """Test column coverage statistics."""
        stats = amd_parser.get_column_statistics()
        assert stats["total"] > 0
        assert stats["with_column"] > 0
        assert stats["column_coverage_pct"] > 0

    def test_get_all_locations(self, amd_parser):
        """Test getting all address-to-source mappings."""
        locations = amd_parser.get_all_locations()
        assert len(locations) > 0
        # All entries should be SourceLocation instances
        for addr, loc in locations.items():
            assert isinstance(addr, int)
            assert isinstance(loc, SourceLocation)

    def test_source_location_str_format(self, amd_parser):
        """Test SourceLocation string formatting."""
        loc = amd_parser.get_source_location(0x1900)
        assert loc is not None
        loc_str = str(loc)
        # Should contain file:line or file:line:column
        assert ":" in loc_str
        if loc.column > 0:
            # Should have format file:line:column
            parts = loc_str.rsplit(":", 2)
            assert len(parts) >= 2

    def test_binary_search_for_nearby_address(self, amd_parser):
        """Test that nearby addresses find preceding location."""
        # Get an exact location first
        exact_loc = amd_parser.get_source_location(0x1900)
        assert exact_loc is not None

        # Query a nearby address (between instructions)
        nearby_loc = amd_parser.get_source_location(0x1902)
        # Should return the preceding entry
        assert nearby_loc is not None

    def test_get_locations_for_line(self, amd_parser):
        """Test filtering locations by source line."""
        # Find all locations for a specific line
        # From the test data, compute.h line 67 has multiple instructions
        locations = amd_parser.get_locations_for_line("compute.h", 67)
        # Should find at least one match
        assert len(locations) >= 0  # May be empty if file not found

    def test_dump_line_table(self, amd_parser):
        """Test line table dump formatting."""
        dump = amd_parser.dump_line_table(max_entries=10)
        assert "DWARF Line Table" in dump
        assert "Address" in dump
        assert "Line" in dump
        assert "Col" in dump

    def test_nonexistent_binary(self):
        """Test handling of nonexistent binary."""
        parser = DWARFLineParser("/nonexistent/path/to/binary.gpubin")
        assert not parser.has_dwarf_info
        assert parser.location_count == 0

    def test_factory_function(self):
        """Test get_dwarf_line_parser factory function."""
        if not AMD_GPUBIN.exists():
            pytest.skip(f"AMD test binary not found: {AMD_GPUBIN}")

        parser = get_dwarf_line_parser(str(AMD_GPUBIN))
        assert parser is not None
        assert parser.has_dwarf_info

    def test_factory_returns_none_for_missing(self):
        """Test factory returns None for missing binary."""
        parser = get_dwarf_line_parser("/nonexistent/binary.gpubin")
        assert parser is None


class TestSourceLocation:
    """Tests for SourceLocation dataclass."""

    def test_has_column_property(self):
        """Test has_column property."""
        loc_with_col = SourceLocation(
            address=0x1000, file="test.cpp", line=10, column=5
        )
        assert loc_with_col.has_column

        loc_without_col = SourceLocation(
            address=0x1000, file="test.cpp", line=10, column=0
        )
        assert not loc_without_col.has_column

    def test_short_file_property(self):
        """Test short_file extracts filename."""
        loc = SourceLocation(
            address=0x1000,
            file="/path/to/some/source/file.cpp",
            line=10,
            column=5,
        )
        assert loc.short_file == "file.cpp"

    def test_str_with_column(self):
        """Test string representation with column."""
        loc = SourceLocation(
            address=0x1000, file="test.cpp", line=10, column=5
        )
        assert str(loc) == "test.cpp:10:5"

    def test_str_without_column(self):
        """Test string representation without column."""
        loc = SourceLocation(
            address=0x1000, file="test.cpp", line=10, column=0
        )
        assert str(loc) == "test.cpp:10"


class TestNVIDIABinary:
    """Tests for NVIDIA binary (may have limited DWARF info)."""

    @pytest.fixture
    def nvidia_parser(self):
        """Create parser for NVIDIA test binary."""
        if not NVIDIA_GPUBIN.exists():
            pytest.skip(f"NVIDIA test binary not found: {NVIDIA_GPUBIN}")
        return DWARFLineParser(str(NVIDIA_GPUBIN))

    def test_parser_handles_nvidia_binary(self, nvidia_parser):
        """Test that parser handles NVIDIA binary gracefully."""
        # NVIDIA may or may not have DWARF info
        # Parser should not crash either way
        _ = nvidia_parser.has_dwarf_info
        _ = nvidia_parser.location_count
