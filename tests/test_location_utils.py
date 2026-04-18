"""Tests for shared source location utilities."""

from leo.utils.location import extract_filename, format_source_location


class TestExtractFilename:
    """Test filename extraction helper."""

    def test_extract_filename_with_path(self):
        assert extract_filename("/tmp/kernel.cu") == "kernel.cu"

    def test_extract_filename_already_name(self):
        assert extract_filename("kernel.cu") == "kernel.cu"

    def test_extract_filename_empty(self):
        assert extract_filename("") == ""


class TestFormatSourceLocation:
    """Test source location formatting."""

    def test_format_full_path_without_column(self):
        loc = format_source_location("/tmp/kernel.cu", 42, col=0, short=False)
        assert loc == "/tmp/kernel.cu:42"

    def test_format_full_path_with_column(self):
        loc = format_source_location("/tmp/kernel.cu", 42, col=7, short=False)
        assert loc == "/tmp/kernel.cu:42:7"

    def test_format_short_path_without_column(self):
        loc = format_source_location("/tmp/kernel.cu", 42, col=0, short=True)
        assert loc == "kernel.cu:42"

    def test_format_short_path_with_column(self):
        loc = format_source_location("/tmp/kernel.cu", 42, col=7, short=True)
        assert loc == "kernel.cu:42:7"
