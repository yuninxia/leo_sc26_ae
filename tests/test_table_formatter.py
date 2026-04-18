"""Tests for the TableBuilder utility."""

import pytest

from leo.output.table_formatter import TableBuilder, calculate_column_widths


class TestTableBuilder:
    """Test the TableBuilder class."""

    def test_simple_table(self):
        """Test creating a simple table with columns and rows."""
        table = TableBuilder()
        table.set_width(40)
        table.add_columns([
            ("Name", 20, "<"),
            ("Value", 20, ">"),
        ])
        table.add_row("Item 1", "100")
        table.add_row("Item 2", "200")

        result = table.build()
        lines = result.split("\n")

        assert "Name" in lines[0]
        assert "Value" in lines[0]
        assert "Item 1" in lines[1]
        assert "100" in lines[1]
        assert "Item 2" in lines[2]

    def test_with_title(self):
        """Test table with title and borders."""
        table = TableBuilder(title="Test Report", width=50)
        table.add_header()
        table.add_line("Content here")
        table.add_footer()

        result = table.build()
        lines = result.split("\n")

        assert lines[0] == "=" * 50
        assert "Test Report" in lines[1]
        assert lines[2] == "=" * 50
        assert "Content here" in lines[3]
        assert lines[4] == "=" * 50

    def test_section_header(self):
        """Test adding section headers."""
        table = TableBuilder(width=40)
        table.add_section_header("SECTION 1")
        table.add_separator("-")
        table.add_line("  Some content")

        result = table.build()
        lines = result.split("\n")

        assert lines[0] == "SECTION 1"
        assert lines[1] == "-" * 40
        assert "Some content" in lines[2]

    def test_separator(self):
        """Test separator lines with different characters."""
        table = TableBuilder(width=30)
        table.add_separator("-")
        table.add_separator("=")
        table.add_separator("*")

        result = table.build()
        lines = result.split("\n")

        assert lines[0] == "-" * 30
        assert lines[1] == "=" * 30
        assert lines[2] == "*" * 30

    def test_key_value_pairs(self):
        """Test key-value pair formatting."""
        table = TableBuilder(width=50)
        table.add_key_value("Total:", "1,234,567", indent=2, key_width=15)
        table.add_key_value("Average:", "456", indent=2, key_width=15)

        result = table.build()
        lines = result.split("\n")

        assert lines[0].startswith("  Total:")
        assert "1,234,567" in lines[0]
        assert lines[1].startswith("  Average:")
        assert "456" in lines[1]

    def test_blank_line(self):
        """Test adding blank lines."""
        table = TableBuilder(width=20)
        table.add_line("Line 1")
        table.add_blank_line()
        table.add_line("Line 2")

        result = table.build()
        lines = result.split("\n")

        assert lines[0] == "Line 1"
        assert lines[1] == ""
        assert lines[2] == "Line 2"

    def test_column_alignment(self):
        """Test column alignment (left, right, center)."""
        table = TableBuilder()
        table.add_columns([
            ("Left", 10, "<"),
            ("Center", 10, "^"),
            ("Right", 10, ">"),
        ])
        table.add_row("L", "C", "R")

        result = table.build()
        lines = result.split("\n")

        # Check header alignment
        assert lines[0][:10].strip() == "Left"
        assert lines[0][10:20].strip() == "Center"
        assert lines[0][20:30].strip() == "Right"

        # Check data alignment
        assert lines[1][0] == "L"  # Left-aligned starts at position 0
        assert lines[1][29] == "R"  # Right-aligned ends at last position

    def test_row_without_columns_raises(self):
        """Test that add_row raises error if columns not defined."""
        table = TableBuilder(width=40)

        with pytest.raises(ValueError, match="Must call add_columns"):
            table.add_row("value1", "value2")

    def test_row_value_count_mismatch_raises(self):
        """Test that add_row raises error if value count mismatches."""
        table = TableBuilder()
        table.add_columns([
            ("Col1", 10, "<"),
            ("Col2", 10, "<"),
        ])

        with pytest.raises(ValueError, match="Expected 2 values, got 3"):
            table.add_row("a", "b", "c")

    def test_add_row_raw(self):
        """Test adding a raw formatted row."""
        table = TableBuilder(width=40)
        table.add_row_raw("{:<10} <-- {:<10} {:>10}", "Stall", "Cause", "100")

        result = table.build()
        assert "Stall" in result
        assert "<--" in result
        assert "Cause" in result
        assert "100" in result

    def test_auto_width_from_columns(self):
        """Test that width is auto-calculated from column definitions."""
        table = TableBuilder()
        table.add_columns([
            ("Name", 20, "<"),
            ("Value", 15, ">"),
        ])
        table.add_separator("-")

        result = table.build()
        lines = result.split("\n")

        # Separator should be total column width
        assert lines[1] == "-" * 35

    def test_clear(self):
        """Test clearing the builder."""
        table = TableBuilder(title="Test", width=40)
        table.add_header()
        table.add_line("Content")

        table.clear()
        result = table.build()

        assert result == ""

    def test_complex_table(self):
        """Test building a complex multi-section table."""
        table = TableBuilder(title="Leo Analysis Results", width=80)
        table.add_header()
        table.add_blank_line()

        # Metadata section
        table.add_key_value("Kernel:", "my_kernel", key_width=15)
        table.add_key_value("Architecture:", "NVIDIA A100", key_width=15)
        table.add_blank_line()

        # Section header
        table.add_section_header("STALL ANALYSIS")
        table.add_separator("-")

        # Column headers
        table.add_columns([
            ("Source", 30, "<"),
            ("Opcode", 15, "<"),
            ("Cycles", 20, ">"),
            ("Percent", 15, ">"),
        ])
        table.add_separator("-")

        # Data rows
        table.add_row("main.cu:42", "LDG.E.64", "1,234,567", "45.2%")
        table.add_row("main.cu:45", "STS", "567,890", "20.8%")

        table.add_separator("-")
        table.add_footer()

        result = table.build()

        # Verify structure
        assert "Leo Analysis Results" in result
        assert "STALL ANALYSIS" in result
        assert "main.cu:42" in result
        assert "LDG.E.64" in result
        assert "1,234,567" in result


class TestCalculateColumnWidths:
    """Test the calculate_column_widths helper function."""

    def test_basic_width_calculation(self):
        """Test basic width calculation from data."""
        headers = ["Name", "Value"]
        data = [
            ("Short", "1"),
            ("Much longer name", "12345"),
        ]

        widths = calculate_column_widths(data, headers, padding=1)

        # "Much longer name" is 16 chars, + 1 padding = 17
        assert widths[0] == 17
        # "12345" is 5 chars, "Value" is 5 chars, + 1 padding = 6
        assert widths[1] == 6

    def test_min_widths(self):
        """Test that minimum widths are respected."""
        headers = ["A", "B"]
        data = [("x", "y")]

        widths = calculate_column_widths(
            data, headers, min_widths=[10, 15], padding=0
        )

        assert widths[0] == 10
        assert widths[1] == 15

    def test_header_wider_than_data(self):
        """Test when header is wider than any data."""
        headers = ["Very Long Header", "Short"]
        data = [("x", "y")]

        widths = calculate_column_widths(data, headers, padding=1)

        assert widths[0] == len("Very Long Header") + 1
        assert widths[1] == len("Short") + 1

    def test_empty_data(self):
        """Test with no data rows."""
        headers = ["Col1", "Col2"]
        data = []

        widths = calculate_column_widths(data, headers, padding=1)

        assert widths[0] == len("Col1") + 1
        assert widths[1] == len("Col2") + 1

    def test_custom_padding(self):
        """Test custom padding value."""
        headers = ["A", "B"]
        data = [("x", "y")]

        widths = calculate_column_widths(data, headers, padding=5)

        assert widths[0] == 1 + 5  # 'A' is 1 char, + 5 padding
        assert widths[1] == 1 + 5


class TestTableBuilderIntegration:
    """Integration tests for TableBuilder with realistic use cases."""

    def test_analyzer_summary_style(self):
        """Test TableBuilder can produce analyzer.py summary style output."""
        table = TableBuilder(title="Leo GPU Performance Analysis")

        # Calculate width based on columns (similar to analyzer.py)
        W_STALL_LOC = 25
        W_STALL_OP = 15
        W_ARROW = 5
        W_CAUSE_LOC = 25
        W_CAUSE_OP = 15
        W_CYCLES = 18
        W_PCT = 8
        W_SPD = 8
        LINE_WIDTH = (
            W_STALL_LOC + W_STALL_OP + W_ARROW +
            W_CAUSE_LOC + W_CAUSE_OP + W_CYCLES + W_PCT + W_SPD
        )
        table.set_width(LINE_WIDTH)

        table.add_header()
        table.add_key_value("Kernel:", "test_kernel", key_width=18)
        table.add_key_value("Total Stall Cycles:", "1,234,567", key_width=18)
        table.add_header()  # Double border after metadata
        table.add_blank_line()

        table.add_section_header("STALL ANALYSIS (PC Sampling -> Back-slicing -> Root Cause)")
        table.add_separator("-")

        # Manual header row for complex format
        header = (
            f"{'Stall Location':<{W_STALL_LOC}}"
            f"{'Stall Opcode':<{W_STALL_OP}}"
            f"{'':^{W_ARROW}}"
            f"{'Root Cause Location':<{W_CAUSE_LOC}}"
            f"{'Root Opcode':<{W_CAUSE_OP}}"
            f"{'Cycles':>{W_CYCLES}}"
            f"{'% Total':>{W_PCT}}"
            f"{'Speedup':>{W_SPD}}"
        )
        table.add_line(header)
        table.add_separator("-")

        # Data row with arrow
        table.add_row_raw(
            "{:<" + str(W_STALL_LOC) + "}"
            "{:<" + str(W_STALL_OP) + "}"
            "<-- "
            "{:<" + str(W_CAUSE_LOC) + "}"
            "{:<" + str(W_CAUSE_OP) + "}"
            "{:>" + str(W_CYCLES) + "}"
            "{:>" + str(W_PCT - 1) + "}%"
            "{:>" + str(W_SPD - 1) + "}x",
            "main.cu:42", "LDG.E.64", "main.cu:40", "FADD",
            "1,234,567", "45.2", "1.05"
        )

        table.add_separator("-")
        table.add_footer()

        result = table.build()

        # Verify key elements
        assert "Leo GPU Performance Analysis" in result
        assert "STALL ANALYSIS" in result
        assert "main.cu:42" in result
        assert "LDG.E.64" in result
        assert "<--" in result
        assert "1,234,567" in result

    def test_program_analysis_style(self):
        """Test TableBuilder can produce program_analysis.py summary style output."""
        table = TableBuilder(title="Leo Whole-Program GPU Performance Analysis", width=100)

        table.add_header()
        table.add_blank_line()

        table.add_key_value("Database:", "/path/to/database")
        table.add_key_value("Measurements:", "/path/to/measurements")
        table.add_blank_line()

        table.add_section_header("PROGRAM TOTALS")
        table.add_separator("-")
        table.add_key_value("Total Execution Time:", "1.2345s", indent=2, key_width=20)
        table.add_key_value("Total Stall Cycles:", "1,234,567,890", indent=2, key_width=20)
        table.add_key_value("Kernels Analyzed:", "5", indent=2, key_width=20)
        table.add_blank_line()

        table.add_section_header("TOP 5 KERNELS BY STALL CYCLES")
        table.add_separator("-")

        # Kernel table
        table.add_columns([
            ("#", 3, "<"),
            ("Kernel", 40, "<"),
            ("Time (s)", 12, ">"),
            ("Stall Cycles", 18, ">"),
            ("Stall %", 10, ">"),
        ])
        table.add_separator("-")

        table.add_row("1", "MyKernel1", "0.1234", "123,456,789", "45.6%")
        table.add_row("2", "MyKernel2", "0.0567", "56,789,012", "21.3%")

        table.add_separator("-")
        table.add_footer()

        result = table.build()

        # Verify structure
        assert "Leo Whole-Program GPU Performance Analysis" in result
        assert "PROGRAM TOTALS" in result
        assert "TOP 5 KERNELS BY STALL CYCLES" in result
        assert "MyKernel1" in result
        assert "123,456,789" in result
