"""Reusable table formatting utility for consistent output styling.

This module provides a TableBuilder class for creating formatted text tables
with headers, separators, columns, and sections. It supports dynamic column
width calculation, multiple alignment options, and flexible section organization.

Example:
    from leo.output.table_formatter import TableBuilder

    table = TableBuilder(title="Analysis Results")
    table.add_section_header("TOP ITEMS")
    table.add_columns([
        ("Name", 20, "<"),
        ("Value", 15, ">"),
        ("Percent", 10, ">"),
    ])
    table.add_row("Item 1", "1,234", "45.6%")
    table.add_row("Item 2", "567", "21.3%")
    print(table.build())
"""

from typing import List, Optional, Tuple, Union


class TableBuilder:
    """Reusable table formatting utility for consistent output styling.

    Supports building tables with:
    - Centered titles with border characters
    - Section headers
    - Custom separator lines
    - Column definitions with alignment
    - Data rows matching column definitions
    - Custom lines for flexibility

    The table can use a fixed width or auto-calculate width from columns.
    """

    def __init__(
        self,
        title: Optional[str] = None,
        width: Optional[int] = None,
        border_char: str = "=",
    ):
        """Initialize table builder.

        Args:
            title: Optional centered title for the table (displayed at top).
            width: Fixed width, or None to auto-calculate from columns.
            border_char: Character to use for top/bottom borders. Default "=".
        """
        self._title = title
        self._width = width
        self._border_char = border_char
        self._lines: List[str] = []
        self._columns: Optional[List[Tuple[str, int, str]]] = None
        self._column_widths: List[int] = []
        self._total_width: Optional[int] = None

    def _get_width(self) -> int:
        """Get effective table width."""
        if self._width is not None:
            return self._width
        if self._total_width is not None:
            return self._total_width
        return 80  # Default fallback

    def add_header(self, border_char: Optional[str] = None) -> None:
        """Add header border and centered title (if set).

        Args:
            border_char: Override border character (default uses instance setting).
        """
        char = border_char or self._border_char
        width = self._get_width()
        self._lines.append(char * width)
        if self._title:
            self._lines.append(self._title.center(width))
            self._lines.append(char * width)

    def add_footer(self, border_char: Optional[str] = None) -> None:
        """Add footer border line.

        Args:
            border_char: Override border character (default uses instance setting).
        """
        char = border_char or self._border_char
        self._lines.append(char * self._get_width())

    def add_section_header(self, text: str) -> None:
        """Add a section header (e.g., 'PROGRAM TOTALS').

        Args:
            text: Section header text (added as-is).
        """
        self._lines.append(text)

    def add_separator(self, char: str = "-") -> None:
        """Add separator line.

        Args:
            char: Character to use for the separator line.
        """
        self._lines.append(char * self._get_width())

    def add_columns(self, columns: List[Tuple[str, int, str]]) -> None:
        """Define and add column headers.

        Args:
            columns: List of (header_text, width, alignment) tuples.
                     alignment: '<' (left), '>' (right), '^' (center).
        """
        self._columns = columns
        self._column_widths = [col[1] for col in columns]
        self._total_width = sum(self._column_widths)

        # Build header line
        parts = []
        for header, width, align in columns:
            parts.append(f"{header:{align}{width}}")
        self._lines.append("".join(parts))

    def add_row(self, *values: Union[str, int, float]) -> None:
        """Add a data row using the defined column widths.

        Args:
            *values: Values for each column. Values are converted to strings
                     and formatted according to column alignment and width.

        Raises:
            ValueError: If add_columns() has not been called first.
            ValueError: If number of values doesn't match number of columns.
        """
        if self._columns is None:
            raise ValueError("Must call add_columns() before add_row()")
        if len(values) != len(self._columns):
            raise ValueError(
                f"Expected {len(self._columns)} values, got {len(values)}"
            )

        parts = []
        for (_, width, align), value in zip(self._columns, values):
            str_value = str(value)
            parts.append(f"{str_value:{align}{width}}")
        self._lines.append("".join(parts))

    def add_row_raw(self, format_str: str, *values) -> None:
        """Add a row using a custom format string.

        This is useful for complex rows that don't fit the simple column model,
        such as rows with arrows or special formatting between columns.

        Args:
            format_str: Python format string with placeholders.
            *values: Values to format into the string.
        """
        self._lines.append(format_str.format(*values))

    def add_line(self, text: str) -> None:
        """Add a custom line of text.

        Args:
            text: Text line to add (added as-is).
        """
        self._lines.append(text)

    def add_blank_line(self) -> None:
        """Add a blank line."""
        self._lines.append("")

    def add_key_value(
        self,
        key: str,
        value: Union[str, int, float],
        indent: int = 0,
        key_width: int = 0,
    ) -> None:
        """Add a key-value pair line (e.g., "  Total Stall Cycles: 1,234,567").

        Args:
            key: Key name (with trailing colon).
            value: Value to display.
            indent: Number of spaces to indent.
            key_width: Fixed key width for alignment (0 = no padding).
        """
        prefix = " " * indent
        if key_width > 0:
            self._lines.append(f"{prefix}{key:<{key_width}} {value}")
        else:
            self._lines.append(f"{prefix}{key} {value}")

    def set_width(self, width: int) -> None:
        """Explicitly set the table width.

        Args:
            width: Table width in characters.
        """
        self._width = width
        self._total_width = width

    def build(self) -> str:
        """Return the formatted table as a string.

        Returns:
            Complete formatted table as a single string with newline separators.
        """
        return "\n".join(self._lines)

    def clear(self) -> None:
        """Clear all content and reset the builder."""
        self._lines = []
        self._columns = None
        self._column_widths = []
        self._total_width = None


def calculate_column_widths(
    data: List[Tuple[str, ...]],
    headers: List[str],
    min_widths: Optional[List[int]] = None,
    padding: int = 1,
) -> List[int]:
    """Calculate optimal column widths based on data and headers.

    Args:
        data: List of row tuples with string values.
        headers: List of column header strings.
        min_widths: Optional minimum widths for each column.
        padding: Extra padding to add to each column.

    Returns:
        List of column widths.
    """
    n_cols = len(headers)
    min_widths = min_widths or [0] * n_cols

    # Start with header widths
    widths = [len(h) for h in headers]

    # Expand to fit data
    for row in data:
        for i, value in enumerate(row):
            if i < n_cols:
                widths[i] = max(widths[i], len(str(value)))

    # Apply minimums and padding
    for i in range(n_cols):
        widths[i] = max(widths[i], min_widths[i]) + padding

    return widths
