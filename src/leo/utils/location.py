"""Source location formatting helpers."""

from pathlib import Path


def extract_filename(path: str) -> str:
    """Return the filename component of a path."""
    if not path:
        return ""
    return Path(path).name


def format_source_location(
    file: str,
    line: int,
    col: int = 0,
    short: bool = False,
) -> str:
    """Format a source location as file:line[:column].

    Args:
        file: Full path or filename to format.
        line: 1-indexed line number.
        col: 1-indexed column number; 0 means unknown.
        short: If True, use only the filename.
    """
    path = extract_filename(file) if short else file
    if col > 0:
        return f"{path}:{line}:{col}"
    return f"{path}:{line}"
