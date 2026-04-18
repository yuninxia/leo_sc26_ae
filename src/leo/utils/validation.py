"""Common validation helpers."""

from pathlib import Path


def require_file_exists(path: Path, description: str) -> Path:
    """Ensure a file exists and return the normalized Path.

    Args:
        path: File path to validate.
        description: Human-readable description for error messages.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path
