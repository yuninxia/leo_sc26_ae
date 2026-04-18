"""Auto-discovery utilities for HPCToolkit databases and GPU binaries.

This module provides functions to automatically locate related files in
HPCToolkit's directory structure:
- Database directories from measurements directories
- GPU binaries (gpubins) from measurements directories
- hpcstruct files for source mapping
"""

from pathlib import Path
from typing import List, Optional


def find_database(measurements_dir: Path) -> Path:
    """Find the HPCToolkit database directory for a measurements directory.

    HPCToolkit creates paired directories:
    - hpctoolkit-<name>-measurements/  (profile data)
    - hpctoolkit-<name>-database/      (processed database)

    Args:
        measurements_dir: Path to the measurements directory.

    Returns:
        Path to the corresponding database directory.

    Raises:
        FileNotFoundError: If no database directory can be found.
    """
    measurements_dir = Path(measurements_dir)
    base_name = measurements_dir.name.replace("-measurements", "")

    # Try common naming patterns
    candidates = [
        measurements_dir.parent / f"{base_name}-database",
        measurements_dir.parent / f"hpctoolkit-{base_name.replace('hpctoolkit-', '')}-database",
        measurements_dir.with_name(measurements_dir.name.replace("measurements", "database")),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find database for {measurements_dir}")


def find_all_gpubins(measurements_dir: Path) -> List[Path]:
    """Find all GPU binaries in a measurements directory.

    HPCToolkit stores GPU binaries in:
    - gpubins/       (all compiled binaries — preferred, always real files)
    - gpubins-used/  (actually profiled binaries — may contain symlinks
                      that break in containers when the original profiling
                      host paths aren't mounted)

    Args:
        measurements_dir: Path to the measurements directory.

    Returns:
        List of paths to GPU binary files, sorted alphabetically.

    Raises:
        FileNotFoundError: If no GPU binaries are found.
    """
    measurements_dir = Path(measurements_dir)

    # Prefer gpubins/ (always real files) over gpubins-used/ (may have symlinks)
    for subdir in ["gpubins", "gpubins-used"]:
        gpubins_dir = measurements_dir / subdir
        if gpubins_dir.exists():
            gpubins = sorted(g for g in gpubins_dir.glob("*.gpubin") if g.exists())
            if gpubins:
                return gpubins

    raise FileNotFoundError(f"No GPU binaries found in {measurements_dir}")


def find_hpcstruct(measurements_dir: Path, gpubin: Path) -> Optional[Path]:
    """Find the hpcstruct file for a GPU binary.

    hpcstruct files provide source line mapping information and are stored in:
    - measurements_dir/structs/<gpubin_name>-gpucfg-yes.hpcstruct
    - measurements_dir/structs/<gpubin_name>.hpcstruct

    Args:
        measurements_dir: Path to the measurements directory.
        gpubin: Path to the GPU binary file.

    Returns:
        Path to the hpcstruct file, or None if not found.
    """
    measurements_dir = Path(measurements_dir)
    structs_dir = measurements_dir / "structs"

    if not structs_dir.exists():
        return None

    gpubin_name = gpubin.name

    # Try common naming patterns
    for pattern in [f"{gpubin_name}-gpucfg-yes.hpcstruct", f"{gpubin_name}.hpcstruct"]:
        candidate = structs_dir / pattern
        if candidate.exists():
            return candidate

    return None


def discover_analysis_inputs(
    measurements_dir: Path,
) -> dict:
    """Discover all inputs needed for Leo analysis from a measurements directory.

    This is a convenience function that finds the database, GPU binaries,
    and hpcstruct files in one call.

    Args:
        measurements_dir: Path to the HPCToolkit measurements directory.

    Returns:
        Dict with:
        - 'database': Path to database directory
        - 'gpubins': List of (gpubin_path, hpcstruct_path or None) tuples

    Raises:
        FileNotFoundError: If database or GPU binaries cannot be found.
    """
    measurements_dir = Path(measurements_dir)

    database = find_database(measurements_dir)
    gpubins = find_all_gpubins(measurements_dir)

    gpubin_info = []
    for gpubin in gpubins:
        hpcstruct = find_hpcstruct(measurements_dir, gpubin)
        gpubin_info.append((gpubin, hpcstruct))

    return {
        "database": database,
        "gpubins": gpubin_info,
    }
