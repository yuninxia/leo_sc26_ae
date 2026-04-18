#!/usr/bin/env python3
"""Flexible script to analyze any HPCToolkit-profiled GPU benchmark with Leo.

Usage:
    python scripts/analyze_benchmark.py <measurements_dir> [--arch ARCH]

Examples:
    # Whole-program analysis (default) - ranks kernels by stall cycles
    python scripts/analyze_benchmark.py /path/to/hpctoolkit-measurements --arch mi300

    # Whole-program analysis ranked by execution time
    python scripts/analyze_benchmark.py /path/to/measurements --arch mi300 --sort-by execution_time

    # Per-binary analysis (legacy mode)
    python scripts/analyze_benchmark.py /path/to/measurements --arch mi300 --mode binary

    # Analyze top 5 kernels only
    python scripts/analyze_benchmark.py /path/to/measurements --arch mi300 --top-n 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leo import analyze_kernel, analyze_program


def find_database(measurements_dir: Path) -> Path:
    """Find the HPCToolkit database directory."""
    # Try common naming patterns
    base_name = measurements_dir.name.replace("-measurements", "")
    candidates = [
        measurements_dir.parent / f"{base_name}-database",
        measurements_dir.parent / f"hpctoolkit-{base_name.replace('hpctoolkit-', '')}-database",
        measurements_dir.with_name(measurements_dir.name.replace("measurements", "database")),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # List available databases
    available = list(measurements_dir.parent.glob("*-database"))
    if available:
        print(f"Available databases: {[d.name for d in available]}")

    raise FileNotFoundError(f"Could not find database for {measurements_dir}")


def find_all_gpubins(measurements_dir: Path) -> list[Path]:
    """Find all GPU binaries in measurements directory.

    Prefers gpubins-used/ (actually profiled) over gpubins/ (all compiled).
    Handles broken symlinks by falling back to gpubins/.
    """
    # Prefer gpubins-used/ - these are the ones actually profiled
    gpubins_used_dir = measurements_dir / "gpubins-used"
    if gpubins_used_dir.exists():
        gpubins = list(gpubins_used_dir.glob("*.gpubin"))
        # Filter out broken symlinks - check if files actually exist
        valid_gpubins = [g for g in gpubins if g.exists()]
        if valid_gpubins:
            return sorted(valid_gpubins)
        # If all symlinks are broken, fall through to gpubins/

    # Fallback to gpubins/
    gpubins_dir = measurements_dir / "gpubins"
    if not gpubins_dir.exists():
        raise FileNotFoundError(f"No gpubins directory in {measurements_dir}")

    gpubins = list(gpubins_dir.glob("*.gpubin"))
    # Filter out broken symlinks
    valid_gpubins = [g for g in gpubins if g.exists()]
    if not valid_gpubins:
        raise FileNotFoundError(f"No valid .gpubin files in {gpubins_dir}")

    return sorted(valid_gpubins)


def find_hpcstruct(measurements_dir: Path, gpubin: Path) -> Path | None:
    """Find the hpcstruct file for source mapping."""
    structs_dir = measurements_dir / "structs"
    if not structs_dir.exists():
        return None

    # Look for GPU struct file matching the gpubin
    gpubin_name = gpubin.name
    candidates = [
        structs_dir / f"{gpubin_name}-gpucfg-yes.hpcstruct",
        structs_dir / f"{gpubin_name}.hpcstruct",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Try any GPU hpcstruct
    gpu_structs = list(structs_dir.glob("*.gpubin*.hpcstruct"))
    if gpu_structs:
        return gpu_structs[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a GPU benchmark with Leo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "measurements_dir",
        type=Path,
        help="Path to HPCToolkit measurements directory",
    )
    parser.add_argument(
        "--arch",
        default="a100",
        help="GPU architecture (default: a100)",
    )
    parser.add_argument(
        "--mode",
        choices=["program", "binary"],
        default="program",
        help="Analysis mode: 'program' for whole-program analysis (default), 'binary' for per-binary analysis",
    )
    parser.add_argument(
        "--sort-by",
        choices=["stall_cycles", "execution_time"],
        default="stall_cycles",
        help="Kernel ranking metric for program mode (default: stall_cycles)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Analyze only top N kernels (default: all)",
    )
    parser.add_argument(
        "--no-full-analysis",
        action="store_true",
        help="Skip full Leo back-slicing analysis (metrics only)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Override database path (auto-detected by default)",
    )
    parser.add_argument(
        "--gpubin",
        type=Path,
        help="Override gpubin path (auto-detected by default)",
    )
    parser.add_argument(
        "--hpcstruct",
        type=Path,
        help="Override hpcstruct path (auto-detected by default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--disable-latency-pruning",
        action="store_true",
        help="Disable latency-based dependency pruning (Stage 3)",
    )
    parser.add_argument(
        "--enable-execution-pruning",
        action="store_true",
        help="Enable execution constraints pruning (GPA Stage 4)",
    )

    args = parser.parse_args()

    measurements_dir = args.measurements_dir.resolve()
    if not measurements_dir.exists():
        print(f"Error: Measurements directory not found: {measurements_dir}")
        return 1

    # Whole-program analysis mode (default)
    if args.mode == "program":
        return run_program_analysis(args, measurements_dir)
    else:
        return run_binary_analysis(args, measurements_dir)


def run_program_analysis(args, measurements_dir: Path) -> int:
    """Run whole-program analysis across all kernels."""
    print(f"Mode:       Whole-Program Analysis")
    print(f"Sort by:    {args.sort_by}")
    print(f"Top N:      {args.top_n or 'all'}")
    print(f"Arch:       {args.arch}")
    print()

    try:
        result = analyze_program(
            measurements_dir=str(measurements_dir),
            arch=args.arch,
            top_n_kernels=args.top_n,
            sort_by=args.sort_by,
            run_full_analysis=not args.no_full_analysis,
            skip_failed_kernels=True,
            debug=args.debug,
            apply_graph_latency_pruning=not args.disable_latency_pruning,
            enable_execution_pruning=args.enable_execution_pruning,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Print summary
    print(result.summary())

    return 0


def run_binary_analysis(args, measurements_dir: Path) -> int:
    """Run per-binary analysis (legacy mode)."""
    print(f"Mode:       Per-Binary Analysis")

    # Auto-detect paths
    try:
        db_path = args.db or find_database(measurements_dir)
        if args.gpubin:
            # User specified a single gpubin
            gpubins = [args.gpubin]
        else:
            # Find all gpubins (prefer gpubins-used/)
            gpubins = find_all_gpubins(measurements_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Database:   {db_path}")
    print(f"GPU Binaries: {len(gpubins)} found")
    print(f"Arch:       {args.arch}")
    print()

    # Analyze each gpubin
    results_with_data = []
    for gpubin_path in gpubins:
        hpcstruct_path = args.hpcstruct or find_hpcstruct(measurements_dir, gpubin_path)

        if args.debug:
            print(f"Analyzing: {gpubin_path.name}")

        result = analyze_kernel(
            db_path=str(db_path),
            gpubin_path=str(gpubin_path),
            hpcstruct_path=str(hpcstruct_path) if hpcstruct_path else None,
            arch=args.arch,
            debug=args.debug,
            apply_graph_latency_pruning=not args.disable_latency_pruning,
            enable_execution_pruning=args.enable_execution_pruning,
        )

        # Only keep results with actual profile data
        if result.total_stall_cycles > 0:
            results_with_data.append((gpubin_path, result))

    # Print results
    if not results_with_data:
        print("No profile data matched any GPU binary.")
        print("GPU binaries analyzed:")
        for gpubin in gpubins:
            print(f"  - {gpubin.name}")
        return 1

    for gpubin_path, result in results_with_data:
        print(f"\n=== {gpubin_path.name} ===")
        print(result.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
