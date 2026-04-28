#!/usr/bin/env python3
"""CLI entry point for ground truth validation.

Usage:
  uv run python -m scripts.validation.main
  uv run python -m scripts.validation.main --rerun-leo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import (
    VENDOR_ARCH, VENDORS, TABLE_VI_KERNELS, REALWORLD_APPS,
)
from .core import (
    ValidationResult,
    find_latest_result_dir,
    validate_kernel, validate_realworld_app,
)
from .formatting import format_report
from .leo_rerun import RerunTask, batch_rerun_leo_docker


def main():
    parser = argparse.ArgumentParser(
        description="Ground truth validation: Leo root causes vs optimization locations."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Per-kernel results directory (default: {leo_root}/results/per-kernel)",
    )
    parser.add_argument(
        "--benchmarks-dir", type=Path, default=None,
        help="RAJAPerf benchmarks directory (default: {leo_root}/benchmarks/rajaperf-h100)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Consider top K root causes from Leo (default: 5)",
    )
    parser.add_argument(
        "--rerun-leo", action="store_true",
        help="Re-run Leo analysis using latest code (CPU-only)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also emit machine-readable JSON",
    )
    args = parser.parse_args()

    # Resolve paths — scripts/validation/ is two levels below leo_root
    script_dir = Path(__file__).resolve().parent
    leo_root = script_dir.parent.parent

    results_dir = args.results_dir or (leo_root / "results" / "per-kernel")
    benchmarks_dir = args.benchmarks_dir or (leo_root / "benchmarks" / "rajaperf-h100")

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover kernels (filtered to Table VI whitelist)
    all_kernel_dirs = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    excluded = [k for k in all_kernel_dirs if k not in TABLE_VI_KERNELS]
    kernel_dirs = [k for k in all_kernel_dirs if k in TABLE_VI_KERNELS]

    print(f"Discovered {len(all_kernel_dirs)} kernels in {results_dir}", file=sys.stderr)
    if excluded:
        print(f"  Excluded {len(excluded)} kernels not in Table VI: {', '.join(excluded)}", file=sys.stderr)
    print(f"  Evaluating {len(kernel_dirs)} Table VI kernels", file=sys.stderr)
    print(f"Benchmarks: {benchmarks_dir}", file=sys.stderr)

    results_root = leo_root / "results"
    benchmarks_root = leo_root / "benchmarks"

    # Batch re-run Leo in Docker (single container for all kernels + apps)
    if args.rerun_leo:
        rerun_tasks: list[RerunTask] = []

        # RAJAPerf kernels
        for kernel in kernel_dirs:
            for vendor in VENDORS:
                vendor_dir = results_dir / kernel / vendor
                if not vendor_dir.exists():
                    continue
                meas_dirs = list(vendor_dir.glob("hpctoolkit-*-measurements"))
                if meas_dirs:
                    rerun_tasks.append(RerunTask(
                        label=f"{kernel}/{vendor}",
                        meas_dir=meas_dirs[0],
                        arch=VENDOR_ARCH.get(vendor, vendor),
                        output_file=vendor_dir / "leo_output_latest.txt",
                    ))

        # Real-world apps
        for app_name, app_config in REALWORLD_APPS.items():
            for vendor in VENDORS:
                pattern = app_config["result_patterns"].get(vendor)
                if not pattern:
                    continue
                result_dir = find_latest_result_dir(results_root, pattern)
                if not result_dir:
                    continue
                meas_dirs = list(result_dir.glob("hpctoolkit-*-measurements"))
                if meas_dirs:
                    rerun_tasks.append(RerunTask(
                        label=f"{app_name}/{vendor}",
                        meas_dir=meas_dirs[0],
                        arch=VENDOR_ARCH.get(vendor, vendor),
                        output_file=result_dir / "leo_output_latest.txt",
                    ))

        batch_rerun_leo_docker(rerun_tasks, leo_root)

    # Run validation for each kernel/vendor
    all_results: list[ValidationResult] = []

    for kernel in kernel_dirs:
        for vendor in VENDORS:
            vendor_dir = results_dir / kernel / vendor
            if not vendor_dir.exists():
                continue

            result = validate_kernel(
                results_dir=results_dir,
                benchmarks_dir=benchmarks_dir,
                kernel=kernel,
                vendor=vendor,
                top_k=args.top_k,
            )
            all_results.append(result)

    # Real-world apps (miniBUDE, XSBench, LULESH, HipKittens)
    print(f"\nReal-world apps: {', '.join(REALWORLD_APPS.keys())}", file=sys.stderr)

    for app_name, app_config in REALWORLD_APPS.items():
        for vendor in VENDORS:
            r = validate_realworld_app(
                results_root=results_root,
                benchmarks_root=benchmarks_root,
                app_name=app_name,
                vendor=vendor,
                app_config=app_config,
                top_k=args.top_k,
            )
            if r is not None:
                all_results.append(r)
                print(f"  {app_name}/{vendor}: {r.top1_match or r.error}", file=sys.stderr)

    # Format and output
    report = format_report(all_results, args.top_k, args.rerun_leo)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    # JSON output
    if args.json:
        json_path = args.output.replace(".txt", ".json") if args.output else "validation_results.json"

        json_data = {
            "settings": {"top_k": args.top_k, "rerun_leo": args.rerun_leo},
            "results": [],
        }
        for r in all_results:
            entry = {
                "kernel": r.kernel,
                "vendor": r.vendor,
                "has_optimization": r.has_optimization,
                "has_leo_data": r.has_leo_data,
                "top1_distance": r.top1_distance,
                "top1_match": r.top1_match,
                "topk_min_distance": r.topk_min_distance,
                "topk_best_match": r.topk_best_match,
                "diff_file": r.diff_file,
                "num_diff_lines": len(r.diff_lines),
                "num_root_causes": len(r.root_causes),
                "root_causes": [
                    {
                        "root_file": rc.root_file,
                        "root_line": rc.root_line,
                        "root_opcode": rc.root_opcode,
                        "stall_file": rc.stall_file,
                        "stall_line": rc.stall_line,
                        "cycles": rc.cycles,
                        "pct": rc.pct,
                        "speedup": rc.speedup,
                    }
                    for rc in r.root_causes
                ],
                "error": r.error,
            }
            json_data["results"].append(entry)

        Path(json_path).write_text(json.dumps(json_data, indent=2))
        print(f"JSON written to {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
