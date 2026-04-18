#!/usr/bin/env python3
"""Cross-vendor GPU kernel comparison for RAJAPerf benchmarks.

Runs Leo analysis on HPCToolkit measurements from multiple GPU vendors,
matches kernels by RAJAPerf canonical name, and generates comparison figures.

Usage:
    uv run python scripts/compare_vendors.py \
        --nvidia tests/data/pc/nvidia/hpctoolkit-rajaperf.cudaoffload.gcc.cudagpu-measurements \
        --amd tests/data/pc/amd/hpctoolkit-rajaperf.hipoffload.amdclang.rocmgpu-measurements \
        --intel tests/data/pc/intel/hpctoolkit-rajaperf.sycloffload.icpx.intelgpu-measurements \
        --top-n 10 --output comparison.pdf
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from leo import analyze_program
from leo.program_analysis import _demangle_name

# Vendor colors (official brand colors)
VENDOR_COLORS = {
    "nvidia": "#76b900",
    "amd": "#ed1c24",
    "intel": "#0071c5",
}
VENDOR_LABELS = {
    "nvidia": "NVIDIA",
    "amd": "AMD",
    "intel": "Intel",
}

# High-level stall categories for cross-vendor comparison
STALL_CATEGORIES = {
    "Memory": ["gcycles:stl:mem", "gcycles:stl:gmem", "gcycles:stl:lmem", "gcycles:stl:tmem"],
    "Sync": ["gcycles:stl:sync"],
    "Dependency": ["gcycles:stl:idep"],
    "Pipeline": ["gcycles:stl:pipe", "gcycles:stl:ifet"],
}
CATEGORY_COLORS = {
    "Memory": "#e74c3c",
    "Sync": "#f39c12",
    "Dependency": "#3498db",
    "Pipeline": "#9b59b6",
    "Other": "#95a5a6",
}


def extract_rajaperf_name_from_string(name: str) -> Optional[str]:
    """Try to extract RAJAPerf canonical name from a single string."""
    demangled = _demangle_name(name)
    match = re.search(r"rajaperf::(\w+)::(\w+)", demangled)
    if match:
        group = match.group(1).capitalize()
        kernel = match.group(2).upper()
        # Skip internal helpers like lambda wrappers, use the actual kernel name
        if kernel in ("RUNHIPVARIANTIMPL", "RUNSYCLVARIANTIMPL",
                       "RUNCUDAVARIANTIMPL", "LAMBDA_HIP_FORALL"):
            return None
        return f"{group}_{kernel}"
    return None


def extract_rajaperf_name(kr) -> Optional[str]:
    """Extract RAJAPerf canonical name from a PerKernelAnalysis.

    Tries multiple strategies:
    1. Primary kernel_name (works for NVIDIA and Intel)
    2. All function_names in analysis stats (works for AMD where .text is first)
    3. Source file paths from blame edges (fallback)
    """
    # Strategy 1: Primary kernel name
    result = extract_rajaperf_name_from_string(kr.kernel_name)
    if result:
        return result

    # Strategy 2: Check all function names (AMD stores .text first, real name second)
    if kr.analysis_result and kr.analysis_result.stats.get("function_names"):
        for fn in kr.analysis_result.stats["function_names"]:
            result = extract_rajaperf_name_from_string(fn)
            if result:
                return result

    # Strategy 3: Extract from source file paths in blame edges
    if kr.analysis_result:
        pairs = kr.analysis_result.get_top_blame_pairs(1)
        for p in pairs:
            for key in ("stall_source", "cause_source"):
                src = p.get(key)
                if src and isinstance(src, tuple):
                    filepath = src[0]
                elif src and isinstance(src, str):
                    filepath = src
                else:
                    continue
                # Pattern: /RAJAPerf/src/{group}/{KERNEL}-Hip.cpp or {KERNEL}_BODY.hpp
                m = re.search(r"/RAJAPerf/src/(\w+)/(\w+?)[-_](?:Hip|Cuda|Sycl|BODY)", filepath)
                if m:
                    group = m.group(1).capitalize()
                    kernel = m.group(2).upper()
                    return f"{group}_{kernel}"

    return None


def run_vendor_analysis(measurements_dir: str, vendor: str, top_n: Optional[int] = None):
    """Run Leo whole-program analysis for a single vendor."""
    print(f"  Analyzing {VENDOR_LABELS[vendor]}: {measurements_dir}")
    result = analyze_program(
        measurements_dir,
        arch=vendor,
        top_n_kernels=top_n,
        run_full_analysis=True,
        skip_failed_kernels=True,
    )
    print(f"    {result.kernels_analyzed} kernels analyzed, "
          f"{result.kernels_skipped} skipped")
    return result


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Match kernels across vendors and build comparison DataFrame.

    Args:
        results: {vendor: ProgramAnalysisResult} dict.

    Returns:
        DataFrame indexed by canonical kernel name with columns for each
        vendor's metrics (e.g., nvidia_stall_ratio, amd_exec_time, etc.).
    """
    # Collect per-vendor kernel data keyed by canonical name
    vendor_kernels = {}  # {vendor: {canonical_name: PerKernelAnalysis}}

    for vendor, result in results.items():
        vendor_kernels[vendor] = {}
        matched = 0
        for kr in result.per_kernel_results:
            canonical = extract_rajaperf_name(kr)
            if canonical:
                vendor_kernels[vendor][canonical] = kr
                matched += 1
        print(f"  {VENDOR_LABELS[vendor]}: {matched}/{len(result.per_kernel_results)} "
              f"kernels matched to RAJAPerf names")

    # Find kernels present in all vendors
    all_names = set()
    for vk in vendor_kernels.values():
        all_names.update(vk.keys())

    vendors = list(results.keys())
    rows = []
    for name in sorted(all_names):
        row = {"kernel": name}
        present_in = 0
        for vendor in vendors:
            kr = vendor_kernels[vendor].get(name)
            if kr:
                present_in += 1
                row[f"{vendor}_stall_ratio"] = kr.stall_ratio
                row[f"{vendor}_exec_time"] = kr.execution_time_s
                row[f"{vendor}_stall_cycles"] = kr.stall_cycles
                row[f"{vendor}_total_cycles"] = kr.total_cycles
                row[f"{vendor}_gpu_util"] = kr.gpu_utilization_pct
                # Stall breakdown
                if kr.analysis_result:
                    breakdown = kr.analysis_result.vma_map.get_stall_type_breakdown()
                    total_stall = sum(breakdown.values()) or 1.0
                    for cat, metrics in STALL_CATEGORIES.items():
                        cat_cycles = sum(breakdown.get(m, 0.0) for m in metrics)
                        row[f"{vendor}_{cat.lower()}_pct"] = cat_cycles / total_stall
                    accounted = sum(
                        row.get(f"{vendor}_{cat.lower()}_pct", 0.0)
                        for cat in STALL_CATEGORIES
                    )
                    row[f"{vendor}_other_pct"] = max(0.0, 1.0 - accounted)
            else:
                row[f"{vendor}_stall_ratio"] = np.nan
                row[f"{vendor}_exec_time"] = np.nan
                row[f"{vendor}_stall_cycles"] = np.nan
        row["vendor_count"] = present_in
        rows.append(row)

    df = pd.DataFrame(rows).set_index("kernel")
    return df


def plot_overview(df: pd.DataFrame, vendors: list, top_n: int, ax: plt.Axes):
    """Plot overview: top-N kernels by max stall cycles, grouped stall ratio bars."""
    # Rank by maximum stall cycles across vendors
    stall_cols = [f"{v}_stall_cycles" for v in vendors]
    df["max_stall"] = df[stall_cols].max(axis=1)
    top = df.nlargest(top_n, "max_stall")

    x = np.arange(len(top))
    width = 0.8 / len(vendors)

    for i, vendor in enumerate(vendors):
        col = f"{vendor}_stall_ratio"
        values = top[col].fillna(0).values * 100
        offset = (i - len(vendors) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=VENDOR_LABELS[vendor],
               color=VENDOR_COLORS[vendor], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Kernel")
    ax.set_ylabel("Stall Ratio (%)")
    ax.set_title(f"Top {top_n} Kernels — Stall Ratio by Vendor")
    ax.set_xticks(x)
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)


def plot_kernel_detail(kernel_name: str, df_row: pd.Series, vendors: list,
                       fig: plt.Figure):
    """Plot detailed comparison for a single kernel (4 panels)."""
    axes = fig.subplots(2, 2)
    fig.suptitle(kernel_name, fontsize=14, fontweight="bold")

    # Panel A: Stall Ratio
    ax = axes[0, 0]
    vals = [df_row.get(f"{v}_stall_ratio", 0) * 100 for v in vendors]
    colors = [VENDOR_COLORS[v] for v in vendors]
    labels = [VENDOR_LABELS[v] for v in vendors]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Stall Ratio (%)")
    ax.set_title("Stall Ratio")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Stall Breakdown (stacked bars)
    ax = axes[0, 1]
    categories = list(STALL_CATEGORIES.keys()) + ["Other"]
    bottoms = [0.0] * len(vendors)
    for cat in categories:
        cat_vals = []
        for v in vendors:
            pct = df_row.get(f"{v}_{cat.lower()}_pct", 0)
            cat_vals.append(pct * 100 if not np.isnan(pct) else 0)
        ax.bar(labels, cat_vals, bottom=bottoms, color=CATEGORY_COLORS[cat],
               label=cat, edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, cat_vals)]
    ax.set_ylabel("Stall Breakdown (%)")
    ax.set_title("Stall Categories")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Execution Time
    ax = axes[1, 0]
    vals = [df_row.get(f"{v}_exec_time", 0) * 1000 for v in vendors]  # ms
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}ms", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Execution Time")
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Stall Cycles
    ax = axes[1, 1]
    vals = [df_row.get(f"{v}_stall_cycles", 0) / 1e9 for v in vendors]  # billions
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val) and val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}B", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Stall Cycles (billions)")
    ax.set_title("Total Stall Cycles")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    parser = argparse.ArgumentParser(
        description="Cross-vendor GPU kernel comparison for RAJAPerf"
    )
    parser.add_argument("--nvidia", type=str, help="NVIDIA measurements directory")
    parser.add_argument("--amd", type=str, help="AMD measurements directory")
    parser.add_argument("--intel", type=str, help="Intel measurements directory")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top kernels to compare (default: 10)")
    parser.add_argument("--output", type=str, default="comparison.pdf",
                        help="Output PDF path (default: comparison.pdf)")
    parser.add_argument("--analysis-top-n", type=int, default=None,
                        help="Limit per-vendor analysis to top N kernels (faster)")
    parser.add_argument("--all-three-only", action="store_true",
                        help="Only include kernels present in all 3 vendors")
    args = parser.parse_args()

    # Collect vendor inputs
    vendor_dirs = {}
    if args.nvidia:
        vendor_dirs["nvidia"] = args.nvidia
    if args.amd:
        vendor_dirs["amd"] = args.amd
    if args.intel:
        vendor_dirs["intel"] = args.intel

    if len(vendor_dirs) < 2:
        print("Error: Need at least 2 vendors to compare.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Step 1: Run analysis per vendor
    print("Step 1: Running Leo analysis per vendor...")
    results = {}
    for vendor, mdir in vendor_dirs.items():
        results[vendor] = run_vendor_analysis(mdir, vendor, args.analysis_top_n)

    # Step 2: Match kernels and build comparison table
    print("\nStep 2: Matching kernels across vendors...")
    df = build_comparison_table(results)
    vendors = list(vendor_dirs.keys())

    # Filter to kernels present in multiple vendors
    shared = df[df["vendor_count"] >= 2]
    all_three = df[df["vendor_count"] == len(vendors)]
    print(f"  {len(shared)} kernels in 2+ vendors, "
          f"{len(all_three)} in all {len(vendors)} vendors")

    if args.all_three_only and len(vendors) >= 3:
        shared = all_three
        print(f"  --all-three-only: using {len(shared)} kernels")

    if len(shared) == 0:
        print("Error: No matching kernels found across vendors.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Generate PDF
    print(f"\nStep 3: Generating comparison figures → {args.output}")
    top_n = min(args.top_n, len(shared))

    with PdfPages(args.output) as pdf:
        # Page 1: Overview
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_overview(shared, vendors, top_n, ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Per-kernel detail pages (top N by max stall cycles)
        stall_cols = [f"{v}_stall_cycles" for v in vendors]
        shared_sorted = shared.copy()
        shared_sorted["max_stall"] = shared_sorted[stall_cols].max(axis=1)
        top_kernels = shared_sorted.nlargest(top_n, "max_stall")

        for kernel_name in top_kernels.index:
            fig = plt.figure(figsize=(11, 8))
            plot_kernel_detail(kernel_name, top_kernels.loc[kernel_name],
                               vendors, fig)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nDone! {top_n + 1} pages written to {args.output}")

    # Print summary table to stdout
    print(f"\n{'Kernel':<30} ", end="")
    for v in vendors:
        print(f" {VENDOR_LABELS[v]:>12}", end="")
    print()
    print("-" * (30 + 13 * len(vendors)))
    for name in top_kernels.index:
        row = top_kernels.loc[name]
        print(f"{name:<30} ", end="")
        for v in vendors:
            sr = row.get(f"{v}_stall_ratio", float("nan"))
            if not np.isnan(sr):
                print(f" {sr:>11.1%}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()


if __name__ == "__main__":
    main()
