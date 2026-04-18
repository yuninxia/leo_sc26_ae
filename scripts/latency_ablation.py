#!/usr/bin/env python3
"""Latency pruning ablation study for Leo.

Compares Leo's blame attribution with latency pruning ON vs OFF on the same
profile data. This validates whether latency pruning improves (or at least
does not degrade) analysis quality.

For each benchmark, runs BackSliceEngine twice:
  1. With latency pruning enabled (default)
  2. With latency pruning disabled

Then compares the top root causes using Jaccard, Kendall tau, and category
shift metrics — the same metrics used in latency_sensitivity.py.

Usage:
    # Analyze top kernel from measurements directory
    python scripts/latency_ablation.py /path/to/measurements --arch mi300

    # Save results as JSON
    python scripts/latency_ablation.py /path/to/measurements --arch mi300 \
        --output ablation_results.json
"""

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leo.analyzer import AnalysisConfig, AnalysisResult, KernelAnalyzer
from leo.analysis.backslice import BackSliceConfig, BackSliceEngine
from leo.arch import get_architecture
from leo.program_analysis import analyze_program


def _kendall_tau(ranking_a: Dict[int, float], ranking_b: Dict[int, float]) -> float:
    """Kendall tau-b rank correlation between two PC->blame mappings."""
    all_pcs = sorted(set(ranking_a) | set(ranking_b))
    n = len(all_pcs)
    if n < 2:
        return 1.0

    vals_a = [ranking_a.get(pc, 0.0) for pc in all_pcs]
    vals_b = [ranking_b.get(pc, 0.0) for pc in all_pcs]

    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff_a = vals_a[i] - vals_a[j]
            diff_b = vals_b[i] - vals_b[j]

            if diff_a == 0 and diff_b == 0:
                continue
            elif diff_a == 0:
                ties_a += 1
            elif diff_b == 0:
                ties_b += 1
            elif (diff_a > 0 and diff_b > 0) or (diff_a < 0 and diff_b < 0):
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt(
        (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    )
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def extract_metrics(result, stats, top_k: int = 5) -> dict:
    """Extract comparison metrics from a single analysis run."""
    top_sources = result.get_top_blame_sources(top_k)

    blame_per_pc: Dict[int, float] = {}
    for edge in result.blames:
        blame_per_pc[edge.src_pc] = (
            blame_per_pc.get(edge.src_pc, 0.0) + edge.total_blame()
        )

    return {
        "edges_pruned_latency": stats.edges_pruned_latency,
        "edges_pruned_opcode": stats.edges_pruned_opcode,
        "edges_pruned_barrier": stats.edges_pruned_barrier,
        "initial_edges": stats.initial_edges,
        "final_edges": stats.final_edges,
        "total_stall_blame": result.total_stall_blame,
        "total_lat_blame": result.total_lat_blame,
        "blame_by_category": dict(result.blame_by_category),
        "num_blame_edges": result.num_blame_edges,
        "blame_per_pc": blame_per_pc,
        "top_sources": [
            {"pc": pc, "blame": blame, "opcode": opcode}
            for pc, blame, opcode in top_sources
        ],
        "top_source_pcs": [pc for pc, _, _ in top_sources],
    }


def run_ablation(
    measurements_dir: str,
    arch_name: str,
    kernel_rank: int = 1,
    top_k: int = 5,
    debug: bool = False,
) -> Tuple[str, dict, dict]:
    """Run latency pruning ablation on a single kernel.

    Returns:
        Tuple of (kernel_name, metrics_pruning_on, metrics_pruning_off).
    """
    print(f"Step 1: Discovering kernels in {measurements_dir}...")
    program_result = analyze_program(
        measurements_dir=measurements_dir,
        arch=arch_name,
        top_n_kernels=kernel_rank,
        sort_by="stall_cycles",
        run_full_analysis=True,
        skip_failed_kernels=True,
        debug=debug,
    )

    analyzed = program_result.get_analyzed_kernels()
    if not analyzed:
        raise ValueError("No kernels successfully analyzed")

    target = analyzed[min(kernel_rank - 1, len(analyzed) - 1)]
    kernel_name = target.kernel_name
    print(f"Step 2: Target kernel: {kernel_name}")
    print(f"         Stall cycles: {target.stall_cycles:,.0f}")

    analysis = target.analysis_result
    if analysis is None:
        raise ValueError(f"No analysis result for kernel {kernel_name}")

    instructions = analysis.instructions
    vma_map = analysis.vma_map
    cfg = analysis.cfg
    config = analysis.config
    arch = get_architecture(arch_name)

    results = {}
    for label, enable_latency in [("ON", True), ("OFF", False)]:
        bs_config = BackSliceConfig(
            arch_name=config.arch,
            enable_predicate_tracking=config.enable_predicate_tracking,
            predicate_track_limit=config.predicate_track_limit,
            use_min_latency=config.use_min_latency,
            apply_opcode_pruning=config.apply_opcode_pruning,
            apply_barrier_pruning=config.apply_barrier_pruning,
            apply_graph_latency_pruning=enable_latency,
            enable_execution_pruning=config.enable_execution_pruning,
            stall_threshold=config.stall_threshold,
            debug=debug,
        )

        engine = BackSliceEngine(
            vma_map=vma_map,
            instructions=instructions,
            cfg=cfg,
            config=bs_config,
            arch_override=arch,
        )
        blame_result = engine.analyze()
        metrics = extract_metrics(blame_result, engine.stats, top_k)
        results[label] = metrics

        print(
            f"  Pruning {label}: "
            f"lat_prune={metrics['edges_pruned_latency']}, "
            f"final_edges={metrics['final_edges']}, "
            f"blame_edges={metrics['num_blame_edges']}"
        )

    return kernel_name, results["ON"], results["OFF"]


def print_ablation_table(
    kernel_name: str,
    arch_name: str,
    on: dict,
    off: dict,
    top_k: int,
):
    """Print formatted ablation comparison table."""
    print()
    print("=" * 80)
    print("Latency Pruning Ablation Study")
    print("=" * 80)
    print(f"Kernel:       {kernel_name}")
    print(f"Architecture: {arch_name}")
    print(f"Top-K:        {top_k}")
    print()

    # Edge counts
    print("Edge Counts:")
    print(f"  {'':20s}  {'Pruning ON':>12s}  {'Pruning OFF':>12s}  {'Delta':>8s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*8}")

    rows = [
        ("Initial edges", on["initial_edges"], off["initial_edges"]),
        ("Opcode pruned", on["edges_pruned_opcode"], off["edges_pruned_opcode"]),
        ("Barrier pruned", on["edges_pruned_barrier"], off["edges_pruned_barrier"]),
        ("Latency pruned", on["edges_pruned_latency"], off["edges_pruned_latency"]),
        ("Final edges", on["final_edges"], off["final_edges"]),
        ("Blame edges", on["num_blame_edges"], off["num_blame_edges"]),
    ]
    for label, v_on, v_off in rows:
        delta = v_on - v_off
        sign = "+" if delta > 0 else ""
        print(f"  {label:20s}  {v_on:>12d}  {v_off:>12d}  {sign}{delta:>7d}")

    # Top-K comparison
    on_pcs = set(on["top_source_pcs"])
    off_pcs = set(off["top_source_pcs"])
    if on_pcs or off_pcs:
        jaccard = len(on_pcs & off_pcs) / len(on_pcs | off_pcs)
    else:
        jaccard = 1.0

    top1_match = (
        on["top_source_pcs"][:1] == off["top_source_pcs"][:1]
        if on["top_source_pcs"] and off["top_source_pcs"]
        else True
    )

    kendall_tau = _kendall_tau(on["blame_per_pc"], off["blame_per_pc"])

    # Category shift
    all_cats = set(on["blame_by_category"]) | set(off["blame_by_category"])
    category_shift = sum(
        abs(on["blame_by_category"].get(c, 0) - off["blame_by_category"].get(c, 0))
        for c in all_cats
    )
    total = on["total_stall_blame"] + on["total_lat_blame"]
    category_shift_pct = (category_shift / total * 100) if total > 0 else 0.0

    print()
    print("Comparison (pruning ON vs OFF):")
    print(f"  Top-1 Match:      {'Yes' if top1_match else 'NO'}")
    print(f"  Top-{top_k} Jaccard:    {jaccard:.2f}")
    print(f"  Kendall tau:      {kendall_tau:.3f}")
    print(f"  Category shift:   {category_shift_pct:.1f}%")

    # Print top root causes side by side
    print()
    max_sources = max(len(on["top_sources"]), len(off["top_sources"]))
    print(f"Top-{top_k} Root Causes:")
    print(f"  {'#':>2s}  {'Pruning ON':42s}  {'Pruning OFF':42s}")
    print(f"  {'':>2s}  {'-'*42}  {'-'*42}")

    for i in range(max_sources):
        on_str = ""
        off_str = ""
        if i < len(on["top_sources"]):
            s = on["top_sources"][i]
            on_str = f"{s['pc']:#x} ({s['opcode']:12s}) {s['blame']:>14,.0f}c"
        if i < len(off["top_sources"]):
            s = off["top_sources"][i]
            off_str = f"{s['pc']:#x} ({s['opcode']:12s}) {s['blame']:>14,.0f}c"

        match = ""
        if i < len(on["top_sources"]) and i < len(off["top_sources"]):
            if on["top_sources"][i]["pc"] == off["top_sources"][i]["pc"]:
                match = "="
            elif on["top_sources"][i]["pc"] in off_pcs:
                match = "~"  # Same PC, different rank
            else:
                match = "!"  # Different PC
        print(f"  {i+1:>2d}  {on_str:42s}  {off_str:42s}  {match}")

    # Category breakdown
    print()
    print("Blame by Category:")
    print(f"  {'Category':20s}  {'Pruning ON':>14s}  {'Pruning OFF':>14s}  {'Delta':>10s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*10}")
    for cat in sorted(all_cats):
        v_on = on["blame_by_category"].get(cat, 0)
        v_off = off["blame_by_category"].get(cat, 0)
        delta = v_on - v_off
        sign = "+" if delta > 0 else ""
        print(f"  {cat:20s}  {v_on:>14,.0f}  {v_off:>14,.0f}  {sign}{delta:>9,.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Latency pruning ablation study for Leo",
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
        "--kernel-rank",
        type=int,
        default=1,
        help="Which kernel to analyze by stall ranking (1=top, default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top root causes to compare (default: 5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results as JSON file",
    )

    args = parser.parse_args()

    if not args.measurements_dir.exists():
        print(f"Error: Measurements directory not found: {args.measurements_dir}")
        return 1

    try:
        kernel_name, on_metrics, off_metrics = run_ablation(
            measurements_dir=str(args.measurements_dir),
            arch_name=args.arch,
            kernel_rank=args.kernel_rank,
            top_k=args.top_k,
            debug=args.debug,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print_ablation_table(kernel_name, args.arch, on_metrics, off_metrics, args.top_k)

    if args.output:
        output_data = {
            "kernel_name": kernel_name,
            "architecture": args.arch,
            "top_k": args.top_k,
            "pruning_on": on_metrics,
            "pruning_off": off_metrics,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
