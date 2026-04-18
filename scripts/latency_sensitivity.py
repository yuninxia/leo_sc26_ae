#!/usr/bin/env python3
"""Latency table sensitivity analysis for Leo.

Proves that Leo's latency pruning results are robust to imprecise latency
values by perturbing all latency values by multiplicative factors and
comparing blame attribution results across perturbation levels.

If the top root causes remain stable across perturbation levels (e.g., 0.5x
to 2.0x), it demonstrates that Leo's analysis is not sensitive to the exact
latency table values — only to the relative ordering of instruction costs.

Usage:
    # Analyze top kernel from measurements directory
    python scripts/latency_sensitivity.py /path/to/measurements --arch mi300

    # Custom perturbation levels
    python scripts/latency_sensitivity.py /path/to/measurements --arch mi300 \
        --scales 0.25 0.5 0.8 1.0 1.2 1.5 2.0 4.0

    # Analyze specific kernel by rank
    python scripts/latency_sensitivity.py /path/to/measurements --arch mi300 \
        --kernel-rank 2  # 2nd-highest stalling kernel

    # Save results as JSON
    python scripts/latency_sensitivity.py /path/to/measurements --arch mi300 \
        --output sensitivity_results.json
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
from leo.arch import get_architecture, PerturbedArchitecture
from leo.program_analysis import analyze_program


DEFAULT_SCALES = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def _kendall_tau(ranking_a: Dict[int, float], ranking_b: Dict[int, float]) -> float:
    """Kendall tau-b rank correlation between two PC->blame mappings.

    PCs present in one ranking but not the other are assigned blame=0.
    Returns tau in [-1, +1]. Returns 1.0 if fewer than 2 PCs.

    Uses tau-b variant which adjusts for ties (important because absent
    PCs get 0 blame, creating many tied-at-zero entries).
    """
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
                continue  # Joint tie — not counted
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
        return 1.0  # All values identical in at least one ranking
    return (concordant - discordant) / denom


def extract_metrics(
    result, stats, top_k: int = 5,
) -> dict:
    """Extract comparison metrics from a single analysis run.

    Args:
        result: KernelBlameResult from BackSliceEngine.analyze().
        stats: BackSliceStats from the engine.
        top_k: Number of top root causes to track.

    Returns:
        Dict of metric name -> value.
    """
    top_sources = result.get_top_blame_sources(top_k)

    # Aggregate blame by source PC for rank correlation
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


def compute_stability(
    results: Dict[float, dict],
    baseline_scale: float = 1.0,
) -> Dict[float, dict]:
    """Compare results across perturbation levels against the baseline.

    Metrics:
    - top1_match: Whether top-1 root cause PC matches baseline
    - top_k_jaccard: Jaccard similarity of top-K root cause PCs
    - kendall_tau: Kendall tau-b rank correlation of full PC blame ranking
    - category_shift_pct: L1 norm of blame_by_category delta (% of total)
    - edge_pruning_delta: Change in edges pruned by latency stage
    """
    baseline = results[baseline_scale]
    baseline_pcs = set(baseline["top_source_pcs"])

    stability = OrderedDict()
    for scale in sorted(results.keys()):
        data = results[scale]
        these_pcs = set(data["top_source_pcs"])

        # Jaccard similarity (1.0 = identical top-K, 0.0 = no overlap)
        if baseline_pcs or these_pcs:
            jaccard = len(baseline_pcs & these_pcs) / len(baseline_pcs | these_pcs)
        else:
            jaccard = 1.0

        # Top-1 match
        top1_match = (
            data["top_source_pcs"][:1] == baseline["top_source_pcs"][:1]
            if data["top_source_pcs"] and baseline["top_source_pcs"]
            else True
        )

        # Edge pruning delta
        edge_delta = data["edges_pruned_latency"] - baseline["edges_pruned_latency"]

        # Category blame shift (L1 norm)
        all_cats = set(baseline["blame_by_category"]) | set(data["blame_by_category"])
        category_shift = sum(
            abs(
                data["blame_by_category"].get(c, 0)
                - baseline["blame_by_category"].get(c, 0)
            )
            for c in all_cats
        )
        total = baseline["total_stall_blame"] + baseline["total_lat_blame"]
        category_shift_pct = (category_shift / total * 100) if total > 0 else 0.0

        # Kendall tau rank correlation (full PC blame ranking)
        kendall_tau = _kendall_tau(baseline["blame_per_pc"], data["blame_per_pc"])

        stability[scale] = {
            "top1_match": top1_match,
            "top_k_jaccard": jaccard,
            "kendall_tau": kendall_tau,
            "edges_pruned_latency": data["edges_pruned_latency"],
            "edge_pruning_delta": edge_delta,
            "final_edges": data["final_edges"],
            "category_shift_pct": category_shift_pct,
            "num_blame_edges": data["num_blame_edges"],
        }

    return stability


def run_sensitivity(
    measurements_dir: str,
    arch_name: str,
    scales: List[float],
    kernel_rank: int = 1,
    top_k: int = 5,
    debug: bool = False,
) -> Tuple[str, Dict[float, dict]]:
    """Run latency sensitivity analysis on a single kernel.

    1. Runs whole-program analysis to identify the target kernel
    2. Runs KernelAnalyzer once to get pre-pruning state
    3. For each scale, re-runs BackSliceEngine with PerturbedArchitecture

    Args:
        measurements_dir: Path to HPCToolkit measurements.
        arch_name: Architecture name (e.g., "mi300", "a100", "pvc").
        scales: List of latency scale factors (1.0 = baseline).
        kernel_rank: Which kernel to analyze (1 = top stalling).
        top_k: Number of top root causes to compare.
        debug: Enable debug output.

    Returns:
        Tuple of (kernel_name, results_dict) where results_dict maps
        scale_factor -> metrics dict.
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

    # Pick the target kernel
    target = analyzed[min(kernel_rank - 1, len(analyzed) - 1)]
    kernel_name = target.kernel_name
    print(f"Step 2: Target kernel: {kernel_name}")
    print(f"         Stall cycles: {target.stall_cycles:,.0f}")

    # We need the KernelAnalyzer's intermediate results
    # The AnalysisResult has instructions, vma_map, and cfg
    analysis = target.analysis_result
    if analysis is None:
        raise ValueError(f"No analysis result for kernel {kernel_name}")

    instructions = analysis.instructions
    vma_map = analysis.vma_map
    cfg = analysis.cfg
    config = analysis.config

    base_arch = get_architecture(arch_name)
    results = OrderedDict()

    print(f"Step 3: Running {len(scales)} perturbation levels...")
    for scale in sorted(scales):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=scale)

        bs_config = BackSliceConfig(
            arch_name=config.arch,
            enable_predicate_tracking=config.enable_predicate_tracking,
            predicate_track_limit=config.predicate_track_limit,
            use_min_latency=config.use_min_latency,
            apply_opcode_pruning=config.apply_opcode_pruning,
            apply_barrier_pruning=config.apply_barrier_pruning,
            apply_graph_latency_pruning=config.apply_graph_latency_pruning,
            enable_execution_pruning=config.enable_execution_pruning,
            stall_threshold=config.stall_threshold,
            debug=debug,
        )

        engine = BackSliceEngine(
            vma_map=vma_map,
            instructions=instructions,
            cfg=cfg,
            config=bs_config,
            arch_override=perturbed,
        )
        blame_result = engine.analyze()
        metrics = extract_metrics(blame_result, engine.stats, top_k)
        results[scale] = metrics

        marker = " <- baseline" if scale == 1.0 else ""
        print(
            f"  {scale:.2f}x: "
            f"lat_prune={metrics['edges_pruned_latency']}, "
            f"final_edges={metrics['final_edges']}, "
            f"blame_edges={metrics['num_blame_edges']}"
            f"{marker}"
        )

    return kernel_name, results


def print_results_table(
    kernel_name: str,
    arch_name: str,
    stability: Dict[float, dict],
    top_k: int,
    results: Dict[float, dict],
):
    """Print a formatted results table."""
    print()
    print("=" * 80)
    print("Latency Sensitivity Analysis")
    print("=" * 80)
    print(f"Kernel:       {kernel_name}")
    print(f"Architecture: {arch_name} (baseline scale = 1.0)")
    print(f"Top-K:        {top_k}")
    print()

    # Header
    header = (
        f"{'Scale':>7s}  "
        f"{'LatPrune':>8s}  "
        f"{'FinalEdg':>8s}  "
        f"{'Top1Match':>9s}  "
        f"{'TopK_Jacc':>9s}  "
        f"{'KendallT':>8s}  "
        f"{'CatShift%':>9s}"
    )
    print(header)
    print("-" * len(header))

    for scale, data in stability.items():
        marker = "  <- baseline" if scale == 1.0 else ""
        top1_str = "Yes" if data["top1_match"] else "NO"
        print(
            f"{scale:>6.2f}x  "
            f"{data['edges_pruned_latency']:>8d}  "
            f"{data['final_edges']:>8d}  "
            f"{top1_str:>9s}  "
            f"{data['top_k_jaccard']:>9.2f}  "
            f"{data['kendall_tau']:>8.3f}  "
            f"{data['category_shift_pct']:>8.1f}%"
            f"{marker}"
        )

    # Summary
    print()
    all_top1_match = all(d["top1_match"] for d in stability.values())
    min_jaccard = min(d["top_k_jaccard"] for d in stability.values())
    min_kendall = min(d["kendall_tau"] for d in stability.values())
    max_cat_shift = max(d["category_shift_pct"] for d in stability.values())

    if all_top1_match:
        print(f"Summary: Top-1 root cause is STABLE across all perturbation levels.")
    else:
        unstable = [f"{s:.2f}x" for s, d in stability.items() if not d["top1_match"]]
        print(f"Summary: Top-1 root cause CHANGES at: {', '.join(unstable)}")

    print(f"         Min Top-{top_k} Jaccard:  {min_jaccard:.2f}")
    print(f"         Min Kendall tau:    {min_kendall:.3f}")
    print(f"         Max category shift: {max_cat_shift:.1f}%")

    # Print top root causes at baseline
    baseline_data = None
    for scale, data in results.items():
        if scale == 1.0:
            baseline_data = data
            break
    if baseline_data and baseline_data["top_sources"]:
        print()
        print(f"Top-{top_k} root causes at baseline (1.0x):")
        for i, src in enumerate(baseline_data["top_sources"], 1):
            print(f"  {i}. PC {src['pc']:#x} ({src['opcode']}): {src['blame']:,.0f} cycles")


def main():
    parser = argparse.ArgumentParser(
        description="Latency table sensitivity analysis for Leo",
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
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help=f"Latency scale factors (default: {DEFAULT_SCALES})",
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

    # Ensure 1.0 is in scales (baseline)
    if 1.0 not in args.scales:
        args.scales.append(1.0)
        args.scales.sort()

    try:
        kernel_name, results = run_sensitivity(
            measurements_dir=str(args.measurements_dir),
            arch_name=args.arch,
            scales=args.scales,
            kernel_rank=args.kernel_rank,
            top_k=args.top_k,
            debug=args.debug,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    stability = compute_stability(results, baseline_scale=1.0)
    print_results_table(kernel_name, args.arch, stability, args.top_k, results)

    # Save JSON if requested
    if args.output:
        output_data = {
            "kernel_name": kernel_name,
            "architecture": args.arch,
            "top_k": args.top_k,
            "scales": args.scales,
            "results": {
                str(k): v for k, v in results.items()
            },
            "stability": {
                str(k): v for k, v in stability.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
