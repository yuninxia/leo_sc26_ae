"""Output formatting and export for validation results."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .constants import VENDORS


def format_diff_range(diff_lines: list[int]) -> str:
    """Summarize diff lines as ranges, e.g., '45-62,88-102'."""
    if not diff_lines:
        return "(none)"

    ranges = []
    start = diff_lines[0]
    end = start

    for line in diff_lines[1:]:
        if line == end + 1:
            end = line
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = line

    ranges.append(f"{start}-{end}" if start != end else str(start))

    text = ",".join(ranges)
    if len(text) > 20:
        text = text[:17] + "..."
    return text


def format_report(results: list, top_k: int, rerun: bool) -> str:
    """Format human-readable validation report."""
    lines = []
    w = 100
    lines.append("=" * w)
    lines.append("Leo Ground Truth Validation Report".center(w))
    lines.append("=" * w)
    lines.append(f"Settings: top_k={top_k}, rerun_leo={rerun}")
    lines.append("")

    # Per-kernel table
    lines.append("PER-KERNEL RESULTS (Top-1 root cause vs optimization location)")
    lines.append("-" * w)
    header = f"{'Kernel':<25} {'Vendor':<8} {'Leo Top-1 Root Cause':<28} {'Opt Lines':<22} {'Dist':>5} {'Match':<10}"
    lines.append(header)
    lines.append("-" * w)

    for r in sorted(results, key=lambda x: (x.kernel, x.vendor)):
        # Format top-1 root cause
        if r.root_causes:
            rc = r.root_causes[0]
            if rc.root_file and rc.root_line:
                top1_str = f"{rc.root_file}:{rc.root_line}"
            elif rc.stall_file and rc.stall_line:
                top1_str = f"{rc.stall_file}:{rc.stall_line}"
            else:
                top1_str = "(hex only)"
        else:
            top1_str = "(no data)"

        if len(top1_str) > 26:
            top1_str = top1_str[:23] + "..."

        opt_str = format_diff_range(r.diff_lines)
        dist_str = str(r.top1_distance) if r.top1_distance is not None else "-"

        lines.append(
            f"{r.kernel:<25} {r.vendor:<8} {top1_str:<28} {opt_str:<22} {dist_str:>5} {r.top1_match:<10}"
        )

    lines.append("-" * w)
    lines.append("")

    # Top-K best match table
    lines.append(f"PER-KERNEL RESULTS (Best match among top-{top_k} root causes)")
    lines.append("-" * w)
    header2 = f"{'Kernel':<25} {'Vendor':<8} {'Best Root Cause':<28} {'Opt Lines':<22} {'Dist':>5} {'Match':<10}"
    lines.append(header2)
    lines.append("-" * w)

    for r in sorted(results, key=lambda x: (x.kernel, x.vendor)):
        # Find the root cause with minimum distance
        best_rc_str = "(no data)"
        if r.root_causes and r.diff_lines and r.diff_file:
            best_dist = None
            best_rc = None
            for rc in r.root_causes:
                for loc_file, loc_line in [(rc.root_file, rc.root_line), (rc.stall_file, rc.stall_line)]:
                    if loc_file and loc_line > 0 and os.path.basename(loc_file) == os.path.basename(r.diff_file):
                        for dl in r.diff_lines:
                            d = abs(loc_line - dl)
                            if best_dist is None or d < best_dist:
                                best_dist = d
                                best_rc = rc
            if best_rc:
                if best_rc.root_file and best_rc.root_line:
                    best_rc_str = f"{best_rc.root_file}:{best_rc.root_line}"
                elif best_rc.stall_file and best_rc.stall_line:
                    best_rc_str = f"{best_rc.stall_file}:{best_rc.stall_line}"

        if len(best_rc_str) > 26:
            best_rc_str = best_rc_str[:23] + "..."

        opt_str = format_diff_range(r.diff_lines)
        dist_str = str(r.topk_min_distance) if r.topk_min_distance is not None else "-"

        lines.append(
            f"{r.kernel:<25} {r.vendor:<8} {best_rc_str:<28} {opt_str:<22} {dist_str:>5} {r.topk_best_match:<10}"
        )

    lines.append("-" * w)
    lines.append("")

    # Summary by vendor
    lines.append("SUMMARY")
    lines.append("-" * w)

    # Count by vendor and overall
    stats: dict[str, dict[str, int]] = {}
    for v in VENDORS + ["overall"]:
        stats[v] = {
            "total": 0, "exact": 0, "near": 0, "same_func": 0,
            "same_file": 0, "no_opt": 0, "unmapped": 0, "no_data": 0,
            "dist_sum": 0, "dist_count": 0,
        }

    for r in results:
        for key in [r.vendor, "overall"]:
            s = stats[key]
            s["total"] += 1
            match = r.topk_best_match
            if match == "EXACT":
                s["exact"] += 1
            elif match == "NEAR":
                s["near"] += 1
            elif match == "SAME-FUNC":
                s["same_func"] += 1
            elif match == "SAME-FILE":
                s["same_file"] += 1
            elif match == "NO-OPT":
                s["no_opt"] += 1
            elif match == "UNMAPPED":
                s["unmapped"] += 1
            elif match == "NO-DATA":
                s["no_data"] += 1

            if r.topk_min_distance is not None:
                s["dist_sum"] += r.topk_min_distance
                s["dist_count"] += 1

    def fmt_count(count: int, total: int) -> str:
        pct = f"{100*count/total:.0f}%" if total > 0 else "-"
        return f"{count}/{total} ({pct})"

    header_cols = f"{'':>22} {'AMD':>14} {'NVIDIA':>14} {'Intel':>14} {'Overall':>14}"
    lines.append(header_cols)

    for label, key in [
        ("Exact (dist=0)", "exact"),
        ("Near (dist<=5)", "near"),
        ("Same-func (<=20)", "same_func"),
        ("Same-file (>20)", "same_file"),
        ("No optimization", "no_opt"),
        ("Unmapped (hex)", "unmapped"),
    ]:
        row = f"{label:>22}"
        for v in VENDORS + ["overall"]:
            s = stats[v]
            row += f" {fmt_count(s[key], s['total']):>14}"
        lines.append(row)

    # Average distance
    avg_row = f"{'Avg distance':>22}"
    for v in VENDORS + ["overall"]:
        s = stats[v]
        if s["dist_count"] > 0:
            avg = s["dist_sum"] / s["dist_count"]
            avg_row += f" {avg:>11.1f}   "
        else:
            avg_row += f" {'N/A':>14}"
    lines.append(avg_row)

    lines.append("")

    # Top-1 accuracy summary
    lines.append(f"TOP-1 ACCURACY (Leo's #1 root cause matches optimization):")
    top1_stats: dict[str, int] = {"exact": 0, "near": 0, "func": 0, "evaluable": 0}
    for r in results:
        if r.top1_match in ("EXACT", "NEAR", "SAME-FUNC", "SAME-FILE"):
            top1_stats["evaluable"] += 1
            if r.top1_match == "EXACT":
                top1_stats["exact"] += 1
            if r.top1_match in ("EXACT", "NEAR"):
                top1_stats["near"] += 1
            if r.top1_match in ("EXACT", "NEAR", "SAME-FUNC"):
                top1_stats["func"] += 1

    n = top1_stats["evaluable"]
    if n > 0:
        lines.append(f"  Exact match:        {top1_stats['exact']}/{n} ({100*top1_stats['exact']/n:.0f}%)")
        lines.append(f"  Near (<=5 lines):   {top1_stats['near']}/{n} ({100*top1_stats['near']/n:.0f}%)")
        lines.append(f"  Same function:      {top1_stats['func']}/{n} ({100*top1_stats['func']/n:.0f}%)")
    else:
        lines.append("  No evaluable results.")

    lines.append("")
    return "\n".join(lines)


def export_diffs(results: list, export_dir: Path) -> None:
    """Export per-kernel diff files for LLM semantic evaluation.

    Each file contains:
      1. Validation summary (kernel, vendor, match, distance)
      2. Leo's STALL ANALYSIS table (root cause diagnosis)
      3. Unified diff (original → optimized source)
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    for r in sorted(results, key=lambda x: (x.kernel, x.vendor)):
        if not r.has_leo_data and not r.diff_text:
            continue  # skip empty results

        filename = f"{r.kernel}__{r.vendor}.txt"
        filepath = export_dir / filename

        lines = []
        lines.append(f"{'=' * 80}")
        lines.append(f"Kernel/App: {r.kernel}")
        lines.append(f"Vendor:     {r.vendor}")
        lines.append(f"{'=' * 80}")
        lines.append("")

        # Section 1: Validation result
        lines.append("VALIDATION RESULT")
        lines.append("-" * 40)
        lines.append(f"  Top-1 match:    {r.top1_match}")
        lines.append(f"  Top-1 distance: {r.top1_distance if r.top1_distance is not None else 'N/A'}")
        lines.append(f"  Top-K match:    {r.topk_best_match}")
        lines.append(f"  Top-K distance: {r.topk_min_distance if r.topk_min_distance is not None else 'N/A'}")
        lines.append(f"  Diff file:      {r.diff_file or 'N/A'}")
        lines.append(f"  Diff lines:     {format_diff_range(r.diff_lines)}")
        if r.error:
            lines.append(f"  Error:          {r.error}")
        lines.append("")

        # Section 2: Leo root causes
        lines.append("LEO ROOT CAUSE ANALYSIS")
        lines.append("-" * 40)
        if r.leo_stall_text:
            lines.append(r.leo_stall_text)
        elif r.root_causes:
            # Fallback: format from parsed data
            for i, rc in enumerate(r.root_causes, 1):
                loc = f"{rc.root_file}:{rc.root_line}" if rc.root_file else "(hex)"
                stall = f"{rc.stall_file}:{rc.stall_line}" if rc.stall_file else "(hex)"
                lines.append(f"  #{i}: Stall at {stall} ({rc.stall_opcode})")
                lines.append(f"       Root cause: {loc} ({rc.root_opcode})")
                lines.append(f"       Cycles: {rc.cycles:,}  ({rc.pct})  Speedup: {rc.speedup}")
        else:
            lines.append("  (no Leo data)")
        lines.append("")

        # Section 3: Optimization diff
        lines.append("OPTIMIZATION DIFF (original → optimized)")
        lines.append("-" * 40)
        if r.diff_text:
            lines.append(r.diff_text)
        else:
            lines.append("  (no diff available)")
        lines.append("")

        filepath.write_text("\n".join(lines))

    print(f"Exported {len([r for r in results if r.has_leo_data or r.diff_text])} "
          f"diff files to {export_dir}/", file=sys.stderr)
