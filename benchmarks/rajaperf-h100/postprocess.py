#!/usr/bin/env python3
"""Post-process RAJAPerf benchmark raw CSV: drop cold-start passes, compute robust statistics.

Each invocation of the RAJAPerf binary has a cold-start first pass (CUDA context init,
memory allocation, first-touch) that is 10-50x slower than steady state. This script
drops the first N passes from each invocation and computes median-based speedup.

Usage:
    python postprocess.py rajaperf-compare-raw.csv
    python postprocess.py rajaperf-compare-raw.csv --drop-passes 2
    python postprocess.py rajaperf-compare-raw.csv --output summary.csv
    python postprocess.py rajaperf-compare-raw.csv --latex
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


def load_raw_csv(path: str, drop_passes: int = 1):
    """Load raw CSV and drop the first `drop_passes` passes from each (kernel, version, run)."""
    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["pass"]) <= drop_passes:
                continue
            key = (row["kernel"], row["version"])
            data[key].append(float(row["time_ms"]))
    return data


def compute_stats(times: list[float]) -> dict:
    """Compute robust statistics for a list of timing samples."""
    n = len(times)
    if n == 0:
        return {"n": 0, "median": 0, "mean": 0, "min": 0, "p25": 0, "p75": 0, "iqr": 0}
    s = sorted(times)
    med = statistics.median(s)
    mean = statistics.mean(s)
    mn = s[0]
    p25 = s[n // 4] if n >= 4 else s[0]
    p75 = s[3 * n // 4] if n >= 4 else s[-1]
    return {"n": n, "median": med, "mean": mean, "min": mn, "p25": p25, "p75": p75, "iqr": p75 - p25}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", help="Path to rajaperf-compare-raw.csv")
    parser.add_argument("--drop-passes", type=int, default=1,
                        help="Number of initial passes to drop per invocation (default: 1)")
    parser.add_argument("--output", "-o", help="Write summary CSV to this path")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX-formatted table rows")
    args = parser.parse_args()

    data = load_raw_csv(args.csv_file, args.drop_passes)

    # Gather all kernels
    kernels = sorted(set(k for (k, v) in data.keys()))

    results = []
    for kernel in kernels:
        orig = data.get((kernel, "original"), [])
        opt = data.get((kernel, "optimized"), [])
        if not orig or not opt:
            continue
        orig_stats = compute_stats(orig)
        opt_stats = compute_stats(opt)
        speedup_med = orig_stats["median"] / opt_stats["median"] if opt_stats["median"] > 0 else 0
        speedup_min = orig_stats["min"] / opt_stats["min"] if opt_stats["min"] > 0 else 0
        results.append({
            "kernel": kernel,
            "orig": orig_stats,
            "opt": opt_stats,
            "speedup_median": speedup_med,
            "speedup_min": speedup_min,
        })

    # Print table
    hdr = (f"{'Kernel':<35} {'Orig med':>10} {'IQR%':>6} {'Opt med':>10} {'IQR%':>6} "
           f"{'Spd(med)':>10} {'Spd(min)':>10} {'n':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        orig_iqr_pct = r['orig']['iqr'] / r['orig']['median'] * 100 if r['orig']['median'] > 0 else 0
        opt_iqr_pct = r['opt']['iqr'] / r['opt']['median'] * 100 if r['opt']['median'] > 0 else 0
        print(f"{r['kernel']:<35} {r['orig']['median']:>10.4f} {orig_iqr_pct:>5.1f}% "
              f"{r['opt']['median']:>10.4f} {opt_iqr_pct:>5.1f}% "
              f"{r['speedup_median']:>9.2f}x {r['speedup_min']:>9.2f}x {r['orig']['n']:>5}")

    # Summary CSV
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel",
                        "orig_median_ms", "orig_min_ms", "orig_iqr_ms", "orig_iqr_pct", "orig_n",
                        "opt_median_ms", "opt_min_ms", "opt_iqr_ms", "opt_iqr_pct", "opt_n",
                        "speedup_median", "speedup_min"])
            for r in results:
                orig_iqr_pct = r['orig']['iqr'] / r['orig']['median'] * 100 if r['orig']['median'] > 0 else 0
                opt_iqr_pct = r['opt']['iqr'] / r['opt']['median'] * 100 if r['opt']['median'] > 0 else 0
                w.writerow([r["kernel"],
                            f"{r['orig']['median']:.6f}", f"{r['orig']['min']:.6f}",
                            f"{r['orig']['iqr']:.6f}", f"{orig_iqr_pct:.1f}", r["orig"]["n"],
                            f"{r['opt']['median']:.6f}", f"{r['opt']['min']:.6f}",
                            f"{r['opt']['iqr']:.6f}", f"{opt_iqr_pct:.1f}", r["opt"]["n"],
                            f"{r['speedup_median']:.4f}", f"{r['speedup_min']:.4f}"])
        print(f"\nSummary CSV written to: {out_path}")

    # LaTeX output
    if args.latex:
        print("\n% LaTeX table rows (speedup_median):")
        for r in results:
            s = r["speedup_median"]
            name = r["kernel"].replace("Apps_", "").replace("Polybench_", "")
            if s >= 1.05:
                cell = f"\\cellcolor{{nvgreen!{min(int(s * 15), 60)}}}{s:.2f}$\\times$"
            elif s <= 0.95:
                cell = f"{s:.2f}$\\times$"
            else:
                cell = f"{s:.2f}$\\times$"
            print(f"% {name:<30} & {cell} \\\\")


if __name__ == "__main__":
    main()
