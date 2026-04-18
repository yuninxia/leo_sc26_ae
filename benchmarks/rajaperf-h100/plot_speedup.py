#!/usr/bin/env python3
"""Plot A100 speedup bar chart with IQR-based error bars from raw benchmark CSV."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_and_process(csv_path: str, drop_passes: int = 1):
    """Load raw CSV, drop cold-start passes, compute per-kernel speedup distribution."""
    # Collect per-(kernel, version, run) timing samples
    raw = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["pass"]) <= drop_passes:
                continue
            key = (row["kernel"], row["version"])
            run = int(row["run"])
            raw[key][run].append(float(row["time_ms"]))

    kernels = sorted(set(k for (k, v) in raw.keys()))
    results = []
    for kernel in kernels:
        orig_runs = raw.get((kernel, "original"), {})
        opt_runs = raw.get((kernel, "optimized"), {})
        if not orig_runs or not opt_runs:
            continue

        # Compute per-run median, then speedup = orig_median / opt_median per run
        orig_all = []
        opt_all = []
        for r in orig_runs.values():
            orig_all.extend(r)
        for r in opt_runs.values():
            opt_all.extend(r)

        orig_med = statistics.median(orig_all)
        opt_med = statistics.median(opt_all)
        speedup = orig_med / opt_med if opt_med > 0 else 1.0

        # IQR of speedup via bootstrap: per-run medians
        run_speedups = []
        run_ids = sorted(set(orig_runs.keys()) & set(opt_runs.keys()))
        for rid in run_ids:
            om = statistics.median(orig_runs[rid])
            optm = statistics.median(opt_runs[rid])
            if optm > 0:
                run_speedups.append(om / optm)

        if len(run_speedups) >= 3:
            s = sorted(run_speedups)
            q1 = s[len(s) // 4]
            q3 = s[3 * len(s) // 4]
        else:
            q1 = q3 = speedup

        # Short display name
        name = kernel.replace("Apps_", "").replace("Polybench_", "")

        results.append({
            "kernel": kernel,
            "name": name,
            "speedup": speedup,
            "q1": q1,
            "q3": q3,
        })

    return results


def plot(results, output_path: str):
    # Sort by speedup descending
    results.sort(key=lambda r: r["speedup"], reverse=True)

    names = [r["name"] for r in results]
    speedups = [r["speedup"] for r in results]
    q1s = [r["q1"] for r in results]
    q3s = [r["q3"] for r in results]

    # Error bars: asymmetric [speedup - q1, q3 - speedup]
    err_lo = [max(0, s - q) for s, q in zip(speedups, q1s)]
    err_hi = [max(0, q - s) for s, q in zip(speedups, q3s)]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    x = np.arange(len(names))
    colors = ["#4CAF50" if s >= 1.05 else "#9E9E9E" if s >= 0.95 else "#F44336"
              for s in speedups]

    bars = ax.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=[err_lo, err_hi], capsize=3, error_kw={"linewidth": 1, "color": "#333"})

    # Reference line at 1.0
    ax.axhline(y=1.0, color="#888", linestyle="--", linewidth=0.8, zorder=0)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Speedup (median)", fontsize=11)
    ax.set_title("NVIDIA A100: Optimization Speedup per Kernel (median, drop cold-start pass 1)", fontsize=12)

    # Value labels on bars
    for i, (xi, s) in enumerate(zip(x, speedups)):
        ax.text(xi, s + err_hi[i] + 0.04, f"{s:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0, max(speedups) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to rajaperf-compare-raw.csv")
    parser.add_argument("--output", "-o", default="a100_speedup.png")
    args = parser.parse_args()

    results = load_and_process(args.csv_file)
    plot(results, args.output)


if __name__ == "__main__":
    main()
