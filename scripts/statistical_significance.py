#!/usr/bin/env python3
"""Paired t-test for GPU benchmark speedups (SC26 statistical significance).

Reads JSONL benchmark results and performs paired t-tests on (orig_ms, opt_ms)
pairs for each kernel. Reports which speedups are statistically significant.

Usage:
    uv run python scripts/statistical_significance.py benchmarks/gh200-results-*/gh200.jsonl
    uv run python scripts/statistical_significance.py benchmarks/mi300a-results-*/mi300a.jsonl
    uv run python scripts/statistical_significance.py benchmarks/pvc-results-*/pvc.jsonl
"""

import json
import sys
from collections import defaultdict
from scipy import stats


def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def analyze(records, label: str):
    groups = defaultdict(lambda: {"orig": [], "opt": []})
    for r in records:
        groups[r["kernel"]]["orig"].append(r["orig_ms"])
        groups[r["kernel"]]["opt"].append(r["opt_ms"])

    print(f"\n{'=' * 74}")
    print(f"  {label}")
    print(f"{'=' * 74}")
    print(f"{'Kernel':<30s} {'N':>3s} {'Speedup':>8s} {'t-stat':>8s} {'p-value':>10s} {'Sig?':>5s}")
    print("-" * 74)

    sig_01 = 0
    sig_05 = 0
    total = 0

    for k in sorted(groups):
        orig = groups[k]["orig"]
        opt = groups[k]["opt"]
        n = min(len(orig), len(opt))
        if n < 2:
            print(f"{k:<30s} {n:>3d}     (insufficient data)")
            continue
        total += 1
        orig = orig[:n]
        opt = opt[:n]
        mean_spd = (sum(orig) / n) / (sum(opt) / n)  # ratio of means
        t, p = stats.ttest_rel(orig, opt)
        if p < 0.01:
            sig_01 += 1
            sig = "**"
        elif p < 0.05:
            sig_05 += 1
            sig = "*"
        else:
            sig = ""
        print(f"{k:<30s} {n:>3d} {mean_spd:>8.2f}x {t:>8.2f} {p:>10.4g} {sig:>5s}")

    print("-" * 74)
    print(f"Significant at p<0.01: {sig_01}/{total}")
    print(f"Significant at p<0.05: {sig_01 + sig_05}/{total}")
    print(f"Not significant:       {total - sig_01 - sig_05}/{total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <jsonl_file> [<jsonl_file> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        records = load_jsonl(path)
        analyze(records, path)
