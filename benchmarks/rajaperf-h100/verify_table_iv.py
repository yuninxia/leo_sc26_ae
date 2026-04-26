#!/usr/bin/env python3
"""verify_table_iv.py — per-kernel PASS/FAIL tolerance check for Table IV.

Compares a reviewer-produced rajaperf-compare-summary.csv (output of
run_compare.sh / runme.sh --with-table-iv) against the paper's reference
rajaperf-compare-summary-reference.csv (committed in this directory),
and prints PASS / FAIL per kernel based on the appendix-declared
tolerance window:

    NVIDIA / AMD: +/- 5%
    Intel PVC:    +/- 9%

Drift formula:
    drift = (reviewer_speedup_mean - reference_speedup_mean) / reference_speedup_mean

Pass condition: |drift| <= tolerance.

Exits 0 if every kernel passes; exit 1 if any kernel exceeds the
tolerance window (so runme.sh can detect and surface the failure).

Usage:
    python verify_table_iv.py                                          # auto, NVIDIA tolerance, default file paths
    python verify_table_iv.py --vendor amd                             # AMD tolerance (+/-5%)
    python verify_table_iv.py --vendor intel                           # Intel tolerance (+/-9%)
    python verify_table_iv.py --reviewer rajaperf-compare-summary.csv \\
                              --reference rajaperf-compare-summary-reference.csv

Note on hardware mismatch: this script assumes reviewer's GPU is the
same family as the paper-reference hardware. If reviewer is on H100/
A100/A10 instead of paper's GH200, some kernels (particularly memory-
bound ones such as MASS3DEA) may exceed +/- 5% due to architectural
differences (Hopper PCIe vs. Grace-Hopper coherent memory). Such
failures are documented as expected drift in the appendix's Known
Deviations section, not artifact-pipeline regressions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

VENDOR_TOLERANCE = {
    "nvidia": 0.05,
    "amd": 0.05,
    "intel": 0.09,
}


def load_summary(path: Path) -> dict[str, float]:
    """Read a rajaperf-compare-summary.csv and return {kernel -> speedup_mean}."""
    if not path.exists():
        sys.stderr.write(f"ERROR: file not found: {path}\n")
        sys.exit(2)
    out: dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        if "kernel" not in (reader.fieldnames or []) or "speedup_mean" not in (reader.fieldnames or []):
            sys.stderr.write(
                f"ERROR: {path} missing required columns 'kernel' and 'speedup_mean'. "
                f"Got: {reader.fieldnames}\n"
            )
            sys.exit(2)
        for row in reader:
            out[row["kernel"]] = float(row["speedup_mean"])
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reviewer", default=str(here / "rajaperf-compare-summary.csv"),
                   help="Reviewer-produced summary CSV (default: ./rajaperf-compare-summary.csv)")
    p.add_argument("--reference", default=str(here / "rajaperf-compare-summary-reference.csv"),
                   help="Committed paper reference CSV (default: ./rajaperf-compare-summary-reference.csv)")
    p.add_argument("--vendor", choices=sorted(VENDOR_TOLERANCE), default="nvidia",
                   help="GPU vendor; sets tolerance window (default: nvidia, +/-5%%)")
    p.add_argument("--tolerance", type=float, default=None,
                   help="Override tolerance fraction (e.g., 0.10 for +/-10%%); default by --vendor")
    args = p.parse_args()

    tol = args.tolerance if args.tolerance is not None else VENDOR_TOLERANCE[args.vendor]

    reviewer_path = Path(args.reviewer)
    reference_path = Path(args.reference)
    reviewer = load_summary(reviewer_path)
    reference = load_summary(reference_path)

    print("=" * 86)
    print(f" Table IV verification ({args.vendor.upper()}, tolerance +/-{tol*100:.0f}%)")
    print(f"   reviewer:  {reviewer_path}")
    print(f"   reference: {reference_path}")
    print("=" * 86)
    print(f"{'Kernel':<32} {'Reference':>12} {'Reviewer':>12} {'Drift':>10}  {'Verdict':<8}")
    print("-" * 86)

    fails: list[tuple[str, float]] = []
    missing_in_reviewer = []
    missing_in_reference = []

    for kernel in sorted(reference):
        ref_speedup = reference[kernel]
        if kernel not in reviewer:
            missing_in_reviewer.append(kernel)
            print(f"{kernel:<32} {ref_speedup:>11.4f}x {'MISSING':>12} {'--':>10}  MISSING")
            continue
        rev_speedup = reviewer[kernel]
        if ref_speedup == 0:
            verdict = "FAIL"
            drift = float("nan")
        else:
            drift = (rev_speedup - ref_speedup) / ref_speedup
            verdict = "PASS" if abs(drift) <= tol else "FAIL"
        if verdict == "FAIL":
            fails.append((kernel, drift))
        drift_str = f"{drift*100:+.2f}%" if drift == drift else "nan"
        print(f"{kernel:<32} {ref_speedup:>11.4f}x {rev_speedup:>11.4f}x {drift_str:>10}  {verdict:<8}")

    for kernel in sorted(reviewer):
        if kernel not in reference:
            missing_in_reference.append(kernel)
            print(f"{kernel:<32} {'MISSING':>12} {reviewer[kernel]:>11.4f}x {'--':>10}  EXTRA")

    print("-" * 86)
    n_total = len(reference)
    n_pass = n_total - len(fails) - len(missing_in_reviewer)
    print(f"{'Summary':<32} {n_pass}/{n_total} kernels passed within +/-{tol*100:.0f}%")
    if fails:
        print(f"   FAIL ({len(fails)}): {', '.join(k for k, _ in fails)}")
    if missing_in_reviewer:
        print(f"   MISSING in reviewer output ({len(missing_in_reviewer)}): {', '.join(missing_in_reviewer)}")
    if missing_in_reference:
        print(f"   EXTRA in reviewer (not in reference) ({len(missing_in_reference)}): {', '.join(missing_in_reference)}")

    if not fails and not missing_in_reviewer:
        print(f"\nPASS: {n_total}/{n_total} kernels within +/-{tol*100:.0f}% tolerance")
        return 0
    print(f"\nFAIL: {len(fails)} kernel(s) outside tolerance" +
          (f"; {len(missing_in_reviewer)} missing" if missing_in_reviewer else ""))
    print("Note: hardware drift on non-paper GPUs (e.g., H100 vs paper GH200) may cause")
    print("expected failures on memory-bound kernels (MASS3DEA, etc.); see appendix Known Deviations.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
