#!/usr/bin/env python3
"""Benchmark Astra-optimized kernels against sgl-kernel baseline.

Loads the best kernel from an Astra optimization run and benchmarks it
head-to-head against the sgl-kernel reference implementation using the
same methodology: CUDA Events timing with 20 warmup + 100 timed iterations.

Usage (inside the leo-astra-nvidia container):
  # After running cuda_kernel_optimizer_multi.py, find the best kernel .cu file
  # in the run directory, then:

  python3.11 /opt/scripts/benchmark_astra.py \
      --kernel /opt/astra/cuda_optimization_runs/run_.../kernels/...v5.cu \
      --kind rmsnorm

  # Or compare multiple versions at once:
  python3.11 /opt/scripts/benchmark_astra.py \
      --kernel best_v5.cu --kernel initial_v1.cu \
      --kind rmsnorm

  # Or point to a run directory and auto-discover best kernel:
  python3.11 /opt/scripts/benchmark_astra.py \
      --run-dir /opt/astra/cuda_optimization_runs/run_2026-02-27_21-39-23 \
      --kind rmsnorm
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# Benchmark shapes (same as Astra's rmsnorm_test.py)
# ---------------------------------------------------------------------------
SHAPES = {
    "rmsnorm": [
        (128, 4096),
        (256, 4096),
        (1024, 4096),
        (2048, 8192),
        (128, 11008),
        (256, 13824),
        (512, 14336),
        (1024, 8192),
    ],
    "silu": [
        (128, 4096),
        (256, 4096),
        (1024, 4096),
        (2048, 8192),
        (128, 11008),
        (256, 13824),
        (512, 14336),
        (1024, 8192),
    ],
    "mergestate": [
        (1, 64, 128),
        (4, 64, 128),
        (16, 64, 128),
        (32, 64, 128),
    ],
}

# Baseline function names per kernel kind
BASELINE_FUNCS = {
    "rmsnorm": "sgl_fused_add_rmsnorm",
    "silu": "sgl_silu_mul",
    "mergestate": "merge_state",
}

EXPORT_FUNCS = {
    "rmsnorm": "sgl_fused_add_rmsnorm",
    "silu": "sgl_silu_mul",
    "mergestate": "merge_state",
}


# ---------------------------------------------------------------------------
# CUDA Events benchmarking (matches Astra's methodology exactly)
# ---------------------------------------------------------------------------
def benchmark_fn(fn, make_args, warmup=20, iters=100):
    """Time a CUDA callable. Returns (mean_ms, std_ms, min_ms, max_ms)."""
    # Warmup
    for _ in range(warmup):
        args = make_args()
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        args = make_args()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


# ---------------------------------------------------------------------------
# Kernel loaders
# ---------------------------------------------------------------------------
def load_sgl_baseline(kind):
    """Load sgl-kernel baseline function."""
    import sgl_kernel as sglk
    func_name = BASELINE_FUNCS[kind]
    if hasattr(sglk, func_name):
        return getattr(sglk, func_name)
    # Try without sgl_ prefix (sgl_kernel uses e.g. fused_add_rmsnorm, silu_and_mul)
    alt_name = func_name.replace("sgl_", "")
    if hasattr(sglk, alt_name):
        return getattr(sglk, alt_name)
    raise AttributeError(
        f"sgl_kernel has no '{func_name}' or '{alt_name}'")


def load_kernel_from_cu(cu_path, export_func, label="opt"):
    """JIT-compile a .cu file and return the exported function."""
    cu_path = str(cu_path)
    module_name = f"bench_{label}_{Path(cu_path).stem}"
    ext = load(
        name=module_name,
        sources=[cu_path],
        extra_cuda_cflags=["-O2", "--use_fast_math", "-std=c++17",
                           "-gencode=arch=compute_90,code=sm_90"],
        verbose=True,
        build_directory=os.getcwd(),
    )
    if not hasattr(ext, export_func):
        avail = [a for a in dir(ext) if not a.startswith("_")]
        raise AttributeError(
            f"Module has no '{export_func}'. Available: {avail}")
    return getattr(ext, export_func)


# ---------------------------------------------------------------------------
# Per-kind argument generators
# ---------------------------------------------------------------------------
def make_rmsnorm_args(B, D, device="cuda", eps=1e-5):
    def gen():
        x = torch.randn(B, D, device=device, dtype=torch.float32)
        r = torch.randn(B, D, device=device, dtype=torch.float32)
        w = torch.randn(D, device=device, dtype=torch.float32)
        return (x, r, w, eps, False)
    return gen


def make_silu_args(B, D, device="cuda"):
    def gen():
        x = torch.randn(B, D, device=device, dtype=torch.float32)
        return (x,)
    return gen


def make_mergestate_args(B, N, D, device="cuda"):
    def gen():
        s = torch.randn(B, N, D, device=device, dtype=torch.float32)
        return (s,)
    return gen


# ---------------------------------------------------------------------------
# Auto-discover best kernel from run directory
# ---------------------------------------------------------------------------
def find_best_kernel(run_dir):
    """Find the best kernel .cu file from an Astra run directory."""
    run_dir = Path(run_dir)

    # Check metadata.json for best_version
    meta = run_dir / "metadata.json"
    if meta.exists():
        with open(meta) as f:
            data = json.load(f)
        best = data.get("best_version")
        if best:
            # Search in kernels/ subdirectory
            kernels_dir = run_dir / "kernels"
            if kernels_dir.exists():
                for cu in kernels_dir.glob(f"*{best}*.cu"):
                    return cu
            # Also search in the run_dir itself
            for cu in run_dir.glob(f"*{best}*.cu"):
                return cu

    # Fallback: find highest-numbered .cu file
    cu_files = sorted(run_dir.glob("kernels/*.cu")) or sorted(run_dir.glob("*.cu"))
    if cu_files:
        return cu_files[-1]
    return None


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def check_correctness_rmsnorm(baseline_fn, opt_fn, B=128, D=4096):
    """Quick correctness check for rmsnorm."""
    device = "cuda"
    x_b = torch.randn(B, D, device=device, dtype=torch.float32)
    r_b = torch.randn(B, D, device=device, dtype=torch.float32)
    w = torch.randn(D, device=device, dtype=torch.float32)
    x_o, r_o = x_b.clone(), r_b.clone()

    baseline_fn(x_b, r_b, w.clone(), 1e-5, False)
    opt_fn(x_o, r_o, w.clone(), 1e-5, False)

    diff = (x_b - x_o).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (x_b.abs() + 1e-8)).max().item()
    ok = torch.allclose(x_b, x_o, rtol=1e-4, atol=1e-5)
    return ok, max_abs, max_rel


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(kind, kernel_paths, run_dir=None, csv_out=None):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = "cuda"

    # Resolve kernel paths
    if not kernel_paths and run_dir:
        best = find_best_kernel(run_dir)
        if best:
            kernel_paths = [str(best)]
            print(f"Auto-discovered best kernel: {best}")
        else:
            print(f"ERROR: No .cu files found in {run_dir}")
            sys.exit(1)

    if not kernel_paths:
        print("ERROR: No kernel files specified. Use --kernel or --run-dir.")
        sys.exit(1)

    export_func = EXPORT_FUNCS[kind]
    shapes = SHAPES[kind]

    # Load baseline
    print(f"\n{'='*70}")
    print(f" Astra Benchmark: {kind} (sgl-kernel vs optimized)")
    print(f"{'='*70}")
    print(f"\nLoading sgl-kernel baseline ({BASELINE_FUNCS[kind]})...")
    baseline_fn = load_sgl_baseline(kind)

    # Load optimized kernels
    opt_fns = {}
    for i, kp in enumerate(kernel_paths):
        label = Path(kp).stem
        print(f"Loading optimized kernel: {label} ...")
        opt_fns[label] = load_kernel_from_cu(kp, export_func, label=f"k{i}")

    # Correctness check
    if kind == "rmsnorm":
        print(f"\n--- Correctness Check ---")
        for label, fn in opt_fns.items():
            ok, max_abs, max_rel = check_correctness_rmsnorm(baseline_fn, fn)
            status = "PASS" if ok else "FAIL"
            print(f"  {label}: {status}  (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})")

    # Benchmark
    print(f"\n--- Performance (20 warmup + 100 timed, CUDA Events) ---")
    print(f"{'Shape':>16s}  {'sgl-kernel':>12s}", end="")
    for label in opt_fns:
        short = label[:20]
        print(f"  {short:>12s}  {'Speedup':>8s}", end="")
    print()
    print("-" * (18 + 14 + len(opt_fns) * 24))

    results = []
    for shape in shapes:
        if kind == "rmsnorm":
            B, D = shape
            shape_str = f"{B}x{D}"
            make_args = make_rmsnorm_args(B, D)
        elif kind == "silu":
            B, D = shape
            shape_str = f"{B}x{D}"
            make_args = make_silu_args(B, D)
        else:
            B, N, D = shape
            shape_str = f"{B}x{N}x{D}"
            make_args = make_mergestate_args(B, N, D)

        # Benchmark baseline
        base_mean, base_std, _, _ = benchmark_fn(baseline_fn, make_args)

        row = {"shape": shape_str, "sgl_ms": base_mean}
        print(f"{shape_str:>16s}  {base_mean:>10.4f}ms", end="")

        for label, fn in opt_fns.items():
            opt_mean, opt_std, _, _ = benchmark_fn(fn, make_args)
            speedup = base_mean / opt_mean if opt_mean > 0 else float("nan")
            row[f"{label}_ms"] = opt_mean
            row[f"{label}_speedup"] = speedup
            short = label[:20]
            print(f"  {opt_mean:>10.4f}ms  {speedup:>7.2f}x", end="")

        print()
        results.append(row)

    # Summary
    print("-" * (18 + 14 + len(opt_fns) * 24))
    avg_base = np.mean([r["sgl_ms"] for r in results])
    print(f"{'AVG':>16s}  {avg_base:>10.4f}ms", end="")
    for label in opt_fns:
        avg_opt = np.mean([r[f"{label}_ms"] for r in results])
        avg_spd = np.mean([r[f"{label}_speedup"] for r in results])
        short = label[:20]
        print(f"  {avg_opt:>10.4f}ms  {avg_spd:>7.2f}x", end="")
    print()

    # CSV output
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to {csv_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Astra-optimized kernels vs sgl-kernel baseline")
    parser.add_argument("--kernel", action="append", default=[],
                        help="Path to optimized .cu file (can be repeated)")
    parser.add_argument("--run-dir",
                        help="Astra run directory (auto-discovers best kernel)")
    parser.add_argument("--kind", default="rmsnorm",
                        choices=["rmsnorm", "silu", "mergestate"],
                        help="Kernel type (default: rmsnorm)")
    parser.add_argument("--csv", default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    run_benchmark(args.kind, args.kernel, args.run_dir, args.csv)


if __name__ == "__main__":
    main()
