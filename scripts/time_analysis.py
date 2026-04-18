#!/usr/bin/env python3
"""Time Leo's analysis on all Table IV workloads. Run inside leo-base-universal container."""

import subprocess
import time
import os

WORKLOADS = [
    ("nvidia-rajaperf",    "/data/nvidia-rajaperf/hpctoolkit-raja-perf.exe-measurements",  "h100",  5),
    ("amd-rajaperf",       "/data/amd-rajaperf/hpctoolkit-raja-perf.exe-measurements",     "mi300", 5),
    ("intel-rajaperf",     "/data/intel-rajaperf/hpctoolkit-raja-perf.exe-measurements",    "pvc",   5),  # uses per-kernel data
    ("nvidia-minibude",    "/data/nvidia-minibude/hpctoolkit-cuda-bude-measurements",       "h100",  2),
    ("amd-minibude",       "/data/amd-minibude/hpctoolkit-hip-bude-measurements",           "mi300", 2),
    ("intel-minibude",     "/data/intel-minibude/hpctoolkit-sycl-bude-measurements",        "pvc",   2),
    ("nvidia-xsbench",     "/data/nvidia-xsbench/hpctoolkit-XSBench-measurements",          "h100",  2),
    ("amd-xsbench",        "/data/amd-xsbench/hpctoolkit-XSBench-measurements",             "mi300", 2),
    ("intel-xsbench",      "/data/intel-xsbench/hpctoolkit-XSBench-measurements",           "pvc",   2),
    ("nvidia-lulesh",      "/data/nvidia-lulesh/hpctoolkit-lulesh2.0-measurements",         "h100",  2),
    ("amd-lulesh",         "/data/amd-lulesh/hpctoolkit-lulesh2.0-measurements",            "mi300", 2),
    ("intel-lulesh",       "/data/intel-lulesh/hpctoolkit-lulesh2.0-measurements",          "pvc",   2),
    ("nvidia-quicksilver", "/data/nvidia-quicksilver/hpctoolkit-qs-measurements",            "h100",  2),
    ("amd-quicksilver",    "/data/amd-quicksilver/hpctoolkit-qs-measurements",               "mi300", 2),
    ("nvidia-llamacpp",    "/data/nvidia-llamacpp/hpctoolkit-llama-bench-measurements",      "h100",  2),
    ("amd-llamacpp",       "/data/amd-llamacpp/hpctoolkit-llama-bench-measurements",         "mi300", 2),
    ("nvidia-kripke",      "/data/nvidia-kripke/hpctoolkit-kripke.exe-measurements",         "h100",  2),
    ("amd-kripke",         "/data/amd-kripke/hpctoolkit-kripke.exe-measurements",            "mi300", 2),
]

def main():
    # Warmup: import Leo once
    print("Warming up Python imports...")
    t0 = time.time()
    subprocess.run(
        ["uv", "run", "python", "-c", "from leo.analysis.program_analysis import analyze_program"],
        capture_output=True,
    )
    print(f"Import warmup: {time.time() - t0:.1f}s\n")

    print(f"{'Workload':<25} {'Arch':<8} {'Top-N':<6} {'Time':>8}")
    print("-" * 50)

    for name, meas, arch, topn in WORKLOADS:
        if not os.path.isdir(meas):
            print(f"{name:<25} {arch:<8} {topn:<6} {'SKIP':>8}")
            continue

        t0 = time.time()
        r = subprocess.run(
            ["uv", "run", "python", "scripts/analyze_benchmark.py", meas,
             "--arch", arch, "--top-n", str(topn)],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        status = f"{elapsed:.1f}s" if r.returncode == 0 else f"FAIL ({elapsed:.1f}s)"
        print(f"{name:<25} {arch:<8} {topn:<6} {status:>8}")

    print("-" * 50)

if __name__ == "__main__":
    main()
