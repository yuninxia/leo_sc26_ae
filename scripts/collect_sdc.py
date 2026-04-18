#!/usr/bin/env python3
"""Collect per-kernel Single Dependency Coverage for all Table IV workloads."""

import subprocess
import re
import os

PK = "/data/per-kernel"

# Table IV RAJAPerf kernels: (display_name, directory_name)
RAJAPERF_KERNELS = [
    ("MASS3DEA",        "Apps_MASS3DEA"),
    ("LTIMES_NOVIEW",   "Apps_LTIMES_NOVIEW"),
    ("LTIMES",          "Apps_LTIMES"),
    ("3MM",             "Polybench_3MM"),
    ("2MM",             "Polybench_2MM"),
    ("GEMM",            "Polybench_GEMM"),
    ("PRESSURE",        "Apps_PRESSURE"),
    ("ENERGY",          "Apps_ENERGY"),
    ("FIR",             "Apps_FIR"),
    ("ZONAL_ACCUM_3D",  "Apps_ZONAL_ACCUMULATION_3D"),
    ("VOL3D",           "Apps_VOL3D"),
    ("DEL_DOT_VEC_2D",  "Apps_DEL_DOT_VEC_2D"),
    ("DIFFUSION3DPA",   "Apps_DIFFUSION3DPA"),
    ("CONVECTION3DPA",  "Apps_CONVECTION3DPA"),
    ("MASS3DPA",        "Apps_MASS3DPA"),
]

VENDORS = [
    ("nvidia", "nvidia-arm", "h100"),  # (display, dir_name, arch)
    ("amd",    "amd",        "mi300"),
    ("intel",  "intel",      "pvc"),
]

# HPC apps: (display_name, data_mount_name, measurements_subdir, arch, vendors_available)
HPC_APPS = [
    ("miniBUDE",     "minibude",     {"nvidia": "hpctoolkit-cuda-bude-measurements", "amd": "hpctoolkit-hip-bude-measurements", "intel": "hpctoolkit-sycl-bude-measurements"}),
    ("XSBench",      "xsbench",      {"nvidia": "hpctoolkit-XSBench-measurements", "amd": "hpctoolkit-XSBench-measurements", "intel": "hpctoolkit-XSBench-measurements"}),
    ("LULESH",       "lulesh",       {"nvidia": "hpctoolkit-lulesh2.0-measurements", "amd": "hpctoolkit-lulesh2.0-measurements", "intel": "hpctoolkit-lulesh2.0-measurements"}),
    ("QuickSilver",  "quicksilver",  {"nvidia": "hpctoolkit-qs-measurements", "amd": "hpctoolkit-qs-measurements"}),
    ("llama.cpp",    "llamacpp",     {"nvidia": "hpctoolkit-llama-bench-measurements", "amd": "hpctoolkit-llama-bench-measurements"}),
    ("Kripke",       "kripke",       {"nvidia": "hpctoolkit-kripke.exe-measurements", "amd": "hpctoolkit-kripke.exe-measurements"}),
]

SDC_RE = re.compile(r"Single Dependency Coverage:\s+([\d.]+)%\s+\(before:\s+([\d.]+)%\)")


def run_and_extract(meas_path, arch):
    """Run Leo and extract SDC from output."""
    if not os.path.isdir(meas_path):
        return None
    r = subprocess.run(
        ["uv", "run", "python", "scripts/analyze_benchmark.py", meas_path,
         "--arch", arch, "--top-n", "1"],
        capture_output=True, text=True,
    )
    output = r.stdout + r.stderr
    m = SDC_RE.search(output)
    if m:
        return (float(m.group(2)), float(m.group(1)))  # (before, after)
    return None


def main():
    # Header
    print(f"{'Kernel':<20} {'NVIDIA':>16} {'AMD':>16} {'Intel':>16}")
    print(f"{'':20} {'before→after':>16} {'before→after':>16} {'before→after':>16}")
    print("=" * 70)

    def fmt(result):
        if result is None:
            return f"{'N/A':>16}"
        before, after = result
        return f" {before:>5.1f}→{after:>5.1f}%"

    # RAJAPerf kernels
    print("RAJAPerf Kernels:")
    for display, dirname in RAJAPERF_KERNELS:
        row = f"{display:<20}"
        for v_display, v_dir, v_arch in VENDORS:
            meas = f"{PK}/{dirname}/{v_dir}/hpctoolkit-raja-perf.exe-measurements"
            row += fmt(run_and_extract(meas, v_arch))
        print(row)

    print()
    print("HPC Applications:")
    for display, mount_name, meas_map in HPC_APPS:
        row = f"{display:<20}"
        for v_display, v_dir, v_arch in VENDORS:
            if v_display not in meas_map:
                row += fmt(None)
                continue
            meas = f"/data/{v_display}-{mount_name}/{meas_map[v_display]}"
            row += fmt(run_and_extract(meas, v_arch))
        print(row)

    print("=" * 70)


if __name__ == "__main__":
    main()
