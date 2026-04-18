"""Constants and mappings for ground truth validation."""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

VENDOR_SUFFIX = {"amd": "-Hip.cpp", "nvidia": "-Cuda.cpp", "intel": "-Sycl.cpp"}
VENDOR_ARCH = {"amd": "mi300", "nvidia": "h100", "intel": "pvc"}
VENDORS = ["amd", "nvidia", "intel"]

# Kernels included in Table VI of the paper.
# 17 RAJAPerf kernels + 4 real-world apps (miniBUDE, XSBench, LULESH, TK/HK).
# Kernels NOT in this list (EDGE3D, MATVEC_3D_STENCIL, FLOYD_WARSHALL) are excluded.
TABLE_VI_KERNELS = {
    # RAJAPerf benchmark kernels
    "Apps_MASS3DEA",
    "Apps_LTIMES_NOVIEW",
    "Apps_LTIMES",
    "Polybench_3MM",
    "Polybench_2MM",
    "Polybench_GEMM",
    "Apps_PRESSURE",
    "Apps_ENERGY",
    "Apps_FIR",
    "Apps_ZONAL_ACCUMULATION_3D",
    "Apps_VOL3D",
    "Apps_DEL_DOT_VEC_2D",
    "Apps_DIFFUSION3DPA",
    "Apps_CONVECTION3DPA",
    "Apps_MASS3DPA",
    "Apps_MASSVEC3DPA",
    "Apps_MASS3DPA_ATOMIC",
}

# Real-world app configurations: result dir patterns and source file paths.
# Source paths are relative to the benchmarks/ root directory.
REALWORLD_APPS = {
    "miniBUDE": {
        "result_patterns": {
            "amd": "amd-minibude-*",
            "nvidia": "nvidia-*minibude-*",
            "intel": "intel-minibude-*",
        },
        "source_files": {
            "amd": ("minibude/original/src/hip/fasten.hpp",
                    "minibude/optimized/src/hip/fasten.hpp"),
            "nvidia": ("minibude/original/src/cuda/fasten.hpp",
                       "minibude/optimized/src/cuda/fasten.hpp"),
            "intel": ("minibude/original/src/sycl/fasten.hpp",
                      "minibude/optimized/src/sycl/fasten.hpp"),
        },
    },
    "XSBench": {
        "result_patterns": {
            "amd": "amd-xsbench-*",
            "nvidia": "nvidia-*xsbench-*",
            "intel": "intel-xsbench-*",
        },
        "source_files": {
            "amd": ("xsbench/original/hip/Simulation.cpp",
                    "xsbench/optimized/hip/Simulation.cpp"),
            "nvidia": ("xsbench/original/cuda/Simulation.cu",
                       "xsbench/optimized/cuda/Simulation.cu"),
            "intel": ("xsbench/original/openmp-offload/Simulation.c",
                      "xsbench/optimized/openmp-offload/Simulation.c"),
        },
    },
    "LULESH": {
        "result_patterns": {
            "amd": "amd-lulesh-*",
            "nvidia": "nvidia-*lulesh-*",
        },
        # Optimization is in git history, not separate directories.
        "git_diff": {
            "repo": "lulesh/fork",
            "base_commit": "680e53a",
            "file": "omp_4.0/lulesh.cc",
        },
    },
    "HipKittens": {
        "result_patterns": {
            "amd": "amd-hipkittens-rmsnorm-*",
        },
        # Source files under scripts/evaluation/docker/ (use ../ from benchmarks root)
        "source_files": {
            "amd": ("../scripts/evaluation/docker/hipkittens/hipkittens_rmsnorm_bench.cpp",
                    "../scripts/evaluation/docker/hipkittens/hipkittens_rmsnorm_optimized_bench.cpp"),
        },
    },
}

# Regex for Leo stall analysis table rows
# "StallLoc  StallOp  <--  RootLoc  RootOp  Cycles  Pct  Speedup"
STALL_LINE_RE = re.compile(
    r"^(\S+)\s+(.+?)\s+<--\s+(\S+)\s+(.+?)\s+([\d,]+)\s+([\d.]+%)\s+([\d.]+x)\s*$"
)

# Extract source:line[:col] from a location string
SOURCE_LOC_RE = re.compile(r"^(.+\.\w+):(\d+)(?::(\d+))?$")

# Unified diff hunk header
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
