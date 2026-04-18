#!/usr/bin/env python3.11
"""Verify HPCToolkit + Leo pipeline on Astra's CUDA kernel.

Run inside the leo-astra-nvidia container:
  python3.11 /opt/scripts/verify_leo_profile.py
"""
import os
import subprocess
import sys
from pathlib import Path

KERNEL_CU = "/opt/astra/test/rms/rms_v1.cu"
WORK_DIR = "/tmp/leo_profile_test"
B, D, ITERS = 256, 8192, 200


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, **kwargs)
    return r.returncode


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)

    # Step 1: Compile .so with -lineinfo
    print("\n[1/5] Compiling kernel with -lineinfo...")
    script = Path(WORK_DIR) / "compile_and_test.py"
    script.write_text(f"""
import torch
from torch.utils.cpp_extension import load
ext = load("rms_li", sources=["{KERNEL_CU}"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-std=c++17",
                       "-gencode=arch=compute_90,code=sm_90"],
    verbose=True, build_directory="{WORK_DIR}")
x = torch.randn({B}, {D}, device="cuda", dtype=torch.float32)
r = torch.randn({B}, {D}, device="cuda", dtype=torch.float32)
w = torch.randn({D}, device="cuda", dtype=torch.float32)
ext.sgl_fused_add_rmsnorm(x, r, w, 1e-5, False)
torch.cuda.synchronize()
print("Compile + test OK")
""")
    if run(f"python3.11 {script}") != 0:
        sys.exit(1)

    # Step 2: Profile with hpcrun
    print("\n[2/5] Profiling with hpcrun...")
    profile_script = Path(WORK_DIR) / "profile.py"
    profile_script.write_text(f"""
import torch
from torch.utils.cpp_extension import load
ext = load("rms_li", sources=["{KERNEL_CU}"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-std=c++17",
                       "-gencode=arch=compute_90,code=sm_90"],
    verbose=False, build_directory="{WORK_DIR}")
x = torch.randn({B}, {D}, device="cuda", dtype=torch.float32)
r = torch.randn({B}, {D}, device="cuda", dtype=torch.float32)
w = torch.randn({D}, device="cuda", dtype=torch.float32)
for _ in range(20):
    ext.sgl_fused_add_rmsnorm(x.clone(), r.clone(), w.clone(), 1e-5, False)
torch.cuda.synchronize()
for i in range({ITERS}):
    ext.sgl_fused_add_rmsnorm(x.clone(), r.clone(), w.clone(), 1e-5, False)
torch.cuda.synchronize()
print(f"Profiled {{i+1}} iterations")
""")
    if run(f"hpcrun -e gpu=cuda,pc python3.11 {profile_script}") != 0:
        print("WARNING: hpcrun returned non-zero")

    # Find measurements
    meas = sorted(Path(WORK_DIR).glob("hpctoolkit-*-measurements"))
    if not meas:
        print("FAILED: no measurements directory"); sys.exit(1)
    meas_dir = meas[-1]
    print(f"  Measurements: {meas_dir}")

    # Step 3: hpcstruct
    print("\n[3/5] Running hpcstruct...")
    run(f"hpcstruct --gpucfg yes {meas_dir}")

    # Step 4: hpcprof
    print("\n[4/5] Running hpcprof...")
    run(f"hpcprof {meas_dir}")

    # Step 5: Leo analysis (via standard pipeline)
    print("\n[5/5] Running Leo analysis...")
    run(f"cd /opt/leo && uv run python scripts/analyze_benchmark.py {meas_dir} --arch h100 --top-n 2")

    print("\nDone.")


if __name__ == "__main__":
    main()
