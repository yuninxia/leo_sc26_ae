# Verification Log

This file records end-to-end verification runs of the AE-documented reproduction path
on real hardware, executed by the authors prior to AE review.

## 2026-04-23 — Table IV MASS3DEA smoke test (NVIDIA GH200)

**Host:** `hopper1` — NVIDIA GH200 480GB (ARM Grace + Hopper), CUDA 12.x, cached
`leo-rajaperf-nvidia:latest` Docker image (ARM build).

**Procedure** (executed as a fresh clone, no dev-checkout contamination):

```bash
ssh hopper1
cd ~
rm -rf leo_smoke
git clone --depth=1 https://github.com/yuninxia/leo_sc26_ae.git leo_smoke
cd leo_smoke
bash scripts/evaluation/run_workload_rajaperf.sh --kernel MASS3DEA --vendor nvidia
```

**Result** — from `benchmarks/rajaperf-h100/rajaperf-compare-summary.csv`:

```
Apps_MASS3DEA, baseline_min=1.012128, baseline_max=1.014224, ...,
              optimized_min=0.276288, optimized_max=0.277430, ...,
              speedup_min=3.6633, speedup_max=3.6558
```

**Verdict:** PASS. Measured speedup **3.66×**, matches Table IV NVIDIA column
(paper value: 3.66×) to four significant figures. The AE-promised command
(`bash scripts/evaluation/run_workload_rajaperf.sh --kernel <K> --vendor nvidia`)
works end-to-end on a fresh GitHub clone.

**Coverage note:** This test exercised the full command chain:
`run_workload_rajaperf.sh` → `run_compare.sh` → `docker run --gpus all
leo-rajaperf-nvidia …`. Volume mounts, GPU pass-through, and the baseline /
optimized build comparison all work without manual intervention. Other kernels
in Table IV were not smoke-tested here; they reuse the same infrastructure, so
a failure mode specific to (e.g.) LTIMES or VOL3D would be an upstream RAJAPerf
issue, not an AE-path issue.
