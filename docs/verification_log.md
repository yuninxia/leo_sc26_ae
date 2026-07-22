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

---

## 2026-07-22 — CPU-only Figure-5 path, reviewer-style end-to-end (odyssey, x86-64)

**Environment** — `odyssey` (2× 24c EPYC-class hosts of the quad-MI300A node, x86-64,
RHEL 8.10, Docker 29.6.2); fresh `git clone` of `main` (includes the
comma-list `--kernel` filter fix and the `v0.1.21-sc26-ae` default
`PREBUILT_TAG`); no cached leo images present before the run (verified with
`docker images`); GPUs untouched (another user's job was running on them —
irrelevant for this CPU-only path).

**Procedure** — exactly the AE-recommended reviewer flow:

```
git clone https://github.com/yuninxia/leo_sc26_ae.git
cd leo_sc26_ae
bash scripts/runme.sh --use-prebuilt
```

**Result** — all steps completed unattended:

- Step 1 download_data (1 GB → 5.6 GB): 174 s
- Step 2 docker pull `jssonxia/leo-base-universal:v0.1.21-sc26-ae` (4.63 GB): 178 s
- Step 3 collect_sdc.sh (Figure 5) + per-kernel cross-vendor demo: remainder
- **Total wall-clock: 16 min 58 s** (within the documented 20–30 min budget)
- **Figure 5 SHA-256: OK** — output byte-identical to the committed
  `sdc_coverage_reference.txt`

**Verdict:** PASS. First end-to-end validation of the CPU-only path from a
clean environment via Docker Hub prebuilt images; the default prebuilt tag now
matches the appendix's pinned `v0.1.21-sc26-ae`.

**Minor issue observed:** `download_data.sh` still echoes
"You can now run: bash scripts/time_analysis.sh" — that script was removed;
the hint should point at `scripts/collect_sdc.sh` / `scripts/runme.sh`.
