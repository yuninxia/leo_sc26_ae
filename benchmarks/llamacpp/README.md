# llama.cpp (Section VI case study)

Leo identifies the `mul_mat_q` quantized matrix-multiply kernel in
`ggml/src/ggml-cuda/mmq.cuh` as the bottleneck on both AMD MI300A and
NVIDIA H100, with **different root causes** per vendor. The optimization
lives in vendor-specific patches.

## Upstream

- Source: https://github.com/ggml-org/llama.cpp
- Baseline used in the paper: upstream `master` at the time of submission.
  The relevant file is `ggml/src/ggml-cuda/mmq.cuh`.

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/llamacpp-original
# then locate ggml/src/ggml-cuda/mmq.cuh
```

## What's in this directory

- `optimized/amd_mmq_optimize.patch` — AMD path: reduce tile width, direct store
- `optimized/nvidia_mmq_optimize.patch` — NVIDIA path: direct store at
  stream-k fixup
- `optimized/mmq_original.cuh` / `mmq_amd_optimized.cuh` — before/after
  snapshots of the modified sections, for easy side-by-side diff
- `optimized/leo_output_{amd,nvidia}_1.5b.txt` — Leo analysis output
- `optimized/README.md` — full diagnosis and rationale
- `run_compare_nvidia.sh` / `run_perkernel_amd.sh` — GPU run harnesses

## Diffing the optimization

Apply the appropriate patch to an upstream clone:
```bash
cd /tmp/llamacpp-original
git apply /path/to/leo_sc26_ae/benchmarks/llamacpp/optimized/amd_mmq_optimize.patch
# or nvidia_mmq_optimize.patch
```

Alternatively, `diff optimized/mmq_original.cuh optimized/mmq_amd_optimized.cuh`
shows the AMD-path changes as a self-contained before/after pair.
