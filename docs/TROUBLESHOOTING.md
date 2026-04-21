# Troubleshooting

Common issues when reproducing LEO results.

## Build Issues

### `uv sync` fails compiling `hpcanalysis` (OpenMP `shared` errors)

**Symptom:** `error: 'profiles_count' is predetermined 'shared' for 'shared'` during the `hpcanalysis` C++ build.

**Cause:** `hpcanalysis` requires GCC 11 or newer; on RHEL 8 / CentOS 8 the default `gcc` is 8.5.

**Fix:** Activate a newer toolchain before running `uv sync`:
```bash
# RHEL/CentOS 8 with gcc-toolset-11 or newer installed via dnf
scl enable gcc-toolset-11 -- uv sync
# or for Ubuntu 22.04+: the default GCC 11+ works — just `uv sync`
```

Verify the compiler is recent enough: `gcc --version` should report 11.x or newer.

### hpcstruct segfaults (exit 139) on Intel

**Cause:** Debug TBB libraries linked into hpcstruct.

**Fix:** Rebuild HPCToolkit with `meson setup build --buildtype=release`.

See: `docs/INTEL_EVAL_TROUBLESHOOTING.md` in the main LEO repo for full analysis.

### GED library not found (Intel)

LEO's Intel disassembler needs `libged.so` from GTPin.

**Fix:** Set env var explicitly:
```bash
export GED_LIBRARY_PATH=/path/to/hpctoolkit/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so
```

Or install GTPin system-wide at `/opt/intel/oneapi/gtpin/latest/`.

### Docker build failures

Each vendor's base image builds HPCToolkit from source, which takes 30-60 minutes.
If build fails:

1. Verify CUDA/ROCm/Level Zero toolkits installed in the build context
2. Check Dockerfile for pinned versions matching your host
3. Consult `scripts/evaluation/docker/Dockerfile.base-<vendor>` comments

## Runtime Issues

### hpcrun produces empty PC samples on Intel

**Cause:** Missing `ZE_ENABLE_TRACING_LAYER=1`.

**Fix:** Export before running:
```bash
export ZE_ENABLE_TRACING_LAYER=1
hpcrun -e gpu=level0 ...
```

### "Module load failed" for lgkmcnt/vmcnt on AMD

**Cause:** PC sampling stochastic mode unavailable (needs MI300+).

**Fix:** Verify GPU supports stochastic sampling:
```bash
rocprofiler-sdk info --stochastic
```

On older AMD GPUs, use host-trap mode (higher overhead).

### NVIDIA Activity API serializes kernels

This is expected behavior of the Activity API (documented in the paper, Section II). The serialization affects concurrency-sensitive benchmarks; we report total kernel time, not wall-clock application time.

## Analysis Issues

### LEO reports "No CFG for function X"

**Cause:** Function's binary section not found by disassembler.

**Fix:** Verify the GPU binary (cubin/hsaco/zebin) is present in the HPCToolkit measurement directory under `gpubins/`.

### SDC coverage much lower than paper

**Cause:** Running LEO without `--enable-sync-tracing` (flag enabled by default in current build).

**Verify:** Check `assign_pcs` counts in LEO output include s_waitcnt→global_load (AMD) or Control.wait→Control.write (NVIDIA) edges.

## Benchmark Variance

### Intel PVC results differ by >5% across runs

**Expected.** Intel PVC shows median CV of ~9% across runs due to platform-specific scheduling (documented in paper Section V). Run 10 times and report median (as the paper does).

### RAJAPerf speedups near 1.0× seem significant vs paper

**Check:**
1. Build type: release vs debug (must be release)
2. Kernel repetition count: RAJAPerf uses internal warmup + specific rep counts
3. Per-kernel GPU time (nsys/rocprofv3/unitrace), not wall-clock

## LLM Reproduction (Table V)

### Gemini API rate limits

Gemini 3.1 Pro (via OpenRouter) has per-minute QPS limits. Reduce concurrency via `--llm-concurrency 2` on `python -m scripts.validation.main --llm-eval`.

### LLM outputs vary across runs

**Expected.** The paper uses k=5 trials and reports best-of-k. Overall geomean (1.29× for C+L(S)) and 100% compile rate should hold, but per-kernel speedups may vary ±0.1×.

## Still Stuck?

Contact: yuning.xia@rice.edu
