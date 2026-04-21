# LULESH (Section VI case study)

LULESH is a shock hydrodynamics proxy application (LLNL). Leo guides a
gather-once AoS→SoA optimization for five hot kernels.

## Baseline (original)

**The baseline is `fork/omp_4.0/`** — this is the upstream LLNL OMP 4.0
source, kept as-is; the `fork/` name refers to the GitHub fork used for
version control, not to a modification.

- Upstream: https://github.com/LLNL/LULESH (`omp4.0` branch)
- Vendored here at: `fork/omp_4.0/`
- `run_compare_*.sh` builds this directory as the "original" side of the
  comparison.

## Optimized

`optimized/` contains the Leo-guided AoS→SoA variant. See the commit history
in the parent repo for the exact set of changes, or run:

```bash
diff -ru benchmarks/lulesh/fork/omp_4.0/ benchmarks/lulesh/optimized/
```

(Also available via `bash run_compare_nvidia.sh --diff` without Docker.)

## Vendor runners

- `run_compare_nvidia.sh` — NVIDIA H100 (`leo-lulesh-nvidia`)
- `run_compare_amd.sh` — AMD MI300A (`leo-lulesh-amd`)
- `run_compare_intel.sh` — Intel PVC (`leo-lulesh-intel`)

Each builds `fork/omp_4.0/` and `optimized/` from source inside its Docker
image and runs a comparative benchmark with per-kernel timing.
