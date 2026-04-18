# RAJAPerf: Original vs Leo-Optimized Comparison

Two copies of the [RAJAPerf](https://github.com/LLNL/RAJAPerf) benchmark suite for A/B comparison of Leo-guided GPU kernel optimizations.

- `original/` — upstream [LLNL/RAJAPerf](https://github.com/LLNL/RAJAPerf) (develop branch, used for diff discovery only)
- `optimized/` — [fork](https://github.com/yuninxia/RAJAPerf) with Leo optimizations applied
- `run_compare.sh` — build, run, and compare both versions inside Docker

## How It Works

Runs inside Docker containers (`leo-rajaperf-{intel,amd,nvidia}`) which already have the upstream RAJAPerf pre-built as `raja-perf.exe`. The script only builds the optimized fork, then compares both binaries on the same GPU.

Build artifacts are persisted in a named Docker volume (`rajaperf-build-{vendor}`), so:

- **First run**: full CMake + build (~10 min)
- **After adding/modifying one kernel**: incremental recompile + relink (~10 sec)
- Volume survives `docker run --rm`; use `docker volume rm rajaperf-build-intel` to clean

## Usage

```bash
# Compare all optimized kernels on Intel PVC
./run_compare.sh --docker leo-rajaperf-intel

# Compare a specific kernel
./run_compare.sh --docker leo-rajaperf-intel --kernel MASS3DEA

# AMD MI300A
./run_compare.sh --docker leo-rajaperf-amd --kernel MASS3DEA

# NVIDIA H100
./run_compare.sh --docker leo-rajaperf-nvidia --kernel MASS3DEA

# Skip build (reuse previous build from volume)
./run_compare.sh --docker leo-rajaperf-intel --skip-build

# More timing passes (default: 3)
./run_compare.sh --docker leo-rajaperf-intel --npasses 5

# Profile with HPCToolkit + Leo analysis
./run_compare.sh --docker leo-rajaperf-intel --kernel MASS3DEA --profile

# List which kernels have been optimized (no Docker needed)
./run_compare.sh --list-optimized

# Full HPCToolkit profiling + Leo root cause analysis (per-kernel)
# Results go to results/per-kernel/<KERNEL>/<vendor>/
cd ../../  # from benchmarks/rajaperf-h100 to leo root
./scripts/evaluation/run_per_kernel.sh nvidia-gilgamesh        # H100 on gilgamesh
./scripts/evaluation/run_per_kernel.sh nvidia-athena            # A100 on athena
./scripts/evaluation/run_per_kernel.sh amd                      # MI300A on odyssey
./scripts/evaluation/run_per_kernel.sh intel                    # PVC on headroom

# Profile specific kernels only
./scripts/evaluation/run_per_kernel.sh nvidia-gilgamesh --kernels Apps_MASS3DEA Apps_LTIMES
```

## Optimizations Applied

Each optimization modifies a single `*-Sycl.cpp` (or `*-Hip.cpp`, `*-Cuda.cpp`) file in `optimized/src/`. Only the `Base_SYCL` (or `Base_HIP`, `Base_CUDA`) kernel variant is changed; RAJA variants are left untouched.

### Current optimizations

| Kernel | File | Optimization | Intel PVC | AMD MI300A | NVIDIA H100 |
|---|---|---|---|---|---|
| MASS3DEA | `src/apps/MASS3DEA-Sycl.cpp` | Precompute basis in registers + j3-innermost loop reorder | 6.88x | 3.45x | 3.01x |

## How to Add a New Optimization

1. Edit the kernel's backend file in `optimized/src/`, e.g. `optimized/src/apps/ENERGY-Sycl.cpp`
2. In the `Base_SYCL` case, replace the macro call (e.g. `MASS3DEA_4`) with the optimized inline code
3. Keep SLM allocation, cooperative loading, and barriers unchanged; only modify the compute phase
4. Verify: `./run_compare.sh --list-optimized` should show the new kernel
5. Build and compare: `./run_compare.sh --docker leo-rajaperf-intel --kernel ENERGY`
6. Incremental build takes ~10 seconds since the Docker volume has the previous build cached

### Key patterns in RAJAPerf source

- Kernel bodies use macros (e.g. `MASS3DEA_4`) defined in the `.hpp` header
- SLM uses typed multi-dimensional `sycl::local_accessor` (2D for matrices, 3D for tensors), not flattened 1D
- Thread mapping: `SYCL_FOREACH_THREAD(var, dim, N)` expands to `for (var = itm.get_local_id(dim); var < N; var += itm.get_local_range(dim))`
- Element index: `itm.get_group(2)` (not `get_group(0)`)
- Work-group size is typically `Q1D x Q1D x Q1D` (e.g. 5x5x5 = 125), not `D1D x D1D x D1D`

## Relationship to Standalone Benchmarks

| | Standalone (`optimizations/rajaperf/`) | Full RAJAPerf (`rajaperf/`) |
|---|---|---|
| Build | Single-file Makefile (`icpx -fsycl`) | CMake (full RAJAPerf + RAJA + BLT) |
| SLM | Flattened 1D `local_accessor` | Native 2D/3D `local_accessor` |
| Thread mapping | Manual `get_local_id()` | `SYCL_FOREACH_THREAD` macro |
| Work-group size | Often D1D^3 (64) | RAJAPerf default Q1D^3 (125) |
| Validation | Checksum comparison | RAJAPerf built-in correctness |
| Use case | Rapid prototyping + iteration | Artifact evaluation + reproducibility |
