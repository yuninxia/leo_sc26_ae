# Evaluation Containers

Docker containers for profiling and analyzing GPU benchmarks with HPCToolkit and Leo.

## Layered Build Architecture

Containers use a two-layer structure to decouple profiling infrastructure from workloads:

```
leo-base-{vendor}           Profiling infrastructure (HPCToolkit + Dyninst + Leo)
  └── leo-{workload}-{vendor}   Benchmark/workload built on top of base
```

**Image naming**:
- `leo-base-nvidia`, `leo-base-amd`, `leo-base-intel` — base images
- `leo-rajaperf-nvidia`, `leo-llamacpp-amd`, etc. — workload images
- `leo-eval-{vendor}` — legacy alias for `leo-rajaperf-{vendor}`

**Benefits**: Build the base once (~30 min), then add any workload in minutes. Switch workloads without rebuilding HPCToolkit.

## Quick Start

```bash
# Build AMD container on odyssey (base image + RAJAPerf workload on top)
./scripts/evaluation/build_containers.sh amd

# Build NVIDIA container with llama.cpp workload instead of RAJAPerf
./scripts/evaluation/build_containers.sh nvidia --workload llamacpp

# Build only the base image (no workload, just HPCToolkit + Leo)
./scripts/evaluation/build_containers.sh intel --base-only

# Build all vendors (base + RAJAPerf for each)
./scripts/evaluation/build_containers.sh all

# Force rebuild of base image (e.g. after HPCToolkit or Leo changes)
./scripts/evaluation/build_containers.sh amd --rebuild-base
```

## Evaluation Workflows

### RAJAPerf (default)

```bash
# AMD: Polybench kernels on odyssey
./scripts/evaluation/run_evaluation.sh amd --kernels Polybench

# NVIDIA: Polybench kernels on gilgamesh
./scripts/evaluation/run_evaluation.sh nvidia --kernels Polybench

# Intel: Polybench kernels on headroom
./scripts/evaluation/run_evaluation.sh intel --kernels Polybench

# Run specific kernels
./scripts/evaluation/run_evaluation.sh amd --kernels DAXPY MULADDSUB

# Run all 82 RAJAPerf kernels
./scripts/evaluation/run_evaluation.sh amd

# Analyze top 10 kernels instead of default 5
./scripts/evaluation/run_evaluation.sh nvidia --top-n 10
```

### llama.cpp

```bash
# Profile llama.cpp inference on NVIDIA
./scripts/evaluation/run_evaluation.sh nvidia --workload llamacpp --model /path/to/model.gguf

# Profile on AMD
./scripts/evaluation/run_evaluation.sh amd --workload llamacpp --model /path/to/model.gguf

# With custom arguments
./scripts/evaluation/run_evaluation.sh nvidia --workload llamacpp --model /path/to/model.gguf --llama-args "-n 256"
```

## Profiling Pipeline

| Step | AMD | NVIDIA | Intel |
|------|-----|--------|-------|
| Profile | `hpcrun -e gpu=rocm,pc=hw` | `hpcrun -e gpu=cuda,pc` | `hpcrun -e gpu=level0,pc` |
| Structure | `hpcstruct --gpucfg yes` | `hpcstruct --gpucfg yes` | `hpcstruct --gpucfg yes` |
| Database | `hpcprof` | `hpcprof` | `hpcprof` |
| Analysis | `... --arch mi300` | `... --arch a100` | `... --arch pvc` |

**Intel note**: Requires `ZE_ENABLE_TRACING_LAYER=1` for PC sampling. See [Intel Troubleshooting](../../docs/INTEL_EVAL_TROUBLESHOOTING.md) for known issues.

## Target Machines

| Vendor | Machine | GPU | Architecture |
|--------|---------|-----|--------------|
| AMD | odyssey | 4x MI300A | `gfx942` |
| NVIDIA | gilgamesh | H100 | `sm_90` |
| Intel | headroom | PVC 1100 | auto |

## Available Workloads

| Workload | Description | Vendors |
|----------|-------------|---------|
| `rajaperf` (default) | RAJAPerf benchmark suite (82 GPU kernels) | NVIDIA, AMD, Intel |
| `llamacpp` | llama.cpp LLM inference benchmark | NVIDIA, AMD, Intel |

## Adding a New Workload

1. Create `docker/Dockerfile.{name}-{nvidia,amd,intel}` (~20 lines each):
   ```dockerfile
   ARG BASE_TAG=latest
   FROM leo-base-nvidia:${BASE_TAG}
   # Build your workload...
   ENV PATH="/opt/your-workload/bin:${PATH}"
   WORKDIR /opt/leo
   ```

2. Add to `VALID_WORKLOADS` in `build_containers.sh`
3. Add `WORKLOAD_EXEC[name]` in `run_evaluation.sh`

## RAJAPerf Kernels (82 total)

| Group | Kernels | Examples |
|-------|---------|---------|
| Basic | 20 | DAXPY, MULADDSUB, PI_REDUCE |
| Lcals | 11 | HYDRO_1D, HYDRO_2D, EOS |
| Polybench | 13 | 2MM, 3MM, GEMM, FLOYD_WARSHALL |
| Stream | 5 | ADD, COPY, DOT, MUL, TRIAD |
| Apps | 20 | DIFFUSION3DPA, LTIMES, ENERGY |
| Algorithm | 8 | SCAN, SORT, REDUCE_SUM, HISTOGRAM |
| Comm | 2 | HALO_PACKING, HALO_PACKING_FUSED |

## What's Inside

**Base image** (`leo-base-*`): HPCToolkit (Meson build, feature/hidden-stall-metrics), Dyninst, Leo with all Python dependencies, cmake.

**RAJAPerf image** (`leo-rajaperf-*`): RAJAPerf with vendor-specific GPU backend and `-lineinfo` for source mapping.

**llama.cpp image** (`leo-llamacpp-*`): llama.cpp with CUDA/HIP/SYCL backend for GPU-accelerated inference.

## File Structure

```
docker/
  Dockerfile.base-nvidia          Base: CUDA + HPCToolkit + Leo
  Dockerfile.base-amd             Base: ROCm + HPCToolkit + Leo
  Dockerfile.base-intel           Base: oneAPI + HPCToolkit + Leo
  Dockerfile.rajaperf-nvidia      RAJAPerf (CUDA)
  Dockerfile.rajaperf-amd         RAJAPerf (HIP)
  Dockerfile.rajaperf-intel       RAJAPerf (SYCL)
  Dockerfile.llamacpp-nvidia      llama.cpp (CUDA)
  Dockerfile.llamacpp-amd         llama.cpp (HIP)
  Dockerfile.llamacpp-intel       llama.cpp (SYCL)
build_containers.sh               Build orchestration (SSH to target machines)
run_evaluation.sh                 Run profiling + analysis pipeline
check_gpus.sh                     Verify GPU availability across machines
```

## Optional: Singularity

```bash
./scripts/evaluation/build_containers.sh amd --sif
singularity exec --rocm containers/leo-rajaperf-amd.sif bash
```
