# Kripke Leo-Guided Optimization

## Leo Diagnosis

Leo identifies the `LTimes` kernel (`LTimes.cpp:62`) as the primary bottleneck in Kripke's Sn transport solver. The root cause is a `global_load` stall traced through 5 files:

```
DFMA @ LTimes.cpp:62
  ← LDG.E.64 @ LTimes.cpp:62
    ← LEA.HI.X @ TypedViewBase.hpp:216      (RAJA View abstraction)
      ← IADD3 @ Operators.hpp:369            (RAJA operator overload)
        ← IADD3 @ Iterators.hpp:291          (RAJA iterator)
```

### Root Cause Analysis

The RAJA execution policy for LTimes with DGZ layout:
```
Moment(block_x) → Direction(block_y) → Group(thread_x) → Zone(sequential)
```

**Zone is sequential** — each GPU thread loops over all 32K zones. Adjacent threads handle different Groups but the same Zone range. With DGZ data layout (`psi[Direction][Group][Zone]`), adjacent threads access `psi(d, g0, z)` and `psi(d, g1, z)` which are **not contiguous** in memory (stride = num_zones between groups).

Leo's stall analysis shows:
- **AMD MI300A**: 73.8% stall cycles from `global_load_dwordx2`, 3.43x speedup potential
- **NVIDIA H100**: 96.7% stall cycles from `LDG.E.64`, 1.92x speedup potential

## Optimization

Swap Group and Zone in the RAJA execution policy:
```
Moment(block_x) → Direction(block_y) → Zone(thread_x) → Group(sequential)
```

Now **Zone is thread-parallel** — adjacent threads handle adjacent zones. With DGZ layout, `psi(d, g, z)` and `psi(d, g, z+1)` ARE contiguous → **coalesced memory access**.

The change is 2 lines in `src/Kripke/Arch/LTimes.h` (one for CUDA, one for HIP).

## Results

### NVIDIA H100

| Metric | Baseline | Optimized |
|--------|---------|-----------|
| LTimes kernel | 1.12s | **0.23s (4.93x)** |
| Solve | 5.71s | **3.77s (1.52x)** |
| Throughput | 1.76e8 | **2.67e8 (1.52x)** |

### AMD MI300A

Two CHAI memory management modes were tested:

**CHAI THIN_GPU_ALLOCATE mode** (requires `HSA_XNACK=1`):
- Uses single memory address space with page fault handling
- High measurement variance (CV 50-65%) due to XNACK page fault overhead
- Results unreliable for per-kernel attribution

**CHAI Resource Manager mode** (no XNACK needed, recommended for benchmarking):
- Uses explicit host↔device data management
- Stable measurements

| Metric | Baseline (RM) | Optimized (RM) |
|--------|--------------|----------------|
| LTimes kernel | 13.2-15.2s | **0.54-1.51s (~10-25x)** |
| Solve | 26.9-29.3s | **3.0-12.9s** |

Note: CHAI RM mode has higher absolute times than THIN_GPU_ALLOCATE due to explicit data transfer overhead, but provides more reliable relative measurements.

## Reproducing

### Prerequisites
- Docker with GPU access (NVIDIA or AMD)
- Leo's Kripke containers: `leo-kripke-nvidia` / `leo-kripke-amd`

### Quick test

```bash
# Build containers (if not already built)
cd scripts/evaluation
docker build -f docker/Dockerfile.kripke-nvidia --build-arg GPU_ARCH=90 -t leo-kripke-nvidia .
docker build -f docker/Dockerfile.kripke-amd -t leo-kripke-amd .

# Run benchmark
cd benchmarks/kripke/optimized
bash run_benchmark.sh --vendor nvidia   # NVIDIA H100
bash run_benchmark.sh --vendor amd      # AMD MI300A
```

### Manual reproduction

```bash
# NVIDIA
docker run --rm --gpus all --entrypoint /bin/bash leo-kripke-nvidia -c '
  echo "=== BASELINE ===" &&
  /opt/kripke/bin/kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E "LTimes|Solve |Throughput" &&
  echo "" &&
  cd /opt/kripke-src/src/Kripke/Arch &&
  sed -i "s/cuda_thread_x_loop, \/\/ group/cuda_thread_x_loop, \/\/ Zone (opt)/" LTimes.h &&
  sed -i "s/seq_exec, \/\/ zone/seq_exec, \/\/ Group (opt)/" LTimes.h &&
  cd /opt/kripke-build && make -j$(nproc) 2>/dev/null | tail -1 &&
  cp kripke.exe /opt/kripke/bin/ &&
  echo "=== OPTIMIZED ===" &&
  /opt/kripke/bin/kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E "LTimes|Solve |Throughput"
'

# AMD (CHAI RM mode, no XNACK needed)
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --entrypoint /bin/bash leo-kripke-amd -c '
  mkdir -p /tmp/kb && cd /tmp/kb &&
  cmake /opt/kripke-src -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
    -DENABLE_HIP=On -DENABLE_CHAI=On \
    -DCHAI_DISABLE_RM=OFF -DCHAI_THIN_GPU_ALLOCATE=OFF -DCHAI_ENABLE_UM=OFF \
    -DENABLE_MPI=Off -DENABLE_OPENMP=Off \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 -DROCM_PATH=/opt/rocm 2>/dev/null | tail -1 &&
  make -j$(nproc) 2>/dev/null | tail -1 &&
  echo "=== BASELINE ===" &&
  ./kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E "LTimes|Solve |Throughput" &&
  echo "" &&
  cd /opt/kripke-src/src/Kripke/Arch &&
  sed -i "s/hip_thread_x_loop, \/\/ group/hip_thread_x_loop, \/\/ Zone (opt)/" LTimes.h &&
  sed -i "s/seq_exec, \/\/ zone/seq_exec, \/\/ Group (opt)/" LTimes.h &&
  cd /tmp/kb && make -j$(nproc) 2>/dev/null | tail -1 &&
  echo "=== OPTIMIZED ===" &&
  ./kripke.exe --zones 32,32,32 --groups 32 --niter 10 2>&1 | grep -E "LTimes|Solve |Throughput"
'
```

## Key Insight

Leo's cross-file back-slicing through 5 layers of RAJA abstraction (LTimes.cpp → TypedViewBase.hpp → Operators.hpp → Iterators.hpp → For.hpp) reveals that the performance bottleneck is NOT in the physics code itself, but in how the RAJA framework maps loop dimensions to GPU threads. Without Leo's trace through the framework layers, a developer would only see "line 62 is slow" without understanding WHY — the strided memory access pattern caused by the Group-parallel/Zone-sequential mapping.
