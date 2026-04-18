# Hardware Setup

Vendor-specific setup notes for reproducing LEO results.

## NVIDIA GH200

- Driver: 570+
- CUDA: 12.8
- PC sampling: CUPTI Activity API
- Build HPCToolkit with `-Dcuda=enabled`

## AMD MI300A

- ROCm 6.3
- PC sampling: ROCprofiler-SDK stochastic mode
- Build HPCToolkit with `-Drocm=enabled`
- Stochastic mode available on MI300+ only

## Intel GPU Max 1100

- Level Zero 1.x (from oneAPI 2025.1+)
- PC sampling via GTPin on Level Zero
- Build HPCToolkit with `-Dlevel0=enabled -Dgtpin=enabled`
- **IMPORTANT:** Set `ZE_ENABLE_TRACING_LAYER=1` before running `hpcrun`
- GED library location: `$HPCTOOLKIT_BUILD/subprojects/gtpin-4.5.0/Profilers/Lib/intel64/libged.so`
  (set `GED_LIBRARY_PATH` env var if LEO can't auto-detect)

## HPCToolkit Fork

```bash
git clone --recursive https://gitlab.com/yuningxia/hpctoolkit.git
cd hpctoolkit
git checkout feature/hidden-stall-metrics

# Build (example for NVIDIA)
meson setup build --buildtype=release \
  -Dcuda=enabled -Dcuda_prefix=/usr/local/cuda
cd build && meson compile && meson install
```

**Note on hpcstruct segfault:** If you hit exit 139 during hpcstruct, ensure `--buildtype=release` (not debug). Debug TBB libraries have been known to segfault.

## Docker Images

All evaluation runs in vendor-specific Docker containers. Build with:

```bash
bash scripts/evaluation/build_containers.sh nvidia   # or amd, intel, universal
```

Base images include HPCToolkit (branch `feature/hidden-stall-metrics`) + LEO. Workload images layer the benchmark on top.

## Partial Reproduction (No GPU)

Use `leo-base-universal` container for analysis-only reproduction. This runs on any x86-64 machine and reads pre-collected HPCToolkit databases from Zenodo.
