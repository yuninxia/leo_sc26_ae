# Kripke (Section VI case study)

Kripke is a proxy application for Sn deterministic particle transport (LLNL).
Leo identifies a global-load stall in the `LTimes` kernel and guides a
Group↔Zone swap in the RAJA execution policy.

## Upstream

- Source: https://github.com/LLNL/Kripke
- Baseline used in the paper: `master` branch, commit representing RAJA
  execution policies prior to the swap. Clone and treat the upstream
  `LTimes.cpp` as the *original*.

```bash
git clone https://github.com/LLNL/Kripke /tmp/kripke-original
```

## What's in this directory

- `optimized/` — the Leo-guided fix: patch (`ltimes_loop_reorder.patch`),
  replacement header (`LTimes.h`), runner scripts (`run_nvidia.sh`,
  `run_amd.sh`), and LEO output logs. See `optimized/README.md` for the
  full diagnosis.
- `standalone-amd/` — a self-contained AMD reproducer used for profiling.

## Diffing the optimization

Apply `optimized/ltimes_loop_reorder.patch` to a clean upstream checkout, or
compare `optimized/LTimes.h` against `Kripke/src/Kripke/Kernel/LTimes.cpp`
from the upstream baseline.

## Reproducing on GPUs

See `optimized/run_nvidia.sh` and `optimized/run_amd.sh`.
