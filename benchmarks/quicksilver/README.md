# QuickSilver (Section VI case study)

QuickSilver is a proxy application for Monte Carlo particle transport (LLNL).
Leo traces a cross-file dependency chain through `CollisionEvent.hh` → 
`MacroscopicCrossSection.hh` → `NuclearData.hh`; the optimization inlines
the cross-file call and hoists loop invariants.

## Upstream

- Source: https://github.com/LLNL/Quicksilver
- Baseline used in the paper: `master`. Clone and treat the upstream files
  as the *original* for diffing.

```bash
git clone https://github.com/LLNL/Quicksilver /tmp/quicksilver-original
```

## What's in this directory

- `optimized/` — modified files for the NVIDIA path (`CollisionEvent.{cc,hh}`,
  `MacroscopicCrossSection.hh`, `NuclearData.hh`, `MCT.hh`, `main.cc`).
  See `optimized/README.md` for the full diagnosis.
- `optimized-amd/` — separate variant for AMD MI300A.
- `run_compare_nvidia.sh` / `run_compare_amd.sh` — GPU run harnesses.

## Diffing the optimization

Drop `optimized/*.{cc,hh}` onto an upstream clone (same filenames and
directory layout inside `src/`) and diff against the upstream versions.
