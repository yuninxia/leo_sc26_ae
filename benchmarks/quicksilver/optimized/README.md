# QuickSilver Leo-Guided Optimization

## Root Cause (Leo Analysis)

**NVIDIA H100**: Leo traces a 3-file dependency chain responsible for 19.8% of stall
cycles. A floating-point multiply in `CollisionEvent.hh` stalls on a global load
that passes through `MacroscopicCrossSection.hh` to `NuclearData.hh`.

**AMD MI300A**: Leo identifies flat load latency through cross-file pointer chains
under extreme register pressure (153 VGPRs, 0% occupancy).

## Optimization

Modified file: `CollisionEvent.hh` (lines 88-109 of original)

Three changes guided by Leo's cross-file dependency chain:

1. **Inline the cross-file call**: Replace `macroscopicCrossSection()` call with
   direct access to `reactions[reactIndex].getCrossSection()`. This eliminates
   the 3-file pointer chain that prevented load reordering.

2. **Add `__restrict__`**: Annotate `NuclearDataReaction_d*` pointer to enable
   compiler load reordering across the inlined code.

3. **Hoist loop invariants**: Extract `cellNumberDensity` before the isotope loop
   and precompute `prefactor = atomFraction * cellNumberDensity`.

## Results

- NVIDIA H100: 1.17x speedup
- AMD MI300A: 1.16x speedup
