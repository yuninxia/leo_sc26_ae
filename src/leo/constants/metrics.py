"""Metric name constants for HPCToolkit GPU profiling."""

# Base metrics
METRIC_GCYCLES = "gcycles"
METRIC_GCYCLES_ALL = "gcycles:*"

# Execution/issue metrics
METRIC_GCYCLES_ISU = "gcycles:isu"
METRIC_GCYCLES_LAT = "gcycles:lat"

# Stall metrics (total + breakdown)
METRIC_GCYCLES_STL = "gcycles:stl"
METRIC_GCYCLES_STL_ALL = "gcycles:stl:*"
METRIC_GCYCLES_STL_MEM = "gcycles:stl:mem"
METRIC_GCYCLES_STL_GMEM = "gcycles:stl:gmem"
METRIC_GCYCLES_STL_LMEM = "gcycles:stl:lmem"
METRIC_GCYCLES_STL_TMEM = "gcycles:stl:tmem"
METRIC_GCYCLES_STL_SYNC = "gcycles:stl:sync"
METRIC_GCYCLES_STL_IDEP = "gcycles:stl:idep"
METRIC_GCYCLES_STL_IFET = "gcycles:stl:ifet"
METRIC_GCYCLES_STL_PIPE = "gcycles:stl:pipe"

# Intel-specific stall metrics (EU stalls from hardware-assisted sampling)
# See docs/intel/INTEL_GPU_SUPPORT_RESEARCH.md for details
METRIC_GCYCLES_STL_ACTIVE = "gcycles:stl:active"      # Active cycles (not stall)
METRIC_GCYCLES_STL_CONTROL = "gcycles:stl:control"    # Control flow stalls
METRIC_GCYCLES_STL_SEND = "gcycles:stl:send"          # Send unit stalls (memory)
METRIC_GCYCLES_STL_DIST = "gcycles:stl:dist"          # Distance/ARF dependency
METRIC_GCYCLES_STL_SBID = "gcycles:stl:sbid"          # Scoreboard stalls (exec dep)
METRIC_GCYCLES_STL_INSFETCH = "gcycles:stl:insfetch"  # Instruction fetch stalls
METRIC_GCYCLES_STL_OTHER = "gcycles:stl:other"        # Other stalls

# Instruction execution metrics (for execution constraints pruning)
# NVIDIA uses GINS:EXE from PC sampling
METRIC_GINS_EXE = "GINS:EXE"  # NVIDIA instruction execution count
METRIC_GINS_EXE_PRED = "GINS:EXE (PRED)"  # NVIDIA predicated execution count

# Vendor-specific fallbacks for execution count detection:
# - NVIDIA: GINS:EXE > 0 means instruction was executed
# - AMD: gcycles:isu > 0 means instruction was issued (proxy for executed)
# - Intel: gcycles:isu > 0 or gcycles:stl > 0 (if stalled, must have been issued)
EXECUTION_METRICS = [
    METRIC_GINS_EXE,      # NVIDIA primary
    METRIC_GCYCLES_ISU,   # AMD/Intel fallback (issue cycles as proxy)
]

# Kernel-level metrics (per-kernel, not per-instruction)
METRIC_GKER = "gker"                    # Kernel execution time in seconds
METRIC_GKER_COUNT = "gker:count"        # Kernel launch count
METRIC_GKER_BLK_THR = "gker:blk_thr_acumu"    # Threads per block
METRIC_GKER_BLKS_AVG = "gker:blks_avg_acumu"  # Average blocks per launch
METRIC_GKER_VREG = "gker:thr_vreg_acumu"      # Vector registers per thread
METRIC_GKER_SREG = "gker:thr_sreg_acumu"      # Scalar registers per thread
METRIC_GKER_STMEM = "gker:stmem_acumu"        # Shared memory per block
METRIC_GKER_LMEM = "gker:lmem_acumu"          # Local memory per thread

# Sampling info metrics (NVIDIA-specific SM efficiency)
METRIC_GSAMP_TOT = "gsamp:tot"              # Total PC samples collected
METRIC_GSAMP_EXP = "gsamp:exp"              # Expected samples (if all SMs active)
