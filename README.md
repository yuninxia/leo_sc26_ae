# LEO SC26 Artifact Evaluation

Artifact for the SC26 paper:

> **LEO: Tracing GPU Stall Root Causes via Cross-Vendor Backward Slicing**
> Yuning Xia and John Mellor-Crummey, Rice University

This repository contains the LEO analysis tool, evaluation infrastructure, benchmark workloads, and optimization patches used to produce the paper's results.

---

## What This Artifact Reproduces

- **Table IV** — Per-kernel speedups for 21 workloads × 3 GPU platforms (NVIDIA GH200, AMD MI300A, Intel GPU Max 1100)
- **Figure 5** — Single Dependency Coverage (SDC) across vendors
- **Section VI case studies** — MASS3DEA, miniBUDE, QuickSilver, ML kernels (llama.cpp, HipKittens), Kripke

---

## Artifact Contents

```
leo_sc26_ae/
├── src/leo/                    # LEO Python source code
├── scripts/                    # Analysis and reproduction scripts
├── scripts/evaluation/         # Docker orchestration and evaluation runners
├── benchmarks/                 # Benchmark forks with optimizations
│   ├── rajaperf-h100/         # 15 RAJAPerf kernels (original + optimized)
│   ├── minibude/              # miniBUDE (molecular docking)
│   ├── lulesh/                # LULESH (hydrodynamics)
│   ├── kripke/                # Kripke (particle transport)
│   ├── llamacpp/              # llama.cpp (LLM inference)
│   ├── quicksilver/           # QuickSilver (Monte Carlo)
│   └── xsbench/               # XSBench (nuclear physics)
├── tests/                      # Unit tests (data downloaded separately)
├── docs/                       # Setup and troubleshooting guides
├── pyproject.toml              # Python dependencies (uv)
├── LICENSE                     # Apache 2.0
└── CITATION.cff                # Citation metadata
```

---

## External Dependencies (Not Vendored)

### HPCToolkit (required for profiling)

LEO reads PC-sampling data from an HPCToolkit fork. Build instructions:

```bash
git clone --recursive https://gitlab.com/yuningxia/hpctoolkit.git
cd hpctoolkit
git checkout feature/hidden-stall-metrics
# Follow HPCToolkit build instructions for your vendor (meson --buildtype=release)
```

See `docs/HARDWARE_SETUP.md` for vendor-specific details.

### Pre-collected Profiling Data (GitHub Release)

No GPU needed for LEO analysis — one command downloads all pre-collected measurements (~1 GB compressed, ~5.6 GB extracted):

```bash
bash scripts/download_data.sh
```

This pulls `leo-sc26-measurements.tar.gz` from the GitHub Release and extracts to `results/`. After that, all analysis scripts (`collect_sdc.sh`, `time_analysis.sh`) run without GPU hardware.

Three Zenodo records back this artifact:

- **Source code** (concept DOI, always latest): [10.5281/zenodo.19704349](https://doi.org/10.5281/zenodo.19704349)
- **Profiling data tarball** (~1 GB, CC0): [10.5281/zenodo.19705577](https://doi.org/10.5281/zenodo.19705577)
- **Pre-built Docker images** (NVIDIA chain, ~12 GB, Apache-2.0): [10.5281/zenodo.19752199](https://doi.org/10.5281/zenodo.19752199) — mirror of `jssonxia/leo-*-nvidia` on Docker Hub. `gunzip -c leo-sc26-ae-images-v0.1.14.tar.gz | docker load` to skip the ~30 min build.

`download_data.sh` primarily pulls from the GitHub Release `v1.0-sc26-data`; if the GitHub URL is unreachable (e.g., the source repo is temporarily private), it automatically falls back to the Zenodo data mirror.

---

## Quick Start (15 min, no GPU required)

```bash
# 1. Clone (502 MB, ~2 min)
git clone https://github.com/yuninxia/leo_sc26_ae.git
cd leo_sc26_ae

# 2. Check / install system build deps (python3-dev, pkg-config, unzip).
#    Most cloud base images (Lambda, Verda, RunPod, Paperspace) ship without
#    these and `uv sync` will fail at the `hpcanalysis` meson build otherwise.
bash scripts/preflight.sh            # dry-run: prints missing + exact install command
bash scripts/preflight.sh --install  # or: auto-install via apt / dnf (uses sudo if not root)

# 3. Install Python deps
curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv not installed
uv sync

# 4. Download pre-collected profiling data (~1 GB download, ~5.6 GB extracted)
bash scripts/download_data.sh

# 5. Build the universal analysis Docker image (~20 min one-time)
bash scripts/evaluation/build_containers.sh universal --base-only

# 6. Reproduce Figure 5 (SDC coverage)
bash scripts/collect_sdc.sh

# 7. (Optional) Verify LEO analyzer unit tests
uv run pytest tests/ -v -m "not slow" --ignore-glob="*test_integration*"
```

> **Compiler note (RHEL/CentOS 8):** `uv sync` builds the vendored `hpcanalysis`
> C++ extension and requires GCC 11+. If `gcc --version` reports 8.x, prefix
> `uv sync` with `scl enable gcc-toolset-11 --`. Ubuntu 22.04+ and recent
> Fedora ship GCC 11+ by default.

---

## Hardware Requirements

Full 3-vendor reproduction requires:

| Vendor | GPU | Software |
|--------|-----|----------|
| NVIDIA | GH200 480GB (sm_90, Hopper) | CUDA 12.8, driver 570+ |
| AMD | MI300A (gfx942, CDNA3 APU) | ROCm 6.3 |
| Intel | Data Center GPU Max 1100 (Ponte Vecchio, Xe-HPC) | Level Zero 1.x, oneAPI 2025.1+ |

**Partial reproduction is supported.** LEO analysis on pre-collected profiles runs on any x86-64 machine without GPU (use the `leo-base-universal` Docker container).

---

## Reproduction Workflow

The evaluation has four tasks. The full AD/AE Appendix is submitted with the paper (SC26 submission system); this README is the reviewer's quick reference.

### Task 1: Analyze pre-collected profiling data (CPU-only)

Uses Docker + pre-collected HPCToolkit databases. No GPU needed.

**Prerequisites:** steps 2–4 of Quick Start (`uv sync`, `download_data.sh`, universal Docker image). The two scripts below expect the `leo-base-universal` image to be built locally and data extracted under `results/`.

```bash
# Compute SDC metrics for Figure 5
bash scripts/collect_sdc.sh

# Measure LEO analysis time
bash scripts/time_analysis.sh
```

### Task 2: Run RAJAPerf benchmarks (requires GPUs)

Per-vendor measurement of original vs. optimized kernels for Table IV.

```bash
# Run from the repo root (not from benchmarks/rajaperf-h100/).
bash scripts/evaluation/run_workload_rajaperf.sh --vendor nvidia   # on GH200
bash scripts/evaluation/run_workload_rajaperf.sh --vendor amd      # on MI300A
bash scripts/evaluation/run_workload_rajaperf.sh --vendor intel    # on GPU Max 1100
cd benchmarks/rajaperf-h100 && python postprocess.py  # generate Table IV data
```

**Expected:** per-kernel speedups match Table IV within ±5% (NVIDIA/AMD) or ±9% (Intel, due to platform variability).

### Task 3: Case studies (Section VI)

```bash
# Example: MASS3DEA across three vendors
bash scripts/evaluation/run_workload_rajaperf.sh --kernel MASS3DEA --vendor all
```

Optimization patches for each case study are in `benchmarks/*/optimized/`. To see the changes:

- `rajaperf-h100/` and `xsbench/` — diff `original/` vs `optimized/` directly.
- `lulesh/` — `fork/omp_4.0/` is the baseline; see `benchmarks/lulesh/README.md`.
- `kripke/`, `quicksilver/`, `llamacpp/` — the baseline is upstream; each
  benchmark's top-level `README.md` lists the upstream URL and the patch/file
  pair that reproduces the Leo-guided change.

---

## Expected Results (Tolerance)

- **Table IV speedups:** ±5% on NVIDIA GH200 and AMD MI300A (median CV < 1%); ±9% on Intel PVC (median CV ~9%)
- **Figure 5 SDC coverage (before → after pruning):** NVIDIA 30–74% → 64–94%; AMD 80–97%; Intel PVC stable at 61–89% (exact values within rounding)

Table V (LLM diagnostic study) is **not** reproduced by this artifact: the numeric Compilable / Speedup / Regressions columns require a GPU-side compile-and-benchmark harness and a paid OpenRouter subscription, and the LLM outputs are non-deterministic. The corresponding code has been removed from the artifact; see the AD appendix Badges-requested paragraph.

---

## Known Deviations from Paper

- Pre-collected data on Zenodo may have slightly different timestamps than those in the paper but produces identical LEO analysis output.
- Intel PVC results exhibit higher run-to-run variability (documented 9% CV); ±9% tolerance recommended.

---

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues:

- HPCToolkit build failures on Intel (TBB / debug library)
- Docker container build errors (vendor-specific base images)
- GTPin library not found (Intel)
- GPU driver version mismatches

---

## License

Apache 2.0. See `LICENSE`.

Third-party benchmarks retain their original licenses (see `benchmarks/*/LICENSE`).

---

## Citation

The paper is currently under peer review. Until acceptance, please cite as:

```bibtex
@unpublished{xia2026leo,
  author = {Xia, Yuning and Mellor-Crummey, John},
  title  = {{LEO}: Tracing {GPU} Stall Root Causes via Cross-Vendor Backward Slicing},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

Once accepted, this entry will be updated to an `@inproceedings` form with the
full SC26 proceedings information.

## Contact

- Yuning Xia — yuning.xia@rice.edu
- John Mellor-Crummey — johnmc@rice.edu

## Acknowledgments

The full AD/AE Appendix is submitted with the paper via the SC26 submission system (not bundled in this artifact to avoid stale-snapshot drift).
