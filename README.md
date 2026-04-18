# LEO SC26 Artifact Evaluation

Artifact for the SC26 paper:

> **LEO: Tracing GPU Stall Root Causes via Cross-Vendor Backward Slicing**
> Yuning Xia and John Mellor-Crummey, Rice University

This repository contains the LEO analysis tool, evaluation infrastructure, benchmark workloads, and optimization patches used to produce the paper's results.

---

## What This Artifact Reproduces

- **Table IV** — Per-kernel speedups for 21 workloads × 3 GPU platforms (NVIDIA GH200, AMD MI300A, Intel GPU Max 1100)
- **Table V** — LLM diagnostic context ablation (Gemini 3.1 Pro)
- **Figure 5** — Single-Dependency Coverage (SDC) across vendors
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
├── ad_appendix.pdf             # Artifact Description from paper
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

### Pre-collected Profiling Data (Zenodo)

To skip profiling (no GPU required for analysis), download measurement databases from Zenodo:

```bash
# TODO: Zenodo DOI pending
# mkdir -p data/hpctoolkit-measurements
# cd data/hpctoolkit-measurements
# wget https://zenodo.org/record/XXXXXXX/files/leo-sc26-data.tar.gz
# tar xzf leo-sc26-data.tar.gz
```

### LLM API (optional, Table V only)

Reproducing Table V requires **Gemini 3.1 Pro** API access (Google Cloud). Set `GOOGLE_API_KEY` in your environment. Table IV and Figure 5 do not need LLM access.

---

## Quick Start (10 min, no GPU required)

```bash
# 1. Clone
git clone https://github.com/yuninxia/leo_sc26_ae.git
cd leo_sc26_ae

# 2. Install Python deps
curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv not installed
uv sync

# 3. Verify LEO analyzer works
uv run pytest tests/ -v -m "not slow" --ignore-glob="*test_integration*"
```

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

The evaluation has four tasks. For full detail, see `ad_appendix.pdf`.

### Task 1: Analyze pre-collected profiling data (CPU-only)

Uses Docker + pre-collected HPCToolkit databases. No GPU needed.

```bash
# Build universal analysis container (one-time, ~20 min)
bash scripts/evaluation/build_containers.sh universal --base-only

# Compute SDC metrics for Figure 5
bash scripts/collect_sdc.sh

# Measure LEO analysis time
bash scripts/time_analysis.sh
```

### Task 2: Run RAJAPerf benchmarks (requires GPUs)

Per-vendor measurement of original vs. optimized kernels for Table IV.

```bash
cd benchmarks/rajaperf-h100
bash run_compare.sh --vendor nvidia   # on GH200
bash run_compare.sh --vendor amd      # on MI300A
bash run_compare.sh --vendor intel    # on GPU Max 1100
python postprocess.py                 # generate Table IV data
```

**Expected:** per-kernel speedups match Table IV within ±5% (NVIDIA/AMD) or ±9% (Intel, due to platform variability).

### Task 3: Case studies (Section VI)

```bash
# Example: MASS3DEA across three vendors
bash scripts/evaluation/run_workload_rajaperf.sh --kernel MASS3DEA --vendor all
```

Optimization patches for each case study are in `benchmarks/*/optimized/`. Diff against `original/` to see the changes.

### Task 4: LLM diagnostic study (Table V, optional)

```bash
export GOOGLE_API_KEY=your_key_here
uv run python scripts/run_llm_eval.py --model gemini-3.1-pro --trials 5
```

Pre-collected LLM outputs are included in the artifact for reference.

---

## Expected Results (Tolerance)

- **Table IV speedups:** ±5% on NVIDIA GH200 and AMD MI300A (median CV < 1%); ±9% on Intel PVC (median CV ~9%)
- **Figure 5 SDC coverage:** NVIDIA 30–74% → 64–94% after pruning (exact values within rounding)
- **Table V:** Gemini 3.1 Pro results subject to LLM sampling variance; overall trends (100% compile rate with C+L(S), 1.29× speedup) should hold

---

## Known Deviations from Paper

- Pre-collected data on Zenodo may have slightly different timestamps than those in the paper but produces identical LEO analysis output.
- Intel PVC results exhibit higher run-to-run variability (documented 9% CV); ±9% tolerance recommended.
- LLM study is non-deterministic; best-of-5 speedups for specific kernels may vary ±0.1×, but geomean and compile rate reproduce.

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

```bibtex
@inproceedings{xia2026leo,
  title     = {{LEO}: Tracing {GPU} Stall Root Causes via Cross-Vendor Backward Slicing},
  author    = {Xia, Yuning and Mellor-Crummey, John},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC26)},
  year      = {2026}
}
```

## Contact

- Yuning Xia — yx87@rice.edu
- John Mellor-Crummey — johnmc@rice.edu

## Acknowledgments

See `ad_appendix.pdf` for full artifact description.
