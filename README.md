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

### Pre-collected Profiling Data (GitHub Release)

No GPU needed for LEO analysis — one command downloads all pre-collected measurements (~1 GB compressed, ~8.5 GB extracted):

```bash
bash scripts/download_data.sh
```

This pulls `leo-sc26-measurements.tar.gz` from the GitHub Release and extracts to `results/`. After that, all analysis scripts (`collect_sdc.sh`, `time_analysis.sh`) run without GPU hardware.

A Zenodo mirror is also available for archival citation (DOI: TBD).

### LLM API (optional, Table V only)

Reproducing Table V requires **Gemini 3.1 Pro** via the OpenRouter proxy (https://openrouter.ai). Set `OPENROUTER_API_KEY` in your environment. Table IV and Figure 5 do not need LLM access.

---

## Quick Start (15 min, no GPU required)

```bash
# 1. Clone (502 MB, ~2 min)
git clone https://github.com/yuninxia/leo_sc26_ae.git
cd leo_sc26_ae

# 2. Install Python deps
curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv not installed
uv sync

# 3. Download pre-collected profiling data (~1 GB download, ~8.5 GB extracted)
bash scripts/download_data.sh

# 4. Build the universal analysis Docker image (~20 min one-time)
bash scripts/evaluation/build_containers.sh universal --base-only

# 5. Reproduce Figure 5 (SDC coverage)
bash scripts/collect_sdc.sh

# 6. (Optional) Verify LEO analyzer unit tests
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

The evaluation has four tasks. For full detail, see `ad_appendix.pdf`.

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

Optimization patches for each case study are in `benchmarks/*/optimized/`. To see the changes:

- `rajaperf-h100/` and `xsbench/` — diff `original/` vs `optimized/` directly.
- `lulesh/` — `fork/omp_4.0/` is the baseline; see `benchmarks/lulesh/README.md`.
- `kripke/`, `quicksilver/`, `llamacpp/` — the baseline is upstream; each
  benchmark's top-level `README.md` lists the upstream URL and the patch/file
  pair that reproduces the Leo-guided change.

### Task 4: LLM diagnostic study (Table V, optional)

Reproduces the Table V **semantic-match** numbers (how well the LLM, given each diagnostic context, identifies the optimization site). The paper's **speedup** column additionally requires compiling and benchmarking LLM-generated kernels on GPUs and is out of scope for CPU-only reviewers.

```bash
export OPENROUTER_API_KEY=your_key_here    # obtain from openrouter.ai
uv run python -m scripts.validation.main \
    --llm-eval \
    --llm-model google/gemini-3.1-pro \
    --llm-log results/llm_eval.log
```

Routes to Gemini 3.1 Pro via OpenRouter. Reduce `--llm-concurrency` (default 5) if you hit per-minute rate limits. Pre-collected LLM outputs are included in the artifact for reference.

---

## Expected Results (Tolerance)

- **Table IV speedups:** ±5% on NVIDIA GH200 and AMD MI300A (median CV < 1%); ±9% on Intel PVC (median CV ~9%)
- **Figure 5 SDC coverage:** NVIDIA 30–74% → 64–94% after pruning (exact values within rounding)
- **Table V:** Gemini 3.1 Pro results subject to LLM sampling variance; overall trends (100% compile rate with C+L(S), 1.29× speedup) should hold

---

## Known Deviations from Paper

- Pre-collected data on Zenodo may have slightly different timestamps than those in the paper but produces identical LEO analysis output.
- Intel PVC results exhibit higher run-to-run variability (documented 9% CV); ±9% tolerance recommended.
- LLM study is non-deterministic. The semantic-match evaluation (`--llm-eval`) reports a 0–100 score per kernel/vendor — per-pair values may vary ±10 across runs, but overall trends (relative ranking of diagnostic contexts) reproduce. The paper's best-of-5 speedup column was produced by a separate compile-and-benchmark harness that requires GPUs and is not included here.

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

- Yuning Xia — yuning.xia@rice.edu
- John Mellor-Crummey — johnmc@rice.edu

## Acknowledgments

See `ad_appendix.pdf` for full artifact description.
