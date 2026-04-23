# miniBUDE (Section VI case study)

miniBUDE is a molecular-docking mini-app (University of Bristol). Leo
identifies three different root causes on the three vendors (LDG stall on
NVIDIA, redundant vector loads on AMD, SYCL-accessor control-flow overhead on
Intel), each needing a vendor-specific code change.

## What's in this directory

- `original/` — upstream [UoB-HPC/miniBUDE](https://github.com/UoB-HPC/miniBUDE)
  source, vendored as-is for A/B diffing.
- `optimized/` — Leo-guided variants (vendor-specific patches applied inline).
- `run_compare.sh` — build and benchmark both sides in the vendor-specific
  Docker image; accepts `--vendor {nvidia,amd,intel}`.
- `sweep_ppwi.sh` — per-kernel parameter sweep (poses-per-work-item).

## Diffing the Leo change

The diff is self-contained in this directory — no upstream clone needed:

```bash
diff -ruN original/ optimized/
```

Each hunk corresponds to the vendor-specific optimization described in
Section VI of the paper.

## Running

```bash
bash benchmarks/minibude/run_compare.sh --vendor nvidia   # on GH200
bash benchmarks/minibude/run_compare.sh --vendor amd      # on MI300A
bash benchmarks/minibude/run_compare.sh --vendor intel    # on GPU Max 1100
```

Prerequisite: build the vendor Docker image, e.g.
`docker build -f scripts/evaluation/docker/Dockerfile.minibude-nvidia -t leo-minibude-nvidia .`
from the artifact root. See the main AE appendix for details.
