# XSBench (Section VI case study)

XSBench is a Monte Carlo neutron-transport proxy application (Argonne).
Leo identifies irregular-load stalls in the cross-section lookup kernel
and guides an integer-hash / extract optimization (plus a hybrid-search +
USM variant on Intel).

## What's in this directory

- `original/` — upstream [ANL-CESAR/XSBench](https://github.com/ANL-CESAR/XSBench)
  source, vendored as-is for A/B diffing.
- `optimized/` — Leo-guided variant (vendor-agnostic primary + Intel-specific
  hybrid path).
- `run_compare_{amd,intel,nvidia}.sh` — build and benchmark both sides in
  the vendor-specific Docker image; one script per vendor.

## Diffing the Leo change

The diff is self-contained — no upstream clone needed:

```bash
diff -ruN original/ optimized/
```

Each hunk corresponds to the lookup-kernel optimization described in
Section VI of the paper.

## Running

```bash
bash benchmarks/xsbench/run_compare_nvidia.sh   # on GH200
bash benchmarks/xsbench/run_compare_amd.sh      # on MI300A
bash benchmarks/xsbench/run_compare_intel.sh    # on GPU Max 1100
```

Prerequisite: build the vendor Docker image, e.g.
`docker build -f scripts/evaluation/docker/Dockerfile.xsbench-nvidia -t leo-xsbench-nvidia .`
from the artifact root. See the main AE appendix for details.
