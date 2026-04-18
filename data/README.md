# Pre-Collected Data

Pre-collected HPCToolkit measurement databases for all 21 workloads × 3 vendors are **not included** in this Git repository due to size (~5.6 GB).

## Download from Zenodo

```bash
# TODO: Zenodo DOI pending upload
# wget https://zenodo.org/record/XXXXXXX/files/leo-sc26-measurements.tar.gz
# tar xzf leo-sc26-measurements.tar.gz -C .
```

After extraction, the directory structure should be:

```
data/hpctoolkit-measurements/
├── nvidia-h100/
│   ├── rajaperf-MASS3DEA/
│   ├── rajaperf-LTIMES/
│   └── ...
├── amd-mi300a/
│   └── ...
└── intel-pvc/
    └── ...
```

Each subdirectory is a complete HPCToolkit measurement database (from `hpcrun`).

## Using Pre-Collected Data

LEO analysis on pre-collected data runs on any x86-64 machine without GPU:

```bash
# Build universal analysis container
bash scripts/evaluation/build_containers.sh universal --base-only

# Run analysis (example)
docker run --rm \
  -v $(pwd)/data/hpctoolkit-measurements/nvidia-h100/rajaperf-MASS3DEA:/data:ro \
  -v $(pwd)/src/leo:/opt/leo/src/leo:ro \
  leo-base-universal -c \
  "uv run python scripts/analyze_benchmark.py /data --arch gh200 --top-n 5"
```

## Why Not in Git?

The full measurement set is ~5.6 GB, exceeding GitHub's file size limits and making clones slow. Zenodo provides persistent DOI-referenced storage and is standard practice for HPC artifacts.
