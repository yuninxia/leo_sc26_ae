# Pre-Collected Data

The `scripts/download_data.sh` script fetches pre-collected HPCToolkit measurement databases and extracts them to `results/` (sibling of this directory), which is where all reproduction scripts expect the data.

**Do not commit the downloaded data.** The `.gitignore` already excludes `results/` so you won't accidentally check it in.

## What Gets Downloaded

17 directories containing HPCToolkit databases for:
- 16 HPC application measurements (6 workloads × 2-3 vendors)
- Per-kernel RAJAPerf data (15 kernels × 4 vendor variants)

Total: ~8.5 GB uncompressed, ~1 GB compressed. Most of the size (~7.3 GB) is per-kernel RAJAPerf data; the 16 HPC app measurements add ~1.2 GB.

## Primary Source: GitHub Release

```bash
bash scripts/download_data.sh
```

Downloads from: https://github.com/yuninxia/leo_sc26_ae/releases/download/v1.0-sc26-data/leo-sc26-measurements.tar.gz

## Archival Mirror: Zenodo (pending DOI)

If the GitHub release is unavailable, set `ZENODO_URL` and re-run:
```bash
ZENODO_URL="https://zenodo.org/record/XXXXXXX/files/leo-sc26-measurements.tar.gz" \
  bash scripts/download_data.sh
```
