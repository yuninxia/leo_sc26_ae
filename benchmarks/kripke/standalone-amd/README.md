# Kripke — AMD MI300A (Standalone Build)

Self-contained build and run for Kripke on AMD MI300A.
No dependencies on Leo or HPCToolkit — only Docker + AMD GPU needed.

## Quick Start

```bash
# 1. Build the container (~5 min, one-time)
docker build -t kripke-amd .

# 2. Run
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video kripke-amd

# 3. Or run with custom parameters
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  kripke-amd kripke.exe --zones 32,32,32 --groups 32 --niter 10
```

## Requirements

- Docker with AMD GPU access (`--device=/dev/kfd --device=/dev/dri`)
- AMD MI300A (gfx942)
