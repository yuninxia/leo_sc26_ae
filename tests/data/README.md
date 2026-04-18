# Test Data

Test fixtures (PC sampling binaries + measurement databases) are **not included** in this repository due to size (~4 GB).

To obtain the test data:

1. **Option A: Hugging Face (small fixtures)**
   ```bash
   # Requires HF_TOKEN in .env
   uv run python scripts/upload_testdata_hf.py download
   ```

2. **Option B: Zenodo archive**
   Download from the Zenodo record referenced in the root README.

After downloading, the directory structure should be:
```
tests/data/
├── pc/
│   ├── nvidia/hpctoolkit-single.*
│   ├── amd/hpctoolkit-single.*
│   └── intel/hpctoolkit-single.*
└── ...
```

## Running Tests Without Data

Unit tests that don't require external data will still run:
```bash
uv run pytest tests/ -v -m "not slow" --ignore-glob="*test_integration*"
```
