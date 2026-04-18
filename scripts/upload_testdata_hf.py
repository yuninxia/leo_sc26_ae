#!/usr/bin/env python3
"""Upload/download Leo test data to/from Hugging Face datasets hub.

Usage:
    # Upload all test data to Hugging Face (requires HF_TOKEN with write permission)
    python scripts/upload_testdata_hf.py upload

    # Download additional test data from Hugging Face (e.g., LAMMPS profiles)
    python scripts/upload_testdata_hf.py download

    # Download specific dataset
    python scripts/upload_testdata_hf.py download amd/hpctoolkit-lammps.rocm7.2.0.kokkos4.6.2-measurements

Setup for upload:
    1. Copy .env.example to .env
    2. Add your HF_TOKEN (with WRITE permission) from https://huggingface.co/settings/tokens
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import login, snapshot_download, upload_folder

REPO_ID = "ynxia/leo-test-data"
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data" / "pc"


def load_env():
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def upload_dataset(local_path: Path, remote_path: str):
    """Upload a directory to Hugging Face."""
    print(f"Uploading {local_path.name} -> {remote_path}")
    upload_folder(
        folder_path=str(local_path),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=remote_path,
    )
    print(f"  Done: https://huggingface.co/datasets/{REPO_ID}/tree/main/{remote_path}")


def download_dataset(pattern: str | None = None):
    """Download test data from Hugging Face.

    Args:
        pattern: Optional pattern to filter downloads (e.g., "amd/hpctoolkit-lammps*").
                 If None, downloads all datasets not already present locally.
    """
    print(f"Downloading from https://huggingface.co/datasets/{REPO_ID}")

    # Download to test data directory
    allow_patterns = None
    if pattern:
        allow_patterns = [f"{pattern}/**"] if not pattern.endswith("*") else [pattern]
        print(f"  Pattern: {pattern}")

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=TEST_DATA_DIR,
        allow_patterns=allow_patterns,
    )
    print(f"  Downloaded to: {local_dir}")
    return local_dir


def upload_all():
    """Upload all local test data to Hugging Face."""
    # Load .env file
    load_env()

    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found.")
        print("Please copy .env.example to .env and add your Hugging Face token.")
        print("Get a token with WRITE permission at: https://huggingface.co/settings/tokens")
        return

    # Login to Hugging Face
    login(token=token)

    # Upload test data for all vendors (amd, nvidia, intel, etc.)
    for vendor_dir in sorted(TEST_DATA_DIR.iterdir()):
        if vendor_dir.is_dir():
            vendor = vendor_dir.name
            if vendor == "per-kernel":
                # Per-kernel data: per-kernel/<kernel>/<vendor>/hpctoolkit-*
                for kernel_dir in sorted(vendor_dir.iterdir()):
                    if kernel_dir.is_dir():
                        upload_dataset(kernel_dir, f"per-kernel/{kernel_dir.name}")
                continue
            for dataset_dir in sorted(vendor_dir.iterdir()):
                if dataset_dir.is_dir() and dataset_dir.name.startswith("hpctoolkit-"):
                    upload_dataset(dataset_dir, f"{vendor}/{dataset_dir.name}")

    print(f"\nAll uploads complete: https://huggingface.co/datasets/{REPO_ID}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload/download Leo test data to/from Hugging Face"
    )
    parser.add_argument(
        "action",
        choices=["upload", "download"],
        help="Action to perform",
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="For download: optional pattern to filter (e.g., 'amd/hpctoolkit-lammps*')",
    )
    args = parser.parse_args()

    if args.action == "upload":
        upload_all()
    elif args.action == "download":
        download_dataset(args.pattern)


if __name__ == "__main__":
    main()
