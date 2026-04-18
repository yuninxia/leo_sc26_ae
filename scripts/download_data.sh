#!/bin/bash
# Download pre-collected HPCToolkit measurements from GitHub Release.
# Reviewers run this once after `git clone` to enable all reproduction scripts.
#
# Usage: bash scripts/download_data.sh
set -euo pipefail

LEO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${LEO_ROOT}/results"
TARBALL="leo-sc26-measurements.tar.gz"
RELEASE_TAG="v1.0-sc26-data"
GH_URL="https://github.com/yuninxia/leo_sc26_ae/releases/download/${RELEASE_TAG}/${TARBALL}"

# Optional Zenodo mirror (set ZENODO_URL env var or edit below once DOI is assigned)
ZENODO_URL="${ZENODO_URL:-}"

if [ -d "${DATA_DIR}/per-kernel" ] && [ -n "$(ls -A "${DATA_DIR}/per-kernel" 2>/dev/null)" ]; then
  echo "Data already present at ${DATA_DIR}"
  echo "To re-download, first remove: rm -rf ${DATA_DIR}"
  exit 0
fi

mkdir -p "${DATA_DIR}"
TMP="/tmp/${TARBALL}"

download_from() {
  local url="$1"
  echo "Downloading from: ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${TMP}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "${TMP}" "${url}"
  else
    echo "ERROR: need curl or wget"
    return 1
  fi
}

if ! download_from "${GH_URL}"; then
  if [ -n "${ZENODO_URL}" ]; then
    echo "GitHub download failed; trying Zenodo mirror..."
    download_from "${ZENODO_URL}"
  else
    echo "ERROR: failed to download from GitHub and no Zenodo mirror configured"
    exit 1
  fi
fi

echo ""
echo "Extracting ~1.9 GB to ${DATA_DIR} ..."
tar -xzf "${TMP}" -C "${LEO_ROOT}" --strip-components=0
rm -f "${TMP}"

echo ""
echo "Done. Data available at ${DATA_DIR}/"
echo ""
echo "You can now run:"
echo "  bash scripts/collect_sdc.sh      # Figure 5 (SDC coverage)"
echo "  bash scripts/time_analysis.sh    # LEO analysis timing"
