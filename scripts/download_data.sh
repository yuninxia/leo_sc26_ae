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
GH_REPO="yuninxia/leo_sc26_ae"
GH_URL="https://github.com/${GH_REPO}/releases/download/${RELEASE_TAG}/${TARBALL}"

# Optional Zenodo archival mirror for the measurement tarball. The Zenodo source
# archive (DOI 10.5281/zenodo.19704349) contains only the repo snapshot, not the
# ~1 GB measurements; if a separate Zenodo data deposit is made later, set
# ZENODO_URL (or edit here) to the direct file URL.
ZENODO_URL="${ZENODO_URL:-}"

if [ -d "${DATA_DIR}/per-kernel" ] && [ -n "$(ls -A "${DATA_DIR}/per-kernel" 2>/dev/null)" ]; then
  echo "Data already present at ${DATA_DIR}"
  echo "To re-download, first remove: rm -rf ${DATA_DIR}"
  exit 0
fi

mkdir -p "${DATA_DIR}"
TMP="/tmp/${TARBALL}"

download_curl() {
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

download_gh() {
  # Use `gh` CLI when available — works even if the repo is private,
  # provided the user has run `gh auth login`.
  if ! command -v gh >/dev/null 2>&1; then
    return 1
  fi
  if ! gh auth status >/dev/null 2>&1; then
    return 1
  fi
  echo "Downloading via gh CLI (repo: ${GH_REPO}, tag: ${RELEASE_TAG})..."
  gh release download "${RELEASE_TAG}" --repo "${GH_REPO}" \
    --pattern "${TARBALL}" --output "${TMP}" --clobber
}

ok=0
if download_curl "${GH_URL}"; then
  ok=1
elif download_gh; then
  ok=1
elif [ -n "${ZENODO_URL}" ] && download_curl "${ZENODO_URL}"; then
  ok=1
fi

if [ "$ok" -ne 1 ]; then
  echo "ERROR: failed to download tarball."
  echo "  - If the GitHub repo is private, authenticate with: gh auth login"
  echo "  - Or set ZENODO_URL to an archival mirror and re-run."
  exit 1
fi

echo ""
echo "Extracting ~5.6 GB to ${DATA_DIR} ..."
tar -xzf "${TMP}" -C "${LEO_ROOT}" --strip-components=0
rm -f "${TMP}"

echo ""
echo "Done. Data available at ${DATA_DIR}/"
echo ""
echo "You can now run:"
echo "  bash scripts/collect_sdc.sh      # Figure 5 (SDC coverage)"
echo "  bash scripts/time_analysis.sh    # LEO analysis timing"
