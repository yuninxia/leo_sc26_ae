#!/bin/bash
# Preflight: verify system build dependencies required for `uv sync` (hpcanalysis
# C++ extension via meson) and archive extraction. Most cloud base images
# (Lambda Stack, RunPod Pytorch, Verda CUDA, Paperspace GPU Droplets) strip these
# out, so reviewers on minimal images hit `Run-time dependency python found: NO`
# during `uv sync` without them.
#
# Usage:
#   bash scripts/preflight.sh            # dry-run: print missing + exact install command; exit 1 if any missing
#   bash scripts/preflight.sh --install  # auto-install via apt-get / dnf (needs sudo unless root)
#   bash scripts/preflight.sh --help
#
# Packages checked:
#   python3-dev (apt) / python3-devel (dnf) — Python.h for meson/pybind11 build of hpcanalysis
#   pkg-config                              — meson dependency detection
#   unzip                                   — extracting the Zenodo archive
set -euo pipefail

INSTALL=false
case "${1:-}" in
  --install) INSTALL=true ;;
  --help|-h) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "Unknown flag: $1 (try --help)" >&2; exit 2 ;;
esac

# --- Detect distro family ---
DISTRO="unknown"
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  DISTRO="${ID:-unknown}"
fi

# --- Check each dependency ---
MISSING=()

command -v pkg-config >/dev/null 2>&1 || MISSING+=("pkg-config")
command -v unzip      >/dev/null 2>&1 || MISSING+=("unzip")

# python3-dev: verified by presence of Python.h under a python3.* include dir
if ! ls /usr/include/python3*/Python.h >/dev/null 2>&1 \
  && ! ls /usr/include/*-linux-gnu/python3*/Python.h >/dev/null 2>&1 ; then
  MISSING+=("python3-dev-headers")
fi

if [ "${#MISSING[@]}" -eq 0 ]; then
  echo "Preflight: all system build dependencies present. (distro=${DISTRO})"
  exit 0
fi

echo "Preflight: missing system build dependencies on ${DISTRO}:"
printf '  - %s\n' "${MISSING[@]}"
echo

# --- Map missing to per-distro package names + install command ---
case "$DISTRO" in
  ubuntu|debian|linuxmint|pop)
    PKGS=()
    for m in "${MISSING[@]}"; do
      case "$m" in
        pkg-config)             PKGS+=(pkg-config) ;;
        unzip)                  PKGS+=(unzip) ;;
        python3-dev-headers)    PKGS+=(python3-dev) ;;
      esac
    done
    CMD="apt-get install -y ${PKGS[*]}"
    UPDATE_CMD="apt-get update -qq"
    ;;
  rhel|centos|fedora|rocky|almalinux)
    PKGS=()
    for m in "${MISSING[@]}"; do
      case "$m" in
        pkg-config)             PKGS+=(pkgconf-pkg-config) ;;
        unzip)                  PKGS+=(unzip) ;;
        python3-dev-headers)    PKGS+=(python3-devel) ;;
      esac
    done
    CMD="dnf install -y ${PKGS[*]}"
    UPDATE_CMD=""
    ;;
  *)
    echo "Unsupported distro: ${DISTRO}. Install equivalents via your package manager:"
    echo "  python3 headers (Python.h), pkg-config, unzip"
    exit 1
    ;;
esac

SUDO=""
[ "$(id -u)" = 0 ] || SUDO="sudo "

echo "Suggested install command:"
[ -n "$UPDATE_CMD" ] && echo "  ${SUDO}${UPDATE_CMD}"
echo "  ${SUDO}${CMD}"
echo

if [ "$INSTALL" != true ]; then
  echo "Re-run with --install to apply automatically: bash scripts/preflight.sh --install"
  exit 1
fi

echo "Installing..."
[ -n "$UPDATE_CMD" ] && eval "${SUDO}${UPDATE_CMD}"
eval "${SUDO}${CMD}"
echo
echo "Preflight: installation complete."
