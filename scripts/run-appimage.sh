#!/bin/bash
# Build and run the nailbite AppImage.
#
# Usage:
#   bash scripts/run-appimage.sh              # Build + run
#   bash scripts/run-appimage.sh --skip-build # Run existing AppImage without rebuilding
#   bash scripts/run-appimage.sh --clean      # Clean build cache and rebuild
#
# Environment variables (passed through to build.sh):
#   GPU_BACKEND       - GPU acceleration: none (default), cuda, tensorrt, migraphx, rocm
#   RUST_LOG          - Logging level (default: info)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${PROJECT_DIR}/dist"
APPIMAGE_DIR="${OUTPUT_DIR}/bundle/appimage"
SKIP_BUILD=0
CLEAN=0
RUST_LOG="${RUST_LOG:-info}"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --skip-build)
            SKIP_BUILD=1
            ;;
        --clean)
            CLEAN=1
            ;;
        --help|-h)
            echo "Usage: bash scripts/run-appimage.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-build   Run existing AppImage without rebuilding"
            echo "  --clean        Remove dist/ and rebuild from scratch"
            echo "  --help, -h     Show this help"
            echo ""
            echo "Environment:"
            echo "  GPU_BACKEND    GPU acceleration (none, cuda, tensorrt, migraphx, rocm)"
            echo "  RUST_LOG       Logging level (default: info)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "${CLEAN}" = "1" ]; then
    echo "Cleaning dist/..."
    rm -rf "${OUTPUT_DIR}"
fi

# Build
if [ "${SKIP_BUILD}" = "0" ]; then
    echo "=========================================="
    echo "Building nailbite AppImage..."
    echo "=========================================="
    echo ""

    cd "${PROJECT_DIR}"
    bash build.sh

    echo ""
fi

# Find the AppImage
APPIMAGE=""
if [ -d "${APPIMAGE_DIR}" ]; then
    APPIMAGE=$(find "${APPIMAGE_DIR}" -maxdepth 1 -name "*.AppImage" -type f | head -n 1)
fi

if [ -z "${APPIMAGE}" ] || [ ! -f "${APPIMAGE}" ]; then
    echo "ERROR: No AppImage found in ${APPIMAGE_DIR}" >&2
    echo "Run without --skip-build to build first." >&2
    exit 1
fi

echo "=========================================="
echo "Running: $(basename "${APPIMAGE}")"
echo "RUST_LOG=${RUST_LOG}"
echo "=========================================="
echo ""

# Run from the project root so the app sees `config.yaml` and `./models/`.
# `appimage-run` on NixOS launches via `bwrap --chdir <invoker cwd> ...`, so
# the invoker's cwd determines what the app sees as `.`. The previous
# `cd "${APPIMAGE_DIR}"` made config + cached models invisible to the app,
# which forced re-download on every launch and printed a misleading
# "Failed to read config file config.yaml" warning.
cd "${PROJECT_DIR}"

if command -v appimage-run &>/dev/null; then
    RUST_LOG="${RUST_LOG}" exec appimage-run "${APPIMAGE}"
else
    # Non-NixOS: run directly (AppImage is self-extracting)
    chmod +x "${APPIMAGE}"
    RUST_LOG="${RUST_LOG}" exec "${APPIMAGE}"
fi
