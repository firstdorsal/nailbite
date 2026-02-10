#!/bin/bash
# Nailbite Tauri Build Script
#
# Builds the Tauri desktop application using Docker for reproducible builds.
# Produces AppImage and deb packages.
#
# Usage:
#   bash build.sh                    # CPU-only build
#   GPU_BACKEND=cuda bash build.sh   # NVIDIA CUDA build
#   GPU_BACKEND=rocm bash build.sh   # AMD ROCm/MIGraphX build
#
# Environment variables:
#   GPU_BACKEND    - GPU acceleration: none (default), cuda, tensorrt, migraphx, rocm
#   OUTPUT_DIR     - Output directory for artifacts (default: ./dist)
#   SKIP_EXTRACT   - Set to 1 to skip extracting artifacts from container
#
set -euo pipefail

SERVICE_NAME="nailbite"
GPU_BACKEND="${GPU_BACKEND:-none}"
OUTPUT_DIR="${OUTPUT_DIR:-./dist}"

export BUILDKIT_PROGRESS="plain"
export DOCKER_BUILDKIT=1

echo "=========================================="
echo "Building ${SERVICE_NAME} Tauri Application"
echo "GPU Backend: ${GPU_BACKEND}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Map GPU backend to Dockerfile and cargo features
case "${GPU_BACKEND}" in
    cuda)
        DOCKERFILE="Dockerfile.cuda"
        CARGO_FEATURES="--features cuda"
        IMAGE_TAG="cuda"
        ;;
    tensorrt)
        DOCKERFILE="Dockerfile.cuda"
        CARGO_FEATURES="--features nvidia-gpu"
        IMAGE_TAG="tensorrt"
        ;;
    migraphx)
        DOCKERFILE="Dockerfile.rocm"
        CARGO_FEATURES="--features migraphx"
        IMAGE_TAG="migraphx"
        ;;
    rocm)
        DOCKERFILE="Dockerfile.rocm"
        CARGO_FEATURES="--features amd-gpu"
        IMAGE_TAG="rocm"
        ;;
    none|*)
        DOCKERFILE="Dockerfile"
        CARGO_FEATURES=""
        IMAGE_TAG="cpu"
        ;;
esac

echo "Using Dockerfile: ${DOCKERFILE}"
echo "Cargo features: ${CARGO_FEATURES:-none}"
echo ""

# Build the Docker image
echo "Building Docker image..."
docker build \
    --build-arg CARGO_FEATURES="${CARGO_FEATURES}" \
    -f "${DOCKERFILE}" \
    -t "${SERVICE_NAME}:build-${IMAGE_TAG}" \
    .

# Extract artifacts
if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
    echo ""
    echo "Extracting build artifacts..."

    mkdir -p "${OUTPUT_DIR}"

    CONTAINER_ID=$(docker create "${SERVICE_NAME}:build-${IMAGE_TAG}")

    # Extract binary
    docker cp "${CONTAINER_ID}:/nailbite" "${OUTPUT_DIR}/nailbite" 2>/dev/null || true

    # Extract bundles (AppImage, deb, rpm)
    docker cp "${CONTAINER_ID}:/bundle" "${OUTPUT_DIR}/" 2>/dev/null || true

    docker rm "${CONTAINER_ID}" > /dev/null

    echo ""
    echo "Build artifacts:"
    echo "----------------"

    # Show binary info
    if [ -f "${OUTPUT_DIR}/nailbite" ]; then
        ls -lh "${OUTPUT_DIR}/nailbite"
        file "${OUTPUT_DIR}/nailbite"
        echo ""
    fi

    # Show bundle info
    if [ -d "${OUTPUT_DIR}/bundle" ]; then
        echo "Bundles:"
        find "${OUTPUT_DIR}/bundle" -type f \( -name "*.AppImage" -o -name "*.deb" -o -name "*.rpm" \) -exec ls -lh {} \;
    fi

    echo ""
    echo "Build complete!"
    echo "Artifacts in: ${OUTPUT_DIR}/"
else
    echo ""
    echo "Build complete (extraction skipped)."
    echo "Use 'docker cp' to extract artifacts manually."
fi
