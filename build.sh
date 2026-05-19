#!/bin/bash
# nailbite Tauri Build Script
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
#   GPU_BACKEND       - GPU acceleration: none (default), cuda, tensorrt, migraphx, rocm
#   BUILD_MODE        - Build mode: dev (default) or release
#   OUTPUT_DIR        - Output directory for artifacts (default: ./dist)
#   RENAME_ARTIFACTS  - Set to 1 to rename artifacts with GPU suffix
#   BUILDX_CACHE      - Cache type: gha (GitHub Actions), local, or none (default)
#   SKIP_EXTRACT      - Set to 1 to skip extracting artifacts from container
#
set -euo pipefail

SERVICE_NAME="nailbite"
GPU_BACKEND="${GPU_BACKEND:-none}"
BUILD_MODE="${BUILD_MODE:-dev}"
OUTPUT_DIR="${OUTPUT_DIR:-./dist}"
RENAME_ARTIFACTS="${RENAME_ARTIFACTS:-0}"
BUILDX_CACHE="${BUILDX_CACHE:-none}"

export BUILDKIT_PROGRESS="plain"
export DOCKER_BUILDKIT=1

# Extract version from Cargo.toml
VERSION=$(grep -m1 '^version = ' src-tauri/Cargo.toml | sed 's/version = "\(.*\)"/\1/')

echo "=========================================="
echo "Building ${SERVICE_NAME} Tauri Application"
echo "Version: ${VERSION}"
echo "GPU Backend: ${GPU_BACKEND}"
echo "Build Mode: ${BUILD_MODE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Rename Artifacts: ${RENAME_ARTIFACTS}"
echo "Buildx Cache: ${BUILDX_CACHE}"
echo "=========================================="

# Map GPU backend to base image, cargo features, and ROCm-ORT flag.
# `ROCM_ORT=1` switches the Dockerfile from the upstream Microsoft CPU
# tarball to the ROCm-enabled libonnxruntime.so extracted from the
# `onnxruntime-rocm` PyPI wheel — Microsoft does not publish ROCm
# tarballs, so the wheel is the only ready-made source.
ROCM_ORT="0"
case "${GPU_BACKEND}" in
    cuda)
        BASE_IMAGE="nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04"
        CARGO_FEATURES="--features cuda"
        IMAGE_TAG="cuda"
        GPU_SUFFIX="cuda"
        ;;
    tensorrt)
        BASE_IMAGE="nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04"
        CARGO_FEATURES="--features nvidia-gpu"
        IMAGE_TAG="tensorrt"
        GPU_SUFFIX="tensorrt"
        ;;
    migraphx)
        BASE_IMAGE="ubuntu:24.04"
        CARGO_FEATURES="--features migraphx"
        IMAGE_TAG="migraphx"
        GPU_SUFFIX="migraphx"
        ROCM_ORT="1"
        ;;
    rocm)
        BASE_IMAGE="ubuntu:24.04"
        CARGO_FEATURES="--features amd-gpu"
        IMAGE_TAG="rocm"
        GPU_SUFFIX="rocm"
        ROCM_ORT="1"
        ;;
    none|*)
        BASE_IMAGE="ubuntu:24.04"
        CARGO_FEATURES=""
        IMAGE_TAG="cpu"
        GPU_SUFFIX="cpu"
        ;;
esac

echo "Base image: ${BASE_IMAGE}"
echo "Cargo features: ${CARGO_FEATURES:-none}"
echo "ROCm ORT mode: ${ROCM_ORT}"
echo ""

# Build cache arguments based on BUILDX_CACHE setting
CACHE_ARGS=""
case "${BUILDX_CACHE}" in
    gha)
        CACHE_ARGS="--cache-from type=gha,scope=${SERVICE_NAME}-${GPU_SUFFIX} --cache-to type=gha,mode=max,scope=${SERVICE_NAME}-${GPU_SUFFIX}"
        ;;
    local)
        CACHE_ARGS="--cache-from type=local,src=/tmp/.buildx-cache --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max"
        ;;
    *)
        CACHE_ARGS=""
        ;;
esac

# Build the Docker image
echo "Building Docker image..."
if [ -n "${CACHE_ARGS}" ]; then
    # Use buildx for caching
    docker buildx build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg CARGO_FEATURES="${CARGO_FEATURES}" \
        --build-arg ROCM_ORT="${ROCM_ORT}" \
        ${CACHE_ARGS} \
        -t "${SERVICE_NAME}:build-${IMAGE_TAG}" \
        --load \
        .
else
    # Standard docker build
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg CARGO_FEATURES="${CARGO_FEATURES}" \
        --build-arg ROCM_ORT="${ROCM_ORT}" \
        -t "${SERVICE_NAME}:build-${IMAGE_TAG}" \
        .
fi

# Handle local cache rotation (avoids cache growing indefinitely)
if [ "${BUILDX_CACHE}" = "local" ] && [ -d "/tmp/.buildx-cache-new" ]; then
    rm -rf /tmp/.buildx-cache
    mv /tmp/.buildx-cache-new /tmp/.buildx-cache
fi

# Extract artifacts
if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
    echo ""
    echo "Extracting build artifacts..."

    mkdir -p "${OUTPUT_DIR}"
    # Convert to absolute path for reliable subshell operations
    OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

    # Pass dummy command since scratch image has no CMD/ENTRYPOINT
    CONTAINER_ID=$(docker create "${SERVICE_NAME}:build-${IMAGE_TAG}" /nonexistent)

    # Extract binary
    docker cp "${CONTAINER_ID}:/nailbite" "${OUTPUT_DIR}/nailbite" 2>/dev/null || true

    # Extract bundles (AppImage, deb, rpm)
    docker cp "${CONTAINER_ID}:/bundle" "${OUTPUT_DIR}/" 2>/dev/null || true

    docker rm "${CONTAINER_ID}" > /dev/null

    # Rename artifacts if requested
    if [ "${RENAME_ARTIFACTS}" = "1" ]; then
        echo ""
        echo "Renaming artifacts with GPU suffix..."

        # Rename binary
        if [ -f "${OUTPUT_DIR}/nailbite" ]; then
            mv "${OUTPUT_DIR}/nailbite" "${OUTPUT_DIR}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}"
        fi

        # Rename bundle files
        if [ -d "${OUTPUT_DIR}/bundle" ]; then
            # Find and rename AppImage files (only original Tauri-generated names)
            find "${OUTPUT_DIR}/bundle" -name "nailbite_*.AppImage" -type f | while read -r file; do
                dir=$(dirname "$file")
                # New name: nailbite_<version>_amd64-<gpu>.AppImage
                new_name="${dir}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}.AppImage"
                mv "$file" "$new_name"
            done

            # Find and rename deb files (only original Tauri-generated names)
            find "${OUTPUT_DIR}/bundle" -name "nailbite_*.deb" -type f | while read -r file; do
                dir=$(dirname "$file")
                # New name: nailbite_<version>_amd64-<gpu>.deb
                new_name="${dir}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}.deb"
                mv "$file" "$new_name"
            done

            # Find and rename rpm files (only original Tauri-generated names)
            find "${OUTPUT_DIR}/bundle" -name "nailbite-*.rpm" -type f | while read -r file; do
                dir=$(dirname "$file")
                # New name: nailbite_<version>_amd64-<gpu>.rpm
                new_name="${dir}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}.rpm"
                mv "$file" "$new_name"
            done
        fi
    fi

    # Generate checksums in release mode
    if [ "${BUILD_MODE}" = "release" ]; then
        echo ""
        echo "Generating SHA256 checksums..."

        CHECKSUM_FILE="${OUTPUT_DIR}/SHA256SUMS-${GPU_SUFFIX}.txt"
        : > "${CHECKSUM_FILE}"

        # Checksum for binary
        if [ -f "${OUTPUT_DIR}/nailbite" ]; then
            (cd "${OUTPUT_DIR}" && sha256sum "nailbite" >> "SHA256SUMS-${GPU_SUFFIX}.txt")
        elif [ -f "${OUTPUT_DIR}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}" ]; then
            (cd "${OUTPUT_DIR}" && sha256sum "nailbite_${VERSION}_amd64-${GPU_SUFFIX}" >> "SHA256SUMS-${GPU_SUFFIX}.txt")
        fi

        # Checksum for bundles
        if [ -d "${OUTPUT_DIR}/bundle" ]; then
            find "${OUTPUT_DIR}/bundle" -type f \( -name "*.AppImage" -o -name "*.deb" -o -name "*.rpm" \) | while read -r file; do
                filename=$(basename "$file")
                (cd "$(dirname "$file")" && sha256sum "$filename" >> "${OUTPUT_DIR}/SHA256SUMS-${GPU_SUFFIX}.txt")
            done
        fi

        echo "Checksums written to: ${CHECKSUM_FILE}"
        cat "${CHECKSUM_FILE}"
    fi

    echo ""
    echo "Build artifacts:"
    echo "----------------"

    # Show binary info
    if [ -f "${OUTPUT_DIR}/nailbite" ]; then
        ls -lh "${OUTPUT_DIR}/nailbite"
        file "${OUTPUT_DIR}/nailbite" 2>/dev/null || true
        echo ""
    elif [ -f "${OUTPUT_DIR}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}" ]; then
        ls -lh "${OUTPUT_DIR}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}"
        file "${OUTPUT_DIR}/nailbite_${VERSION}_amd64-${GPU_SUFFIX}" 2>/dev/null || true
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
