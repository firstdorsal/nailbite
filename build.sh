#!/bin/bash
set -euo pipefail

SERVICE_NAME="nailbite"
BINARY_NAME="nailbite"
PROFILE="${PROFILE:-release}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

export BUILDKIT_PROGRESS="plain"
export DOCKER_BUILDKIT=1

echo "Building ${SERVICE_NAME} (profile: ${PROFILE})"

# Build the Docker image
docker build \
    --build-arg PROFILE="${PROFILE}" \
    --build-arg SERVICE_NAME="${SERVICE_NAME}" \
    --build-arg BINARY_NAME="${BINARY_NAME}" \
    -t "${SERVICE_NAME}:build" \
    .

# Extract the binary from the image
CONTAINER_ID=$(docker create "${SERVICE_NAME}:build")
docker cp "${CONTAINER_ID}:/${BINARY_NAME}" "${OUTPUT_DIR}/${BINARY_NAME}"
docker rm "${CONTAINER_ID}" > /dev/null

# Show binary info
ls -lh "${OUTPUT_DIR}/${BINARY_NAME}"
file "${OUTPUT_DIR}/${BINARY_NAME}"

echo "Binary extracted to ${OUTPUT_DIR}/${BINARY_NAME}"
