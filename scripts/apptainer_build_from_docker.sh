#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-log-monitor-train:latest}"
SIF_PATH="${2:-log-monitor-train.sif}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

echo "Building Docker image: ${IMAGE_TAG}"
docker build \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
  -t "${IMAGE_TAG}" \
  -f "${PROJECT_DIR}/Dockerfile" \
  "${PROJECT_DIR}"

echo "Building Apptainer image: ${SIF_PATH}"
apptainer build "${SIF_PATH}" "docker-daemon://${IMAGE_TAG}"

echo "Done: ${SIF_PATH}"
