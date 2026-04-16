#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-log-monitor-train:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

docker build \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
  -t "${IMAGE_TAG}" \
  -f "${PROJECT_DIR}/Dockerfile" \
  "${PROJECT_DIR}"
