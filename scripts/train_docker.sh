#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <labeled_csv_path> [extra train.py args...]"
  exit 1
fi

DATA_PATH="$1"
shift

if [[ "${DATA_PATH}" != /* ]]; then
  DATA_PATH="$(cd "$(dirname "${DATA_PATH}")" && pwd)/$(basename "${DATA_PATH}")"
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Data file not found: ${DATA_PATH}"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-log-monitor-train:latest}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"

mkdir -p "${OUTPUT_DIR}"

GPU_FLAGS=()
if [[ "${DEVICE}" == "cuda" ]]; then
  GPU_FLAGS=(--gpus all)
fi

ENV_FLAGS=()
if [[ "${DEVICE}" == "cpu" ]]; then
  ENV_FLAGS=(-e CUDA_VISIBLE_DEVICES=-1)
fi

docker run --rm \
  "${GPU_FLAGS[@]}" \
  "${ENV_FLAGS[@]}" \
  -v "${OUTPUT_DIR}:/workspace/outputs" \
  -v "$(dirname "${DATA_PATH}"):/data:ro" \
  "${IMAGE_TAG}" \
  python train.py --data "/data/$(basename "${DATA_PATH}")" "$@"
