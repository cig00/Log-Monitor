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

resolve_local_tracking_uri() {
  local uri="$1"
  if [[ -z "${uri}" ]]; then
    return 1
  fi
  if [[ "${uri}" == file://* ]]; then
    printf '%s\n' "${uri#file://}"
    return 0
  fi
  if [[ "${uri}" == *"://"* ]]; then
    return 1
  fi
  if [[ "${uri}" == /* ]]; then
    printf '%s\n' "${uri}"
    return 0
  fi
  printf '%s\n' "$(cd "$(dirname "${uri}")" && pwd)/$(basename "${uri}")"
  return 0
}

GPU_FLAGS=()
if [[ "${DEVICE}" == "cuda" ]]; then
  GPU_FLAGS=(--gpus all)
fi

ENV_FLAGS=()
VOLUME_FLAGS=(
  -v "${OUTPUT_DIR}:/workspace/outputs"
  -v "$(dirname "${DATA_PATH}"):/data:ro"
)
if [[ "${DEVICE}" == "cpu" ]]; then
  ENV_FLAGS=(-e CUDA_VISIBLE_DEVICES=-1)
fi

MLOPS_KEYS=(
  MLOPS_ENABLED
  MLFLOW_TRACKING_URI
  MLFLOW_EXPERIMENT_NAME
  MLFLOW_PIPELINE_ID
  MLFLOW_PARENT_RUN_ID
  MLFLOW_RUN_SOURCE
  MLFLOW_TAGS_JSON
)

for key in "${MLOPS_KEYS[@]}"; do
  if [[ -n "${!key:-}" ]]; then
    ENV_FLAGS+=(-e "${key}=${!key}")
  fi
done

if [[ "${MLOPS_ENABLED:-0}" == "1" ]]; then
  if HOST_MLFLOW_PATH="$(resolve_local_tracking_uri "${MLFLOW_TRACKING_URI:-}")"; then
    mkdir -p "${HOST_MLFLOW_PATH}"
    VOLUME_FLAGS+=(-v "${HOST_MLFLOW_PATH}:/workspace/mlruns")
    ENV_FLAGS+=(-e "MLFLOW_TRACKING_URI=/workspace/mlruns")
  fi
fi

docker run --rm \
  "${GPU_FLAGS[@]}" \
  "${ENV_FLAGS[@]}" \
  "${VOLUME_FLAGS[@]}" \
  "${IMAGE_TAG}" \
  python train.py --data "/data/$(basename "${DATA_PATH}")" "$@"
