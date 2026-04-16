#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sif_path> <labeled_csv_path> [extra train.py args...]"
  exit 1
fi

SIF_PATH="$1"
DATA_PATH="$2"
shift 2

if [[ "${DATA_PATH}" != /* ]]; then
  DATA_PATH="$(cd "$(dirname "${DATA_PATH}")" && pwd)/$(basename "${DATA_PATH}")"
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "SIF image not found: ${SIF_PATH}"
  exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Data file not found: ${DATA_PATH}"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

NV_FLAG=()
if [[ "${DEVICE}" == "cuda" ]]; then
  NV_FLAG=(--nv)
fi

if [[ "${DEVICE}" == "cpu" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES=-1
else
  unset APPTAINERENV_CUDA_VISIBLE_DEVICES || true
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
    export "APPTAINERENV_${key}=${!key}"
  fi
done

BIND_FLAGS=(
  --bind "${OUTPUT_DIR}:/workspace/outputs"
  --bind "$(dirname "${DATA_PATH}"):/data:ro"
)

if [[ "${MLOPS_ENABLED:-0}" == "1" ]]; then
  if HOST_MLFLOW_PATH="$(resolve_local_tracking_uri "${MLFLOW_TRACKING_URI:-}")"; then
    mkdir -p "${HOST_MLFLOW_PATH}"
    export APPTAINERENV_MLFLOW_TRACKING_URI="/workspace/mlruns"
    BIND_FLAGS+=(--bind "${HOST_MLFLOW_PATH}:/workspace/mlruns")
  fi
fi

apptainer exec \
  "${NV_FLAG[@]}" \
  "${BIND_FLAGS[@]}" \
  "${SIF_PATH}" \
  python /workspace/train.py --data "/data/$(basename "${DATA_PATH}")" "$@"
