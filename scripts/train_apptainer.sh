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

apptainer exec \
  "${NV_FLAG[@]}" \
  --bind "${OUTPUT_DIR}:/workspace/outputs" \
  --bind "$(dirname "${DATA_PATH}"):/data:ro" \
  "${SIF_PATH}" \
  python /workspace/train.py --data "/data/$(basename "${DATA_PATH}")" "$@"
