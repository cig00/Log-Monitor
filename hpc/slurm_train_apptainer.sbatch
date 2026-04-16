#!/usr/bin/env bash
#SBATCH --job-name=log-monitor-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# Expected to be provided at submit time:
# sbatch --export=ALL,SIF_PATH=/path/log-monitor-train.sif,DATA_PATH=/path/labeled.csv hpc/slurm_train_apptainer.sbatch

if [[ -z "${SIF_PATH:-}" ]]; then
  echo "SIF_PATH is required."
  exit 1
fi

if [[ -z "${DATA_PATH:-}" ]]; then
  echo "DATA_PATH is required."
  exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "SIF image not found: ${SIF_PATH}"
  exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Data file not found: ${DATA_PATH}"
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/outputs}"
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

apptainer exec --nv \
  "${BIND_FLAGS[@]}" \
  "${SIF_PATH}" \
  python /workspace/train.py --data "/data/$(basename "${DATA_PATH}")"
