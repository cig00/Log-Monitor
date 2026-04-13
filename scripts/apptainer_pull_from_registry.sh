#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <docker_image_ref> [sif_path]"
  echo "Example: $0 ghcr.io/<org>/log-monitor-train:latest log-monitor-train.sif"
  exit 1
fi

IMAGE_REF="$1"
SIF_PATH="${2:-log-monitor-train.sif}"

apptainer pull "${SIF_PATH}" "docker://${IMAGE_REF}"
echo "Pulled Apptainer image to ${SIF_PATH}"
