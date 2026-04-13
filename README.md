# Log-Monitor

Log-Monitor is a desktop app for:
- Preparing labeled log data with GPT-4o.
- Training a DeBERTa classifier on Azure ML (CPU or GPU) or locally.
- Running local/HPC training through reproducible Docker/Apptainer workflows.

## Repository Layout

- `app.py`: Tkinter UI + orchestration for data prep and training.
- `train.py`: DeBERTa training script.
- `prompt.txt`: prompt used by the data labeling pipeline.
- `requirements.txt`: app/runtime dependencies.
- `requirements.train.txt`: training/container dependencies.
- `Dockerfile`: container build for training.
- `scripts/`: helper scripts for Docker and Apptainer.
- `hpc/slurm_train_apptainer.sbatch`: Slurm example for Apptainer GPU jobs.

## Prerequisites

- Python 3.10+ (3.11 recommended).
- Azure subscription + tenant access (for Azure training).
- Docker (for container runtime).
- Apptainer (for HPC / `.sif` workflows).

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

## UI Overview

### 1) Data Processing

- `Log File (CSV)`: input raw logs.
- `Prepare Data (GPT-4o)`: labels logs and saves a labeled CSV.
- Uses the GitHub PAT entered in `Hosting` for API authentication.

### 2) Model Training (DeBERTa)

- `Azure Sub ID`, `Tenant ID`: required for Azure mode.
- `Azure Compute`: `cpu` or `gpu` (used only in Azure mode).
- `Labeled Data (CSV)`: training file used by both Azure and local training.
- `Environment`:
  - `Azure Cloud`
  - `Local Device (CPU/GPU)`
- `Local Device`: `auto`, `cpu`, `cuda`
- `Local Runtime`: `host` or `container` (policy-driven)

Local runtime policy:
- If local device is `auto` or `cpu`, runtime is forced to `host`.
- If local device is `cuda`, runtime selector is enabled and defaults to `container` when switching to `cuda`.

### 3) Hosting

- GitHub PAT, repository, branch controls.
- `Host Service` is currently a placeholder button.

## Training Behavior

### Local Training

`app.py` launches `train.py` in a background process (host) or via Docker (container).

- Local + `cpu` + `host`:
  - Runs `python train.py --data <csv>`
  - Sets `CUDA_VISIBLE_DEVICES=-1` to force CPU.
- Local + `auto` + `host`:
  - Runs `python train.py --data <csv>`
  - `train.py` auto-selects CUDA if available, else CPU.
- Local + `cuda` + `host`:
  - Preflight checks `torch.cuda.is_available()` in host Python.
  - Runs host training only if CUDA is available.
- Local + `cuda` + `container`:
  - Runs `scripts/train_docker.sh` with `DEVICE=cuda`.

Artifacts are saved to:

```text
./outputs/final_model
```

### Azure Training

Azure mode provisions a temporary Azure ML compute cluster and submits a command job.

Current mapping:
- `Azure Compute=cpu` -> `Standard_D2as_v4` on `cpu-cluster-temp`
- `Azure Compute=gpu` -> `Standard_NC4as_T4_v3` on `gpu-cluster-temp`

The Azure job keeps the training fix currently implemented in `app.py`:
- normalizes Windows paths for Azure input URIs
- installs pinned packages compatible with the curated Azure image
- runs `USE_TF=0 python train.py --data ...`

After completion (or failure path), the temporary compute cluster is deleted by cleanup logic.

## Labeled CSV Requirements

`train.py` expects:
- `LogMessage` column
- `class` column with values mapped from:
  - `Error`
  - `CONFIGURATION`
  - `SYSTEM`
  - `Noise`

Invalid rows are dropped before training. If all rows are invalid after mapping, training fails fast.

## Container Workflows

The container source of truth is:
- `Dockerfile`
- `requirements.train.txt`

Build Docker image:

```bash
./scripts/docker_build_train_image.sh log-monitor-train:latest
```

Optional Torch index override (for CUDA wheels, etc.):

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./scripts/docker_build_train_image.sh log-monitor-train:latest
```

Run training in Docker:

```bash
./scripts/train_docker.sh /absolute/path/to/labeled.csv
```

Supported environment variables:
- `DEVICE=auto|cpu|cuda` (default `auto`)
- `IMAGE_TAG` (default `log-monitor-train:latest`)
- `OUTPUT_DIR` (default `./outputs`)

CUDA Docker example:

```bash
DEVICE=cuda ./scripts/train_docker.sh /absolute/path/to/labeled.csv
```

## Apptainer / HPC Workflows

Build `.sif` from local Docker image:

```bash
./scripts/apptainer_build_from_docker.sh log-monitor-train:latest log-monitor-train.sif
```

Pull `.sif` from registry image:

```bash
./scripts/apptainer_pull_from_registry.sh ghcr.io/<org>/log-monitor-train:latest log-monitor-train.sif
```

Run training with Apptainer:

```bash
DEVICE=cuda ./scripts/train_apptainer.sh /path/to/log-monitor-train.sif /absolute/path/to/labeled.csv
```

Slurm example:

```bash
sbatch --export=ALL,SIF_PATH=/path/to/log-monitor-train.sif,DATA_PATH=/path/to/labeled.csv hpc/slurm_train_apptainer.sbatch
```

## Validation Status

Validated successfully:
- Local CPU training
- Azure Cloud CPU training

Pending validation:
- Local GPU training
- Azure Cloud GPU training

## Notes / Troubleshooting

- HF Hub warning about unauthenticated requests is non-fatal; set `HF_TOKEN` to improve rate limits.
- Azure GPU requires quota and region availability for selected GPU SKU.
- If local CUDA host training fails preflight, switch local runtime to `container` or install CUDA-enabled PyTorch on host.
