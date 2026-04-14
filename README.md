# Log-Monitor

Log-Monitor is a desktop app for:
- Preparing labeled log data with GPT-4o.
- Training a DeBERTa classifier on Azure ML (CPU or GPU) or locally.
- Running local/HPC training through reproducible Docker/Apptainer workflows.

## Repository Layout

- `app.py`: Tkinter UI + orchestration for data prep and training.
- `train.py`: DeBERTa training script.
- `mlops_utils.py`: shared MLOps helpers (hashes, sidecar metadata, MLflow env helpers).
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
- `scikit-learn` is required for splits/CV and classification metrics (installed via requirements files).

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

UI behavior:
- The main window is scrollable, so all controls remain reachable on smaller windows.

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
- Environment-specific visibility:
  - Azure fields appear only when `Azure Cloud` is selected.
  - Local device/runtime fields appear only when `Local Device (CPU/GPU)` is selected.
- `Local Device`: `auto`, `cpu`, `cuda`
- `Local Runtime`: `host` or `container` (policy-driven)
- `Training Config`:
  - hidden by default and shown via `Show Model Parameters`
  - `Mode`: `default`, `tune`, `tune_cv`
  - base params: epochs, batch size, learning rate, weight decay, max length
  - tuning candidates (comma-separated): LRs, batch sizes, epochs, weight decays, max lengths
  - `CV Folds` (used only for `tune_cv`, default `3`)
  - `Max Trials` (used for tuning modes)
- `Interrupt Training`: stops any active training run (local host/container or Azure job).

Local runtime policy:
- If local device is `auto` or `cpu`, runtime is forced to `host`.
- If local device is `cuda`, runtime selector is enabled and defaults to `container` when switching to `cuda`.

### 3) Hosting

- GitHub PAT, repository, branch controls.
- `Host Service` is currently a placeholder button.
- MLflow controls:
  - `MLflow Enabled`
  - `MLflow Backend`: `local`, `azure`, `custom_uri`
  - `Tracking URI` (local default: `<current_working_dir>/mlruns`)
  - `Experiment Name` (no default; required when MLflow is enabled)
  - `Registered Model` (no default; required when MLflow is enabled)
  - `Open MLflow Console`
  - `Register Last Model`

### MLflow Backend Policy

- Default backend is `local` (`./mlruns`) for cost-safe tracking.
- `custom_uri` requires a non-empty tracking URI.
- `azure` resolves the workspace MLflow tracking URI during Azure auth.
- Azure training blocks when MLflow is enabled with `local` backend (remote backend is required for Azure jobs).

## MLOps Tracking

The app tracks both data preparation and training when MLflow is enabled.

### Data Preparation Tracking

- Starts a pipeline parent run and nested data-prep run.
- Logs:
  - prompt hash + preview artifact (first 2000 chars)
  - input/output dataset SHA256
  - dataset metadata (rows/columns/label distribution)
  - sanitized sample artifact (first 100 rows)
- Writes sidecar metadata next to the labeled CSV:

```text
<labeled_csv>.mlmeta.json
```

Sidecar fields:
- `pipeline_id`
- `parent_run_id`
- `data_prep_run_id`
- `prompt_hash`
- `input_dataset_hash`
- `output_dataset_hash`
- `created_at`
- `tracking_uri`
- `experiment_name`

### Training Tracking

`train.py` consumes MLOps env vars and logs:
- params: model, epochs, batch size, lr, resolved device, runtime mode, run source
- metrics:
  - per-epoch `train_loss`
  - selection metrics (validation or CV averages depending on mode)
  - test metrics: `accuracy`, weighted/macro precision, recall, F1, and test loss
  - selected configuration summary score (`weighted_f1`)
- artifacts: `final_model` directory and run metadata JSON
- dataset hash/metadata + sample artifact
- evaluation artifacts:
  - selection summary JSON
  - selection report/confusion artifacts
  - test classification report + confusion matrix JSON

Training writes metadata file for manual model registration:

```text
./outputs/last_training_mlflow.json
```

Required MLOps env vars passed to `train.py`:
- `MLOPS_ENABLED`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `MLFLOW_PIPELINE_ID`
- `MLFLOW_PARENT_RUN_ID`
- `MLFLOW_RUN_SOURCE`
- `MLFLOW_TAGS_JSON`

## Training Behavior

### Modes and Splits

`train.py` now applies a fixed holdout test set and mode-dependent model selection:

- `test_ratio=0.15` (held out, never used for model selection)
- `val_ratio=0.15` (used for holdout validation in `default` and `tune`)

Training mode behavior:
- `default`:
  - single configuration from base parameters
  - train/validation holdout on dev split
- `tune`:
  - random trial subset from candidate parameter grid (`Max Trials`)
  - holdout validation on dev split
- `tune_cv`:
  - random trial subset from candidate parameter grid (`Max Trials`)
  - stratified K-fold CV on dev split using `CV Folds` (default recommended: `3`)

After selection, the best configuration is retrained on the full dev split, then evaluated once on test.

### Local Training

`app.py` launches `train.py` in a background process (host) or via Docker (container).

Interruption behavior:
- Clicking `Interrupt Training` sends a stop signal to local training processes (host/container) and marks the run as interrupted.

- Local + `cpu` + `host`:
  - Runs `python train.py --data <csv> ...training config args...`
  - Sets `CUDA_VISIBLE_DEVICES=-1` to force CPU.
- Local + `auto` + `host`:
  - Runs `python train.py --data <csv> ...training config args...`
  - `train.py` auto-selects CUDA if available, else CPU.
- Local + `cuda` + `host`:
  - Preflight checks `torch.cuda.is_available()` in host Python.
  - Runs host training only if CUDA is available.
- Local + `cuda` + `container`:
  - Runs `scripts/train_docker.sh` with `DEVICE=cuda` and forwards training config args.
  - Passes MLflow env vars into Docker for consistent tracking.

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
- runs `USE_TF=0 python train.py --data ...` with training config args
- injects MLflow env vars into the Azure command when tracking is enabled

After completion (or failure path), the temporary compute cluster is deleted by cleanup logic.

Interruption behavior:
- Clicking `Interrupt Training` requests job cancellation on Azure and the app keeps polling until Azure reports a terminal state (`Canceled`, `Failed`, or `Completed`), then cleanup runs.

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

The Docker and Apptainer scripts pass through the same MLOps env vars used by host training.

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
- If Azure training is selected with MLflow backend `local`, the app blocks start and asks for `azure` or `custom_uri`.
