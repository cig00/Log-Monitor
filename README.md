# Log-Monitor

Log-Monitor is a Tkinter desktop application for the full log-classification lifecycle:

1. ingest raw log CSVs
2. label them with an OpenAI model
3. train a DeBERTa classifier locally or on Azure ML
4. track lineage and metrics with MLflow-style MLOps/LLMOps metadata
5. host the trained model locally or on Azure
6. expose either a local prediction API or an Azure batch inference endpoint

The project also includes Docker, Apptainer, and Slurm helpers for reproducible training outside the desktop UI.

## What The Project Does

At a high level, the application turns unlabeled operational logs into a hosted classifier:

1. `Data Processing`
   - reads a raw CSV
   - finds the log text column automatically
   - batches logs to the OpenAI Chat Completions API using the prompt in `prompt.txt`
   - writes a labeled CSV with `LogMessage` and `class`

2. `Training`
   - trains `microsoft/deberta-v3-xsmall`
   - supports local host training, local container training, and Azure ML training
   - supports `default`, `tune`, and `tune_cv` selection modes
   - saves the final model and evaluation metadata

3. `Tracking`
   - records data-prep lineage and training lineage when MLflow is enabled
   - writes sidecar metadata next to labeled CSV files
   - writes training metadata JSON next to saved models

4. `Hosting`
   - starts a local prediction API, or
   - deploys an Azure ML batch endpoint backed by scale-from-zero compute

5. `Consumption`
   - local clients send:

```json
{
  "errorMessage": ""
}
```

   - the service returns:

```json
{
  "prediction": ""
}
```

   - Azure batch jobs submit files or folders and read predictions from batch job outputs in storage

## Repository Layout

- `app.py`: Tkinter UI and workflow orchestrator.
- `train.py`: model training script.
- `mlops_utils.py`: shared helpers for hashes, sidecars, JSON, model discovery, and prompt metadata.
- `inference_utils.py`: shared inference loader and prediction helper.
- `serve_model.py`: local HTTP prediction service.
- `azure_score.py`: legacy Azure ML online endpoint scoring entrypoint.
- `azure_batch_score.py`: Azure ML batch endpoint scoring entrypoint.
- `prompt.txt`: system prompt used during OpenAI labeling.
- `requirements.txt`: desktop app/runtime dependencies.
- `requirements.train.txt`: training/container dependencies.
- `Dockerfile`: training image definition.
- `azure_inference_conda.yml`: legacy Azure online inference environment definition.
- `azure_batch_inference_conda.yml`: Azure batch inference environment definition.
- `scripts/train_docker.sh`: run training inside Docker.
- `scripts/train_apptainer.sh`: run training inside Apptainer.
- `scripts/docker_build_train_image.sh`: build the Docker training image.
- `scripts/apptainer_build_from_docker.sh`: build a `.sif` from the Docker image.
- `scripts/apptainer_pull_from_registry.sh`: pull a `.sif` from a registry image.
- `hpc/slurm_train_apptainer.sbatch`: Slurm example for Apptainer GPU jobs.

## Labels And Model

The classifier predicts one of four classes:

- `Error`
- `CONFIGURATION`
- `SYSTEM`
- `Noise`

The current model backbone is:

- `microsoft/deberta-v3-xsmall`

The labeling prompt in `prompt.txt` explicitly tells the LLM to classify by root cause, not by superficial keywords like "error" or "exception".

## Requirements

Minimum practical requirements:

- Python 3.10 or newer
- `pip`
- internet access for OpenAI labeling, Hugging Face model download, and Azure workflows

Optional, depending on workflow:

- Azure subscription and tenant access for Azure training/hosting
- Docker for local container training
- Apptainer for HPC or `.sif`-based training
- NVIDIA drivers / CUDA-capable environment for GPU workflows

Platform notes:

- The Tkinter app is desktop-oriented and can be run on Windows or Linux.
- The Docker, Apptainer, and Slurm helper scripts are Bash-based and assume a Linux-style shell environment.
- Azure training and Azure hosting currently create or reuse resources in `eastus`.

## Installation

Create a clean environment and install the app dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the desktop app:

```bash
python app.py
```

Important:

- `requirements.txt` includes `mlflow==2.9.2`. If the local dashboard says MLflow is unavailable, the Python environment running the app is missing that dependency.
- Local container training assumes the Docker image already exists. The UI does not build the image for you.

## Full Lifecycle

### 1. Prepare Raw Logs

Use the `Data Processing` section in the UI:

- `Log File (CSV)`: raw input file
- `OpenAI API Key`: used for labeling
- `OpenAI Model`: any supported chat model you want to use for classification
- `Prepare Data (OpenAI)`: starts the labeling workflow

How data preparation works:

- the app loads `prompt.txt`
- detects a likely log text column by checking names like `log`, `message`, `msg`, or `text`
- falls back to the first CSV column if no obvious log column is found
- sends logs in batches of 10
- requests JSON output from OpenAI
- retries API rate-limit failures with exponential backoff
- writes a labeled CSV even if the run partially fails

Output:

- a labeled CSV chosen by the user via Save As
- a sidecar file next to that CSV:

```text
<labeled_csv>.mlmeta.json
```

### 2. Train The Classifier

Use the `Model Training (DeBERTa)` section:

- `Labeled Data (CSV)`: training input
- `Environment`:
  - `Azure Cloud`
  - `Local Device (CPU/GPU)`
- `Show Model Parameters`: reveals the training config panel
- `Interrupt Training`: requests cancellation for local or Azure runs

Training options:

- `Mode`: `default`, `tune`, `tune_cv`
- `Epochs`
- `Batch Size`
- `Learning Rate`
- `Weight Decay`
- `Max Length`
- `CV Folds`
- `Max Trials`
- tuning lists:
  - `Tune LRs`
  - `Tune Batch Sizes`
  - `Tune Epochs`
  - `Tune Weight Decays`
  - `Tune Max Lengths`

Default training split behavior from `train.py`:

- `test_ratio = 0.15`
- `val_ratio = 0.15`
- `seed = 42`

Mode behavior:

- `default`
  - uses the base hyperparameters directly
  - performs a holdout validation split on the dev subset
- `tune`
  - builds a candidate grid from the tuning fields
  - shuffles with the configured seed
  - evaluates up to `Max Trials`
  - selects by validation weighted F1
- `tune_cv`
  - builds the same candidate grid
  - evaluates up to `Max Trials`
  - uses stratified K-fold CV on the dev subset
  - selects by average weighted F1

After selection:

- the best configuration is retrained on the full dev split
- the final model is evaluated once on the held-out test split

### 3. Review Outputs And Metrics

Training creates:

```text
./outputs/final_model
./outputs/last_training_mlflow.json
./outputs/model_versions/<model_version_id>/final_model
./outputs/data_versions/<data_version_id>/dataset.csv
```

The saved model directory contains standard Hugging Face model files such as:

- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- tokenizer files
- `last_training_mlflow.json`

The metadata JSON contains:

- `run_id`
- `tracking_uri`
- `experiment_name`
- `model_uri`
- `model_version_id`
- `model_version_dir`
- `model_version_model_dir`
- `pipeline_id`
- `parent_run_id`
- `run_source`
- `resolved_device`
- `runtime_mode`
- dataset hashes
- data version fields
- data-prep lineage
- `selection_summary`
- `test_metrics`

Printed training output includes:

- final-train epoch losses
- best selection metrics
- final test metrics such as:
  - `accuracy`
  - `weighted_precision`
  - `weighted_recall`
  - `weighted_f1`
  - `loss`

### 4. Host The Model

Use the `Hosting` section:

- `Generated Model`: select the trained model directory
- `Available Models`: review and select discovered local versioned models, latest local output, downloaded models, or the current manual selection
- `Host Target`:
  - `Local`
  - `Azure`

The selected model path can be:

- the exact `final_model` directory, or
- a parent directory that contains a discoverable model folder

The app resolves the actual model directory by searching for:

- `config.json`
- and one of:
  - `pytorch_model.bin`
  - `model.safetensors`
  - `tf_model.h5`

### 5. Query The Prediction API

Local hosting uses a direct synchronous HTTP request contract.

Request:

```json
{
  "errorMessage": "processed Canceled"
}
```

Response:

```json
{
  "prediction": "Noise"
}
```

Azure hosting is asynchronous batch inference rather than a direct `POST /predict` call.

Typical Azure batch input file example:

```csv
LogMessage
processed Canceled
timeout while opening socket
```

Azure batch output:

- one prediction row is written per processed input row
- results are written to Azure Storage when the batch job finishes

### 6. Inspect Dashboards

Depending on backend and environment:

- local dashboard HTML summarizes hosting, training metadata, and lineage
- local MLflow UI can be opened if `mlflow` is installed
- Azure dashboards open in Azure ML Studio

The desktop UI provides:

- `Open Dashboard`
- `Open MLOps`
- `Open LLMOps`
- `Register Last Model`

## UI Reference

### Data Processing

- `Log File (CSV)`: raw logs
- `OpenAI API Key`: required for labeling
- `OpenAI Model`: OpenAI model used for labeling
- `Prepare Data (OpenAI)`: starts LLM labeling

### Model Training

- `Labeled Data (CSV)`: prepared dataset
- `Environment`
  - `Azure Cloud`
  - `Local Device (CPU/GPU)`
- Azure-only fields:
  - `Azure Sub ID`
  - `Tenant ID`
  - `Azure Compute`
- local-only fields:
  - `Local Device`: `auto`, `cpu`, `cuda`
  - `Local Runtime`: `host`, `container`

Local runtime policy:

- if local device is `cpu` or `auto`, runtime is forced to `host`
- if local device is `cuda`, runtime selection is enabled
- when switching to `cuda`, the UI defaults runtime to `container`

### Hosting

- `GitHub PAT`
- `Repository`
- `Branch`
- `Generated Model`
- `Host Target`
- Azure-host-only fields:
  - `Azure Host Sub ID`
  - `Azure Host Tenant`
  - `Azure Host Compute`
- outputs:
  - `Endpoint URL`
  - `Hosting Status`
  - `Azure MLOps URL`
  - `Azure LLMOps URL`

Important note about GitHub fields:

- the GitHub PAT, repository, and branch controls currently load repo and branch lists from GitHub
- they do not currently drive training or hosting behavior

### MLflow Controls

- `MLflow Enabled`
- `MLflow Backend`
  - `local`
  - `azure`
  - `custom_uri`
- `Tracking URI`
- `Experiment Name`
- `Registered Model`
- `Open Dashboard`
- `Register Last Model`

Backend policy:

- `local` uses the local `mlruns` directory under the project
- `custom_uri` requires a non-empty tracking URI
- `azure` resolves the workspace MLflow URI during Azure auth
- Azure training is blocked if MLflow is enabled with backend `local`

## Input And Output Data Contracts

### Raw Input CSV

The data-prep stage expects a CSV with at least one text-like column containing log messages.

Preferred column names include:

- `LogMessage`
- `log`
- `message`
- `msg`
- `text`

If none of those are present, the first column is used.

### Labeled Training CSV

`train.py` expects:

- `LogMessage`
- `class`

Allowed `class` values:

- `Error`
- `CONFIGURATION`
- `SYSTEM`
- `Noise`

Invalid rows are dropped before training. Training fails if no valid rows remain.

## Local Training

### Host Training

Examples of what the app runs conceptually:

- CPU:

```bash
python train.py --data /path/to/labeled.csv
```

- CPU-forced host mode:

```bash
CUDA_VISIBLE_DEVICES=-1 python train.py --data /path/to/labeled.csv
```

- CUDA host mode:
  - the app first checks `torch.cuda.is_available()`
  - if CUDA is unavailable, the UI blocks the run and suggests container mode

### Container Training

Build the image first:

```bash
./scripts/docker_build_train_image.sh log-monitor-train:latest
```

Optional CUDA wheel index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./scripts/docker_build_train_image.sh log-monitor-train:latest
```

Run training:

```bash
./scripts/train_docker.sh /absolute/path/to/labeled.csv
```

Supported environment variables:

- `DEVICE=auto|cpu|cuda`
- `IMAGE_TAG`
- `OUTPUT_DIR`

The Docker workflow:

- mounts `./outputs` into the container
- mounts the labeled CSV directory read-only as `/data`
- forwards MLOps env vars
- bind-mounts the local MLflow directory when the tracking URI is filesystem-based

## Azure Training

Azure training is fully orchestrated from `app.py`.

Current resource behavior:

- resource group: `LogClassifier-RG`
- workspace: `LogClassifier-Workspace`
- region: `eastus`

Compute mapping:

- `Azure Compute = cpu`
  - cluster name: `cpu-cluster-temp`
  - size: `Standard_D2as_v4`
- `Azure Compute = gpu`
  - cluster name: `gpu-cluster-temp`
  - size: `Standard_NC4as_T4_v3`

Azure training flow:

1. authenticate with `InteractiveBrowserCredential`
2. verify or create the Azure resource group
3. register `Microsoft.MachineLearningServices`
4. verify or create the Azure ML workspace
5. provision a temporary AML compute cluster
6. submit an Azure ML command job
7. install pinned training dependencies inside the Azure job
8. run `train.py`
9. poll until `Completed`, `Failed`, or `Canceled`
10. download artifacts into:

```text
./downloaded_model
```

11. cache training metadata into:

```text
./outputs/last_training_mlflow.json
```

12. delete the temporary training compute cluster during cleanup

Azure job notes:

- Windows dataset paths are normalized to forward-slash form for Azure input URIs.
- The job uses the curated environment `AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest`.
- MLflow env vars are injected into the Azure job when tracking is enabled.

Interruption behavior:

- clicking `Interrupt Training` requests Azure job cancellation
- the app keeps polling until Azure reaches a terminal state
- cleanup still runs afterward

## Local Hosting

Local hosting starts `serve_model.py` in a background process and exposes:

- `GET /`
  - HTML landing page
- `GET /health`
  - returns:

```json
{"status": "ok"}
```

- `POST /predict`
  - accepts the prediction request JSON
- `GET /predict`
  - returns `405 Method Not Allowed` with a usage hint

Behavior:

- the app chooses a free localhost port dynamically
- the hosted API binds to `127.0.0.1`
- the app health-checks `/health` before marking hosting as ready
- the app records which model version and training run were deployed when metadata is available
- hosting metadata is saved to:

```text
./outputs/last_hosting.json
```

After successful local hosting, the app opens a local dashboard page at:

```text
./outputs/local_dashboard.html
```

The local dashboard:

- summarizes hosted API information
- shows training metadata if it can find `last_training_mlflow.json`
- shows the hosted model version when known
- lists discovered available models from local version archives and downloaded bundles
- searches the project `outputs`, the project `downloaded_model`, and the selected hosted model path
- links to the local API and local MLflow UI when available

## Azure Hosting

Azure hosting deploys the selected model to an Azure ML batch endpoint.

Azure hosting flow:

1. authenticate and ensure workspace existence
2. register the model as an Azure ML model asset
3. create an Azure ML batch inference environment from `azure_batch_inference_conda.yml`
4. create or update a scale-from-zero AML compute cluster with `min_instances=0`
5. create a batch endpoint
6. deploy `azure_batch_score.py`
7. set the deployment as the endpoint default
8. return the endpoint scoring URI

Azure scoring notes:

- the endpoint is asynchronous and intended for background scoring
- batch jobs submit files, folders, or Azure ML data assets
- results are written to Azure Storage when each batch job completes
- batch endpoint authentication uses Microsoft Entra ID
- the scoring entrypoint loads the model from `AZUREML_MODEL_DIR`

Azure hosting instance selection:

- CPU tries these sizes in order:
  - `Standard_D2as_v4`
  - `Standard_DS2_v2`
  - `Standard_DS1_v2`
  - `Standard_F2s_v2`
  - `Standard_DS3_v2`
- GPU tries:
  - `Standard_NC4as_T4_v3`
  - `Standard_NC6s_v3`

If Azure hosting fails because of quota:

- the app now shows a reduced quota-focused message
- the error includes the attempted instance types

Important limitation:

- unlike Azure training, Azure hosting does not currently clean up partially created endpoint assets automatically after a failed deployment
- if a deployment fails midway, review Azure ML Studio for leftover endpoint, environment, model, or compute assets

## MLOps And LLMOps

When MLflow is enabled, the app tracks both data preparation and training.

### Data-Prep Tracking

The labeling pipeline logs:

- prompt hash
- prompt preview artifact
- input dataset hash
- output dataset hash
- input/output dataset metadata
- output sample artifact
- aggregate OpenAI token counts:
  - `prompt_tokens`
  - `completion_tokens`
  - `total_tokens`

It also writes a sidecar file next to the labeled CSV:

```text
<labeled_csv>.mlmeta.json
```

That sidecar carries lineage such as:

- `pipeline_id`
- `parent_run_id`
- `data_prep_run_id`
- `prompt_hash`
- `llm_model`
- `input_dataset_hash`
- `output_dataset_hash`
- `data_version_id`
- `data_version_dir`
- `data_version_path`
- tracking URI / experiment name
- data-prep tracking URI / experiment name
- training tracking URI / experiment name

### Training Tracking

Training consumes the following env vars:

- `MLOPS_ENABLED`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `MLFLOW_PIPELINE_ID`
- `MLFLOW_PARENT_RUN_ID`
- `MLFLOW_RUN_SOURCE`
- `MLFLOW_TAGS_JSON`

Training logs:

- base parameters
- selected config parameters
- `model_version_id`
- `data_version_id` when dataset lineage is available
- resolved device and runtime mode
- dataset metadata
- split metadata
- dataset sample artifact
- per-epoch final-train loss
- validation or CV selection metrics
- test metrics
- final model artifacts
- evaluation JSON artifacts
- training metadata JSON

Lineage propagation:

- data-prep lineage is carried into training tags and metadata when available
- labeled datasets are copied into immutable content-addressed folders under `./outputs/data_versions`
- trained models are copied into immutable run-addressed folders under `./outputs/model_versions`
- if the app detects a sidecar pointing to a different MLflow target, it starts a new pipeline parent run instead of reusing stale lineage

### Local Dashboard vs MLflow UI

These are different things:

- `local_dashboard.html`
  - generated by the app
  - shows the latest hosting and training metadata even if the full MLflow UI is unavailable
- local MLflow UI
  - launched with `python -m mlflow ui`
  - usually served at `http://127.0.0.1:5001`
  - requires the `mlflow` package in the Python environment running the app

### Registering A Model Version

`Register Last Model` uses:

- the latest `last_training_mlflow.json`
- the `Registered Model` name from the Hosting section

It registers:

- `model_uri` from metadata when present, or
- `runs:/<run_id>/final_model` as a fallback

## Container And HPC Workflows

### Docker

Build:

```bash
./scripts/docker_build_train_image.sh log-monitor-train:latest
```

Run:

```bash
./scripts/train_docker.sh /absolute/path/to/labeled.csv --train-mode tune --max-trials 8
```

### Apptainer

Build from local Docker image:

```bash
./scripts/apptainer_build_from_docker.sh log-monitor-train:latest log-monitor-train.sif
```

Pull from a registry:

```bash
./scripts/apptainer_pull_from_registry.sh ghcr.io/<org>/log-monitor-train:latest log-monitor-train.sif
```

Run:

```bash
DEVICE=cuda ./scripts/train_apptainer.sh /path/to/log-monitor-train.sif /absolute/path/to/labeled.csv
```

### Slurm

Example:

```bash
sbatch --export=ALL,SIF_PATH=/path/to/log-monitor-train.sif,DATA_PATH=/path/to/labeled.csv hpc/slurm_train_apptainer.sbatch
```

The Docker and Apptainer scripts:

- pass through the same MLOps environment variables used by host training
- bind local MLflow filesystem paths into the container when applicable
- write model artifacts to a host-mounted `outputs` directory

## Troubleshooting

### The Local Dashboard Opens But MLOps Fields Are Empty

This usually means the app found hosting metadata but could not find `last_training_mlflow.json`.

Check:

- the selected hosted model directory
- `./outputs/last_training_mlflow.json`
- `./downloaded_model`
- whether the model bundle actually contains `last_training_mlflow.json`

### The Local Dashboard Says MLflow Is Unavailable

The Python environment running `app.py` does not have `mlflow` installed, even if another environment does.

Fix:

```bash
pip install -r requirements.txt
```

### `ImportError: cannot import name 'MLOPS_ENV_VARS'`

The app now defines that constant locally in `app.py` so startup is resilient to stale copies of `mlops_utils.py`. If you still see this error, you are likely launching a different copy of the project than the one you edited.

### Local API URL Returns `connection refused`

That means the local hosting process exited after startup or never became healthy.

Check:

- the selected model directory really contains a trained model
- the Python environment includes `torch`, `transformers`, and `sentencepiece`
- the hosting error dialog or terminal output for the local process

### Azure Hosting Fails With Quota Errors

That is an Azure subscription capacity issue, not a model issue.

Options:

- request quota for the relevant VM family
- use a different subscription
- change the Azure region in code if you want a region other than `eastus`

### Azure Training Or Hosting Opens The Browser Repeatedly

Azure authentication uses `InteractiveBrowserCredential`, so browser-based login is expected.

### CUDA Host Training Fails Preflight

If local device is `cuda` and host training fails the preflight check:

- switch local runtime to `container`, or
- install a CUDA-enabled PyTorch build on the host

## Known Limitations

- The GitHub repo/branch UI controls currently browse GitHub metadata only; they are not wired into deployment automation.
- Azure training cleanup deletes temporary compute clusters, but Azure hosting does not yet fully clean up partially created endpoint assets on failure.
- Azure region is currently hardcoded to `eastus`.
- Local container training assumes the Docker training image already exists.

## Suggested End-To-End Demo

For a clean demo of the full lifecycle:

1. launch `python app.py`
2. label a raw CSV with `Prepare Data (OpenAI)`
3. enable MLflow and set an experiment name
4. train locally on CPU first
5. inspect `./outputs/final_model` and `./outputs/last_training_mlflow.json`
6. host the saved model locally
7. open the local dashboard
8. send a `POST /predict` request
9. if needed, repeat with Azure training or Azure hosting once Azure quota is available, then submit a batch scoring job
