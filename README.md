# Log-Monitor

Log-Monitor is a Tkinter desktop application for the full log-classification lifecycle:

1. ingest raw log CSVs
2. label them with an OpenAI model
3. train a DeBERTa classifier locally or on Azure ML
4. track lineage and metrics with MLflow-style MLOps/LLMOps metadata (Azure training always logs metrics to Azure ML)
5. host the trained model locally or on Azure
6. expose either a local prediction API or an Azure batch inference endpoint

The desktop app now uses a modular-monolith architecture: `app.py` remains the Tkinter UI and workflow orchestrator, while the business logic lives in internal service and runtime modules under `app_core/`.

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
   - records data-prep lineage when MLflow is enabled
   - always logs Azure training metrics to the Azure workspace MLflow backend
   - writes sidecar metadata next to labeled CSV files
   - writes training metadata JSON next to saved models

4. `Hosting`
   - starts a local prediction API, or
   - deploys an Azure ML batch endpoint backed by scale-from-zero compute
   - enforces a deployment gate against a golden labeled dataset before any local or Azure deploy is allowed

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
- `app_core/`: internal modular-monolith package used by `app.py`.
  - `contracts.py`: typed request, status, artifact, and job dataclasses.
  - `runtime.py`: shared `JobManager`, `StateStore` (SQLite), and `ArtifactStore`.
  - `github_service.py`: GitHub repo and branch discovery service.
  - `data_prep_service.py`: OpenAI-backed data-prep workflow service.
  - `training_service.py`: local and Azure training orchestration service.
  - `model_catalog_service.py`: model inventory and metadata lookup service.
  - `hosting_service.py`: local and Azure hosting orchestration service.
  - `observability_service.py`: Prometheus/Grafana install, launch, and provisioning service.
  - `mlops_service.py`: MLflow config, lineage, UI, and registration service.
  - `azure_platform_service.py`: Azure SDK adapter used by training and hosting.
- `train.py`: model training script.
- `mlops_utils.py`: shared helpers for hashes, sidecars, JSON, model discovery, and prompt metadata.
- `inference_utils.py`: shared inference loader and prediction helper.
- `serve_model.py`: local HTTP prediction service.
- `gates/deployment_policy.json`: default deployment-gate threshold policy.
- `gates/drift_policy.json`: default drift-monitoring threshold policy.
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
- `tests/`: unit tests for the new runtime and service layer.

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

CSV parsing behavior:

- `train.py` first tries strict `pandas.read_csv` parsing.
- if strict parsing fails, it retries with malformed-row skipping and logs the skipped-row count.
- accepted training-column aliases are:
  - label: `class`, `label`
  - message: `LogMessage`, `log_message`, `message`, `msg`, `text`, `log`
- class labels are normalized case-insensitively before mapping to model IDs.

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
- Azure data asset fields, when training runs on Azure
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
- `Gate Golden Set`: labeled CSV used as the deployment acceptance gate (`LogMessage` + `class`, with common aliases accepted)
- `Gate Policy`: JSON thresholds (`gates/deployment_policy.json` by default)
- `Drift Golden Set`: labeled CSV used for drift monitoring baseline checks
- `Drift Policy`: JSON warning/critical drift thresholds (`gates/drift_policy.json` by default)
- `Available Models`: review and select discovered local versioned models, latest local output, downloaded models, or the current manual selection
- `Host Target`:
  - `Local`
  - `Azure`
- `Azure Service`:
  - `Queued batch API (daily)` for the app's log API + daily Azure ML batch scoring pipeline
  - `Real-time endpoint` for a managed online endpoint backed by Azure compute
  - `Serverless endpoint` for a model-catalog or Foundry model ID shown in Azure ML Studio

The selected model path can be:

- the exact `final_model` directory, or
- a parent directory that contains a discoverable model folder

Serverless endpoint hosting still requires selecting a local generated model directory because deployment gate evaluation runs locally before deployment. The model folder is used only for gate evaluation and is not uploaded for serverless deployment. The app pre-fills `Serverless Model ID` with `azureml://registries/azureml/models/Phi-4-mini-instruct` and generates a unique endpoint name automatically. You can edit either field; if the model ID includes `/versions/...`, the app strips that suffix because Azure serverless deployments use the latest catalog version.

The app resolves the actual model directory by searching for:

- `config.json`
- and one of:
  - `pytorch_model.bin`
  - `model.safetensors`
  - `tf_model.h5`

Before deployment starts, the app evaluates the selected model against the golden set and blocks deployment if thresholds are not met.

Deployment gate outputs are written to:

- `outputs/gates/gate_eval_<timestamp>.json`
- `outputs/gates/gate_eval_<timestamp>_predictions.csv`
- `outputs/gates/latest_gate_eval.json`

Gate PASS evaluations are cached by `model_hash + golden_set_hash + policy_hash` and reused automatically.

After deployment succeeds, observability runs a drift-monitoring baseline against the configured drift golden set and stores results in:

- `outputs/drift_monitoring/<deployment>/drift_eval_<timestamp>.json`
- `outputs/drift_monitoring/<deployment>/drift_eval_<timestamp>_predictions.csv`
- `outputs/drift_monitoring/<deployment>/latest_drift_eval.json`

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

Azure batch and queued-batch hosting are asynchronous batch inference flows rather than direct `POST /predict` calls. Azure real-time hosting returns a managed online scoring URI, and the Serverless endpoint option creates an Azure ML Serverless endpoint from a catalog model ID and returns that endpoint target URI.

Typical Azure batch input file example:

```csv
LogMessage
processed Canceled
timeout while opening socket
```

Azure batch output:

- one prediction row is written per processed input row
- results are written to Azure Storage when the batch job finishes

### 6. Correct Labels And Retrain

Every Azure hosting mode also deploys a companion Azure Function feedback API. This is separate from the inference endpoint so it works consistently for:

- Azure ML serverless catalog endpoints
- Azure ML managed online endpoints
- Azure ML batch endpoints
- queued-batch Function API deployments

Feedback request:

```json
{
  "errorMessage": "Error: application startup failed because MAIL_HOST environment variable is missing",
  "correctLabel": "CONFIGURATION",
  "predictedLabel": "Error",
  "source": "production"
}
```

Accepted label values are:

- `Error`
- `CONFIGURATION`
- `SYSTEM`
- `Noise`

When feedback is accepted, the Function:

1. stores the feedback event in Azure Blob Storage
2. updates or appends the row in a corrected labeled CSV
3. writes the corrected CSV as an immutable blob
4. registers the corrected CSV as a new Azure ML Data asset version named `log-monitor-feedback-labeled-data`
5. submits an Azure ML retraining command job against that new data asset version

The hosting summary includes:

- `Feedback API`: `POST` corrected labels here
- `Feedback status`: stored in hosting metadata as `feedback_status_url`

For model-based Azure hosting, the app uploads the current local labeled data version as the feedback base dataset when it can find it from training metadata. If no base dataset is available, the feedback pipeline still creates data versions from submitted corrections, but retraining may need enough feedback rows to satisfy the training split and label distribution requirements.

### 7. Azure Real-Time Triage Automation

Azure real-time endpoint hosting can also expose a Function-backed triage endpoint in front of the prediction endpoint. The app fetches the Azure ML endpoint key automatically after deployment. The desktop form asks for:

- GitHub PAT, repository, and branch
- Configuration email and System email
- Azure Communication Services connection string and Email sender address. Use the single `Load Azure Resources` button in hosting to authenticate once, load Azure ML models, and fill the ACS connection/sender dropdowns. If no ACS resources exist, the app creates a default Communication Services resource, Email service, Azure-managed domain, and sender so the workflow is not blocked.
- Jira site URL, account email, API token, project key, issue type, optional priority, and labels
- Jira priority must be a Jira priority name such as `Highest`, `High`, `Medium`, or `Low`; if Jira rejects the optional priority, the Function retries issue creation without that field
- Jira site URL can be either the base Atlassian URL or a copied project/board URL; the Function normalizes it to the site root before calling Jira
- If Jira rejects the configured issue type, the Function retries the issue as `Task`

Triage behavior:

- `Noise`: no action
- `CONFIGURATION`: sends email to the configured configuration recipient
- `SYSTEM`: sends email to the configured system recipient
- `Error`: creates a Jira issue with the log payload, prediction response, endpoint metadata, and GitHub commit context
- if Jira issue creation fails for an `Error` prediction, the triage endpoint returns `action_status: partial_failure`, `jira_created: false`, and the Jira error/config summary in `action_errors`
- every triage prediction updates a daily Jira `Log Monitor Prediction Summary - YYYY-MM-DD` issue with counts for `Error`, `CONFIGURATION`, `SYSTEM`, and `Noise`, plus Jira incident success/failure counts
- every triage response includes sanitized `diagnostics` showing the payload, prediction, selected action, Jira creation, monitoring update, and response stages without returning tokens or connection strings
- Azure real-time scoring also receives a fire-and-forget `triage/action` Function URL. When callers hit the Azure ML scoring URI directly, `azure_score.py` returns the prediction immediately and starts a background POST to `triage/action` so Jira/email/monitoring can run without blocking the prediction response. The action route uses the already computed prediction and does not call the model again.

GitHub history is included only as a non-conclusive investigation signal. Jira issues show a developer-impact verdict such as `possible_developer_impact`, `message_only_developer_candidate`, `no_direct_developer_evidence`, or `no_developer_candidate_found`, then list candidate commits for manual review. The lookup checks exact commit metadata, commit ranges, source paths, commit-message search, and then hydrates candidates with the GitHub commit diff so the changed files and patch text are scored against the log evidence. If no direct diff candidate is found, the report states that no developer may be responsible while still keeping the raw GitHub evidence visible.

GitHub PAT, repository, and branch are required for Azure hosting so the Function app has enough repository context for incident triage.

If the Azure Function App exists but the Portal shows no functions, the bridge package did not finish indexing. Hosting now waits for the named Function trigger to appear before returning URLs; if indexing fails, the error points to Function App Log stream and the Flex Consumption Deployment diagnostic blade.

### 8. Inspect Dashboards

Depending on backend and environment:

- local dashboard HTML summarizes hosting, training metadata, and lineage
- local MLflow UI can be opened if `mlflow` is installed
- Azure dashboards open in Azure ML Studio

The desktop UI provides:

- `Open MLOps`
- `Open LLMOps`

## UI Reference

### Data Processing

- `Log File (CSV)`: raw logs
- `OpenAI API Key`: required for labeling
- `OpenAI Model`: OpenAI model used for labeling
- `Prompt Lab`: editable labeling prompt, initialized from `prompt.txt`
- `Version`: choose `default` or an archived prompt version as the editable starting point
- `Run Prompt Tests`: runs the current prompt against built-in and custom test cases and shows `Expected`, `Got`, and pass/fail results
- `Compare Prompt Versions`: opens a diff between the two latest archived prompt versions
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
- `Create PR`: after hosting succeeds, create a GitHub Copilot coding-agent task in the selected repo/branch to add async log forwarding to the created endpoint
- `Generated Model`
- `Gate Golden Set`
- `Gate Policy`
- `Drift Golden Set`
- `Drift Policy`
- `Host Target`
- Azure-host-only fields:
  - `Azure Service`
  - `Azure Host Sub ID`
  - `Azure Host Tenant`
  - `Azure Host Compute`
  - `Serverless Model ID`
  - `Endpoint Name`
- outputs:
  - `Endpoint URL`
  - `Feedback API`, with a copy button for the POST endpoint created by Azure hosting
  - `Feedback Status`
  - `Hosting Status`
  - `GitHub PR Task`
  - `Azure MLOps URL`
  - `Azure LLMOps URL`

Important note about GitHub PR automation:

- the app creates a GitHub issue assigned to `copilot-swe-agent[bot]` using GitHub's Copilot coding-agent API
- Copilot opens or updates the implementation PR in the background when the repository/account supports Copilot coding agent
- the Copilot task prompt focuses on non-blocking log forwarding so the target app does not wait for endpoint calls in the user-facing path
- the Copilot task prompt includes the deployed endpoint URL and the Azure ML Studio endpoint page for traceability
- when Azure hosting creates a Function bridge, the Copilot task uses the triage/log Function URL first so the committed default URL already contains the Function access code
- the prompt asks Copilot to commit the deployed endpoint URL as the default target in the app code/config, without requiring server environment variable setup by the user
- the generated Copilot task prompt is versioned under `./outputs/copilot_pr_prompts` and, when Azure MLflow tracking is available, logged to the `log-monitor-copilot-prompts` experiment in Azure ML Studio

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
5. register the labeled CSV as an Azure ML Data asset named `log-monitor-labeled-data`
6. provision a temporary AML compute cluster
7. submit an Azure ML command job that reads from the registered data asset version
8. install pinned training dependencies inside the Azure job
9. run `train.py`
10. poll until `Completed`, `Failed`, or `Canceled`
11. download artifacts into:

```text
./downloaded_model
```

12. cache training metadata into:

```text
./outputs/last_training_mlflow.json
```

13. delete the temporary training compute cluster during cleanup

Azure job notes:

- Azure data asset versions are derived from the local dataset SHA-256 hash and truncated to Azure's 30-character version limit, so identical labeled CSV content reuses the same version.
- The registered data asset URI is written into the CSV sidecar and training metadata as `azure_data_asset_uri`.
- The job uses the curated environment `AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest`.
- MLflow env vars are always injected into Azure training jobs so metrics are logged to Azure ML even when MLflow is disabled in the UI.

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
- `GET /metrics`
  - returns Prometheus-compatible metrics for the local API
- `GET /predict`
  - returns `405 Method Not Allowed` with a usage hint

Behavior:

- the app chooses a free localhost port dynamically
- the hosted API binds to `127.0.0.1`
- the app health-checks `/health` before marking hosting as ready
- the app starts Prometheus and Grafana locally to visualize API health and prediction traffic
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

Internal runtime note:

- background workflow metadata is also stored in:

```text
./outputs/runtime_state.sqlite3
```

- the SQLite runtime store backs job status and lightweight orchestration state for the modular service layer used by `app.py`

Local observability dependency handling:

- Windows:
  - the app can download portable Grafana and Prometheus builds into the project automatically
- macOS:
  - the app can install Grafana and Prometheus automatically with Homebrew
  - manual fallback paths include common Homebrew locations under `/opt/homebrew` and `/usr/local`
- Debian/Ubuntu Linux:
  - the app can install Grafana and Prometheus automatically with `pkexec` + `apt-get`

## Azure Hosting

Azure hosting supports three service choices:

- `Queued batch API (daily)` deploys the selected model behind an Azure Function log API, queue, storage container, and daily Azure ML batch run.
- `Real-time endpoint` deploys the selected model to a managed online endpoint backed by Azure compute.
- `Serverless endpoint` creates an Azure ML Serverless endpoint from a model catalog or Foundry model ID. This path does not upload the local generated model directory.

All Azure hosting choices also deploy the same Azure Function feedback bridge. The bridge exposes `POST /api/feedback`, creates corrected Azure ML Data asset versions, and starts retraining jobs from those versions. In queued-batch mode, the same Function App also exposes `POST /api/logs` for production log ingestion.

Azure hosting flow:

1. authenticate and ensure workspace existence
2. register the model as an Azure ML model asset
3. create an Azure ML batch inference environment from `azure_batch_inference_conda.yml`
4. create or update a scale-from-zero AML compute cluster with `min_instances=0`
5. create a batch endpoint
6. deploy `azure_batch_score.py`
7. set the deployment as the endpoint default
8. deploy the feedback bridge Function App
9. return the endpoint scoring URI and feedback API URL

Serverless endpoint flow:

1. authenticate and ensure workspace existence
2. default the model catalog ID when the field is empty
3. generate an Azure-valid endpoint name from the model name and a timestamp
4. normalize the model catalog ID by removing any trailing `/versions/...`
5. create the endpoint through the documented ARM `Microsoft.MachineLearningServices/workspaces/serverlessEndpoints` resource using API version `2024-04-01-preview`, which is the API version used in the Foundry serverless deployment docs
6. fall back to `ml_client.serverless_endpoints` only if ARM creation fails
7. verify the endpoint appears in both the ARM resource list and the workspace serverless endpoint list returned by the Azure SDK
8. deploy the feedback bridge Function App
9. return the serverless target URI, feedback API URL, the Azure ML Studio endpoints URL, and a direct Azure Portal hidden-resource URL

Azure scoring notes:

- the endpoint is asynchronous and intended for background scoring
- batch jobs submit files, folders, or Azure ML data assets
- results are written to Azure Storage when each batch job completes
- batch endpoint authentication uses Microsoft Entra ID
- the scoring entrypoint loads the model from `AZUREML_MODEL_DIR`
- serverless endpoints use endpoint keys from Azure ML Studio and are billed as serverless/standard deployments
- if Studio does not immediately show a newly created serverless endpoint, check the summary for the endpoint name, target URI, SDK list verification, and direct Portal resource link

Azure hosting instance selection:

- CPU tries these sizes in order:
  - `Standard_D2as_v4`
  - `Standard_DS2_v2`
  - `Standard_DS1_v2`
  - `Standard_F2s_v2`
  - `Standard_E2s_v3`
  - `Standard_DS3_v2`
  - `Standard_E4s_v3`
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

Data-prep tracking still depends on MLflow being enabled in the UI, but Azure training metrics are always logged to Azure MLflow.

### Copilot PR Prompt Tracking

When Azure hosting creates a GitHub Copilot PR task, Log Monitor archives the exact prompt under `./outputs/copilot_pr_prompts`. If the Azure workspace exposes an MLflow tracking URI, the same prompt is also logged to Azure ML Studio in the `log-monitor-copilot-prompts` experiment with the prompt version ID, endpoint URL, Azure ML Studio endpoint page, target repo, branch, and GitHub issue URL.

### Data-Prep Tracking

The labeling pipeline logs:

- prompt hash
- immutable prompt version ID
- full prompt artifact and prompt preview artifact
- prompt metadata, including prompt length and previous prompt version
- prompt comparison diff when a previous prompt version exists
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
- `prompt_version_id`
- `prompt_version_label`
- `prompt_version_dir`
- `prompt_version_path`
- `prompt_metadata_path`
- `prompt_comparison_path`
- `previous_prompt_version_id`
- `llm_model`
- `input_dataset_hash`
- `output_dataset_hash`
- `data_version_id`
- `data_version_dir`
- `data_version_path`
- tracking URI / experiment name
- data-prep tracking URI / experiment name
- training tracking URI / experiment name

The Prompt Lab in the desktop UI starts with the contents of `prompt.txt`, but users can edit the prompt before running data preparation. The exact UI prompt used for labeling is what gets archived and logged.

The Prompt Lab version dropdown shows `default` for the current `prompt.txt` plus archived prompt versions. Selecting an archived version loads it into the editor as a starting point; edits do not mutate that version. Running data preparation archives the edited text as a new immutable prompt version.

Prompt versions are archived locally under:

```text
./outputs/prompt_versions/<prompt_sha256>/
```

Each prompt version directory contains:

- `prompt.txt`
- `metadata.json`
- `comparison_from_previous.diff` when there is a previous version

The desktop UI also provides `Compare Prompt Versions`, which opens a unified diff between the two latest archived prompt versions.

Prompt unit tests can be run from the UI before labeling. Built-in and custom test cases compare expected classes against LLM-returned classes and color each row green or red.

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
- `prompt_version_id` when prompt lineage is available
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

### Local Dashboard

The local dashboard is generated by the app as `local_dashboard.html` and summarizes the latest hosting and training metadata.

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

### Service-Layer Changes Need Validation

The UI is now a thin orchestrator over `app_core/`, so service/runtime regressions can be checked without launching Tkinter:

```bash
python -m unittest discover -s tests -v
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

- Create PR requires GitHub Issues and Copilot coding agent to be enabled for the selected repository/account, and the PAT must have enough repository, issue, and pull request permissions.
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
