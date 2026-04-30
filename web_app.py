from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app_core.azure_platform_service import AZURE_AVAILABLE, AzurePlatformService
from app_core.contracts import DataPrepRequest, HostingRequest, MlflowConfig, TrainingRequest
from app_core.data_prep_service import DataPrepService
from app_core.github_service import GitHubService
from app_core.hosting_service import HostingService
from app_core.mlops_service import MlopsService
from app_core.model_catalog_service import ModelCatalogService
from app_core.observability_service import ObservabilityService
from app_core.runtime import ArtifactStore, JobManager, StateStore
from app_core.training_service import TrainingService
from mlops_utils import clean_optional_string, local_mlflow_tracking_uri


PROJECT_DIR = Path(os.environ.get("LOG_MONITOR_PROJECT_DIR", Path(__file__).resolve().parent)).resolve()
UPLOAD_DIR = Path(os.environ.get("LOG_MONITOR_UPLOAD_DIR", PROJECT_DIR / "uploads")).resolve()
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "LogClassifier-RG")
WORKSPACE_NAME = os.environ.get("AZURE_WORKSPACE_NAME", "LogClassifier-Workspace")

artifact_store = ArtifactStore(str(PROJECT_DIR))
state_store = StateStore(artifact_store.state_db_path)
job_manager = JobManager(state_store)
github_service = GitHubService()
model_catalog_service = ModelCatalogService(str(PROJECT_DIR), artifact_store)
mlops_service = MlopsService(
    str(PROJECT_DIR),
    artifact_store,
    model_catalog_service,
    resource_group=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
    local_tracking_uri=clean_optional_string(os.environ.get("MLFLOW_TRACKING_URI")) or local_mlflow_tracking_uri(str(PROJECT_DIR)),
)
azure_platform_service = AzurePlatformService(str(PROJECT_DIR), resource_group=RESOURCE_GROUP, workspace_name=WORKSPACE_NAME)
observability_service = ObservabilityService(str(PROJECT_DIR), artifact_store, state_store)
data_prep_service = DataPrepService(job_manager, mlops_service, model_catalog_service)
training_service = TrainingService(str(PROJECT_DIR), job_manager, model_catalog_service, mlops_service, azure_platform_service)
hosting_service = HostingService(
    str(PROJECT_DIR),
    job_manager,
    model_catalog_service,
    mlops_service,
    azure_platform_service,
    observability_service,
    github_service,
)

app = FastAPI(title="Log Monitor", version="1.0")

OPENAI_LABEL_MODELS = (
    "gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5.2",
)

DEFAULT_PROMPT_TEST_CASES = [
    {
        "name": "Missing Environment Variable",
        "message": "Error: application startup failed because MAIL_HOST environment variable is missing",
        "expected": "CONFIGURATION",
    },
    {
        "name": "Disk Space",
        "message": "kernel: write failed on /var/log/app.log, no space left on device",
        "expected": "SYSTEM",
    },
    {
        "name": "Code Exception",
        "message": "Traceback: TypeError unsupported operand type for +: 'NoneType' and 'str'",
        "expected": "Error",
    },
    {
        "name": "Canceled Work",
        "message": "processed Canceled after user interrupted the background task",
        "expected": "Noise",
    },
]


class MlflowPayload(BaseModel):
    enabled: bool = True
    backend: str = "custom_uri"
    tracking_uri: str = Field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    experiment_name: str = "log-monitor-docker"
    registered_model_name: str = "log-monitor-deberta"

    def to_config(self) -> MlflowConfig:
        return MlflowConfig(
            enabled=self.enabled,
            backend=self.backend,
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
            registered_model_name=self.registered_model_name,
        )


class TrainingPayload(BaseModel):
    csv_path: str
    environment_mode: str = "local"
    local_device: str = "cpu"
    local_runtime: str = "host"
    azure_sub_id: str = Field(default_factory=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID", ""))
    azure_tenant_id: str = Field(default_factory=lambda: os.environ.get("AZURE_TENANT_ID", ""))
    azure_compute: str = "cpu"
    azure_instance_type: str = ""
    train_mode: str = "default"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = 128
    cv_folds: int = 3
    max_trials: int = 1
    tune_learning_rates: list[float] = Field(default_factory=list)
    tune_batch_sizes: list[int] = Field(default_factory=list)
    tune_epochs: list[int] = Field(default_factory=list)
    tune_weight_decays: list[float] = Field(default_factory=list)
    tune_max_lengths: list[int] = Field(default_factory=list)
    mlflow: MlflowPayload = Field(default_factory=MlflowPayload)


class DataPrepPayload(BaseModel):
    input_path: str
    output_path: str = "/workspace/outputs/prepared_labeled.csv"
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    prompt_text: str = ""
    prompt_source: str = "web"
    mlflow: MlflowPayload = Field(default_factory=MlflowPayload)


class HostingPayload(BaseModel):
    model_dir: str = "/workspace/outputs/final_model"
    mode: str = "local"
    deployment_gate_golden_path: str = "/workspace/gates/deployment_golden.csv"
    deployment_gate_policy_path: str = "/workspace/gates/deployment_policy.json"
    drift_golden_path: str = "/workspace/gates/drift_golden.csv"
    drift_policy_path: str = "/workspace/gates/drift_policy.json"
    azure_sub_id: str = Field(default_factory=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID", ""))
    azure_tenant_id: str = Field(default_factory=lambda: os.environ.get("AZURE_TENANT_ID", ""))
    azure_compute: str = "cpu"
    azure_instance_type: str = ""
    azure_service: str = "queued_batch"
    azure_serverless_model_id: str = ""
    azure_serverless_endpoint_name: str = ""
    batch_input_uri: str = ""
    batch_hour: int = 0
    batch_minute: int = 0
    batch_timezone: str = "UTC"
    create_github_pr: bool = False
    github_token: str = ""
    github_repo: str = ""
    github_branch: str = ""


class GitHubReposPayload(BaseModel):
    token: str = Field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))


class GitHubBranchesPayload(BaseModel):
    token: str = Field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))
    repo_name: str


class PromptTestCasePayload(BaseModel):
    name: str = ""
    message: str
    expected: str = "Error"


class PromptTestsPayload(BaseModel):
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    prompt_text: str
    cases: list[PromptTestCasePayload] = Field(default_factory=list)


def _job_to_dict(job) -> dict[str, Any]:
    return asdict(job) if job is not None else {}


def _resolve_path(raw_path: str) -> Path:
    value = clean_optional_string(raw_path)
    if not value:
        raise HTTPException(status_code=400, detail="Path is empty.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def _training_options(payload: TrainingPayload) -> dict[str, Any]:
    return {
        "train_mode": payload.train_mode,
        "epochs": payload.epochs,
        "batch_size": payload.batch_size,
        "learning_rate": payload.learning_rate,
        "weight_decay": payload.weight_decay,
        "max_length": payload.max_length,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "cv_folds": payload.cv_folds,
        "max_trials": payload.max_trials,
        "tune_learning_rates": payload.tune_learning_rates or [payload.learning_rate],
        "tune_batch_sizes": payload.tune_batch_sizes or [payload.batch_size],
        "tune_epochs": payload.tune_epochs or [payload.epochs],
        "tune_weight_decays": payload.tune_weight_decays or [payload.weight_decay],
        "tune_max_lengths": payload.tune_max_lengths or [payload.max_length],
    }


def _prompt_version_choices() -> list[dict[str, str]]:
    choices = [{"label": "default", "version_id": "", "source": "default"}]
    for version in reversed(mlops_service.list_prompt_versions()):
        version_id = clean_optional_string(version.get("prompt_version_id"))
        label = clean_optional_string(version.get("prompt_version_label")) or version_id[:12]
        created_at = clean_optional_string(version.get("created_at")).replace("T", " ")[:19]
        source = clean_optional_string(version.get("prompt_source"))
        display_parts = [label]
        if created_at:
            display_parts.append(created_at)
        if source:
            display_parts.append(source)
        choices.append(
            {
                "label": " | ".join(display_parts),
                "version_id": version_id,
                "source": source or "archived",
            }
        )
    return choices


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "project_dir": str(PROJECT_DIR), "docker_runtime": observability_service.is_docker_runtime()}


@app.get("/api/config")
def config() -> dict[str, Any]:
    urls = observability_service.get_docker_observability_urls() if observability_service.is_docker_runtime() else {}
    return {
        "project_dir": str(PROJECT_DIR),
        "upload_dir": str(UPLOAD_DIR),
        "docker_runtime": observability_service.is_docker_runtime(),
        "urls": urls,
        "port_commands": {
            "web": "docker compose port web 8080",
            "inference": "docker compose port inference 8000",
            "mlflow": "docker compose port mlflow 5000",
            "prometheus": "docker compose port prometheus 9090",
            "grafana": "docker compose port grafana 3000",
        },
        "defaults": {
            "training_csv": str(UPLOAD_DIR),
            "model_dir": "/workspace/outputs/final_model",
            "deployment_gate_golden_path": "/workspace/gates/deployment_golden.csv",
            "deployment_gate_policy_path": "/workspace/gates/deployment_policy.json",
            "drift_golden_path": "/workspace/gates/drift_golden.csv",
            "drift_policy_path": "/workspace/gates/drift_policy.json",
            "openai_models": list(OPENAI_LABEL_MODELS),
            "prompt_test_cases": DEFAULT_PROMPT_TEST_CASES,
            "mlflow": {
                "enabled": True,
                "backend": "custom_uri",
                "tracking_uri": clean_optional_string(os.environ.get("MLFLOW_TRACKING_URI")) or "http://mlflow:5000",
                "experiment_name": "log-monitor-docker",
                "registered_model_name": "log-monitor-deberta",
            },
            "azure": {
                "available": AZURE_AVAILABLE,
                "training_instances": {
                    "cpu": azure_platform_service.get_azure_training_instance_candidates("cpu"),
                    "gpu": azure_platform_service.get_azure_training_instance_candidates("gpu"),
                },
                "hosting_instances": {
                    "cpu": azure_platform_service.get_azure_host_instance_candidates("cpu"),
                    "gpu": azure_platform_service.get_azure_host_instance_candidates("gpu"),
                },
                "batch_timezones": azure_platform_service.get_azure_batch_timezone_options(),
                "default_serverless_model_id": azure_platform_service.get_default_serverless_model_id(),
                "default_serverless_endpoint_name": azure_platform_service.build_default_serverless_endpoint_name(
                    azure_platform_service.get_default_serverless_model_id(),
                    suffix=str(int(time.time())),
                ),
            },
        },
    }


@app.get("/api/jobs")
def list_jobs(limit: int = 50, job_type: str = "") -> dict[str, Any]:
    return {"jobs": [_job_to_dict(job) for job in state_store.list_jobs(limit=limit, job_type=job_type)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_to_dict(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    return {"cancel_requested": job_manager.cancel_job(job_id)}


@app.get("/api/events")
def events() -> dict[str, Any]:
    return {"events": [asdict(event) for event in job_manager.drain_events()]}


@app.post("/api/upload")
def upload(file: UploadFile = File(...), category: str = "uploads") -> dict[str, Any]:
    clean_category = clean_optional_string(category).lower()
    if clean_category == "gates":
        target_dir = PROJECT_DIR / "gates"
    else:
        target_dir = UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "upload.csv").name
    target = target_dir / filename
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"path": str(target), "filename": filename}


@app.post("/api/upload-model-directory")
def upload_model_directory(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No model files were uploaded.")
    root = PROJECT_DIR / "downloaded_model" / f"uploaded_model_{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for item in files:
        raw_name = clean_optional_string(item.filename)
        if not raw_name:
            continue
        safe_parts = [Path(part).name for part in raw_name.replace("\\", "/").split("/") if part not in {"", ".", ".."}]
        if not safe_parts:
            continue
        target = root.joinpath(*safe_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            shutil.copyfileobj(item.file, handle)
        saved_count += 1
    if saved_count < 1:
        raise HTTPException(status_code=400, detail="No model files could be saved.")
    return {"path": str(root), "file_count": saved_count}


@app.get("/api/prompts/current")
def current_prompt() -> dict[str, Any]:
    return {"text": mlops_service.load_prompt(), "source": "default"}


@app.get("/api/prompts/versions")
def prompt_versions() -> dict[str, Any]:
    return {"versions": _prompt_version_choices()}


@app.get("/api/prompts/versions/{version_id}")
def prompt_version(version_id: str) -> dict[str, Any]:
    clean_version = clean_optional_string(version_id)
    if not clean_version or clean_version == "default":
        return current_prompt()
    return {"text": mlops_service.read_prompt_version_text(clean_version), "source": f"derived_from:{clean_version}"}


@app.get("/api/prompts/compare")
def compare_prompts(old_version_id: str = "", new_version_id: str = "") -> dict[str, Any]:
    return mlops_service.compare_prompt_versions(old_version_id=old_version_id, new_version_id=new_version_id)


@app.post("/api/prompts/tests/start")
def start_prompt_tests(payload: PromptTestsPayload) -> dict[str, Any]:
    api_key = clean_optional_string(payload.api_key) or clean_optional_string(os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required for prompt tests.")
    model_name = clean_optional_string(payload.model_name)
    if not model_name:
        raise HTTPException(status_code=400, detail="OpenAI model name is required for prompt tests.")
    prompt_text = clean_optional_string(payload.prompt_text)
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt text is empty.")
    cases = [case.model_dump() if hasattr(case, "model_dump") else case.dict() for case in payload.cases]
    if not cases:
        raise HTTPException(status_code=400, detail="At least one prompt test case is required.")
    job = job_manager.submit(
        "prompt_tests",
        lambda ctx: data_prep_service.evaluate_prompt_test_cases(
            api_key=api_key,
            model_name=model_name,
            prompt_text=prompt_text,
            cases=cases,
        ),
        metadata={"operation": "prompt_tests", "case_count": len(cases)},
    )
    return _job_to_dict(job)


@app.post("/api/github/repos")
def github_repos(payload: GitHubReposPayload) -> dict[str, Any]:
    token = clean_optional_string(payload.token) or clean_optional_string(os.environ.get("GITHUB_TOKEN"))
    if not token:
        raise HTTPException(status_code=400, detail="GitHub PAT is required.")
    return {"repo_names": github_service.fetch_repos(token)}


@app.post("/api/github/branches")
def github_branches(payload: GitHubBranchesPayload) -> dict[str, Any]:
    token = clean_optional_string(payload.token) or clean_optional_string(os.environ.get("GITHUB_TOKEN"))
    repo_name = clean_optional_string(payload.repo_name)
    if not token:
        raise HTTPException(status_code=400, detail="GitHub PAT is required.")
    if not repo_name:
        raise HTTPException(status_code=400, detail="Repository name is required.")
    return {"branch_names": github_service.fetch_branches(token, repo_name)}


@app.get("/api/azure/dashboard-urls")
def azure_dashboard_urls(sub_id: str = "", tenant_id: str = "") -> dict[str, str]:
    mlops_url, llmops_url = azure_platform_service.build_azure_dashboard_urls(sub_id, tenant_id)
    return {"mlops_url": mlops_url, "llmops_url": llmops_url}


@app.get("/api/azure/serverless-defaults")
def azure_serverless_defaults(model_id: str = "") -> dict[str, str]:
    clean_model_id = azure_platform_service.normalize_serverless_model_id(model_id)
    if not clean_model_id:
        clean_model_id = azure_platform_service.get_default_serverless_model_id()
    return {
        "model_id": clean_model_id,
        "endpoint_name": azure_platform_service.build_default_serverless_endpoint_name(
            clean_model_id,
            suffix=str(int(time.time())),
        ),
    }


@app.post("/api/data-prep/start")
def start_data_prep(payload: DataPrepPayload) -> dict[str, Any]:
    input_path = _resolve_path(payload.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Input file not found: {input_path}")
    output_path = _resolve_path(payload.output_path)
    api_key = clean_optional_string(payload.api_key) or clean_optional_string(os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required for data preparation.")
    job = data_prep_service.submit_data_prep(
        DataPrepRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            api_key=api_key,
            model_name=payload.model_name,
            mlflow_config=payload.mlflow.to_config(),
            prompt_text=payload.prompt_text,
            prompt_source=payload.prompt_source,
        )
    )
    return _job_to_dict(job)


@app.post("/api/training/start")
def start_training(payload: TrainingPayload) -> dict[str, Any]:
    csv_path = _resolve_path(payload.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")
    environment_mode = clean_optional_string(payload.environment_mode) or "local"
    azure_sub_id = clean_optional_string(payload.azure_sub_id) or clean_optional_string(os.environ.get("AZURE_SUBSCRIPTION_ID"))
    azure_tenant_id = clean_optional_string(payload.azure_tenant_id) or clean_optional_string(os.environ.get("AZURE_TENANT_ID"))
    job = training_service.submit_training(
        TrainingRequest(
            csv_path=str(csv_path),
            environment_mode=environment_mode,
            local_device=payload.local_device,
            local_runtime=payload.local_runtime,
            azure_sub_id=azure_sub_id,
            azure_tenant_id=azure_tenant_id,
            azure_compute=payload.azure_compute,
            azure_instance_type=payload.azure_instance_type,
            training_options=_training_options(payload),
            mlflow_config=payload.mlflow.to_config(),
        )
    )
    return _job_to_dict(job)


@app.get("/api/models")
def models(selected_path: str = "") -> dict[str, Any]:
    return {"models": [asdict(model) for model in model_catalog_service.discover_available_hosted_models(selected_path)]}


@app.post("/api/hosting/start")
def start_hosting(payload: HostingPayload) -> dict[str, Any]:
    model_dir = _resolve_path(payload.model_dir)
    azure_sub_id = clean_optional_string(payload.azure_sub_id) or clean_optional_string(os.environ.get("AZURE_SUBSCRIPTION_ID"))
    azure_tenant_id = clean_optional_string(payload.azure_tenant_id) or clean_optional_string(os.environ.get("AZURE_TENANT_ID"))
    job = hosting_service.submit_hosting(
        HostingRequest(
            model_dir=str(model_dir),
            mode=payload.mode,
            deployment_gate_golden_path=str(_resolve_path(payload.deployment_gate_golden_path)),
            deployment_gate_policy_path=str(_resolve_path(payload.deployment_gate_policy_path)),
            drift_golden_path=str(_resolve_path(payload.drift_golden_path)),
            drift_policy_path=str(_resolve_path(payload.drift_policy_path)),
            azure_sub_id=azure_sub_id,
            azure_tenant_id=azure_tenant_id,
            azure_compute=payload.azure_compute,
            azure_instance_type=payload.azure_instance_type,
            azure_service=payload.azure_service,
            azure_serverless_model_id=payload.azure_serverless_model_id,
            azure_serverless_endpoint_name=payload.azure_serverless_endpoint_name,
            batch_input_uri=payload.batch_input_uri,
            batch_hour=payload.batch_hour,
            batch_minute=payload.batch_minute,
            batch_timezone=payload.batch_timezone,
            create_github_pr=payload.create_github_pr,
            github_token=payload.github_token,
            github_repo=payload.github_repo,
            github_branch=payload.github_branch,
        )
    )
    return _job_to_dict(job)


@app.post("/api/hosting/stop")
def stop_hosting() -> dict[str, Any]:
    hosting_service.stop_local_stack()
    return {"stopped": True}


@app.get("/api/hosting/current")
def current_hosting() -> dict[str, Any]:
    return {"hosting": model_catalog_service.read_last_hosting_metadata()}


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log Monitor Docker Control Plane</title>
  <style>
    :root { --ink: #18201d; --muted: #66736e; --paper: #f5efe2; --card: #fffaf0; --line: #d9cbb2; --accent: #0f766e; --accent2: #a16207; --bad: #9f1239; --good: #166534; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif; color: var(--ink); background: radial-gradient(circle at top left, #d6eadf 0, transparent 34rem), linear-gradient(135deg, #f8f1df, #efe2c9); }
    header { padding: 34px clamp(18px, 4vw, 56px) 16px; }
    h1 { margin: 0; font-size: clamp(34px, 6vw, 72px); line-height: .92; letter-spacing: -0.06em; max-width: 980px; }
    header p { max-width: 850px; color: var(--muted); font-size: 18px; }
    main { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(360px, .9fr); gap: 18px; padding: 0 clamp(18px, 4vw, 56px) 56px; }
    section, dialog { background: rgba(255, 250, 240, .92); border: 1px solid var(--line); border-radius: 24px; padding: 18px; box-shadow: 0 24px 60px rgba(64, 49, 27, .08); }
    dialog { max-width: min(980px, calc(100vw - 24px)); width: 980px; }
    h2 { margin: 0 0 12px; font-size: 26px; letter-spacing: -0.03em; }
    h3 { margin: 14px 0 6px; font-size: 19px; }
    label { display: block; font-weight: 700; margin-top: 10px; }
    input, select, textarea, button { width: 100%; border-radius: 14px; border: 1px solid var(--line); padding: 10px 12px; font: inherit; background: #fffdf7; color: var(--ink); }
    input[type="checkbox"], input[type="radio"] { width: auto; margin-right: 8px; }
    textarea { min-height: 116px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
    button { cursor: pointer; margin-top: 12px; background: var(--ink); color: #fffaf0; border: none; font-weight: 800; }
    button.secondary { background: var(--accent); }
    button.warn { background: var(--accent2); }
    button.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
    button.inline { width: auto; margin: 0 6px 6px 0; padding: 8px 12px; }
    button:disabled, fieldset:disabled { opacity: .55; cursor: not-allowed; }
    details { border: 1px dashed var(--line); border-radius: 18px; padding: 10px 12px; margin-top: 12px; background: rgba(255,255,255,.45); }
    summary { cursor: pointer; font-weight: 900; }
    fieldset { border: 0; padding: 0; margin: 0; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .stack { display: grid; gap: 18px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
    .toolbar button, .toolbar a { width: auto; margin-top: 0; }
    .radio-row, .check-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 8px 0; }
    .radio-row label, .check-row label { margin: 0; font-weight: 700; display: inline-flex; align-items: center; }
    .links { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 0; }
    .links a { color: var(--accent); font-weight: 800; }
    .hints, .help { color: var(--muted); font-size: 14px; margin-top: 8px; }
    .pill { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; color: var(--muted); background: #fffdf7; }
    .hidden { display: none !important; }
    .readonly { background: #f6efe3; color: var(--muted); }
    pre { white-space: pre-wrap; word-break: break-word; background: #191f1c; color: #eef8ef; border-radius: 18px; padding: 14px; max-height: 380px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    td, th { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
    .status-succeeded, .pass { color: var(--good); font-weight: 800; }
    .status-failed, .fail { color: var(--bad); font-weight: 800; }
    .status-running, .status-queued { color: var(--accent2); font-weight: 800; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .grid, .three { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <span class="pill">Docker Compose Control Plane</span>
    <h1>Log Classifier & DeBERTa Trainer</h1>
    <p>The browser UI mirrors the desktop workflow: data prep, prompt lab, training, deployment gate, hosting, drift baseline, GitHub PR tasking, and observability.</p>
    <div class="links" id="links"></div>
    <div class="hints" id="port_hints"></div>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>Data Processing</h2>
        <label>Log File (CSV)</label>
        <div class="grid">
          <input id="csv_path" value="/workspace/uploads/training.csv">
          <div><input id="upload" type="file" accept=".csv,text/csv"><button class="secondary" onclick="uploadCsv()">Browse / Upload CSV</button></div>
        </div>
        <div class="grid">
          <label>OpenAI API Key<input id="openai_api_key" type="password" placeholder="Uses server OPENAI_API_KEY when left blank"></label>
          <label>OpenAI Model<select id="openai_model"></select></label>
        </div>
        <label>Output Labeled CSV<input id="prep_output_path" value="/workspace/outputs/prepared_labeled.csv"></label>

        <details open>
          <summary>Prompt Lab</summary>
          <label>Version<select id="prompt_version" onchange="loadPromptVersion()"></select></label>
          <label>Prompt Text<textarea id="prompt_text" spellcheck="false"></textarea></label>
          <div class="toolbar">
            <button class="ghost" onclick="reloadPromptText()">Reload Prompt</button>
            <button class="ghost" onclick="refreshPromptVersions()">Refresh Versions</button>
            <button class="secondary" onclick="openPromptTests(true)">Run Prompt Tests</button>
            <button class="ghost" onclick="comparePrompts()">Compare Prompt Versions</button>
          </div>
        </details>
        <button class="secondary" onclick="startDataPrep()">Prepare Data (OpenAI)</button>
      </section>

      <section>
        <h2>Model Training (DeBERTa)</h2>
        <label>Labeled Data (CSV)<input id="training_csv_path" value="/workspace/uploads/training_labeled.csv"></label>
        <div class="toolbar">
          <input id="labeled_upload" type="file" accept=".csv,text/csv">
          <button class="secondary" onclick="uploadLabeledCsv()">Browse / Upload Labeled CSV</button>
          <button class="ghost" onclick="usePreparedForTraining()">Use Prepared Output</button>
        </div>
        <label>Environment</label>
        <div class="radio-row">
          <label><input type="radio" name="training_environment" value="azure" onchange="updateTrainingMode()">Azure Cloud</label>
          <label><input type="radio" name="training_environment" value="local" checked onchange="updateTrainingMode()">Local Device (CPU/GPU)</label>
        </div>
        <fieldset id="training_azure_fields">
          <div class="grid">
            <label>Azure Sub ID<input id="train_azure_sub" placeholder="Uses AZURE_SUBSCRIPTION_ID when blank"></label>
            <label>Tenant ID<input id="train_azure_tenant" placeholder="Uses AZURE_TENANT_ID when blank"></label>
            <label>Azure Compute<select id="train_azure_compute" onchange="updateAzureInstanceSelects()"><option>cpu</option><option>gpu</option></select></label>
            <label>Azure VM Size<select id="train_azure_instance"></select></label>
          </div>
        </fieldset>
        <fieldset id="training_local_fields">
          <div class="grid">
            <label>Local Device<select id="local_device" onchange="updateLocalRuntime()"><option>auto</option><option>cpu</option><option>cuda</option></select></label>
            <label>Local Runtime<select id="local_runtime"><option>host</option><option>container</option></select></label>
          </div>
        </fieldset>
        <details id="training_parameters">
          <summary id="training_parameters_summary">Show Model Parameters</summary>
          <div class="grid">
            <label>Mode<select id="train_mode" onchange="updateTrainingStrategy()"><option>default</option><option>tune</option><option>tune_cv</option></select></label>
            <label>CV Folds<input id="cv_folds" type="number" value="3"></label>
            <label>Epochs<input id="epochs" type="number" value="3"></label>
            <label>Batch Size<input id="batch_size" type="number" value="8"></label>
            <label>Learning Rate<input id="learning_rate" value="5e-5"></label>
            <label>Weight Decay<input id="weight_decay" value="0.01"></label>
            <label>Max Length<input id="max_length" type="number" value="128"></label>
            <label>Max Trials<input id="max_trials" type="number" value="8"></label>
          </div>
          <fieldset id="tuning_fields">
            <label>Tune LRs<input id="tune_lrs" value="5e-5,3e-5,1e-4"></label>
            <label>Tune Batch Sizes<input id="tune_batch_sizes" value="8,16"></label>
            <label>Tune Epochs<input id="tune_epochs" value="3,4"></label>
            <label>Tune Weight Decays<input id="tune_weight_decays" value="0.0,0.01"></label>
            <label>Tune Max Lengths<input id="tune_max_lengths" value="128"></label>
          </fieldset>
        </details>
        <div class="grid">
          <button onclick="startTraining()">Get Model (Train)</button>
          <button class="warn" onclick="interruptTraining()">Interrupt Training</button>
        </div>
      </section>

      <section>
        <h2>Hosting</h2>
        <div class="grid">
          <label>GitHub PAT<input id="github_token" type="password" placeholder="Uses server GITHUB_TOKEN when blank"></label>
          <div class="check-row"><label><input id="create_pr" type="checkbox">Create PR</label></div>
        </div>
        <button class="ghost" onclick="loadRepos()">Load Repos</button>
        <div class="grid">
          <label>Repository<select id="repo_select" onchange="loadBranches()"></select></label>
          <label>Branch<select id="branch_select"></select></label>
        </div>

        <label>Generated Model<input id="hosted_model_path" value="/workspace/outputs/final_model"></label>
        <div class="toolbar">
          <input id="model_dir_upload" type="file" webkitdirectory directory multiple>
          <button class="secondary" onclick="uploadModelDirectory()">Browse / Upload Model Directory</button>
        </div>
        <label>Available Models<select id="model_select" onchange="onModelSelected()"></select></label>
        <button class="ghost" onclick="refreshModels()">Refresh Models</button>

        <label>Host Target</label>
        <div class="radio-row">
          <label><input type="radio" name="hosting_mode" value="local" checked onchange="updateHostingMode()">Local</label>
          <label><input type="radio" name="hosting_mode" value="azure" onchange="updateHostingMode()">Azure</label>
        </div>
        <fieldset id="hosting_azure_fields">
          <label>Azure Service</label>
          <div class="radio-row">
            <label><input type="radio" name="azure_service" value="online" onchange="updateHostingMode()">Real-time endpoint</label>
            <label><input type="radio" name="azure_service" value="queued_batch" checked onchange="updateHostingMode()">Queued batch API (daily)</label>
            <label><input type="radio" name="azure_service" value="serverless" onchange="updateHostingMode()">Serverless endpoint</label>
          </div>
          <fieldset id="serverless_fields">
            <div class="grid">
              <label>Serverless Model ID<input id="serverless_model_id" onblur="refreshServerlessDefaults(false)"></label>
              <label>Endpoint Name<input id="serverless_endpoint" oninput="serverlessEndpointAuto=false"></label>
            </div>
            <p class="help">Serverless hosting uses the Azure ML catalog model ID; the selected local model is only used for gate evaluation.</p>
          </fieldset>
          <div class="grid">
            <label>Azure Host Sub ID<input id="host_azure_sub" placeholder="Uses AZURE_SUBSCRIPTION_ID when blank" oninput="updateAzureUrls()"></label>
            <label>Azure Host Tenant<input id="host_azure_tenant" placeholder="Uses AZURE_TENANT_ID when blank" oninput="updateAzureUrls()"></label>
          </div>
          <fieldset id="azure_host_compute_fields">
            <div class="grid">
              <label>Azure Host Compute<select id="host_azure_compute" onchange="updateAzureInstanceSelects()"><option>cpu</option><option>gpu</option></select></label>
              <label>Azure VM Size<select id="host_azure_instance"></select></label>
            </div>
          </fieldset>
          <fieldset id="batch_fields">
            <div class="grid">
              <label>Batch Input URI<input id="batch_input_uri" placeholder="Required for legacy batch; queued batch can leave blank"></label>
              <label>Daily Time (HH:MM)<input id="batch_time" value="02:00"></label>
              <label>Time Zone<select id="batch_timezone"></select></label>
            </div>
            <p class="help">Queued batch hosting deploys an Azure Function log API, Service Bus queue, Blob Storage, and a daily Azure ML batch launcher.</p>
          </fieldset>
        </fieldset>

        <div class="grid">
          <label>Gate Golden Set<input id="gate_golden" value="/workspace/gates/deployment_golden.csv"></label>
          <div><input id="gate_golden_upload" type="file" accept=".csv,text/csv"><button class="secondary" onclick="uploadGateGolden()">Browse / Upload Gate CSV</button></div>
          <label>Gate Policy<input id="gate_policy" value="/workspace/gates/deployment_policy.json"></label>
          <div><input id="gate_policy_upload" type="file" accept=".json,application/json"><button class="secondary" onclick="uploadGatePolicy()">Browse / Upload Gate Policy</button></div>
          <label>Drift Golden Set<input id="drift_golden" value="/workspace/gates/drift_golden.csv"></label>
          <div><input id="drift_golden_upload" type="file" accept=".csv,text/csv"><button class="secondary" onclick="uploadDriftGolden()">Browse / Upload Drift CSV</button></div>
          <label>Drift Policy<input id="drift_policy" value="/workspace/gates/drift_policy.json"></label>
          <div><input id="drift_policy_upload" type="file" accept=".json,application/json"><button class="secondary" onclick="uploadDriftPolicy()">Browse / Upload Drift Policy</button></div>
        </div>
        <div class="grid">
          <button onclick="startHosting()">Host Service</button>
          <button class="warn" onclick="stopHosting()">Stop Local Stack</button>
        </div>

        <h3>Hosting Outputs</h3>
        <label>Endpoint URL<input id="endpoint_url" class="readonly" readonly></label>
        <div class="toolbar"><button class="ghost" onclick="openField('endpoint_url')">Open Endpoint</button></div>
        <label>Feedback API<input id="feedback_api_url" class="readonly" readonly></label>
        <div class="toolbar"><button class="ghost" onclick="copyField('feedback_api_url')">Copy API</button></div>
        <label>Feedback Status<input id="feedback_status_url" class="readonly" readonly></label>
        <div class="toolbar"><button class="ghost" onclick="openField('feedback_status_url')">Open Status</button></div>
        <label>GitHub PR Task<input id="github_pr_url" class="readonly" readonly></label>
        <div class="toolbar"><button class="ghost" onclick="openField('github_pr_url')">Open PR Task</button></div>
        <label>Azure MLOps URL<input id="azure_mlops_url" class="readonly" readonly></label>
        <div class="toolbar"><button class="ghost" onclick="openField('azure_mlops_url')">Open MLOps</button></div>
        <label>Azure LLMOps URL<input id="azure_llmops_url" class="readonly" readonly></label>
        <div class="toolbar"><button class="ghost" onclick="openField('azure_llmops_url')">Open LLMOps</button></div>
        <label>Hosting Status<textarea id="hosting_summary" class="readonly" readonly></textarea></label>
      </section>
    </div>

    <div class="stack">
      <section>
        <h2>Status</h2>
        <div id="status" class="pill">Ready</div>
      </section>
      <section>
        <h2>Current Hosting</h2>
        <pre id="hosting">{}</pre>
      </section>
      <section>
        <h2>Jobs</h2>
        <table><thead><tr><th>Type</th><th>Status</th><th>Submitted</th><th></th></tr></thead><tbody id="jobs"></tbody></table>
      </section>
      <section>
        <h2>Events</h2>
        <pre id="events"></pre>
      </section>
    </div>
  </main>

  <dialog id="prompt_test_dialog">
    <h2>Prompt Unit Tests</h2>
    <table><thead><tr><th>Test Case</th><th>Log Message</th><th>Expected</th><th>Got</th><th>Result</th></tr></thead><tbody id="prompt_test_rows"></tbody></table>
    <h3>Add Custom Test Case</h3>
    <div class="grid">
      <label>Name<input id="prompt_test_name" value="Custom Case"></label>
      <label>Expected<select id="prompt_test_expected"><option>Error</option><option>CONFIGURATION</option><option>SYSTEM</option><option>Noise</option></select></label>
    </div>
    <label>Log Message<textarea id="prompt_test_message"></textarea></label>
    <div class="toolbar">
      <button class="ghost" onclick="addPromptTestCase()">Add Test Case</button>
      <button class="secondary" onclick="runPromptTests()">Run Tests</button>
      <button class="ghost" onclick="resetPromptTests()">Reset Built-ins</button>
      <button class="warn" onclick="qs('prompt_test_dialog').close()">Close</button>
    </div>
  </dialog>

  <dialog id="prompt_compare_dialog">
    <h2>Prompt Version Comparison</h2>
    <pre id="prompt_compare_text"></pre>
    <button class="warn" onclick="qs('prompt_compare_dialog').close()">Close</button>
  </dialog>

<script>
const qs = id => document.getElementById(id);
const qsa = selector => Array.from(document.querySelectorAll(selector));
let cfg = {};
let eventLog = [];
let promptSource = 'default';
let promptTestCases = [];
let currentPromptTestJobId = '';
let currentTrainingJobId = '';
let currentHostingJobId = '';
let serverlessEndpointAuto = true;
let cachedJobs = [];
let mlflowDefaults = {
  enabled: true,
  backend: 'custom_uri',
  tracking_uri: 'http://mlflow:5000',
  experiment_name: 'log-monitor-docker',
  registered_model_name: 'log-monitor-deberta',
};

function setStatus(message) { qs('status').textContent = message || 'Ready'; }
function reportUiError(context, err) {
  const detail = err && err.message ? err.message : String(err || 'Unknown error');
  const message = `${context}: ${detail}`;
  setStatus(message);
  eventLog.push({timestamp: new Date().toISOString(), status: 'failed', message});
  qs('events').textContent = eventLog.map(e => `[${e.timestamp}] ${e.status || e.stage}: ${e.message}`).join('\n');
  alert(message);
}
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch { payload = {raw: text}; }
  if (!response.ok) throw new Error(payload.detail || payload.message || text || response.statusText);
  return payload;
}
function postJson(path, body) { return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}); }
function radioValue(name) { const item = document.querySelector(`input[name="${name}"]:checked`); return item ? item.value : ''; }
function setRadioValue(name, value) { const item = document.querySelector(`input[name="${name}"][value="${value}"]`); if (item) item.checked = true; }
function setHidden(id, hidden) { qs(id).classList.toggle('hidden', hidden); }
function fillSelect(id, values, preferred = '') {
  const select = qs(id);
  const current = preferred || select.value;
  select.innerHTML = (values || []).map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  if (current && (values || []).includes(current)) select.value = current;
}
function parseList(id, parser) { return qs(id).value.split(',').map(x => x.trim()).filter(Boolean).map(parser).filter(x => Number.isFinite(x)); }
function mlflowPayload() {
  return {...mlflowDefaults};
}
function timeParts() {
  const [hour, minute] = (qs('batch_time').value || '00:00').split(':').map(x => Number(x || 0));
  return {batch_hour: Number.isFinite(hour) ? hour : 0, batch_minute: Number.isFinite(minute) ? minute : 0};
}

async function init() {
  cfg = await api('/api/config');
  const urls = cfg.urls || {};
  const linkHtml = [['Inference', urls.health_url], ['MLflow', urls.mlflow_url], ['Grafana', urls.grafana_url], ['Prometheus', urls.prometheus_url]]
    .filter(x => x[1]).map(x => `<a href="${escapeHtml(x[1])}" target="_blank">${escapeHtml(x[0])}</a>`).join('');
  qs('links').innerHTML = linkHtml;
  if (!linkHtml) qs('port_hints').textContent = 'Docker assigned host ports dynamically. Use `docker compose port web 8080` for this UI and similar `docker compose port ...` commands for service URLs.';

  const defaults = cfg.defaults || {};
  fillSelect('openai_model', defaults.openai_models || ['gpt-4o'], 'gpt-5-mini');
  fillSelect('train_azure_instance', (defaults.azure || {}).training_instances?.cpu || [], 'Standard_D2as_v4');
  fillSelect('host_azure_instance', (defaults.azure || {}).hosting_instances?.cpu || [], 'Standard_DS3_v2');
  fillSelect('batch_timezone', (defaults.azure || {}).batch_timezones || ['UTC'], 'UTC');
  qs('serverless_model_id').value = (defaults.azure || {}).default_serverless_model_id || '';
  qs('serverless_endpoint').value = (defaults.azure || {}).default_serverless_endpoint_name || '';
  promptTestCases = (defaults.prompt_test_cases || []).map(x => ({...x}));
  mlflowDefaults = {...mlflowDefaults, ...(defaults.mlflow || {})};

  qs('training_parameters').addEventListener('toggle', () => qs('training_parameters_summary').textContent = qs('training_parameters').open ? 'Hide Model Parameters' : 'Show Model Parameters');
  await reloadPromptText();
  await refreshPromptVersions();
  await refreshModels();
  updateTrainingMode(); updateTrainingStrategy(); updateHostingMode(); updateLocalRuntime(); renderPromptTests();
  await refreshAll();
  setInterval(refreshAll, 2500);
}

async function uploadFileTo({inputId, targetId, category = 'uploads', label = 'file'}) {
  const file = qs(inputId).files[0];
  if (!file) {
    alert(`Choose a ${label} first.`);
    return null;
  }
  setStatus(`Uploading ${label}...`);
  const form = new FormData();
  form.append('file', file);
  const payload = await api(`/api/upload?category=${encodeURIComponent(category)}`, {method: 'POST', body: form});
  qs(targetId).value = payload.path;
  setStatus(`Uploaded ${payload.filename || label}`);
  return payload;
}
async function uploadCsv() {
  try {
    await uploadFileTo({inputId: 'upload', targetId: 'csv_path', category: 'uploads', label: 'raw CSV'});
  } catch (err) {
    reportUiError('CSV upload failed', err);
  }
}
async function uploadLabeledCsv() {
  try {
    await uploadFileTo({inputId: 'labeled_upload', targetId: 'training_csv_path', category: 'uploads', label: 'labeled CSV'});
  } catch (err) {
    reportUiError('Labeled CSV upload failed', err);
  }
}
async function uploadGateGolden() {
  try {
    await uploadFileTo({inputId: 'gate_golden_upload', targetId: 'gate_golden', category: 'gates', label: 'deployment gate CSV'});
  } catch (err) {
    reportUiError('Deployment gate CSV upload failed', err);
  }
}
async function uploadGatePolicy() {
  try {
    await uploadFileTo({inputId: 'gate_policy_upload', targetId: 'gate_policy', category: 'gates', label: 'deployment gate policy'});
  } catch (err) {
    reportUiError('Deployment gate policy upload failed', err);
  }
}
async function uploadDriftGolden() {
  try {
    await uploadFileTo({inputId: 'drift_golden_upload', targetId: 'drift_golden', category: 'gates', label: 'drift CSV'});
  } catch (err) {
    reportUiError('Drift CSV upload failed', err);
  }
}
async function uploadDriftPolicy() {
  try {
    await uploadFileTo({inputId: 'drift_policy_upload', targetId: 'drift_policy', category: 'gates', label: 'drift policy'});
  } catch (err) {
    reportUiError('Drift policy upload failed', err);
  }
}
async function uploadModelDirectory() {
  try {
    const files = Array.from(qs('model_dir_upload').files || []);
    if (!files.length) return alert('Choose a model directory first.');
    setStatus('Uploading model directory...');
    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.webkitRelativePath || file.name);
    }
    const payload = await api('/api/upload-model-directory', {method: 'POST', body: form});
    qs('hosted_model_path').value = payload.path;
    setStatus(`Uploaded model directory (${payload.file_count || files.length} files).`);
    await refreshModels();
  } catch (err) {
    reportUiError('Model directory upload failed', err);
  }
}
function usePreparedForTraining() { qs('training_csv_path').value = qs('prep_output_path').value; }

async function reloadPromptText() {
  const payload = await api('/api/prompts/current');
  qs('prompt_text').value = payload.text || '';
  promptSource = 'default';
  if (qs('prompt_version')) qs('prompt_version').value = 'default';
}
async function refreshPromptVersions() {
  const payload = await api('/api/prompts/versions');
  const versions = payload.versions || [];
  qs('prompt_version').innerHTML = versions.map(v => `<option value="${escapeHtml(v.version_id || 'default')}" data-source="${escapeHtml(v.source || '')}">${escapeHtml(v.label || 'default')}</option>`).join('');
}
async function loadPromptVersion() {
  const versionId = qs('prompt_version').value;
  if (!versionId || versionId === 'default') return reloadPromptText();
  const payload = await api(`/api/prompts/versions/${encodeURIComponent(versionId)}`);
  qs('prompt_text').value = payload.text || '';
  promptSource = payload.source || `derived_from:${versionId}`;
}
async function comparePrompts() {
  const payload = await api('/api/prompts/compare');
  const oldLabel = payload.old?.prompt_version_label || (payload.old?.prompt_version_id || '').slice(0, 12) || 'not available';
  const newLabel = payload.new?.prompt_version_label || (payload.new?.prompt_version_id || '').slice(0, 12) || 'not available';
  qs('prompt_compare_text').textContent = payload.message || `Old prompt: ${oldLabel}\nNew prompt: ${newLabel}\n\n${payload.diff || 'No textual prompt changes.'}`;
  qs('prompt_compare_dialog').showModal();
}
function openPromptTests(runImmediately = false) { qs('prompt_test_dialog').showModal(); renderPromptTests(); if (runImmediately) runPromptTests(); }
function renderPromptTests(cases = promptTestCases) {
  qs('prompt_test_rows').innerHTML = cases.map(c => `<tr><td>${escapeHtml(c.name || '')}</td><td>${escapeHtml(c.message || '')}</td><td>${escapeHtml(c.expected || '')}</td><td>${escapeHtml(c.got || '')}</td><td class="${c.match === true ? 'pass' : c.match === false ? 'fail' : ''}">${c.match === true ? 'Pass' : c.match === false ? 'Fail' : 'Not run'}</td></tr>`).join('');
}
function addPromptTestCase() {
  const message = qs('prompt_test_message').value.trim();
  if (!message) return alert('Enter a log message for the custom test case.');
  promptTestCases.push({name: qs('prompt_test_name').value.trim() || 'Custom Case', message, expected: qs('prompt_test_expected').value});
  qs('prompt_test_message').value = ''; renderPromptTests();
}
function resetPromptTests() { promptTestCases = ((cfg.defaults || {}).prompt_test_cases || []).map(x => ({...x})); renderPromptTests(); }
async function runPromptTests() {
  const job = await postJson('/api/prompts/tests/start', {api_key: qs('openai_api_key').value, model_name: qs('openai_model').value, prompt_text: qs('prompt_text').value, cases: promptTestCases});
  currentPromptTestJobId = job.job_id; setStatus('Running prompt tests...'); await refreshAll();
}

async function startDataPrep() {
  try {
    const inputPath = qs('csv_path').value.trim();
    const outputPath = qs('prep_output_path').value.trim();
    const modelName = qs('openai_model').value.trim();
    const promptText = qs('prompt_text').value.trim();
    if (!inputPath) return alert('Please select or upload a log CSV first.');
    if (!outputPath) return alert('Please set an output labeled CSV path.');
    if (!modelName) return alert('Please select or enter an OpenAI model.');
    if (!promptText) return alert('Prompt text is empty.');
    setStatus('Submitting data preparation...');
    const job = await postJson('/api/data-prep/start', {
      input_path: inputPath,
      output_path: outputPath,
      api_key: qs('openai_api_key').value,
      model_name: modelName,
      prompt_text: promptText,
      prompt_source: promptSource || 'default',
      mlflow: mlflowPayload(),
    });
    setStatus(`Data preparation started: ${job.job_id || ''}`);
    await refreshAll();
  } catch (err) {
    reportUiError('Data preparation could not start', err);
  }
}
function trainingPayload() {
  return { csv_path: qs('training_csv_path').value, environment_mode: radioValue('training_environment'), local_device: qs('local_device').value, local_runtime: qs('local_runtime').value,
    azure_sub_id: qs('train_azure_sub').value, azure_tenant_id: qs('train_azure_tenant').value, azure_compute: qs('train_azure_compute').value, azure_instance_type: qs('train_azure_instance').value,
    train_mode: qs('train_mode').value, epochs: Number(qs('epochs').value), batch_size: Number(qs('batch_size').value), learning_rate: Number(qs('learning_rate').value),
    weight_decay: Number(qs('weight_decay').value), max_length: Number(qs('max_length').value), cv_folds: Number(qs('cv_folds').value), max_trials: Number(qs('max_trials').value),
    tune_learning_rates: parseList('tune_lrs', Number), tune_batch_sizes: parseList('tune_batch_sizes', Number), tune_epochs: parseList('tune_epochs', Number),
    tune_weight_decays: parseList('tune_weight_decays', Number), tune_max_lengths: parseList('tune_max_lengths', Number), mlflow: mlflowPayload() };
}
async function startTraining() { const job = await postJson('/api/training/start', trainingPayload()); currentTrainingJobId = job.job_id; setStatus('Training started.'); await refreshAll(); }
async function interruptTraining() {
  const job = cachedJobs.find(j => j.job_type === 'training' && ['queued','running'].includes(j.status));
  if (!job && !currentTrainingJobId) return setStatus('No training is currently running.');
  await api(`/api/jobs/${encodeURIComponent((job || {}).job_id || currentTrainingJobId)}/cancel`, {method: 'POST'}); setStatus('Interrupt requested.'); await refreshAll();
}
function updateTrainingMode() {
  const isAzure = radioValue('training_environment') === 'azure';
  setHidden('training_azure_fields', !isAzure); setHidden('training_local_fields', isAzure); updateLocalRuntime();
}
function updateLocalRuntime() {
  const isLocal = radioValue('training_environment') === 'local';
  const useCuda = qs('local_device').value === 'cuda';
  qs('local_runtime').disabled = !(isLocal && useCuda);
  if (!isLocal || !useCuda) qs('local_runtime').value = 'host';
  if (isLocal && useCuda && qs('local_runtime').value === 'host') qs('local_runtime').value = 'container';
}
function updateTrainingStrategy() {
  const mode = qs('train_mode').value;
  const tuning = ['tune', 'tune_cv'].includes(mode);
  qs('tuning_fields').disabled = !tuning;
  qs('cv_folds').disabled = mode !== 'tune_cv';
  qs('max_trials').disabled = !tuning;
}
function updateAzureInstanceSelects() {
  const azure = (cfg.defaults || {}).azure || {};
  fillSelect('train_azure_instance', (azure.training_instances || {})[qs('train_azure_compute').value] || [], qs('train_azure_instance').value);
  fillSelect('host_azure_instance', (azure.hosting_instances || {})[qs('host_azure_compute').value] || [], qs('host_azure_instance').value);
}
async function updateAzureUrls() {
  const payload = await api(`/api/azure/dashboard-urls?sub_id=${encodeURIComponent(qs('host_azure_sub').value)}&tenant_id=${encodeURIComponent(qs('host_azure_tenant').value)}`);
  qs('azure_mlops_url').value = payload.mlops_url || ''; qs('azure_llmops_url').value = payload.llmops_url || '';
}
async function refreshServerlessDefaults(force = false) {
  if (!force && !serverlessEndpointAuto) return;
  const payload = await api(`/api/azure/serverless-defaults?model_id=${encodeURIComponent(qs('serverless_model_id').value)}`);
  qs('serverless_model_id').value = payload.model_id || '';
  qs('serverless_endpoint').value = payload.endpoint_name || '';
  serverlessEndpointAuto = true;
}

async function refreshModels() {
  const payload = await api(`/api/models?selected_path=${encodeURIComponent(qs('hosted_model_path').value)}`);
  const models = payload.models || [];
  qs('model_select').innerHTML = models.map(m => `<option value="${escapeHtml(m.path)}">${escapeHtml(m.label || m.path)}</option>`).join('') || '<option value="/workspace/outputs/final_model">/workspace/outputs/final_model</option>';
  onModelSelected();
}
function onModelSelected() { if (qs('model_select').value) qs('hosted_model_path').value = qs('model_select').value; }
async function loadRepos() {
  const payload = await postJson('/api/github/repos', {token: qs('github_token').value});
  fillSelect('repo_select', payload.repo_names || []); setStatus('Repositories loaded.'); await loadBranches();
}
async function loadBranches() {
  const repo = qs('repo_select').value;
  if (!repo) return;
  const payload = await postJson('/api/github/branches', {token: qs('github_token').value, repo_name: repo});
  fillSelect('branch_select', payload.branch_names || []); setStatus('Branches loaded.');
}
function updateHostingMode() {
  const isAzure = radioValue('hosting_mode') === 'azure';
  const service = radioValue('azure_service') || 'queued_batch';
  setHidden('hosting_azure_fields', !isAzure);
  setHidden('serverless_fields', !(isAzure && service === 'serverless'));
  setHidden('azure_host_compute_fields', !(isAzure && service !== 'serverless'));
  setHidden('batch_fields', !(isAzure && service === 'queued_batch'));
  if (isAzure && service === 'serverless') refreshServerlessDefaults(false);
  updateAzureUrls().catch(() => {});
}
async function startHosting() {
  const t = timeParts();
  const job = await postJson('/api/hosting/start', {model_dir: qs('hosted_model_path').value, mode: radioValue('hosting_mode'), azure_service: radioValue('azure_service'),
    azure_sub_id: qs('host_azure_sub').value, azure_tenant_id: qs('host_azure_tenant').value, azure_compute: qs('host_azure_compute').value, azure_instance_type: qs('host_azure_instance').value,
    azure_serverless_model_id: qs('serverless_model_id').value, azure_serverless_endpoint_name: qs('serverless_endpoint').value, batch_input_uri: qs('batch_input_uri').value,
    batch_hour: t.batch_hour, batch_minute: t.batch_minute, batch_timezone: qs('batch_timezone').value,
    deployment_gate_golden_path: qs('gate_golden').value, deployment_gate_policy_path: qs('gate_policy').value, drift_golden_path: qs('drift_golden').value, drift_policy_path: qs('drift_policy').value,
    create_github_pr: qs('create_pr').checked, github_token: qs('github_token').value, github_repo: qs('repo_select').value, github_branch: qs('branch_select').value});
  currentHostingJobId = job.job_id; setStatus('Hosting workflow started.'); await refreshAll();
}
async function stopHosting() { await api('/api/hosting/stop', {method: 'POST'}); setStatus('Local hosted model unloaded.'); await refreshAll(); }
function applyHostingPayload(hosting) {
  if (hosting.api_url) qs('endpoint_url').value = hosting.api_url;
  if (hosting.feedback_api_url) qs('feedback_api_url').value = hosting.feedback_api_url;
  if (hosting.feedback_status_url) qs('feedback_status_url').value = hosting.feedback_status_url;
  if (hosting.github_pr_url) qs('github_pr_url').value = hosting.github_pr_url;
  if (hosting.mlops_url) qs('azure_mlops_url').value = hosting.mlops_url;
  if (hosting.llmops_url) qs('azure_llmops_url').value = hosting.llmops_url;
  if (hosting.summary || hosting.message) qs('hosting_summary').value = hosting.summary || hosting.message;
}
function openField(id) { const url = qs(id).value.trim(); if (!url) return alert('No URL is available yet.'); window.open(url, '_blank'); }
async function copyField(id) { const value = qs(id).value.trim(); if (!value) return alert('No value is available yet.'); await navigator.clipboard.writeText(value); setStatus('Copied to clipboard.'); }
async function cancelJob(jobId) { await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {method: 'POST'}); await refreshAll(); }
async function refreshAll() {
  try {
    const [jobsPayload, eventsPayload, hostingPayload] = await Promise.all([api('/api/jobs'), api('/api/events'), api('/api/hosting/current')]);
    cachedJobs = jobsPayload.jobs || [];
    qs('jobs').innerHTML = cachedJobs.map(j => `<tr><td>${escapeHtml(j.job_type)}</td><td class="status-${escapeHtml(j.status)}">${escapeHtml(j.status)}</td><td>${escapeHtml(j.submitted_at || '')}</td><td>${['queued','running'].includes(j.status) ? `<button class="inline warn" onclick="cancelJob('${escapeHtml(j.job_id)}')">Cancel</button>` : ''}</td></tr>`).join('');
    const events = eventsPayload.events || [];
    if (events.length) {
      eventLog = eventLog.concat(events).slice(-120);
      for (const event of events) {
        if (event.message) setStatus(event.message);
        if (event.job_id === currentPromptTestJobId && event.status === 'succeeded' && event.payload?.cases) { promptTestCases = event.payload.cases; renderPromptTests(promptTestCases); }
        if (event.status === 'succeeded' && event.payload?.output_path) { qs('training_csv_path').value = event.payload.output_path; await refreshPromptVersions(); }
        if (event.status === 'succeeded' && event.payload?.selected_model_dir) { qs('hosted_model_path').value = event.payload.selected_model_dir; await refreshModels(); }
        if (event.status === 'succeeded' && event.payload?.operation === 'hosting') { applyHostingPayload(event.payload); }
      }
    }
    qs('events').textContent = eventLog.map(e => `[${e.timestamp}] ${e.status || e.stage}: ${e.message}`).join('\n');
    const hosting = hostingPayload.hosting || {};
    qs('hosting').textContent = JSON.stringify(hosting, null, 2);
    applyHostingPayload(hosting);
  } catch (err) {
    eventLog.push({timestamp: new Date().toISOString(), status: 'failed', message: 'Refresh failed: ' + err.message});
    qs('events').textContent = eventLog.map(e => `[${e.timestamp}] ${e.status || e.stage}: ${e.message}`).join('\n');
  }
}
init().catch(err => alert(err.message));
</script>
</body>
</html>
"""
