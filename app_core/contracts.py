from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELED = "canceled"


@dataclass(slots=True)
class MlflowConfig:
    enabled: bool
    backend: str
    tracking_uri: str = ""
    experiment_name: str = ""
    registered_model_name: str = ""
    disabled_reason: str = ""


@dataclass(slots=True)
class DataPrepRequest:
    input_path: str
    output_path: str
    api_key: str
    model_name: str
    mlflow_config: MlflowConfig


@dataclass(slots=True)
class TrainingRequest:
    csv_path: str
    environment_mode: str
    local_device: str
    local_runtime: str
    azure_sub_id: str
    azure_tenant_id: str
    azure_compute: str
    azure_instance_type: str
    training_options: dict[str, Any]
    mlflow_config: MlflowConfig


@dataclass(slots=True)
class HostingRequest:
    model_dir: str
    mode: str
    auto_install_missing_tools: bool = False
    azure_sub_id: str = ""
    azure_tenant_id: str = ""
    azure_compute: str = "cpu"
    azure_instance_type: str = ""
    azure_service: str = "queued_batch"
    batch_input_uri: str = ""
    batch_hour: int = 0
    batch_minute: int = 0
    batch_timezone: str = "UTC"


@dataclass(slots=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelRecord:
    path: str
    source: str
    created_at: str = ""
    model_version_id: str = ""
    run_id: str = ""
    label: str = ""


@dataclass(slots=True)
class HostingStatus:
    mode: str
    service_kind: str
    api_url: str = ""
    summary: str = ""
    metadata_path: str = ""
    mlops_url: str = ""
    llmops_url: str = ""
    endpoint_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobRecord:
    job_id: str
    job_type: str
    status: str
    submitted_at: str
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProgressEvent:
    job_id: str
    stage: str
    message: str
    percent: float | None
    timestamp: str
    status: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
