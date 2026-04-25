from __future__ import annotations

import json
import os
import re
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote as url_quote

import requests
try:
    from azure.ai.ml import MLClient, Input, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (
        AccountKeyConfiguration,
        AmlCompute,
        AzureBlobDatastore,
        BatchEndpoint,
        CodeConfiguration,
        Data,
        Environment,
        JobSchedule,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        Model,
        ModelBatchDeployment,
        ModelBatchDeploymentSettings,
        RecurrencePattern,
        RecurrenceTrigger,
        Workspace,
    )
    from azure.identity import InteractiveBrowserCredential
    from azure.mgmt.resource import ResourceManagementClient
    AZURE_AVAILABLE = True
except Exception:
    MLClient = Any
    Input = Any
    command = None
    AssetTypes = Any
    AccountKeyConfiguration = Any
    AmlCompute = Any
    AzureBlobDatastore = Any
    BatchEndpoint = Any
    CodeConfiguration = Any
    Data = Any
    Environment = Any
    JobSchedule = Any
    ManagedOnlineDeployment = Any
    ManagedOnlineEndpoint = Any
    Model = Any
    ModelBatchDeployment = Any
    ModelBatchDeploymentSettings = Any
    RecurrencePattern = Any
    RecurrenceTrigger = Any
    Workspace = Any
    InteractiveBrowserCredential = Any
    ResourceManagementClient = Any
    AZURE_AVAILABLE = False

from mlops_utils import clean_optional_string, read_json


Reporter = Callable[[str], None]


class AzurePlatformService:
    def __init__(self, project_dir: str, resource_group: str, workspace_name: str):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.resource_group = resource_group
        self.workspace_name = workspace_name

    def sanitize_azure_name(self, raw_value: str, max_length: int = 32) -> str:
        cleaned = re.sub(r"[^a-z0-9-]", "-", clean_optional_string(raw_value).lower())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        cleaned = cleaned[:max_length].strip("-")
        return cleaned or "log-monitor"

    def ensure_azure_dependencies(self) -> None:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")

    def sanitize_azure_storage_name(self, raw_value: str, max_length: int = 24) -> str:
        cleaned = re.sub(r"[^a-z0-9]", "", clean_optional_string(raw_value).lower())
        cleaned = cleaned[:max_length]
        if len(cleaned) < 3:
            cleaned = (cleaned + "logmonitor")[:max_length]
        return cleaned or "logmonitorstore"

    def sanitize_azure_asset_version(self, raw_value: str, max_length: int = 50) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", clean_optional_string(raw_value))
        cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
        cleaned = cleaned[:max_length].strip("-._")
        return cleaned or "1"

    def build_azure_workspace_id(self, sub_id: str) -> str:
        return (
            f"/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{self.workspace_name}"
        )

    def build_azure_studio_workspace_id(self, sub_id: str) -> str:
        return (
            f"/subscriptions/{sub_id}/resourcegroups/{self.resource_group}"
            f"/workspaces/{self.workspace_name}"
        )

    def build_azure_studio_query(self, sub_id: str, tenant_id: str = "") -> str:
        clean_sub_id = clean_optional_string(sub_id)
        if not clean_sub_id:
            return ""
        studio_wsid = self.build_azure_studio_workspace_id(clean_sub_id)
        query = f"wsid={url_quote(studio_wsid, safe='')}"
        clean_tenant_id = clean_optional_string(tenant_id)
        if clean_tenant_id:
            query += f"&tid={url_quote(clean_tenant_id, safe='')}"
        return query

    def build_azure_studio_url(self, sub_id: str, tenant_id: str = "") -> str:
        query = self.build_azure_studio_query(sub_id, tenant_id)
        if not query:
            return ""
        return f"https://ml.azure.com/experiments?{query}"

    def build_azure_dashboard_urls(self, sub_id: str, tenant_id: str = "") -> tuple[str, str]:
        query = self.build_azure_studio_query(sub_id, tenant_id)
        if not query:
            return "", ""
        url = f"https://ml.azure.com/experiments?{query}"
        return url, url

    def dedupe_instance_candidates(self, candidates: list[str]) -> list[str]:
        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            clean_candidate = clean_optional_string(candidate)
            if clean_candidate and clean_candidate not in seen:
                seen.add(clean_candidate)
                unique_candidates.append(clean_candidate)
        return unique_candidates

    def prioritize_instance_candidates(self, candidates: list[str], preferred: str) -> list[str]:
        ordered = self.dedupe_instance_candidates(candidates)
        preferred_clean = clean_optional_string(preferred)
        if preferred_clean and preferred_clean in ordered:
            ordered.remove(preferred_clean)
            ordered.insert(0, preferred_clean)
        return ordered

    def get_azure_training_instance_candidates(self, azure_compute: str) -> list[str]:
        if clean_optional_string(azure_compute).lower() == "gpu":
            return self.dedupe_instance_candidates(["Standard_NC4as_T4_v3", "Standard_NC6s_v3"])
        return self.dedupe_instance_candidates(
            ["Standard_D2as_v4", "Standard_DS2_v2", "Standard_DS1_v2", "Standard_F2s_v2", "Standard_E2s_v3", "Standard_E4s_v3"]
        )

    def get_azure_host_instance_candidates(self, azure_compute: str) -> list[str]:
        if clean_optional_string(azure_compute).lower() == "gpu":
            return self.dedupe_instance_candidates(["Standard_NC4as_T4_v3", "Standard_NC6s_v3"])
        return self.dedupe_instance_candidates(
            ["Standard_D2as_v4", "Standard_DS2_v2", "Standard_DS1_v2", "Standard_F2s_v2", "Standard_E2s_v3", "Standard_E4s_v3", "Standard_DS3_v2"]
        )

    def get_azure_batch_timezone_options(self) -> list[str]:
        return ["UTC", "Eastern Standard Time", "Central Standard Time", "Mountain Standard Time", "Pacific Standard Time"]

    def get_azure_batch_timezone_iana(self, timezone_name: str) -> str:
        mapping = {
            "UTC": "UTC",
            "Eastern Standard Time": "America/New_York",
            "Central Standard Time": "America/Chicago",
            "Mountain Standard Time": "America/Denver",
            "Pacific Standard Time": "America/Los_Angeles",
        }
        return mapping.get(clean_optional_string(timezone_name), "UTC")

    def parse_daily_time(self, raw_value: str) -> tuple[int, int]:
        clean_value = clean_optional_string(raw_value)
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", clean_value)
        if not match:
            raise ValueError("Enter the batch time as HH:MM using 24-hour time, for example 02:00 or 14:30.")
        hour = int(match.group(1))
        minute = int(match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Batch time must use hours 00-23 and minutes 00-59.")
        return hour, minute

    def is_cloud_accessible_batch_input(self, raw_value: str) -> bool:
        clean_value = clean_optional_string(raw_value)
        if not clean_value:
            return False
        if clean_value.startswith("/subscriptions/") or clean_value.startswith("azureml:"):
            return True
        return re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", clean_value) is not None

    def is_azure_quota_error(self, exc: Exception) -> bool:
        message = clean_optional_string(str(exc)).lower()
        quota_signals = ["not enough quota available", "quota", "additional needed", "vmsize"]
        return any(signal in message for signal in quota_signals)

    def format_azure_hosting_error(self, exc: Exception, attempted_instance_types: list[str] | None = None) -> str:
        message = clean_optional_string(str(exc))
        if not self.is_azure_quota_error(exc):
            return message
        attempted_text = ", ".join(attempted_instance_types or [])
        guidance = "Azure hosting failed because this subscription does not have enough quota in `eastus` for the VM sizes we tried.\n\n"
        if attempted_text:
            guidance += f"Tried instance types:\n- {attempted_text.replace(', ', chr(10) + '- ')}\n\n"
        guidance += (
            "What you can do next:\n"
            "- Request a quota increase in Azure for one of those VM families.\n"
            "- Use a different Azure subscription with available quota.\n"
            "- Change the workspace region from `eastus` if you want to target a region with quota.\n"
            "- Pick a smaller VM size in the hosting form and try again.\n"
        )
        return guidance

    def normalize_arm_template_outputs(self, outputs: Any) -> dict:
        if outputs is None or not hasattr(outputs, "items"):
            return {}
        normalized: dict[str, Any] = {}
        for key, value in outputs.items():
            if isinstance(value, dict) and "value" in value:
                normalized[key] = value.get("value")
            else:
                normalized[key] = value
        return normalized

    def get_azure_management_token(self, credential) -> str:
        self.ensure_azure_dependencies()
        token = credential.get_token("https://management.azure.com/.default")
        return clean_optional_string(getattr(token, "token", ""))

    def azure_management_json_request(
        self,
        credential,
        method: str,
        url: str,
        payload: dict | None = None,
        expected_statuses: tuple[int, ...] = (200, 201, 202),
    ) -> dict:
        headers = {"Authorization": f"Bearer {self.get_azure_management_token(credential)}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        response = requests.request(method.upper(), url, headers=headers, json=payload, timeout=180)
        body_text = response.text.strip()
        if response.status_code not in expected_statuses:
            detail = body_text
            try:
                detail = json.dumps(response.json(), indent=2, ensure_ascii=True)
            except Exception:
                pass
            raise RuntimeError(f"Azure management request failed ({response.status_code} {response.reason}).\n\n{detail[:4000]}")
        if not body_text:
            return {}
        try:
            return response.json()
        except Exception:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(body_text)
                return parsed
            except Exception:
                return {"raw_body": body_text}

    def load_azure_function_bridge_template(self) -> dict:
        template = read_json(str(self.project_dir / "azure_function_bridge_infra.json"))
        if not isinstance(template, dict):
            raise RuntimeError("Could not load the Azure Function bridge ARM template.")
        return template

    def wait_for_resource_group_deployment(self, credential, sub_id: str, deployment_name: str, timeout_seconds: int = 1800) -> dict:
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Resources/deployments/{deployment_name}?api-version=2025-04-01"
        )
        deadline = time.time() + timeout_seconds
        last_state = ""
        while time.time() < deadline:
            payload = self.azure_management_json_request(credential, "GET", url, expected_statuses=(200,))
            properties = payload.get("properties") if isinstance(payload.get("properties"), dict) else {}
            provisioning_state = clean_optional_string(properties.get("provisioningState"))
            if provisioning_state:
                last_state = provisioning_state
            if provisioning_state == "Succeeded":
                return payload
            if provisioning_state in {"Failed", "Canceled"}:
                error_payload = properties.get("error")
                detail = json.dumps(error_payload, indent=2, ensure_ascii=True) if error_payload else json.dumps(payload, indent=2, ensure_ascii=True)
                raise RuntimeError(f"Azure infrastructure deployment {provisioning_state.lower()}.\n\n{detail[:4000]}")
            time.sleep(10)
        raise TimeoutError(
            "Timed out waiting for the Azure infrastructure deployment to finish."
            + (f" Last known state: {last_state}." if last_state else "")
        )

    def deploy_azure_function_bridge_infrastructure(
        self,
        credential,
        sub_id: str,
        function_app_name: str,
        function_plan_name: str,
        storage_account_name: str,
        service_bus_namespace_name: str,
        service_bus_queue_name: str,
    ) -> dict:
        deployment_name = self.sanitize_azure_name(f"log-monitor-bridge-{function_app_name}", max_length=50)
        template = self.load_azure_function_bridge_template()
        parameters = {
            "location": {"value": "eastus"},
            "workspaceResourceId": {"value": self.build_azure_workspace_id(sub_id)},
            "functionPlanName": {"value": function_plan_name},
            "functionAppName": {"value": function_app_name},
            "storageAccountName": {"value": storage_account_name},
            "serviceBusNamespaceName": {"value": service_bus_namespace_name},
            "serviceBusQueueName": {"value": service_bus_queue_name},
        }
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Resources/deployments/{deployment_name}?api-version=2025-04-01"
        )
        self.azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={"properties": {"mode": "Incremental", "template": template, "parameters": parameters}},
            expected_statuses=(200, 201, 202),
        )
        deployment = self.wait_for_resource_group_deployment(credential=credential, sub_id=sub_id, deployment_name=deployment_name)
        properties = deployment.get("properties") if isinstance(deployment.get("properties"), dict) else {}
        normalized = self.normalize_arm_template_outputs(properties.get("outputs"))
        if not normalized:
            raise RuntimeError("Azure infrastructure deployment completed but did not return the expected outputs.")
        return normalized

    def build_function_bridge_package(self, package_name: str) -> str:
        bridge_root = self.project_dir / "azure_function_bridge"
        if not bridge_root.exists():
            raise RuntimeError("The Azure Function bridge files are missing from this project.")
        package_path = Path(tempfile.gettempdir()) / f"{package_name}.zip"
        if package_path.exists():
            package_path.unlink()
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in bridge_root.rglob("*"):
                if file_path.is_file() and "__pycache__" not in file_path.parts:
                    archive.write(file_path, arcname=str(file_path.relative_to(bridge_root)))
        return str(package_path)

    def upload_function_bridge_package(
        self,
        storage_connection_string: str,
        storage_account_name: str,
        storage_account_key: str,
        package_path: str,
        package_container_name: str = "functionpkgs",
    ) -> str:
        from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

        blob_service = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service.get_container_client(package_container_name)
        package_blob_name = f"releases/{Path(package_path).name}"
        with open(package_path, "rb") as handle:
            container_client.upload_blob(name=package_blob_name, data=handle, overwrite=True)

        sas_token = generate_blob_sas(
            account_name=storage_account_name,
            container_name=package_container_name,
            blob_name=package_blob_name,
            account_key=storage_account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        return f"https://{storage_account_name}.blob.core.windows.net/{package_container_name}/{package_blob_name}?{sas_token}"

    def set_function_app_settings(self, credential, sub_id: str, function_app_name: str, settings: dict) -> None:
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/config/appsettings?api-version=2025-03-01"
        )
        self.azure_management_json_request(credential, "PUT", url, payload={"properties": settings}, expected_statuses=(200,))

    def trigger_function_app_onedeploy(self, credential, sub_id: str, function_app_name: str, package_uri: str) -> None:
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/extensions/onedeploy?api-version=2025-03-01"
        )
        self.azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={"properties": {"packageUri": package_uri, "remoteBuild": True}, "type": "zip"},
            expected_statuses=(200, 202),
        )

    def wait_for_function_bridge_endpoint(
        self,
        credential,
        sub_id: str,
        function_app_name: str,
        function_host_name: str,
        function_name: str = "ingest_log",
    ) -> tuple[str, str]:
        host_keys_url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/host/default/listkeys?api-version=2025-03-01"
        )
        function_secrets_url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/functions/{function_name}/listsecrets?api-version=2025-05-01"
        )
        deadline = time.time() + 900
        last_error = ""
        while time.time() < deadline:
            try:
                payload = self.azure_management_json_request(credential, "POST", host_keys_url, payload={}, expected_statuses=(200,))
                function_keys = payload.get("functionKeys") if isinstance(payload.get("functionKeys"), dict) else {}
                function_key = clean_optional_string(function_keys.get("default"))
                if not function_key and function_keys:
                    function_key = clean_optional_string(next(iter(function_keys.values()), ""))
                if function_key:
                    return f"https://{function_host_name}/api/logs?code={function_key}", function_key
            except Exception as exc:
                last_error = str(exc)
            try:
                payload = self.azure_management_json_request(credential, "POST", function_secrets_url, payload={}, expected_statuses=(200,))
                trigger_url = clean_optional_string(payload.get("trigger_url"))
                function_key = clean_optional_string(payload.get("key"))
                if trigger_url:
                    return trigger_url, function_key
                if function_key:
                    return f"https://{function_host_name}/api/logs?code={function_key}", function_key
            except Exception as exc:
                last_error = str(exc)
            time.sleep(10)
        raise RuntimeError(
            "The Azure Function API was deployed, but the app could not retrieve the trigger URL in time.\n\n"
            f"Last error:\n{last_error}"
        )

    def ensure_azure_blob_datastore(self, ml_client, datastore_name: str, storage_account_name: str, container_name: str, storage_account_key: str):
        datastore = AzureBlobDatastore(
            name=datastore_name,
            account_name=storage_account_name,
            container_name=container_name,
            credentials=AccountKeyConfiguration(account_key=storage_account_key),
            description="Queued log batches for Azure ML inference.",
        )
        return ml_client.datastores.create_or_update(datastore)

    def ensure_azure_workspace(self, sub_id: str, tenant_id: str, emit: Reporter | None = None, credential=None) -> MLClient:
        self.ensure_azure_dependencies()
        if credential is None:
            if emit:
                emit("Please log in to Azure in your web browser...")
            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
        ml_client = MLClient(credential, sub_id, self.resource_group, self.workspace_name)
        resource_client = ResourceManagementClient(credential, sub_id)

        if emit:
            emit("Checking Azure Resource Group...")
        try:
            resource_client.resource_groups.get(self.resource_group)
        except Exception:
            if emit:
                emit("Creating Azure Resource Group...")
            resource_client.resource_groups.create_or_update(self.resource_group, {"location": "eastus"})

        if emit:
            emit("Verifying Azure ML registration...")
        resource_client.providers.register("Microsoft.MachineLearningServices")
        while True:
            provider_info = resource_client.providers.get("Microsoft.MachineLearningServices")
            if provider_info.registration_state == "Registered":
                break
            if emit:
                emit("Activating Azure ML services...")
            time.sleep(10)

        if emit:
            emit("Ensuring Azure ML Workspace exists...")
        try:
            ml_client.workspaces.get(self.workspace_name)
        except Exception:
            if emit:
                emit("Creating Azure ML Workspace...")
            workspace = Workspace(name=self.workspace_name, location="eastus")
            ml_client.workspaces.begin_create(workspace).result()
        return ml_client

    def create_interactive_credential(self, tenant_id: str):
        self.ensure_azure_dependencies()
        return InteractiveBrowserCredential(tenant_id=tenant_id)

    def cancel_job(self, ml_client, job_name: str, wait: bool = False) -> None:
        self.ensure_azure_dependencies()
        clean_job_name = clean_optional_string(job_name)
        if not clean_job_name:
            return
        jobs_client = ml_client.jobs
        if hasattr(jobs_client, "begin_cancel"):
            poller = jobs_client.begin_cancel(clean_job_name)
            if wait and hasattr(poller, "result"):
                poller.result()
            return
        if hasattr(jobs_client, "cancel"):
            jobs_client.cancel(clean_job_name)
            return
        raise RuntimeError("Azure ML SDK does not expose a job cancellation method on this client version.")

    def ensure_aml_compute(self, ml_client, compute_name: str, instance_type: str) -> None:
        self.ensure_azure_dependencies()
        compute = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size=instance_type,
            min_instances=0,
            max_instances=1,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(compute).result()

    def register_azure_data_asset(
        self,
        ml_client,
        csv_path: str,
        *,
        asset_name: str,
        asset_version: str,
        description: str = "",
        tags: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        self.ensure_azure_dependencies()
        resolved_csv = Path(csv_path).expanduser().resolve()
        if not resolved_csv.exists():
            raise FileNotFoundError(f"Training CSV does not exist: {resolved_csv}")

        clean_name = self.sanitize_azure_name(asset_name, max_length=255)
        clean_version = self.sanitize_azure_asset_version(asset_version)
        clean_tags = {
            clean_optional_string(key): clean_optional_string(value)
            for key, value in (tags or {}).items()
            if clean_optional_string(key) and clean_optional_string(value)
        }
        azureml_uri = f"azureml:{clean_name}:{clean_version}"
        try:
            existing_asset = ml_client.data.get(name=clean_name, version=clean_version)
        except Exception:
            existing_asset = None
        if existing_asset is not None:
            return {
                "azure_data_asset_name": clean_name,
                "azure_data_asset_version": clean_version,
                "azure_data_asset_uri": azureml_uri,
                "azure_data_asset_id": clean_optional_string(getattr(existing_asset, "id", "")),
                "azure_data_asset_path": clean_optional_string(getattr(existing_asset, "path", "")),
            }

        data_asset = Data(
            path=str(resolved_csv),
            type=AssetTypes.URI_FILE,
            name=clean_name,
            version=clean_version,
            description=clean_optional_string(description),
            tags=clean_tags,
        )
        registered_asset = ml_client.data.create_or_update(data_asset)
        return {
            "azure_data_asset_name": clean_name,
            "azure_data_asset_version": clean_version,
            "azure_data_asset_uri": azureml_uri,
            "azure_data_asset_id": clean_optional_string(getattr(registered_asset, "id", "")),
            "azure_data_asset_path": clean_optional_string(getattr(registered_asset, "path", "")),
        }

    def submit_azure_training_job(
        self,
        ml_client,
        csv_path: str,
        compute_name: str,
        mlflow_install_fragment: str,
        export_segment: str,
        train_cli_segment: str,
        data_input_path: str = "",
        experiment_name: str = "deberta-log-classification",
    ) -> str:
        self.ensure_azure_dependencies()
        if command is None:
            raise RuntimeError("Azure ML command job creation is unavailable in this Python environment.")
        safe_csv_path = clean_optional_string(data_input_path) or csv_path.replace("\\", "/")
        train_job = command(
            inputs={"training_data": Input(type="uri_file", path=safe_csv_path, mode="download")},
            compute=compute_name,
            environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest",
            code=".",
            command=(
                "pip install --upgrade numpy==1.23.5 pandas==1.5.3 transformers==4.24.0 sentencepiece==0.1.99 protobuf==3.20.3 scikit-learn==1.1.3 "
                f"{mlflow_install_fragment}"
                f"&& {export_segment} && USE_TF=0 python train.py --data ${{inputs.training_data}}{train_cli_segment}"
            ),
            experiment_name=experiment_name,
        )
        returned_job = ml_client.jobs.create_or_update(train_job)
        return clean_optional_string(getattr(returned_job, "name", ""))

    def get_job_status(self, ml_client, job_name: str) -> str:
        self.ensure_azure_dependencies()
        details = ml_client.jobs.get(job_name)
        return clean_optional_string(getattr(details, "status", ""))

    def download_job_output(self, ml_client, job_name: str, download_path: str) -> None:
        self.ensure_azure_dependencies()
        ml_client.jobs.download(name=job_name, download_path=download_path, all=False)

    def delete_compute(self, ml_client, compute_name: str) -> None:
        self.ensure_azure_dependencies()
        ml_client.compute.begin_delete(compute_name).result()

    def deploy_azure_online_endpoint(
        self,
        ml_client,
        model_dir: str,
        endpoint_name: str,
        deployment_name: str,
        environment_name: str,
        model_name: str,
        azure_compute: str,
        preferred_instance_type: str,
        endpoint_auth_mode: str = "key",
        emit: Reporter | None = None,
    ) -> dict:
        self.ensure_azure_dependencies()
        attempted_instance_types: list[str] = []
        instance_candidates = self.prioritize_instance_candidates(
            self.get_azure_host_instance_candidates(azure_compute),
            preferred_instance_type,
        )
        if emit:
            emit("Registering model in Azure ML...")
        registered_model = ml_client.models.create_or_update(
            Model(path=model_dir.replace("\\", "/"), name=model_name, type=AssetTypes.CUSTOM_MODEL, description="Log Monitor generated DeBERTa model")
        )
        if emit:
            emit("Creating Azure inference environment...")
        environment = ml_client.environments.create_or_update(
            Environment(
                name=environment_name,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
                conda_file=os.path.join(self.project_dir, "azure_inference_conda.yml"),
                description="Inference environment for Log Monitor hosted API",
            )
        )
        if emit:
            emit("Creating Azure online endpoint...")
        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode=endpoint_auth_mode, description="Hosted prediction API for Log Monitor")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        selected_instance_type = ""
        last_error = None
        for instance_type in instance_candidates:
            attempted_instance_types.append(instance_type)
            if emit:
                emit(f"Deploying model to Azure endpoint ({instance_type})...")
            try:
                deployment = ManagedOnlineDeployment(
                    name=deployment_name,
                    endpoint_name=endpoint_name,
                    model=registered_model,
                    environment=environment,
                    code_configuration=CodeConfiguration(code=self.project_dir, scoring_script="azure_score.py"),
                    instance_type=instance_type,
                    instance_count=1,
                )
                ml_client.online_deployments.begin_create_or_update(deployment).result()
                selected_instance_type = instance_type
                break
            except Exception as exc:
                last_error = exc
                if self.is_azure_quota_error(exc) and instance_type != instance_candidates[-1]:
                    continue
                raise
        if not selected_instance_type:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Azure deployment did not complete and no scoring instance was selected.")
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        endpoint_details = ml_client.online_endpoints.get(endpoint_name)
        scoring_uri = clean_optional_string(getattr(endpoint_details, "scoring_uri", ""))
        if not scoring_uri:
            raise RuntimeError("Azure deployment completed but no scoring URI was returned.")
        return {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "instance_type": selected_instance_type,
            "api_url": scoring_uri,
            "attempted_instance_types": attempted_instance_types,
        }

    def create_daily_batch_schedule(
        self,
        ml_client,
        endpoint_name: str,
        deployment_name: str,
        batch_input_uri: str,
        schedule_name: str,
        batch_hour: int,
        batch_minute: int,
        batch_timezone: str,
        emit: Reporter | None = None,
    ) -> str:
        self.ensure_azure_dependencies()
        if emit:
            emit("Creating daily Azure batch schedule...")
        seed_job = ml_client.batch_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            input=Input(path=batch_input_uri),
            experiment_name="log-monitor-batch-schedules",
        )
        seed_job_name = clean_optional_string(getattr(seed_job, "name", ""))
        schedule = JobSchedule(
            name=schedule_name,
            display_name="Log Monitor Daily Batch Schedule",
            description="Runs the Log Monitor batch endpoint once per day.",
            trigger=RecurrenceTrigger(
                frequency="day",
                interval=1,
                schedule=RecurrencePattern(hours=batch_hour, minutes=batch_minute),
                time_zone=batch_timezone or "UTC",
            ),
            create_job=seed_job_name,
        )
        ml_client.schedules.begin_create_or_update(schedule=schedule).result()
        try:
            self.cancel_job(ml_client, seed_job_name, wait=True)
        except Exception:
            pass
        return seed_job_name

    def deploy_azure_batch_endpoint(
        self,
        ml_client,
        model_dir: str,
        azure_compute: str,
        preferred_instance_type: str,
        endpoint_name: str,
        environment_name: str,
        model_name: str,
        endpoint_auth_mode: str = "aad_token",
        emit: Reporter | None = None,
    ) -> dict:
        self.ensure_azure_dependencies()
        attempted_instance_types: list[str] = []
        deployment_name = "default"
        compute_name = "log-monitor-batch-gpu" if clean_optional_string(azure_compute).lower() == "gpu" else "log-monitor-batch-cpu"
        instance_candidates = self.prioritize_instance_candidates(self.get_azure_host_instance_candidates(azure_compute), preferred_instance_type)
        selected_instance_type = ""

        if emit:
            emit("Registering model in Azure ML...")
        registered_model = ml_client.models.create_or_update(
            Model(path=model_dir.replace("\\", "/"), name=model_name, type=AssetTypes.CUSTOM_MODEL, description="Log Monitor generated DeBERTa model")
        )
        if emit:
            emit("Creating Azure batch inference environment...")
        environment = ml_client.environments.create_or_update(
            Environment(
                name=environment_name,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
                conda_file=os.path.join(self.project_dir, "azure_batch_inference_conda.yml"),
                description="Batch inference environment for Log Monitor hosted API",
            )
        )
        last_deployment_error = None
        for instance_type in instance_candidates:
            attempted_instance_types.append(instance_type)
            if emit:
                emit(f"Ensuring Azure batch compute ({instance_type})...")
            try:
                compute = AmlCompute(name=compute_name, type="amlcompute", size=instance_type, min_instances=0, max_instances=1, idle_time_before_scale_down=120)
                ml_client.compute.begin_create_or_update(compute).result()
                selected_instance_type = instance_type
                break
            except Exception as exc:
                last_deployment_error = exc
                if self.is_azure_quota_error(exc) and instance_type != instance_candidates[-1]:
                    continue
                raise
        if not selected_instance_type:
            if last_deployment_error is not None:
                raise last_deployment_error
            raise RuntimeError("Azure hosting could not provision any batch compute cluster.")

        if emit:
            emit("Creating Azure batch endpoint...")
        endpoint = BatchEndpoint(name=endpoint_name, auth_mode=endpoint_auth_mode, description="Batch endpoint for Log Monitor")
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

        if emit:
            emit("Creating Azure batch deployment...")
        deployment = ModelBatchDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=registered_model,
            environment=environment,
            compute=compute_name,
            code_configuration=CodeConfiguration(code=self.project_dir, scoring_script="azure_batch_score.py"),
            settings=ModelBatchDeploymentSettings(mini_batch_size=1, instance_count=1),
        )
        ml_client.batch_deployments.begin_create_or_update(deployment).result()

        endpoint = ml_client.batch_endpoints.get(endpoint_name)
        try:
            endpoint.default_deployment_name = deployment_name
        except Exception:
            pass
        try:
            if getattr(endpoint, "defaults", None) is not None:
                endpoint.defaults.deployment_name = deployment_name
        except Exception:
            pass
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
        endpoint_details = ml_client.batch_endpoints.get(endpoint_name)
        scoring_uri = clean_optional_string(getattr(endpoint_details, "scoring_uri", ""))
        if not scoring_uri:
            raise RuntimeError("Azure deployment completed but no scoring URI was returned.")
        return {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "instance_type": selected_instance_type,
            "compute_name": compute_name,
            "api_url": scoring_uri,
            "attempted_instance_types": attempted_instance_types,
        }
