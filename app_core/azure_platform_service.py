from __future__ import annotations

import json
import os
import re
import tempfile
import time
import zipfile
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote as url_quote, unquote as url_unquote

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
    try:
        from azure.ai.ml.entities import ServerlessEndpoint
    except Exception:
        ServerlessEndpoint = None
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
    ServerlessEndpoint = None
    Workspace = Any
    InteractiveBrowserCredential = Any
    ResourceManagementClient = Any
    AZURE_AVAILABLE = False

from mlops_utils import clean_optional_string, read_json


Reporter = Callable[[str], None]
DEFAULT_SERVERLESS_MODEL_ID = "azureml://registries/azureml/models/Phi-4-mini-instruct"
COMMUNICATION_API_VERSION = "2025-09-01"
FUNCTION_BRIDGE_WORKER_SETTINGS = {
    "AzureWebJobsFeatureFlags": "EnableWorkerIndexing",
    "PYTHON_ENABLE_WORKER_EXTENSIONS": "1",
    "PYTHON_ISOLATE_WORKER_DEPENDENCIES": "1",
}


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

    def sanitize_azure_endpoint_name(self, raw_value: str, max_length: int = 52) -> str:
        cleaned = re.sub(r"[^a-z0-9-]", "-", clean_optional_string(raw_value).lower())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        if not cleaned or not cleaned[0].isalpha():
            cleaned = f"log-monitor-{cleaned}".strip("-")
        cleaned = cleaned[:max_length].strip("-")
        return cleaned or "log-monitor-endpoint"

    def get_default_serverless_model_id(self) -> str:
        return DEFAULT_SERVERLESS_MODEL_ID

    def normalize_serverless_model_id(self, raw_value: str) -> str:
        clean_value = clean_optional_string(raw_value).strip()
        clean_value = re.sub(r"/versions/[^/]+/?$", "", clean_value)
        return clean_value

    def extract_serverless_model_name(self, model_id: str) -> str:
        normalized_model_id = self.normalize_serverless_model_id(model_id)
        if not normalized_model_id:
            return "model"
        parts = [part for part in normalized_model_id.replace("\\", "/").split("/") if part]
        return parts[-1] if parts else "model"

    def build_default_serverless_endpoint_name(self, model_id: str, suffix: str = "") -> str:
        clean_suffix = self.sanitize_azure_name(suffix, max_length=12) if clean_optional_string(suffix) else ""
        base_max_length = 52 - len(clean_suffix) - (1 if clean_suffix else 0)
        base_name = self.sanitize_azure_endpoint_name(
            f"log-monitor-{self.extract_serverless_model_name(model_id)}",
            max_length=max(base_max_length, 20),
        )
        if clean_suffix:
            return self.sanitize_azure_endpoint_name(f"{base_name}-{clean_suffix}")
        return base_name

    def ensure_azure_dependencies(self) -> None:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")

    def sanitize_azure_storage_name(self, raw_value: str, max_length: int = 24) -> str:
        cleaned = re.sub(r"[^a-z0-9]", "", clean_optional_string(raw_value).lower())
        cleaned = cleaned[:max_length]
        if len(cleaned) < 3:
            cleaned = (cleaned + "logmonitor")[:max_length]
        return cleaned or "logmonitorstore"

    def sanitize_azure_asset_version(self, raw_value: str, max_length: int = 30) -> str:
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

    def build_azure_endpoints_studio_url(self, sub_id: str, tenant_id: str = "") -> str:
        query = self.build_azure_studio_query(sub_id, tenant_id)
        if not query:
            return ""
        return f"https://ml.azure.com/endpoints?{query}&reloadCount=1"

    def build_azure_dashboard_urls(self, sub_id: str, tenant_id: str = "") -> tuple[str, str]:
        query = self.build_azure_studio_query(sub_id, tenant_id)
        if not query:
            return "", ""
        url = f"https://ml.azure.com/experiments?{query}"
        return url, url

    def build_serverless_endpoint_resource_id(self, sub_id: str, endpoint_name: str) -> str:
        clean_sub_id = clean_optional_string(sub_id)
        clean_endpoint_name = clean_optional_string(endpoint_name)
        if not clean_sub_id or not clean_endpoint_name:
            return ""
        return (
            f"/subscriptions/{clean_sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{self.workspace_name}"
            f"/serverlessEndpoints/{clean_endpoint_name}"
        )

    def build_serverless_endpoint_portal_url(self, sub_id: str, tenant_id: str, endpoint_name: str) -> str:
        resource_id = self.build_serverless_endpoint_resource_id(sub_id, endpoint_name)
        if not resource_id:
            return ""
        tenant_fragment = f"@{tenant_id}/" if clean_optional_string(tenant_id) else ""
        return f"https://portal.azure.com/#{tenant_fragment}resource{resource_id}/overview"

    def build_serverless_endpoint_management_url(self, sub_id: str, endpoint_name: str, api_version: str = "2024-04-01-preview") -> str:
        clean_sub_id = clean_optional_string(sub_id)
        clean_endpoint_name = clean_optional_string(endpoint_name)
        if not clean_sub_id or not clean_endpoint_name:
            return ""
        return (
            f"https://management.azure.com/subscriptions/{clean_sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{self.workspace_name}"
            f"/serverlessEndpoints/{clean_endpoint_name}?api-version={api_version}"
        )

    def build_serverless_endpoint_collection_management_url(self, sub_id: str, api_version: str = "2024-04-01-preview") -> str:
        clean_sub_id = clean_optional_string(sub_id)
        if not clean_sub_id:
            return ""
        return (
            f"https://management.azure.com/subscriptions/{clean_sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{self.workspace_name}"
            f"/serverlessEndpoints?api-version={api_version}"
        )

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
            [
                "Standard_D2as_v4",
                "Standard_DS2_v2",
                "Standard_DS1_v2",
                "Standard_F2s_v2",
                "Standard_E2s_v3",
                "Standard_DS3_v2",
                "Standard_E4s_v3",
            ]
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

    def wait_for_azure_poller(
        self,
        poller,
        *,
        action: str,
        emit: Reporter | None = None,
        timeout_seconds: int = 3600,
        poll_interval_seconds: int = 30,
    ):
        deadline = time.time() + max(int(timeout_seconds), 1)
        started_at = time.time()
        interval = max(int(poll_interval_seconds), 1)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for Azure operation: {action}.")
            try:
                return poller.result(timeout=min(interval, remaining))
            except AttributeError as exc:
                message = clean_optional_string(exc)
                if "auth_mode" in message or "endpoint_compute_type" in message:
                    if emit:
                        emit(f"{action} returned an incomplete Azure SDK response; verifying the resource directly...")
                    return None
                raise
            except FutureTimeoutError:
                if emit:
                    elapsed_minutes = (time.time() - started_at) / 60.0
                    status_text = ""
                    try:
                        status_text = clean_optional_string(poller.status())
                    except Exception:
                        status_text = ""
                    suffix = f" Status: {status_text}" if status_text else ""
                    emit(f"{action} still running ({elapsed_minutes:.1f} min elapsed).{suffix}")

    def wait_for_online_endpoint_ready(
        self,
        ml_client,
        endpoint_name: str,
        *,
        action: str = "Waiting for Azure online endpoint",
        emit: Reporter | None = None,
        timeout_seconds: int = 1800,
        poll_interval_seconds: int = 15,
    ):
        deadline = time.time() + max(int(timeout_seconds), 1)
        interval = max(int(poll_interval_seconds), 1)
        last_state = ""
        while time.time() < deadline:
            endpoint = ml_client.online_endpoints.get(endpoint_name)
            if endpoint is not None:
                state = clean_optional_string(getattr(endpoint, "provisioning_state", ""))
                if state:
                    last_state = state
                state_lower = state.lower()
                if state_lower == "succeeded":
                    return endpoint
                if state_lower in {"failed", "canceled"}:
                    raise RuntimeError(f"Azure endpoint '{endpoint_name}' provisioning {state_lower}.")
            if emit:
                emit(f"{action}: {last_state or 'pending'}")
            time.sleep(interval)
        raise TimeoutError(
            f"Timed out waiting for Azure endpoint '{endpoint_name}' to become ready."
            + (f" Last state: {last_state}." if last_state else "")
        )

    def wait_for_serverless_endpoint_ready(
        self,
        ml_client,
        endpoint_name: str,
        *,
        action: str = "Waiting for Azure serverless endpoint",
        emit: Reporter | None = None,
        timeout_seconds: int = 1800,
        poll_interval_seconds: int = 15,
    ):
        deadline = time.time() + max(int(timeout_seconds), 1)
        interval = max(int(poll_interval_seconds), 1)
        last_state = ""
        while time.time() < deadline:
            endpoint = ml_client.serverless_endpoints.get(endpoint_name)
            if endpoint is not None:
                state = clean_optional_string(
                    getattr(endpoint, "provisioning_state", "")
                    or getattr(endpoint, "endpoint_state", "")
                )
                if state:
                    last_state = state
                state_lower = state.lower()
                if state_lower in {"succeeded", "ready", "healthy"}:
                    return endpoint
                if state_lower in {"failed", "canceled", "unhealthy"}:
                    raise RuntimeError(f"Azure serverless endpoint '{endpoint_name}' provisioning {state_lower}.")
            if emit:
                emit(f"{action}: {last_state or 'pending'}")
            time.sleep(interval)
        raise TimeoutError(
            f"Timed out waiting for Azure serverless endpoint '{endpoint_name}' to become ready."
            + (f" Last state: {last_state}." if last_state else "")
        )

    def get_raw_online_deployment(self, ml_client, endpoint_name: str, deployment_name: str):
        deployments_client = ml_client.online_deployments
        raw_client = getattr(deployments_client, "_online_deployment", None)
        resource_group_name = clean_optional_string(getattr(deployments_client, "_resource_group_name", ""))
        workspace_name = clean_optional_string(getattr(deployments_client, "_workspace_name", ""))
        init_kwargs = getattr(deployments_client, "_init_kwargs", {})
        if raw_client is None or not resource_group_name or not workspace_name:
            return deployments_client.get(name=deployment_name, endpoint_name=endpoint_name)
        return raw_client.get(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
            **init_kwargs,
        )

    def wait_for_online_deployment_ready(
        self,
        ml_client,
        endpoint_name: str,
        deployment_name: str,
        *,
        action: str = "Waiting for Azure online deployment",
        emit: Reporter | None = None,
        timeout_seconds: int = 3600,
        poll_interval_seconds: int = 30,
    ):
        deadline = time.time() + max(int(timeout_seconds), 1)
        interval = max(int(poll_interval_seconds), 1)
        last_state = ""
        while time.time() < deadline:
            try:
                deployment = self.get_raw_online_deployment(ml_client, endpoint_name, deployment_name)
                properties = getattr(deployment, "properties", None)
                state = clean_optional_string(
                    getattr(deployment, "provisioning_state", "")
                    or getattr(properties, "provisioning_state", "")
                )
                if state:
                    last_state = state
                state_lower = state.lower()
                if state_lower == "succeeded":
                    return deployment
                if state_lower in {"failed", "canceled"}:
                    raise RuntimeError(f"Azure deployment '{deployment_name}' provisioning {state_lower}.")
            except AttributeError as exc:
                if "endpoint_compute_type" not in clean_optional_string(exc):
                    raise
                last_state = last_state or "response incomplete"
            except Exception as exc:
                if "not found" not in clean_optional_string(exc).lower():
                    raise
                last_state = last_state or "not found"
            if emit:
                emit(f"{action}: {last_state or 'pending'}")
            time.sleep(interval)
        raise TimeoutError(
            f"Timed out waiting for Azure deployment '{deployment_name}' to become ready."
            + (f" Last state: {last_state}." if last_state else "")
        )

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

    def is_transient_azure_resource_error(self, exc: Exception) -> bool:
        message = clean_optional_string(str(exc)).lower()
        signals = (
            "parentresourcenotfound",
            "invalidresourceoperation",
            "being provisioned",
            "provisioningstate",
            "state: 'accepted'",
            "state 'accepted'",
            "409 conflict",
        )
        return any(signal in message for signal in signals)

    def retry_azure_management_json_request(
        self,
        credential,
        method: str,
        url: str,
        payload: dict | None = None,
        expected_statuses: tuple[int, ...] = (200, 201, 202),
        action: str = "Azure resource operation",
        emit: Reporter | None = None,
        timeout_seconds: int = 600,
        poll_interval: int = 10,
    ) -> dict:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while True:
            try:
                return self.azure_management_json_request(
                    credential,
                    method,
                    url,
                    payload=payload,
                    expected_statuses=expected_statuses,
                )
            except RuntimeError as exc:
                last_error = exc
                if not self.is_transient_azure_resource_error(exc) or time.time() >= deadline:
                    raise
                if emit:
                    emit(f"{action} is waiting for Azure provisioning to finish...")
                time.sleep(poll_interval)

        if last_error is not None:
            raise last_error
        raise TimeoutError(f"{action} did not complete before the timeout.")

    def wait_for_azure_resource_ready(
        self,
        credential,
        resource_url: str,
        resource_name: str,
        emit: Reporter | None = None,
        timeout_seconds: int = 600,
        poll_interval: int = 10,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        clean_name = clean_optional_string(resource_name) or "Azure resource"
        last_payload: dict[str, Any] = {}
        while True:
            try:
                payload = self.azure_management_json_request(credential, "GET", resource_url)
                last_payload = payload if isinstance(payload, dict) else {}
                provisioning_state = self.get_arm_provisioning_state(last_payload).lower()
                if provisioning_state in {"", "succeeded"}:
                    return last_payload
                if provisioning_state in {"failed", "canceled", "cancelled"}:
                    raise RuntimeError(f"{clean_name} provisioning failed with state: {provisioning_state}")
                if emit:
                    emit(f"Waiting for {clean_name} provisioning ({provisioning_state})...")
            except RuntimeError as exc:
                if not self.is_transient_azure_resource_error(exc) and "not found" not in clean_optional_string(str(exc)).lower():
                    raise
                if emit:
                    emit(f"Waiting for {clean_name} to become available...")

            if time.time() >= deadline:
                last_state = self.get_arm_provisioning_state(last_payload) or "unknown"
                raise TimeoutError(f"Timed out waiting for {clean_name} provisioning. Last state: {last_state}")
            time.sleep(poll_interval)

    def azure_management_list_values(self, credential, url: str) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        next_url = clean_optional_string(url)
        while next_url:
            payload = self.azure_management_json_request(credential, "GET", next_url)
            page_values = payload.get("value", []) if isinstance(payload, dict) else []
            if isinstance(page_values, list):
                values.extend(item for item in page_values if isinstance(item, dict))
            next_url = clean_optional_string(payload.get("nextLink")) if isinstance(payload, dict) else ""
        return values

    def extract_resource_group_from_arm_id(self, resource_id: str) -> str:
        match = re.search(r"/resourceGroups/([^/]+)", clean_optional_string(resource_id), flags=re.IGNORECASE)
        return match.group(1) if match else ""

    def ensure_resource_group(self, credential, sub_id: str, resource_group: str = "", location: str = "eastus") -> str:
        self.ensure_azure_dependencies()
        clean_group = clean_optional_string(resource_group) or self.resource_group
        resource_client = ResourceManagementClient(credential, sub_id)
        try:
            resource_client.resource_groups.get(clean_group)
        except Exception:
            resource_client.resource_groups.create_or_update(clean_group, {"location": location})
        return clean_group

    def list_acs_communication_services(self, credential, sub_id: str) -> list[dict[str, Any]]:
        clean_sub_id = clean_optional_string(sub_id)
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
            f"/providers/Microsoft.Communication/communicationServices?api-version={COMMUNICATION_API_VERSION}"
        )
        return self.azure_management_list_values(credential, url)

    def create_or_update_acs_communication_service(
        self,
        credential,
        sub_id: str,
        service_name: str,
        resource_group: str,
        linked_domain_id: str = "",
        emit: Reporter | None = None,
    ) -> dict[str, Any]:
        clean_sub_id = clean_optional_string(sub_id)
        clean_group = self.ensure_resource_group(credential, clean_sub_id, resource_group)
        clean_name = self.sanitize_azure_name(service_name, max_length=63)
        linked_domains = []
        clean_domain_id = clean_optional_string(linked_domain_id)
        if clean_domain_id:
            linked_domains.append(clean_domain_id)
        if emit:
            emit(f"Creating Azure Communication Services resource {clean_name}...")
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
            f"/resourceGroups/{url_quote(clean_group, safe='')}"
            f"/providers/Microsoft.Communication/communicationServices/{url_quote(clean_name, safe='')}"
            f"?api-version={COMMUNICATION_API_VERSION}"
        )
        properties: dict[str, Any] = {"dataLocation": "United States"}
        if linked_domains:
            properties["linkedDomains"] = linked_domains
        resource = self.azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={"location": "Global", "properties": properties},
        )
        resource["id"] = clean_optional_string(resource.get("id")) or (
            f"/subscriptions/{clean_sub_id}/resourceGroups/{clean_group}"
            f"/providers/Microsoft.Communication/communicationServices/{clean_name}"
        )
        resource["name"] = clean_optional_string(resource.get("name")) or clean_name
        resource_properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
        resource_properties.setdefault("dataLocation", "United States")
        resource_properties.setdefault("hostName", f"{clean_name}.communication.azure.com")
        if linked_domains:
            resource_properties.setdefault("linkedDomains", linked_domains)
        resource["properties"] = resource_properties
        resource["location"] = clean_optional_string(resource.get("location")) or "Global"
        self.wait_for_azure_resource_ready(
            credential,
            url,
            f"Communication Services resource {clean_name}",
            emit=emit,
        )
        return resource

    def link_acs_domain_to_communication_service(
        self,
        credential,
        sub_id: str,
        communication_service: dict[str, Any],
        domain_id: str,
        emit: Reporter | None = None,
    ) -> dict[str, Any]:
        clean_domain_id = clean_optional_string(domain_id)
        service_name = clean_optional_string(communication_service.get("name"))
        resource_group = self.extract_resource_group_from_arm_id(clean_optional_string(communication_service.get("id")))
        if not clean_domain_id or not service_name or not resource_group:
            return communication_service
        properties = communication_service.get("properties") if isinstance(communication_service.get("properties"), dict) else {}
        linked_domains = [
            clean_optional_string(domain)
            for domain in properties.get("linkedDomains", [])
            if clean_optional_string(domain)
        ]
        if clean_domain_id not in linked_domains:
            linked_domains.append(clean_domain_id)
        if emit:
            emit(f"Linking ACS email domain to {service_name}...")
        payload_properties = {
            "dataLocation": clean_optional_string(properties.get("dataLocation")) or "United States",
            "linkedDomains": linked_domains,
        }
        for optional_key in ("disableLocalAuth", "publicNetworkAccess"):
            if optional_key in properties:
                payload_properties[optional_key] = properties[optional_key]
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_optional_string(sub_id), safe='')}"
            f"/resourceGroups/{url_quote(resource_group, safe='')}"
            f"/providers/Microsoft.Communication/communicationServices/{url_quote(service_name, safe='')}"
            f"?api-version={COMMUNICATION_API_VERSION}"
        )
        resource = self.retry_azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={
                "location": clean_optional_string(communication_service.get("location")) or "Global",
                "properties": payload_properties,
            },
            action=f"Linking ACS email domain to {service_name}",
            emit=emit,
        )
        resource["id"] = clean_optional_string(resource.get("id")) or clean_optional_string(communication_service.get("id"))
        resource["name"] = clean_optional_string(resource.get("name")) or service_name
        resource_properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
        resource_properties.update(payload_properties)
        resource_properties.setdefault("hostName", clean_optional_string(properties.get("hostName")) or f"{service_name}.communication.azure.com")
        resource["properties"] = resource_properties
        resource["location"] = clean_optional_string(resource.get("location")) or clean_optional_string(communication_service.get("location")) or "Global"
        self.wait_for_azure_resource_ready(
            credential,
            url,
            f"Communication Services resource {service_name}",
            emit=emit,
        )
        return resource

    def ensure_default_acs_communication_service(
        self,
        credential,
        sub_id: str,
        linked_domain_id: str = "",
        emit: Reporter | None = None,
    ) -> dict[str, Any]:
        resources = self.list_acs_communication_services(credential, sub_id)
        if resources:
            selected = resources[0]
            if clean_optional_string(linked_domain_id):
                return self.link_acs_domain_to_communication_service(credential, sub_id, selected, linked_domain_id, emit=emit)
            return selected
        suffix = str(int(time.time()))[-8:]
        service_name = self.sanitize_azure_name(f"log-monitor-acs-{suffix}", max_length=63)
        return self.create_or_update_acs_communication_service(
            credential=credential,
            sub_id=sub_id,
            service_name=service_name,
            resource_group=self.resource_group,
            linked_domain_id=linked_domain_id,
            emit=emit,
        )

    def list_acs_email_services(self, credential, sub_id: str) -> list[dict[str, Any]]:
        clean_sub_id = clean_optional_string(sub_id)
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
            f"/providers/Microsoft.Communication/emailServices?api-version={COMMUNICATION_API_VERSION}"
        )
        return self.azure_management_list_values(credential, url)

    def create_or_update_acs_email_service(
        self,
        credential,
        sub_id: str,
        email_service_name: str,
        resource_group: str,
        emit: Reporter | None = None,
    ) -> dict[str, Any]:
        clean_sub_id = clean_optional_string(sub_id)
        clean_group = self.ensure_resource_group(credential, clean_sub_id, resource_group)
        clean_name = self.sanitize_azure_name(email_service_name, max_length=63)
        if emit:
            emit(f"Creating ACS Email service {clean_name}...")
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
            f"/resourceGroups/{url_quote(clean_group, safe='')}"
            f"/providers/Microsoft.Communication/emailServices/{url_quote(clean_name, safe='')}"
            f"?api-version={COMMUNICATION_API_VERSION}"
        )
        resource = self.azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={"location": "Global", "properties": {"dataLocation": "United States"}},
        )
        resource["id"] = clean_optional_string(resource.get("id")) or (
            f"/subscriptions/{clean_sub_id}/resourceGroups/{clean_group}"
            f"/providers/Microsoft.Communication/emailServices/{clean_name}"
        )
        resource["name"] = clean_optional_string(resource.get("name")) or clean_name
        resource_properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
        resource_properties.setdefault("dataLocation", "United States")
        resource["properties"] = resource_properties
        resource["location"] = clean_optional_string(resource.get("location")) or "Global"
        self.wait_for_azure_resource_ready(
            credential,
            url,
            f"ACS Email service {clean_name}",
            emit=emit,
        )
        return resource

    def create_or_update_acs_email_domain(
        self,
        credential,
        sub_id: str,
        email_service_name: str,
        resource_group: str,
        domain_name: str = "AzureManagedDomain",
        emit: Reporter | None = None,
    ) -> dict[str, Any]:
        clean_sub_id = clean_optional_string(sub_id)
        clean_domain = clean_optional_string(domain_name) or "AzureManagedDomain"
        if emit:
            emit(f"Creating ACS Email domain {clean_domain}...")
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
            f"/resourceGroups/{url_quote(resource_group, safe='')}"
            f"/providers/Microsoft.Communication/emailServices/{url_quote(email_service_name, safe='')}"
            f"/domains/{url_quote(clean_domain, safe='')}"
            f"?api-version={COMMUNICATION_API_VERSION}"
        )
        resource = self.retry_azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={"location": "Global", "properties": {"domainManagement": "AzureManaged"}},
            action=f"Creating ACS Email domain {clean_domain}",
            emit=emit,
        )
        resource["id"] = clean_optional_string(resource.get("id")) or (
            f"/subscriptions/{clean_sub_id}/resourceGroups/{resource_group}"
            f"/providers/Microsoft.Communication/emailServices/{email_service_name}/domains/{clean_domain}"
        )
        resource["name"] = clean_optional_string(resource.get("name")) or clean_domain
        resource_properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
        resource_properties.setdefault("domainManagement", "AzureManaged")
        resource["properties"] = resource_properties
        resource["location"] = clean_optional_string(resource.get("location")) or "Global"
        self.wait_for_azure_resource_ready(
            credential,
            url,
            f"ACS Email domain {clean_domain}",
            emit=emit,
        )
        return resource

    def create_or_update_acs_sender_username(
        self,
        credential,
        sub_id: str,
        email_service_name: str,
        resource_group: str,
        domain_name: str,
        username: str = "DoNotReply",
        display_name: str = "Log Monitor",
        emit: Reporter | None = None,
    ) -> dict[str, Any]:
        clean_username = clean_optional_string(username) or "DoNotReply"
        if emit:
            emit(f"Creating ACS sender {clean_username}...")
        url = (
            f"https://management.azure.com/subscriptions/{url_quote(clean_optional_string(sub_id), safe='')}"
            f"/resourceGroups/{url_quote(resource_group, safe='')}"
            f"/providers/Microsoft.Communication/emailServices/{url_quote(email_service_name, safe='')}"
            f"/domains/{url_quote(domain_name, safe='')}"
            f"/senderUsernames/{url_quote(clean_username, safe='')}"
            f"?api-version={COMMUNICATION_API_VERSION}"
        )
        resource = self.retry_azure_management_json_request(
            credential,
            "PUT",
            url,
            payload={"properties": {"username": clean_username, "displayName": display_name}},
            action=f"Creating ACS sender {clean_username}",
            emit=emit,
        )
        resource["name"] = clean_optional_string(resource.get("name")) or clean_username
        resource_properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
        resource_properties.setdefault("username", clean_username)
        resource_properties.setdefault("displayName", display_name)
        resource["properties"] = resource_properties
        self.wait_for_azure_resource_ready(
            credential,
            url,
            f"ACS sender {clean_username}",
            emit=emit,
        )
        return resource

    def ensure_default_acs_email_sender(self, credential, sub_id: str, emit: Reporter | None = None) -> dict[str, str]:
        suffix = str(int(time.time()))[-8:]
        email_service_name = self.sanitize_azure_name(f"log-monitor-email-{suffix}", max_length=63)
        resource_group = self.resource_group
        email_service = self.create_or_update_acs_email_service(
            credential=credential,
            sub_id=sub_id,
            email_service_name=email_service_name,
            resource_group=resource_group,
            emit=emit,
        )
        email_service_name = clean_optional_string(email_service.get("name")) or email_service_name
        resource_group = self.extract_resource_group_from_arm_id(clean_optional_string(email_service.get("id"))) or resource_group
        domain = self.create_or_update_acs_email_domain(
            credential=credential,
            sub_id=sub_id,
            email_service_name=email_service_name,
            resource_group=resource_group,
            domain_name="AzureManagedDomain",
            emit=emit,
        )
        domain_name = clean_optional_string(domain.get("name")) or "AzureManagedDomain"
        domain_properties = domain.get("properties") if isinstance(domain.get("properties"), dict) else {}
        mail_from_domain = clean_optional_string(domain_properties.get("mailFromSenderDomain")) or domain_name
        sender = self.create_or_update_acs_sender_username(
            credential=credential,
            sub_id=sub_id,
            email_service_name=email_service_name,
            resource_group=resource_group,
            domain_name=domain_name,
            username="DoNotReply",
            display_name="Log Monitor",
            emit=emit,
        )
        sender_properties = sender.get("properties") if isinstance(sender.get("properties"), dict) else {}
        username = clean_optional_string(sender_properties.get("username")) or clean_optional_string(sender.get("name")) or "DoNotReply"
        address = username if "@" in username else f"{username}@{mail_from_domain}"
        domain_id = clean_optional_string(domain.get("id"))
        if domain_id:
            self.ensure_default_acs_communication_service(credential, sub_id, linked_domain_id=domain_id, emit=emit)
        return {
            "address": address,
            "label": address,
            "display_name": clean_optional_string(sender_properties.get("displayName")) or "Log Monitor",
            "email_service": email_service_name,
            "resource_group": resource_group,
            "domain": domain_name,
            "mail_from_domain": mail_from_domain,
            "created": "1",
        }

    def list_acs_email_senders(
        self,
        credential,
        sub_id: str,
        emit: Reporter | None = None,
        limit: int = 200,
        create_if_missing: bool = True,
    ) -> list[dict[str, str]]:
        self.ensure_azure_dependencies()
        clean_sub_id = clean_optional_string(sub_id)
        if not clean_sub_id:
            raise ValueError("Azure Subscription ID is required to load ACS sender addresses.")
        if emit:
            emit("Loading Azure Communication Services email resources...")
        email_services = self.list_acs_email_services(credential, clean_sub_id)
        senders: list[dict[str, str]] = []
        seen_addresses: set[str] = set()

        for email_service in email_services:
            email_service_name = clean_optional_string(email_service.get("name"))
            resource_group = self.extract_resource_group_from_arm_id(clean_optional_string(email_service.get("id")))
            if not email_service_name or not resource_group:
                continue
            if emit:
                emit(f"Loading ACS email domains for {email_service_name}...")
            domains_url = (
                f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
                f"/resourceGroups/{url_quote(resource_group, safe='')}"
                f"/providers/Microsoft.Communication/emailServices/{url_quote(email_service_name, safe='')}"
                f"/domains?api-version={COMMUNICATION_API_VERSION}"
            )
            domains = self.azure_management_list_values(credential, domains_url)
            for domain in domains:
                domain_name = clean_optional_string(domain.get("name"))
                domain_properties = domain.get("properties") if isinstance(domain.get("properties"), dict) else {}
                mail_from_domain = clean_optional_string(domain_properties.get("mailFromSenderDomain")) or domain_name
                if not domain_name or not mail_from_domain:
                    continue
                if emit:
                    emit(f"Loading ACS senders for {mail_from_domain}...")
                sender_url = (
                    f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
                    f"/resourceGroups/{url_quote(resource_group, safe='')}"
                    f"/providers/Microsoft.Communication/emailServices/{url_quote(email_service_name, safe='')}"
                    f"/domains/{url_quote(domain_name, safe='')}"
                    f"/senderUsernames?api-version={COMMUNICATION_API_VERSION}"
                )
                sender_resources = self.azure_management_list_values(credential, sender_url)
                for sender in sender_resources:
                    sender_properties = sender.get("properties") if isinstance(sender.get("properties"), dict) else {}
                    username = clean_optional_string(sender_properties.get("username")) or clean_optional_string(sender.get("name"))
                    if not username:
                        continue
                    address = username if "@" in username else f"{username}@{mail_from_domain}"
                    address_key = address.casefold()
                    if address_key in seen_addresses:
                        continue
                    seen_addresses.add(address_key)
                    display_name = clean_optional_string(sender_properties.get("displayName"))
                    label = address
                    if display_name:
                        label = f"{address} | {display_name}"
                    senders.append(
                        {
                            "address": address,
                            "label": label,
                            "display_name": display_name,
                            "email_service": email_service_name,
                            "resource_group": resource_group,
                            "domain": domain_name,
                            "mail_from_domain": mail_from_domain,
                        }
                    )
                    if len(senders) >= limit:
                        return sorted(senders, key=lambda item: item["address"].casefold())
        if not senders and create_if_missing:
            if emit:
                emit("No ACS email sender was found. Creating a default ACS Email setup...")
            senders.append(self.ensure_default_acs_email_sender(credential, clean_sub_id, emit=emit))
        return sorted(senders, key=lambda item: item["address"].casefold())

    def build_acs_connection_string(self, host_name: str, access_key: str) -> str:
        clean_host = clean_optional_string(host_name)
        clean_key = clean_optional_string(access_key)
        if not clean_host or not clean_key:
            return ""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", clean_host):
            clean_host = f"https://{clean_host}"
        clean_host = clean_host.rstrip("/") + "/"
        return f"endpoint={clean_host};accesskey={clean_key}"

    def list_acs_connection_strings(
        self,
        credential,
        sub_id: str,
        emit: Reporter | None = None,
        limit: int = 200,
        create_if_missing: bool = True,
    ) -> list[dict[str, str]]:
        self.ensure_azure_dependencies()
        clean_sub_id = clean_optional_string(sub_id)
        if not clean_sub_id:
            raise ValueError("Azure Subscription ID is required to load ACS connection strings.")
        if emit:
            emit("Loading Azure Communication Services resources...")
        resources = self.list_acs_communication_services(credential, clean_sub_id)
        if not resources and create_if_missing:
            if emit:
                emit("No Azure Communication Services resource was found. Creating one automatically...")
            resources = [self.ensure_default_acs_communication_service(credential, clean_sub_id, emit=emit)]
        connections: list[dict[str, str]] = []

        for resource in resources:
            service_name = clean_optional_string(resource.get("name"))
            resource_group = self.extract_resource_group_from_arm_id(clean_optional_string(resource.get("id")))
            properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
            host_name = clean_optional_string(properties.get("hostName")) or f"{service_name}.communication.azure.com"
            if not service_name or not resource_group:
                continue
            if emit:
                emit(f"Fetching ACS keys for {service_name}...")
            keys_url = (
                f"https://management.azure.com/subscriptions/{url_quote(clean_sub_id, safe='')}"
                f"/resourceGroups/{url_quote(resource_group, safe='')}"
                f"/providers/Microsoft.Communication/communicationServices/{url_quote(service_name, safe='')}"
                f"/listKeys?api-version={COMMUNICATION_API_VERSION}"
            )
            keys = self.retry_azure_management_json_request(
                credential,
                "POST",
                keys_url,
                action=f"Fetching ACS keys for {service_name}",
                emit=emit,
            )
            key_options = [
                ("Primary", "primaryConnectionString", "primaryKey"),
                ("Secondary", "secondaryConnectionString", "secondaryKey"),
            ]
            for key_label, connection_key, access_key_name in key_options:
                connection_string = clean_optional_string(keys.get(connection_key)) if isinstance(keys, dict) else ""
                if not connection_string:
                    access_key = clean_optional_string(keys.get(access_key_name)) if isinstance(keys, dict) else ""
                    connection_string = self.build_acs_connection_string(host_name, access_key)
                if not connection_string:
                    continue
                label = f"{service_name} | {key_label}"
                if resource_group:
                    label = f"{resource_group}/{label}"
                connections.append(
                    {
                        "label": label,
                        "connection_string": connection_string,
                        "service_name": service_name,
                        "resource_group": resource_group,
                        "key_name": key_label.lower(),
                    }
                )
                if len(connections) >= limit:
                    return sorted(connections, key=lambda item: item["label"].casefold())
        return sorted(connections, key=lambda item: item["label"].casefold())

    def get_azure_workspace_location(self, ml_client) -> str:
        try:
            workspace = ml_client.workspaces.get(self.workspace_name)
            return clean_optional_string(getattr(workspace, "location", "")) or "eastus"
        except Exception:
            return "eastus"

    def get_arm_provisioning_state(self, resource: dict) -> str:
        properties = resource.get("properties") if isinstance(resource, dict) else {}
        if not isinstance(properties, dict):
            properties = {}
        return clean_optional_string(
            properties.get("provisioningState")
            or properties.get("provisioning_state")
            or resource.get("provisioningState")
            or resource.get("provisioning_state")
        )

    def wait_for_arm_serverless_endpoint_ready(
        self,
        credential,
        sub_id: str,
        endpoint_name: str,
        *,
        api_version: str = "2024-04-01-preview",
        emit: Reporter | None = None,
        timeout_seconds: int = 1800,
        poll_interval_seconds: int = 15,
    ) -> dict:
        url = self.build_serverless_endpoint_management_url(sub_id, endpoint_name, api_version=api_version)
        deadline = time.time() + max(int(timeout_seconds), 1)
        interval = max(int(poll_interval_seconds), 1)
        last_state = ""
        last_resource: dict = {}
        while time.time() < deadline:
            resource = self.azure_management_json_request(credential, "GET", url)
            if isinstance(resource, dict):
                last_resource = resource
                state = self.get_arm_provisioning_state(resource)
                if state:
                    last_state = state
                state_lower = state.lower()
                if state_lower == "succeeded":
                    return resource
                if state_lower in {"failed", "canceled"}:
                    raise RuntimeError(f"Azure serverless endpoint '{endpoint_name}' ARM provisioning {state_lower}.")
            if emit:
                emit(f"Waiting for Azure serverless endpoint ARM resource: {last_state or 'pending'}")
            time.sleep(interval)
        raise TimeoutError(
            f"Timed out waiting for Azure serverless endpoint ARM resource '{endpoint_name}' to become ready."
            + (f" Last state: {last_state}." if last_state else "")
            + (f" Last response: {json.dumps(last_resource)[:1000]}" if last_resource else "")
        )

    def summarize_arm_serverless_endpoint_resource(self, resource: dict) -> dict[str, Any]:
        if not isinstance(resource, dict):
            return {}
        properties = resource.get("properties") if isinstance(resource.get("properties"), dict) else {}
        model_settings = properties.get("modelSettings") if isinstance(properties.get("modelSettings"), dict) else {}
        inference_endpoint = properties.get("inferenceEndpoint") if isinstance(properties.get("inferenceEndpoint"), dict) else {}
        sku = resource.get("sku") if isinstance(resource.get("sku"), dict) else {}
        return {
            "id": clean_optional_string(resource.get("id")),
            "name": clean_optional_string(resource.get("name")),
            "type": clean_optional_string(resource.get("type")),
            "location": clean_optional_string(resource.get("location")),
            "kind": clean_optional_string(resource.get("kind")),
            "sku_name": clean_optional_string(sku.get("name")),
            "provisioning_state": self.get_arm_provisioning_state(resource),
            "auth_mode": clean_optional_string(properties.get("authMode") or properties.get("auth_mode")),
            "model_id": clean_optional_string(model_settings.get("modelId")),
            "target_uri": clean_optional_string(
                inference_endpoint.get("targetUri")
                or inference_endpoint.get("target_uri")
                or inference_endpoint.get("uri")
                or properties.get("scoringUri")
                or properties.get("scoring_uri")
            ),
        }

    def list_arm_serverless_endpoint_resources(
        self,
        credential,
        sub_id: str,
        api_version: str = "2024-04-01-preview",
    ) -> list[dict[str, Any]]:
        url = self.build_serverless_endpoint_collection_management_url(sub_id, api_version=api_version)
        response = self.azure_management_json_request(credential, "GET", url)
        raw_resources = response.get("value", []) if isinstance(response, dict) else []
        resources: list[dict[str, Any]] = []
        for resource in raw_resources:
            if isinstance(resource, dict):
                resources.append(self.summarize_arm_serverless_endpoint_resource(resource))
        return resources

    def create_arm_serverless_endpoint(
        self,
        credential,
        sub_id: str,
        endpoint_name: str,
        model_id: str,
        location: str,
        auth_mode: str = "Key",
        api_version: str = "2024-04-01-preview",
        emit: Reporter | None = None,
    ) -> dict:
        clean_endpoint_name = self.sanitize_azure_endpoint_name(endpoint_name)
        clean_model_id = self.normalize_serverless_model_id(model_id)
        clean_location = clean_optional_string(location) or "eastus"
        clean_auth_mode = clean_optional_string(auth_mode) or "Key"
        payload = {
            "location": clean_location,
            "kind": "ServerlessEndpoint",
            "sku": {"name": "Consumption"},
            "properties": {
                "authMode": clean_auth_mode,
                "modelSettings": {
                    "modelId": clean_model_id,
                },
            },
        }
        if emit:
            emit("Creating Azure serverless endpoint ARM resource...")
        self.azure_management_json_request(
            credential,
            "PUT",
            self.build_serverless_endpoint_management_url(sub_id, clean_endpoint_name, api_version=api_version),
            payload=payload,
            expected_statuses=(200, 201, 202),
        )
        return self.wait_for_arm_serverless_endpoint_ready(
            credential,
            sub_id,
            clean_endpoint_name,
            api_version=api_version,
            emit=emit,
            timeout_seconds=1800,
            poll_interval_seconds=15,
        )

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
        required_files = ["function_app.py", "host.json", "requirements.txt"]
        missing_files = [name for name in required_files if not (bridge_root / name).is_file()]
        if missing_files:
            raise RuntimeError(
                "The Azure Function bridge package is missing required files: " + ", ".join(missing_files)
            )
        package_dir = Path(tempfile.gettempdir()) / self.sanitize_azure_name(package_name, max_length=60)
        package_dir.mkdir(parents=True, exist_ok=True)
        package_path = package_dir / "released-package.zip"
        if package_path.exists():
            package_path.unlink()
        extra_root_files = [
            self.project_dir / "train.py",
            self.project_dir / "mlops_utils.py",
            self.project_dir / "requirements.train.txt",
        ]
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in bridge_root.rglob("*"):
                if file_path.is_file() and "__pycache__" not in file_path.parts:
                    archive.write(file_path, arcname=str(file_path.relative_to(bridge_root)))
            for file_path in extra_root_files:
                if file_path.exists() and file_path.is_file():
                    archive.write(file_path, arcname=file_path.name)
        return str(package_path)

    def upload_blob_file(
        self,
        storage_connection_string: str,
        container_name: str,
        source_path: str,
        blob_name: str,
    ) -> str:
        from azure.storage.blob import BlobServiceClient

        resolved_source = Path(source_path).expanduser().resolve()
        if not resolved_source.exists():
            raise FileNotFoundError(f"Feedback base dataset does not exist: {resolved_source}")
        blob_service = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service.get_container_client(container_name)
        with open(resolved_source, "rb") as handle:
            container_client.upload_blob(name=blob_name, data=handle, overwrite=True)
        return blob_name

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
        package_blob_name = f"releases/{Path(package_path).parent.name}/released-package.zip"
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
            payload={"properties": {"packageUri": package_uri, "remoteBuild": True}},
            expected_statuses=(200, 202),
        )

    def list_function_app_function_names(self, credential, sub_id: str, function_app_name: str) -> list[str]:
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/functions?api-version=2025-03-01"
        )
        payload = self.azure_management_json_request(credential, "GET", url, expected_statuses=(200,))
        raw_values = payload.get("value", []) if isinstance(payload, dict) else []
        names: list[str] = []
        for item in raw_values:
            if not isinstance(item, dict):
                continue
            raw_name = clean_optional_string(item.get("name"))
            if not raw_name:
                continue
            names.append(raw_name.split("/")[-1])
        return names

    def sync_function_app_triggers(self, credential, sub_id: str, function_app_name: str) -> None:
        url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/host/default/sync?api-version=2025-03-01"
        )
        self.azure_management_json_request(credential, "POST", url, payload={}, expected_statuses=(200, 202, 204))

    def update_online_deployment_environment_variables(
        self,
        ml_client,
        endpoint_name: str,
        deployment_name: str,
        variables: dict[str, str],
        emit: Reporter | None = None,
    ) -> dict[str, str]:
        clean_variables = {
            clean_optional_string(key): clean_optional_string(value)
            for key, value in (variables or {}).items()
            if clean_optional_string(key) and clean_optional_string(value)
        }
        if not clean_variables:
            return {}
        deployment = ml_client.online_deployments.get(name=deployment_name, endpoint_name=endpoint_name)
        existing_variables = getattr(deployment, "environment_variables", None)
        if not isinstance(existing_variables, dict):
            existing_variables = {}
        merged_variables = {**existing_variables, **clean_variables}
        setattr(deployment, "environment_variables", merged_variables)
        if emit:
            emit("Updating Azure online deployment environment variables...")
        self.wait_for_azure_poller(
            ml_client.online_deployments.begin_create_or_update(deployment),
            action="Updating Azure online deployment environment variables",
            emit=emit,
            timeout_seconds=1800,
        )
        self.wait_for_online_deployment_ready(
            ml_client,
            endpoint_name,
            deployment_name,
            action="Waiting for Azure online deployment environment update",
            emit=emit,
            timeout_seconds=1800,
        )
        return merged_variables

    def wait_for_function_bridge_endpoint(
        self,
        credential,
        sub_id: str,
        function_app_name: str,
        function_host_name: str,
        function_name: str = "ingest_log",
        route_path: str = "logs",
        timeout_seconds: int = 900,
        poll_interval_seconds: int = 10,
    ) -> tuple[str, str]:
        host_keys_url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/host/default/listkeys?api-version=2025-03-01"
        )
        function_secrets_url = (
            f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Web/sites/{function_app_name}/functions/{function_name}/listsecrets?api-version=2025-05-01"
        )
        deadline = time.time() + max(int(timeout_seconds), 1)
        poll_interval = max(int(poll_interval_seconds), 1)
        last_error = ""
        last_function_names: list[str] = []
        sync_attempted = False
        while time.time() < deadline:
            try:
                last_function_names = self.list_function_app_function_names(credential, sub_id, function_app_name)
            except Exception as exc:
                last_error = str(exc)
            if function_name in last_function_names:
                try:
                    payload = self.azure_management_json_request(
                        credential,
                        "POST",
                        function_secrets_url,
                        payload={},
                        expected_statuses=(200,),
                    )
                    trigger_url = clean_optional_string(payload.get("trigger_url"))
                    function_key = clean_optional_string(payload.get("key"))
                    if trigger_url:
                        return trigger_url, function_key
                    if function_key:
                        return f"https://{function_host_name}/api/{route_path.strip('/')}?code={function_key}", function_key
                except Exception as exc:
                    last_error = str(exc)
                try:
                    payload = self.azure_management_json_request(credential, "POST", host_keys_url, payload={}, expected_statuses=(200,))
                    function_keys = payload.get("functionKeys") if isinstance(payload.get("functionKeys"), dict) else {}
                    function_key = clean_optional_string(function_keys.get("default"))
                    if not function_key and function_keys:
                        function_key = clean_optional_string(next(iter(function_keys.values()), ""))
                    if function_key:
                        return f"https://{function_host_name}/api/{route_path.strip('/')}?code={function_key}", function_key
                except Exception as exc:
                    last_error = str(exc)
            elif not sync_attempted:
                try:
                    self.sync_function_app_triggers(credential, sub_id, function_app_name)
                except Exception as exc:
                    last_error = str(exc)
                sync_attempted = True
            time.sleep(poll_interval)
        visible_functions = ", ".join(last_function_names) if last_function_names else "none"
        raise RuntimeError(
            f"The Azure Function API package was deployed, but Azure did not index `{function_name}` in time.\n"
            f"Visible functions: {visible_functions}.\n\n"
            "Open the Function App > Log stream or Diagnose and solve problems > Flex Consumption Deployment "
            "to see Python import/build errors from the deployment.\n\n"
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

    def build_azure_model_label(self, model_info: dict[str, Any]) -> str:
        name = clean_optional_string(model_info.get("name"))
        version = clean_optional_string(model_info.get("version"))
        asset_type = clean_optional_string(model_info.get("type"))
        created_at = clean_optional_string(model_info.get("created_at"))
        label = name
        if version:
            label += f":{version}"
        if asset_type:
            label += f" | {asset_type}"
        if created_at:
            label += f" | {created_at.replace('T', ' ')[:19]}"
        return label or clean_optional_string(model_info.get("id")) or "Azure model"

    def parse_azure_model_name_version_from_id(self, raw_value: str) -> tuple[str, str]:
        clean_value = clean_optional_string(raw_value)
        if not clean_value:
            return "", ""
        if clean_value.startswith("azureml:") and not clean_value.startswith("azureml://"):
            parts = clean_value.split(":")
            if len(parts) >= 3:
                return clean_optional_string(parts[-2]), clean_optional_string(parts[-1])

        match = re.search(r"/models/([^/?#]+)/versions/([^/?#]+)", clean_value, flags=re.IGNORECASE)
        if match:
            return url_unquote(match.group(1)), url_unquote(match.group(2))

        match = re.search(r"models/([^/?#]+)/versions/([^/?#]+)", clean_value, flags=re.IGNORECASE)
        if match:
            return url_unquote(match.group(1)), url_unquote(match.group(2))
        return "", ""

    def build_azure_model_payload_from_entity(self, model: Any) -> dict[str, Any]:
        name = clean_optional_string(getattr(model, "name", ""))
        version = clean_optional_string(getattr(model, "version", ""))
        model_id = clean_optional_string(getattr(model, "id", ""))
        path = clean_optional_string(getattr(model, "path", ""))
        if not (name and version):
            parsed_name, parsed_version = self.parse_azure_model_name_version_from_id(model_id or path)
            name = name or parsed_name
            version = version or parsed_version
        creation_context = getattr(model, "creation_context", None)
        payload = {
            "id": model_id,
            "name": name,
            "version": version,
            "path": path,
            "type": clean_optional_string(getattr(model, "type", "")),
            "description": clean_optional_string(getattr(model, "description", "")),
            "created_at": clean_optional_string(getattr(creation_context, "created_at", "")) if creation_context is not None else "",
        }
        payload["label"] = self.build_azure_model_label(payload)
        return payload

    def list_azure_workspace_models(self, ml_client, limit: int = 200) -> list[dict[str, Any]]:
        self.ensure_azure_dependencies()
        models: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_model_payload(payload: dict[str, Any]) -> bool:
            name = clean_optional_string(payload.get("name"))
            version = clean_optional_string(payload.get("version"))
            if not name or not version:
                return False
            model_id = clean_optional_string(payload.get("id"))
            key = model_id or f"{name}:{version}"
            if key in seen:
                return False
            seen.add(key)
            models.append(payload)
            return len(models) >= limit

        try:
            iterable = ml_client.models.list()
        except TypeError:
            iterable = ml_client.models.list(name=None)
        for model in iterable:
            payload = self.build_azure_model_payload_from_entity(model)
            name = clean_optional_string(payload.get("name"))
            version = clean_optional_string(payload.get("version"))
            if name and not version:
                try:
                    versioned_iterable = ml_client.models.list(name=name)
                except Exception:
                    versioned_iterable = []
                for versioned_model in versioned_iterable:
                    if add_model_payload(self.build_azure_model_payload_from_entity(versioned_model)):
                        break
            else:
                add_model_payload(payload)
            if len(models) >= limit:
                break
        return sorted(
            models,
            key=lambda item: (
                clean_optional_string(item.get("name")),
                clean_optional_string(item.get("version")),
                clean_optional_string(item.get("id")),
            ),
            reverse=True,
        )

    def resolve_azure_model_asset(self, ml_client, *, model_id: str = "", model_name: str = "", model_version: str = ""):
        self.ensure_azure_dependencies()
        clean_id = clean_optional_string(model_id)
        clean_name = clean_optional_string(model_name)
        clean_version = clean_optional_string(model_version)

        if not (clean_name and clean_version):
            parsed_name, parsed_version = self.parse_azure_model_name_version_from_id(clean_id)
            clean_name = clean_name or parsed_name
            clean_version = clean_version or parsed_version

        if clean_name and clean_version:
            return ml_client.models.get(name=clean_name, version=clean_version)

        if clean_id:
            for model in self.list_azure_workspace_models(ml_client):
                if clean_optional_string(model.get("id")) == clean_id:
                    name = clean_optional_string(model.get("name"))
                    version = clean_optional_string(model.get("version"))
                    if name and version:
                        return ml_client.models.get(name=name, version=version)

        raise ValueError("Select an Azure ML registered model version before hosting.")

    def get_online_endpoint_key(self, ml_client, endpoint_name: str, emit: Reporter | None = None) -> str:
        self.ensure_azure_dependencies()
        clean_endpoint_name = clean_optional_string(endpoint_name)
        if not clean_endpoint_name:
            return ""
        if emit:
            emit("Fetching Azure online endpoint key...")
        try:
            credentials = ml_client.online_endpoints.get_keys(name=clean_endpoint_name)
        except TypeError:
            credentials = ml_client.online_endpoints.get_keys(clean_endpoint_name)

        if isinstance(credentials, dict):
            for key in ("primary_key", "primaryKey", "key1", "primary", "value"):
                value = clean_optional_string(credentials.get(key))
                if value:
                    return value
        for attr in ("primary_key", "primaryKey", "key1", "primary", "value"):
            value = clean_optional_string(getattr(credentials, attr, ""))
            if value:
                return value
        as_dict = getattr(credentials, "as_dict", None)
        if callable(as_dict):
            payload = as_dict()
            if isinstance(payload, dict):
                for key in ("primary_key", "primaryKey", "key1", "primary", "value"):
                    value = clean_optional_string(payload.get(key))
                    if value:
                        return value
        raise RuntimeError(
            f"Azure returned no access key for online endpoint '{clean_endpoint_name}'. "
            "Confirm the endpoint uses key authentication or fetch credentials from Azure ML Studio."
        )

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
        azure_model_id: str = "",
        azure_model_name: str = "",
        azure_model_version: str = "",
        emit: Reporter | None = None,
    ) -> dict:
        self.ensure_azure_dependencies()
        attempted_instance_types: list[str] = []
        instance_candidates = self.prioritize_instance_candidates(
            self.get_azure_host_instance_candidates(azure_compute),
            preferred_instance_type,
        )
        if clean_optional_string(azure_model_id) or clean_optional_string(azure_model_name):
            if emit:
                emit("Using selected Azure ML registered model...")
            registered_model = self.resolve_azure_model_asset(
                ml_client,
                model_id=azure_model_id,
                model_name=azure_model_name,
                model_version=azure_model_version,
            )
        else:
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
        self.wait_for_azure_poller(
            ml_client.online_endpoints.begin_create_or_update(endpoint),
            action="Creating Azure online endpoint",
            emit=emit,
            timeout_seconds=1800,
        )
        endpoint_details = self.wait_for_online_endpoint_ready(
            ml_client,
            endpoint_name,
            action="Waiting for Azure online endpoint to finish creating",
            emit=emit,
            timeout_seconds=1800,
        )
        if endpoint_details is None:
            raise RuntimeError(f"Azure endpoint '{endpoint_name}' was not found after create/update completed.")
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
                    code_configuration=CodeConfiguration(code=str(self.project_dir), scoring_script="azure_score.py"),
                    instance_type=instance_type,
                    instance_count=1,
                )
                self.wait_for_azure_poller(
                    ml_client.online_deployments.begin_create_or_update(deployment),
                    action=f"Deploying model to Azure endpoint ({instance_type})",
                    emit=emit,
                    timeout_seconds=3600,
                )
                self.wait_for_online_deployment_ready(
                    ml_client,
                    endpoint_name,
                    deployment_name,
                    action=f"Waiting for Azure deployment ({instance_type})",
                    emit=emit,
                    timeout_seconds=3600,
                )
                selected_instance_type = instance_type
                break
            except Exception as exc:
                last_error = exc
                if self.is_azure_quota_error(exc) and instance_type != instance_candidates[-1]:
                    if emit:
                        emit(f"Azure quota was not available for {instance_type}; trying the next VM size...")
                    continue
                if self.is_azure_quota_error(exc):
                    raise RuntimeError(self.format_azure_hosting_error(exc, attempted_instance_types)) from exc
                raise
        if not selected_instance_type:
            if last_error is not None:
                if self.is_azure_quota_error(last_error):
                    raise RuntimeError(self.format_azure_hosting_error(last_error, attempted_instance_types)) from last_error
                raise last_error
            raise RuntimeError("Azure deployment did not complete and no scoring instance was selected.")
        endpoint = ml_client.online_endpoints.get(endpoint_name) or ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode=endpoint_auth_mode,
            description="Hosted prediction API for Log Monitor",
        )
        if not clean_optional_string(getattr(endpoint, "auth_mode", "")):
            endpoint.auth_mode = endpoint_auth_mode
        endpoint.traffic = {deployment_name: 100}
        self.wait_for_azure_poller(
            ml_client.online_endpoints.begin_create_or_update(endpoint),
            action="Routing Azure online endpoint traffic",
            emit=emit,
            timeout_seconds=900,
        )
        endpoint_details = self.wait_for_online_endpoint_ready(
            ml_client,
            endpoint_name,
            action="Waiting for Azure online endpoint traffic routing",
            emit=emit,
            timeout_seconds=900,
        )
        scoring_uri = clean_optional_string(getattr(endpoint_details, "scoring_uri", ""))
        if not scoring_uri:
            raise RuntimeError("Azure deployment completed but no scoring URI was returned.")
        return {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "instance_type": selected_instance_type,
            "api_url": scoring_uri,
            "azure_model_id": clean_optional_string(getattr(registered_model, "id", "")) or clean_optional_string(azure_model_id),
            "azure_model_name": clean_optional_string(getattr(registered_model, "name", "")) or clean_optional_string(azure_model_name),
            "azure_model_version": clean_optional_string(getattr(registered_model, "version", "")) or clean_optional_string(azure_model_version),
            "attempted_instance_types": attempted_instance_types,
        }

    def extract_serverless_scoring_uri(self, endpoint: Any) -> str:
        scoring_uri = clean_optional_string(getattr(endpoint, "scoring_uri", ""))
        if scoring_uri:
            return scoring_uri
        inference_endpoint = getattr(endpoint, "inference_endpoint", None)
        if inference_endpoint is None:
            inference_endpoint = getattr(endpoint, "inferenceEndpoint", None)
        if isinstance(inference_endpoint, dict):
            for key in ("uri", "target_uri", "targetUri", "scoring_uri", "scoringUri"):
                scoring_uri = clean_optional_string(inference_endpoint.get(key))
                if scoring_uri:
                    return scoring_uri
        if inference_endpoint is not None:
            for attr in ("uri", "target_uri", "targetUri", "scoring_uri", "scoringUri"):
                scoring_uri = clean_optional_string(getattr(inference_endpoint, attr, ""))
                if scoring_uri:
                    return scoring_uri
        return ""

    def summarize_serverless_endpoint(self, endpoint: Any) -> dict[str, str]:
        return {
            "name": clean_optional_string(getattr(endpoint, "name", "")),
            "model_id": clean_optional_string(getattr(endpoint, "model_id", "")),
            "api_url": self.extract_serverless_scoring_uri(endpoint),
            "auth_mode": clean_optional_string(getattr(endpoint, "auth_mode", "")),
            "provisioning_state": clean_optional_string(getattr(endpoint, "provisioning_state", "")),
        }

    def list_azure_serverless_endpoints(self, ml_client) -> list[dict[str, str]]:
        self.ensure_azure_dependencies()
        endpoints: list[dict[str, str]] = []
        for endpoint in ml_client.serverless_endpoints.list():
            summary = self.summarize_serverless_endpoint(endpoint)
            if summary["name"]:
                endpoints.append(summary)
        return endpoints

    def deploy_azure_serverless_endpoint(
        self,
        ml_client,
        model_id: str,
        endpoint_name: str,
        endpoint_auth_mode: str = "key",
        credential=None,
        sub_id: str = "",
        emit: Reporter | None = None,
    ) -> dict:
        self.ensure_azure_dependencies()
        clean_model_id = self.normalize_serverless_model_id(model_id)
        if not clean_model_id:
            raise ValueError("Azure serverless hosting needs a model ID from the Azure ML model catalog.")
        clean_endpoint_name = self.sanitize_azure_endpoint_name(endpoint_name)
        arm_resource: dict[str, Any] = {}
        arm_creation_error = ""
        arm_api_version = "2024-04-01-preview"
        creation_method = f"arm-{arm_api_version}"
        if clean_model_id != clean_optional_string(model_id).strip() and emit:
            emit("Removed the version suffix from the serverless model ID; Azure deploys the latest catalog version.")

        if credential is not None and clean_optional_string(sub_id):
            try:
                arm_resource = self.summarize_arm_serverless_endpoint_resource(
                    self.create_arm_serverless_endpoint(
                        credential=credential,
                        sub_id=sub_id,
                        endpoint_name=clean_endpoint_name,
                        model_id=clean_model_id,
                        location=self.get_azure_workspace_location(ml_client),
                        auth_mode="Key" if endpoint_auth_mode.lower() == "key" else endpoint_auth_mode,
                        api_version=arm_api_version,
                        emit=emit,
                    )
                )
            except Exception as exc:
                arm_creation_error = clean_optional_string(exc)
                if emit:
                    emit(f"ARM serverless endpoint creation failed; falling back to Azure ML SDK. {arm_creation_error}")

        if not arm_resource:
            creation_method = "sdk-serverless-endpoints"
            if ServerlessEndpoint is None:
                raise RuntimeError("This Python environment needs a newer `azure-ai-ml` package to create serverless endpoints.")
            if emit:
                emit("Creating Azure serverless endpoint with Azure ML SDK...")
            endpoint = ServerlessEndpoint(
                name=clean_endpoint_name,
                model_id=clean_model_id,
                auth_mode=endpoint_auth_mode,
            )
            self.wait_for_azure_poller(
                ml_client.serverless_endpoints.begin_create_or_update(endpoint),
                action="Creating Azure serverless endpoint",
                emit=emit,
                timeout_seconds=1800,
                poll_interval_seconds=15,
            )
        endpoint_details = self.wait_for_serverless_endpoint_ready(
            ml_client,
            clean_endpoint_name,
            action="Waiting for Azure serverless endpoint to finish creating",
            emit=emit,
            timeout_seconds=1800,
            poll_interval_seconds=15,
        )
        scoring_uri = self.extract_serverless_scoring_uri(endpoint_details)
        if not scoring_uri:
            raise RuntimeError("Azure serverless endpoint completed but no target URI was returned.")
        workspace_serverless_endpoints: list[dict[str, str]] = []
        workspace_serverless_endpoint_names: list[str] = []
        serverless_list_error = ""
        visible_in_workspace_list = False
        arm_serverless_endpoint_names: list[str] = []
        arm_serverless_endpoints: list[dict[str, Any]] = []
        arm_list_error = ""
        visible_in_arm_resource_list = False
        if emit:
            emit("Verifying serverless endpoint in the workspace list...")
        try:
            workspace_serverless_endpoints = self.list_azure_serverless_endpoints(ml_client)
            workspace_serverless_endpoint_names = [
                clean_optional_string(endpoint.get("name"))
                for endpoint in workspace_serverless_endpoints
                if clean_optional_string(endpoint.get("name"))
            ]
            visible_in_workspace_list = clean_endpoint_name in workspace_serverless_endpoint_names
        except Exception as exc:
            serverless_list_error = clean_optional_string(exc)
        if credential is not None and clean_optional_string(sub_id):
            try:
                arm_serverless_endpoints = self.list_arm_serverless_endpoint_resources(credential, sub_id)
                arm_serverless_endpoint_names = [
                    clean_optional_string(endpoint.get("name"))
                    for endpoint in arm_serverless_endpoints
                    if clean_optional_string(endpoint.get("name"))
                ]
                visible_in_arm_resource_list = clean_endpoint_name in arm_serverless_endpoint_names
            except Exception as exc:
                arm_list_error = clean_optional_string(exc)
        return {
            "endpoint_name": clean_endpoint_name,
            "model_id": clean_model_id,
            "endpoint_auth_mode": clean_optional_string(getattr(endpoint_details, "auth_mode", "")) or endpoint_auth_mode,
            "api_url": scoring_uri,
            "provisioning_state": clean_optional_string(getattr(endpoint_details, "provisioning_state", "")),
            "creation_method": creation_method,
            "arm_api_version": arm_api_version,
            "arm_creation_error": arm_creation_error,
            "arm_resource": arm_resource,
            "visible_in_arm_resource_list": visible_in_arm_resource_list,
            "arm_list_error": arm_list_error,
            "arm_serverless_endpoint_names": arm_serverless_endpoint_names,
            "arm_serverless_endpoints": arm_serverless_endpoints,
            "visible_in_workspace_list": visible_in_workspace_list,
            "serverless_list_error": serverless_list_error,
            "workspace_serverless_endpoint_names": workspace_serverless_endpoint_names,
            "workspace_serverless_endpoints": workspace_serverless_endpoints,
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
        azure_model_id: str = "",
        azure_model_name: str = "",
        azure_model_version: str = "",
        emit: Reporter | None = None,
    ) -> dict:
        self.ensure_azure_dependencies()
        attempted_instance_types: list[str] = []
        deployment_name = "default"
        compute_name = "log-monitor-batch-gpu" if clean_optional_string(azure_compute).lower() == "gpu" else "log-monitor-batch-cpu"
        instance_candidates = self.prioritize_instance_candidates(self.get_azure_host_instance_candidates(azure_compute), preferred_instance_type)
        selected_instance_type = ""

        if clean_optional_string(azure_model_id) or clean_optional_string(azure_model_name):
            if emit:
                emit("Using selected Azure ML registered model...")
            registered_model = self.resolve_azure_model_asset(
                ml_client,
                model_id=azure_model_id,
                model_name=azure_model_name,
                model_version=azure_model_version,
            )
        else:
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
                    if emit:
                        emit(f"Azure quota was not available for {instance_type}; trying the next VM size...")
                    continue
                if self.is_azure_quota_error(exc):
                    raise RuntimeError(self.format_azure_hosting_error(exc, attempted_instance_types)) from exc
                raise
        if not selected_instance_type:
            if last_deployment_error is not None:
                if self.is_azure_quota_error(last_deployment_error):
                    raise RuntimeError(self.format_azure_hosting_error(last_deployment_error, attempted_instance_types)) from last_deployment_error
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
            code_configuration=CodeConfiguration(code=str(self.project_dir), scoring_script="azure_batch_score.py"),
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
            "azure_model_id": clean_optional_string(getattr(registered_model, "id", "")) or clean_optional_string(azure_model_id),
            "azure_model_name": clean_optional_string(getattr(registered_model, "name", "")) or clean_optional_string(azure_model_name),
            "azure_model_version": clean_optional_string(getattr(registered_model, "version", "")) or clean_optional_string(azure_model_version),
            "attempted_instance_types": attempted_instance_types,
        }
