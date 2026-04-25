from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from mlops_utils import (
    clean_optional_string,
    now_utc_iso,
)

from .azure_platform_service import AZURE_AVAILABLE, AzurePlatformService
from .contracts import HostingRequest
from .github_service import GitHubService
from .mlops_service import MlopsService
from .model_catalog_service import ModelCatalogService
from .observability_service import ObservabilityService
from .runtime import JobContext, JobManager


class HostingService:
    def __init__(
        self,
        project_dir: str,
        job_manager: JobManager,
        model_catalog_service: ModelCatalogService,
        mlops_service: MlopsService,
        azure_platform_service: AzurePlatformService,
        observability_service: ObservabilityService,
        github_service: GitHubService | None = None,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.job_manager = job_manager
        self.model_catalog_service = model_catalog_service
        self.mlops_service = mlops_service
        self.azure_platform_service = azure_platform_service
        self.observability_service = observability_service
        self.github_service = github_service or GitHubService()

    def submit_hosting(self, request: HostingRequest):
        return self.job_manager.submit(
            "hosting",
            lambda ctx: self._run(ctx, request),
            metadata={"operation": "hosting", "mode": request.mode, "model_dir": request.model_dir},
        )

    def cancel(self, job_id: str) -> bool:
        return self.job_manager.cancel_job(job_id)

    def stop_local_stack(self) -> None:
        self.observability_service.shutdown_local_hosting_stack()

    def _run(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if request.mode == "azure":
            result = self._run_azure_hosting(ctx, request)
        else:
            result = self._run_local_hosting(ctx, request)
        if request.create_github_pr:
            self._attach_github_copilot_pr_task(ctx, request, result)
        return result

    def _attach_github_copilot_pr_task(self, ctx: JobContext, request: HostingRequest, result: dict[str, Any]) -> None:
        def persist_pr_metadata(extra: dict[str, Any]) -> None:
            try:
                hosting_meta = self.model_catalog_service.read_last_hosting_metadata() or {}
                hosting_meta.update(extra)
                result["metadata_path"] = self.model_catalog_service.save_last_hosting_metadata(hosting_meta)
            except Exception:
                pass

        endpoint_url = clean_optional_string(result.get("api_url"))
        if not endpoint_url:
            result["github_pr_error"] = "Could not create the Copilot PR task because no endpoint URL was returned."
            result["summary"] = clean_optional_string(result.get("summary")) + f"\nCopilot PR task: {result['github_pr_error']}"
            persist_pr_metadata({"github_pr_error": result["github_pr_error"]})
            return
        ctx.emit("progress", "Creating GitHub Copilot PR task...")
        prompt_info: dict[str, Any] = {}
        try:
            prompt_text = self.github_service.build_log_forwarding_copilot_prompt(
                repo_name=request.github_repo,
                base_branch=request.github_branch,
                endpoint_url=endpoint_url,
                endpoint_name=clean_optional_string(result.get("endpoint_name")),
                endpoint_auth_mode=clean_optional_string(result.get("endpoint_auth_mode")),
                service_kind=clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                hosting_mode=request.mode,
            )
            prompt_info = self.mlops_service.archive_copilot_pr_prompt(
                prompt_text,
                {
                    "repo_name": request.github_repo,
                    "base_branch": request.github_branch,
                    "endpoint_url": endpoint_url,
                    "endpoint_name": clean_optional_string(result.get("endpoint_name")),
                    "service_kind": clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                    "hosting_mode": request.mode,
                    "copilot_model": "github-default-best-available",
                    "copilot_assignee": "copilot-swe-agent[bot]",
                },
            )
            pr_task = self.github_service.create_copilot_log_forwarding_pr_task(
                token=request.github_token,
                repo_name=request.github_repo,
                base_branch=request.github_branch,
                endpoint_url=endpoint_url,
                endpoint_name=clean_optional_string(result.get("endpoint_name")),
                endpoint_auth_mode=clean_optional_string(result.get("endpoint_auth_mode")),
                service_kind=clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                hosting_mode=request.mode,
                copilot_model="",
                prompt_text=prompt_text,
            )
            prompt_info.update(
                self.mlops_service.archive_copilot_pr_prompt(
                    prompt_text,
                    {
                        "repo_name": request.github_repo,
                        "base_branch": request.github_branch,
                        "endpoint_url": endpoint_url,
                        "endpoint_name": clean_optional_string(result.get("endpoint_name")),
                        "service_kind": clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                        "hosting_mode": request.mode,
                        "github_issue_number": pr_task.get("issue_number"),
                        "github_issue_url": clean_optional_string(pr_task.get("html_url")),
                        "copilot_model": clean_optional_string(pr_task.get("copilot_model")),
                        "copilot_assignee": clean_optional_string(pr_task.get("copilot_assignee")),
                    },
                )
            )
            safe_pr_task = {key: value for key, value in pr_task.items() if key != "prompt_text"}
            safe_pr_task.update(prompt_info)
            result["github_pr_task"] = safe_pr_task
            result["github_pr_url"] = clean_optional_string(pr_task.get("html_url"))
            result["github_pr_error"] = ""
            result["message"] = f"{clean_optional_string(result.get('message')) or 'Hosting is ready.'} Copilot PR task created."
            result["summary"] = (
                clean_optional_string(result.get("summary"))
                + f"\nCopilot PR task: {clean_optional_string(pr_task.get('html_url'))}"
                + f"\nCopilot prompt version: {clean_optional_string(prompt_info.get('copilot_prompt_version_label'))}"
            )
            persist_pr_metadata(
                {
                    "github_pr_task": safe_pr_task,
                    "github_pr_url": result["github_pr_url"],
                    "github_pr_error": "",
                }
            )
        except Exception as exc:
            result["github_pr_error"] = clean_optional_string(exc)
            result["summary"] = (
                clean_optional_string(result.get("summary"))
                + "\nCopilot PR task failed: "
                + result["github_pr_error"]
            )
            persist_pr_metadata({"github_pr_error": result["github_pr_error"], "github_pr_prompt": prompt_info})

    def _run_local_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        serve_script = os.path.join(self.project_dir, "serve_model.py")
        if not os.path.exists(serve_script):
            raise FileNotFoundError("Could not find 'serve_model.py' in the app directory.")

        self.observability_service.shutdown_local_hosting_stack()
        if request.auto_install_missing_tools:
            ctx.emit("progress", "Installing Grafana and Prometheus...")
            self.observability_service.install_local_observability_dependencies(emit=lambda msg: ctx.emit("progress", msg))

        port = self.observability_service.find_free_port()
        prometheus_port = self.observability_service.find_free_port()
        grafana_port = self.observability_service.find_free_port()
        host = "127.0.0.1"
        api_url = f"http://{host}:{port}/predict"
        health_url = f"http://{host}:{port}/health"
        metrics_url = f"http://{host}:{port}/metrics"
        prometheus_url = f"http://127.0.0.1:{prometheus_port}"
        grafana_url = f"http://127.0.0.1:{grafana_port}"
        dashboard_url = f"{grafana_url}/d/log-monitor-local/log-monitor-local-hosting?orgId=1&refresh=10s"

        ctx.emit("progress", "Starting local prediction API...")
        process = subprocess.Popen(
            [sys.executable, serve_script, "--model-dir", request.model_dir, "--host", host, "--port", str(port)],
            cwd=self.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ctx.register_subprocess("local_api", process)
        deadline = time.time() + 60
        ready = False
        while time.time() < deadline:
            ctx.check_cancelled()
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Local hosting exited unexpectedly.\n\n{output}")
            try:
                response = requests.get(health_url, timeout=2)
                if response.ok:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            raise TimeoutError("Timed out waiting for the local prediction API to become ready.")

        training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        local_hosting_meta = {
            "mode": "local",
            "service_kind": "grafana_local",
            "model_dir": request.model_dir,
            "model_version_id": clean_optional_string(training_metadata.get("model_version_id", "")),
            "training_run_id": clean_optional_string(training_metadata.get("run_id", "")),
            "data_version_id": clean_optional_string(training_metadata.get("data_version_id", "")),
            "api_url": api_url,
            "health_url": health_url,
            "metrics_url": metrics_url,
            "prometheus_url": prometheus_url,
            "grafana_url": grafana_url,
            "dashboard_url": dashboard_url,
            "created_at": now_utc_iso(),
        }
        tracking_console_url, tracking_console_note = self.mlops_service.resolve_dashboard_tracking_console(
            backend="local",
            tracking_uri=self.mlops_service.local_tracking_uri,
            azure_studio_url="",
            launch_live_console=True,
            hosted_model_path=request.model_dir,
        )
        observability_files = self.observability_service.write_local_observability_files(
            hosting_meta=local_hosting_meta,
            training_meta=training_metadata,
            tracking_console_url=tracking_console_url,
            tracking_console_note=tracking_console_note,
        )

        ctx.emit("progress", "Starting local Prometheus...")
        prometheus_process = self.observability_service.start_local_prometheus(
            config_path=observability_files["prometheus_config_path"],
            data_path=observability_files["prometheus_data_path"],
            port=prometheus_port,
            log_path=observability_files["prometheus_launch_log_path"],
        )
        ctx.register_subprocess("prometheus", prometheus_process)
        ctx.emit("progress", "Starting local Grafana...")
        grafana_process = self.observability_service.start_local_grafana(
            provisioning_path=observability_files["grafana_provisioning_path"],
            dashboard_path=observability_files["grafana_dashboard_path"],
            data_path=observability_files["grafana_data_path"],
            logs_path=observability_files["grafana_logs_path"],
            plugins_path=observability_files["grafana_plugins_path"],
            port=grafana_port,
            log_path=observability_files["grafana_launch_log_path"],
        )
        ctx.register_subprocess("grafana", grafana_process)

        if not self.observability_service.wait_for_http_endpoint(dashboard_url, timeout_seconds=45, ready_statuses=(200,)):
            dashboard_url = grafana_url
            local_hosting_meta["dashboard_url"] = dashboard_url

        self.observability_service.hosting_process = process
        self.observability_service.prometheus_process = prometheus_process
        self.observability_service.grafana_process = grafana_process
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(local_hosting_meta)
        return {
            "operation": "hosting",
            "message": "Local Grafana hosting stack is ready.",
            "api_url": api_url,
            "dashboard_url": dashboard_url,
            "prometheus_url": prometheus_url,
            "grafana_url": grafana_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Local Grafana hosting stack is running.\nPOST {api_url}\nGrafana: {dashboard_url}\n"
                f"Prometheus: {prometheus_url}\nMetrics: {metrics_url}\nBody: {{\"errorMessage\": \"...\"}}\n"
                "Response: {\"prediction\": \"...\"}"
            ),
        }

    def _run_azure_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        service_kind = clean_optional_string(request.azure_service) or "queued_batch"
        if service_kind == "serverless":
            return self._run_azure_serverless_hosting(ctx, request)
        if service_kind == "online":
            return self._run_azure_online_hosting(ctx, request)
        if service_kind == "queued_batch":
            return self._run_azure_queued_batch_hosting(ctx, request)
        return self._run_azure_batch_hosting(ctx, request)

    def _run_azure_serverless_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        model_id = self.azure_platform_service.normalize_serverless_model_id(request.azure_serverless_model_id)
        if not model_id:
            raise ValueError("Azure serverless hosting needs a model ID from the Azure ML model catalog.")

        ctx.emit("progress", "Preparing Azure serverless hosting...")
        credential = self.azure_platform_service.create_interactive_credential(request.azure_tenant_id)
        ml_client = self.azure_platform_service.ensure_azure_workspace(
            request.azure_sub_id,
            request.azure_tenant_id,
            emit=lambda msg: ctx.emit("progress", msg),
            credential=credential,
        )
        timestamp = int(time.time())
        model_hint = model_id.rstrip("/").split("/")[-1] or "model"
        endpoint_name = clean_optional_string(request.azure_serverless_endpoint_name)
        if not endpoint_name:
            endpoint_name = f"log-monitor-serverless-{model_hint}-{timestamp}"
        endpoint_name = self.azure_platform_service.sanitize_azure_endpoint_name(endpoint_name)
        deployment_meta = self.azure_platform_service.deploy_azure_serverless_endpoint(
            ml_client=ml_client,
            model_id=model_id,
            endpoint_name=endpoint_name,
            endpoint_auth_mode="key",
            credential=credential,
            sub_id=request.azure_sub_id,
            emit=lambda msg: ctx.emit("progress", msg),
        )
        scoring_uri = clean_optional_string(deployment_meta.get("api_url"))
        endpoints_studio_url = self.azure_platform_service.build_azure_endpoints_studio_url(request.azure_sub_id, request.azure_tenant_id)
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        endpoint_resource_id = self.azure_platform_service.build_serverless_endpoint_resource_id(
            request.azure_sub_id,
            clean_optional_string(deployment_meta.get("endpoint_name")) or endpoint_name,
        )
        endpoint_portal_url = self.azure_platform_service.build_serverless_endpoint_portal_url(
            request.azure_sub_id,
            request.azure_tenant_id,
            clean_optional_string(deployment_meta.get("endpoint_name")) or endpoint_name,
        )
        visible_in_workspace_list = bool(deployment_meta.get("visible_in_workspace_list"))
        serverless_list_error = clean_optional_string(deployment_meta.get("serverless_list_error"))
        visible_in_arm_resource_list = bool(deployment_meta.get("visible_in_arm_resource_list"))
        arm_list_error = clean_optional_string(deployment_meta.get("arm_list_error"))
        creation_method = clean_optional_string(deployment_meta.get("creation_method"))
        visibility_note = "Verified in both the ARM resource list and the Azure ML SDK workspace list."
        if not visible_in_arm_resource_list and arm_list_error:
            visibility_note = f"Azure SDK created the endpoint, but ARM resource listing failed: {arm_list_error}"
        elif not visible_in_arm_resource_list:
            visibility_note = "Azure SDK created the endpoint, but the ARM resource list did not return it yet."
        if serverless_list_error:
            visibility_note = f"Azure created the endpoint, but the workspace list check failed: {serverless_list_error}"
        elif visible_in_arm_resource_list and not visible_in_workspace_list:
            visibility_note = "Azure ARM created the endpoint, but the Azure ML SDK list did not return it yet. Refresh Studio or use the Portal link below."
        elif not visible_in_workspace_list:
            visibility_note = "Azure created the endpoint, but the workspace list did not return it yet. Use the resource link below or refresh Studio."
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure_serverless",
                "service_kind": "serverless",
                "model_dir": clean_optional_string(request.model_dir),
                "serverless_model_id": model_id,
                "endpoint_name": clean_optional_string(deployment_meta.get("endpoint_name")) or endpoint_name,
                "endpoint_auth_mode": clean_optional_string(deployment_meta.get("endpoint_auth_mode")) or "key",
                "provisioning_state": clean_optional_string(deployment_meta.get("provisioning_state")),
                "creation_method": creation_method,
                "arm_api_version": clean_optional_string(deployment_meta.get("arm_api_version")),
                "arm_creation_error": clean_optional_string(deployment_meta.get("arm_creation_error")),
                "arm_resource": deployment_meta.get("arm_resource", {}),
                "visible_in_arm_resource_list": visible_in_arm_resource_list,
                "arm_list_error": arm_list_error,
                "arm_serverless_endpoint_names": deployment_meta.get("arm_serverless_endpoint_names", []),
                "arm_serverless_endpoints": deployment_meta.get("arm_serverless_endpoints", []),
                "visible_in_workspace_list": visible_in_workspace_list,
                "serverless_list_error": serverless_list_error,
                "workspace_serverless_endpoint_names": deployment_meta.get("workspace_serverless_endpoint_names", []),
                "serverless_endpoint_resource_id": endpoint_resource_id,
                "serverless_endpoint_portal_url": endpoint_portal_url,
                "serverless_endpoints_studio_url": endpoints_studio_url,
                "api_url": scoring_uri,
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "mlops_url": mlops_url,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure serverless endpoint is ready.",
            "api_url": scoring_uri,
            "endpoint_name": clean_optional_string(deployment_meta.get("endpoint_name")) or endpoint_name,
            "mlops_url": endpoints_studio_url or mlops_url,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure serverless endpoint is ready.\nTarget URI: {scoring_uri}\n"
                f"Model ID: {model_id}\nEndpoint Name: {clean_optional_string(deployment_meta.get('endpoint_name')) or endpoint_name}\n"
                f"Creation: {creation_method}\n"
                f"{visibility_note}\n"
                f"Studio: {endpoints_studio_url}\nPortal Resource: {endpoint_portal_url}\n"
                "Authentication: endpoint keys from Azure ML Studio."
            ),
        }

    def _run_azure_online_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        ctx.emit("progress", "Preparing Azure hosting...")
        ml_client = self.azure_platform_service.ensure_azure_workspace(
            request.azure_sub_id,
            request.azure_tenant_id,
            emit=lambda msg: ctx.emit("progress", msg),
        )
        timestamp = int(time.time())
        model_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-model-{Path(request.model_dir).name}-{timestamp}")
        endpoint_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-endpoint-{timestamp}")
        deployment_name = "blue"
        env_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-inference-env-{timestamp}")
        deployment_meta = self.azure_platform_service.deploy_azure_online_endpoint(
            ml_client=ml_client,
            model_dir=request.model_dir,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            environment_name=env_name,
            model_name=model_name,
            azure_compute=request.azure_compute,
            preferred_instance_type=request.azure_instance_type,
            endpoint_auth_mode="key",
            emit=lambda msg: ctx.emit("progress", msg),
        )
        scoring_uri = clean_optional_string(deployment_meta.get("api_url"))
        selected_instance_type = clean_optional_string(deployment_meta.get("instance_type"))
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure",
                "service_kind": "online",
                "model_dir": request.model_dir,
                "model_version_id": clean_optional_string(training_metadata.get("model_version_id", "")),
                "training_run_id": clean_optional_string(training_metadata.get("run_id", "")),
                "data_version_id": clean_optional_string(training_metadata.get("data_version_id", "")),
                "endpoint_name": endpoint_name,
                "deployment_name": deployment_name,
                "instance_type": selected_instance_type,
                "endpoint_auth_mode": "key",
                "api_url": scoring_uri,
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "azure_compute": request.azure_compute,
                "mlops_url": mlops_url,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure real-time endpoint is ready.",
            "api_url": scoring_uri,
            "endpoint_name": endpoint_name,
            "mlops_url": mlops_url,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure real-time endpoint is ready.\nPOST {scoring_uri}\nInstance Type: {selected_instance_type}\n"
                "Body: {\"errorMessage\": \"...\"}\nResponse: {\"prediction\": \"...\"}\nAuthentication: endpoint keys."
            ),
        }

    def _run_azure_batch_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        ctx.emit("progress", "Preparing Azure hosting...")
        ml_client = self.azure_platform_service.ensure_azure_workspace(
            request.azure_sub_id,
            request.azure_tenant_id,
            emit=lambda msg: ctx.emit("progress", msg),
        )
        timestamp = int(time.time())
        endpoint_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-endpoint-{timestamp}")
        schedule_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-schedule-{timestamp}")
        env_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-env-{timestamp}")
        model_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-model-{Path(request.model_dir).name}-{timestamp}")
        deployment_meta = self.azure_platform_service.deploy_azure_batch_endpoint(
            ml_client=ml_client,
            model_dir=request.model_dir,
            azure_compute=request.azure_compute,
            preferred_instance_type=request.azure_instance_type,
            endpoint_name=endpoint_name,
            environment_name=env_name,
            model_name=model_name,
            endpoint_auth_mode="aad_token",
            emit=lambda msg: ctx.emit("progress", msg),
        )
        deployment_name = clean_optional_string(deployment_meta.get("deployment_name")) or "default"
        seed_job_name = self.azure_platform_service.create_daily_batch_schedule(
            ml_client=ml_client,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            batch_input_uri=request.batch_input_uri,
            schedule_name=schedule_name,
            batch_hour=request.batch_hour,
            batch_minute=request.batch_minute,
            batch_timezone=request.batch_timezone or "UTC",
            emit=lambda msg: ctx.emit("progress", msg),
        )
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure_batch",
                "service_kind": "batch",
                "model_dir": request.model_dir,
                "model_version_id": clean_optional_string(training_metadata.get("model_version_id", "")),
                "training_run_id": clean_optional_string(training_metadata.get("run_id", "")),
                "data_version_id": clean_optional_string(training_metadata.get("data_version_id", "")),
                "endpoint_name": endpoint_name,
                "deployment_name": deployment_name,
                "schedule_name": schedule_name,
                "schedule_time": f"{request.batch_hour:02d}:{request.batch_minute:02d}",
                "schedule_time_zone": request.batch_timezone or "UTC",
                "batch_input_uri": request.batch_input_uri,
                "seed_job_name": seed_job_name,
                "instance_type": clean_optional_string(deployment_meta.get("instance_type")),
                "compute_name": clean_optional_string(deployment_meta.get("compute_name")),
                "endpoint_auth_mode": "aad_token",
                "api_url": clean_optional_string(deployment_meta.get("api_url")),
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "azure_compute": request.azure_compute,
                "mlops_url": mlops_url,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure batch endpoint and daily schedule are ready.",
            "api_url": clean_optional_string(deployment_meta.get("api_url")),
            "endpoint_name": endpoint_name,
            "mlops_url": mlops_url,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure batch endpoint is ready.\nInvoke: {clean_optional_string(deployment_meta.get('api_url'))}\n"
                f"Cluster: {clean_optional_string(deployment_meta.get('compute_name'))} ({clean_optional_string(deployment_meta.get('instance_type'))}, min nodes 0)\n"
                f"Schedule: every day at {request.batch_hour:02d}:{request.batch_minute:02d} {request.batch_timezone or 'UTC'}\n"
                f"Input: {request.batch_input_uri}\nOutput: prediction rows are written to Azure Storage when each batch job finishes.\nAuthentication: Microsoft Entra ID."
            ),
        }

    def _run_azure_queued_batch_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        ctx.emit("progress", "Preparing Azure queued batch hosting...")
        credential = self.azure_platform_service.create_interactive_credential(request.azure_tenant_id)
        ml_client = self.azure_platform_service.ensure_azure_workspace(
            request.azure_sub_id,
            request.azure_tenant_id,
            emit=lambda msg: ctx.emit("progress", msg),
            credential=credential,
        )
        timestamp = int(time.time())
        function_app_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-func-{timestamp}", max_length=60)
        function_plan_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-flex-{timestamp}", max_length=40)
        storage_account_name = self.azure_platform_service.sanitize_azure_storage_name(f"logmonitor{timestamp}")
        service_bus_namespace_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-sb-{timestamp}", max_length=50)
        service_bus_queue_name = "logs"
        datastore_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batches-{timestamp}", max_length=30).replace("-", "")
        batch_endpoint_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-endpoint-{timestamp}")
        batch_env_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-env-{timestamp}")
        model_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-model-{Path(request.model_dir).name}-{timestamp}")
        batch_timezone_iana = self.azure_platform_service.get_azure_batch_timezone_iana(request.batch_timezone)

        ctx.emit("progress", "Creating Azure queue, storage, and Function App...")
        infra_outputs = self.azure_platform_service.deploy_azure_function_bridge_infrastructure(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            function_plan_name=function_plan_name,
            storage_account_name=storage_account_name,
            service_bus_namespace_name=service_bus_namespace_name,
            service_bus_queue_name=service_bus_queue_name,
        )
        storage_connection_string = clean_optional_string(infra_outputs.get("storageConnectionString"))
        storage_account_key = clean_optional_string(infra_outputs.get("storageAccountKey"))
        service_bus_connection_string = clean_optional_string(infra_outputs.get("serviceBusConnectionString"))
        function_host_name = clean_optional_string(infra_outputs.get("functionAppHostName"))
        if not storage_connection_string or not storage_account_key or not service_bus_connection_string or not function_host_name:
            raise RuntimeError("Azure infrastructure deployment did not return the required connection details.")

        ctx.emit("progress", "Registering Azure Blob datastore for queued batches...")
        self.azure_platform_service.ensure_azure_blob_datastore(
            ml_client=ml_client,
            datastore_name=datastore_name,
            storage_account_name=storage_account_name,
            container_name="log-batches",
            storage_account_key=storage_account_key,
        )
        deployment_meta = self.azure_platform_service.deploy_azure_batch_endpoint(
            ml_client=ml_client,
            model_dir=request.model_dir,
            azure_compute=request.azure_compute,
            preferred_instance_type=request.azure_instance_type,
            endpoint_name=batch_endpoint_name,
            environment_name=batch_env_name,
            model_name=model_name,
            endpoint_auth_mode="aad_token",
            emit=lambda msg: ctx.emit("progress", msg),
        )
        deployment_name = clean_optional_string(deployment_meta.get("deployment_name")) or "default"
        ctx.emit("progress", "Updating Azure Function settings...")
        self.azure_platform_service.set_function_app_settings(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            settings={
                "AzureWebJobsStorage__accountName": storage_account_name,
                "LOGMONITOR_STORAGE_CONNECTION": storage_connection_string,
                "LOGMONITOR_BLOB_CONTAINER": "log-batches",
                "LOGMONITOR_SERVICEBUS_CONNECTION": service_bus_connection_string,
                "LOGMONITOR_QUEUE_NAME": service_bus_queue_name,
                "LOGMONITOR_BATCH_TIME": f"{request.batch_hour:02d}:{request.batch_minute:02d}",
                "LOGMONITOR_BATCH_TIME_ZONE": batch_timezone_iana,
                "LOGMONITOR_BATCH_ENDPOINT_NAME": batch_endpoint_name,
                "LOGMONITOR_BATCH_DEPLOYMENT_NAME": deployment_name,
                "LOGMONITOR_AML_SUBSCRIPTION_ID": request.azure_sub_id,
                "LOGMONITOR_AML_RESOURCE_GROUP": self.azure_platform_service.resource_group,
                "LOGMONITOR_AML_WORKSPACE_NAME": self.azure_platform_service.workspace_name,
                "LOGMONITOR_DATASTORE_NAME": datastore_name,
                "LOGMONITOR_INPUT_PREFIX": "queue-batches",
                "LOGMONITOR_STATE_BLOB": "queue-state/scheduler-state.json",
            },
        )
        ctx.emit("progress", "Packaging the Azure Function bridge...")
        package_path = self.azure_platform_service.build_function_bridge_package(f"log-monitor-function-{timestamp}")
        package_uri = self.azure_platform_service.upload_function_bridge_package(
            storage_connection_string=storage_connection_string,
            storage_account_name=storage_account_name,
            storage_account_key=storage_account_key,
            package_path=package_path,
            package_container_name="functionpkgs",
        )
        ctx.emit("progress", "Deploying the Azure Function bridge...")
        self.azure_platform_service.trigger_function_app_onedeploy(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            package_uri=package_uri,
        )
        ctx.emit("progress", "Waiting for the Azure log API to become ready...")
        log_api_url, function_key = self.azure_platform_service.wait_for_function_bridge_endpoint(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            function_host_name=function_host_name,
        )
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure_queue_batch",
                "service_kind": "queued_batch",
                "model_dir": request.model_dir,
                "model_version_id": clean_optional_string(training_metadata.get("model_version_id", "")),
                "training_run_id": clean_optional_string(training_metadata.get("run_id", "")),
                "data_version_id": clean_optional_string(training_metadata.get("data_version_id", "")),
                "api_url": log_api_url,
                "function_key": function_key,
                "function_app_name": function_app_name,
                "function_host_name": function_host_name,
                "service_bus_namespace": service_bus_namespace_name,
                "service_bus_queue": service_bus_queue_name,
                "storage_account_name": storage_account_name,
                "log_container_name": "log-batches",
                "datastore_name": datastore_name,
                "endpoint_name": batch_endpoint_name,
                "deployment_name": deployment_name,
                "batch_endpoint_url": clean_optional_string(deployment_meta.get("api_url")),
                "schedule_time": f"{request.batch_hour:02d}:{request.batch_minute:02d}",
                "schedule_time_zone": request.batch_timezone,
                "schedule_time_zone_iana": batch_timezone_iana,
                "instance_type": clean_optional_string(deployment_meta.get("instance_type")),
                "compute_name": clean_optional_string(deployment_meta.get("compute_name")),
                "endpoint_auth_mode": "aad_token",
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "azure_compute": request.azure_compute,
                "mlops_url": mlops_url,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure queued batch pipeline is ready.",
            "api_url": log_api_url,
            "endpoint_name": batch_endpoint_name,
            "mlops_url": mlops_url,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure queued batch pipeline is ready.\nLog API: {log_api_url}\nQueue: {service_bus_namespace_name}/{service_bus_queue_name}\n"
                f"Batch Endpoint: {batch_endpoint_name}\nSchedule: every day at {request.batch_hour:02d}:{request.batch_minute:02d} {request.batch_timezone}\n"
                f"Cluster: {clean_optional_string(deployment_meta.get('compute_name'))} ({clean_optional_string(deployment_meta.get('instance_type'))}, min nodes 0)\n"
                "Flow: POST logs to the Function API, queue them, and process the accumulated logs once per day in the background."
            ),
        }
