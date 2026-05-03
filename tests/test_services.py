from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app_core.contracts import HostingRequest, MlflowConfig
from app_core.azure_platform_service import AzurePlatformService
from app_core.data_prep_service import DataPrepService
from app_core.github_service import GitHubService
from app_core.hosting_service import HostingService
from app_core.mlops_service import MlopsService
from app_core.model_catalog_service import ModelCatalogService
from app_core.observability_service import ObservabilityService
from app_core.runtime import ArtifactStore, JobManager, StateStore


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        (self.project_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "prompt.txt").write_text("prompt body", encoding="utf-8")
        self.artifact_store = ArtifactStore(str(self.project_dir))
        self.state_store = StateStore(self.artifact_store.state_db_path)
        self.job_manager = JobManager(self.state_store)
        self.model_catalog = ModelCatalogService(str(self.project_dir), self.artifact_store)
        self.mlops = MlopsService(
            str(self.project_dir),
            self.artifact_store,
            self.model_catalog,
            resource_group="rg",
            workspace_name="ws",
            local_tracking_uri=str(self.project_dir / "mlruns"),
        )
        self.data_prep = DataPrepService(self.job_manager, self.mlops, self.model_catalog)
        self.azure_platform = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        self.observability = ObservabilityService(str(self.project_dir), self.artifact_store, self.state_store)
        self.hosting = HostingService(
            str(self.project_dir),
            self.job_manager,
            self.model_catalog,
            self.mlops,
            self.azure_platform,
            self.observability,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def load_function_bridge_module(self):
        class FakeFunctionApp:
            def function_name(self, **_kwargs):
                return lambda func: func

            def route(self, **_kwargs):
                return lambda func: func

            def schedule(self, **_kwargs):
                return lambda func: func

        class FakeHttpResponse:
            def __init__(self, body=None, status_code=200, mimetype=None):
                self.body = body
                self.status_code = status_code
                self.mimetype = mimetype

        class FakeResourceNotFoundError(Exception):
            pass

        class FakeContentSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        azure = types.ModuleType("azure")
        azure.__path__ = []
        azure_functions = types.ModuleType("azure.functions")
        azure_functions.FunctionApp = FakeFunctionApp
        azure_functions.HttpRequest = object
        azure_functions.HttpResponse = FakeHttpResponse
        azure_functions.TimerRequest = object
        azure_functions.AuthLevel = SimpleNamespace(FUNCTION="FUNCTION")
        azure.functions = azure_functions

        azure_ai = types.ModuleType("azure.ai")
        azure_ai.__path__ = []
        azure_ai_ml = types.ModuleType("azure.ai.ml")
        azure_ai_ml.__path__ = []
        azure_ai_ml.Input = lambda **kwargs: SimpleNamespace(**kwargs)
        azure_ai_ml.MLClient = object
        azure_ai_ml.command = lambda **kwargs: SimpleNamespace(**kwargs)
        azure_ai_ml_constants = types.ModuleType("azure.ai.ml.constants")
        azure_ai_ml_constants.AssetTypes = SimpleNamespace(URI_FILE="uri_file")
        azure_ai_ml_entities = types.ModuleType("azure.ai.ml.entities")
        azure_ai_ml_entities.AmlCompute = object
        azure_ai_ml_entities.Data = object
        azure.ai = azure_ai
        azure_ai.ml = azure_ai_ml
        azure_ai_ml.constants = azure_ai_ml_constants
        azure_ai_ml.entities = azure_ai_ml_entities

        azure_core = types.ModuleType("azure.core")
        azure_core.__path__ = []
        azure_core_exceptions = types.ModuleType("azure.core.exceptions")
        azure_core_exceptions.ResourceNotFoundError = FakeResourceNotFoundError
        azure.core = azure_core
        azure_core.exceptions = azure_core_exceptions

        azure_identity = types.ModuleType("azure.identity")
        azure_identity.DefaultAzureCredential = object
        azure_servicebus = types.ModuleType("azure.servicebus")
        azure_servicebus.ServiceBusClient = object
        azure_servicebus.ServiceBusMessage = object
        azure_storage = types.ModuleType("azure.storage")
        azure_storage.__path__ = []
        azure_storage_blob = types.ModuleType("azure.storage.blob")
        azure_storage_blob.BlobServiceClient = object
        azure_storage_blob.ContentSettings = FakeContentSettings
        azure.identity = azure_identity
        azure.servicebus = azure_servicebus
        azure.storage = azure_storage
        azure_storage.blob = azure_storage_blob

        stubs = {
            "azure": azure,
            "azure.functions": azure_functions,
            "azure.ai": azure_ai,
            "azure.ai.ml": azure_ai_ml,
            "azure.ai.ml.constants": azure_ai_ml_constants,
            "azure.ai.ml.entities": azure_ai_ml_entities,
            "azure.core": azure_core,
            "azure.core.exceptions": azure_core_exceptions,
            "azure.identity": azure_identity,
            "azure.servicebus": azure_servicebus,
            "azure.storage": azure_storage,
            "azure.storage.blob": azure_storage_blob,
        }
        bridge_path = Path(__file__).resolve().parents[1] / "azure_function_bridge" / "function_app.py"
        spec = importlib.util.spec_from_file_location("bridge_function_app_under_test", bridge_path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        return module

    def test_github_service_fetches_repo_and_branch_names(self):
        service = GitHubService()
        repos_response = mock.Mock()
        repos_response.json.return_value = [{"full_name": "owner/repo"}]
        repos_response.raise_for_status.return_value = None
        branches_response = mock.Mock()
        branches_response.json.return_value = [{"name": "main"}]
        branches_response.raise_for_status.return_value = None
        with mock.patch("app_core.github_service.requests.get", side_effect=[repos_response, branches_response]):
            self.assertEqual(service.fetch_repos("token"), ["owner/repo"])
            self.assertEqual(service.fetch_branches("token", "owner/repo"), ["main"])

    def test_github_copilot_prompt_does_not_require_token_settings_by_default(self):
        service = GitHubService()

        prompt = service.build_log_forwarding_copilot_prompt(
            repo_name="owner/repo",
            base_branch="main",
            endpoint_url="https://example.test/api/logs?code=function-key",
            endpoint_auth_mode="",
        )

        self.assertIn("https://example.test/api/logs?code=function-key", prompt)
        self.assertIn("Hard-code the deployed Log Monitor endpoint URL above as the default target", prompt)
        self.assertIn("Do not require the user to access the server or set environment variables", prompt)
        self.assertIn("already includes the required Function access code", prompt)
        self.assertNotIn("LOG_MONITOR_ENDPOINT_URL", prompt)
        self.assertNotIn("LOG_MONITOR_FORWARDING_ENABLED", prompt)
        self.assertNotIn("LOG_MONITOR_ENDPOINT_KEY", prompt)
        self.assertNotIn("bearer-token", prompt.lower())
        self.assertNotIn("bearer token", prompt.lower())

    def test_github_copilot_prompt_mentions_endpoint_key_only_for_key_auth(self):
        service = GitHubService()

        prompt = service.build_log_forwarding_copilot_prompt(
            repo_name="owner/repo",
            base_branch="main",
            endpoint_url="https://example.test/score",
            endpoint_auth_mode="key",
        )

        self.assertIn("Authentication mode: key", prompt)
        self.assertIn("Use the exact deployed endpoint URL above as the committed forwarding target", prompt)
        self.assertIn("Function proxy URL", prompt)
        self.assertIn("TODO", prompt)
        self.assertNotIn("LOG_MONITOR_ENDPOINT_KEY", prompt)
        self.assertNotIn("bearer-token", prompt.lower())
        self.assertNotIn("bearer token", prompt.lower())

    def test_github_copilot_prompt_includes_azure_studio_and_prompt_version(self):
        service = GitHubService()
        studio_url = "https://ml.azure.com/endpoints?wsid=abc&tid=tenant&reloadCount=1"

        prompt = service.build_log_forwarding_copilot_prompt(
            repo_name="owner/repo",
            base_branch="main",
            endpoint_url="https://example.test/api/logs?code=function-key",
            endpoint_name="log-monitor-endpoint",
            endpoint_auth_mode="key",
            service_kind="online",
            hosting_mode="azure",
            azure_studio_endpoint_url=studio_url,
            copilot_prompt_version_label="abc123",
            copilot_prompt_version_id="abc123full",
        )
        body = service.build_log_forwarding_issue_body(prompt, "https://example.test/api/logs?code=function-key", studio_url)

        self.assertIn(studio_url, prompt)
        self.assertIn("Copilot prompt version: abc123 (abc123full)", prompt)
        self.assertIn("Azure ML Studio endpoint page", body)
        self.assertIn(studio_url, body)

    def test_hosting_copilot_pr_task_uses_studio_url_and_logs_prompt_version(self):
        studio_url = "https://ml.azure.com/endpoints?wsid=abc&tid=tenant&reloadCount=1"
        request = HostingRequest(
            model_dir="",
            mode="azure",
            azure_service="online",
            github_token="gh-token",
            github_repo="owner/repo",
            github_branch="main",
        )
        result = {
            "operation": "hosting",
            "message": "Azure endpoint is ready.",
            "summary": "Azure endpoint is ready.",
            "api_url": "https://score.azureml.net/score",
            "log_api_url": "https://func.azurewebsites.net/api/logs?code=function-key",
            "endpoint_name": "log-monitor-endpoint",
            "endpoint_auth_mode": "key",
            "service_kind": "online",
            "azure_endpoint_studio_url": studio_url,
            "azure_mlflow_tracking_uri": "azureml://tracking",
        }
        captured = {}

        def fake_create_pr_task(**kwargs):
            captured["create_pr_task"] = kwargs
            return {
                "title": "Integrate async Log Monitor log forwarding",
                "repo_name": kwargs["repo_name"],
                "base_branch": kwargs["base_branch"],
                "issue_number": 17,
                "issue_url": "https://api.github.test/issues/17",
                "html_url": "https://github.test/owner/repo/issues/17",
                "endpoint_url": kwargs["endpoint_url"],
                "endpoint_name": kwargs["endpoint_name"],
                "azure_studio_endpoint_url": kwargs["azure_studio_endpoint_url"],
                "copilot_assignee": "copilot-swe-agent[bot]",
                "copilot_model": "github-default-best-available",
                "prompt_text": kwargs["prompt_text"],
            }

        def fake_log_prompt_mlflow(**kwargs):
            captured["log_prompt_mlflow"] = kwargs
            return {"copilot_prompt_mlflow_status": "logged", "copilot_prompt_mlflow_run_id": "run-123"}

        with mock.patch.object(self.hosting.github_service, "create_copilot_log_forwarding_pr_task", side_effect=fake_create_pr_task), mock.patch.object(
            self.hosting.mlops_service,
            "log_copilot_pr_prompt_mlflow",
            side_effect=fake_log_prompt_mlflow,
        ):
            self.hosting._attach_github_copilot_pr_task(SimpleNamespace(emit=lambda *args, **kwargs: None), request, result)

        prompt_text = captured["create_pr_task"]["prompt_text"]
        self.assertEqual(captured["create_pr_task"]["azure_studio_endpoint_url"], studio_url)
        self.assertEqual(captured["create_pr_task"]["endpoint_url"], "https://func.azurewebsites.net/api/logs?code=function-key")
        self.assertIn(studio_url, prompt_text)
        self.assertIn("https://func.azurewebsites.net/api/logs?code=function-key", prompt_text)
        self.assertNotIn("https://score.azureml.net/score", prompt_text)
        self.assertIn("Hard-code the deployed Log Monitor endpoint URL above as the default target", prompt_text)
        self.assertNotIn("LOG_MONITOR_ENDPOINT_URL", prompt_text)
        self.assertEqual(captured["log_prompt_mlflow"]["tracking_uri"], "azureml://tracking")
        self.assertEqual(captured["log_prompt_mlflow"]["metadata"]["azure_studio_endpoint_url"], studio_url)
        self.assertEqual(result["github_pr_task"]["azure_studio_endpoint_url"], studio_url)
        self.assertEqual(result["github_pr_task"]["copilot_prompt_mlflow_status"], "logged")

    def test_model_catalog_archives_dataset_and_finds_models(self):
        dataset_path = self.project_dir / "sample.csv"
        dataset_path.write_text("LogMessage,class\nhello,Noise\n", encoding="utf-8")
        archive_info = self.model_catalog.archive_data_version(str(dataset_path), {"source": "test"})
        self.assertTrue(Path(archive_info["data_version_path"]).exists())

        model_dir = self.project_dir / "outputs" / "model_versions" / "v1" / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "pytorch_model.bin").write_bytes(b"0")
        (model_dir.parent / "last_training_mlflow.json").write_text(json.dumps({"run_id": "123", "created_at": "2024-01-01T00:00:00"}), encoding="utf-8")

        inventory = self.model_catalog.discover_available_hosted_models()
        self.assertTrue(any(item.path.endswith("final_model") for item in inventory))

    def test_mlops_service_resolves_local_config_and_loads_prompt(self):
        config, error = self.mlops.resolve_mlflow_config(
            enabled=False,
            backend="local",
            experiment_name="",
            registered_model_name="",
            tracking_uri="",
            require_tracking_uri=False,
        )
        self.assertIsNone(error)
        self.assertFalse(config.enabled)
        self.assertEqual(self.mlops.load_prompt(), "prompt body")

    def test_mlops_training_env_keeps_remote_tracking_enabled_without_local_mlflow(self):
        config = MlflowConfig(
            enabled=True,
            backend="azure",
            tracking_uri="azureml://mlflow/v1.0/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws",
            experiment_name="deberta-log-classification",
        )
        pipeline_context = {"pipeline_id": "pipe-1", "parent_run_id": "parent-1"}
        with mock.patch("app_core.mlops_service.mlflow", None):
            env = self.mlops.build_training_mlflow_env(
                config,
                pipeline_context,
                run_source="azure_cpu",
                environment_mode="azure",
            )
        self.assertEqual(env["MLOPS_ENABLED"], "1")
        self.assertEqual(env["MLFLOW_TRACKING_URI"], config.tracking_uri)
        self.assertEqual(env["MLFLOW_EXPERIMENT_NAME"], "deberta-log-classification")

    def test_mlops_service_versions_and_compares_prompts(self):
        first = self.mlops.archive_prompt_version("classify logs\nreturn json", {"llm_model": "gpt-test"})
        second = self.mlops.archive_prompt_version("classify logs by root cause\nreturn json", {"llm_model": "gpt-test"})

        self.assertTrue(Path(first["prompt_version_path"]).exists())
        self.assertTrue(Path(second["prompt_version_path"]).exists())
        self.assertEqual(first["prompt_hash"], first["prompt_version_id"])
        self.assertEqual(second["previous_prompt_version_id"], first["prompt_version_id"])
        self.assertTrue(Path(second["prompt_comparison_path"]).exists())

        versions = self.mlops.list_prompt_versions()
        self.assertEqual(len(versions), 2)
        comparison = self.mlops.compare_prompt_versions()
        self.assertIn("-classify logs", comparison["diff"])
        self.assertIn("+classify logs by root cause", comparison["diff"])

    def test_mlops_service_logs_copilot_prompt_to_mlflow(self):
        prompt_text = "Forward logs to https://func.azurewebsites.net/api/logs?code=function-key"
        prompt_info = self.mlops.archive_copilot_pr_prompt(prompt_text, {"repo_name": "owner/repo"})

        class FakeRun:
            def __init__(self):
                self.info = SimpleNamespace(run_id="run-123")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeMlflow:
            def __init__(self):
                self.tracking_uri = ""
                self.experiment_name = ""
                self.params = {}
                self.metrics = {}
                self.artifact_path = ""
                self.run_name = ""
                self.tags = {}

            def set_tracking_uri(self, value):
                self.tracking_uri = value

            def set_experiment(self, value):
                self.experiment_name = value

            def start_run(self, run_name="", tags=None):
                self.run_name = run_name
                self.tags = tags or {}
                return FakeRun()

            def log_param(self, key, value):
                self.params[key] = value

            def log_metric(self, key, value):
                self.metrics[key] = value

            def log_artifacts(self, local_dir, artifact_path=None):
                self.artifact_path = artifact_path
                self.artifact_names = {path.name for path in Path(local_dir).iterdir()}

        fake_mlflow = FakeMlflow()
        with mock.patch("app_core.mlops_service.mlflow", fake_mlflow):
            result = self.mlops.log_copilot_pr_prompt_mlflow(
                tracking_uri="azureml://tracking",
                experiment_name="log-monitor-copilot-prompts",
                prompt_text=prompt_text,
                prompt_info=prompt_info,
                metadata={
                    "repo_name": "owner/repo",
                    "endpoint_url": "https://func.azurewebsites.net/api/logs?code=function-key",
                    "azure_studio_endpoint_url": "https://ml.azure.com/endpoints?wsid=abc",
                },
            )

        self.assertEqual(result["copilot_prompt_mlflow_status"], "logged")
        self.assertEqual(result["copilot_prompt_mlflow_run_id"], "run-123")
        self.assertEqual(fake_mlflow.tracking_uri, "azureml://tracking")
        self.assertEqual(fake_mlflow.experiment_name, "log-monitor-copilot-prompts")
        self.assertEqual(fake_mlflow.params["repo_name"], "owner/repo")
        self.assertEqual(fake_mlflow.artifact_path, "copilot_pr_prompt")
        self.assertIn("copilot_prompt.txt", fake_mlflow.artifact_names)

    def test_data_prep_service_evaluates_prompt_test_cases(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"results": [{"class": "CONFIGURATION"}, {"class": "Noise"}]})}}],
            "usage": {"total_tokens": 12},
        }
        cases = [
            {"name": "Config", "message": "missing MAIL_HOST", "expected": "CONFIGURATION"},
            {"name": "Noise", "message": "processed Canceled", "expected": "Noise"},
        ]
        with mock.patch("app_core.data_prep_service.requests.post", return_value=response):
            result = self.data_prep.evaluate_prompt_test_cases(
                api_key="key",
                model_name="gpt-test",
                prompt_text="classify logs",
                cases=cases,
            )

        self.assertEqual(result["usage"]["total_tokens"], 12)
        self.assertTrue(all(case["match"] for case in result["cases"]))
        self.assertEqual(result["cases"][0]["got"], "CONFIGURATION")

    def test_azure_platform_registers_versioned_data_asset(self):
        dataset_path = self.project_dir / "labeled.csv"
        dataset_path.write_text("LogMessage,class\nhello,Noise\n", encoding="utf-8")
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None

        class FakeDataEntity:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeDataOperations:
            def __init__(self):
                self.created = None

            def create_or_update(self, data_asset):
                self.created = data_asset
                return SimpleNamespace(id="/asset/id", path="azureml://datastores/workspaceblobstore/paths/labeled.csv")

        fake_client = SimpleNamespace(data=FakeDataOperations())
        fake_asset_types = SimpleNamespace(URI_FILE="uri_file")

        with mock.patch("app_core.azure_platform_service.Data", FakeDataEntity), mock.patch(
            "app_core.azure_platform_service.AssetTypes",
            fake_asset_types,
        ):
            result = service.register_azure_data_asset(
                fake_client,
                str(dataset_path),
                asset_name="Log Monitor Labeled Data",
                asset_version="sha:bad version",
                tags={"pipeline_id": "abc", "empty": ""},
            )

        self.assertEqual(result["azure_data_asset_name"], "log-monitor-labeled-data")
        self.assertEqual(result["azure_data_asset_version"], "sha-bad-version")
        self.assertEqual(result["azure_data_asset_uri"], "azureml:log-monitor-labeled-data:sha-bad-version")
        self.assertEqual(result["azure_data_asset_id"], "/asset/id")
        self.assertEqual(fake_client.data.created.type, "uri_file")
        self.assertEqual(fake_client.data.created.tags, {"pipeline_id": "abc"})
        self.assertEqual(len(service.sanitize_azure_asset_version("a" * 64)), 30)

    def test_azure_host_instance_candidates_start_with_smaller_cpu_sizes(self):
        candidates = self.azure_platform.get_azure_host_instance_candidates("cpu")

        self.assertEqual(candidates[0], "Standard_D2as_v4")
        self.assertIn("Standard_DS2_v2", candidates)
        self.assertIn("Standard_DS3_v2", candidates)
        self.assertIn("Standard_E4s_v3", candidates)
        self.assertLess(candidates.index("Standard_D2as_v4"), candidates.index("Standard_E4s_v3"))

    def test_azure_quota_error_is_formatted_with_attempted_host_sizes(self):
        raw_error = (
            "BadRequest: Not enough quota available for Standard_E4s_v3 in Subscription. "
            "Additional needed: 8"
        )

        message = self.azure_platform.format_azure_hosting_error(
            RuntimeError(raw_error),
            ["Standard_D2as_v4", "Standard_DS2_v2", "Standard_E4s_v3"],
        )

        self.assertIn("does not have enough quota", message)
        self.assertIn("Standard_D2as_v4", message)
        self.assertIn("Standard_E4s_v3", message)
        self.assertIn("Request a quota increase", message)
        self.assertNotIn("BadRequest", message)

    def test_function_bridge_package_includes_feedback_retraining_code(self):
        bridge_root = self.project_dir / "azure_function_bridge"
        bridge_root.mkdir(parents=True, exist_ok=True)
        (bridge_root / "function_app.py").write_text("# function app\n", encoding="utf-8")
        (bridge_root / "requirements.txt").write_text("azure-functions\n", encoding="utf-8")
        (self.project_dir / "train.py").write_text("# train\n", encoding="utf-8")
        (self.project_dir / "mlops_utils.py").write_text("# utils\n", encoding="utf-8")
        (self.project_dir / "requirements.train.txt").write_text("pandas\n", encoding="utf-8")

        package_path = self.azure_platform.build_function_bridge_package("unit-feedback-bridge")

        with zipfile.ZipFile(package_path) as archive:
            names = set(archive.namelist())
        self.assertIn("function_app.py", names)
        self.assertIn("requirements.txt", names)
        self.assertIn("train.py", names)
        self.assertIn("mlops_utils.py", names)
        self.assertIn("requirements.train.txt", names)

    def test_function_bridge_normalizes_copied_jira_project_url(self):
        bridge = self.load_function_bridge_module()

        site_url = bridge._normalize_jira_site_url("https://eece00.atlassian.net/jira/software/projects/KAN/boards/1")

        self.assertEqual(site_url, "https://eece00.atlassian.net")

    def test_function_bridge_jira_issue_retries_without_invalid_priority(self):
        bridge = self.load_function_bridge_module()

        class FakeResponse:
            def __init__(self, status_code, text, payload=None):
                self.status_code = status_code
                self.text = text
                self._payload = payload or {}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise bridge.requests.HTTPError(self.text)

            def json(self):
                return self._payload

        fields = {
            "project": {"key": "KAN"},
            "summary": "summary",
            "description": {},
            "issuetype": {"name": "Bug"},
            "priority": {"name": "200"},
        }
        responses = [
            FakeResponse(400, '{"errors":{"priority":"Priority is not valid"}}'),
            FakeResponse(201, "{}", {"key": "KAN-12"}),
        ]
        posted_fields = []

        def fake_post(_url, headers=None, json=None, timeout=None):
            del headers, timeout
            posted_fields.append(dict(json["fields"]))
            return responses.pop(0)

        with mock.patch.object(bridge.requests, "post", side_effect=fake_post):
            result, warnings = bridge._post_jira_issue_with_fallbacks(
                "https://eece00.atlassian.net",
                {"Authorization": "Basic token"},
                fields,
                "Bug",
                "200",
            )

        self.assertEqual(result["key"], "KAN-12")
        self.assertIn("priority", posted_fields[0])
        self.assertNotIn("priority", posted_fields[1])
        self.assertTrue(warnings)

    def test_function_bridge_prediction_monitoring_counts_and_links_daily_jira_summary(self):
        bridge = self.load_function_bridge_module()
        state_writes = []

        def capture_state_write(state):
            state_writes.append(json.loads(json.dumps(state)))

        with mock.patch.dict(
            bridge.os.environ,
            {
                "LOGMONITOR_JIRA_MONITORING_ENABLED": "1",
                "LOGMONITOR_SOURCE_ENDPOINT_NAME": "log-monitor-endpoint",
                "LOGMONITOR_HOSTING_SERVICE_KIND": "real-time endpoint",
            },
            clear=False,
        ), mock.patch.object(bridge, "_load_monitoring_state", return_value={}), mock.patch.object(
            bridge,
            "_write_monitoring_state",
            side_effect=capture_state_write,
        ), mock.patch.object(
            bridge,
            "_upsert_jira_monitoring_issue",
            return_value={
                "issue_key": "KAN-55",
                "issue_url": "https://eece00.atlassian.net/browse/KAN-55",
                "operation": "created",
            },
        ):
            summary = bridge._record_prediction_monitoring(
                {"received_at": "2026-05-03T10:15:00Z", "message": "handled"},
                {"prediction": "Noise", "raw_response": {"prediction": "Noise"}},
                "Noise",
                [{"type": "ignore"}],
                [],
                {},
            )

        self.assertEqual(summary["date"], "2026-05-03")
        self.assertEqual(summary["counts"]["Noise"], 1)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["jira_summary_issue_key"], "KAN-55")
        self.assertEqual(state_writes[-1]["days"]["2026-05-03"]["actions_by_type"]["ignore"], 1)

    def test_hosting_feedback_bridge_configures_mode_agnostic_feedback_endpoint(self):
        dataset_path = self.project_dir / "base.csv"
        dataset_path.write_text("LogMessage,class\nhello,Noise\n", encoding="utf-8")
        captured = {"settings": {}, "uploaded_blob": ""}

        self.azure_platform.deploy_azure_function_bridge_infrastructure = lambda **kwargs: {
            "storageConnectionString": "UseDevelopmentStorage=true",
            "storageAccountKey": "key",
            "serviceBusConnectionString": "Endpoint=sb://example/",
            "functionAppHostName": "func.azurewebsites.net",
        }
        self.azure_platform.ensure_azure_blob_datastore = lambda **kwargs: None
        self.azure_platform.upload_blob_file = lambda **kwargs: captured.update({"uploaded_blob": kwargs["blob_name"]}) or kwargs["blob_name"]
        self.azure_platform.set_function_app_settings = lambda **kwargs: captured.update({"settings": kwargs["settings"]})
        self.azure_platform.build_function_bridge_package = lambda package_name: str(dataset_path)
        self.azure_platform.upload_function_bridge_package = lambda **kwargs: "https://blob/package.zip"
        self.azure_platform.trigger_function_app_onedeploy = lambda **kwargs: None
        self.azure_platform.wait_for_function_bridge_endpoint = (
            lambda **kwargs: (f"https://func.azurewebsites.net/api/{kwargs.get('route_path', 'logs')}?code=key", "key")
        )

        request = HostingRequest(
            model_dir=str(self.project_dir),
            mode="azure",
            azure_sub_id="sub",
            azure_tenant_id="tenant",
            azure_compute="cpu",
            azure_instance_type="Standard_D2as_v4",
            azure_service="serverless",
        )
        result = self.hosting._deploy_azure_feedback_bridge(
            ctx=SimpleNamespace(emit=lambda *args, **kwargs: None),
            credential=object(),
            ml_client=object(),
            request=request,
            service_kind="serverless",
            timestamp=123,
            training_metadata={"data_version_path": str(dataset_path)},
            source_endpoint_name="serverless-endpoint",
            source_api_url="https://endpoint",
            batch_enabled=False,
        )

        self.assertEqual(result["feedback_api_url"], "https://func.azurewebsites.net/api/feedback?code=key")
        self.assertEqual(captured["settings"]["LOGMONITOR_BATCH_ENABLED"], "0")
        self.assertEqual(captured["settings"]["LOGMONITOR_HOSTING_SERVICE_KIND"], "serverless")
        self.assertEqual(captured["settings"]["LOGMONITOR_RETRAIN_ENABLED"], "1")
        self.assertEqual(captured["settings"]["LOGMONITOR_TRIAGE_ENABLED"], "0")
        self.assertEqual(captured["settings"]["LOGMONITOR_BASE_DATASET_BLOB"], "feedback/base/123/dataset.csv")
        self.assertEqual(captured["uploaded_blob"], "feedback/base/123/dataset.csv")

        self.hosting._deploy_azure_feedback_bridge(
            ctx=SimpleNamespace(emit=lambda *args, **kwargs: None),
            credential=object(),
            ml_client=object(),
            request=request,
            service_kind="queued_batch",
            timestamp=124,
            training_metadata={"data_version_path": str(dataset_path)},
            source_endpoint_name="batch-endpoint",
            source_api_url="https://batch",
            batch_enabled=True,
            batch_endpoint_name="batch-endpoint",
            batch_deployment_name="default",
            retrain_compute_name="log-monitor-batch-cpu",
        )
        self.assertEqual(captured["settings"]["LOGMONITOR_BATCH_ENABLED"], "1")
        self.assertEqual(captured["settings"]["LOGMONITOR_BATCH_ENDPOINT_NAME"], "batch-endpoint")
        self.assertEqual(captured["settings"]["LOGMONITOR_RETRAIN_COMPUTE_NAME"], "log-monitor-batch-cpu")

        triage_request = HostingRequest(
            model_dir=str(self.project_dir),
            mode="azure",
            azure_sub_id="sub",
            azure_tenant_id="tenant",
            azure_compute="cpu",
            azure_instance_type="Standard_D2as_v4",
            azure_service="online",
            github_token="gh-token",
            github_repo="owner/repo",
            github_branch="main",
            triage_enabled=True,
            configuration_email="email1@example.test",
            system_email="email2@example.test",
            acs_connection_string="endpoint=https://acs.example/;accesskey=secret",
            acs_sender_address="alerts@example.test",
            jira_site_url="https://jira.example.test",
            jira_account_email="jira@example.test",
            jira_api_token="jira-token",
            jira_project_key="OPS",
            jira_issue_type="Bug",
            jira_priority="High",
            jira_labels="log-monitor,triage",
        )
        triage_result = self.hosting._deploy_azure_feedback_bridge(
            ctx=SimpleNamespace(emit=lambda *args, **kwargs: None),
            credential=object(),
            ml_client=object(),
            request=triage_request,
            service_kind="online",
            timestamp=125,
            training_metadata={"data_version_path": str(dataset_path)},
            source_endpoint_name="online-endpoint",
            source_api_url="https://score",
            batch_enabled=False,
            prediction_key="prediction-key",
        )
        self.assertEqual(triage_result["triage_api_url"], "https://func.azurewebsites.net/api/triage?code=key")
        self.assertEqual(captured["settings"]["LOGMONITOR_TRIAGE_ENABLED"], "1")
        self.assertEqual(captured["settings"]["LOGMONITOR_PREDICTION_ENDPOINT_URL"], "https://score")
        self.assertEqual(captured["settings"]["LOGMONITOR_PREDICTION_KEY"], "prediction-key")
        self.assertEqual(captured["settings"]["LOGMONITOR_CONFIGURATION_EMAIL"], "email1@example.test")
        self.assertEqual(captured["settings"]["LOGMONITOR_SYSTEM_EMAIL"], "email2@example.test")
        self.assertEqual(captured["settings"]["LOGMONITOR_GITHUB_REPO"], "owner/repo")
        self.assertEqual(captured["settings"]["LOGMONITOR_JIRA_PROJECT_KEY"], "OPS")
        self.assertEqual(captured["settings"]["LOGMONITOR_JIRA_MONITORING_ENABLED"], "1")
        self.assertEqual(captured["settings"]["LOGMONITOR_JIRA_MONITORING_ISSUE_TYPE"], "Task")
        self.assertEqual(captured["settings"]["LOGMONITOR_MONITORING_STATE_BLOB"], "monitoring/prediction-summary-state.json")

    def test_hosting_run_blocks_when_gate_fails(self):
        request = HostingRequest(model_dir=str(self.project_dir), mode="local")
        ctx = SimpleNamespace(emit=lambda *args, **kwargs: None)
        with mock.patch.object(self.hosting, "_enforce_deployment_gate", side_effect=RuntimeError("gate failed")), mock.patch.object(
            self.hosting,
            "_run_local_hosting",
        ) as local_host_mock:
            with self.assertRaises(RuntimeError):
                self.hosting._run(ctx, request)
        local_host_mock.assert_not_called()

    def test_hosting_run_attaches_gate_summary_when_gate_passes(self):
        request = HostingRequest(model_dir=str(self.project_dir), mode="local")
        ctx = SimpleNamespace(emit=lambda *args, **kwargs: None)
        gate_payload = {
            "gate_pass": True,
            "cached": False,
            "report_path": str(self.project_dir / "outputs" / "gates" / "gate_eval_1.json"),
            "predictions_path": str(self.project_dir / "outputs" / "gates" / "gate_eval_1_predictions.csv"),
            "model_hash": "m",
            "golden_set_hash": "g",
            "policy_hash": "p",
            "sample_count": 4,
            "metrics": {"accuracy": 0.9, "weighted_f1": 0.89},
            "golden_set_path": str(self.project_dir / "gates" / "deployment_golden.csv"),
            "policy_path": str(self.project_dir / "gates" / "deployment_policy.json"),
        }
        with mock.patch.object(self.hosting, "_enforce_deployment_gate", return_value=gate_payload), mock.patch.object(
            self.hosting,
            "_run_local_hosting",
            return_value={"operation": "hosting", "message": "Local stack is ready.", "summary": "Local stack is ready."},
        ):
            result = self.hosting._run(ctx, request)

        self.assertTrue(result["gate"]["gate_pass"])
        self.assertIn("Deployment gate: PASS", result["message"])
        self.assertIn("Deployment gate: PASS", result["summary"])
        self.assertEqual(result["gate"]["sample_count"], 4)

    def test_hosting_run_skips_local_gate_for_azure_registered_model(self):
        request = HostingRequest(
            model_dir="",
            mode="azure",
            azure_service="online",
            azure_model_name="registered-log-model",
            azure_model_version="3",
        )
        emitted = []
        ctx = SimpleNamespace(emit=lambda _kind, message: emitted.append(message))
        with mock.patch.object(self.hosting, "_enforce_deployment_gate") as gate_mock, mock.patch.object(
            self.hosting,
            "_run_azure_hosting",
            return_value={"operation": "hosting", "message": "Azure endpoint is ready.", "summary": "Azure endpoint is ready."},
        ) as azure_host_mock:
            result = self.hosting._run(ctx, request)

        gate_mock.assert_not_called()
        azure_host_mock.assert_called_once()
        self.assertEqual(result["drift_monitoring"]["status"], "skipped")
        self.assertTrue(any("Skipping local deployment gate" in message for message in emitted))

    def test_enforce_deployment_gate_reuses_cached_pass(self):
        model_dir = self.project_dir / "outputs" / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "pytorch_model.bin").write_bytes(b"1")
        gates_dir = self.project_dir / "gates"
        gates_dir.mkdir(parents=True, exist_ok=True)
        golden_path = gates_dir / "deployment_golden.csv"
        policy_path = gates_dir / "deployment_policy.json"
        golden_path.write_text("LogMessage,class\nprocessed Canceled,Noise\n", encoding="utf-8")
        policy_path.write_text(
            json.dumps(
                {
                    "min_accuracy": 0.8,
                    "min_weighted_f1": 0.8,
                    "min_macro_f1": 0.8,
                    "min_recall_per_class": {"Noise": 0.8},
                }
            ),
            encoding="utf-8",
        )
        request = HostingRequest(
            model_dir=str(model_dir),
            mode="local",
            deployment_gate_golden_path=str(golden_path),
            deployment_gate_policy_path=str(policy_path),
        )
        ctx = SimpleNamespace(emit=lambda *args, **kwargs: None)

        with mock.patch("app_core.hosting_service.load_model_bundle", return_value=object()), mock.patch(
            "app_core.hosting_service.predict_error_message",
            return_value="Noise",
        ), mock.patch.object(
            self.hosting,
            "_compute_gate_metrics",
            return_value={
                "accuracy": 1.0,
                "weighted_precision": 1.0,
                "weighted_recall": 1.0,
                "weighted_f1": 1.0,
                "macro_precision": 1.0,
                "macro_recall": 1.0,
                "macro_f1": 1.0,
                "per_class": {"Noise": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}},
                "confusion_matrix": {"labels": ["Noise"], "matrix": [[1]]},
            },
        ):
            first = self.hosting._enforce_deployment_gate(ctx, request)
        self.assertTrue(first["gate_pass"])
        self.assertFalse(first["cached"])

        with mock.patch(
            "app_core.hosting_service.load_model_bundle",
            side_effect=AssertionError("load_model_bundle should not run on cache hit"),
        ):
            second = self.hosting._enforce_deployment_gate(ctx, request)
        self.assertTrue(second["gate_pass"])
        self.assertTrue(second["cached"])

    def test_enforce_deployment_gate_requires_golden_file(self):
        model_dir = self.project_dir / "outputs" / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "pytorch_model.bin").write_bytes(b"1")
        gates_dir = self.project_dir / "gates"
        gates_dir.mkdir(parents=True, exist_ok=True)
        policy_path = gates_dir / "deployment_policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "min_accuracy": 0.8,
                    "min_weighted_f1": 0.8,
                    "min_macro_f1": 0.8,
                    "min_recall_per_class": {"Noise": 0.8},
                }
            ),
            encoding="utf-8",
        )
        request = HostingRequest(
            model_dir=str(model_dir),
            mode="local",
            deployment_gate_golden_path=str(gates_dir / "missing.csv"),
            deployment_gate_policy_path=str(policy_path),
        )
        ctx = SimpleNamespace(emit=lambda *args, **kwargs: None)
        with self.assertRaises(FileNotFoundError):
            self.hosting._enforce_deployment_gate(ctx, request)

    def test_observability_evaluate_drift_for_model_writes_report(self):
        model_dir = self.project_dir / "outputs" / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "pytorch_model.bin").write_bytes(b"1")
        gates_dir = self.project_dir / "gates"
        gates_dir.mkdir(parents=True, exist_ok=True)
        golden_path = gates_dir / "drift_golden.csv"
        policy_path = gates_dir / "drift_policy.json"
        golden_path.write_text("LogMessage,class\nprocessed Canceled,Noise\n", encoding="utf-8")
        policy_path.write_text(
            json.dumps(
                {
                    "warning": {
                        "min_accuracy": 0.9,
                        "min_weighted_f1": 0.9,
                        "min_macro_f1": 0.9,
                        "min_recall_per_class": {"Noise": 0.9},
                    },
                    "critical": {
                        "min_accuracy": 0.7,
                        "min_weighted_f1": 0.7,
                        "min_macro_f1": 0.7,
                        "min_recall_per_class": {"Noise": 0.7},
                    },
                }
            ),
            encoding="utf-8",
        )
        with mock.patch("app_core.observability_service.load_model_bundle", return_value=object()), mock.patch(
            "app_core.observability_service.predict_error_message",
            return_value="Noise",
        ), mock.patch.object(
            self.observability,
            "compute_drift_metrics",
            return_value={
                "accuracy": 1.0,
                "weighted_precision": 1.0,
                "weighted_recall": 1.0,
                "weighted_f1": 1.0,
                "macro_precision": 1.0,
                "macro_recall": 1.0,
                "macro_f1": 1.0,
                "per_class": {"Noise": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}},
                "confusion_matrix": {"labels": ["Noise"], "matrix": [[1]]},
            },
        ):
            payload = self.observability.evaluate_drift_for_model(
                model_dir=str(model_dir),
                golden_path=str(golden_path),
                policy_path=str(policy_path),
                deployment_id="unit-deploy",
                mode="local",
                service_kind="local",
                source="unit-test",
            )
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(Path(payload["report_path"]).exists())
        self.assertTrue(Path(payload["predictions_path"]).exists())
        latest_path = Path(payload["report_path"]).parent / "latest_drift_eval.json"
        self.assertTrue(latest_path.exists())

    def test_hosting_run_attaches_drift_monitoring_result(self):
        request = HostingRequest(
            model_dir=str(self.project_dir),
            mode="local",
            drift_golden_path=str(self.project_dir / "gates" / "drift_golden.csv"),
            drift_policy_path=str(self.project_dir / "gates" / "drift_policy.json"),
        )
        ctx = SimpleNamespace(emit=lambda *args, **kwargs: None)
        gate_payload = {
            "gate_pass": True,
            "cached": False,
            "report_path": str(self.project_dir / "outputs" / "gates" / "gate_eval_1.json"),
            "predictions_path": str(self.project_dir / "outputs" / "gates" / "gate_eval_1_predictions.csv"),
            "model_hash": "m",
            "golden_set_hash": "g",
            "policy_hash": "p",
            "sample_count": 4,
            "metrics": {"accuracy": 0.9, "weighted_f1": 0.89},
            "golden_set_path": str(self.project_dir / "gates" / "deployment_golden.csv"),
            "policy_path": str(self.project_dir / "gates" / "deployment_policy.json"),
        }
        drift_payload = {
            "status": "warning",
            "created_at": "2026-04-30T00:00:00Z",
            "report_path": str(self.project_dir / "outputs" / "drift_monitoring" / "report.json"),
            "predictions_path": str(self.project_dir / "outputs" / "drift_monitoring" / "predictions.csv"),
            "golden_set_path": str(self.project_dir / "gates" / "drift_golden.csv"),
            "golden_set_hash": "dg",
            "policy_path": str(self.project_dir / "gates" / "drift_policy.json"),
            "policy_hash": "dp",
            "sample_count": 8,
            "metrics": {"accuracy": 0.84, "weighted_f1": 0.83},
            "warning_failures": ["warning.accuracy 0.8400 < warning.min_accuracy 0.8800"],
            "critical_failures": [],
        }
        with mock.patch.object(self.hosting, "_enforce_deployment_gate", return_value=gate_payload), mock.patch.object(
            self.hosting,
            "_run_local_hosting",
            return_value={"operation": "hosting", "message": "Local stack is ready.", "summary": "Local stack is ready."},
        ), mock.patch.object(self.observability, "evaluate_drift_for_model", return_value=drift_payload):
            result = self.hosting._run(ctx, request)

        self.assertEqual(result["drift_monitoring"]["status"], "warning")
        self.assertIn("Drift monitoring baseline: WARNING", result["summary"])
        metadata = self.model_catalog.read_last_hosting_metadata()
        self.assertEqual(metadata.get("drift_monitoring", {}).get("status"), "warning")

    def test_hosting_run_keeps_success_when_drift_monitoring_errors(self):
        request = HostingRequest(
            model_dir=str(self.project_dir),
            mode="local",
            drift_golden_path=str(self.project_dir / "gates" / "drift_golden.csv"),
            drift_policy_path=str(self.project_dir / "gates" / "drift_policy.json"),
        )
        ctx = SimpleNamespace(emit=lambda *args, **kwargs: None)
        gate_payload = {
            "gate_pass": True,
            "cached": False,
            "report_path": str(self.project_dir / "outputs" / "gates" / "gate_eval_1.json"),
            "predictions_path": str(self.project_dir / "outputs" / "gates" / "gate_eval_1_predictions.csv"),
            "model_hash": "m",
            "golden_set_hash": "g",
            "policy_hash": "p",
            "sample_count": 4,
            "metrics": {"accuracy": 0.9, "weighted_f1": 0.89},
            "golden_set_path": str(self.project_dir / "gates" / "deployment_golden.csv"),
            "policy_path": str(self.project_dir / "gates" / "deployment_policy.json"),
        }
        with mock.patch.object(self.hosting, "_enforce_deployment_gate", return_value=gate_payload), mock.patch.object(
            self.hosting,
            "_run_local_hosting",
            return_value={"operation": "hosting", "message": "Local stack is ready.", "summary": "Local stack is ready."},
        ), mock.patch.object(
            self.observability,
            "evaluate_drift_for_model",
            side_effect=RuntimeError("drift failed"),
        ):
            result = self.hosting._run(ctx, request)

        self.assertEqual(result["operation"], "hosting")
        self.assertIn("drift failed", result.get("drift_monitoring_error", ""))

    def test_azure_platform_lists_workspace_models_for_hosting_picker(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        created_at = "2026-05-01T12:30:00Z"
        fake_models = [
            SimpleNamespace(
                id="azureml:log-model:2",
                name="log-model",
                version="2",
                path="azureml://models/log-model/versions/2",
                type="custom_model",
                description="candidate",
                creation_context=SimpleNamespace(created_at=created_at),
            ),
            SimpleNamespace(
                id="azureml:log-model:2",
                name="log-model",
                version="2",
                path="duplicate",
                type="custom_model",
                description="duplicate",
                creation_context=SimpleNamespace(created_at=created_at),
            ),
        ]
        ml_client = SimpleNamespace(models=SimpleNamespace(list=lambda: fake_models))

        models = service.list_azure_workspace_models(ml_client)

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "azureml:log-model:2")
        self.assertEqual(models[0]["name"], "log-model")
        self.assertEqual(models[0]["version"], "2")
        self.assertIn("log-model:2", models[0]["label"])
        self.assertIn("2026-05-01 12:30:00", models[0]["label"])

    def test_azure_platform_expands_model_containers_to_versions(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        container = SimpleNamespace(
            id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws/models/log-model",
            name="log-model",
            version="",
            type="custom_model",
        )
        versions = [
            SimpleNamespace(
                id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws/models/log-model/versions/3",
                name="log-model",
                version="3",
                type="custom_model",
            )
        ]

        class FakeModels:
            def list(self, name=None):
                return versions if name == "log-model" else [container]

        models = service.list_azure_workspace_models(SimpleNamespace(models=FakeModels()))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "log-model")
        self.assertEqual(models[0]["version"], "3")
        self.assertIn("log-model:3", models[0]["label"])

    def test_azure_platform_parses_model_name_version_from_asset_ids(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        self.assertEqual(
            service.parse_azure_model_name_version_from_id("azureml:log-model:7"),
            ("log-model", "7"),
        )
        self.assertEqual(
            service.parse_azure_model_name_version_from_id(
                "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws/models/log-model/versions/8"
            ),
            ("log-model", "8"),
        )
        self.assertEqual(
            service.parse_azure_model_name_version_from_id(
                "azureml://registries/azureml/models/Phi-4-mini-instruct/versions/12"
            ),
            ("Phi-4-mini-instruct", "12"),
        )

    def test_azure_platform_resolves_model_from_full_asset_id(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        captured = {}

        def fake_get(name, version):
            captured["name"] = name
            captured["version"] = version
            return SimpleNamespace(name=name, version=version)

        ml_client = SimpleNamespace(models=SimpleNamespace(get=fake_get))

        model = service.resolve_azure_model_asset(
            ml_client,
            model_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws/models/log-model/versions/9",
        )

        self.assertEqual(model.name, "log-model")
        self.assertEqual(model.version, "9")
        self.assertEqual(captured, {"name": "log-model", "version": "9"})

    def test_azure_platform_reads_online_endpoint_key(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        calls = []

        class FakeOnlineEndpoints:
            def get_keys(self, name):
                calls.append(name)
                return SimpleNamespace(primary_key="primary-secret")

        ml_client = SimpleNamespace(online_endpoints=FakeOnlineEndpoints())

        self.assertEqual(service.get_online_endpoint_key(ml_client, "endpoint-a"), "primary-secret")
        self.assertEqual(calls, ["endpoint-a"])

    def test_azure_platform_lists_acs_email_senders(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        requested_urls = []

        def fake_management_request(credential, method, url, payload=None, expected_statuses=(200, 201, 202)):
            requested_urls.append(url)
            if "/providers/Microsoft.Communication/emailServices?" in url:
                return {
                    "value": [
                        {
                            "id": "/subscriptions/sub/resourceGroups/email-rg/providers/Microsoft.Communication/EmailServices/email-service",
                            "name": "email-service",
                        }
                    ]
                }
            if "/domains?" in url:
                return {
                    "value": [
                        {
                            "name": "contoso.com",
                            "properties": {"mailFromSenderDomain": "mail.contoso.com"},
                        }
                    ]
                }
            if "/senderUsernames?" in url:
                return {
                    "value": [
                        {
                            "name": "alerts",
                            "properties": {"username": "alerts", "displayName": "Alerts"},
                        },
                        {
                            "name": "noreply",
                            "properties": {"username": "noreply"},
                        },
                    ]
                }
            return {"value": []}

        service.azure_management_json_request = fake_management_request

        senders = service.list_acs_email_senders(credential=object(), sub_id="sub")

        self.assertEqual([sender["address"] for sender in senders], ["alerts@mail.contoso.com", "noreply@mail.contoso.com"])
        self.assertEqual(senders[0]["display_name"], "Alerts")
        self.assertEqual(senders[0]["email_service"], "email-service")
        self.assertTrue(any("/senderUsernames?" in url for url in requested_urls))
        self.assertTrue(all("api-version=2025-09-01" in url for url in requested_urls))

    def test_azure_platform_creates_default_acs_email_sender_when_missing(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        service.ensure_resource_group = lambda credential, sub_id, resource_group="", location="eastus": resource_group or "rg"
        calls = []

        def fake_management_request(credential, method, url, payload=None, expected_statuses=(200, 201, 202)):
            calls.append({"method": method, "url": url, "payload": payload})
            if method == "GET" and "/emailServices?" in url:
                return {"value": []}
            if method == "GET" and "/communicationServices?" in url:
                return {"value": []}
            if method == "PUT" and "/emailServices/" in url and "/domains/" not in url:
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Communication/emailServices/log-monitor-email",
                    "name": "log-monitor-email",
                }
            if method == "PUT" and "/domains/AzureManagedDomain" in url and "/senderUsernames/" not in url:
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Communication/emailServices/log-monitor-email/domains/AzureManagedDomain",
                    "name": "AzureManagedDomain",
                    "properties": {"mailFromSenderDomain": "abc.azurecomm.net"},
                }
            if method == "PUT" and "/senderUsernames/DoNotReply" in url:
                return {
                    "name": "DoNotReply",
                    "properties": {"username": "DoNotReply", "displayName": "Log Monitor"},
                }
            if method == "PUT" and "/communicationServices/" in url:
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Communication/communicationServices/log-monitor-acs",
                    "name": "log-monitor-acs",
                    "properties": {"linkedDomains": payload["properties"].get("linkedDomains", [])},
                }
            return {"value": []}

        service.azure_management_json_request = fake_management_request

        senders = service.list_acs_email_senders(credential=object(), sub_id="sub")

        self.assertEqual(senders[0]["address"], "DoNotReply@abc.azurecomm.net")
        self.assertTrue(any(call["method"] == "PUT" and "/emailServices/" in call["url"] for call in calls))
        self.assertTrue(any(call["method"] == "PUT" and "/domains/AzureManagedDomain" in call["url"] for call in calls))
        self.assertTrue(any(call["method"] == "PUT" and "/senderUsernames/DoNotReply" in call["url"] for call in calls))
        self.assertTrue(any(call["method"] == "PUT" and "/communicationServices/" in call["url"] for call in calls))

    def test_azure_platform_lists_acs_connection_strings(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        requested = []

        def fake_management_request(credential, method, url, payload=None, expected_statuses=(200, 201, 202)):
            requested.append((method, url))
            if "/providers/Microsoft.Communication/communicationServices?" in url:
                return {
                    "value": [
                        {
                            "id": "/subscriptions/sub/resourceGroups/acs-rg/providers/Microsoft.Communication/communicationServices/acs-service",
                            "name": "acs-service",
                            "properties": {"hostName": "acs-service.communication.azure.com"},
                        }
                    ]
                }
            if "/listKeys?" in url:
                return {
                    "primaryConnectionString": "endpoint=https://acs-service.communication.azure.com/;accesskey=primary",
                    "secondaryKey": "secondary",
                }
            return {"value": []}

        service.azure_management_json_request = fake_management_request

        connections = service.list_acs_connection_strings(credential=object(), sub_id="sub")

        self.assertEqual(len(connections), 2)
        self.assertEqual(connections[0]["label"], "acs-rg/acs-service | Primary")
        self.assertEqual(connections[0]["connection_string"], "endpoint=https://acs-service.communication.azure.com/;accesskey=primary")
        self.assertEqual(connections[1]["label"], "acs-rg/acs-service | Secondary")
        self.assertEqual(connections[1]["connection_string"], "endpoint=https://acs-service.communication.azure.com/;accesskey=secondary")
        self.assertTrue(any(method == "POST" and "/listKeys?" in url for method, url in requested))

    def test_azure_platform_creates_default_acs_connection_when_missing(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        service.ensure_resource_group = lambda credential, sub_id, resource_group="", location="eastus": resource_group or "rg"
        calls = []

        def fake_management_request(credential, method, url, payload=None, expected_statuses=(200, 201, 202)):
            calls.append({"method": method, "url": url, "payload": payload})
            if method == "GET" and "/communicationServices?" in url:
                return {"value": []}
            if method == "PUT" and "/communicationServices/" in url:
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Communication/communicationServices/log-monitor-acs",
                    "name": "log-monitor-acs",
                    "properties": {"hostName": "log-monitor-acs.communication.azure.com"},
                }
            if method == "POST" and "/listKeys?" in url:
                return {
                    "primaryKey": "primary",
                    "secondaryKey": "secondary",
                }
            return {"value": []}

        service.azure_management_json_request = fake_management_request

        connections = service.list_acs_connection_strings(credential=object(), sub_id="sub")

        self.assertEqual(connections[0]["connection_string"], "endpoint=https://log-monitor-acs.communication.azure.com/;accesskey=primary")
        self.assertTrue(any(call["method"] == "PUT" and "/communicationServices/" in call["url"] for call in calls))
        self.assertTrue(any(call["method"] == "POST" and "/listKeys?" in call["url"] for call in calls))

    def test_azure_platform_waits_for_acs_provisioning_before_keys(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None
        service.ensure_resource_group = lambda credential, sub_id, resource_group="", location="eastus": resource_group or "rg"
        state_checks = {"resource": 0, "keys": 0}

        def fake_management_request(credential, method, url, payload=None, expected_statuses=(200, 201, 202)):
            if method == "GET" and "/providers/Microsoft.Communication/communicationServices?" in url:
                return {"value": []}
            if method == "PUT" and "/communicationServices/" in url:
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Communication/communicationServices/log-monitor-acs",
                    "name": "log-monitor-acs",
                    "properties": {
                        "hostName": "log-monitor-acs.communication.azure.com",
                        "provisioningState": "Accepted",
                    },
                }
            if method == "GET" and "/communicationServices/log-monitor-acs" in url:
                state_checks["resource"] += 1
                state = "Accepted" if state_checks["resource"] == 1 else "Succeeded"
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Communication/communicationServices/log-monitor-acs",
                    "name": "log-monitor-acs",
                    "properties": {
                        "hostName": "log-monitor-acs.communication.azure.com",
                        "provisioningState": state,
                    },
                }
            if method == "POST" and "/listKeys?" in url:
                state_checks["keys"] += 1
                if state_checks["keys"] == 1:
                    raise RuntimeError(
                        "Azure management request failed (409 Conflict).\n\n"
                        "{'error': {'code': 'InvalidResourceOperation', 'message': "
                        "\"The operation for resource is invalid as it is being provisioned with state: 'Accepted'.\"}}"
                    )
                return {"primaryKey": "primary"}
            return {"value": []}

        service.azure_management_json_request = fake_management_request

        with mock.patch("app_core.azure_platform_service.time.sleep", lambda _seconds: None):
            connections = service.list_acs_connection_strings(credential=object(), sub_id="sub")

        self.assertEqual(connections[0]["connection_string"], "endpoint=https://log-monitor-acs.communication.azure.com/;accesskey=primary")
        self.assertEqual(state_checks["resource"], 2)
        self.assertEqual(state_checks["keys"], 2)

    def test_azure_platform_deploys_serverless_endpoint_from_catalog_model_id(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        service.ensure_azure_dependencies = lambda: None

        class FakeServerlessEndpoint:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakePoller:
            def result(self, timeout=None):
                return SimpleNamespace(provisioning_state="Succeeded")

        class FakeServerlessEndpointOperations:
            def __init__(self):
                self.created = None

            def begin_create_or_update(self, endpoint):
                self.created = endpoint
                return FakePoller()

            def get(self, name):
                return SimpleNamespace(
                    name=name,
                    provisioning_state="Succeeded",
                    scoring_uri=f"https://{name}.eastus.inference.ai.azure.com/v1/chat/completions",
                    auth_mode="key",
                )

            def list(self):
                return [
                    SimpleNamespace(
                        name=self.created.name,
                        model_id=self.created.model_id,
                        provisioning_state="Succeeded",
                        scoring_uri=f"https://{self.created.name}.eastus.inference.ai.azure.com/v1/chat/completions",
                        auth_mode="key",
                    )
                ]

        fake_ops = FakeServerlessEndpointOperations()
        fake_client = SimpleNamespace(serverless_endpoints=fake_ops)

        self.assertEqual(service.get_default_serverless_model_id(), "azureml://registries/azureml/models/Phi-4-mini-instruct")
        self.assertEqual(
            service.build_default_serverless_endpoint_name(
                "azureml://registries/azureml/models/Phi-4-mini-instruct/versions/1",
                suffix="1234567890",
            ),
            "log-monitor-phi-4-mini-instruct-1234567890",
        )

        with mock.patch("app_core.azure_platform_service.ServerlessEndpoint", FakeServerlessEndpoint):
            result = service.deploy_azure_serverless_endpoint(
                fake_client,
                model_id="azureml://registries/azureml/models/Phi-3-mini/versions/7",
                endpoint_name="1 Custom Endpoint",
            )

        self.assertEqual(fake_ops.created.name, "log-monitor-1-custom-endpoint")
        self.assertEqual(fake_ops.created.model_id, "azureml://registries/azureml/models/Phi-3-mini")
        self.assertEqual(fake_ops.created.auth_mode, "key")
        self.assertEqual(result["endpoint_name"], "log-monitor-1-custom-endpoint")
        self.assertIn("/v1/chat/completions", result["api_url"])
        self.assertTrue(result["visible_in_workspace_list"])
        self.assertEqual(result["workspace_serverless_endpoint_names"], ["log-monitor-1-custom-endpoint"])

    def test_azure_platform_creates_serverless_endpoint_with_documented_arm_shape(self):
        service = AzurePlatformService(str(self.project_dir), resource_group="rg", workspace_name="ws")
        calls = []

        def fake_management_request(credential, method, url, payload=None, expected_statuses=(200, 201, 202)):
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "payload": payload,
                    "expected_statuses": expected_statuses,
                }
            )
            if method == "GET":
                return {
                    "id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws/serverlessEndpoints/log-monitor-phi",
                    "name": "log-monitor-phi",
                    "type": "Microsoft.MachineLearningServices/workspaces/serverlessEndpoints",
                    "location": "eastus",
                    "kind": "ServerlessEndpoint",
                    "sku": {"name": "Consumption"},
                    "properties": {
                        "provisioningState": "Succeeded",
                        "authMode": "Key",
                        "modelSettings": {"modelId": "azureml://registries/azureml/models/Phi-4-mini-instruct"},
                    },
                }
            return {}

        service.azure_management_json_request = fake_management_request
        resource = service.create_arm_serverless_endpoint(
            credential=object(),
            sub_id="sub",
            endpoint_name="log-monitor-phi",
            model_id="azureml://registries/azureml/models/Phi-4-mini-instruct",
            location="eastus",
        )

        put_call = calls[0]
        self.assertEqual(put_call["method"], "PUT")
        self.assertIn("api-version=2024-04-01-preview", put_call["url"])
        self.assertEqual(put_call["payload"]["kind"], "ServerlessEndpoint")
        self.assertEqual(put_call["payload"]["sku"], {"name": "Consumption"})
        self.assertEqual(put_call["payload"]["properties"]["authMode"], "Key")
        self.assertEqual(
            put_call["payload"]["properties"]["modelSettings"]["modelId"],
            "azureml://registries/azureml/models/Phi-4-mini-instruct",
        )
        self.assertEqual(service.get_arm_provisioning_state(resource), "Succeeded")


if __name__ == "__main__":
    unittest.main()
