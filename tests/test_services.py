from __future__ import annotations

import json
import tempfile
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
