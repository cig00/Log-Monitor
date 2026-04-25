from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app_core.azure_platform_service import AzurePlatformService
from app_core.data_prep_service import DataPrepService
from app_core.github_service import GitHubService
from app_core.mlops_service import MlopsService
from app_core.model_catalog_service import ModelCatalogService
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


if __name__ == "__main__":
    unittest.main()
