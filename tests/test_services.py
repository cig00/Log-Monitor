from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_core.github_service import GitHubService
from app_core.mlops_service import MlopsService
from app_core.model_catalog_service import ModelCatalogService
from app_core.runtime import ArtifactStore


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        (self.project_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "prompt.txt").write_text("prompt body", encoding="utf-8")
        self.artifact_store = ArtifactStore(str(self.project_dir))
        self.model_catalog = ModelCatalogService(str(self.project_dir), self.artifact_store)
        self.mlops = MlopsService(
            str(self.project_dir),
            self.artifact_store,
            self.model_catalog,
            resource_group="rg",
            workspace_name="ws",
            local_tracking_uri=str(self.project_dir / "mlruns"),
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


if __name__ == "__main__":
    unittest.main()
