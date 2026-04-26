from __future__ import annotations

import difflib
import html
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_utils import (
    clean_optional_string,
    dataframe_metadata,
    dataframe_sample,
    discover_model_dir,
    now_utc_iso,
    prompt_sha256,
    read_json,
    read_sidecar_for_csv,
    safe_prompt_preview,
    write_json,
    write_sidecar_for_csv,
)

from .contracts import MlflowConfig
from .model_catalog_service import ModelCatalogService
from .runtime import ArtifactStore

try:
    import mlflow
except Exception:
    mlflow = None


class MlopsService:
    MLOPS_ENV_VARS = [
        "MLOPS_ENABLED",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_PIPELINE_ID",
        "MLFLOW_PARENT_RUN_ID",
        "MLFLOW_RUN_SOURCE",
        "MLFLOW_TAGS_JSON",
    ]

    def __init__(
        self,
        project_dir: str,
        artifact_store: ArtifactStore,
        model_catalog_service: ModelCatalogService,
        resource_group: str,
        workspace_name: str,
        local_tracking_uri: str,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.artifact_store = artifact_store
        self.model_catalog_service = model_catalog_service
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.local_tracking_uri = clean_optional_string(local_tracking_uri)
        self.azure_tracking_uri = ""
        self.mlflow_ui_process = None

    def load_prompt(self) -> str:
        prompt_path = self.project_dir / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError("Could not find 'prompt.txt' in the application directory.")
        return prompt_path.read_text(encoding="utf-8")

    def get_prompt_versions_root(self) -> Path:
        root = self.project_dir / "outputs" / "prompt_versions"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list_prompt_versions(self) -> list[dict]:
        root = self.get_prompt_versions_root()
        versions: list[dict] = []
        for metadata_path in root.glob("*/metadata.json"):
            payload = read_json(str(metadata_path)) or {}
            if payload:
                payload.setdefault("prompt_metadata_path", str(metadata_path.resolve()))
                versions.append(payload)
        return sorted(
            versions,
            key=lambda item: (
                clean_optional_string(item.get("created_at")),
                clean_optional_string(item.get("prompt_version_id")),
            ),
        )

    def read_prompt_version_text(self, prompt_version_id: str) -> str:
        clean_version = clean_optional_string(prompt_version_id)
        if not clean_version:
            raise ValueError("Prompt version id is empty.")
        matches = [
            version
            for version in self.list_prompt_versions()
            if clean_optional_string(version.get("prompt_version_id")).startswith(clean_version)
            or clean_optional_string(version.get("prompt_hash")).startswith(clean_version)
        ]
        if not matches:
            raise FileNotFoundError(f"Prompt version not found: {prompt_version_id}")
        if len(matches) > 1 and all(
            clean_optional_string(match.get("prompt_version_id")) != clean_version
            and clean_optional_string(match.get("prompt_hash")) != clean_version
            for match in matches
        ):
            raise ValueError(f"Prompt version prefix is ambiguous: {prompt_version_id}")
        prompt_path = clean_optional_string(matches[-1].get("prompt_version_path"))
        if not prompt_path:
            raise FileNotFoundError(f"Prompt version has no prompt artifact path: {prompt_version_id}")
        return Path(prompt_path).read_text(encoding="utf-8")

    def build_prompt_diff(self, old_prompt: str, new_prompt: str, old_label: str, new_label: str) -> str:
        diff_lines = difflib.unified_diff(
            old_prompt.splitlines(),
            new_prompt.splitlines(),
            fromfile=old_label,
            tofile=new_label,
            lineterm="",
        )
        return "\n".join(diff_lines)

    def compare_prompt_versions(self, old_version_id: str = "", new_version_id: str = "") -> dict:
        versions = self.list_prompt_versions()
        if not versions:
            return {"message": "No prompt versions have been archived yet.", "diff": "", "old": {}, "new": {}}
        if not new_version_id:
            new = versions[-1]
        else:
            new = next(
                (
                    version
                    for version in versions
                    if clean_optional_string(version.get("prompt_version_id")).startswith(clean_optional_string(new_version_id))
                    or clean_optional_string(version.get("prompt_hash")).startswith(clean_optional_string(new_version_id))
                ),
                None,
            )
            if new is None:
                raise FileNotFoundError(f"Prompt version not found: {new_version_id}")

        if not old_version_id:
            earlier_versions = [
                version
                for version in versions
                if clean_optional_string(version.get("prompt_version_id")) != clean_optional_string(new.get("prompt_version_id"))
            ]
            if not earlier_versions:
                return {"message": "Only one prompt version is available.", "diff": "", "old": {}, "new": new}
            old = earlier_versions[-1]
        else:
            old = next(
                (
                    version
                    for version in versions
                    if clean_optional_string(version.get("prompt_version_id")).startswith(clean_optional_string(old_version_id))
                    or clean_optional_string(version.get("prompt_hash")).startswith(clean_optional_string(old_version_id))
                ),
                None,
            )
            if old is None:
                raise FileNotFoundError(f"Prompt version not found: {old_version_id}")

        old_text = self.read_prompt_version_text(clean_optional_string(old.get("prompt_version_id")))
        new_text = self.read_prompt_version_text(clean_optional_string(new.get("prompt_version_id")))
        old_label = f"prompt:{clean_optional_string(old.get('prompt_version_id'))[:12]}"
        new_label = f"prompt:{clean_optional_string(new.get('prompt_version_id'))[:12]}"
        diff_text = self.build_prompt_diff(old_text, new_text, old_label, new_label)
        return {"message": "", "diff": diff_text, "old": old, "new": new}

    def archive_prompt_version(self, prompt_text: str, metadata: dict | None = None) -> dict:
        prompt_hash = prompt_sha256(prompt_text)
        root = self.get_prompt_versions_root()
        version_dir = root / prompt_hash
        version_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = version_dir / "prompt.txt"
        metadata_path = version_dir / "metadata.json"
        comparison_path = version_dir / "comparison_from_previous.diff"

        existing_metadata = read_json(str(metadata_path)) or {}
        previous_versions = [
            version
            for version in self.list_prompt_versions()
            if clean_optional_string(version.get("prompt_version_id")) != prompt_hash
        ]
        previous_version = previous_versions[-1] if previous_versions else {}
        previous_version_id = clean_optional_string(previous_version.get("prompt_version_id"))

        prompt_path.write_text(prompt_text, encoding="utf-8")
        payload = {
            "prompt_version_id": prompt_hash,
            "prompt_hash": prompt_hash,
            "prompt_version_label": prompt_hash[:12],
            "prompt_version_dir": str(version_dir.resolve()),
            "prompt_version_path": str(prompt_path.resolve()),
            "prompt_metadata_path": str(metadata_path.resolve()),
            "prompt_comparison_path": str(comparison_path.resolve()) if previous_version_id else "",
            "previous_prompt_version_id": previous_version_id,
            "created_at": clean_optional_string(existing_metadata.get("created_at")) or now_utc_iso(),
            "source_prompt_path": str((self.project_dir / "prompt.txt").resolve()),
            "char_count": len(prompt_text),
            "line_count": len(prompt_text.splitlines()),
        }
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if value is None:
                    continue
                payload[clean_optional_string(key)] = value

        if previous_version_id:
            previous_prompt = self.read_prompt_version_text(previous_version_id)
            diff_text = self.build_prompt_diff(
                previous_prompt,
                prompt_text,
                f"prompt:{previous_version_id[:12]}",
                f"prompt:{prompt_hash[:12]}",
            )
            comparison_path.write_text(diff_text or "No textual prompt changes.\n", encoding="utf-8")
            (root / "latest_comparison.diff").write_text(diff_text or "No textual prompt changes.\n", encoding="utf-8")

        write_json(str(metadata_path), payload)
        return payload

    def get_copilot_pr_prompts_root(self) -> Path:
        root = self.project_dir / "outputs" / "copilot_pr_prompts"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list_copilot_pr_prompt_versions(self) -> list[dict]:
        root = self.get_copilot_pr_prompts_root()
        versions: list[dict] = []
        for metadata_path in root.glob("*/metadata.json"):
            payload = read_json(str(metadata_path)) or {}
            if payload:
                payload.setdefault("copilot_prompt_metadata_path", str(metadata_path.resolve()))
                versions.append(payload)
        return sorted(
            versions,
            key=lambda item: (
                clean_optional_string(item.get("created_at")),
                clean_optional_string(item.get("copilot_prompt_version_id")),
            ),
        )

    def archive_copilot_pr_prompt(self, prompt_text: str, metadata: dict | None = None) -> dict:
        prompt_hash = prompt_sha256(prompt_text)
        root = self.get_copilot_pr_prompts_root()
        version_dir = root / prompt_hash
        version_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = version_dir / "copilot_prompt.txt"
        metadata_path = version_dir / "metadata.json"
        comparison_path = version_dir / "comparison_from_previous.diff"

        existing_metadata = read_json(str(metadata_path)) or {}
        previous_versions = [
            version
            for version in self.list_copilot_pr_prompt_versions()
            if clean_optional_string(version.get("copilot_prompt_version_id")) != prompt_hash
        ]
        previous_version = previous_versions[-1] if previous_versions else {}
        previous_version_id = clean_optional_string(previous_version.get("copilot_prompt_version_id"))

        prompt_path.write_text(prompt_text, encoding="utf-8")
        payload = {
            "copilot_prompt_version_id": prompt_hash,
            "copilot_prompt_hash": prompt_hash,
            "copilot_prompt_version_label": prompt_hash[:12],
            "copilot_prompt_version_dir": str(version_dir.resolve()),
            "copilot_prompt_path": str(prompt_path.resolve()),
            "copilot_prompt_metadata_path": str(metadata_path.resolve()),
            "copilot_prompt_comparison_path": str(comparison_path.resolve()) if previous_version_id else "",
            "previous_copilot_prompt_version_id": previous_version_id,
            "created_at": clean_optional_string(existing_metadata.get("created_at")) or now_utc_iso(),
            "char_count": len(prompt_text),
            "line_count": len(prompt_text.splitlines()),
        }
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if value is None or clean_optional_string(key) == "prompt_text":
                    continue
                payload[clean_optional_string(key)] = value

        if previous_version_id:
            previous_prompt_path = clean_optional_string(previous_version.get("copilot_prompt_path"))
            previous_prompt = Path(previous_prompt_path).read_text(encoding="utf-8") if previous_prompt_path else ""
            diff_text = self.build_prompt_diff(
                previous_prompt,
                prompt_text,
                f"copilot-pr:{previous_version_id[:12]}",
                f"copilot-pr:{prompt_hash[:12]}",
            )
            comparison_path.write_text(diff_text or "No textual Copilot prompt changes.\n", encoding="utf-8")
            (root / "latest_comparison.diff").write_text(diff_text or "No textual Copilot prompt changes.\n", encoding="utf-8")

        write_json(str(metadata_path), payload)
        return payload

    def resolve_azure_mlflow_tracking_uri(self, ml_client: Any | None) -> str:
        if self.azure_tracking_uri:
            return self.azure_tracking_uri
        if ml_client is None:
            return ""
        try:
            workspace = ml_client.workspaces.get(self.workspace_name)
            uri = clean_optional_string(getattr(workspace, "mlflow_tracking_uri", ""))
            if uri:
                self.azure_tracking_uri = uri
            return uri
        except Exception:
            traceback.print_exc()
            return ""

    def resolve_mlflow_config(
        self,
        *,
        enabled: bool,
        backend: str,
        experiment_name: str,
        registered_model_name: str,
        tracking_uri: str,
        require_tracking_uri: bool,
        ml_client: Any | None = None,
        soft_disable: bool = False,
    ) -> tuple[MlflowConfig, str | None]:
        config = MlflowConfig(
            enabled=enabled,
            backend=backend,
            tracking_uri="",
            experiment_name=clean_optional_string(experiment_name),
            registered_model_name=clean_optional_string(registered_model_name),
        )
        if not enabled:
            return config, None

        def soft_disabled(reason: str) -> tuple[MlflowConfig, None]:
            return (
                MlflowConfig(
                    enabled=False,
                    backend=config.backend,
                    tracking_uri="",
                    experiment_name=config.experiment_name,
                    registered_model_name=config.registered_model_name,
                    disabled_reason=reason,
                ),
                None,
            )

        if mlflow is None:
            if soft_disable:
                return soft_disabled("MLflow is enabled in UI, but the `mlflow` package is not available.")
            return config, "MLflow is enabled in UI, but the `mlflow` package is not available."

        if not config.experiment_name:
            error = (
                "MLflow is enabled, but `Experiment Name` is empty.\n\n"
                "Set a non-empty value in Hosting:\n"
                "- Experiment Name (example: `log-monitor`)"
            )
            if soft_disable:
                return soft_disabled(error)
            return config, error

        if backend == "local":
            config.tracking_uri = self.local_tracking_uri
        elif backend == "custom_uri":
            config.tracking_uri = clean_optional_string(tracking_uri)
            if require_tracking_uri and not config.tracking_uri:
                if soft_disable:
                    return soft_disabled("MLflow backend is `custom_uri` but Tracking URI is empty.")
                return config, "MLflow backend is `custom_uri` but Tracking URI is empty."
        elif backend == "azure":
            config.tracking_uri = self.resolve_azure_mlflow_tracking_uri(ml_client)
            if not config.tracking_uri:
                current_value = clean_optional_string(tracking_uri)
                if current_value and "resolved during Azure run" not in current_value:
                    config.tracking_uri = current_value
            if require_tracking_uri and not config.tracking_uri:
                error = "Azure MLflow URI is not resolved yet. Run Azure auth or provide `custom_uri` backend."
                if soft_disable:
                    return soft_disabled(error)
                return config, error
        else:
            if soft_disable:
                return soft_disabled(f"Unsupported MLflow backend: {backend}")
            return config, f"Unsupported MLflow backend: {backend}"
        return config, None

    def sidecar_matches_mlflow_target(self, sidecar: dict, mlflow_config: MlflowConfig) -> bool:
        if not mlflow_config.enabled:
            return True
        current_tracking_uri = clean_optional_string(mlflow_config.tracking_uri)
        current_experiment = clean_optional_string(mlflow_config.experiment_name)
        recorded_tracking_uri = clean_optional_string(
            sidecar.get("training_tracking_uri") or sidecar.get("tracking_uri") or sidecar.get("data_prep_tracking_uri")
        )
        recorded_experiment = clean_optional_string(
            sidecar.get("training_experiment_name") or sidecar.get("experiment_name") or sidecar.get("data_prep_experiment_name")
        )
        if recorded_tracking_uri and current_tracking_uri and recorded_tracking_uri != current_tracking_uri:
            return False
        if recorded_experiment and current_experiment and recorded_experiment != current_experiment:
            return False
        return True

    def build_training_mlflow_env(
        self,
        mlflow_config: MlflowConfig,
        pipeline_context: dict,
        run_source: str,
        environment_mode: str,
    ) -> dict[str, str]:
        env = {key: "" for key in self.MLOPS_ENV_VARS}
        tags = {
            "run_type": "training",
            "pipeline_id": str(pipeline_context.get("pipeline_id", "")),
            "run_source": run_source,
            "environment_mode": environment_mode or "unknown",
        }
        for key in (
            "data_prep_run_id",
            "prompt_hash",
            "llm_model",
            "data_prep_tracking_uri",
            "data_prep_experiment_name",
            "data_version_id",
            "data_version_dir",
            "data_version_path",
            "prompt_version_id",
            "prompt_version_label",
            "prompt_version_dir",
            "prompt_version_path",
            "prompt_metadata_path",
            "prompt_comparison_path",
            "previous_prompt_version_id",
            "azure_data_asset_name",
            "azure_data_asset_version",
            "azure_data_asset_uri",
            "azure_data_asset_id",
            "azure_data_asset_path",
        ):
            value = clean_optional_string(pipeline_context.get(key, ""))
            if value:
                tags[key] = value

        input_hash = clean_optional_string(pipeline_context.get("input_dataset_hash", ""))
        if input_hash:
            tags["data_prep_input_dataset_hash"] = input_hash
        output_hash = clean_optional_string(pipeline_context.get("output_dataset_hash", ""))
        if output_hash:
            tags["data_prep_output_dataset_hash"] = output_hash

        env["MLFLOW_PIPELINE_ID"] = str(pipeline_context.get("pipeline_id", ""))
        env["MLFLOW_PARENT_RUN_ID"] = str(pipeline_context.get("parent_run_id", ""))
        env["MLFLOW_RUN_SOURCE"] = str(run_source)
        env["MLFLOW_TAGS_JSON"] = json.dumps(tags)

        enabled = bool(mlflow_config.enabled and mlflow is not None and clean_optional_string(mlflow_config.tracking_uri))
        env["MLOPS_ENABLED"] = "1" if enabled else "0"
        if enabled:
            env["MLFLOW_TRACKING_URI"] = mlflow_config.tracking_uri
            env["MLFLOW_EXPERIMENT_NAME"] = mlflow_config.experiment_name
        return env

    def build_shell_export_segment(self, env_map: dict[str, str]) -> str:
        exports: list[str] = []
        for key, value in env_map.items():
            safe_key = clean_optional_string(key)
            if safe_key:
                exports.append(f"export {safe_key}={shlex.quote(str(value))}")
        return " && ".join(exports)

    def create_pipeline_parent_run(self, mlflow_config: MlflowConfig, pipeline_id: str, run_source: str) -> str:
        if not mlflow_config.enabled or mlflow is None or not clean_optional_string(mlflow_config.tracking_uri):
            return ""
        try:
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            mlflow.set_experiment(mlflow_config.experiment_name)
            tags = {
                "run_type": "pipeline",
                "pipeline_id": pipeline_id,
                "run_source": run_source,
                "created_by": "app.py",
            }
            with mlflow.start_run(run_name=f"pipeline-{pipeline_id}", tags=tags) as run:
                mlflow.log_dict(
                    {"pipeline_id": pipeline_id, "run_source": run_source, "created_at": now_utc_iso()},
                    "pipeline_context.json",
                )
                return run.info.run_id
        except Exception:
            traceback.print_exc()
            return ""

    def prepare_training_pipeline_context(self, csv_path: str, mlflow_config: MlflowConfig, run_source: str) -> dict:
        sidecar = read_sidecar_for_csv(csv_path) or {}
        data_version_info = self.model_catalog_service.archive_data_version(csv_path, sidecar)
        if data_version_info:
            sidecar.update(data_version_info)
        pipeline_id = clean_optional_string(sidecar.get("pipeline_id")) or str(uuid.uuid4())
        parent_run_id = clean_optional_string(sidecar.get("parent_run_id"))
        if not self.sidecar_matches_mlflow_target(sidecar, mlflow_config):
            parent_run_id = ""
        if mlflow_config.enabled and not parent_run_id:
            parent_run_id = self.create_pipeline_parent_run(mlflow_config, pipeline_id, run_source)
        context = {
            "pipeline_id": pipeline_id,
            "parent_run_id": parent_run_id,
            "data_prep_run_id": clean_optional_string(sidecar.get("data_prep_run_id")),
            "prompt_hash": clean_optional_string(sidecar.get("prompt_hash")),
            "input_dataset_hash": clean_optional_string(sidecar.get("input_dataset_hash")),
            "output_dataset_hash": clean_optional_string(sidecar.get("output_dataset_hash")),
            "llm_model": clean_optional_string(sidecar.get("llm_model")),
            "prompt_version_id": clean_optional_string(sidecar.get("prompt_version_id")),
            "prompt_version_label": clean_optional_string(sidecar.get("prompt_version_label")),
            "prompt_version_dir": clean_optional_string(sidecar.get("prompt_version_dir")),
            "prompt_version_path": clean_optional_string(sidecar.get("prompt_version_path")),
            "prompt_metadata_path": clean_optional_string(sidecar.get("prompt_metadata_path")),
            "prompt_comparison_path": clean_optional_string(sidecar.get("prompt_comparison_path")),
            "previous_prompt_version_id": clean_optional_string(sidecar.get("previous_prompt_version_id")),
            "data_version_id": clean_optional_string(sidecar.get("data_version_id")),
            "data_version_dir": clean_optional_string(sidecar.get("data_version_dir")),
            "data_version_path": clean_optional_string(sidecar.get("data_version_path")),
            "data_prep_tracking_uri": clean_optional_string(sidecar.get("data_prep_tracking_uri") or sidecar.get("tracking_uri")),
            "data_prep_experiment_name": clean_optional_string(sidecar.get("data_prep_experiment_name") or sidecar.get("experiment_name")),
        }
        sidecar_payload = {
            "pipeline_id": pipeline_id,
            "parent_run_id": parent_run_id,
            "data_prep_run_id": context["data_prep_run_id"],
            "prompt_hash": context["prompt_hash"],
            "input_dataset_hash": context["input_dataset_hash"],
            "output_dataset_hash": context["output_dataset_hash"],
            "llm_model": context["llm_model"],
            "prompt_version_id": context["prompt_version_id"],
            "prompt_version_label": context["prompt_version_label"],
            "prompt_version_dir": context["prompt_version_dir"],
            "prompt_version_path": context["prompt_version_path"],
            "prompt_metadata_path": context["prompt_metadata_path"],
            "prompt_comparison_path": context["prompt_comparison_path"],
            "previous_prompt_version_id": context["previous_prompt_version_id"],
            "data_version_id": context["data_version_id"],
            "data_version_dir": context["data_version_dir"],
            "data_version_path": context["data_version_path"],
            "created_at": clean_optional_string(sidecar.get("created_at")) or now_utc_iso(),
            "tracking_uri": clean_optional_string(sidecar.get("tracking_uri")) or mlflow_config.tracking_uri,
            "experiment_name": clean_optional_string(sidecar.get("experiment_name")) or mlflow_config.experiment_name,
            "data_prep_tracking_uri": context["data_prep_tracking_uri"],
            "data_prep_experiment_name": context["data_prep_experiment_name"],
            "training_tracking_uri": mlflow_config.tracking_uri or clean_optional_string(sidecar.get("training_tracking_uri")),
            "training_experiment_name": mlflow_config.experiment_name or clean_optional_string(sidecar.get("training_experiment_name")),
        }
        try:
            write_sidecar_for_csv(csv_path, sidecar_payload)
        except Exception:
            traceback.print_exc()
        return context

    def log_data_prep_mlflow(
        self,
        mlflow_config: MlflowConfig,
        input_path: str,
        output_path: str,
        system_prompt: str,
        llm_model: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        input_hash: str,
        output_hash: str,
        usage_totals: dict[str, int],
        data_version_info: dict | None = None,
        prompt_version_info: dict | None = None,
    ) -> dict:
        prompt_version_info = prompt_version_info or {}
        payload = {
            "pipeline_id": str(uuid.uuid4()),
            "parent_run_id": "",
            "data_prep_run_id": "",
            "prompt_hash": prompt_sha256(system_prompt),
            "prompt_version_id": clean_optional_string(prompt_version_info.get("prompt_version_id", "")),
            "prompt_version_label": clean_optional_string(prompt_version_info.get("prompt_version_label", "")),
            "prompt_version_dir": clean_optional_string(prompt_version_info.get("prompt_version_dir", "")),
            "prompt_version_path": clean_optional_string(prompt_version_info.get("prompt_version_path", "")),
            "prompt_metadata_path": clean_optional_string(prompt_version_info.get("prompt_metadata_path", "")),
            "prompt_comparison_path": clean_optional_string(prompt_version_info.get("prompt_comparison_path", "")),
            "previous_prompt_version_id": clean_optional_string(prompt_version_info.get("previous_prompt_version_id", "")),
            "llm_model": clean_optional_string(llm_model),
            "input_dataset_hash": input_hash,
            "output_dataset_hash": output_hash,
            "created_at": now_utc_iso(),
            "tracking_uri": clean_optional_string(mlflow_config.tracking_uri),
            "experiment_name": clean_optional_string(mlflow_config.experiment_name),
            "data_prep_tracking_uri": clean_optional_string(mlflow_config.tracking_uri),
            "data_prep_experiment_name": clean_optional_string(mlflow_config.experiment_name),
            "data_version_id": clean_optional_string((data_version_info or {}).get("data_version_id", "")),
            "data_version_dir": clean_optional_string((data_version_info or {}).get("data_version_dir", "")),
            "data_version_path": clean_optional_string((data_version_info or {}).get("data_version_path", "")),
        }
        if not mlflow_config.enabled or mlflow is None or not payload["tracking_uri"]:
            return payload
        try:
            mlflow.set_tracking_uri(payload["tracking_uri"])
            mlflow.set_experiment(payload["experiment_name"])
            data_prep_tags = {"run_type": "data_prep", "pipeline_id": payload["pipeline_id"]}
            if payload["prompt_version_id"]:
                data_prep_tags["prompt_version_id"] = payload["prompt_version_id"]
            if payload["prompt_hash"]:
                data_prep_tags["prompt_hash"] = payload["prompt_hash"]
            with mlflow.start_run(
                run_name=f"pipeline-{payload['pipeline_id']}",
                tags={"run_type": "pipeline", "pipeline_id": payload["pipeline_id"], "run_source": "data_prep"},
            ) as parent_run:
                payload["parent_run_id"] = parent_run.info.run_id
                with mlflow.start_run(
                    run_name="data-prep",
                    nested=True,
                    tags=data_prep_tags,
                ) as child_run:
                    payload["data_prep_run_id"] = child_run.info.run_id
                    input_meta = dataframe_metadata(input_df, label_col="class")
                    output_meta = dataframe_metadata(output_df, label_col="class")
                    mlflow.log_param("llm_model", clean_optional_string(llm_model) or "unknown")
                    mlflow.log_param("input_filename", os.path.basename(input_path))
                    mlflow.log_param("output_filename", os.path.basename(output_path))
                    mlflow.log_param("prompt_hash", payload["prompt_hash"])
                    if payload["prompt_version_id"]:
                        mlflow.log_param("prompt_version_id", payload["prompt_version_id"])
                    if payload["prompt_version_label"]:
                        mlflow.log_param("prompt_version_label", payload["prompt_version_label"])
                    if payload["previous_prompt_version_id"]:
                        mlflow.log_param("previous_prompt_version_id", payload["previous_prompt_version_id"])
                    mlflow.log_param("input_dataset_hash", payload["input_dataset_hash"])
                    mlflow.log_param("output_dataset_hash", payload["output_dataset_hash"])
                    if payload["data_version_id"]:
                        mlflow.log_param("data_version_id", payload["data_version_id"])
                    if payload["data_version_path"]:
                        mlflow.log_param("data_version_path", payload["data_version_path"])
                    mlflow.log_metric("input_rows", int(input_meta.get("row_count", 0)))
                    mlflow.log_metric("output_rows", int(output_meta.get("row_count", 0)))
                    mlflow.log_metric("prompt_chars", int(prompt_version_info.get("char_count", len(system_prompt)) or 0))
                    mlflow.log_metric("prompt_lines", int(prompt_version_info.get("line_count", len(system_prompt.splitlines())) or 0))
                    for metric_name, metric_value in usage_totals.items():
                        mlflow.log_metric(metric_name, int(metric_value))
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        prompt_preview_path = os.path.join(tmp_dir, "prompt_preview.txt")
                        with open(prompt_preview_path, "w", encoding="utf-8") as file_obj:
                            file_obj.write(safe_prompt_preview(system_prompt, max_chars=2000))
                        prompt_artifact_path = os.path.join(tmp_dir, "prompt.txt")
                        with open(prompt_artifact_path, "w", encoding="utf-8") as file_obj:
                            file_obj.write(system_prompt)
                        metadata_path = os.path.join(tmp_dir, "data_prep_metadata.json")
                        write_json(
                            metadata_path,
                            {
                                "pipeline_id": payload["pipeline_id"],
                                "input_path": input_path,
                                "output_path": output_path,
                                "prompt_hash": payload["prompt_hash"],
                                "prompt_version_id": payload["prompt_version_id"],
                                "prompt_version_label": payload["prompt_version_label"],
                                "prompt_version_path": payload["prompt_version_path"],
                                "prompt_comparison_path": payload["prompt_comparison_path"],
                                "previous_prompt_version_id": payload["previous_prompt_version_id"],
                                "input_dataset_hash": payload["input_dataset_hash"],
                                "output_dataset_hash": payload["output_dataset_hash"],
                                "data_version_id": payload["data_version_id"],
                                "data_version_path": payload["data_version_path"],
                                "usage_totals": usage_totals,
                                "input_metadata": input_meta,
                                "output_metadata": output_meta,
                                "created_at": payload["created_at"],
                            },
                        )
                        local_prompt_metadata_path = os.path.join(tmp_dir, "prompt_metadata.json")
                        write_json(local_prompt_metadata_path, prompt_version_info)
                        comparison_source_path = payload["prompt_comparison_path"]
                        if comparison_source_path and os.path.exists(comparison_source_path):
                            shutil.copy2(comparison_source_path, os.path.join(tmp_dir, "prompt_comparison.diff"))
                        sample_path = os.path.join(tmp_dir, "output_sample.csv")
                        dataframe_sample(output_df, max_rows=100, max_cell_chars=200).to_csv(sample_path, index=False)
                        mlflow.log_artifacts(tmp_dir, artifact_path="data_prep")
        except Exception:
            traceback.print_exc()
        return payload

    def find_latest_training_mlflow_metadata(self, hosted_model_path: str = "") -> dict | None:
        candidate_paths: list[Path] = []
        seen: set[str] = set()
        recursive_roots: list[Path] = []
        hosted_model_value = clean_optional_string(hosted_model_path)
        if hosted_model_value:
            try:
                recursive_roots.append(Path(discover_model_dir(hosted_model_value)).resolve())
            except Exception:
                pass
        project_download_root = (self.project_dir / "downloaded_model").resolve()
        if project_download_root.exists():
            recursive_roots.append(project_download_root)
        for root in self.model_catalog_service.get_training_metadata_search_roots(hosted_model_path):
            direct_meta = root / "last_training_mlflow.json"
            if direct_meta.exists():
                key = str(direct_meta.resolve())
                if key not in seen:
                    seen.add(key)
                    candidate_paths.append(direct_meta)
            should_search_recursively = any(root == recursive_root for recursive_root in recursive_roots)
            if root.is_dir() and should_search_recursively:
                try:
                    nested_matches = root.rglob("last_training_mlflow.json")
                except Exception:
                    nested_matches = []
                for match in nested_matches:
                    try:
                        key = str(match.resolve())
                    except Exception:
                        key = str(match)
                    if key not in seen:
                        seen.add(key)
                        candidate_paths.append(match)
        if not candidate_paths:
            return None
        latest = max(candidate_paths, key=lambda path: path.stat().st_mtime)
        payload = read_json(str(latest)) or {}
        if payload:
            payload["_metadata_path"] = str(latest)
        return payload if payload else None

    def cache_downloaded_training_mlflow_metadata(self, download_path: str) -> None:
        search_root = Path(download_path)
        if not search_root.exists():
            return
        candidates = list(search_root.rglob("last_training_mlflow.json"))
        if not candidates:
            return
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        target = Path(self.artifact_store.last_training_metadata_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(latest), str(target))

    def start_local_mlflow_ui(self, tracking_uri: str = "") -> str:
        if mlflow is None:
            raise RuntimeError("The `mlflow` package is not available in this Python environment.")
        backend_store_uri = clean_optional_string(tracking_uri) or self.local_tracking_uri
        if backend_store_uri.startswith("file://"):
            backend_store_uri = backend_store_uri[7:]
        if self.mlflow_ui_process is None or self.mlflow_ui_process.poll() is not None:
            self.mlflow_ui_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "ui",
                    "--backend-store-uri",
                    backend_store_uri,
                    "--port",
                    "5001",
                ],
                cwd=self.project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            if self.mlflow_ui_process.poll() is not None:
                raise RuntimeError("Failed to start the local MLflow UI process.")
        return "http://127.0.0.1:5001"

    def stop_local_mlflow_ui(self) -> None:
        if self.mlflow_ui_process is None or self.mlflow_ui_process.poll() is not None:
            return
        try:
            self.mlflow_ui_process.terminate()
            self.mlflow_ui_process.wait(timeout=3)
        except Exception:
            try:
                self.mlflow_ui_process.kill()
            except Exception:
                pass

    def resolve_dashboard_tracking_console(
        self,
        backend: str,
        tracking_uri: str,
        azure_studio_url: str,
        *,
        launch_live_console: bool,
        hosted_model_path: str = "",
    ) -> tuple[str, str]:
        training_meta = self.find_latest_training_mlflow_metadata(hosted_model_path) or {}
        effective_tracking_uri = clean_optional_string(training_meta.get("tracking_uri")) or clean_optional_string(tracking_uri)
        if backend == "local":
            if mlflow is None:
                return "", "The local MLflow UI is unavailable because the `mlflow` package is not installed in this Python environment."
            if launch_live_console:
                try:
                    return self.start_local_mlflow_ui(self.local_tracking_uri), ""
                except Exception as exc:
                    return "", str(exc)
            return "http://127.0.0.1:5001", ""
        if backend == "custom_uri":
            if effective_tracking_uri.startswith("http://") or effective_tracking_uri.startswith("https://"):
                return effective_tracking_uri, ""
            return "", "The custom tracking URI is not an HTTP URL, so it cannot be opened in the browser."
        if azure_studio_url:
            return azure_studio_url, ""
        return "", "Azure dashboard URL is unavailable because the Azure subscription ID is empty."

    def _dashboard_value_html(self, value: Any) -> str:
        text = clean_optional_string(value)
        return html.escape(text) if text else "<span class='muted'>Not available</span>"

    def _dashboard_link_html(self, url: str, label: str | None = None) -> str:
        clean_url = clean_optional_string(url)
        if not clean_url:
            return "<span class='muted'>Not available</span>"
        link_label = clean_optional_string(label) or clean_url
        return (
            f"<a href=\"{html.escape(clean_url, quote=True)}\" target=\"_blank\" rel=\"noreferrer\">"
            f"{html.escape(link_label)}</a>"
        )

    def open_dashboard_url(self, url: str) -> None:
        if clean_optional_string(url):
            webbrowser.open(url)
