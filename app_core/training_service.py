from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from mlops_utils import (
    clean_optional_string,
    compute_file_sha256,
    discover_model_dir,
    now_utc_iso,
    read_json,
    read_sidecar_for_csv,
    write_sidecar_for_csv,
)

from .azure_platform_service import AZURE_AVAILABLE, AzurePlatformService
from .contracts import MlflowConfig, TrainingRequest
from .mlops_service import MlopsService
from .model_catalog_service import ModelCatalogService
from .runtime import JobCancelled, JobContext, JobManager


class TrainingService:
    def __init__(
        self,
        project_dir: str,
        job_manager: JobManager,
        model_catalog_service: ModelCatalogService,
        mlops_service: MlopsService,
        azure_platform_service: AzurePlatformService,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.job_manager = job_manager
        self.model_catalog_service = model_catalog_service
        self.mlops_service = mlops_service
        self.azure_platform_service = azure_platform_service

    def parse_numeric_list(self, raw_value: str, parser, field_name: str):
        cleaned = raw_value.strip()
        if not cleaned:
            return []
        parsed_values = []
        for part in cleaned.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                parsed_values.append(parser(token))
            except Exception as exc:
                raise ValueError(f"Invalid value '{token}' in {field_name}: {exc}") from exc
        return parsed_values

    def collect_training_options(self, values: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            strategy = (values.get("strategy", "").strip() or "default").lower()
            if strategy not in {"default", "tune", "tune_cv"}:
                return None, f"Unsupported training mode: {strategy}"
            epochs = int(values.get("epochs", "3").strip())
            batch_size = int(values.get("batch_size", "8").strip())
            learning_rate = float(values.get("learning_rate", "5e-5").strip())
            weight_decay = float(values.get("weight_decay", "0.01").strip())
            max_length = int(values.get("max_length", "128").strip())
            cv_folds = int(values.get("cv_folds", "3").strip() or "3")
            max_trials = int(values.get("max_trials", "8").strip() or "8")
        except Exception as exc:
            return None, f"Invalid training configuration values: {exc}"
        if epochs < 1:
            return None, "Epochs must be >= 1."
        if batch_size < 1:
            return None, "Batch Size must be >= 1."
        if learning_rate <= 0:
            return None, "Learning Rate must be > 0."
        if weight_decay < 0:
            return None, "Weight Decay must be >= 0."
        if max_length < 16:
            return None, "Max Length must be >= 16."
        if cv_folds < 2:
            return None, "CV Folds must be >= 2."
        if max_trials < 1:
            return None, "Max Trials must be >= 1."
        try:
            tune_lrs = self.parse_numeric_list(values.get("tune_lrs", ""), float, "Tune LRs")
            tune_batch_sizes = self.parse_numeric_list(values.get("tune_batch_sizes", ""), int, "Tune Batch Sizes")
            tune_epochs = self.parse_numeric_list(values.get("tune_epochs", ""), int, "Tune Epochs")
            tune_weight_decays = self.parse_numeric_list(values.get("tune_weight_decays", ""), float, "Tune Weight Decays")
            tune_max_lengths = self.parse_numeric_list(values.get("tune_max_lengths", ""), int, "Tune Max Lengths")
        except ValueError as exc:
            return None, str(exc)
        tune_lrs = [value for value in tune_lrs if value > 0] or [learning_rate]
        tune_batch_sizes = [value for value in tune_batch_sizes if value > 0] or [batch_size]
        tune_epochs = [value for value in tune_epochs if value > 0] or [epochs]
        tune_weight_decays = [value for value in tune_weight_decays if value >= 0] or [weight_decay]
        tune_max_lengths = [value for value in tune_max_lengths if value >= 16] or [max_length]
        if strategy == "default":
            max_trials = 1
        if strategy != "tune_cv":
            cv_folds = 3
        return {
            "train_mode": strategy,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "cv_folds": cv_folds,
            "max_trials": max_trials,
            "tune_learning_rates": tune_lrs,
            "tune_batch_sizes": tune_batch_sizes,
            "tune_epochs": tune_epochs,
            "tune_weight_decays": tune_weight_decays,
            "tune_max_lengths": tune_max_lengths,
        }, None

    def build_train_cli_args(self, training_options: dict) -> list[str]:
        def list_to_csv(values):
            return ",".join(str(value) for value in values)
        return [
            "--train-mode", str(training_options["train_mode"]),
            "--epochs", str(training_options["epochs"]),
            "--batch-size", str(training_options["batch_size"]),
            "--learning-rate", str(training_options["learning_rate"]),
            "--weight-decay", str(training_options["weight_decay"]),
            "--max-length", str(training_options["max_length"]),
            "--val-ratio", str(training_options["val_ratio"]),
            "--test-ratio", str(training_options["test_ratio"]),
            "--cv-folds", str(training_options["cv_folds"]),
            "--max-trials", str(training_options["max_trials"]),
            "--tune-learning-rates", list_to_csv(training_options["tune_learning_rates"]),
            "--tune-batch-sizes", list_to_csv(training_options["tune_batch_sizes"]),
            "--tune-epochs", list_to_csv(training_options["tune_epochs"]),
            "--tune-weight-decays", list_to_csv(training_options["tune_weight_decays"]),
            "--tune-max-lengths", list_to_csv(training_options["tune_max_lengths"]),
        ]

    def build_train_cli_segment(self, training_options: dict) -> str:
        args = self.build_train_cli_args(training_options)
        return " ".join(shlex.quote(arg) for arg in args) if args else ""

    def check_host_cuda_available(self) -> tuple[bool, str | None]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() == "1", None
        except Exception as exc:
            return False, str(exc)

    def submit_training(self, request: TrainingRequest):
        return self.job_manager.submit(
            "training",
            lambda ctx: self._run(ctx, request),
            metadata={"operation": "training", "csv_path": request.csv_path, "environment_mode": request.environment_mode},
        )

    def cancel(self, job_id: str) -> bool:
        return self.job_manager.cancel_job(job_id)

    def _run(self, ctx: JobContext, request: TrainingRequest) -> dict[str, Any]:
        if request.environment_mode == "azure":
            return self._run_azure_training(ctx, request)
        return self._run_local_training(ctx, request)

    def _run_local_training(self, ctx: JobContext, request: TrainingRequest) -> dict[str, Any]:
        backend = "container" if request.local_runtime == "container" else "host"
        project_dir = str(self.project_dir)
        extra_train_args = self.build_train_cli_args(request.training_options)
        if backend == "container":
            docker_script = os.path.join(project_dir, "scripts", "train_docker.sh")
            if not os.path.exists(docker_script):
                raise FileNotFoundError("Could not find scripts/train_docker.sh in the app directory.")
            command_args = ["bash", docker_script, request.csv_path, *extra_train_args]
            process_env = os.environ.copy()
            process_env["DEVICE"] = request.local_device
        else:
            train_script = os.path.join(project_dir, "train.py")
            if not os.path.exists(train_script):
                raise FileNotFoundError("Could not find 'train.py' in the app directory.")
            command_args = [sys.executable, train_script, "--data", request.csv_path, *extra_train_args]
            process_env = os.environ.copy()
            if request.local_device == "cpu":
                process_env["CUDA_VISIBLE_DEVICES"] = "-1"

        run_source = "local_container" if backend == "container" else "local_host"
        pipeline_context = self.mlops_service.prepare_training_pipeline_context(request.csv_path, request.mlflow_config, run_source)
        process_env.update(
            self.mlops_service.build_training_mlflow_env(
                request.mlflow_config,
                pipeline_context,
                run_source,
                environment_mode=request.environment_mode,
            )
        )
        ctx.emit("progress", f"Starting local {backend} training on device: {request.local_device}...")
        process = subprocess.Popen(
            command_args,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )
        ctx.register_subprocess("local_training", process)
        try:
            if process.stdout:
                while True:
                    ctx.check_cancelled()
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                        continue
                    clean_line = line.strip()
                    if clean_line:
                        ctx.emit("progress", clean_line[:150])
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Local training failed with exit code {return_code}.")
        finally:
            ctx.clear_subprocess("local_training")

        model_path = os.path.abspath(os.path.join(project_dir, "outputs", "final_model"))
        training_metadata = read_json(os.path.join(project_dir, "outputs", "last_training_mlflow.json")) or {}
        archived_model_dir = clean_optional_string(training_metadata.get("model_version_model_dir"))
        selected_model_dir = archived_model_dir if archived_model_dir and os.path.isdir(archived_model_dir) else model_path
        return {
            "operation": "training",
            "message": f"Local {backend} model training completed.",
            "model_path": model_path,
            "selected_model_dir": selected_model_dir,
            "backend": backend,
        }

    def _run_azure_training(self, ctx: JobContext, request: TrainingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        ml_client = None
        returned_job_name = ""
        compute_mode = "gpu" if request.azure_compute == "gpu" else "cpu"
        compute_name = "gpu-cluster-temp" if compute_mode == "gpu" else "cpu-cluster-temp"
        compute_size = ""
        attempted_compute_sizes: list[str] = []
        compute_created = False
        try:
            credential = self.azure_platform_service.create_interactive_credential(request.azure_tenant_id)
            ml_client = self.azure_platform_service.ensure_azure_workspace(
                request.azure_sub_id,
                request.azure_tenant_id,
                emit=lambda msg: ctx.emit("progress", msg),
                credential=credential,
            )

            ctx.add_cleanup(
                lambda client=ml_client, get_job=lambda: returned_job_name: self.azure_platform_service.cancel_job(client, get_job()) if get_job() else None
            )

            run_source = "azure_gpu" if compute_mode == "gpu" else "azure_cpu"
            azure_tracking_uri = clean_optional_string(self.mlops_service.resolve_azure_mlflow_tracking_uri(ml_client))
            if not azure_tracking_uri:
                raise ValueError("Azure MLflow tracking URI is unresolved for this workspace.")
            azure_experiment_name = clean_optional_string(request.mlflow_config.experiment_name) or "deberta-log-classification"
            resolved_mlflow_config = MlflowConfig(
                enabled=True,
                backend="azure",
                tracking_uri=azure_tracking_uri,
                experiment_name=azure_experiment_name,
                registered_model_name=clean_optional_string(request.mlflow_config.registered_model_name),
            )
            ctx.emit("progress", f"Azure MLflow tracking enabled for experiment: {azure_experiment_name}")

            pipeline_context_local = self.mlops_service.prepare_training_pipeline_context(request.csv_path, resolved_mlflow_config, run_source)
            if resolved_mlflow_config.enabled and not clean_optional_string(pipeline_context_local.get("parent_run_id")):
                parent_run_id = self.mlops_service.create_pipeline_parent_run(resolved_mlflow_config, str(pipeline_context_local.get("pipeline_id", "")), run_source)
                pipeline_context_local["parent_run_id"] = parent_run_id
                sidecar_existing = read_sidecar_for_csv(request.csv_path) or {}
                sidecar_existing["parent_run_id"] = parent_run_id
                sidecar_existing["tracking_uri"] = clean_optional_string(sidecar_existing.get("tracking_uri")) or resolved_mlflow_config.tracking_uri
                sidecar_existing["experiment_name"] = clean_optional_string(sidecar_existing.get("experiment_name")) or resolved_mlflow_config.experiment_name
                sidecar_existing["training_tracking_uri"] = resolved_mlflow_config.tracking_uri
                sidecar_existing["training_experiment_name"] = resolved_mlflow_config.experiment_name
                if "created_at" not in sidecar_existing:
                    sidecar_existing["created_at"] = now_utc_iso()
                write_sidecar_for_csv(request.csv_path, sidecar_existing)

            data_version_id = clean_optional_string(pipeline_context_local.get("data_version_id")) or compute_file_sha256(request.csv_path)
            azure_data_asset_info: dict[str, str] = {}
            ctx.emit("progress", "Registering labeled dataset as an Azure ML Data asset...")
            azure_data_asset_info = self.azure_platform_service.register_azure_data_asset(
                ml_client=ml_client,
                csv_path=request.csv_path,
                asset_name="log-monitor-labeled-data",
                asset_version=data_version_id,
                description="Log Monitor labeled training dataset.",
                tags={
                    "created_by": "log-monitor",
                    "pipeline_id": clean_optional_string(pipeline_context_local.get("pipeline_id")),
                    "data_version_id": data_version_id,
                    "source_filename": Path(request.csv_path).name,
                    "run_source": run_source,
                },
            )
            pipeline_context_local.update(azure_data_asset_info)
            sidecar_existing = read_sidecar_for_csv(request.csv_path) or {}
            sidecar_existing.update(azure_data_asset_info)
            sidecar_existing["azure_data_asset_registered_at"] = now_utc_iso()
            write_sidecar_for_csv(request.csv_path, sidecar_existing)
            azure_data_asset_uri = clean_optional_string(azure_data_asset_info.get("azure_data_asset_uri"))
            if azure_data_asset_uri:
                ctx.emit("progress", f"Dataset registered in Azure ML Studio: {azure_data_asset_uri}")

            mlflow_env = self.mlops_service.build_training_mlflow_env(
                resolved_mlflow_config,
                pipeline_context_local,
                run_source,
                environment_mode=request.environment_mode,
            )
            export_segment = self.mlops_service.build_shell_export_segment(mlflow_env)
            mlflow_install_fragment = "mlflow==2.9.2 azureml-mlflow "
            train_cli_segment = self.build_train_cli_segment(request.training_options)
            train_cli_segment = f" {train_cli_segment}" if train_cli_segment else ""

            compute_candidates = self.azure_platform_service.prioritize_instance_candidates(
                self.azure_platform_service.get_azure_training_instance_candidates(compute_mode),
                request.azure_instance_type,
            )
            last_compute_error = None
            for instance_type in compute_candidates:
                ctx.check_cancelled()
                attempted_compute_sizes.append(instance_type)
                ctx.emit("progress", f"Provisioning {compute_mode.upper()} cluster ({instance_type})...")
                try:
                    self.azure_platform_service.ensure_aml_compute(ml_client, compute_name, instance_type)
                    compute_size = instance_type
                    break
                except Exception as exc:
                    last_compute_error = exc
                    if self.azure_platform_service.is_azure_quota_error(exc) and instance_type != compute_candidates[-1]:
                        continue
                    raise
            if not compute_size:
                if last_compute_error is not None:
                    raise last_compute_error
                raise RuntimeError("No Azure training compute size could be provisioned.")
            compute_created = True

            ctx.emit("progress", "Uploading data and starting DeBERTa training...")
            returned_job_name = self.azure_platform_service.submit_azure_training_job(
                ml_client=ml_client,
                csv_path=request.csv_path,
                compute_name=compute_name,
                mlflow_install_fragment=mlflow_install_fragment,
                export_segment=export_segment,
                train_cli_segment=train_cli_segment,
                data_input_path=clean_optional_string(azure_data_asset_info.get("azure_data_asset_uri")),
                experiment_name=azure_experiment_name,
            )
            while True:
                ctx.check_cancelled()
                job_status = self.azure_platform_service.get_job_status(ml_client, returned_job_name)
                ctx.emit("progress", f"Training in progress. Azure Status: {job_status}")
                if job_status in ["Completed", "Failed", "Canceled"]:
                    break
                time.sleep(15)
            if job_status == "Failed":
                raise RuntimeError("The training job failed on the Azure machine. Check Azure Portal logs.")
            if job_status == "Canceled":
                raise JobCancelled("Azure training was interrupted before completion.")

            ctx.emit("progress", "Training Complete! Downloading model...")
            download_path = "./downloaded_model"
            self.azure_platform_service.download_job_output(ml_client, returned_job_name, download_path)
            self.mlops_service.cache_downloaded_training_mlflow_metadata(download_path)
            preferred_model_path = ""
            try:
                preferred_model_path = discover_model_dir(download_path)
            except Exception:
                pass
            completion_message = f"Model trained and downloaded to {os.path.abspath(download_path)}"
            if azure_data_asset_uri:
                completion_message += f"\nAzure data asset: {azure_data_asset_uri}"
            return {
                "operation": "training",
                "message": completion_message,
                "download_path": os.path.abspath(download_path),
                "selected_model_dir": preferred_model_path,
                "instance_type": compute_size,
                "azure_data_asset_name": azure_data_asset_info.get("azure_data_asset_name", ""),
                "azure_data_asset_version": azure_data_asset_info.get("azure_data_asset_version", ""),
                "azure_data_asset_uri": azure_data_asset_info.get("azure_data_asset_uri", ""),
            }
        finally:
            if ml_client and compute_created:
                try:
                    ctx.emit("progress", "Destroying compute cluster to prevent charges...")
                    self.azure_platform_service.delete_compute(ml_client, compute_name)
                except Exception:
                    traceback.print_exc()
