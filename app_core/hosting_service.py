from __future__ import annotations

import csv
import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from mlops_utils import (
    clean_optional_string,
    compute_file_sha256,
    discover_model_dir,
    now_utc_iso,
    read_json,
    write_json,
)

from .azure_platform_service import AZURE_AVAILABLE, AzurePlatformService
from .contracts import HostingRequest
from .github_service import GitHubService
from .mlops_service import MlopsService
from .model_catalog_service import ModelCatalogService
from .observability_service import ObservabilityService
from .runtime import JobContext, JobManager

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
except Exception:
    accuracy_score = None
    confusion_matrix = None
    precision_recall_fscore_support = None

LABEL_NAMES = ["Error", "CONFIGURATION", "SYSTEM", "Noise"]


def _load_inference_helpers():
    try:
        from inference_utils import load_model_bundle as bundle_loader
        from inference_utils import predict_error_message as predictor
    except Exception as exc:
        raise RuntimeError(
            "Model inference dependencies could not be loaded. "
            "Fix the local PyTorch installation before running deployment gates or local inference."
        ) from exc
    return bundle_loader, predictor


def load_model_bundle(model_path: str):
    bundle_loader, _ = _load_inference_helpers()
    return bundle_loader(model_path)


def predict_error_message(bundle, error_message: str) -> str:
    _, predictor = _load_inference_helpers()
    return predictor(bundle, error_message)


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

    def _resolve_gate_input_path(self, raw_path: str, default_relative_path: str) -> Path:
        candidate = clean_optional_string(raw_path) or default_relative_path
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (self.project_dir / path).resolve()
        else:
            path = path.resolve()
        return path

    def _load_gate_policy(self, policy_path: Path) -> dict[str, Any]:
        payload = read_json(str(policy_path))
        if not isinstance(payload, dict):
            raise ValueError(f"Gate policy must be a JSON object: {policy_path}")

        required_scalar_keys = ("min_accuracy", "min_weighted_f1", "min_macro_f1")
        normalized: dict[str, Any] = {}
        for key in required_scalar_keys:
            if key not in payload:
                raise ValueError(f"Gate policy is missing required key `{key}`: {policy_path}")
            try:
                value = float(payload[key])
            except Exception as exc:
                raise ValueError(f"Gate policy key `{key}` must be numeric: {exc}") from exc
            if value < 0 or value > 1:
                raise ValueError(f"Gate policy key `{key}` must be within [0, 1].")
            normalized[key] = value

        recalls = payload.get("min_recall_per_class")
        if not isinstance(recalls, dict):
            raise ValueError("Gate policy key `min_recall_per_class` must be an object.")
        normalized_recalls: dict[str, float] = {}
        for raw_label, raw_value in recalls.items():
            label = clean_optional_string(raw_label)
            if not label:
                continue
            try:
                value = float(raw_value)
            except Exception as exc:
                raise ValueError(f"Gate policy recall threshold for `{label}` must be numeric: {exc}") from exc
            if value < 0 or value > 1:
                raise ValueError(f"Gate policy recall threshold for `{label}` must be within [0, 1].")
            normalized_recalls[label] = value
        normalized["min_recall_per_class"] = normalized_recalls
        return normalized

    def _resolve_gate_dataset_columns(self, fieldnames: list[str]) -> tuple[str, str]:
        normalized_map: dict[str, str] = {clean_optional_string(name).lower(): name for name in fieldnames}
        message_candidates = ("logmessage", "log_message", "message", "errormessage", "error_message", "log")
        label_candidates = ("class", "label", "classification", "target", "y")

        message_column = ""
        for candidate in message_candidates:
            if candidate in normalized_map:
                message_column = normalized_map[candidate]
                break
        if not message_column:
            raise ValueError("Golden set is missing a message column (expected LogMessage/message/errorMessage).")

        label_column = ""
        for candidate in label_candidates:
            if candidate in normalized_map:
                label_column = normalized_map[candidate]
                break
        if not label_column:
            raise ValueError("Golden set is missing a label column (expected class/label).")
        return message_column, label_column

    def _load_gate_dataset_rows(self, golden_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(golden_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Golden set has no header columns: {golden_path}")
            message_column, label_column = self._resolve_gate_dataset_columns(list(reader.fieldnames))
            for line_number, row in enumerate(reader, start=2):
                message = clean_optional_string((row or {}).get(message_column, ""))
                label = clean_optional_string((row or {}).get(label_column, ""))
                if not message:
                    raise ValueError(f"Golden set row {line_number} has empty message text.")
                if not label:
                    raise ValueError(f"Golden set row {line_number} has empty label.")
                rows.append(
                    {
                        "line_number": line_number,
                        "message": message,
                        "label": label,
                    }
                )
        if not rows:
            raise ValueError(f"Golden set has no data rows: {golden_path}")
        return rows

    def _compute_model_dir_hash(self, model_dir: Path) -> str:
        digest = hashlib.sha256()
        files = sorted([path for path in model_dir.rglob("*") if path.is_file()])
        if not files:
            raise ValueError(f"No files found under model directory: {model_dir}")
        for file_path in files:
            relative_path = file_path.relative_to(model_dir).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(compute_file_sha256(str(file_path)).encode("utf-8"))
        return digest.hexdigest()

    def _compute_gate_metrics(self, y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
        if accuracy_score is None or precision_recall_fscore_support is None or confusion_matrix is None:
            raise RuntimeError("scikit-learn is required for deployment gate metrics, but it is not available.")
        if len(y_true) != len(y_pred):
            raise ValueError("Gate metric computation received mismatched true/predicted label lengths.")
        if not y_true:
            raise ValueError("Gate metric computation received an empty dataset.")

        labels = list(LABEL_NAMES)
        for label in y_true:
            clean_label = clean_optional_string(label)
            if clean_label and clean_label not in labels:
                labels.append(clean_label)
        for label in y_pred:
            clean_label = clean_optional_string(label)
            if clean_label and clean_label not in labels:
                labels.append(clean_label)

        accuracy = float(accuracy_score(y_true, y_pred))
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        per_class: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(labels):
            per_class[label] = {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }

        confusion = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        return {
            "accuracy": accuracy,
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "per_class": per_class,
            "confusion_matrix": {"labels": labels, "matrix": confusion},
        }

    def _check_deployment_gate_cache(
        self,
        *,
        gate_root: Path,
        model_hash: str,
        golden_hash: str,
        policy_hash: str,
    ) -> dict[str, Any] | None:
        if not gate_root.exists():
            return None
        try:
            candidates = sorted(gate_root.glob("gate_eval_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        except Exception:
            return None
        for candidate in candidates[:200]:
            payload = read_json(str(candidate))
            if not isinstance(payload, dict):
                continue
            if not bool(payload.get("gate_pass")):
                continue
            if clean_optional_string(payload.get("model_hash")) != model_hash:
                continue
            if clean_optional_string(payload.get("golden_set_hash")) != golden_hash:
                continue
            if clean_optional_string(payload.get("policy_hash")) != policy_hash:
                continue
            if not clean_optional_string(payload.get("report_path")):
                payload["report_path"] = str(candidate.resolve())
            payload["_cache_hit"] = True
            return payload
        return None

    def _write_deployment_gate_artifacts(
        self,
        *,
        gate_root: Path,
        payload: dict[str, Any],
        predictions: list[dict[str, Any]],
    ) -> dict[str, str]:
        gate_root.mkdir(parents=True, exist_ok=True)
        stamp = str(int(time.time() * 1000))
        report_path = gate_root / f"gate_eval_{stamp}.json"
        predictions_path = gate_root / f"gate_eval_{stamp}_predictions.csv"

        with open(predictions_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["line_number", "message", "true_label", "predicted_label", "match"],
            )
            writer.writeheader()
            for row in predictions:
                writer.writerow(
                    {
                        "line_number": row.get("line_number", ""),
                        "message": clean_optional_string(row.get("message")),
                        "true_label": clean_optional_string(row.get("true_label")),
                        "predicted_label": clean_optional_string(row.get("predicted_label")),
                        "match": bool(row.get("match")),
                    }
                )

        payload_with_paths = dict(payload)
        payload_with_paths["report_path"] = str(report_path.resolve())
        payload_with_paths["predictions_path"] = str(predictions_path.resolve())
        write_json(str(report_path), payload_with_paths)
        write_json(str((gate_root / "latest_gate_eval.json").resolve()), payload_with_paths)
        return {
            "report_path": str(report_path.resolve()),
            "predictions_path": str(predictions_path.resolve()),
        }

    def _enforce_deployment_gate(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        ctx.emit("progress", "Evaluating deployment gate against golden dataset...")

        resolved_model_dir = Path(discover_model_dir(request.model_dir)).resolve()
        golden_path = self._resolve_gate_input_path(request.deployment_gate_golden_path, "gates/deployment_golden.csv")
        policy_path = self._resolve_gate_input_path(request.deployment_gate_policy_path, "gates/deployment_policy.json")
        if not golden_path.exists():
            raise FileNotFoundError(f"Deployment gate golden set file not found: {golden_path}")
        if not policy_path.exists():
            raise FileNotFoundError(f"Deployment gate policy file not found: {policy_path}")

        policy = self._load_gate_policy(policy_path)
        golden_rows = self._load_gate_dataset_rows(golden_path)
        model_hash = self._compute_model_dir_hash(resolved_model_dir)
        golden_hash = compute_file_sha256(str(golden_path))
        policy_hash = compute_file_sha256(str(policy_path))
        gate_root = (self.project_dir / "outputs" / "gates").resolve()

        cached = self._check_deployment_gate_cache(
            gate_root=gate_root,
            model_hash=model_hash,
            golden_hash=golden_hash,
            policy_hash=policy_hash,
        )
        if isinstance(cached, dict):
            cached_report = clean_optional_string(cached.get("report_path"))
            ctx.emit(
                "progress",
                "Deployment gate cache hit: prior PASS was reused."
                + (f" ({cached_report})" if cached_report else ""),
            )
            return {
                "gate_pass": True,
                "cached": True,
                "report_path": cached_report,
                "predictions_path": clean_optional_string(cached.get("predictions_path")),
                "model_hash": model_hash,
                "golden_set_hash": golden_hash,
                "policy_hash": policy_hash,
                "sample_count": int(cached.get("sample_count") or len(golden_rows)),
                "metrics": cached.get("metrics", {}),
                "failed_checks": [],
                "golden_set_path": str(golden_path.resolve()),
                "policy_path": str(policy_path.resolve()),
            }

        bundle = load_model_bundle(str(resolved_model_dir))
        y_true: list[str] = []
        y_pred: list[str] = []
        predictions: list[dict[str, Any]] = []
        for row in golden_rows:
            true_label = clean_optional_string(row.get("label"))
            predicted_label = clean_optional_string(predict_error_message(bundle, clean_optional_string(row.get("message"))))
            y_true.append(true_label)
            y_pred.append(predicted_label)
            predictions.append(
                {
                    "line_number": int(row.get("line_number", 0)),
                    "message": clean_optional_string(row.get("message")),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "match": predicted_label == true_label,
                }
            )

        metrics = self._compute_gate_metrics(y_true, y_pred)
        failed_checks: list[str] = []
        if float(metrics.get("accuracy", 0.0)) < float(policy["min_accuracy"]):
            failed_checks.append(
                f"accuracy {float(metrics.get('accuracy', 0.0)):.4f} < min_accuracy {float(policy['min_accuracy']):.4f}"
            )
        if float(metrics.get("weighted_f1", 0.0)) < float(policy["min_weighted_f1"]):
            failed_checks.append(
                f"weighted_f1 {float(metrics.get('weighted_f1', 0.0)):.4f} < min_weighted_f1 {float(policy['min_weighted_f1']):.4f}"
            )
        if float(metrics.get("macro_f1", 0.0)) < float(policy["min_macro_f1"]):
            failed_checks.append(
                f"macro_f1 {float(metrics.get('macro_f1', 0.0)):.4f} < min_macro_f1 {float(policy['min_macro_f1']):.4f}"
            )

        per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
        recall_thresholds = policy.get("min_recall_per_class") if isinstance(policy.get("min_recall_per_class"), dict) else {}
        for raw_label, raw_threshold in recall_thresholds.items():
            label = clean_optional_string(raw_label)
            if not label:
                continue
            threshold = float(raw_threshold)
            class_recall = 0.0
            class_payload = per_class.get(label) if isinstance(per_class, dict) else None
            if isinstance(class_payload, dict):
                class_recall = float(class_payload.get("recall", 0.0))
            if class_recall < threshold:
                failed_checks.append(
                    f"recall[{label}] {class_recall:.4f} < min_recall_per_class[{label}] {threshold:.4f}"
                )

        gate_pass = len(failed_checks) == 0
        report_payload: dict[str, Any] = {
            "created_at": now_utc_iso(),
            "gate_pass": gate_pass,
            "failed_checks": failed_checks,
            "sample_count": len(golden_rows),
            "model_dir": str(resolved_model_dir),
            "model_hash": model_hash,
            "golden_set_path": str(golden_path.resolve()),
            "golden_set_hash": golden_hash,
            "policy_path": str(policy_path.resolve()),
            "policy_hash": policy_hash,
            "policy": policy,
            "metrics": metrics,
            "cached": False,
        }
        artifact_paths = self._write_deployment_gate_artifacts(
            gate_root=gate_root,
            payload=report_payload,
            predictions=predictions,
        )
        report_payload.update(artifact_paths)

        if not gate_pass:
            raise RuntimeError(
                "Deployment gate rejected this model.\n\n"
                + "\n".join(f"- {reason}" for reason in failed_checks)
                + f"\n\nReport: {artifact_paths['report_path']}"
            )

        ctx.emit(
            "progress",
            (
                "Deployment gate PASSED. "
                f"accuracy={float(metrics.get('accuracy', 0.0)):.4f}, "
                f"weighted_f1={float(metrics.get('weighted_f1', 0.0)):.4f}, "
                f"macro_f1={float(metrics.get('macro_f1', 0.0)):.4f}"
            ),
        )
        return report_payload

    def _attach_deployment_gate_to_result(self, result: dict[str, Any], gate_payload: dict[str, Any]) -> None:
        gate_pass = bool(gate_payload.get("gate_pass"))
        if not gate_pass:
            return
        gate_summary = (
            "Deployment gate: PASS"
            + (" (cache hit)" if bool(gate_payload.get("cached")) else "")
            + f" | accuracy={float((gate_payload.get('metrics') or {}).get('accuracy', 0.0)):.4f}"
            + f" | weighted_f1={float((gate_payload.get('metrics') or {}).get('weighted_f1', 0.0)):.4f}"
        )
        result["gate"] = {
            "gate_pass": True,
            "cached": bool(gate_payload.get("cached")),
            "report_path": clean_optional_string(gate_payload.get("report_path")),
            "predictions_path": clean_optional_string(gate_payload.get("predictions_path")),
            "model_hash": clean_optional_string(gate_payload.get("model_hash")),
            "golden_set_hash": clean_optional_string(gate_payload.get("golden_set_hash")),
            "policy_hash": clean_optional_string(gate_payload.get("policy_hash")),
            "sample_count": int(gate_payload.get("sample_count") or 0),
            "metrics": gate_payload.get("metrics", {}),
            "golden_set_path": clean_optional_string(gate_payload.get("golden_set_path")),
            "policy_path": clean_optional_string(gate_payload.get("policy_path")),
        }
        result["message"] = (
            clean_optional_string(result.get("message")) + "\n" + gate_summary
            if clean_optional_string(result.get("message"))
            else gate_summary
        )
        result["summary"] = (
            clean_optional_string(result.get("summary")) + "\n" + gate_summary
            if clean_optional_string(result.get("summary"))
            else gate_summary
        )

        try:
            hosting_meta = self.model_catalog_service.read_last_hosting_metadata() or {}
            hosting_meta["deployment_gate"] = result["gate"]
            result["metadata_path"] = self.model_catalog_service.save_last_hosting_metadata(hosting_meta)
        except Exception:
            pass

    def _attach_drift_monitoring_to_result(self, ctx: JobContext, request: HostingRequest, result: dict[str, Any]) -> None:
        if request.mode == "azure" and not clean_optional_string(request.model_dir):
            result["drift_monitoring"] = {
                "status": "skipped",
                "reason": "Azure hosting used a registered Azure ML model, so no local model directory was available for drift baseline evaluation.",
            }
            return
        drift_golden = clean_optional_string(request.drift_golden_path) or "gates/drift_golden.csv"
        drift_policy = clean_optional_string(request.drift_policy_path) or "gates/drift_policy.json"
        deployment_id = clean_optional_string(result.get("endpoint_name")) or clean_optional_string(result.get("api_url"))
        service_kind = clean_optional_string(request.azure_service) if request.mode == "azure" else "local"
        summary_prefix = "Drift monitoring baseline:"
        drift_binding = {
            "golden_set_path": str(self.observability_service.resolve_drift_input_path(drift_golden, "gates/drift_golden.csv")),
            "policy_path": str(self.observability_service.resolve_drift_input_path(drift_policy, "gates/drift_policy.json")),
        }

        try:
            ctx.emit("progress", "Running drift monitoring baseline against drift golden set...")
            drift_payload = self.observability_service.evaluate_drift_for_model(
                model_dir=request.model_dir,
                golden_path=drift_golden,
                policy_path=drift_policy,
                deployment_id=deployment_id,
                endpoint_name=clean_optional_string(result.get("endpoint_name")),
                mode=request.mode,
                service_kind=service_kind,
                source="deploy_baseline",
                emit=lambda msg: ctx.emit("progress", msg),
            )
            status = clean_optional_string(drift_payload.get("status")) or "unknown"
            metrics = drift_payload.get("metrics", {}) if isinstance(drift_payload.get("metrics"), dict) else {}
            drift_summary = (
                f"{summary_prefix} {status.upper()} "
                f"| accuracy={float(metrics.get('accuracy', 0.0)):.4f} "
                f"| weighted_f1={float(metrics.get('weighted_f1', 0.0)):.4f}"
            )
            result["drift_monitoring"] = {
                "status": status,
                "report_path": clean_optional_string(drift_payload.get("report_path")),
                "predictions_path": clean_optional_string(drift_payload.get("predictions_path")),
                "golden_set_path": clean_optional_string(drift_payload.get("golden_set_path")),
                "golden_set_hash": clean_optional_string(drift_payload.get("golden_set_hash")),
                "policy_path": clean_optional_string(drift_payload.get("policy_path")),
                "policy_hash": clean_optional_string(drift_payload.get("policy_hash")),
                "sample_count": int(drift_payload.get("sample_count") or 0),
                "metrics": metrics,
                "warning_failures": drift_payload.get("warning_failures", []),
                "critical_failures": drift_payload.get("critical_failures", []),
            }
            result["summary"] = (
                clean_optional_string(result.get("summary")) + "\n" + drift_summary
                if clean_optional_string(result.get("summary"))
                else drift_summary
            )
            result["message"] = (
                clean_optional_string(result.get("message")) + "\n" + drift_summary
                if clean_optional_string(result.get("message"))
                else drift_summary
            )
            drift_binding.update(
                {
                    "status": status,
                    "report_path": clean_optional_string(drift_payload.get("report_path")),
                    "predictions_path": clean_optional_string(drift_payload.get("predictions_path")),
                    "metrics": metrics,
                    "sample_count": int(drift_payload.get("sample_count") or 0),
                    "last_checked_at": clean_optional_string(drift_payload.get("created_at")),
                    "warning_failures": drift_payload.get("warning_failures", []),
                    "critical_failures": drift_payload.get("critical_failures", []),
                }
            )
        except Exception as exc:
            drift_error = clean_optional_string(exc) or "Drift monitoring baseline failed."
            result["drift_monitoring_error"] = drift_error
            result["summary"] = (
                clean_optional_string(result.get("summary"))
                + "\n"
                + summary_prefix
                + " ERROR: "
                + drift_error
            )
            drift_binding.update({"status": "error", "error": drift_error})

        try:
            hosting_meta = self.model_catalog_service.read_last_hosting_metadata() or {}
            hosting_meta["drift_monitoring"] = drift_binding
            result["metadata_path"] = self.model_catalog_service.save_last_hosting_metadata(hosting_meta)
        except Exception:
            pass

    def _run(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        gate_payload: dict[str, Any] = {}
        if request.mode == "azure" and not clean_optional_string(request.model_dir):
            ctx.emit(
                "progress",
                "Skipping local deployment gate because Azure hosting is using a registered Azure ML model.",
            )
        else:
            gate_payload = self._enforce_deployment_gate(ctx, request)
        if request.mode == "azure":
            result = self._run_azure_hosting(ctx, request)
        else:
            result = self._run_local_hosting(ctx, request)
        if gate_payload:
            self._attach_deployment_gate_to_result(result, gate_payload)
        self._attach_drift_monitoring_to_result(ctx, request, result)
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

        endpoint_url = (
            clean_optional_string(result.get("triage_api_url"))
            or clean_optional_string(result.get("log_api_url"))
            or clean_optional_string(result.get("api_url"))
        )
        if not endpoint_url:
            result["github_pr_error"] = "Could not create the Copilot PR task because no endpoint URL was returned."
            result["summary"] = clean_optional_string(result.get("summary")) + f"\nCopilot PR task: {result['github_pr_error']}"
            persist_pr_metadata({"github_pr_error": result["github_pr_error"]})
            return
        ctx.emit("progress", "Creating GitHub Copilot PR task...")
        prompt_info: dict[str, Any] = {}
        try:
            azure_studio_endpoint_url = (
                clean_optional_string(result.get("azure_endpoint_studio_url"))
                or clean_optional_string(result.get("serverless_endpoints_studio_url"))
                or clean_optional_string(result.get("mlops_url"))
            )
            prompt_text = self.github_service.build_log_forwarding_copilot_prompt(
                repo_name=request.github_repo,
                base_branch=request.github_branch,
                endpoint_url=endpoint_url,
                endpoint_name=clean_optional_string(result.get("endpoint_name")),
                endpoint_auth_mode=clean_optional_string(result.get("endpoint_auth_mode")),
                service_kind=clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                hosting_mode=request.mode,
                azure_studio_endpoint_url=azure_studio_endpoint_url,
            )
            prompt_info = self.mlops_service.archive_copilot_pr_prompt(
                prompt_text,
                {
                    "repo_name": request.github_repo,
                    "base_branch": request.github_branch,
                    "endpoint_url": endpoint_url,
                    "endpoint_name": clean_optional_string(result.get("endpoint_name")),
                    "azure_studio_endpoint_url": azure_studio_endpoint_url,
                    "service_kind": clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                    "hosting_mode": request.mode,
                    "copilot_model": "github-default-best-available",
                    "copilot_assignee": "copilot-swe-agent[bot]",
                },
            )
            prompt_text = self.github_service.build_log_forwarding_copilot_prompt(
                repo_name=request.github_repo,
                base_branch=request.github_branch,
                endpoint_url=endpoint_url,
                endpoint_name=clean_optional_string(result.get("endpoint_name")),
                endpoint_auth_mode=clean_optional_string(result.get("endpoint_auth_mode")),
                service_kind=clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                hosting_mode=request.mode,
                azure_studio_endpoint_url=azure_studio_endpoint_url,
                copilot_prompt_version_label=clean_optional_string(prompt_info.get("copilot_prompt_version_label")),
                copilot_prompt_version_id=clean_optional_string(prompt_info.get("copilot_prompt_version_id")),
            )
            prompt_info = self.mlops_service.archive_copilot_pr_prompt(
                prompt_text,
                {
                    "repo_name": request.github_repo,
                    "base_branch": request.github_branch,
                    "endpoint_url": endpoint_url,
                    "endpoint_name": clean_optional_string(result.get("endpoint_name")),
                    "azure_studio_endpoint_url": azure_studio_endpoint_url,
                    "service_kind": clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                    "hosting_mode": request.mode,
                    "copilot_model": "github-default-best-available",
                    "copilot_assignee": "copilot-swe-agent[bot]",
                    "bootstrap_copilot_prompt_version_id": clean_optional_string(prompt_info.get("copilot_prompt_version_id")),
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
                azure_studio_endpoint_url=azure_studio_endpoint_url,
            )
            prompt_info.update(
                self.mlops_service.archive_copilot_pr_prompt(
                    prompt_text,
                    {
                        "repo_name": request.github_repo,
                        "base_branch": request.github_branch,
                        "endpoint_url": endpoint_url,
                        "endpoint_name": clean_optional_string(result.get("endpoint_name")),
                        "azure_studio_endpoint_url": azure_studio_endpoint_url,
                        "service_kind": clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                        "hosting_mode": request.mode,
                        "github_issue_number": pr_task.get("issue_number"),
                        "github_issue_url": clean_optional_string(pr_task.get("html_url")),
                        "copilot_model": clean_optional_string(pr_task.get("copilot_model")),
                        "copilot_assignee": clean_optional_string(pr_task.get("copilot_assignee")),
                    },
                )
            )
            prompt_mlflow_info = self.mlops_service.log_copilot_pr_prompt_mlflow(
                tracking_uri=clean_optional_string(result.get("azure_mlflow_tracking_uri")),
                experiment_name="log-monitor-copilot-prompts",
                prompt_text=prompt_text,
                prompt_info=prompt_info,
                metadata={
                    "repo_name": request.github_repo,
                    "base_branch": request.github_branch,
                    "endpoint_url": endpoint_url,
                    "endpoint_name": clean_optional_string(result.get("endpoint_name")),
                    "azure_studio_endpoint_url": azure_studio_endpoint_url,
                    "service_kind": clean_optional_string(request.azure_service) or clean_optional_string(result.get("service_kind")),
                    "hosting_mode": request.mode,
                    "github_issue_url": clean_optional_string(pr_task.get("html_url")),
                    "copilot_model": clean_optional_string(pr_task.get("copilot_model")),
                },
            )
            prompt_info.update(prompt_mlflow_info)
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

    def _resolve_feedback_base_dataset_path(self, training_metadata: dict[str, Any]) -> str:
        candidates: list[Path] = []
        data_version_id = clean_optional_string(training_metadata.get("data_version_id"))
        if data_version_id:
            local_data_path = self.project_dir / "outputs" / "data_versions" / data_version_id / "dataset.csv"
            candidates.append(local_data_path)
        for key in ("data_version_path", "archived_dataset_path", "source_dataset_path"):
            value = clean_optional_string(training_metadata.get(key))
            if value:
                candidates.append(Path(value).expanduser())
        data_version_dir = clean_optional_string(training_metadata.get("data_version_dir"))
        if data_version_dir:
            candidates.append(Path(data_version_dir).expanduser() / "dataset.csv")
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists() and resolved.is_file():
                return str(resolved)
        return ""

    def _build_feedback_retrain_args(self, training_metadata: dict[str, Any]) -> str:
        default_config = {
            "train_mode": "default",
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_length": 128,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "cv_folds": 3,
            "max_trials": 1,
        }
        selection_summary = training_metadata.get("selection_summary")
        best_config = selection_summary.get("best_config") if isinstance(selection_summary, dict) else {}
        if isinstance(best_config, dict):
            for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "max_length"):
                if best_config.get(key) not in (None, ""):
                    default_config[key] = best_config[key]
        args = [
            "--train-mode",
            str(default_config["train_mode"]),
            "--epochs",
            str(default_config["epochs"]),
            "--batch-size",
            str(default_config["batch_size"]),
            "--learning-rate",
            str(default_config["learning_rate"]),
            "--weight-decay",
            str(default_config["weight_decay"]),
            "--max-length",
            str(default_config["max_length"]),
            "--val-ratio",
            str(default_config["val_ratio"]),
            "--test-ratio",
            str(default_config["test_ratio"]),
            "--cv-folds",
            str(default_config["cv_folds"]),
            "--max-trials",
            str(default_config["max_trials"]),
        ]
        return " ".join(shlex.quote(arg) for arg in args)

    def _deploy_azure_feedback_bridge(
        self,
        *,
        ctx: JobContext,
        credential: Any,
        ml_client: Any,
        request: HostingRequest,
        service_kind: str,
        timestamp: int,
        training_metadata: dict[str, Any],
        source_endpoint_name: str,
        source_api_url: str,
        batch_enabled: bool = False,
        batch_endpoint_name: str = "",
        batch_deployment_name: str = "",
        retrain_compute_name: str = "",
        prediction_key: str = "",
    ) -> dict[str, Any]:
        clean_service_kind = clean_optional_string(service_kind) or "azure"
        function_app_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-func-{clean_service_kind}-{timestamp}", max_length=60)
        function_plan_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-flex-{clean_service_kind}-{timestamp}", max_length=40)
        storage_account_name = self.azure_platform_service.sanitize_azure_storage_name(f"logmonitor{timestamp}")
        service_bus_namespace_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-sb-{clean_service_kind}-{timestamp}", max_length=50)
        service_bus_queue_name = "logs"
        datastore_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-feedback-{timestamp}", max_length=30).replace("-", "")
        blob_container_name = "log-batches"

        ctx.emit("progress", "Creating Azure Function feedback bridge infrastructure...")
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
            raise RuntimeError("Azure feedback bridge deployment did not return the required connection details.")

        ctx.emit("progress", "Registering Azure Blob datastore for feedback datasets...")
        self.azure_platform_service.ensure_azure_blob_datastore(
            ml_client=ml_client,
            datastore_name=datastore_name,
            storage_account_name=storage_account_name,
            container_name=blob_container_name,
            storage_account_key=storage_account_key,
        )

        base_dataset_path = self._resolve_feedback_base_dataset_path(training_metadata)
        base_dataset_blob = ""
        if base_dataset_path:
            base_dataset_blob = f"feedback/base/{timestamp}/dataset.csv"
            ctx.emit("progress", "Uploading the current labeled dataset for feedback correction...")
            self.azure_platform_service.upload_blob_file(
                storage_connection_string=storage_connection_string,
                container_name=blob_container_name,
                source_path=base_dataset_path,
                blob_name=base_dataset_blob,
            )

        retrain_instance_type = clean_optional_string(request.azure_instance_type) or "Standard_D2as_v4"
        retrain_compute = clean_optional_string(retrain_compute_name) or "log-monitor-feedback-cpu"
        batch_timezone_iana = self.azure_platform_service.get_azure_batch_timezone_iana(request.batch_timezone)
        triage_enabled = bool(request.triage_enabled)
        prediction_endpoint_key = clean_optional_string(prediction_key)
        if triage_enabled and not prediction_endpoint_key:
            raise RuntimeError("Azure real-time triage needs the Azure ML online endpoint key, but no key was available.")
        settings = {
            "AzureWebJobsStorage__accountName": storage_account_name,
            "LOGMONITOR_STORAGE_CONNECTION": storage_connection_string,
            "LOGMONITOR_BLOB_CONTAINER": blob_container_name,
            "LOGMONITOR_SERVICEBUS_CONNECTION": service_bus_connection_string,
            "LOGMONITOR_QUEUE_NAME": service_bus_queue_name,
            "LOGMONITOR_BATCH_ENABLED": "1" if batch_enabled else "0",
            "LOGMONITOR_BATCH_TIME": f"{request.batch_hour:02d}:{request.batch_minute:02d}",
            "LOGMONITOR_BATCH_TIME_ZONE": batch_timezone_iana,
            "LOGMONITOR_BATCH_ENDPOINT_NAME": clean_optional_string(batch_endpoint_name),
            "LOGMONITOR_BATCH_DEPLOYMENT_NAME": clean_optional_string(batch_deployment_name),
            "LOGMONITOR_AML_SUBSCRIPTION_ID": request.azure_sub_id,
            "LOGMONITOR_AML_RESOURCE_GROUP": self.azure_platform_service.resource_group,
            "LOGMONITOR_AML_WORKSPACE_NAME": self.azure_platform_service.workspace_name,
            "LOGMONITOR_DATASTORE_NAME": datastore_name,
            "LOGMONITOR_INPUT_PREFIX": "queue-batches",
            "LOGMONITOR_STATE_BLOB": "queue-state/scheduler-state.json",
            "LOGMONITOR_FEEDBACK_STATE_BLOB": "feedback/state.json",
            "LOGMONITOR_MONITORING_STATE_BLOB": "monitoring/prediction-summary-state.json",
            "LOGMONITOR_FEEDBACK_DATASET_PREFIX": "feedback/datasets",
            "LOGMONITOR_FEEDBACK_EVENTS_PREFIX": "feedback/events",
            "LOGMONITOR_FEEDBACK_DATA_ASSET_NAME": "log-monitor-feedback-labeled-data",
            "LOGMONITOR_BASE_DATASET_BLOB": base_dataset_blob,
            "LOGMONITOR_RETRAIN_ENABLED": "1",
            "LOGMONITOR_RETRAIN_COMPUTE_NAME": retrain_compute,
            "LOGMONITOR_RETRAIN_INSTANCE_TYPE": retrain_instance_type,
            "LOGMONITOR_RETRAIN_TRAIN_ARGS": self._build_feedback_retrain_args(training_metadata),
            "LOGMONITOR_RETRAIN_EXPERIMENT_NAME": "log-monitor-feedback-retraining",
            "LOGMONITOR_HOSTING_SERVICE_KIND": clean_service_kind,
            "LOGMONITOR_SOURCE_ENDPOINT_NAME": clean_optional_string(source_endpoint_name),
            "LOGMONITOR_SOURCE_ENDPOINT_URL": clean_optional_string(source_api_url),
            "LOGMONITOR_TRIAGE_ENABLED": "1" if triage_enabled else "0",
            "LOGMONITOR_PREDICTION_ENDPOINT_URL": clean_optional_string(source_api_url),
            "LOGMONITOR_PREDICTION_AUTH_MODE": "key",
            "LOGMONITOR_PREDICTION_KEY": prediction_endpoint_key,
            "LOGMONITOR_CONFIGURATION_EMAIL": clean_optional_string(request.configuration_email),
            "LOGMONITOR_SYSTEM_EMAIL": clean_optional_string(request.system_email),
            "LOGMONITOR_ACS_CONNECTION_STRING": clean_optional_string(request.acs_connection_string),
            "LOGMONITOR_ACS_SENDER_ADDRESS": clean_optional_string(request.acs_sender_address),
            "LOGMONITOR_GITHUB_TOKEN": clean_optional_string(request.github_token),
            "LOGMONITOR_GITHUB_REPO": clean_optional_string(request.github_repo),
            "LOGMONITOR_GITHUB_BRANCH": clean_optional_string(request.github_branch),
            "LOGMONITOR_JIRA_SITE_URL": clean_optional_string(request.jira_site_url),
            "LOGMONITOR_JIRA_ACCOUNT_EMAIL": clean_optional_string(request.jira_account_email),
            "LOGMONITOR_JIRA_API_TOKEN": clean_optional_string(request.jira_api_token),
            "LOGMONITOR_JIRA_PROJECT_KEY": clean_optional_string(request.jira_project_key),
            "LOGMONITOR_JIRA_ISSUE_TYPE": clean_optional_string(request.jira_issue_type) or "Bug",
            "LOGMONITOR_JIRA_PRIORITY": clean_optional_string(request.jira_priority),
            "LOGMONITOR_JIRA_LABELS": clean_optional_string(request.jira_labels) or "log-monitor,ml-triage",
            "LOGMONITOR_JIRA_MONITORING_ENABLED": "1" if triage_enabled else "0",
            "LOGMONITOR_JIRA_MONITORING_ISSUE_TYPE": "Task",
            "LOGMONITOR_JIRA_MONITORING_LABELS": "log-monitor,ml-monitoring,prediction-summary",
        }
        ctx.emit("progress", "Updating Azure Function feedback settings...")
        self.azure_platform_service.set_function_app_settings(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            settings=settings,
        )
        ctx.emit("progress", "Packaging the Azure Function feedback bridge...")
        package_path = self.azure_platform_service.build_function_bridge_package(f"log-monitor-function-{clean_service_kind}-{timestamp}")
        package_uri = self.azure_platform_service.upload_function_bridge_package(
            storage_connection_string=storage_connection_string,
            storage_account_name=storage_account_name,
            storage_account_key=storage_account_key,
            package_path=package_path,
            package_container_name="functionpkgs",
        )
        ctx.emit("progress", "Deploying the Azure Function feedback bridge...")
        self.azure_platform_service.trigger_function_app_onedeploy(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            package_uri=package_uri,
        )
        log_api_url, function_key = self.azure_platform_service.wait_for_function_bridge_endpoint(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            function_host_name=function_host_name,
            function_name="ingest_log",
            route_path="logs",
        )
        feedback_api_url, _ = self.azure_platform_service.wait_for_function_bridge_endpoint(
            credential=credential,
            sub_id=request.azure_sub_id,
            function_app_name=function_app_name,
            function_host_name=function_host_name,
            function_name="submit_feedback",
            route_path="feedback",
        )
        triage_api_url = ""
        if triage_enabled:
            triage_api_url, _ = self.azure_platform_service.wait_for_function_bridge_endpoint(
                credential=credential,
                sub_id=request.azure_sub_id,
                function_app_name=function_app_name,
                function_host_name=function_host_name,
                function_name="triage_log",
                route_path="triage",
            )
        feedback_status_url = f"https://{function_host_name}/api/feedback/status?code={function_key}" if function_key else ""
        return {
            "function_app_name": function_app_name,
            "function_host_name": function_host_name,
            "function_key": function_key,
            "log_api_url": log_api_url,
            "triage_enabled": triage_enabled,
            "triage_api_url": triage_api_url,
            "feedback_api_url": feedback_api_url,
            "feedback_status_url": feedback_status_url,
            "service_bus_namespace": service_bus_namespace_name,
            "service_bus_queue": service_bus_queue_name,
            "storage_account_name": storage_account_name,
            "log_container_name": blob_container_name,
            "datastore_name": datastore_name,
            "base_dataset_path": base_dataset_path,
            "base_dataset_blob": base_dataset_blob,
            "retrain_compute_name": retrain_compute,
            "retrain_instance_type": retrain_instance_type,
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
        azure_mlflow_tracking_uri = clean_optional_string(self.mlops_service.resolve_azure_mlflow_tracking_uri(ml_client))
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
        training_metadata = {}
        if clean_optional_string(request.model_dir):
            training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        ctx.emit("progress", "Deploying feedback API for corrected labels and retraining...")
        feedback_meta = self._deploy_azure_feedback_bridge(
            ctx=ctx,
            credential=credential,
            ml_client=ml_client,
            request=request,
            service_kind="serverless",
            timestamp=timestamp,
            training_metadata=training_metadata,
            source_endpoint_name=clean_optional_string(deployment_meta.get("endpoint_name")) or endpoint_name,
            source_api_url=scoring_uri,
            batch_enabled=False,
        )
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
                "azure_endpoint_studio_url": endpoints_studio_url,
                "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
                "api_url": scoring_uri,
                "log_api_url": clean_optional_string(feedback_meta.get("log_api_url")),
                "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
                "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
                "feedback_bridge": feedback_meta,
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
            "log_api_url": clean_optional_string(feedback_meta.get("log_api_url")),
            "endpoint_name": clean_optional_string(deployment_meta.get("endpoint_name")) or endpoint_name,
            "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
            "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
            "mlops_url": endpoints_studio_url or mlops_url,
            "azure_endpoint_studio_url": endpoints_studio_url,
            "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure serverless endpoint is ready.\nTarget URI: {scoring_uri}\n"
                f"Feedback API: {clean_optional_string(feedback_meta.get('feedback_api_url'))}\n"
                f"Model ID: {model_id}\nEndpoint Name: {clean_optional_string(deployment_meta.get('endpoint_name')) or endpoint_name}\n"
                f"Creation: {creation_method}\n"
                f"{visibility_note}\n"
                f"Studio: {endpoints_studio_url}\nPortal Resource: {endpoint_portal_url}\n"
                "Authentication: endpoint keys from Azure ML Studio; feedback API uses its Azure Function key."
            ),
        }

    def _run_azure_online_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        ctx.emit("progress", "Preparing Azure hosting...")
        credential = self.azure_platform_service.create_interactive_credential(request.azure_tenant_id)
        ml_client = self.azure_platform_service.ensure_azure_workspace(
            request.azure_sub_id,
            request.azure_tenant_id,
            emit=lambda msg: ctx.emit("progress", msg),
            credential=credential,
        )
        timestamp = int(time.time())
        model_hint = clean_optional_string(request.azure_model_name) or clean_optional_string(request.azure_model_id) or Path(clean_optional_string(request.model_dir) or "model").name
        model_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-model-{model_hint}-{timestamp}")
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
            azure_model_id=request.azure_model_id,
            azure_model_name=request.azure_model_name,
            azure_model_version=request.azure_model_version,
            emit=lambda msg: ctx.emit("progress", msg),
        )
        scoring_uri = clean_optional_string(deployment_meta.get("api_url"))
        selected_instance_type = clean_optional_string(deployment_meta.get("instance_type"))
        prediction_endpoint_key = ""
        if request.triage_enabled:
            prediction_endpoint_key = self.azure_platform_service.get_online_endpoint_key(
                ml_client,
                endpoint_name,
                emit=lambda msg: ctx.emit("progress", msg),
            )
        endpoints_studio_url = self.azure_platform_service.build_azure_endpoints_studio_url(request.azure_sub_id, request.azure_tenant_id)
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        azure_mlflow_tracking_uri = clean_optional_string(self.mlops_service.resolve_azure_mlflow_tracking_uri(ml_client))
        training_metadata = {}
        if clean_optional_string(request.model_dir):
            training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        ctx.emit("progress", "Deploying feedback API for corrected labels and retraining...")
        feedback_meta = self._deploy_azure_feedback_bridge(
            ctx=ctx,
            credential=credential,
            ml_client=ml_client,
            request=request,
            service_kind="online",
            timestamp=timestamp,
            training_metadata=training_metadata,
            source_endpoint_name=endpoint_name,
            source_api_url=scoring_uri,
            batch_enabled=False,
            prediction_key=prediction_endpoint_key,
        )
        triage_api_url = clean_optional_string(feedback_meta.get("triage_api_url"))
        public_api_url = triage_api_url or scoring_uri
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure",
                "service_kind": "online",
                "model_dir": clean_optional_string(request.model_dir),
                "azure_model_id": clean_optional_string(deployment_meta.get("azure_model_id")) or clean_optional_string(request.azure_model_id),
                "azure_model_name": clean_optional_string(deployment_meta.get("azure_model_name")) or clean_optional_string(request.azure_model_name),
                "azure_model_version": clean_optional_string(deployment_meta.get("azure_model_version")) or clean_optional_string(request.azure_model_version),
                "azure_model_label": clean_optional_string(request.azure_model_label),
                "model_version_id": clean_optional_string(training_metadata.get("model_version_id", "")),
                "training_run_id": clean_optional_string(training_metadata.get("run_id", "")),
                "data_version_id": clean_optional_string(training_metadata.get("data_version_id", "")),
                "endpoint_name": endpoint_name,
                "deployment_name": deployment_name,
                "instance_type": selected_instance_type,
                "endpoint_auth_mode": "key",
                "api_url": public_api_url,
                "log_api_url": clean_optional_string(feedback_meta.get("log_api_url")),
                "prediction_api_url": scoring_uri,
                "triage_enabled": bool(feedback_meta.get("triage_enabled")),
                "triage_api_url": triage_api_url,
                "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
                "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
                "feedback_bridge": feedback_meta,
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "azure_compute": request.azure_compute,
                "mlops_url": mlops_url,
                "azure_endpoint_studio_url": endpoints_studio_url,
                "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure real-time endpoint is ready.",
            "api_url": public_api_url,
            "log_api_url": clean_optional_string(feedback_meta.get("log_api_url")),
            "prediction_api_url": scoring_uri,
            "endpoint_name": endpoint_name,
            "endpoint_auth_mode": "key",
            "azure_model_id": clean_optional_string(deployment_meta.get("azure_model_id")) or clean_optional_string(request.azure_model_id),
            "azure_model_name": clean_optional_string(deployment_meta.get("azure_model_name")) or clean_optional_string(request.azure_model_name),
            "azure_model_version": clean_optional_string(deployment_meta.get("azure_model_version")) or clean_optional_string(request.azure_model_version),
            "triage_api_url": triage_api_url,
            "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
            "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
            "mlops_url": mlops_url,
            "azure_endpoint_studio_url": endpoints_studio_url,
            "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure real-time endpoint is ready.\nPOST {public_api_url}\n"
                f"Azure Model: {clean_optional_string(request.azure_model_label) or clean_optional_string(request.azure_model_name)}\n"
                f"Prediction target: {scoring_uri}\nInstance Type: {selected_instance_type}\n"
                f"Feedback API: {clean_optional_string(feedback_meta.get('feedback_api_url'))}\n"
                + (f"Triage API: {triage_api_url}\n" if triage_api_url else "")
                + f"Studio: {endpoints_studio_url}\n"
                + "Body: {\"errorMessage\": \"...\"}\nResponse: {\"prediction\": \"...\"}\nAuthentication: endpoint keys."
            ),
        }

    def _run_azure_batch_hosting(self, ctx: JobContext, request: HostingRequest) -> dict[str, Any]:
        if not AZURE_AVAILABLE:
            raise RuntimeError("Azure dependencies are not installed in this Python environment.")
        ctx.emit("progress", "Preparing Azure hosting...")
        credential = self.azure_platform_service.create_interactive_credential(request.azure_tenant_id)
        ml_client = self.azure_platform_service.ensure_azure_workspace(
            request.azure_sub_id,
            request.azure_tenant_id,
            emit=lambda msg: ctx.emit("progress", msg),
            credential=credential,
        )
        timestamp = int(time.time())
        endpoint_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-endpoint-{timestamp}")
        schedule_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-schedule-{timestamp}")
        env_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-env-{timestamp}")
        model_hint = clean_optional_string(request.azure_model_name) or clean_optional_string(request.azure_model_id) or Path(clean_optional_string(request.model_dir) or "model").name
        model_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-model-{model_hint}-{timestamp}")
        deployment_meta = self.azure_platform_service.deploy_azure_batch_endpoint(
            ml_client=ml_client,
            model_dir=request.model_dir,
            azure_compute=request.azure_compute,
            preferred_instance_type=request.azure_instance_type,
            endpoint_name=endpoint_name,
            environment_name=env_name,
            model_name=model_name,
            endpoint_auth_mode="aad_token",
            azure_model_id=request.azure_model_id,
            azure_model_name=request.azure_model_name,
            azure_model_version=request.azure_model_version,
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
        endpoints_studio_url = self.azure_platform_service.build_azure_endpoints_studio_url(request.azure_sub_id, request.azure_tenant_id)
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        azure_mlflow_tracking_uri = clean_optional_string(self.mlops_service.resolve_azure_mlflow_tracking_uri(ml_client))
        training_metadata = {}
        if clean_optional_string(request.model_dir):
            training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        ctx.emit("progress", "Deploying feedback API for corrected labels and retraining...")
        feedback_meta = self._deploy_azure_feedback_bridge(
            ctx=ctx,
            credential=credential,
            ml_client=ml_client,
            request=request,
            service_kind="batch",
            timestamp=timestamp,
            training_metadata=training_metadata,
            source_endpoint_name=endpoint_name,
            source_api_url=clean_optional_string(deployment_meta.get("api_url")),
            batch_enabled=False,
            retrain_compute_name=clean_optional_string(deployment_meta.get("compute_name")),
        )
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure_batch",
                "service_kind": "batch",
                "model_dir": clean_optional_string(request.model_dir),
                "azure_model_id": clean_optional_string(deployment_meta.get("azure_model_id")) or clean_optional_string(request.azure_model_id),
                "azure_model_name": clean_optional_string(deployment_meta.get("azure_model_name")) or clean_optional_string(request.azure_model_name),
                "azure_model_version": clean_optional_string(deployment_meta.get("azure_model_version")) or clean_optional_string(request.azure_model_version),
                "azure_model_label": clean_optional_string(request.azure_model_label),
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
                "log_api_url": clean_optional_string(feedback_meta.get("log_api_url")),
                "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
                "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
                "feedback_bridge": feedback_meta,
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "azure_compute": request.azure_compute,
                "mlops_url": mlops_url,
                "azure_endpoint_studio_url": endpoints_studio_url,
                "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure batch endpoint and daily schedule are ready.",
            "api_url": clean_optional_string(deployment_meta.get("api_url")),
            "log_api_url": clean_optional_string(feedback_meta.get("log_api_url")),
            "endpoint_name": endpoint_name,
            "azure_model_id": clean_optional_string(deployment_meta.get("azure_model_id")) or clean_optional_string(request.azure_model_id),
            "azure_model_name": clean_optional_string(deployment_meta.get("azure_model_name")) or clean_optional_string(request.azure_model_name),
            "azure_model_version": clean_optional_string(deployment_meta.get("azure_model_version")) or clean_optional_string(request.azure_model_version),
            "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
            "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
            "mlops_url": mlops_url,
            "azure_endpoint_studio_url": endpoints_studio_url,
            "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure batch endpoint is ready.\nInvoke: {clean_optional_string(deployment_meta.get('api_url'))}\n"
                f"Azure Model: {clean_optional_string(request.azure_model_label) or clean_optional_string(request.azure_model_name)}\n"
                f"Feedback API: {clean_optional_string(feedback_meta.get('feedback_api_url'))}\n"
                f"Studio: {endpoints_studio_url}\n"
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
        batch_endpoint_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-endpoint-{timestamp}")
        batch_env_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-batch-env-{timestamp}")
        model_hint = clean_optional_string(request.azure_model_name) or clean_optional_string(request.azure_model_id) or Path(clean_optional_string(request.model_dir) or "model").name
        model_name = self.azure_platform_service.sanitize_azure_name(f"log-monitor-model-{model_hint}-{timestamp}")
        deployment_meta = self.azure_platform_service.deploy_azure_batch_endpoint(
            ml_client=ml_client,
            model_dir=request.model_dir,
            azure_compute=request.azure_compute,
            preferred_instance_type=request.azure_instance_type,
            endpoint_name=batch_endpoint_name,
            environment_name=batch_env_name,
            model_name=model_name,
            endpoint_auth_mode="aad_token",
            azure_model_id=request.azure_model_id,
            azure_model_name=request.azure_model_name,
            azure_model_version=request.azure_model_version,
            emit=lambda msg: ctx.emit("progress", msg),
        )
        deployment_name = clean_optional_string(deployment_meta.get("deployment_name")) or "default"
        endpoints_studio_url = self.azure_platform_service.build_azure_endpoints_studio_url(request.azure_sub_id, request.azure_tenant_id)
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(request.azure_sub_id, request.azure_tenant_id)
        azure_mlflow_tracking_uri = clean_optional_string(self.mlops_service.resolve_azure_mlflow_tracking_uri(ml_client))
        training_metadata = {}
        if clean_optional_string(request.model_dir):
            training_metadata = self.model_catalog_service.find_training_metadata_for_model_dir(Path(request.model_dir))
        ctx.emit("progress", "Deploying queued log API and feedback retraining API...")
        feedback_meta = self._deploy_azure_feedback_bridge(
            ctx=ctx,
            credential=credential,
            ml_client=ml_client,
            request=request,
            service_kind="queued_batch",
            timestamp=timestamp,
            training_metadata=training_metadata,
            source_endpoint_name=batch_endpoint_name,
            source_api_url=clean_optional_string(deployment_meta.get("api_url")),
            batch_enabled=True,
            batch_endpoint_name=batch_endpoint_name,
            batch_deployment_name=deployment_name,
            retrain_compute_name=clean_optional_string(deployment_meta.get("compute_name")),
        )
        log_api_url = clean_optional_string(feedback_meta.get("log_api_url"))
        function_key = clean_optional_string(feedback_meta.get("function_key"))
        metadata_path = self.model_catalog_service.save_last_hosting_metadata(
            {
                "mode": "azure_queue_batch",
                "service_kind": "queued_batch",
                "model_dir": clean_optional_string(request.model_dir),
                "azure_model_id": clean_optional_string(deployment_meta.get("azure_model_id")) or clean_optional_string(request.azure_model_id),
                "azure_model_name": clean_optional_string(deployment_meta.get("azure_model_name")) or clean_optional_string(request.azure_model_name),
                "azure_model_version": clean_optional_string(deployment_meta.get("azure_model_version")) or clean_optional_string(request.azure_model_version),
                "azure_model_label": clean_optional_string(request.azure_model_label),
                "model_version_id": clean_optional_string(training_metadata.get("model_version_id", "")),
                "training_run_id": clean_optional_string(training_metadata.get("run_id", "")),
                "data_version_id": clean_optional_string(training_metadata.get("data_version_id", "")),
                "api_url": log_api_url,
                "log_api_url": log_api_url,
                "function_key": function_key,
                "function_app_name": clean_optional_string(feedback_meta.get("function_app_name")),
                "function_host_name": clean_optional_string(feedback_meta.get("function_host_name")),
                "service_bus_namespace": clean_optional_string(feedback_meta.get("service_bus_namespace")),
                "service_bus_queue": clean_optional_string(feedback_meta.get("service_bus_queue")),
                "storage_account_name": clean_optional_string(feedback_meta.get("storage_account_name")),
                "log_container_name": clean_optional_string(feedback_meta.get("log_container_name")),
                "datastore_name": clean_optional_string(feedback_meta.get("datastore_name")),
                "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
                "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
                "feedback_bridge": feedback_meta,
                "endpoint_name": batch_endpoint_name,
                "deployment_name": deployment_name,
                "batch_endpoint_url": clean_optional_string(deployment_meta.get("api_url")),
                "schedule_time": f"{request.batch_hour:02d}:{request.batch_minute:02d}",
                "schedule_time_zone": request.batch_timezone,
                "schedule_time_zone_iana": self.azure_platform_service.get_azure_batch_timezone_iana(request.batch_timezone),
                "instance_type": clean_optional_string(deployment_meta.get("instance_type")),
                "compute_name": clean_optional_string(deployment_meta.get("compute_name")),
                "endpoint_auth_mode": "aad_token",
                "azure_subscription_id": request.azure_sub_id,
                "azure_tenant_id": request.azure_tenant_id,
                "azure_compute": request.azure_compute,
                "mlops_url": mlops_url,
                "azure_endpoint_studio_url": endpoints_studio_url,
                "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
                "llmops_url": llmops_url,
                "created_at": now_utc_iso(),
            }
        )
        return {
            "operation": "hosting",
            "message": "Azure queued batch pipeline is ready.",
            "api_url": log_api_url,
            "log_api_url": log_api_url,
            "endpoint_name": batch_endpoint_name,
            "azure_model_id": clean_optional_string(deployment_meta.get("azure_model_id")) or clean_optional_string(request.azure_model_id),
            "azure_model_name": clean_optional_string(deployment_meta.get("azure_model_name")) or clean_optional_string(request.azure_model_name),
            "azure_model_version": clean_optional_string(deployment_meta.get("azure_model_version")) or clean_optional_string(request.azure_model_version),
            "feedback_api_url": clean_optional_string(feedback_meta.get("feedback_api_url")),
            "feedback_status_url": clean_optional_string(feedback_meta.get("feedback_status_url")),
            "mlops_url": mlops_url,
            "azure_endpoint_studio_url": endpoints_studio_url,
            "azure_mlflow_tracking_uri": azure_mlflow_tracking_uri,
            "llmops_url": llmops_url,
            "metadata_path": metadata_path,
            "summary": (
                f"Azure queued batch pipeline is ready.\nLog API: {log_api_url}\nFeedback API: {clean_optional_string(feedback_meta.get('feedback_api_url'))}\n"
                f"Azure Model: {clean_optional_string(request.azure_model_label) or clean_optional_string(request.azure_model_name)}\n"
                f"Studio: {endpoints_studio_url}\n"
                f"Queue: {clean_optional_string(feedback_meta.get('service_bus_namespace'))}/{clean_optional_string(feedback_meta.get('service_bus_queue'))}\n"
                f"Batch Endpoint: {batch_endpoint_name}\nSchedule: every day at {request.batch_hour:02d}:{request.batch_minute:02d} {request.batch_timezone}\n"
                f"Cluster: {clean_optional_string(deployment_meta.get('compute_name'))} ({clean_optional_string(deployment_meta.get('instance_type'))}, min nodes 0)\n"
                "Flow: POST logs to the Function API for daily batch scoring, or POST corrected labels to the feedback API for data-versioning and retraining."
            ),
        }
