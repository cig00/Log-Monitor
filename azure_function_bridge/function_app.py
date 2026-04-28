import csv
import hashlib
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import azure.functions as func
from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Data
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.storage.blob import BlobServiceClient, ContentSettings


app = func.FunctionApp()
LOGGER = logging.getLogger("log_monitor_bridge")

MESSAGE_KEYS = (
    "errorMessage",
    "LogMessage",
    "logMessage",
    "message",
    "log",
    "msg",
    "text",
)
LABEL_NAMES = ("Error", "CONFIGURATION", "SYSTEM", "Noise")
LABEL_ALIASES = {label.casefold(): label for label in LABEL_NAMES}

_BLOB_SERVICE_CLIENT = None
_SERVICE_BUS_CLIENT = None
_ML_CLIENT = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _require_env(name: str) -> str:
    value = _get_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_daily_time(raw_value: str) -> tuple[int, int]:
    parts = raw_value.split(":", 1)
    if len(parts) != 2:
        raise ValueError("LOGMONITOR_BATCH_TIME must use HH:MM format.")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("LOGMONITOR_BATCH_TIME must use 24-hour time.")
    return hour, minute


def _get_local_now() -> datetime:
    timezone_name = _get_env("LOGMONITOR_BATCH_TIME_ZONE", "UTC") or "UTC"
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except Exception:
        LOGGER.warning("Falling back to UTC because timezone '%s' was invalid.", timezone_name)
        return datetime.now(timezone.utc)


def _get_state_blob_name() -> str:
    return _get_env("LOGMONITOR_STATE_BLOB", "queue-state/scheduler-state.json")


def _get_feedback_state_blob_name() -> str:
    return _get_env("LOGMONITOR_FEEDBACK_STATE_BLOB", "feedback/state.json")


def _build_blob_name(local_now: datetime) -> str:
    prefix = _get_env("LOGMONITOR_INPUT_PREFIX", "queue-batches").strip("/") or "queue-batches"
    return (
        f"{prefix}/"
        f"{local_now.strftime('%Y-%m-%d')}/"
        f"logs-{local_now.strftime('%H%M%S')}.jsonl"
    )


def _extract_message(record) -> str:
    if isinstance(record, dict):
        for key in MESSAGE_KEYS:
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
        for value in record.values():
            if value not in (None, ""):
                return str(value)
        return ""
    if record in (None, ""):
        return ""
    return str(record)


def _extract_correct_label(record) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("correctLabel", "correct_label", "label", "class"):
        value = record.get(key)
        if value in (None, ""):
            continue
        label = LABEL_ALIASES.get(str(value).strip().casefold(), "")
        if label:
            return label
    return ""


def _normalize_payload(payload) -> dict:
    if isinstance(payload, dict):
        normalized = dict(payload)
    elif isinstance(payload, str):
        normalized = {"message": payload}
    else:
        normalized = {"message": json.dumps(payload, ensure_ascii=True)}
    normalized.setdefault("received_at", _now_utc_iso())
    return normalized


def _to_json_response(body: dict, status_code: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body, ensure_ascii=True),
        status_code=status_code,
        mimetype="application/json",
    )


def _get_blob_service_client() -> BlobServiceClient:
    global _BLOB_SERVICE_CLIENT
    if _BLOB_SERVICE_CLIENT is None:
        _BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(
            _require_env("LOGMONITOR_STORAGE_CONNECTION")
        )
    return _BLOB_SERVICE_CLIENT


def _get_container_client():
    container_name = _require_env("LOGMONITOR_BLOB_CONTAINER")
    return _get_blob_service_client().get_container_client(container_name)


def _download_blob_text(blob_name: str) -> str:
    clean_blob_name = _get_env("LOGMONITOR_BASE_DATASET_BLOB") if not blob_name else blob_name
    if not clean_blob_name:
        return ""
    blob_client = _get_container_client().get_blob_client(clean_blob_name)
    try:
        body = blob_client.download_blob().readall()
    except ResourceNotFoundError:
        return ""
    return body.decode("utf-8-sig")


def _upload_blob_text(blob_name: str, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
    blob_client = _get_container_client().get_blob_client(blob_name)
    blob_client.upload_blob(
        body.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )


def _get_service_bus_client() -> ServiceBusClient:
    global _SERVICE_BUS_CLIENT
    if _SERVICE_BUS_CLIENT is None:
        _SERVICE_BUS_CLIENT = ServiceBusClient.from_connection_string(
            _require_env("LOGMONITOR_SERVICEBUS_CONNECTION")
        )
    return _SERVICE_BUS_CLIENT


def _get_ml_client() -> MLClient:
    global _ML_CLIENT
    if _ML_CLIENT is None:
        credential = DefaultAzureCredential()
        _ML_CLIENT = MLClient(
            credential=credential,
            subscription_id=_require_env("LOGMONITOR_AML_SUBSCRIPTION_ID"),
            resource_group_name=_require_env("LOGMONITOR_AML_RESOURCE_GROUP"),
            workspace_name=_require_env("LOGMONITOR_AML_WORKSPACE_NAME"),
        )
    return _ML_CLIENT


def _load_scheduler_state() -> dict:
    blob_client = _get_container_client().get_blob_client(_get_state_blob_name())
    try:
        body = blob_client.download_blob().readall()
    except ResourceNotFoundError:
        return {}
    except Exception:
        LOGGER.exception("Failed to read scheduler state blob.")
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        LOGGER.exception("Failed to parse scheduler state blob.")
        return {}


def _write_scheduler_state(state: dict) -> None:
    blob_client = _get_container_client().get_blob_client(_get_state_blob_name())
    payload = json.dumps(state, ensure_ascii=True, indent=2)
    blob_client.upload_blob(payload, overwrite=True)


def _load_feedback_state() -> dict:
    blob_client = _get_container_client().get_blob_client(_get_feedback_state_blob_name())
    try:
        body = blob_client.download_blob().readall()
    except ResourceNotFoundError:
        return {}
    except Exception:
        LOGGER.exception("Failed to read feedback state blob.")
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        LOGGER.exception("Failed to parse feedback state blob.")
        return {}


def _write_feedback_state(state: dict) -> None:
    blob_client = _get_container_client().get_blob_client(_get_feedback_state_blob_name())
    payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True)
    blob_client.upload_blob(
        payload.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )


def _upload_batch_blob(blob_name: str, lines: list[str]) -> None:
    if not lines:
        raise RuntimeError("Refusing to upload an empty batch file.")
    blob_client = _get_container_client().get_blob_client(blob_name)
    body = "\n".join(lines) + "\n"
    blob_client.upload_blob(body.encode("utf-8"), overwrite=True)


def _message_body_to_text(message) -> str:
    body = message.body
    if isinstance(body, str):
        return body
    try:
        chunks = []
        for item in body:
            if isinstance(item, (bytes, bytearray)):
                chunks.append(bytes(item))
            else:
                chunks.append(bytes(item))
        return b"".join(chunks).decode("utf-8")
    except Exception:
        return str(message)


def _invoke_batch_endpoint(blob_name: str) -> str:
    datastore_name = _require_env("LOGMONITOR_DATASTORE_NAME")
    endpoint_name = _require_env("LOGMONITOR_BATCH_ENDPOINT_NAME")
    deployment_name = _get_env("LOGMONITOR_BATCH_DEPLOYMENT_NAME")
    datastore_uri = f"azureml://datastores/{datastore_name}/paths/{blob_name}"
    invoke_kwargs = {
        "endpoint_name": endpoint_name,
        "input": Input(path=datastore_uri, type="uri_file"),
        "experiment_name": "log-monitor-queued-batch",
    }
    if deployment_name:
        invoke_kwargs["deployment_name"] = deployment_name
    job = _get_ml_client().batch_endpoints.invoke(**invoke_kwargs)
    return getattr(job, "name", "") or ""


def _read_labeled_csv_rows(csv_text: str) -> list[dict]:
    if not csv_text.strip():
        return []
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = list(reader.fieldnames or [])
    message_col = ""
    label_col = ""
    normalized = {str(name).strip().casefold(): name for name in fieldnames}
    for candidate in ("logmessage", "log_message", "message", "msg", "text", "log", "errormessage"):
        if candidate in normalized:
            message_col = normalized[candidate]
            break
    for candidate in ("class", "label", "correctlabel", "correct_label"):
        if candidate in normalized:
            label_col = normalized[candidate]
            break
    if not message_col or not label_col:
        return []

    rows: list[dict] = []
    for row in reader:
        message = _extract_message(row)
        raw_label = row.get(label_col)
        label = LABEL_ALIASES.get(str(raw_label or "").strip().casefold(), "")
        if message and label:
            rows.append({"LogMessage": message, "class": label})
    return rows


def _rows_to_labeled_csv(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["LogMessage", "class"], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({"LogMessage": row.get("LogMessage", ""), "class": row.get("class", "")})
    return output.getvalue()


def _merge_feedback_row(base_rows: list[dict], message: str, correct_label: str) -> tuple[list[dict], str]:
    clean_message = str(message or "").strip()
    clean_label = LABEL_ALIASES.get(str(correct_label or "").strip().casefold(), "")
    if not clean_message:
        raise ValueError("Feedback payload needs a non-empty log message.")
    if not clean_label:
        raise ValueError("Feedback payload needs correctLabel/class/label set to Error, CONFIGURATION, SYSTEM, or Noise.")

    merged: list[dict] = []
    replaced = False
    for row in base_rows:
        row_message = str(row.get("LogMessage", ""))
        if row_message == clean_message:
            merged.append({"LogMessage": clean_message, "class": clean_label})
            replaced = True
        else:
            row_label = LABEL_ALIASES.get(str(row.get("class", "")).strip().casefold(), "")
            if row_message and row_label:
                merged.append({"LogMessage": row_message, "class": row_label})
    if not replaced:
        merged.append({"LogMessage": clean_message, "class": clean_label})
    return merged, "corrected" if replaced else "appended"


def _feedback_event_blob_name(event_id: str) -> str:
    now = datetime.now(timezone.utc)
    prefix = _get_env("LOGMONITOR_FEEDBACK_EVENTS_PREFIX", "feedback/events").strip("/") or "feedback/events"
    return f"{prefix}/{now.strftime('%Y/%m/%d')}/{event_id}.json"


def _feedback_dataset_blob_name(dataset_hash: str) -> str:
    prefix = _get_env("LOGMONITOR_FEEDBACK_DATASET_PREFIX", "feedback/datasets").strip("/") or "feedback/datasets"
    return f"{prefix}/{dataset_hash}/dataset.csv"


def _register_feedback_data_asset(blob_name: str, dataset_hash: str, row_count: int, event_blob_name: str) -> dict:
    datastore_name = _require_env("LOGMONITOR_DATASTORE_NAME")
    asset_name = _get_env("LOGMONITOR_FEEDBACK_DATA_ASSET_NAME", "log-monitor-feedback-labeled-data")
    asset_version = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in dataset_hash)[:30].strip("-._") or "1"
    datastore_uri = f"azureml://datastores/{datastore_name}/paths/{blob_name}"
    data_asset = Data(
        path=datastore_uri,
        type=AssetTypes.URI_FILE,
        name=asset_name,
        version=asset_version,
        description="Log Monitor corrected labeled dataset created from feedback.",
        tags={
            "created_by": "log-monitor-feedback",
            "dataset_hash": dataset_hash,
            "row_count": str(int(row_count)),
            "feedback_event_blob": event_blob_name,
            "hosting_service_kind": _get_env("LOGMONITOR_HOSTING_SERVICE_KIND", ""),
        },
    )
    registered = _get_ml_client().data.create_or_update(data_asset)
    return {
        "azure_data_asset_name": asset_name,
        "azure_data_asset_version": asset_version,
        "azure_data_asset_uri": f"azureml:{asset_name}:{asset_version}",
        "azure_data_asset_id": str(getattr(registered, "id", "") or ""),
        "azure_data_asset_path": str(getattr(registered, "path", "") or datastore_uri),
    }


def _ensure_retrain_compute() -> str:
    compute_name = _get_env("LOGMONITOR_RETRAIN_COMPUTE_NAME", "log-monitor-feedback-cpu")
    instance_type = _get_env("LOGMONITOR_RETRAIN_INSTANCE_TYPE", "Standard_D2as_v4")
    compute = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size=instance_type,
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,
    )
    _get_ml_client().compute.begin_create_or_update(compute).result()
    return compute_name


def _submit_feedback_retraining_job(data_asset_uri: str, dataset_hash: str) -> str:
    if _get_env("LOGMONITOR_RETRAIN_ENABLED", "1").lower() not in {"1", "true", "yes", "on"}:
        return ""
    code_dir = Path(__file__).resolve().parent
    if not (code_dir / "train.py").exists() or not (code_dir / "mlops_utils.py").exists():
        raise RuntimeError("The Function package is missing train.py or mlops_utils.py, so retraining cannot start.")

    compute_name = _ensure_retrain_compute()
    train_args = _get_env(
        "LOGMONITOR_RETRAIN_TRAIN_ARGS",
        "--train-mode default --epochs 3 --batch-size 8 --learning-rate 5e-5 --weight-decay 0.01 --max-length 128 --val-ratio 0.15 --test-ratio 0.15 --cv-folds 3 --max-trials 1",
    )
    mlflow_install_fragment = "mlflow==2.9.2 "
    job = command(
        inputs={"training_data": Input(type="uri_file", path=data_asset_uri, mode="download")},
        compute=compute_name,
        environment=_get_env("LOGMONITOR_RETRAIN_ENVIRONMENT", "AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest"),
        code=str(code_dir),
        command=(
            "pip install --upgrade numpy==1.23.5 pandas==1.5.3 transformers==4.24.0 "
            "sentencepiece==0.1.99 protobuf==3.20.3 scikit-learn==1.1.3 "
            f"{mlflow_install_fragment}"
            f"&& USE_TF=0 python train.py --data ${{inputs.training_data}} {train_args}"
        ),
        experiment_name=_get_env("LOGMONITOR_RETRAIN_EXPERIMENT_NAME", "log-monitor-feedback-retraining"),
        tags={
            "run_type": "feedback_retraining",
            "dataset_hash": dataset_hash,
            "hosting_service_kind": _get_env("LOGMONITOR_HOSTING_SERVICE_KIND", ""),
            "source_endpoint_name": _get_env("LOGMONITOR_SOURCE_ENDPOINT_NAME", ""),
        },
    )
    returned_job = _get_ml_client().jobs.create_or_update(job)
    return str(getattr(returned_job, "name", "") or "")


def _apply_feedback(payload: dict) -> dict:
    normalized = _normalize_payload(payload)
    message = _extract_message(normalized)
    correct_label = _extract_correct_label(normalized)
    if not message:
        raise ValueError("Feedback payload needs a log message in errorMessage, LogMessage, message, log, msg, or text.")
    if not correct_label:
        raise ValueError("Feedback payload needs correctLabel/class/label set to Error, CONFIGURATION, SYSTEM, or Noise.")

    state = _load_feedback_state()
    latest_dataset_blob = str(state.get("latest_dataset_blob") or "").strip()
    base_blob = latest_dataset_blob or _get_env("LOGMONITOR_BASE_DATASET_BLOB")
    base_rows = _read_labeled_csv_rows(_download_blob_text(base_blob)) if base_blob else []
    merged_rows, action = _merge_feedback_row(base_rows, message, correct_label)
    csv_body = _rows_to_labeled_csv(merged_rows)
    dataset_hash = hashlib.sha256(csv_body.encode("utf-8")).hexdigest()
    dataset_blob = _feedback_dataset_blob_name(dataset_hash)
    event_id = str(uuid.uuid4())
    event_blob = _feedback_event_blob_name(event_id)
    event_payload = {
        "event_id": event_id,
        "received_at": _now_utc_iso(),
        "message": message,
        "correct_label": correct_label,
        "predicted_label": str(normalized.get("predictedLabel") or normalized.get("predicted_label") or ""),
        "action": action,
        "source": str(normalized.get("source") or ""),
        "metadata": normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {},
    }
    _upload_blob_text(event_blob, json.dumps(event_payload, ensure_ascii=True, indent=2), content_type="application/json")
    _upload_blob_text(dataset_blob, csv_body, content_type="text/csv; charset=utf-8")
    data_asset = _register_feedback_data_asset(dataset_blob, dataset_hash, len(merged_rows), event_blob)
    job_name = _submit_feedback_retraining_job(data_asset["azure_data_asset_uri"], dataset_hash)

    state.update(
        {
            "latest_dataset_blob": dataset_blob,
            "latest_dataset_hash": dataset_hash,
            "latest_data_asset": data_asset,
            "latest_retraining_job_name": job_name,
            "latest_feedback_event_blob": event_blob,
            "latest_feedback_action": action,
            "latest_row_count": len(merged_rows),
            "updated_at": _now_utc_iso(),
        }
    )
    _write_feedback_state(state)
    return {
        "accepted": True,
        "action": action,
        "row_count": len(merged_rows),
        "dataset_hash": dataset_hash,
        "dataset_blob": dataset_blob,
        "feedback_event_blob": event_blob,
        "retraining_job_name": job_name,
        **data_asset,
    }


def _flush_queue_for_today(local_now: datetime) -> dict:
    queue_name = _require_env("LOGMONITOR_QUEUE_NAME")
    records = []
    messages = []

    service_bus_client = _get_service_bus_client()
    with service_bus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=5) as receiver:
        while True:
            batch = receiver.receive_messages(max_message_count=50, max_wait_time=2)
            if not batch:
                break
            messages.extend(batch)
            for message in batch:
                raw_text = _message_body_to_text(message)
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    payload = {"message": raw_text}
                payload = _normalize_payload(payload)
                payload.setdefault("errorMessage", _extract_message(payload))
                records.append(json.dumps(payload, ensure_ascii=True))

        if not messages:
            return {
                "status": "empty",
                "blob_name": "",
                "job_name": "",
                "message_count": 0,
            }

        blob_name = _build_blob_name(local_now)
        _upload_batch_blob(blob_name, records)
        job_name = _invoke_batch_endpoint(blob_name)

        for message in messages:
            receiver.complete_message(message)

    return {
        "status": "queued",
        "blob_name": blob_name,
        "job_name": job_name,
        "message_count": len(messages),
    }


@app.function_name(name="ingest_log")
@app.route(route="logs", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def ingest_log(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        raw_text = req.get_body().decode("utf-8", errors="ignore").strip()
        if not raw_text:
            return _to_json_response(
                {
                    "accepted": False,
                    "queued": False,
                    "error": "Send a JSON body or plain-text log message.",
                },
                400,
            )
        payload = raw_text

    normalized = _normalize_payload(payload)
    queue_name = _require_env("LOGMONITOR_QUEUE_NAME")
    message_body = json.dumps(normalized, ensure_ascii=True)

    try:
        service_bus_client = _get_service_bus_client()
        with service_bus_client.get_queue_sender(queue_name=queue_name) as sender:
            sender.send_messages(ServiceBusMessage(message_body, content_type="application/json"))
    except Exception as exc:
        LOGGER.exception("Failed to enqueue log record.")
        return _to_json_response(
            {
                "accepted": False,
                "queued": False,
                "error": str(exc),
            },
            500,
        )

    return _to_json_response(
        {
            "accepted": True,
            "queued": True,
            "received_at": normalized.get("received_at"),
        },
        202,
    )


@app.function_name(name="daily_batch_driver")
@app.schedule(schedule="0 * * * * *", arg_name="timer", run_on_startup=False, use_monitor=False)
def daily_batch_driver(timer: func.TimerRequest) -> None:
    try:
        del timer
        if _get_env("LOGMONITOR_BATCH_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
            return
        local_now = _get_local_now()
        target_hour, target_minute = _parse_daily_time(_require_env("LOGMONITOR_BATCH_TIME"))
        scheduled_time = local_now.replace(
            hour=target_hour,
            minute=target_minute,
            second=0,
            microsecond=0,
        )
        if local_now < scheduled_time:
            return

        current_local_date = local_now.date().isoformat()
        state = _load_scheduler_state()
        if state.get("last_processed_date") == current_local_date:
            return

        result = _flush_queue_for_today(local_now)
        state.update(
            {
                "last_processed_date": current_local_date,
                "last_status": result.get("status", ""),
                "last_job_name": result.get("job_name", ""),
                "last_blob_name": result.get("blob_name", ""),
                "last_message_count": int(result.get("message_count", 0)),
                "updated_at": _now_utc_iso(),
            }
        )
        _write_scheduler_state(state)
        LOGGER.info("Daily batch processing result: %s", state)
    except Exception:
        LOGGER.exception("Daily batch driver failed.")


@app.function_name(name="submit_feedback")
@app.route(route="feedback", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def submit_feedback(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return _to_json_response(
            {
                "accepted": False,
                "error": "Send a JSON body with errorMessage/message and correctLabel/class/label.",
            },
            400,
        )
    if not isinstance(payload, dict):
        return _to_json_response(
            {
                "accepted": False,
                "error": "Feedback payload must be a JSON object.",
            },
            400,
        )
    try:
        result = _apply_feedback(payload)
    except ValueError as exc:
        return _to_json_response({"accepted": False, "error": str(exc)}, 400)
    except Exception as exc:
        LOGGER.exception("Failed to process feedback.")
        return _to_json_response({"accepted": False, "error": str(exc)}, 500)
    return _to_json_response(result, 202)


@app.function_name(name="feedback_status")
@app.route(route="feedback/status", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def feedback_status(req: func.HttpRequest) -> func.HttpResponse:
    del req
    state = _load_feedback_state()
    return _to_json_response({"feedback_state": state}, 200)
