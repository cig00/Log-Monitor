import json
import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import azure.functions as func
from azure.ai.ml import Input, MLClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.storage.blob import BlobServiceClient


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
