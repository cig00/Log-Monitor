import json
import os
import threading
from datetime import datetime, timezone
from urllib import request as url_request

from inference_utils import load_model_bundle, predict_error_message


MODEL_BUNDLE = None
TRIAGE_ACTION_TIMEOUT_SECONDS = 2.0


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _triage_action_enabled() -> bool:
    value = os.getenv("LOGMONITOR_TRIAGE_ACTION_ENABLED", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _triage_action_timeout() -> float:
    try:
        return max(float(os.getenv("LOGMONITOR_TRIAGE_ACTION_TIMEOUT", TRIAGE_ACTION_TIMEOUT_SECONDS)), 0.1)
    except Exception:
        return TRIAGE_ACTION_TIMEOUT_SECONDS


def _post_triage_action_async(payload: dict, error_message: str, prediction: str) -> None:
    action_url = (os.getenv("LOGMONITOR_TRIAGE_ACTION_URL") or "").strip()
    if not action_url or not _triage_action_enabled() or not prediction:
        return

    action_payload = {
        "errorMessage": error_message,
        "message": error_message,
        "prediction": prediction,
        "prediction_result": {
            "prediction": prediction,
            "raw_response": {"prediction": prediction},
            "endpoint_url": os.getenv("LOGMONITOR_SOURCE_ENDPOINT_URL", ""),
        },
        "received_at": _now_utc_iso(),
        "source": "azure_ml_online_endpoint",
        "source_endpoint_name": os.getenv("LOGMONITOR_SOURCE_ENDPOINT_NAME", ""),
        "source_endpoint_url": os.getenv("LOGMONITOR_SOURCE_ENDPOINT_URL", ""),
        "original_payload": payload if isinstance(payload, dict) else {},
    }
    body = json.dumps(action_payload, ensure_ascii=True).encode("utf-8")
    timeout_seconds = _triage_action_timeout()

    def send_action() -> None:
        try:
            req = url_request.Request(
                action_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with url_request.urlopen(req, timeout=timeout_seconds) as response:
                response.read(256)
            print(
                json.dumps(
                    {
                        "event": "log_monitor_triage_action_forwarded",
                        "prediction": prediction,
                        "status": "submitted",
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "log_monitor_triage_action_forward_failed",
                        "prediction": prediction,
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    threading.Thread(target=send_action, daemon=True).start()


def init():
    global MODEL_BUNDLE
    MODEL_BUNDLE = load_model_bundle("/var/azureml-app/azureml-models")


def run(raw_data):
    try:
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = raw_data.decode("utf-8")
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(payload, dict):
            return json.dumps({"prediction": ""})
        error_message = str(payload.get("errorMessage", ""))
        prediction = predict_error_message(MODEL_BUNDLE, error_message)
        _post_triage_action_async(payload, error_message, prediction)
        return json.dumps({"prediction": prediction})
    except Exception:
        return json.dumps({"prediction": ""})
