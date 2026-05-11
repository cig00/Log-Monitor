import csv
import base64
import hashlib
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import azure.functions as func
import requests
try:
    from azure.ai.ml import Input, MLClient, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import AmlCompute, Data
except Exception as exc:
    Input = None
    MLClient = None
    command = None
    AssetTypes = None
    AmlCompute = None
    Data = None
    _AZURE_ML_IMPORT_ERROR = exc
else:
    _AZURE_ML_IMPORT_ERROR = None
try:
    from azure.core.exceptions import ResourceNotFoundError
except Exception:
    class ResourceNotFoundError(Exception):
        pass
try:
    from azure.identity import DefaultAzureCredential
except Exception as exc:
    DefaultAzureCredential = None
    _AZURE_IDENTITY_IMPORT_ERROR = exc
else:
    _AZURE_IDENTITY_IMPORT_ERROR = None
try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage
except Exception as exc:
    ServiceBusClient = None
    ServiceBusMessage = None
    _SERVICE_BUS_IMPORT_ERROR = exc
else:
    _SERVICE_BUS_IMPORT_ERROR = None
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception as exc:
    BlobServiceClient = None
    ContentSettings = None
    _BLOB_STORAGE_IMPORT_ERROR = exc
else:
    _BLOB_STORAGE_IMPORT_ERROR = None


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
PREDICTION_LABEL_KEYS = (
    "prediction",
    "predictions",
    "class",
    "classes",
    "label",
    "labels",
    "predictedLabel",
    "predicted_label",
    "predictedClass",
    "predicted_class",
    "category",
    "categories",
    "result",
    "results",
    "output",
    "outputs",
    "response",
    "answer",
    "generated_text",
)
PREDICTION_SKIP_KEYS = {
    "errorMessage",
    "LogMessage",
    "logMessage",
    "message",
    "log",
    "msg",
    "text",
    "input",
    "inputs",
    "payload",
    "request",
    "raw_payload",
}
DIAGNOSTIC_SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "authorization",
    "connection_string",
    "connectionstring",
    "access_key",
    "account_key",
    "api_key",
    "endpoint_key",
    "function_key",
    "prediction_key",
)
DIAGNOSTIC_SENSITIVE_KEYS = {"code", "pat", "connection"}

_BLOB_SERVICE_CLIENT = None
_SERVICE_BUS_CLIENT = None
_ML_CLIENT = None

SOURCE_PATH_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?(?:[\w ._-]+[\\/])*[\w.-]+\.(?:py|js|jsx|ts|tsx|java|cs|go|rb|php|c|cc|cpp|h|hpp|rs|kt|swift))"
)
COMMIT_SHA_PATTERN = re.compile(r"\b[0-9a-fA-F]{7,40}\b")
COMMIT_SHA_KEYS = (
    "commitSha",
    "commit_sha",
    "commitId",
    "commit_id",
    "gitSha",
    "git_sha",
    "githubSha",
    "github_sha",
    "headSha",
    "head_sha",
    "sourceVersion",
    "source_version",
    "buildSourceVersion",
    "build_source_version",
    "revision",
    "commit",
    "GITHUB_SHA",
    "sha",
)
PREVIOUS_COMMIT_SHA_KEYS = (
    "previousSha",
    "previous_sha",
    "baseSha",
    "base_sha",
    "beforeSha",
    "before_sha",
    "before",
    "baseCommit",
    "base_commit",
    "GITHUB_BASE_SHA",
)
BRANCH_KEYS = (
    "branch",
    "gitBranch",
    "git_branch",
    "githubBranch",
    "github_branch",
    "sourceBranch",
    "source_branch",
    "GITHUB_REF",
    "github_ref",
    "ref",
)
GITHUB_SEARCH_STOP_WORDS = {
    "after",
    "azure",
    "because",
    "before",
    "called",
    "class",
    "code",
    "configuration",
    "could",
    "error",
    "failed",
    "failure",
    "from",
    "function",
    "github",
    "http",
    "jira",
    "log",
    "message",
    "monitor",
    "noise",
    "none",
    "null",
    "prediction",
    "request",
    "response",
    "runtime",
    "server",
    "service",
    "system",
    "trace",
    "true",
    "with",
}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _require_env(name: str) -> str:
    value = _get_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _raise_import_error(package_name: str, exc: Exception | None) -> None:
    if exc is not None:
        raise RuntimeError(f"Azure Function dependency `{package_name}` is not installed or failed to import: {exc}") from exc


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


def _get_monitoring_state_blob_name() -> str:
    return _get_env("LOGMONITOR_MONITORING_STATE_BLOB", "monitoring/prediction-summary-state.json")


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


def _redact_for_diagnostics(value: object, max_depth: int = 4) -> object:
    if max_depth <= 0:
        return _truncate(value, 300)
    if isinstance(value, dict):
        redacted = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 50:
                redacted["..."] = f"{len(value) - index} more keys truncated"
                break
            clean_key = str(key)
            clean_key_lower = clean_key.casefold()
            if clean_key_lower in DIAGNOSTIC_SENSITIVE_KEYS or any(
                part in clean_key_lower for part in DIAGNOSTIC_SENSITIVE_KEY_PARTS
            ):
                redacted[clean_key] = "<redacted>"
            else:
                redacted[clean_key] = _redact_for_diagnostics(item, max_depth - 1)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_for_diagnostics(item, max_depth - 1) for item in list(value)[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate(value, 1000) if isinstance(value, str) else value
    return _truncate(str(value), 1000)


def _diagnostic_shape(value: object) -> dict:
    shape = {"type": type(value).__name__}
    if isinstance(value, dict):
        shape["keys"] = [str(key) for key in list(value.keys())[:20]]
    elif isinstance(value, list):
        shape["length"] = len(value)
        if value:
            shape["first_item_type"] = type(value[0]).__name__
    elif isinstance(value, str):
        shape["length"] = len(value)
    return shape


def _add_diagnostic(diagnostics: list[dict], stage: str, status: str = "ok", **details) -> None:
    diagnostics.append(
        {
            "at": _now_utc_iso(),
            "stage": stage,
            "status": status,
            "details": _redact_for_diagnostics(details),
        }
    )


def _truncate(value: object, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30] + "\n...[truncated by Log Monitor]"


def _jira_summary_text(value: object, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        text = "Log Monitor event"
    if len(text) <= max_chars:
        return text
    suffix = "...[truncated]"
    return text[: max(1, max_chars - len(suffix))].rstrip() + suffix


def _dedupe_match_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _jira_jql_string(value: object) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _get_payload_value(payload: dict, keys: tuple[str, ...]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        clean_value = _payload_scalar(value)
        if clean_value:
            return clean_value
    for container_key in ("metadata", "context", "deployment", "git", "github"):
        nested = payload.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                clean_value = _payload_scalar(value)
                if clean_value:
                    return clean_value
    return ""


def _payload_scalar(value: object) -> str:
    if value in (None, "") or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _normalize_payload_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key or "").casefold())


def _get_nested_payload_value(payload: dict, keys: tuple[str, ...], max_depth: int = 5) -> str:
    direct = _get_payload_value(payload, keys)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return ""

    target_keys = {_normalize_payload_key(key) for key in keys if str(key or "").strip()}
    seen: set[int] = set()
    stack: list[tuple[object, int]] = [(payload, 0)]
    while stack:
        current, depth = stack.pop(0)
        if depth > max_depth or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, dict):
            for key, value in current.items():
                clean_value = _payload_scalar(value)
                if _normalize_payload_key(key) in target_keys and clean_value:
                    return clean_value
                if isinstance(value, (dict, list, tuple)):
                    stack.append((value, depth + 1))
        elif isinstance(current, (list, tuple)):
            for value in current:
                if isinstance(value, (dict, list, tuple)):
                    stack.append((value, depth + 1))
    return ""


def _clean_github_branch(raw_branch: object) -> str:
    branch = str(raw_branch or "").strip()
    if not branch:
        return ""
    for prefix in ("refs/heads/", "origin/"):
        if branch.startswith(prefix):
            return branch[len(prefix) :].strip()
    return branch


def _extract_commit_sha(raw_value: object) -> str:
    text = _payload_scalar(raw_value)
    if not text:
        return ""
    matches = COMMIT_SHA_PATTERN.findall(text)
    if not matches:
        return ""
    return max(matches, key=len)[:40]


def _parse_utc_datetime(raw_value: object) -> datetime | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _github_lookback_since(payload: dict) -> str:
    try:
        lookback_days = max(int(_get_env("LOGMONITOR_GITHUB_LOOKBACK_DAYS", "14") or "14"), 1)
    except Exception:
        lookback_days = 14
    received_at = _parse_utc_datetime(
        _get_nested_payload_value(payload, ("received_at", "timestamp", "time", "eventTime", "created_at", "createdAt"))
    ) or datetime.now(timezone.utc)
    return (received_at - timedelta(days=lookback_days)).date().isoformat()


def _extract_github_search_terms(payload: dict, message: str, source_paths: list[str]) -> list[str]:
    text_parts = [message]
    if isinstance(payload, dict):
        text_parts.append(json.dumps(payload, ensure_ascii=True)[:8000])
    for path in source_paths:
        text_parts.extend(Path(path).parts)
    text = "\n".join(part for part in text_parts if part)

    terms: list[str] = []

    def add_term(raw_term: object) -> None:
        term = re.sub(r"[^A-Za-z0-9_.-]+", "", str(raw_term or "").strip()).strip("._-")
        if len(term) < 4 or len(term) > 80:
            return
        if term.casefold() in GITHUB_SEARCH_STOP_WORDS:
            return
        if term.casefold() in {label.casefold() for label in LABEL_NAMES}:
            return
        if term not in terms:
            terms.append(term)

    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Exception|Error|Failure|Fault|Timeout|Denied|NotFound|Conflict|BadRequest|Unauthorized|Forbidden|Crash))\b",
        text,
    ):
        add_term(match.group(1))
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_.-]{3,})\b", text):
        add_term(match.group(1))
        if len(terms) >= 12:
            break
    return terms[:12]


def _label_from_prediction_text(raw_prediction: object) -> str:
    text = str(raw_prediction or "").strip()
    if not text:
        return ""
    cleaned = re.sub(r"^[\s\"'`*_]+|[\s\"'`*_.:;-]+$", "", text).strip()
    for candidate in (text, cleaned):
        label = LABEL_ALIASES.get(candidate.casefold())
        if label:
            return label

    label_pattern = "|".join(re.escape(label) for label in LABEL_NAMES)
    context_patterns = (
        rf"(?:prediction|predicted\s+class|predicted\s+label|class|label|category|result|classification|answer)"
        rf"\s*(?:is|:|=|-)?\s*[\"'`*]*({label_pattern})\b",
        rf"(?:classified\s+as|predicted\s+as)\s*[\"'`*]*({label_pattern})\b",
        rf"\b({label_pattern})\b\s*(?:prediction|class|label|category|classification)\b",
    )
    for pattern in context_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return LABEL_ALIASES.get(match.group(1).casefold(), "")

    standalone_matches = {
        LABEL_ALIASES[match.group(1).casefold()]
        for match in re.finditer(rf"(?<![A-Za-z0-9_])({label_pattern})(?![A-Za-z0-9_])", text, flags=re.IGNORECASE)
        if match.group(1).casefold() in LABEL_ALIASES
    }
    if len(text) <= 80 and len(standalone_matches) == 1:
        return next(iter(standalone_matches))
    return ""


def _normalize_prediction_label(raw_prediction: object, allow_raw_text: bool = True) -> str:
    if isinstance(raw_prediction, dict):
        case_key_lookup = {str(key).casefold(): key for key in raw_prediction.keys()}
        for key in PREDICTION_LABEL_KEYS:
            actual_key = case_key_lookup.get(key.casefold())
            if actual_key is None:
                continue
            value = raw_prediction.get(actual_key)
            if value not in (None, ""):
                nested = _normalize_prediction_label(value, allow_raw_text=True)
                if nested:
                    return nested
        skip_keys = {key.casefold() for key in PREDICTION_SKIP_KEYS}
        for key, value in raw_prediction.items():
            if str(key).casefold() in skip_keys or value in (None, ""):
                continue
            nested = _normalize_prediction_label(value, allow_raw_text=False)
            if nested:
                return nested
        return ""
    if isinstance(raw_prediction, list):
        for item in raw_prediction:
            nested = _normalize_prediction_label(item, allow_raw_text=allow_raw_text)
            if nested:
                return nested
        return ""
    text = str(raw_prediction or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if parsed is not text:
            nested = _normalize_prediction_label(parsed, allow_raw_text=allow_raw_text)
            if nested:
                return nested
    except Exception:
        pass
    label = _label_from_prediction_text(text)
    if label:
        return label
    return text if allow_raw_text else ""


def _call_prediction_endpoint(payload: dict) -> dict:
    endpoint_url = _get_env("LOGMONITOR_PREDICTION_ENDPOINT_URL") or _get_env("LOGMONITOR_SOURCE_ENDPOINT_URL")
    if not endpoint_url:
        raise RuntimeError("Prediction endpoint URL is not configured.")
    message = _extract_message(payload)
    if not message:
        raise ValueError("Triage payload needs a log message in errorMessage, LogMessage, message, log, msg, or text.")

    headers = {"Content-Type": "application/json"}
    auth_mode = _get_env("LOGMONITOR_PREDICTION_AUTH_MODE", "key").lower()
    if auth_mode == "key":
        prediction_key = _require_env("LOGMONITOR_PREDICTION_KEY")
        headers["Authorization"] = f"Bearer {prediction_key}"
    elif auth_mode in {"aad", "aad_token", "entra", "managed_identity"}:
        _raise_import_error("azure-identity", _AZURE_IDENTITY_IMPORT_ERROR)
        credential = DefaultAzureCredential()
        token = credential.get_token("https://ml.azure.com/.default").token
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(endpoint_url, headers=headers, json={"errorMessage": message}, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _truncate(response.text, 2000)
        raise RuntimeError(f"Prediction endpoint failed ({response.status_code}). {detail}") from exc

    try:
        response_payload = response.json()
    except Exception:
        response_payload = response.text
    prediction = _normalize_prediction_label(response_payload)
    if not prediction:
        raise RuntimeError("Prediction endpoint returned no prediction label.")
    return {
        "prediction": prediction,
        "raw_response": response_payload,
        "endpoint_url": endpoint_url,
    }


def _send_email(recipient: str, subject: str, plain_text: str) -> dict:
    clean_recipient = str(recipient or "").strip()
    if not clean_recipient:
        raise RuntimeError("Email recipient is empty.")
    connection_string = _require_env("LOGMONITOR_ACS_CONNECTION_STRING")
    sender_address = _require_env("LOGMONITOR_ACS_SENDER_ADDRESS")
    try:
        from azure.communication.email import EmailClient
    except Exception as exc:
        raise RuntimeError("The Function package is missing azure-communication-email.") from exc

    client = EmailClient.from_connection_string(connection_string)
    message = {
        "senderAddress": sender_address,
        "recipients": {"to": [{"address": clean_recipient}]},
        "content": {
            "subject": _truncate(subject, 250),
            "plainText": _truncate(plain_text, 12000),
        },
    }
    poller = client.begin_send(message)
    result = poller.result()
    message_id = ""
    if isinstance(result, dict):
        message_id = str(result.get("id") or result.get("messageId") or "")
    else:
        message_id = str(getattr(result, "id", "") or getattr(result, "message_id", "") or "")
    return {"recipient": clean_recipient, "message_id": message_id}


def _github_headers() -> dict:
    token = _get_env("LOGMONITOR_GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GitHub token is not configured.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _github_request(path: str, params: dict | None = None) -> object:
    repo = _require_env("LOGMONITOR_GITHUB_REPO")
    url = f"https://api.github.com/repos/{repo}/{path.lstrip('/')}"
    response = requests.get(url, headers=_github_headers(), params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def _github_post(path: str, payload: dict, timeout: int = 60) -> dict:
    repo = _require_env("LOGMONITOR_GITHUB_REPO")
    url = f"https://api.github.com/repos/{repo}/{path.lstrip('/')}"
    response = requests.post(url, headers=_github_headers(), json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _truncate(response.text, 2000)
        raise RuntimeError(f"GitHub request failed ({response.status_code}). {detail}") from exc
    parsed = response.json()
    return parsed if isinstance(parsed, dict) else {}


def _summarize_github_issue_reference(item: dict) -> dict:
    number = item.get("number")
    reference = {
        "number": number,
        "title": str(item.get("title") or ""),
        "state": str(item.get("state") or ""),
        "html_url": str(item.get("html_url") or ""),
        "api_url": str(item.get("url") or ""),
    }
    if isinstance(item.get("pull_request"), dict):
        reference["pull_request"] = True
        reference["pull_request_url"] = str(item.get("pull_request", {}).get("url") or "")
    return reference


def _github_search_issue_references(issue_key: str, reference_type: str) -> dict:
    clean_issue_key = str(issue_key or "").strip()
    repo = _get_env("LOGMONITOR_GITHUB_REPO")
    if not clean_issue_key or not repo or not _get_env("LOGMONITOR_GITHUB_TOKEN"):
        return {
            "status": "skipped",
            "reason": "GitHub token, repository, or Jira issue key is not configured.",
            "items": [],
            "total_count": 0,
        }
    clean_reference_type = "pr" if str(reference_type or "").strip().casefold() in {"pr", "pull_request"} else "issue"
    query = f'repo:{repo} "{clean_issue_key}" type:{clean_reference_type}'
    try:
        response = requests.get(
            "https://api.github.com/search/issues",
            headers=_github_headers(),
            params={"q": query, "per_page": 5},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        LOGGER.warning("GitHub remediation dedupe lookup failed for %s: %s", clean_issue_key, exc)
        return {
            "status": "failed",
            "reason": str(exc),
            "query": query,
            "items": [],
            "total_count": 0,
        }
    items = payload.get("items", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []
    references = [_summarize_github_issue_reference(item) for item in items if isinstance(item, dict)]
    return {
        "status": "found" if references else "not_found",
        "query": query,
        "items": references,
        "total_count": int(payload.get("total_count", len(references)) or 0) if isinstance(payload, dict) else len(references),
    }


def _find_existing_github_remediation(issue_key: str) -> dict:
    pull_requests = _github_search_issue_references(issue_key, "pr")
    if pull_requests.get("items"):
        return {
            "status": "found",
            "operation": "reused",
            "kind": "pull_request",
            "has_pr": True,
            "jira_issue_key": str(issue_key or "").strip(),
            "pull_request": pull_requests["items"][0],
            "pull_request_search": pull_requests,
        }
    if pull_requests.get("status") == "failed":
        return {
            "status": "failed",
            "kind": "pull_request",
            "has_pr": False,
            "jira_issue_key": str(issue_key or "").strip(),
            "error": pull_requests.get("reason", ""),
            "pull_request_search": pull_requests,
        }

    issues = _github_search_issue_references(issue_key, "issue")
    if issues.get("items"):
        return {
            "status": "found",
            "operation": "reused",
            "kind": "issue",
            "has_pr": False,
            "jira_issue_key": str(issue_key or "").strip(),
            "issue": issues["items"][0],
            "issue_search": issues,
            "pull_request_search": pull_requests,
        }
    if issues.get("status") == "failed":
        return {
            "status": "failed",
            "kind": "issue",
            "has_pr": False,
            "jira_issue_key": str(issue_key or "").strip(),
            "error": issues.get("reason", ""),
            "issue_search": issues,
            "pull_request_search": pull_requests,
        }
    if pull_requests.get("status") == "skipped" or issues.get("status") == "skipped":
        return {
            "status": "skipped",
            "kind": "",
            "has_pr": False,
            "jira_issue_key": str(issue_key or "").strip(),
            "reason": pull_requests.get("reason") or issues.get("reason", ""),
            "pull_request_search": pull_requests,
            "issue_search": issues,
        }
    return {
        "status": "not_found",
        "kind": "",
        "has_pr": False,
        "jira_issue_key": str(issue_key or "").strip(),
        "pull_request_search": pull_requests,
        "issue_search": issues,
    }


def _github_search_commits(repo: str, terms: list[str], since_date: str) -> list[dict]:
    search_terms = [term for term in terms if term][:6]
    if not repo or not search_terms:
        return []
    term_groups = [search_terms[:4], search_terms[:2], search_terms[:1]]
    for term_group in term_groups:
        if not term_group:
            continue
        query_parts = [f"repo:{repo}", f"committer-date:>={since_date}", *term_group]
        response = requests.get(
            "https://api.github.com/search/commits",
            headers=_github_headers(),
            params={"q": " ".join(query_parts), "per_page": 5},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("items"), list) and payload["items"]:
            return [item for item in payload["items"] if isinstance(item, dict)]
    return []


def _github_history_diff_limit() -> int:
    try:
        limit = int(_get_env("LOGMONITOR_GITHUB_HISTORY_DIFF_LIMIT", "10") or "10")
    except Exception:
        limit = 10
    return min(max(limit, 1), 25)


def _extract_source_paths(payload: dict, message: str) -> list[str]:
    candidates: list[str] = []
    for value in (
        _get_nested_payload_value(payload, ("sourcePath", "source_path", "file", "filename", "path")),
        message,
        json.dumps(payload, ensure_ascii=True) if isinstance(payload, dict) else "",
    ):
        if not value:
            continue
        for match in SOURCE_PATH_PATTERN.finditer(value):
            path = match.group("path").replace("\\", "/").strip("/")
            path = re.sub(r"^.*\b(?:in|at|from|file)\s+(?=[^\s]+/)", "", path, flags=re.IGNORECASE).strip("/")
            if path and path not in candidates:
                candidates.append(path)
    return candidates[:8]


def _summarize_github_commit(commit: dict, source: str, confidence: str) -> dict:
    payload = commit.get("commit") if isinstance(commit.get("commit"), dict) else {}
    author = payload.get("author") if isinstance(payload.get("author"), dict) else {}
    github_author = commit.get("author") if isinstance(commit.get("author"), dict) else {}
    committer = payload.get("committer") if isinstance(payload.get("committer"), dict) else {}
    files = commit.get("files") if isinstance(commit.get("files"), list) else []
    return {
        "sha": str(commit.get("sha") or "")[:40],
        "html_url": str(commit.get("html_url") or ""),
        "message": str(payload.get("message") or "").splitlines()[0][:240],
        "author_name": str(author.get("name") or github_author.get("login") or ""),
        "author_email": str(author.get("email") or ""),
        "author_login": str(github_author.get("login") or ""),
        "authored_at": str(author.get("date") or ""),
        "committed_at": str(committer.get("date") or author.get("date") or ""),
        "source": source,
        "confidence": confidence,
        "files": [str(file_info.get("filename") or "") for file_info in files[:20] if isinstance(file_info, dict)],
    }


def _source_path_match(filename: str, source_path: str) -> str:
    clean_filename = filename.replace("\\", "/").strip("/")
    clean_source = source_path.replace("\\", "/").strip("/")
    if not clean_filename or not clean_source:
        return ""
    if clean_filename == clean_source:
        return "exact"
    if clean_filename.endswith("/" + clean_source) or clean_source.endswith("/" + clean_filename):
        return "suffix"
    if Path(clean_filename).name == Path(clean_source).name:
        return "basename"
    return ""


def _diff_snippet_for_term(filename: str, patch_text: str, term: str) -> str:
    term_folded = term.casefold()
    for line in patch_text.splitlines():
        if term_folded in line.casefold():
            return _truncate(f"{filename}: {line.strip()}", 240)
    return ""


def _diff_snippet_for_source_path(filename: str, patch_text: str, source_path: str) -> str:
    clean_filename = filename.replace("\\", "/").strip("/")
    if _source_path_match(clean_filename, source_path):
        for line in patch_text.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                return _truncate(f"{filename}: {line.strip()}", 240)
        return _truncate(f"{filename}: changed file matched {source_path}", 240)
    return ""


def _github_diff_evidence(commit: dict, source_paths: list[str], evidence_terms: list[str]) -> dict:
    files = commit.get("files") if isinstance(commit.get("files"), list) else []
    changed_files = [str(file_info.get("filename") or "") for file_info in files if isinstance(file_info, dict)]
    matched_source_paths: list[str] = []
    matched_files: list[str] = []
    matched_terms_in_diff: list[str] = []
    matched_terms_in_changed_files: list[str] = []
    diff_snippets: list[str] = []
    score = 0

    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        filename = str(file_info.get("filename") or "")
        patch_text = str(file_info.get("patch") or "")
        filename_folded = filename.casefold()
        patch_folded = patch_text.casefold()

        for source_path in source_paths:
            match_kind = _source_path_match(filename, source_path)
            if not match_kind:
                continue
            if source_path not in matched_source_paths:
                matched_source_paths.append(source_path)
            if filename not in matched_files:
                matched_files.append(filename)
            snippet = _diff_snippet_for_source_path(filename, patch_text, source_path)
            if snippet and snippet not in diff_snippets:
                diff_snippets.append(snippet)
            score += 70 if match_kind in {"exact", "suffix"} else 25

        for term in evidence_terms:
            term_folded = term.casefold()
            if term_folded in patch_folded:
                if term not in matched_terms_in_diff:
                    matched_terms_in_diff.append(term)
                    snippet = _diff_snippet_for_term(filename, patch_text, term)
                    if snippet and snippet not in diff_snippets:
                        diff_snippets.append(snippet)
                score += 15
            elif term_folded in filename_folded:
                if term not in matched_terms_in_changed_files:
                    matched_terms_in_changed_files.append(term)
                score += 8

    return {
        "changed_files": changed_files[:20],
        "matched_source_paths": matched_source_paths[:8],
        "matched_files": matched_files[:8],
        "matched_terms_in_diff": matched_terms_in_diff[:12],
        "matched_terms_in_changed_files": matched_terms_in_changed_files[:12],
        "diff_snippets": diff_snippets[:5],
        "score": min(score, 160),
    }


def _commit_sha(commit: dict) -> str:
    return str(commit.get("sha") or "").strip()[:40] if isinstance(commit, dict) else ""


def _commit_parent_sha(commit: dict) -> str:
    if not isinstance(commit, dict):
        return ""
    parents = commit.get("parents") if isinstance(commit.get("parents"), list) else []
    for parent in parents:
        if isinstance(parent, dict):
            sha = _commit_sha(parent)
            if sha:
                return sha
    return ""


def _commit_from_compare_payload(compare_payload: dict, head_commit: dict, base_sha: str, head_sha: str) -> dict:
    commit = dict(head_commit) if isinstance(head_commit, dict) else {}
    commit["sha"] = head_sha or _commit_sha(commit)
    commit.setdefault("html_url", str(compare_payload.get("html_url") or ""))
    if isinstance(compare_payload.get("files"), list):
        commit["files"] = compare_payload["files"]
    if not isinstance(commit.get("commit"), dict) and isinstance(compare_payload.get("commits"), list):
        for item in compare_payload["commits"]:
            if isinstance(item, dict) and _commit_sha(item) == commit["sha"]:
                commit["commit"] = item.get("commit") if isinstance(item.get("commit"), dict) else {}
                commit.setdefault("author", item.get("author") if isinstance(item.get("author"), dict) else {})
                commit.setdefault("html_url", item.get("html_url") or commit.get("html_url") or "")
                break
    commit["base_sha"] = base_sha
    commit["head_sha"] = commit["sha"]
    commit["compare_url"] = str(compare_payload.get("html_url") or "")
    return commit


def _build_copilot_diff_review_prompt(message: str, source_paths: list[str], evidence_terms: list[str]) -> str:
    return (
        "You are GitHub Copilot acting as a senior software engineer. Review this exact git diff against the "
        "production log evidence. Decide whether the changed files or patch text are related to the error. "
        "Do not rely on the commit message alone. Return related, confidence, and a short reason.\n\n"
        f"Error log:\n{_truncate(message, 2000)}\n\n"
        f"Source paths: {', '.join(source_paths[:8]) or 'none extracted'}\n"
        f"Evidence terms: {', '.join(evidence_terms[:12]) or 'none extracted'}"
    )


def _copilot_diff_relevance_review(
    commit: dict,
    message: str,
    source_paths: list[str],
    evidence_terms: list[str],
    diff_evidence: dict | None = None,
) -> dict:
    evidence = diff_evidence if isinstance(diff_evidence, dict) else _github_diff_evidence(commit, source_paths, evidence_terms)
    score = int(evidence.get("score") or 0)
    matched_paths = evidence.get("matched_source_paths") if isinstance(evidence.get("matched_source_paths"), list) else []
    matched_terms = evidence.get("matched_terms_in_diff") if isinstance(evidence.get("matched_terms_in_diff"), list) else []
    matched_file_terms = (
        evidence.get("matched_terms_in_changed_files")
        if isinstance(evidence.get("matched_terms_in_changed_files"), list)
        else []
    )
    related = bool(matched_paths or matched_terms or score >= 35)
    confidence = _github_confidence_from_score(score)
    if matched_paths:
        reason = "The diff changes file paths extracted from the log, so this commit is a strong candidate for review."
    elif matched_terms:
        reason = "The patch text contains error terms extracted from the log, so this commit may be related."
    elif matched_file_terms:
        reason = "Changed filenames contain log evidence terms, but the patch text did not directly match."
    else:
        reason = "No changed file or patch-text evidence matched this error; this diff is likely unrelated."
    return {
        "reviewer": "copilot_diff_rubric",
        "related": related,
        "confidence": confidence,
        "score": score,
        "reason": reason,
        "matched_source_paths": matched_paths[:8],
        "matched_terms_in_diff": matched_terms[:12],
        "matched_terms_in_changed_files": matched_file_terms[:12],
        "diff_snippets": evidence.get("diff_snippets", [])[:5] if isinstance(evidence.get("diff_snippets"), list) else [],
        "prompt": _build_copilot_diff_review_prompt(message, source_paths, evidence_terms),
    }


def _merge_unique(left: object, right: object, limit: int) -> list[str]:
    merged: list[str] = []
    for values in (left, right):
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value or "")
            if text and text not in merged:
                merged.append(text)
    return merged[:limit]


def _merge_github_diff_evidence(existing: object, new: dict) -> dict:
    old = existing if isinstance(existing, dict) else {}
    return {
        "changed_files": _merge_unique(old.get("changed_files"), new.get("changed_files"), 20),
        "matched_source_paths": _merge_unique(old.get("matched_source_paths"), new.get("matched_source_paths"), 8),
        "matched_files": _merge_unique(old.get("matched_files"), new.get("matched_files"), 8),
        "matched_terms_in_diff": _merge_unique(old.get("matched_terms_in_diff"), new.get("matched_terms_in_diff"), 12),
        "matched_terms_in_changed_files": _merge_unique(
            old.get("matched_terms_in_changed_files"), new.get("matched_terms_in_changed_files"), 12
        ),
        "diff_snippets": _merge_unique(old.get("diff_snippets"), new.get("diff_snippets"), 5),
        "score": min(int(old.get("score") or 0) + int(new.get("score") or 0), 160),
    }


def _github_confidence_from_score(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _github_impact_assessment(candidates: list[dict], lookup_warnings: list[dict]) -> tuple[str, str]:
    if any(candidate.get("diff_match") or candidate.get("deployment_sha_match") for candidate in candidates):
        return (
            "possible_developer_impact",
            (
                "Candidate commits have deployment SHA evidence or their actual changed files/patch text match the log. "
                "A developer change may be related, but this is not proof."
            ),
        )
    if any(candidate.get("source") == "commit_message_search" for candidate in candidates):
        return (
            "message_only_developer_candidate",
            (
                "Commit-message search found candidates, but their changed files/patch text did not match the extracted log evidence. "
                "No developer may be responsible; review the diff evidence before assigning ownership."
            ),
        )
    if candidates:
        return (
            "no_direct_developer_evidence",
            (
                "No deployment SHA, changed-file, or patch-text match was found. Recent commits are included only for context; "
                "no developer may be responsible."
            ),
        )
    if lookup_warnings:
        return (
            "lookup_incomplete",
            (
                "GitHub lookup did not return candidate commits because one or more lookup steps failed. "
                "No developer may be responsible, but the lookup was incomplete."
            ),
        )
    return (
        "no_developer_candidate_found",
        (
            "No developer commit candidate was found by the automatic search. "
            "This can mean no developer is responsible, or that the relevant code change was not visible in the provided GitHub history."
        ),
    )


def _find_github_impact_context(payload: dict, message: str) -> dict:
    caveat = (
        "GitHub commit data is a non-conclusive signal only. "
        "Log Monitor never decides that a developer did or did not cause the error automatically. "
        "The error may be caused by code, configuration, infrastructure, external services, data, or runtime state."
    )
    repo = _get_env("LOGMONITOR_GITHUB_REPO")
    branch = _clean_github_branch(
        _get_nested_payload_value(payload, BRANCH_KEYS) or _get_env("LOGMONITOR_GITHUB_BRANCH")
    )
    if not _get_env("LOGMONITOR_GITHUB_TOKEN") or not repo:
        return {
            "status": "skipped",
            "developer_impact_verdict": "lookup_skipped",
            "conclusion": (
                "GitHub lookup was skipped because the token or repository was not configured. "
                "No developer may be responsible, but Log Monitor could not inspect commit history."
            ),
            "caveat": caveat,
            "reason": "GitHub token or repository is not configured.",
            "candidates": [],
        }

    candidate_map: dict[str, dict] = {}
    lookup_warnings: list[dict] = []
    source_paths = _extract_source_paths(payload, message)
    evidence_terms = _extract_github_search_terms(payload, message, source_paths)

    def commit_with_diff(commit: dict, source: str) -> dict:
        files = commit.get("files") if isinstance(commit.get("files"), list) else []
        if files:
            return commit
        sha = str(commit.get("sha") or "")
        if not sha:
            return commit
        detail = request_or_warn(f"commits/{sha}", lookup=f"{source}_diff")
        return detail if isinstance(detail, dict) else commit

    def add_candidate(
        commit: dict,
        source: str,
        score: int,
        matched_terms: list[str] | None = None,
        extra: dict | None = None,
    ) -> None:
        sha = str(commit.get("sha") or "")
        if not sha:
            return
        detailed_commit = commit_with_diff(commit, source)
        diff_evidence = _github_diff_evidence(detailed_commit, source_paths, evidence_terms)
        diff_score = int(diff_evidence.get("score") or 0)
        adjusted_score = score + diff_score
        summary = _summarize_github_commit(detailed_commit, source, _github_confidence_from_score(adjusted_score))
        diff_match = bool(diff_evidence.get("matched_source_paths") or diff_evidence.get("matched_terms_in_diff"))
        deployment_sha_match = source in {"payload_commit_sha", "payload_sha_range"}
        signal = {"source": source, "score": score, "diff_score": diff_score}
        if matched_terms:
            signal["matched_terms"] = matched_terms[:6]
        if isinstance(extra, dict):
            signal.update({key: value for key, value in extra.items() if key in {"base_sha", "head_sha", "compare_url"}})
        if sha in candidate_map:
            existing = candidate_map[sha]
            existing["score"] = int(existing.get("score") or 0) + adjusted_score
            existing["confidence"] = _github_confidence_from_score(int(existing["score"]))
            existing["diff_score"] = int(existing.get("diff_score") or 0) + diff_score
            existing["diff_match"] = bool(existing.get("diff_match") or diff_match)
            existing["deployment_sha_match"] = bool(existing.get("deployment_sha_match") or deployment_sha_match)
            if source not in existing.get("sources", []):
                existing.setdefault("sources", []).append(source)
            existing.setdefault("signals", []).append(signal)
            existing["diff_evidence"] = _merge_github_diff_evidence(existing.get("diff_evidence", {}), diff_evidence)
            if matched_terms:
                existing_terms = existing.setdefault("matched_terms", [])
                for term in matched_terms:
                    if term not in existing_terms:
                        existing_terms.append(term)
            if isinstance(extra, dict):
                if extra.get("copilot_diff_review"):
                    existing.setdefault("copilot_diff_reviews", []).append(extra["copilot_diff_review"])
                for key in ("base_sha", "head_sha", "compare_url"):
                    if extra.get(key):
                        existing[key] = extra[key]
            return
        summary["score"] = adjusted_score
        summary["diff_score"] = diff_score
        summary["diff_match"] = diff_match
        summary["deployment_sha_match"] = deployment_sha_match
        summary["diff_evidence"] = diff_evidence
        summary["sources"] = [source]
        summary["signals"] = [signal]
        summary["matched_terms"] = matched_terms[:] if matched_terms else []
        if isinstance(extra, dict):
            if extra.get("copilot_diff_review"):
                summary["copilot_diff_review"] = extra["copilot_diff_review"]
                summary["copilot_diff_reviews"] = [extra["copilot_diff_review"]]
            for key in ("base_sha", "head_sha", "compare_url"):
                if extra.get(key):
                    summary[key] = extra[key]
        candidate_map[sha] = summary

    def request_or_warn(path: str, params: dict | None = None, lookup: str = "") -> object:
        try:
            return _github_request(path, params=params)
        except Exception as exc:
            LOGGER.warning("GitHub impact lookup step failed for %s: %s", lookup or path, exc)
            lookup_warnings.append({"lookup": lookup or path, "error": str(exc)})
            return None

    def review_recent_consecutive_diffs() -> list[dict]:
        limit = _github_history_diff_limit()
        recent_params = {"sha": branch, "per_page": limit + 1} if branch else {"per_page": limit + 1}
        recent = request_or_warn("commits", params=recent_params, lookup="recent_consecutive_history")
        if not isinstance(recent, list):
            return []
        reviews: list[dict] = []
        for index, head_commit in enumerate(recent[:limit]):
            if not isinstance(head_commit, dict):
                continue
            head_sha = _commit_sha(head_commit)
            base_sha = _commit_parent_sha(head_commit)
            if not base_sha and index + 1 < len(recent) and isinstance(recent[index + 1], dict):
                base_sha = _commit_sha(recent[index + 1])
            if not head_sha or not base_sha:
                reviews.append(
                    {
                        "index": index,
                        "head_sha": head_sha,
                        "base_sha": base_sha,
                        "status": "skipped",
                        "reason": "Could not determine consecutive base/head SHA.",
                    }
                )
                continue
            compare_path = f"compare/{base_sha}...{head_sha}"
            comparison = request_or_warn(compare_path, lookup=f"consecutive_diff:{base_sha[:7]}...{head_sha[:7]}")
            if not isinstance(comparison, dict):
                reviews.append(
                    {
                        "index": index,
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "status": "failed",
                        "reason": "GitHub compare API did not return a diff payload.",
                    }
                )
                continue
            compared_commit = _commit_from_compare_payload(comparison, head_commit, base_sha, head_sha)
            diff_evidence = _github_diff_evidence(compared_commit, source_paths, evidence_terms)
            copilot_review = _copilot_diff_relevance_review(
                compared_commit,
                message,
                source_paths,
                evidence_terms,
                diff_evidence,
            )
            review_summary = {
                "index": index,
                "base_sha": base_sha,
                "head_sha": head_sha,
                "commit_message": _summarize_github_commit(compared_commit, "consecutive_history_diff", "low").get("message", ""),
                "compare_url": compared_commit.get("compare_url", ""),
                "changed_files": diff_evidence.get("changed_files", [])[:12],
                "copilot_review": {
                    key: value
                    for key, value in copilot_review.items()
                    if key != "prompt"
                },
            }
            reviews.append(review_summary)
            if copilot_review.get("related"):
                add_candidate(
                    compared_commit,
                    "consecutive_history_diff",
                    25,
                    extra={
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "compare_url": compared_commit.get("compare_url", ""),
                        "copilot_diff_review": copilot_review,
                    },
                )
        return reviews

    commit_sha = _extract_commit_sha(_get_nested_payload_value(payload, COMMIT_SHA_KEYS))
    previous_sha = _extract_commit_sha(_get_nested_payload_value(payload, PREVIOUS_COMMIT_SHA_KEYS))
    consecutive_diff_reviews = review_recent_consecutive_diffs()

    if commit_sha:
        commit = request_or_warn(f"commits/{commit_sha}", lookup="payload_commit_sha")
        if isinstance(commit, dict):
            add_candidate(commit, "payload_commit_sha", 100)
    if previous_sha and commit_sha:
        comparison = request_or_warn(f"compare/{previous_sha}...{commit_sha}", lookup="payload_sha_range")
        if isinstance(comparison, dict):
            for commit in comparison.get("commits", [])[-5:]:
                if isinstance(commit, dict):
                    add_candidate(commit, "payload_sha_range", 75)

    for path in source_paths[:5]:
        path_commits = request_or_warn(
            "commits",
            params={"sha": branch, "path": path, "per_page": 3} if branch else {"path": path, "per_page": 3},
            lookup=f"path:{path}",
        )
        if isinstance(path_commits, list):
            for commit in path_commits:
                if isinstance(commit, dict):
                    add_candidate(commit, f"path:{path}", 45)

    if not candidate_map:
        if evidence_terms:
            try:
                search_commits = _github_search_commits(repo, evidence_terms, _github_lookback_since(payload))
            except Exception as exc:
                LOGGER.warning("GitHub commit search failed: %s", exc)
                lookup_warnings.append({"lookup": "commit_search", "error": str(exc)})
                search_commits = []
            for commit in search_commits:
                commit_text = json.dumps(commit.get("commit", commit), ensure_ascii=True).casefold()
                matched_terms = [term for term in evidence_terms if term.casefold() in commit_text]
                add_candidate(commit, "commit_message_search", 35, matched_terms=matched_terms)

    if not candidate_map:
        recent = request_or_warn(
            "commits",
            params={"sha": branch, "per_page": 10, "since": f"{_github_lookback_since(payload)}T00:00:00Z"}
            if branch
            else {"per_page": 10, "since": f"{_github_lookback_since(payload)}T00:00:00Z"},
            lookup="recent_branch_history",
        )
        if isinstance(recent, list):
            for commit in recent:
                if isinstance(commit, dict):
                    add_candidate(commit, "recent_branch_history", 10)

    candidates = sorted(
        candidate_map.values(),
        key=lambda candidate: (int(candidate.get("score") or 0), candidate.get("committed_at") or candidate.get("authored_at") or ""),
        reverse=True,
    )
    developer_impact_verdict, conclusion = _github_impact_assessment(candidates, lookup_warnings)
    if candidates:
        status = (
            "recent_history_only"
            if all(candidate.get("source") == "recent_branch_history" for candidate in candidates)
            else "candidates_found"
        )
    elif lookup_warnings:
        status = "lookup_failed"
    else:
        status = "no_candidates_found"

    return {
        "status": status,
        "developer_impact_verdict": developer_impact_verdict,
        "conclusion": conclusion,
        "caveat": caveat,
        "repo": repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "previous_sha": previous_sha,
        "source_paths": source_paths,
        "search_terms": evidence_terms,
        "consecutive_diff_limit": _github_history_diff_limit(),
        "consecutive_diff_reviews": consecutive_diff_reviews[: _github_history_diff_limit()],
        "non_developer_cause_possible": True,
        "lookup_warnings": lookup_warnings[:5],
        "candidates": candidates[:5],
    }


def _jira_adf_from_text(text: str) -> dict:
    content = []
    for line in _truncate(text, 25000).splitlines()[:220]:
        content.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": line if line else " "}],
            }
        )
    if not content:
        content.append({"type": "paragraph", "content": [{"type": "text", "text": "No details provided."}]})
    return {"type": "doc", "version": 1, "content": content}


def _parse_jira_labels(raw_labels: str) -> list[str]:
    labels = []
    for token in str(raw_labels or "").split(","):
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", token.strip()).strip("-")
        if label and label not in labels:
            labels.append(label)
    return labels


def _looks_like_jira_priority_error(response_text: str) -> bool:
    text = str(response_text or "").casefold()
    return "priority" in text and any(
        token in text for token in ("valid", "allowed", "field", "cannot", "could not", "does not exist")
    )


def _looks_like_jira_issue_type_error(response_text: str) -> bool:
    text = str(response_text or "").casefold()
    return any(token in text for token in ("issuetype", "issue type")) and any(
        token in text for token in ("valid", "allowed", "field", "cannot", "could not", "does not exist")
    )


def _normalize_jira_site_url(raw_url: str) -> str:
    clean_url = str(raw_url or "").strip().rstrip("/")
    if not clean_url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", clean_url):
        clean_url = "https://" + clean_url
    parsed = urlparse(clean_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return clean_url


def _jira_config_status() -> dict:
    return {
        "site_url": _normalize_jira_site_url(_get_env("LOGMONITOR_JIRA_SITE_URL")),
        "account_email_configured": bool(_get_env("LOGMONITOR_JIRA_ACCOUNT_EMAIL")),
        "api_token_configured": bool(_get_env("LOGMONITOR_JIRA_API_TOKEN")),
        "project_key": _get_env("LOGMONITOR_JIRA_PROJECT_KEY"),
        "issue_type": _get_env("LOGMONITOR_JIRA_ISSUE_TYPE", "Bug") or "Bug",
        "priority_configured": bool(_get_env("LOGMONITOR_JIRA_PRIORITY")),
        "labels": _parse_jira_labels(_get_env("LOGMONITOR_JIRA_LABELS", "log-monitor,ml-triage")),
    }


def _jira_auth_headers(account_email: str, api_token: str) -> dict:
    auth_token = base64.b64encode(f"{account_email}:{api_token}".encode("utf-8")).decode("ascii")
    return {
        "Authorization": f"Basic {auth_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _post_jira_issue_with_fallbacks(
    site_url: str,
    headers: dict,
    fields: dict,
    issue_type: str,
    priority: str,
    issue_type_fallbacks: list[str] | None = None,
) -> tuple[dict, list[str]]:
    issue_url = f"{site_url}/rest/api/3/issue"
    jira_warnings: list[str] = []
    if "summary" in fields:
        fields["summary"] = _jira_summary_text(fields.get("summary"))

    def post_issue() -> requests.Response:
        return requests.post(issue_url, headers=headers, json={"fields": fields}, timeout=30)

    def retry_issue_type_fallbacks(response: requests.Response, reason: str) -> requests.Response:
        fallback_types = issue_type_fallbacks if issue_type_fallbacks is not None else ["Task"]
        current_type = ""
        if isinstance(fields.get("issuetype"), dict):
            current_type = str(fields.get("issuetype", {}).get("name") or "").strip()
        used_issue_types = {str(issue_type or "").casefold(), current_type.casefold()}
        for fallback_type in fallback_types:
            clean_fallback = str(fallback_type or "").strip()
            if not clean_fallback or clean_fallback.casefold() in used_issue_types:
                continue
            used_issue_types.add(clean_fallback.casefold())
            LOGGER.warning("Jira rejected issue creation as '%s'. Retrying as %s.", current_type or issue_type, clean_fallback)
            fields["issuetype"] = {"name": clean_fallback}
            jira_warnings.append(
                f"Jira rejected issue creation as '{current_type or issue_type}' ({reason}), "
                f"so Log Monitor retried it as '{clean_fallback}'."
            )
            response = post_issue()
            if response.status_code < 400:
                return response
            if response.status_code != 400:
                return response
            current_type = clean_fallback
        return response

    response = post_issue()
    if response.status_code >= 400 and priority and _looks_like_jira_priority_error(response.text):
        LOGGER.warning("Jira rejected priority '%s'. Retrying issue creation without priority.", priority)
        fields.pop("priority", None)
        jira_warnings.append(f"Jira rejected configured priority '{priority}', so the issue was created without an explicit priority.")
        response = post_issue()
    if response.status_code >= 400 and _looks_like_jira_issue_type_error(response.text):
        response = retry_issue_type_fallbacks(response, "invalid issue type")
    elif response.status_code == 400 and issue_type_fallbacks:
        response = retry_issue_type_fallbacks(response, "Jira returned 400 Bad Request")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Jira issue creation failed ({response.status_code}). {_truncate(response.text, 2000)}") from exc
    return response.json(), jira_warnings


def _build_jira_description(payload: dict, prediction_result: dict, github_context: dict) -> str:
    message = _extract_message(payload)
    github_candidates = github_context.get("candidates", []) if isinstance(github_context, dict) else []
    if not isinstance(github_candidates, list):
        github_candidates = []
    developer_verdict = (
        str(github_context.get("developer_impact_verdict") or "undetermined")
        if isinstance(github_context, dict)
        else "undetermined"
    )
    github_conclusion = (
        str(github_context.get("conclusion") or "Developer impact remains undetermined.")
        if isinstance(github_context, dict)
        else "Developer impact remains undetermined."
    )
    candidate_lines = []
    for candidate in github_candidates[:5]:
        if not isinstance(candidate, dict):
            continue
        sha = str(candidate.get("sha") or "")[:12]
        author = str(candidate.get("author_login") or candidate.get("author_name") or candidate.get("author_email") or "unknown")
        source = str(candidate.get("source") or "unknown")
        confidence = str(candidate.get("confidence") or "unknown")
        authored_at = str(candidate.get("authored_at") or "")
        commit_message = str(candidate.get("message") or "")
        url = str(candidate.get("html_url") or "")
        diff_evidence = candidate.get("diff_evidence", {}) if isinstance(candidate.get("diff_evidence"), dict) else {}
        diff_bits = []
        if diff_evidence.get("matched_source_paths"):
            diff_bits.append("matched source paths: " + ", ".join(str(path) for path in diff_evidence["matched_source_paths"][:4]))
        if diff_evidence.get("matched_terms_in_diff"):
            diff_bits.append("matched patch terms: " + ", ".join(str(term) for term in diff_evidence["matched_terms_in_diff"][:6]))
        if diff_evidence.get("matched_terms_in_changed_files"):
            diff_bits.append(
                "matched changed-file terms: " + ", ".join(str(term) for term in diff_evidence["matched_terms_in_changed_files"][:6])
            )
        if diff_evidence.get("changed_files"):
            diff_bits.append("changed files: " + ", ".join(str(path) for path in diff_evidence["changed_files"][:5]))
        diff_summary = " | ".join(diff_bits) if diff_bits else "no changed-file or patch-text match found"
        copilot_review = candidate.get("copilot_diff_review") if isinstance(candidate.get("copilot_diff_review"), dict) else {}
        copilot_summary = ""
        if copilot_review:
            copilot_summary = (
                f"\n  Copilot diff review: related={copilot_review.get('related')} "
                f"confidence={copilot_review.get('confidence')} reason={copilot_review.get('reason', '')}"
            )
        candidate_lines.append(
            f"- {sha} by {author} at {authored_at} [{confidence}, {source}] {commit_message} {url}\n"
            f"  Diff evidence: {diff_summary}{copilot_summary}".strip()
        )
        for snippet in diff_evidence.get("diff_snippets", [])[:2] if isinstance(diff_evidence.get("diff_snippets"), list) else []:
            candidate_lines.append(f"  Patch snippet: {snippet}")
    if not candidate_lines:
        candidate_lines.append(
            "- No candidate commit was identified by the automatic lookup. No developer may be responsible, "
            "or the relevant commit evidence may be missing from the payload/history."
        )
    lines = [
        "Log Monitor classified this event as Error.",
        "",
        "Important: GitHub history is included only as investigation context. It can suggest possible developer impact, but it is not proof.",
        f"Developer Impact Verdict: {developer_verdict}",
        f"GitHub Lookup Conclusion: {github_conclusion}",
        "Non-Developer Cause Possible: yes",
        "",
        f"Received At: {payload.get('received_at', _now_utc_iso()) if isinstance(payload, dict) else _now_utc_iso()}",
        f"Prediction: {prediction_result.get('prediction', '')}",
        f"Prediction Endpoint: {prediction_result.get('endpoint_url', '')}",
        f"Source Endpoint Name: {_get_env('LOGMONITOR_SOURCE_ENDPOINT_NAME')}",
        f"Hosting Service Kind: {_get_env('LOGMONITOR_HOSTING_SERVICE_KIND')}",
        "",
        "Log Message:",
        _truncate(message, 5000),
        "",
        "Candidate Commits For Manual Review:",
        "\n".join(candidate_lines),
        "",
        "GitHub Impact Context JSON:",
        json.dumps(github_context, ensure_ascii=True, indent=2)[:10000],
        "",
        "Raw Payload:",
        json.dumps(payload, ensure_ascii=True, indent=2)[:10000],
        "",
        "Raw Prediction Response:",
        json.dumps(prediction_result.get("raw_response", ""), ensure_ascii=True, indent=2)[:5000],
    ]
    return "\n".join(lines)


def _summarize_existing_jira_issue(site_url: str, issue: dict, expected_summary: str) -> dict:
    fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
    status = fields.get("status") if isinstance(fields.get("status"), dict) else {}
    issue_key = str(issue.get("key") or "")
    return {
        "issue_key": issue_key,
        "issue_url": f"{site_url}/browse/{issue_key}" if issue_key else "",
        "operation": "reused",
        "jira_response": issue,
        "jira_summary": str(fields.get("summary") or ""),
        "jira_status": str(status.get("name") or ""),
        "jira_warnings": [],
        "dedupe": {
            "matched": True,
            "match_kind": "summary",
            "expected_summary": expected_summary,
        },
    }


def _find_existing_jira_error_issue(payload: dict) -> dict:
    site_url = _normalize_jira_site_url(_get_env("LOGMONITOR_JIRA_SITE_URL"))
    account_email = _get_env("LOGMONITOR_JIRA_ACCOUNT_EMAIL")
    api_token = _get_env("LOGMONITOR_JIRA_API_TOKEN")
    project_key = _get_env("LOGMONITOR_JIRA_PROJECT_KEY")
    message = _extract_message(payload)
    summary = _jira_summary_text("Log Monitor Error: " + (message or "unclassified runtime error"))
    if not (site_url and account_email and api_token and project_key and message):
        return {
            "status": "skipped",
            "reason": "Jira site, account, token, project key, or log message is not configured.",
            "summary": summary,
        }

    jql = f'project = "{_jira_jql_string(project_key)}" AND summary ~ "{_jira_jql_string(summary)}" ORDER BY updated DESC'
    try:
        response = requests.get(
            f"{site_url}/rest/api/3/search",
            headers=_jira_auth_headers(account_email, api_token),
            params={"jql": jql, "maxResults": 10, "fields": "summary,status,labels"},
            timeout=30,
        )
        response.raise_for_status()
        payload_json = response.json()
    except Exception as exc:
        LOGGER.warning("Jira dedupe lookup failed for summary '%s': %s", summary, exc)
        return {
            "status": "failed",
            "reason": str(exc),
            "summary": summary,
            "jql": jql,
        }

    issues = payload_json.get("issues", []) if isinstance(payload_json, dict) else []
    if not isinstance(issues, list):
        issues = []
    expected_summary = _dedupe_match_text(summary)
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
        if _dedupe_match_text(fields.get("summary")) == expected_summary:
            return {
                "status": "found",
                "summary": summary,
                "jql": jql,
                "issue": _summarize_existing_jira_issue(site_url, issue, summary),
            }
    return {
        "status": "not_found",
        "summary": summary,
        "jql": jql,
        "result_count": len(issues),
    }


def _create_jira_issue(payload: dict, prediction_result: dict, github_context: dict) -> dict:
    site_url = _normalize_jira_site_url(_require_env("LOGMONITOR_JIRA_SITE_URL"))
    account_email = _require_env("LOGMONITOR_JIRA_ACCOUNT_EMAIL")
    api_token = _require_env("LOGMONITOR_JIRA_API_TOKEN")
    project_key = _require_env("LOGMONITOR_JIRA_PROJECT_KEY")
    issue_type = _get_env("LOGMONITOR_JIRA_ISSUE_TYPE", "Bug") or "Bug"
    priority = _get_env("LOGMONITOR_JIRA_PRIORITY")
    labels = _parse_jira_labels(_get_env("LOGMONITOR_JIRA_LABELS", "log-monitor,ml-triage"))
    message = _extract_message(payload)
    summary = _jira_summary_text("Log Monitor Error: " + (message or "unclassified runtime error"))
    description_text = _build_jira_description(payload, prediction_result, github_context)
    fields = {
        "project": {"key": project_key},
        "summary": summary,
        "description": _jira_adf_from_text(description_text),
        "issuetype": {"name": issue_type},
    }
    if labels:
        fields["labels"] = labels
    if priority:
        fields["priority"] = {"name": priority}

    headers = _jira_auth_headers(account_email, api_token)
    result, jira_warnings = _post_jira_issue_with_fallbacks(site_url, headers, fields, issue_type, priority, ["Task"])
    issue_key = str(result.get("key") or "")
    return {
        "issue_key": issue_key,
        "issue_url": f"{site_url}/browse/{issue_key}" if issue_key else "",
        "operation": "created",
        "jira_response": result,
        "jira_warnings": jira_warnings,
    }


def _copilot_remediation_enabled() -> bool:
    return _get_env("LOGMONITOR_COPILOT_REMEDIATION_ENABLED", "1").lower() in {"1", "true", "yes", "on"}


def _github_copilot_assignee() -> str:
    return _get_env("LOGMONITOR_COPILOT_ASSIGNEE", "copilot-swe-agent[bot]") or "copilot-swe-agent[bot]"


def _build_copilot_remediation_prompt(
    payload: dict,
    prediction_result: dict,
    github_context: dict,
    jira_issue: dict,
) -> str:
    message = _extract_message(payload)
    jira_issue_key = str(jira_issue.get("issue_key") or "").strip()
    jira_issue_url = str(jira_issue.get("issue_url") or "").strip()
    repo = _get_env("LOGMONITOR_GITHUB_REPO")
    branch = _clean_github_branch(_get_env("LOGMONITOR_GITHUB_BRANCH")) or "main"
    return "\n".join(
        [
            f"You are GitHub Copilot coding agent acting as a senior software engineer in `{repo}` from `{branch}`.",
            "",
            "Goal:",
            "Fix the production/runtime error captured by Log Monitor, open a pull request, and link it back to the Jira incident.",
            "",
            "Jira incident:",
            f"- Key: {jira_issue_key or 'not provided'}",
            f"- URL: {jira_issue_url or 'not provided'}",
            "",
            "Error message:",
            "```text",
            _truncate(message, 6000),
            "```",
            "",
            "Implementation guidance:",
            "- Treat the Jira ticket as the source incident and this GitHub issue as the engineering repair task.",
            "- Inspect the repository before editing; identify the smallest safe fix for the root cause.",
            "- Use the GitHub history context below as investigation aid only. Do not assume a developer is responsible unless the diff proves it.",
            "- Prefer fixing the real defect over suppressing the error. Add defensive handling only when it preserves intended behavior.",
            "- Add or update focused tests when the repository has an obvious test pattern.",
            "- Run the cheapest relevant formatter/linter/tests available in the repository.",
            "- In the pull request body, include the Jira incident key, the observed error, the root cause, the fix, and verification results.",
            "- If the error is caused by external configuration or infrastructure rather than application code, open the PR with the safest code-side guard or diagnostic improvement and explain the operational action needed.",
            "",
            "Prediction result:",
            json.dumps(prediction_result, ensure_ascii=True, indent=2)[:5000],
            "",
            "GitHub impact context from Log Monitor:",
            json.dumps(github_context, ensure_ascii=True, indent=2)[:10000],
            "",
            "Raw Log Monitor payload:",
            json.dumps(payload, ensure_ascii=True, indent=2)[:10000],
            "",
            f"Created by Log Monitor at {_now_utc_iso()}.",
        ]
    )


def _build_copilot_remediation_issue_body(prompt_text: str, jira_issue: dict) -> str:
    jira_issue_key = str(jira_issue.get("issue_key") or "").strip()
    jira_issue_url = str(jira_issue.get("issue_url") or "").strip()
    jira_line = f"[{jira_issue_key}]({jira_issue_url})" if jira_issue_key and jira_issue_url else jira_issue_key or jira_issue_url
    return "\n".join(
        [
            "Log Monitor created this GitHub issue so Copilot coding agent can prepare a pull request for a Jira incident.",
            "",
            f"Jira incident: {jira_line or 'not provided'}",
            "",
            "Copilot should fix the underlying error, run relevant checks, and include the Jira incident key in the PR body.",
            "",
            "```text",
            prompt_text.strip(),
            "```",
        ]
    )


def _add_jira_comment(issue_key: str, text: str) -> dict:
    clean_issue_key = str(issue_key or "").strip()
    if not clean_issue_key:
        raise RuntimeError("Jira issue key is required to add a comment.")
    site_url = _normalize_jira_site_url(_require_env("LOGMONITOR_JIRA_SITE_URL"))
    account_email = _require_env("LOGMONITOR_JIRA_ACCOUNT_EMAIL")
    api_token = _require_env("LOGMONITOR_JIRA_API_TOKEN")
    response = requests.post(
        f"{site_url}/rest/api/3/issue/{clean_issue_key}/comment",
        headers=_jira_auth_headers(account_email, api_token),
        json={"body": _jira_adf_from_text(text)},
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Jira comment creation failed ({response.status_code}). {_truncate(response.text, 2000)}") from exc
    result = response.json()
    comment_id = str(result.get("id") or "") if isinstance(result, dict) else ""
    return {"comment_id": comment_id}


def _create_github_copilot_remediation_task(
    payload: dict,
    prediction_result: dict,
    github_context: dict,
    jira_issue: dict,
) -> dict:
    if not _copilot_remediation_enabled():
        return {"enabled": False, "reason": "GitHub Copilot remediation is disabled."}
    issue_key = str(jira_issue.get("issue_key") or "").strip()
    if not issue_key:
        raise RuntimeError("A Jira issue must be created before assigning Copilot remediation.")

    repo = _require_env("LOGMONITOR_GITHUB_REPO")
    branch = _clean_github_branch(_get_env("LOGMONITOR_GITHUB_BRANCH")) or "main"
    assignee = _github_copilot_assignee()
    prompt_text = _build_copilot_remediation_prompt(payload, prediction_result, github_context, jira_issue)
    title = _jira_summary_text(f"Fix {issue_key}: {_extract_message(payload) or 'Log Monitor error'}")
    request_payload = {
        "title": title,
        "body": _build_copilot_remediation_issue_body(prompt_text, jira_issue),
        "assignees": [assignee],
        "agent_assignment": {
            "target_repo": repo,
            "base_branch": branch,
            "custom_instructions": prompt_text,
            "custom_agent": _get_env("LOGMONITOR_COPILOT_CUSTOM_AGENT"),
            "model": _get_env("LOGMONITOR_COPILOT_MODEL"),
        },
    }
    issue = _github_post("issues", request_payload)
    issue_number = issue.get("number")
    html_url = str(issue.get("html_url") or "")
    task = {
        "enabled": True,
        "repo_name": repo,
        "base_branch": branch,
        "issue_number": issue_number,
        "issue_url": str(issue.get("url") or ""),
        "html_url": html_url,
        "copilot_assignee": assignee,
        "copilot_model": _get_env("LOGMONITOR_COPILOT_MODEL") or "github-default-best-available",
        "jira_issue_key": issue_key,
        "jira_issue_url": str(jira_issue.get("issue_url") or ""),
        "created_at": _now_utc_iso(),
    }
    try:
        comment = _add_jira_comment(
            issue_key,
            "\n".join(
                [
                    "GitHub Copilot remediation task created.",
                    "",
                    f"GitHub issue: {html_url or task['issue_url']}",
                    f"Repository: {repo}",
                    f"Base branch: {branch}",
                    f"Assignee: {assignee}",
                    "",
                    "Copilot should create a pull request from that GitHub issue when the task is complete.",
                ]
            ),
        )
        task["jira_comment_id"] = comment.get("comment_id", "")
    except Exception as exc:
        task["jira_comment_error"] = str(exc)
    return task


def _monitoring_enabled() -> bool:
    return _get_env("LOGMONITOR_JIRA_MONITORING_ENABLED", "1").lower() in {"1", "true", "yes", "on"}


def _prediction_monitoring_day(received_at: object) -> str:
    text = str(received_at or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    return datetime.now(timezone.utc).date().isoformat()


def _new_monitoring_day(day_key: str) -> dict:
    return {
        "date": day_key,
        "total": 0,
        "counts": {label: 0 for label in LABEL_NAMES},
        "unknown_predictions": {},
        "action_status_counts": {"ok": 0, "partial_failure": 0},
        "jira_created": 0,
        "jira_reused": 0,
        "jira_failed": 0,
        "jira_failure_reasons": {},
        "actions_by_type": {},
        "first_seen_at": _now_utc_iso(),
        "updated_at": _now_utc_iso(),
    }


def _increment_count(container: dict, key: str, amount: int = 1) -> None:
    clean_key = str(key or "unknown")
    container[clean_key] = int(container.get(clean_key, 0) or 0) + int(amount)


def _compact_action_errors(action_errors: list[dict], limit: int = 4) -> list[dict]:
    compact_errors = []
    for action_error in action_errors[:limit]:
        if not isinstance(action_error, dict):
            compact_errors.append({"type": "unknown", "error": _truncate(str(action_error), 1000)})
            continue
        compact_errors.append(
            {
                "type": str(action_error.get("type") or "unknown"),
                "error": _truncate(str(action_error.get("error") or ""), 1000),
            }
        )
    return compact_errors


def _build_monitoring_summary_text(day: dict) -> str:
    counts = day.get("counts") if isinstance(day.get("counts"), dict) else {}
    action_counts = day.get("action_status_counts") if isinstance(day.get("action_status_counts"), dict) else {}
    actions_by_type = day.get("actions_by_type") if isinstance(day.get("actions_by_type"), dict) else {}
    unknown_predictions = day.get("unknown_predictions") if isinstance(day.get("unknown_predictions"), dict) else {}
    jira_failure_reasons = day.get("jira_failure_reasons") if isinstance(day.get("jira_failure_reasons"), dict) else {}
    lines = [
        f"Log Monitor Prediction Summary - {day.get('date', '')}",
        "",
        f"Endpoint: {_get_env('LOGMONITOR_SOURCE_ENDPOINT_NAME')}",
        f"Service kind: {_get_env('LOGMONITOR_HOSTING_SERVICE_KIND')}",
        f"Updated at: {day.get('updated_at', '')}",
        "",
        "Prediction counts:",
    ]
    for label in LABEL_NAMES:
        lines.append(f"- {label}: {int(counts.get(label, 0) or 0)}")
    if unknown_predictions:
        lines.append("- Unknown: " + json.dumps(unknown_predictions, ensure_ascii=True, sort_keys=True))
    lines.extend(
        [
            "",
            f"Total predictions: {int(day.get('total', 0) or 0)}",
            f"Jira incident issues created from Error predictions: {int(day.get('jira_created', 0) or 0)}",
            f"Existing Jira incidents reused from Error predictions: {int(day.get('jira_reused', 0) or 0)}",
            f"Error predictions where Jira incident creation failed: {int(day.get('jira_failed', 0) or 0)}",
        ]
    )
    if jira_failure_reasons:
        lines.extend(["", "Jira incident failure reasons:"])
        sorted_reasons = sorted(
            jira_failure_reasons.items(),
            key=lambda item: int(item[1] or 0),
            reverse=True,
        )
        for reason, count in sorted_reasons[:6]:
            lines.append(f"- {int(count or 0)}x {_truncate(str(reason), 900)}")
    lines.extend(["", "Action status counts:"])
    for key in sorted(action_counts):
        lines.append(f"- {key}: {int(action_counts.get(key, 0) or 0)}")
    lines.append("")
    lines.append("Actions by type:")
    if actions_by_type:
        for key in sorted(actions_by_type):
            lines.append(f"- {key}: {int(actions_by_type.get(key, 0) or 0)}")
    else:
        lines.append("- none: 0")
    last_prediction = day.get("last_prediction") if isinstance(day.get("last_prediction"), dict) else {}
    if last_prediction:
        lines.extend(
            [
                "",
                "Last prediction:",
                json.dumps(last_prediction, ensure_ascii=True, indent=2)[:5000],
            ]
        )
    return "\n".join(lines)


def _upsert_jira_monitoring_issue(day: dict) -> dict:
    site_url = _normalize_jira_site_url(_require_env("LOGMONITOR_JIRA_SITE_URL"))
    account_email = _require_env("LOGMONITOR_JIRA_ACCOUNT_EMAIL")
    api_token = _require_env("LOGMONITOR_JIRA_API_TOKEN")
    project_key = _require_env("LOGMONITOR_JIRA_PROJECT_KEY")
    issue_type = _get_env("LOGMONITOR_JIRA_MONITORING_ISSUE_TYPE", "Task") or "Task"
    configured_issue_type = _get_env("LOGMONITOR_JIRA_ISSUE_TYPE", "Bug") or "Bug"
    labels = _parse_jira_labels(
        _get_env("LOGMONITOR_JIRA_MONITORING_LABELS", "log-monitor,ml-monitoring,prediction-summary")
    )
    summary = _jira_summary_text(f"Log Monitor Prediction Summary - {day.get('date', '')}")
    description = _jira_adf_from_text(_build_monitoring_summary_text(day))
    headers = _jira_auth_headers(account_email, api_token)
    issue_key = str(day.get("jira_summary_issue_key") or "").strip()

    if issue_key:
        response = requests.put(
            f"{site_url}/rest/api/3/issue/{issue_key}",
            headers=headers,
            json={"fields": {"summary": summary, "description": description}},
            timeout=30,
        )
        if response.status_code != 404:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(f"Jira summary update failed ({response.status_code}). {_truncate(response.text, 2000)}") from exc
            return {
                "issue_key": issue_key,
                "issue_url": f"{site_url}/browse/{issue_key}",
                "operation": "updated",
            }

    fields = {
        "project": {"key": project_key},
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type},
    }
    if labels:
        fields["labels"] = labels
    result, jira_warnings = _post_jira_issue_with_fallbacks(
        site_url,
        headers,
        fields,
        issue_type,
        "",
        [configured_issue_type, "Bug", "Task"],
    )
    issue_key = str(result.get("key") or "")
    return {
        "issue_key": issue_key,
        "issue_url": f"{site_url}/browse/{issue_key}" if issue_key else "",
        "operation": "created",
        "jira_warnings": jira_warnings,
    }


def _record_prediction_monitoring(
    payload: dict,
    prediction_result: dict,
    prediction: str,
    actions: list[dict],
    action_errors: list[dict],
    jira_issue: dict,
) -> dict:
    if not _monitoring_enabled():
        return {"enabled": False, "reason": "Jira prediction monitoring is disabled."}

    state = _load_monitoring_state()
    days = state.setdefault("days", {})
    day_key = _prediction_monitoring_day(payload.get("received_at") if isinstance(payload, dict) else "")
    day = days.setdefault(day_key, _new_monitoring_day(day_key))
    counts = day.setdefault("counts", {label: 0 for label in LABEL_NAMES})
    action_status_counts = day.setdefault("action_status_counts", {"ok": 0, "partial_failure": 0})
    actions_by_type = day.setdefault("actions_by_type", {})
    unknown_predictions = day.setdefault("unknown_predictions", {})
    jira_failure_reasons = day.setdefault("jira_failure_reasons", {})
    day.setdefault("jira_reused", 0)

    normalized_prediction = _normalize_prediction_label(prediction) or str(prediction or "Unknown")
    day["total"] = int(day.get("total", 0) or 0) + 1
    if normalized_prediction in LABEL_NAMES:
        _increment_count(counts, normalized_prediction)
    else:
        _increment_count(unknown_predictions, normalized_prediction)

    action_status = "partial_failure" if action_errors else "ok"
    _increment_count(action_status_counts, action_status)
    for action in actions:
        if isinstance(action, dict):
            _increment_count(actions_by_type, str(action.get("type") or "unknown"))
    if normalized_prediction == "Error":
        if jira_issue.get("issue_key"):
            if str(jira_issue.get("operation") or "").casefold() == "reused":
                day["jira_reused"] = int(day.get("jira_reused", 0) or 0) + 1
            else:
                day["jira_created"] = int(day.get("jira_created", 0) or 0) + 1
        else:
            day["jira_failed"] = int(day.get("jira_failed", 0) or 0) + 1
            jira_errors = [
                str(action_error.get("error") or "").strip()
                for action_error in action_errors
                if isinstance(action_error, dict) and str(action_error.get("type") or "") == "jira"
            ]
            if not jira_errors:
                jira_errors = ["Jira issue was not created, and no Jira error details were recorded."]
            for jira_error in jira_errors:
                _increment_count(jira_failure_reasons, _truncate(jira_error, 900))

    day["last_prediction"] = {
        "prediction": normalized_prediction,
        "received_at": payload.get("received_at") if isinstance(payload, dict) else "",
        "action_status": action_status,
        "jira_created": bool(jira_issue.get("issue_key")) and str(jira_issue.get("operation") or "").casefold() != "reused",
        "jira_reused": bool(jira_issue.get("issue_key")) and str(jira_issue.get("operation") or "").casefold() == "reused",
        "jira_issue_key": str(jira_issue.get("issue_key") or ""),
        "action_errors": _compact_action_errors(action_errors),
        "raw_prediction": prediction_result.get("raw_response", ""),
    }
    day["updated_at"] = _now_utc_iso()
    state["latest_day"] = day_key
    state["updated_at"] = day["updated_at"]
    _write_monitoring_state(state)

    summary_issue = _upsert_jira_monitoring_issue(day)
    day["jira_summary_issue_key"] = str(summary_issue.get("issue_key") or "")
    day["jira_summary_issue_url"] = str(summary_issue.get("issue_url") or "")
    day["jira_summary_updated_at"] = _now_utc_iso()
    day["jira_summary_operation"] = str(summary_issue.get("operation") or "")
    state["updated_at"] = day["jira_summary_updated_at"]
    _write_monitoring_state(state)
    return {
        "enabled": True,
        "date": day_key,
        "counts": day.get("counts", {}),
        "total": day.get("total", 0),
        "jira_created": day.get("jira_created", 0),
        "jira_reused": day.get("jira_reused", 0),
        "jira_failed": day.get("jira_failed", 0),
        "jira_failure_reasons": day.get("jira_failure_reasons", {}),
        "jira_summary_issue_key": day.get("jira_summary_issue_key", ""),
        "jira_summary_issue_url": day.get("jira_summary_issue_url", ""),
        "jira_summary_operation": day.get("jira_summary_operation", ""),
    }


def _build_notification_body(payload: dict, prediction_result: dict) -> str:
    return "\n".join(
        [
            f"Log Monitor prediction: {prediction_result.get('prediction', '')}",
            f"Received at: {payload.get('received_at', _now_utc_iso()) if isinstance(payload, dict) else _now_utc_iso()}",
            f"Source endpoint: {_get_env('LOGMONITOR_SOURCE_ENDPOINT_NAME')}",
            "",
            "Log message:",
            _truncate(_extract_message(payload), 6000),
            "",
            "Raw payload:",
            json.dumps(payload, ensure_ascii=True, indent=2)[:10000],
        ]
    )


def _get_blob_service_client() -> BlobServiceClient:
    global _BLOB_SERVICE_CLIENT
    if _BLOB_SERVICE_CLIENT is None:
        _raise_import_error("azure-storage-blob", _BLOB_STORAGE_IMPORT_ERROR)
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
    _raise_import_error("azure-storage-blob", _BLOB_STORAGE_IMPORT_ERROR)
    blob_client = _get_container_client().get_blob_client(blob_name)
    blob_client.upload_blob(
        body.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )


def _get_service_bus_client() -> ServiceBusClient:
    global _SERVICE_BUS_CLIENT
    if _SERVICE_BUS_CLIENT is None:
        _raise_import_error("azure-servicebus", _SERVICE_BUS_IMPORT_ERROR)
        _SERVICE_BUS_CLIENT = ServiceBusClient.from_connection_string(
            _require_env("LOGMONITOR_SERVICEBUS_CONNECTION")
        )
    return _SERVICE_BUS_CLIENT


def _get_ml_client() -> MLClient:
    global _ML_CLIENT
    if _ML_CLIENT is None:
        _raise_import_error("azure-ai-ml", _AZURE_ML_IMPORT_ERROR)
        _raise_import_error("azure-identity", _AZURE_IDENTITY_IMPORT_ERROR)
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
    _raise_import_error("azure-storage-blob", _BLOB_STORAGE_IMPORT_ERROR)
    blob_client = _get_container_client().get_blob_client(_get_feedback_state_blob_name())
    payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True)
    blob_client.upload_blob(
        payload.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )


def _load_monitoring_state() -> dict:
    blob_client = _get_container_client().get_blob_client(_get_monitoring_state_blob_name())
    try:
        body = blob_client.download_blob().readall()
    except ResourceNotFoundError:
        return {}
    except Exception:
        LOGGER.exception("Failed to read prediction monitoring state blob.")
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        LOGGER.exception("Failed to parse prediction monitoring state blob.")
        return {}


def _write_monitoring_state(state: dict) -> None:
    _raise_import_error("azure-storage-blob", _BLOB_STORAGE_IMPORT_ERROR)
    blob_client = _get_container_client().get_blob_client(_get_monitoring_state_blob_name())
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
    _raise_import_error("azure-ai-ml", _AZURE_ML_IMPORT_ERROR)
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
    _raise_import_error("azure-ai-ml", _AZURE_ML_IMPORT_ERROR)
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
    _raise_import_error("azure-ai-ml", _AZURE_ML_IMPORT_ERROR)
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
    _raise_import_error("azure-ai-ml", _AZURE_ML_IMPORT_ERROR)
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
    monitoring_state = _load_monitoring_state()
    return _to_json_response(
        {
            "feedback_state": state,
            "prediction_monitoring_state": monitoring_state,
            "triage_enabled": _get_env("LOGMONITOR_TRIAGE_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
            "jira_config": _jira_config_status(),
        },
        200,
    )


def _execute_triage_actions(normalized: dict, prediction_result: dict, prediction: str, diagnostics: list[dict]) -> dict:
    actions: list[dict] = []
    action_errors: list[dict] = []
    monitoring_errors: list[dict] = []
    github_context: dict = {}
    jira_issue: dict = {}
    monitoring_summary: dict = {}
    jira_config = _jira_config_status()
    message = _extract_message(normalized)
    _add_diagnostic(diagnostics, "jira_config", **jira_config)
    _add_diagnostic(diagnostics, "action_route", prediction=prediction)

    if prediction == "Noise":
        actions.append({"type": "ignore", "reason": "Prediction was Noise."})
        _add_diagnostic(diagnostics, "action_route", action="ignore")
    elif prediction == "CONFIGURATION":
        try:
            email_result = _send_email(
                _require_env("LOGMONITOR_CONFIGURATION_EMAIL"),
                "Log Monitor CONFIGURATION alert",
                _build_notification_body(normalized, prediction_result),
            )
            actions.append({"type": "email", "recipient_kind": "configuration", **email_result})
            _add_diagnostic(diagnostics, "email_send", recipient_kind="configuration", message_id=email_result.get("message_id", ""))
        except Exception as exc:
            LOGGER.exception("Failed to send CONFIGURATION email.")
            action_errors.append({"type": "email", "recipient_kind": "configuration", "error": str(exc)})
            _add_diagnostic(
                diagnostics,
                "email_send",
                "failed",
                recipient_kind="configuration",
                error_type=type(exc).__name__,
                error=str(exc),
            )
    elif prediction == "SYSTEM":
        try:
            email_result = _send_email(
                _require_env("LOGMONITOR_SYSTEM_EMAIL"),
                "Log Monitor SYSTEM alert",
                _build_notification_body(normalized, prediction_result),
            )
            actions.append({"type": "email", "recipient_kind": "system", **email_result})
            _add_diagnostic(diagnostics, "email_send", recipient_kind="system", message_id=email_result.get("message_id", ""))
        except Exception as exc:
            LOGGER.exception("Failed to send SYSTEM email.")
            action_errors.append({"type": "email", "recipient_kind": "system", "error": str(exc)})
            _add_diagnostic(
                diagnostics,
                "email_send",
                "failed",
                recipient_kind="system",
                error_type=type(exc).__name__,
                error=str(exc),
            )
    elif prediction == "Error":
        _add_diagnostic(diagnostics, "github_lookup", "started")
        github_context = _find_github_impact_context(normalized, message)
        _add_diagnostic(
            diagnostics,
            "github_lookup",
            status=str(github_context.get("status") or "unknown"),
            candidates=len(github_context.get("candidates", []) if isinstance(github_context.get("candidates"), list) else []),
            error=github_context.get("error", ""),
        )
        _add_diagnostic(diagnostics, "jira_dedupe_lookup", "started")
        jira_lookup = _find_existing_jira_error_issue(normalized)
        if jira_lookup.get("status") == "found" and isinstance(jira_lookup.get("issue"), dict):
            jira_issue = jira_lookup["issue"]
            actions.append({"type": "jira", **jira_issue})
            _add_diagnostic(
                diagnostics,
                "jira_dedupe_lookup",
                "found",
                issue_key=jira_issue.get("issue_key", ""),
                issue_url=jira_issue.get("issue_url", ""),
                summary=jira_lookup.get("summary", ""),
            )
        else:
            _add_diagnostic(
                diagnostics,
                "jira_dedupe_lookup",
                str(jira_lookup.get("status") or "not_found"),
                reason=jira_lookup.get("reason", ""),
                result_count=jira_lookup.get("result_count", 0),
            )

        if not jira_issue.get("issue_key"):
            _add_diagnostic(diagnostics, "jira_create", "started")
            try:
                jira_issue = _create_jira_issue(normalized, prediction_result, github_context)
                actions.append({"type": "jira", **jira_issue})
                _add_diagnostic(
                    diagnostics,
                    "jira_create",
                    issue_key=jira_issue.get("issue_key", ""),
                    issue_url=jira_issue.get("issue_url", ""),
                    warnings=jira_issue.get("jira_warnings", []),
                )
            except Exception as exc:
                LOGGER.exception("Failed to create Jira issue.")
                action_errors.append({"type": "jira", "error": str(exc), "github_context": github_context, "jira_config": jira_config})
                _add_diagnostic(
                    diagnostics,
                    "jira_create",
                    "failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        else:
            _add_diagnostic(
                diagnostics,
                "jira_create",
                "skipped",
                reason="Existing Jira issue matched this Error event.",
                issue_key=jira_issue.get("issue_key", ""),
            )

        if jira_issue.get("issue_key"):
            _add_diagnostic(diagnostics, "github_remediation_dedupe", "started")
            remediation_lookup = _find_existing_github_remediation(str(jira_issue.get("issue_key") or ""))
            if remediation_lookup.get("status") == "found" and remediation_lookup.get("has_pr"):
                pull_request = remediation_lookup.get("pull_request") if isinstance(remediation_lookup.get("pull_request"), dict) else {}
                actions.append(
                    {
                        "type": "github_pull_request",
                        "operation": "reused",
                        "jira_issue_key": jira_issue.get("issue_key", ""),
                        "number": pull_request.get("number"),
                        "html_url": pull_request.get("html_url", ""),
                        "state": pull_request.get("state", ""),
                        "title": pull_request.get("title", ""),
                    }
                )
                _add_diagnostic(
                    diagnostics,
                    "github_remediation_dedupe",
                    "found",
                    kind="pull_request",
                    number=pull_request.get("number"),
                    html_url=pull_request.get("html_url", ""),
                )
            elif remediation_lookup.get("status") == "found":
                issue = remediation_lookup.get("issue") if isinstance(remediation_lookup.get("issue"), dict) else {}
                actions.append(
                    {
                        "type": "github_copilot_remediation",
                        "operation": "reused",
                        "jira_issue_key": jira_issue.get("issue_key", ""),
                        "issue_number": issue.get("number"),
                        "html_url": issue.get("html_url", ""),
                        "state": issue.get("state", ""),
                        "title": issue.get("title", ""),
                    }
                )
                _add_diagnostic(
                    diagnostics,
                    "github_remediation_dedupe",
                    "found",
                    kind="issue",
                    issue_number=issue.get("number"),
                    html_url=issue.get("html_url", ""),
                )
            else:
                _add_diagnostic(
                    diagnostics,
                    "github_remediation_dedupe",
                    str(remediation_lookup.get("status") or "not_found"),
                    reason=remediation_lookup.get("reason") or remediation_lookup.get("error", ""),
                )
                _add_diagnostic(diagnostics, "copilot_remediation", "started")
                try:
                    copilot_task = _create_github_copilot_remediation_task(
                        normalized,
                        prediction_result,
                        github_context,
                        jira_issue,
                    )
                    if copilot_task.get("enabled") is False:
                        _add_diagnostic(
                            diagnostics,
                            "copilot_remediation",
                            "skipped",
                            reason=copilot_task.get("reason", ""),
                        )
                    else:
                        actions.append({"type": "github_copilot_remediation", **copilot_task})
                        _add_diagnostic(
                            diagnostics,
                            "copilot_remediation",
                            issue_number=copilot_task.get("issue_number", ""),
                            issue_url=copilot_task.get("html_url", ""),
                            assignee=copilot_task.get("copilot_assignee", ""),
                            jira_comment_error=copilot_task.get("jira_comment_error", ""),
                        )
                except Exception as exc:
                    LOGGER.exception("Failed to create GitHub Copilot remediation task.")
                    action_errors.append(
                        {
                            "type": "github_copilot_remediation",
                            "error": str(exc),
                            "jira_issue": jira_issue,
                            "github_repo": _get_env("LOGMONITOR_GITHUB_REPO"),
                        }
                    )
                    _add_diagnostic(
                        diagnostics,
                        "copilot_remediation",
                        "failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
    else:
        actions.append({"type": "none", "reason": f"No triage rule is configured for prediction '{prediction}'."})
        _add_diagnostic(diagnostics, "action_route", "skipped", reason=f"No triage rule is configured for prediction '{prediction}'.")

    _add_diagnostic(diagnostics, "monitoring", "started")
    try:
        monitoring_summary = _record_prediction_monitoring(
            normalized,
            prediction_result,
            prediction,
            actions,
            action_errors,
            jira_issue,
        )
        _add_diagnostic(
            diagnostics,
            "monitoring",
            enabled=monitoring_summary.get("enabled"),
            date=monitoring_summary.get("date", ""),
            total=monitoring_summary.get("total", ""),
            counts=monitoring_summary.get("counts", {}),
            jira_summary_issue_key=monitoring_summary.get("jira_summary_issue_key", ""),
            jira_summary_operation=monitoring_summary.get("jira_summary_operation", ""),
            reason=monitoring_summary.get("reason", ""),
        )
    except Exception as exc:
        LOGGER.exception("Failed to update Jira prediction monitoring summary.")
        monitoring_errors.append({"type": "jira_monitoring", "error": str(exc), "jira_config": jira_config})
        _add_diagnostic(
            diagnostics,
            "monitoring",
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )

    jira_available = bool(jira_issue.get("issue_key"))
    jira_reused = jira_available and str(jira_issue.get("operation") or "").casefold() == "reused"
    jira_created = jira_available and not jira_reused
    status_code = 200
    if prediction == "Error" and not jira_available:
        status_code = 502
    action_status = "partial_failure" if action_errors or monitoring_errors else "ok"
    return {
        "status_code": status_code,
        "accepted": status_code == 200,
        "prediction": prediction,
        "action_status": action_status,
        "actions": actions,
        "action_errors": action_errors,
        "monitoring": monitoring_summary,
        "monitoring_errors": monitoring_errors,
        "github_context": github_context,
        "jira_config": jira_config,
        "jira_issue": jira_issue,
        "jira_available": jira_available,
        "jira_created": jira_created,
        "jira_reused": jira_reused,
        "received_at": normalized.get("received_at"),
    }


def _to_triage_action_response(normalized: dict, result: dict, diagnostics: list[dict]) -> func.HttpResponse:
    _add_diagnostic(
        diagnostics,
        "response",
        status_code=result.get("status_code", 200),
        accepted=bool(result.get("accepted")),
        action_status=result.get("action_status", ""),
        jira_available=bool(result.get("jira_available")),
        jira_created=bool(result.get("jira_created")),
        jira_reused=bool(result.get("jira_reused")),
        action_error_count=len(result.get("action_errors", [])),
        monitoring_error_count=len(result.get("monitoring_errors", [])),
    )
    status_code = int(result.get("status_code", 200) or 200)
    return _to_json_response(
        {
            "accepted": status_code == 200,
            "prediction": result.get("prediction", ""),
            "action_status": result.get("action_status", ""),
            "actions": result.get("actions", []),
            "action_errors": result.get("action_errors", []),
            "monitoring": result.get("monitoring", {}),
            "monitoring_errors": result.get("monitoring_errors", []),
            "github_context": result.get("github_context", {}),
            "jira_config": result.get("jira_config", {}),
            "jira_issue": result.get("jira_issue", {}),
            "jira_available": bool(result.get("jira_available")),
            "jira_created": bool(result.get("jira_created")),
            "jira_reused": bool(result.get("jira_reused")),
            "received_at": normalized.get("received_at"),
            "caveat": (
                "GitHub history is an investigation aid only. A matching or recent commit does not prove developer impact, "
                "and no match can mean no developer is responsible or that evidence was unavailable."
            ),
            "diagnostics": diagnostics,
        },
        status_code,
    )


def _parse_request_payload(req: func.HttpRequest, diagnostics: list[dict]) -> object | func.HttpResponse:
    try:
        payload = req.get_json()
        _add_diagnostic(diagnostics, "payload_parse", payload_shape=_diagnostic_shape(payload))
        return payload
    except ValueError:
        raw_text = req.get_body().decode("utf-8", errors="ignore").strip()
        if not raw_text:
            _add_diagnostic(diagnostics, "payload_parse", "failed", reason="empty body")
            return _to_json_response(
                {
                    "accepted": False,
                    "error": "Send a JSON body or plain-text log message.",
                    "diagnostics": diagnostics,
                },
                400,
            )
        _add_diagnostic(diagnostics, "payload_parse", payload_shape=_diagnostic_shape(raw_text), fallback="plain_text")
        return raw_text


def _validate_triage_message(normalized: dict, diagnostics: list[dict]) -> str | func.HttpResponse:
    message = _extract_message(normalized)
    _add_diagnostic(
        diagnostics,
        "payload_normalized",
        payload_shape=_diagnostic_shape(normalized),
        message_present=bool(message),
        message_length=len(message),
        received_at=normalized.get("received_at"),
    )
    if not message:
        _add_diagnostic(diagnostics, "payload_validation", "failed", reason="missing log message")
        return _to_json_response(
            {
                "accepted": False,
                "error": "Triage payload needs a log message in errorMessage, LogMessage, message, log, msg, or text.",
                "diagnostics": diagnostics,
            },
            400,
        )
    return message


@app.function_name(name="triage_action")
@app.route(route="triage/action", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def triage_action(req: func.HttpRequest) -> func.HttpResponse:
    diagnostics: list[dict] = []
    triage_enabled = _get_env("LOGMONITOR_TRIAGE_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
    _add_diagnostic(diagnostics, "start", route="triage/action", triage_enabled=triage_enabled)
    if not triage_enabled:
        _add_diagnostic(diagnostics, "config", "failed", reason="Azure triage automation is not enabled.")
        return _to_json_response(
            {
                "accepted": False,
                "error": "Azure triage automation is not enabled for this deployment.",
                "diagnostics": diagnostics,
            },
            404,
        )

    payload = _parse_request_payload(req, diagnostics)
    if isinstance(payload, func.HttpResponse):
        return payload

    normalized = _normalize_payload(payload)
    message_or_response = _validate_triage_message(normalized, diagnostics)
    if isinstance(message_or_response, func.HttpResponse):
        return message_or_response

    raw_prediction_result = normalized.get("prediction_result") if isinstance(normalized.get("prediction_result"), dict) else {}
    prediction = (
        _normalize_prediction_label(normalized.get("prediction"))
        or _normalize_prediction_label(raw_prediction_result)
        or _normalize_prediction_label(normalized.get("raw_response"))
    )
    if not prediction:
        _add_diagnostic(diagnostics, "prediction_result", "failed", reason="missing prediction")
        return _to_json_response(
            {
                "accepted": False,
                "error": "Triage action payload needs prediction or prediction_result because prediction has already happened.",
                "diagnostics": diagnostics,
            },
            400,
        )
    prediction_result = dict(raw_prediction_result)
    prediction_result["prediction"] = prediction
    prediction_result.setdefault("raw_response", normalized.get("raw_response", raw_prediction_result or {"prediction": prediction}))
    prediction_result.setdefault(
        "endpoint_url",
        str(normalized.get("source_endpoint_url") or "").strip() or _get_env("LOGMONITOR_SOURCE_ENDPOINT_URL"),
    )
    _add_diagnostic(
        diagnostics,
        "prediction_result",
        prediction=prediction,
        prediction_shape=_diagnostic_shape(prediction_result.get("prediction", "")),
        raw_response_shape=_diagnostic_shape(prediction_result.get("raw_response", "")),
        raw_response_preview=prediction_result.get("raw_response", ""),
        endpoint_url=prediction_result.get("endpoint_url", ""),
        source="already_predicted",
    )
    result = _execute_triage_actions(normalized, prediction_result, prediction, diagnostics)
    return _to_triage_action_response(normalized, result, diagnostics)


@app.function_name(name="triage_log")
@app.route(route="triage", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def triage_log(req: func.HttpRequest) -> func.HttpResponse:
    diagnostics: list[dict] = []
    _add_diagnostic(
        diagnostics,
        "start",
        triage_enabled=_get_env("LOGMONITOR_TRIAGE_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
    )
    if _get_env("LOGMONITOR_TRIAGE_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
        _add_diagnostic(diagnostics, "config", "failed", reason="Azure triage automation is not enabled.")
        return _to_json_response(
            {
                "accepted": False,
                "error": "Azure triage automation is not enabled for this deployment.",
                "diagnostics": diagnostics,
            },
            404,
        )

    try:
        payload = req.get_json()
        _add_diagnostic(diagnostics, "payload_parse", payload_shape=_diagnostic_shape(payload))
    except ValueError:
        raw_text = req.get_body().decode("utf-8", errors="ignore").strip()
        if not raw_text:
            _add_diagnostic(diagnostics, "payload_parse", "failed", reason="empty body")
            return _to_json_response(
                {
                    "accepted": False,
                    "error": "Send a JSON body or plain-text log message.",
                    "diagnostics": diagnostics,
                },
                400,
            )
        payload = raw_text
        _add_diagnostic(diagnostics, "payload_parse", payload_shape=_diagnostic_shape(payload), fallback="plain_text")

    normalized = _normalize_payload(payload)
    message = _extract_message(normalized)
    _add_diagnostic(
        diagnostics,
        "payload_normalized",
        payload_shape=_diagnostic_shape(normalized),
        message_present=bool(message),
        message_length=len(message),
        received_at=normalized.get("received_at"),
    )
    if not message:
        _add_diagnostic(diagnostics, "payload_validation", "failed", reason="missing log message")
        return _to_json_response(
            {
                "accepted": False,
                "error": "Triage payload needs a log message in errorMessage, LogMessage, message, log, msg, or text.",
                "diagnostics": diagnostics,
            },
            400,
        )

    _add_diagnostic(
        diagnostics,
        "prediction_call",
        endpoint_configured=bool(_get_env("LOGMONITOR_PREDICTION_ENDPOINT_URL") or _get_env("LOGMONITOR_SOURCE_ENDPOINT_URL")),
        auth_mode=_get_env("LOGMONITOR_PREDICTION_AUTH_MODE", "key").lower(),
    )
    try:
        prediction_result = _call_prediction_endpoint(normalized)
    except Exception as exc:
        LOGGER.exception("Prediction call failed during triage.")
        _add_diagnostic(
            diagnostics,
            "prediction_call",
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _to_json_response({"accepted": False, "error": str(exc), "diagnostics": diagnostics}, 502)

    prediction = _normalize_prediction_label(prediction_result.get("prediction", "")) or prediction_result.get("prediction", "")
    _add_diagnostic(
        diagnostics,
        "prediction_result",
        prediction=prediction,
        prediction_shape=_diagnostic_shape(prediction_result.get("prediction", "")),
        raw_response_shape=_diagnostic_shape(prediction_result.get("raw_response", "")),
        raw_response_preview=prediction_result.get("raw_response", ""),
        endpoint_url=prediction_result.get("endpoint_url", ""),
    )
    result = _execute_triage_actions(normalized, prediction_result, prediction, diagnostics)
    return _to_triage_action_response(normalized, result, diagnostics)
