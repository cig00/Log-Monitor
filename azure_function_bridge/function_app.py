import csv
import base64
import hashlib
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import azure.functions as func
import requests
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

SOURCE_PATH_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?(?:[\w ._-]+[\\/])*[\w.-]+\.(?:py|js|jsx|ts|tsx|java|cs|go|rb|php|c|cc|cpp|h|hpp|rs|kt|swift))"
)


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


def _truncate(value: object, max_chars: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 30] + "\n...[truncated by Log Monitor]"


def _get_payload_value(payload: dict, keys: tuple[str, ...]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value).strip()
    for container_key in ("metadata", "context", "deployment", "git", "github"):
        nested = payload.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if value not in (None, ""):
                    return str(value).strip()
    return ""


def _normalize_prediction_label(raw_prediction: object) -> str:
    if isinstance(raw_prediction, dict):
        for key in ("prediction", "class", "label", "predictedLabel", "predicted_label"):
            value = raw_prediction.get(key)
            if value not in (None, ""):
                return _normalize_prediction_label(value)
        return ""
    if isinstance(raw_prediction, list) and raw_prediction:
        return _normalize_prediction_label(raw_prediction[0])
    text = str(raw_prediction or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if parsed is not text:
            nested = _normalize_prediction_label(parsed)
            if nested:
                return nested
    except Exception:
        pass
    return LABEL_ALIASES.get(text.casefold(), text)


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


def _extract_source_paths(payload: dict, message: str) -> list[str]:
    candidates: list[str] = []
    for value in (
        _get_payload_value(payload, ("sourcePath", "source_path", "file", "filename", "path")),
        message,
        json.dumps(payload, ensure_ascii=True) if isinstance(payload, dict) else "",
    ):
        if not value:
            continue
        for match in SOURCE_PATH_PATTERN.finditer(value):
            path = match.group("path").replace("\\", "/").strip("/")
            if path and path not in candidates:
                candidates.append(path)
    return candidates[:8]


def _summarize_github_commit(commit: dict, source: str, confidence: str) -> dict:
    payload = commit.get("commit") if isinstance(commit.get("commit"), dict) else {}
    author = payload.get("author") if isinstance(payload.get("author"), dict) else {}
    github_author = commit.get("author") if isinstance(commit.get("author"), dict) else {}
    files = commit.get("files") if isinstance(commit.get("files"), list) else []
    return {
        "sha": str(commit.get("sha") or "")[:40],
        "html_url": str(commit.get("html_url") or ""),
        "message": str(payload.get("message") or "").splitlines()[0][:240],
        "author_name": str(author.get("name") or github_author.get("login") or ""),
        "author_email": str(author.get("email") or ""),
        "author_login": str(github_author.get("login") or ""),
        "authored_at": str(author.get("date") or ""),
        "source": source,
        "confidence": confidence,
        "files": [str(file_info.get("filename") or "") for file_info in files[:20] if isinstance(file_info, dict)],
    }


def _find_github_impact_context(payload: dict, message: str) -> dict:
    caveat = (
        "GitHub commit data is a non-conclusive signal only. "
        "The error may be caused by configuration, infrastructure, external services, data, or runtime state."
    )
    repo = _get_env("LOGMONITOR_GITHUB_REPO")
    branch = _get_env("LOGMONITOR_GITHUB_BRANCH")
    if not _get_env("LOGMONITOR_GITHUB_TOKEN") or not repo:
        return {"status": "skipped", "caveat": caveat, "reason": "GitHub token or repository is not configured.", "candidates": []}

    candidates: list[dict] = []
    seen: set[str] = set()

    def add_candidate(commit: dict, source: str, confidence: str) -> None:
        sha = str(commit.get("sha") or "")
        if not sha or sha in seen:
            return
        seen.add(sha)
        candidates.append(_summarize_github_commit(commit, source, confidence))

    try:
        commit_sha = _get_payload_value(payload, ("commitSha", "commit_sha", "gitSha", "git_sha", "sha"))
        previous_sha = _get_payload_value(payload, ("previousSha", "previous_sha", "baseSha", "base_sha", "beforeSha", "before"))
        if commit_sha:
            commit = _github_request(f"commits/{commit_sha}")
            if isinstance(commit, dict):
                add_candidate(commit, "payload.commitSha", "medium")
        if previous_sha and commit_sha:
            comparison = _github_request(f"compare/{previous_sha}...{commit_sha}")
            if isinstance(comparison, dict):
                for commit in comparison.get("commits", [])[-5:]:
                    if isinstance(commit, dict):
                        add_candidate(commit, "payload.sha_range", "medium")

        source_paths = _extract_source_paths(payload, message)
        for path in source_paths[:5]:
            path_commits = _github_request(
                "commits",
                params={"sha": branch, "path": path, "per_page": 3} if branch else {"path": path, "per_page": 3},
            )
            if isinstance(path_commits, list):
                for commit in path_commits:
                    if isinstance(commit, dict):
                        add_candidate(commit, f"path:{path}", "low")

        if not candidates:
            recent = _github_request("commits", params={"sha": branch, "per_page": 5} if branch else {"per_page": 5})
            if isinstance(recent, list):
                for commit in recent:
                    if isinstance(commit, dict):
                        add_candidate(commit, "recent_branch_history", "low")
    except Exception as exc:
        LOGGER.exception("GitHub impact lookup failed.")
        return {"status": "error", "caveat": caveat, "error": str(exc), "repo": repo, "branch": branch, "candidates": candidates[:5]}

    return {
        "status": "ok" if candidates else "empty",
        "caveat": caveat,
        "repo": repo,
        "branch": branch,
        "source_paths": _extract_source_paths(payload, message),
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

    def post_issue() -> requests.Response:
        return requests.post(issue_url, headers=headers, json={"fields": fields}, timeout=30)

    response = post_issue()
    if response.status_code >= 400 and priority and _looks_like_jira_priority_error(response.text):
        LOGGER.warning("Jira rejected priority '%s'. Retrying issue creation without priority.", priority)
        fields.pop("priority", None)
        jira_warnings.append(f"Jira rejected configured priority '{priority}', so the issue was created without an explicit priority.")
        response = post_issue()
    if response.status_code >= 400 and _looks_like_jira_issue_type_error(response.text):
        fallback_types = issue_type_fallbacks if issue_type_fallbacks is not None else ["Task"]
        used_issue_types = {issue_type.casefold()}
        for fallback_type in fallback_types:
            clean_fallback = str(fallback_type or "").strip()
            if not clean_fallback or clean_fallback.casefold() in used_issue_types:
                continue
            used_issue_types.add(clean_fallback.casefold())
            LOGGER.warning("Jira rejected issue type '%s'. Retrying issue creation as %s.", issue_type, clean_fallback)
            fields["issuetype"] = {"name": clean_fallback}
            jira_warnings.append(
                f"Jira rejected issue type '{issue_type}', so Log Monitor retried issue creation as '{clean_fallback}'."
            )
            response = post_issue()
            if response.status_code < 400 or not _looks_like_jira_issue_type_error(response.text):
                break
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Jira issue creation failed ({response.status_code}). {_truncate(response.text, 2000)}") from exc
    return response.json(), jira_warnings


def _build_jira_description(payload: dict, prediction_result: dict, github_context: dict) -> str:
    message = _extract_message(payload)
    lines = [
        "Log Monitor classified this event as Error.",
        "",
        "Important: GitHub history is included only as investigation context. It is not proof that a developer change caused the error.",
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
        "GitHub Impact Context (non-conclusive):",
        json.dumps(github_context, ensure_ascii=True, indent=2)[:10000],
        "",
        "Raw Payload:",
        json.dumps(payload, ensure_ascii=True, indent=2)[:10000],
        "",
        "Raw Prediction Response:",
        json.dumps(prediction_result.get("raw_response", ""), ensure_ascii=True, indent=2)[:5000],
    ]
    return "\n".join(lines)


def _create_jira_issue(payload: dict, prediction_result: dict, github_context: dict) -> dict:
    site_url = _normalize_jira_site_url(_require_env("LOGMONITOR_JIRA_SITE_URL"))
    account_email = _require_env("LOGMONITOR_JIRA_ACCOUNT_EMAIL")
    api_token = _require_env("LOGMONITOR_JIRA_API_TOKEN")
    project_key = _require_env("LOGMONITOR_JIRA_PROJECT_KEY")
    issue_type = _get_env("LOGMONITOR_JIRA_ISSUE_TYPE", "Bug") or "Bug"
    priority = _get_env("LOGMONITOR_JIRA_PRIORITY")
    labels = _parse_jira_labels(_get_env("LOGMONITOR_JIRA_LABELS", "log-monitor,ml-triage"))
    message = _extract_message(payload)
    summary = _truncate("Log Monitor Error: " + (message.splitlines()[0] if message else "unclassified runtime error"), 240)
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
        "jira_response": result,
        "jira_warnings": jira_warnings,
    }


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
        "jira_failed": 0,
        "actions_by_type": {},
        "first_seen_at": _now_utc_iso(),
        "updated_at": _now_utc_iso(),
    }


def _increment_count(container: dict, key: str, amount: int = 1) -> None:
    clean_key = str(key or "unknown")
    container[clean_key] = int(container.get(clean_key, 0) or 0) + int(amount)


def _build_monitoring_summary_text(day: dict) -> str:
    counts = day.get("counts") if isinstance(day.get("counts"), dict) else {}
    action_counts = day.get("action_status_counts") if isinstance(day.get("action_status_counts"), dict) else {}
    actions_by_type = day.get("actions_by_type") if isinstance(day.get("actions_by_type"), dict) else {}
    unknown_predictions = day.get("unknown_predictions") if isinstance(day.get("unknown_predictions"), dict) else {}
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
            f"Error predictions where Jira incident creation failed: {int(day.get('jira_failed', 0) or 0)}",
            "",
            "Action status counts:",
        ]
    )
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
    summary = _truncate(f"Log Monitor Prediction Summary - {day.get('date', '')}", 240)
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

    normalized_prediction = LABEL_ALIASES.get(str(prediction or "").casefold(), str(prediction or "Unknown"))
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
            day["jira_created"] = int(day.get("jira_created", 0) or 0) + 1
        else:
            day["jira_failed"] = int(day.get("jira_failed", 0) or 0) + 1

    day["last_prediction"] = {
        "prediction": normalized_prediction,
        "received_at": payload.get("received_at") if isinstance(payload, dict) else "",
        "action_status": action_status,
        "jira_created": bool(jira_issue.get("issue_key")),
        "jira_issue_key": str(jira_issue.get("issue_key") or ""),
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


@app.function_name(name="triage_log")
@app.route(route="triage", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def triage_log(req: func.HttpRequest) -> func.HttpResponse:
    if _get_env("LOGMONITOR_TRIAGE_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
        return _to_json_response(
            {
                "accepted": False,
                "error": "Azure triage automation is not enabled for this deployment.",
            },
            404,
        )

    try:
        payload = req.get_json()
    except ValueError:
        raw_text = req.get_body().decode("utf-8", errors="ignore").strip()
        if not raw_text:
            return _to_json_response(
                {
                    "accepted": False,
                    "error": "Send a JSON body or plain-text log message.",
                },
                400,
            )
        payload = raw_text

    normalized = _normalize_payload(payload)
    message = _extract_message(normalized)
    if not message:
        return _to_json_response(
            {
                "accepted": False,
                "error": "Triage payload needs a log message in errorMessage, LogMessage, message, log, msg, or text.",
            },
            400,
        )

    try:
        prediction_result = _call_prediction_endpoint(normalized)
    except Exception as exc:
        LOGGER.exception("Prediction call failed during triage.")
        return _to_json_response({"accepted": False, "error": str(exc)}, 502)

    prediction = LABEL_ALIASES.get(str(prediction_result.get("prediction", "")).casefold(), prediction_result.get("prediction", ""))
    actions: list[dict] = []
    action_errors: list[dict] = []
    monitoring_errors: list[dict] = []
    github_context: dict = {}
    jira_issue: dict = {}
    monitoring_summary: dict = {}
    jira_config = _jira_config_status()

    if prediction == "Noise":
        actions.append({"type": "ignore", "reason": "Prediction was Noise."})
    elif prediction == "CONFIGURATION":
        try:
            email_result = _send_email(
                _require_env("LOGMONITOR_CONFIGURATION_EMAIL"),
                "Log Monitor CONFIGURATION alert",
                _build_notification_body(normalized, prediction_result),
            )
            actions.append({"type": "email", "recipient_kind": "configuration", **email_result})
        except Exception as exc:
            LOGGER.exception("Failed to send CONFIGURATION email.")
            action_errors.append({"type": "email", "recipient_kind": "configuration", "error": str(exc)})
    elif prediction == "SYSTEM":
        try:
            email_result = _send_email(
                _require_env("LOGMONITOR_SYSTEM_EMAIL"),
                "Log Monitor SYSTEM alert",
                _build_notification_body(normalized, prediction_result),
            )
            actions.append({"type": "email", "recipient_kind": "system", **email_result})
        except Exception as exc:
            LOGGER.exception("Failed to send SYSTEM email.")
            action_errors.append({"type": "email", "recipient_kind": "system", "error": str(exc)})
    elif prediction == "Error":
        github_context = _find_github_impact_context(normalized, message)
        try:
            jira_issue = _create_jira_issue(normalized, prediction_result, github_context)
            actions.append({"type": "jira", **jira_issue})
        except Exception as exc:
            LOGGER.exception("Failed to create Jira issue.")
            action_errors.append({"type": "jira", "error": str(exc), "github_context": github_context, "jira_config": jira_config})
    else:
        actions.append({"type": "none", "reason": f"No triage rule is configured for prediction '{prediction}'."})

    try:
        monitoring_summary = _record_prediction_monitoring(
            normalized,
            prediction_result,
            prediction,
            actions,
            action_errors,
            jira_issue,
        )
    except Exception as exc:
        LOGGER.exception("Failed to update Jira prediction monitoring summary.")
        monitoring_errors.append({"type": "jira_monitoring", "error": str(exc), "jira_config": jira_config})

    jira_created = bool(jira_issue.get("issue_key"))
    status_code = 200
    if prediction == "Error" and not jira_created:
        status_code = 502

    return _to_json_response(
        {
            "accepted": status_code == 200,
            "prediction": prediction,
            "action_status": "partial_failure" if action_errors or monitoring_errors else "ok",
            "actions": actions,
            "action_errors": action_errors,
            "monitoring": monitoring_summary,
            "monitoring_errors": monitoring_errors,
            "github_context": github_context,
            "jira_config": jira_config,
            "jira_issue": jira_issue,
            "jira_created": jira_created,
            "received_at": normalized.get("received_at"),
            "caveat": (
                "GitHub history is an investigation aid only. A matching or recent commit does not prove developer impact."
            ),
        },
        status_code,
    )
