from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


ISO_TIMESTAMP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T ][0-9:.]+(?:Z|[+-]\d{2}:?\d{2})?)"
)
APACHE_TIMESTAMP_RE = re.compile(r"\[(?P<ts>\d{2}/[A-Za-z]{3}/\d{4}:[0-9:]+ [+-]\d{4})\]")
SYSLOG_TIMESTAMP_RE = re.compile(r"(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class LogOccurrence:
    log_file: str
    line_number: int
    line: str
    timestamp: str | None
    timestamp_source: str


@dataclass(frozen=True)
class CommitDiff:
    commit: str
    author: str
    date: str
    subject: str
    diff: str


@dataclass(frozen=True)
class CopilotVerdict:
    related: bool | None
    impacted_commit: str | None
    confidence: str | None
    rationale: str
    raw_response: str
    error: str | None = None


@dataclass(frozen=True)
class OpenAIFixPlan:
    unified_diff: str
    summary: str
    tests: str
    raw_response: str


@dataclass(frozen=True)
class JiraIssue:
    key: str
    url: str


@dataclass(frozen=True)
class ErrorAnalysisReport:
    error_message: str
    occurrence: LogOccurrence
    git_window: dict[str, str]
    commits: list[CommitDiff]
    copilot: CopilotVerdict
    jira_issue: JiraIssue | None


def find_first_occurrence(log_file: str | Path, error_message: str, *, ignore_case: bool = False) -> LogOccurrence:
    path = Path(log_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Log file does not exist: {path}")
    if not error_message.strip():
        raise ValueError("error_message must not be empty")

    needle = error_message if not ignore_case else error_message.lower()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            haystack = line if not ignore_case else line.lower()
            if needle in haystack:
                timestamp = extract_timestamp(line)
                if timestamp is None:
                    fallback = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    return LogOccurrence(str(path), line_number, line, fallback.isoformat(), "file_mtime")
                return LogOccurrence(str(path), line_number, line, timestamp.isoformat(), "log_line")

    raise ValueError(f"Error message was not found in {path}")


def extract_timestamp(line: str) -> datetime | None:
    iso_match = ISO_TIMESTAMP_RE.search(line)
    if iso_match:
        value = iso_match.group("ts").replace(" ", "T", 1)
        if value.endswith("Z"):
            value = f"{value[:-1]}+00:00"
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass

    apache_match = APACHE_TIMESTAMP_RE.search(line)
    if apache_match:
        try:
            return datetime.strptime(apache_match.group("ts"), "%d/%b/%Y:%H:%M:%S %z")
        except ValueError:
            pass

    syslog_match = SYSLOG_TIMESTAMP_RE.search(line)
    if syslog_match:
        try:
            parsed = datetime.strptime(syslog_match.group("ts"), "%b %d %H:%M:%S")
            now = datetime.now().astimezone()
            return parsed.replace(year=now.year, tzinfo=now.tzinfo)
        except ValueError:
            pass

    return None


def collect_commit_diffs(
    repo_path: str | Path,
    anchor_timestamp: str,
    *,
    days_before: int = 3,
    days_after: int = 1,
    max_diff_chars: int = 250_000,
) -> tuple[dict[str, str], list[CommitDiff]]:
    repo = Path(repo_path).expanduser().resolve()
    anchor = datetime.fromisoformat(anchor_timestamp.replace("Z", "+00:00"))
    since = anchor - timedelta(days=days_before)
    until = anchor + timedelta(days=days_after)
    window = {"since": since.isoformat(), "until": until.isoformat(), "anchor": anchor.isoformat()}

    commit_rows = _run_git(
        repo,
        [
            "log",
            f"--since={window['since']}",
            f"--until={window['until']}",
            "--format=%H%x00%an%x00%ad%x00%s",
            "--date=iso-strict",
            "--reverse",
        ],
    )
    commits: list[CommitDiff] = []
    remaining_chars = max_diff_chars
    for row in commit_rows.splitlines():
        parts = row.split("\x00", 3)
        if len(parts) != 4:
            continue
        commit, author, date, subject = parts
        if remaining_chars <= 0:
            diff = "\n[Diff omitted because max_diff_chars was reached]\n"
        else:
            diff = _run_git(
                repo,
                [
                    "show",
                    "--format=fuller",
                    "--stat",
                    "--patch",
                    "--find-renames",
                    "--find-copies",
                    "--unified=80",
                    commit,
                ],
            )
            if len(diff) > remaining_chars:
                diff = diff[:remaining_chars] + "\n[Diff truncated]\n"
            remaining_chars -= len(diff)
        commits.append(CommitDiff(commit=commit, author=author, date=date, subject=subject, diff=diff))
    return window, commits


def ask_copilot_about_diffs(
    error_message: str,
    occurrence: LogOccurrence,
    commits: list[CommitDiff],
    *,
    command: str | None = None,
    openai_model: str | None = None,
    timeout_seconds: int = 180,
) -> CopilotVerdict:
    command = command or os.getenv("COPILOT_COMMAND")
    prompt = build_copilot_analysis_prompt(error_message, occurrence, commits)
    if not command:
        return ask_openai_about_diffs(
            error_message,
            occurrence,
            commits,
            model=openai_model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            timeout_seconds=timeout_seconds,
        )

    try:
        completed = subprocess.run(
            shlex.split(command),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CopilotVerdict(
            related=None,
            impacted_commit=None,
            confidence=None,
            rationale=f"Copilot command failed before returning a verdict: {exc}",
            raw_response="",
            error="copilot_command_failed",
        )

    raw_response = completed.stdout.strip() or completed.stderr.strip()
    if completed.returncode != 0:
        return CopilotVerdict(
            related=None,
            impacted_commit=None,
            confidence=None,
            rationale=f"Copilot command exited with status {completed.returncode}.",
            raw_response=raw_response,
            error="copilot_nonzero_exit",
        )

    payload = _extract_json_object(raw_response)
    if payload:
        return CopilotVerdict(
            related=payload.get("related"),
            impacted_commit=payload.get("impacted_commit"),
            confidence=payload.get("confidence"),
            rationale=str(payload.get("rationale", "")).strip(),
            raw_response=raw_response,
        )

    return CopilotVerdict(
        related=None,
        impacted_commit=None,
        confidence=None,
        rationale="Copilot returned a response, but it did not include the requested JSON verdict.",
        raw_response=raw_response,
        error="copilot_unparseable_response",
    )


def ask_openai_about_diffs(
    error_message: str,
    occurrence: LogOccurrence,
    commits: list[CommitDiff],
    *,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    timeout_seconds: int = 180,
) -> CopilotVerdict:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    prompt = build_copilot_analysis_prompt(error_message, occurrence, commits)
    if not api_key:
        return CopilotVerdict(
            related=None,
            impacted_commit=None,
            confidence=None,
            rationale="No OpenAI API key configured. Set OPENAI_API_KEY, or pass --copilot-command to use a local command adapter.",
            raw_response="",
            error="missing_openai_api_key",
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior production debugging assistant. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "error_regression_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "related": {"type": "boolean"},
                        "impacted_commit": {"type": ["string", "null"]},
                        "confidence": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["related", "impacted_commit", "confidence", "rationale"],
                },
            },
        },
    }
    verdict, raw_response, error = _call_openai_json(payload, api_key=api_key, timeout_seconds=timeout_seconds)
    if error:
        return CopilotVerdict(
            related=None,
            impacted_commit=None,
            confidence=None,
            rationale=error,
            raw_response=raw_response,
            error="openai_api_failed",
        )
    return CopilotVerdict(
        related=verdict.get("related"),
        impacted_commit=verdict.get("impacted_commit"),
        confidence=verdict.get("confidence"),
        rationale=str(verdict.get("rationale", "")).strip(),
        raw_response=raw_response,
    )


def ask_openai_for_fix(
    report: dict[str, Any],
    repository_context: str,
    *,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    timeout_seconds: int = 300,
) -> OpenAIFixPlan:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    prompt = "\n\n".join(
        [
            "You are fixing a bug in a git repository.",
            "Return a unified git diff that can be applied with git apply.",
            "Make the smallest correct code change based on the analysis report and repository context.",
            "Do not include markdown fences in the diff.",
            "Analysis report JSON:",
            json.dumps(report, indent=2),
            "Repository context:",
            repository_context,
        ]
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior software engineer. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "error_fix_patch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unified_diff": {"type": "string"},
                        "summary": {"type": "string"},
                        "tests": {"type": "string"},
                    },
                    "required": ["unified_diff", "summary", "tests"],
                },
            },
        },
    }
    result, raw_response, error = _call_openai_json(payload, api_key=api_key, timeout_seconds=timeout_seconds)
    if error:
        raise RuntimeError(error)
    return OpenAIFixPlan(
        unified_diff=str(result.get("unified_diff", "")).strip(),
        summary=str(result.get("summary", "")).strip(),
        tests=str(result.get("tests", "")).strip(),
        raw_response=raw_response,
    )


def build_copilot_analysis_prompt(error_message: str, occurrence: LogOccurrence, commits: list[CommitDiff]) -> str:
    commit_sections = []
    for commit in commits:
        commit_sections.append(
            "\n".join(
                [
                    f"COMMIT: {commit.commit}",
                    f"AUTHOR: {commit.author}",
                    f"DATE: {commit.date}",
                    f"SUBJECT: {commit.subject}",
                    "DIFF:",
                    commit.diff,
                ]
            )
        )
    return "\n\n".join(
        [
            "You are investigating whether a production error was introduced by recent git changes.",
            "Return only a JSON object with keys: related, impacted_commit, confidence, rationale.",
            "Use related=true only when the diff plausibly caused the observed error.",
            f"ERROR MESSAGE:\n{error_message}",
            f"FIRST OCCURRENCE:\n{json.dumps(asdict(occurrence), indent=2)}",
            "CANDIDATE COMMITS AND DIFFS:",
            "\n\n---\n\n".join(commit_sections) if commit_sections else "No commits found in the selected window.",
        ]
    )


def post_report_to_jira(
    report: dict[str, Any],
    *,
    base_url: str | None = None,
    email: str | None = None,
    api_token: str | None = None,
    project_key: str | None = None,
    issue_type: str | None = None,
) -> JiraIssue:
    base_url = (base_url or os.getenv("JIRA_BASE_URL") or "").rstrip("/")
    email = email or os.getenv("JIRA_EMAIL")
    api_token = api_token or os.getenv("JIRA_API_TOKEN")
    project_key = project_key or os.getenv("JIRA_PROJECT_KEY")
    issue_type = issue_type or os.getenv("JIRA_ISSUE_TYPE", "Bug")
    missing = [
        name
        for name, value in {
            "JIRA_BASE_URL": base_url,
            "JIRA_EMAIL": email,
            "JIRA_API_TOKEN": api_token,
            "JIRA_PROJECT_KEY": project_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Jira configuration: {', '.join(missing)}")

    occurrence = report["occurrence"]
    summary = f"Log error regression: {report['error_message'][:180]}"
    description = _jira_adf_document(report)
    response = requests.post(
        f"{base_url}/rest/api/3/issue",
        auth=(email, api_token),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={
            "fields": {
                "project": {"key": project_key},
                "issuetype": {"name": issue_type},
                "summary": summary,
                "description": description,
                "labels": ["log-regression", occurrence["timestamp_source"]],
            }
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    key = payload["key"]
    return JiraIssue(key=key, url=f"{base_url}/browse/{key}")


def report_to_markdown(report: ErrorAnalysisReport | dict[str, Any]) -> str:
    data = asdict(report) if isinstance(report, ErrorAnalysisReport) else report
    lines = [
        f"# Error Analysis Report",
        "",
        f"Error: `{data['error_message']}`",
        "",
        "## First Occurrence",
        f"- File: `{data['occurrence']['log_file']}`",
        f"- Line: {data['occurrence']['line_number']}",
        f"- Timestamp: {data['occurrence']['timestamp']} ({data['occurrence']['timestamp_source']})",
        f"- Log line: `{data['occurrence']['line']}`",
        "",
        "## Git Window",
        f"- Since: {data['git_window']['since']}",
        f"- Until: {data['git_window']['until']}",
        "",
        "## OpenAI Verdict",
        f"- Related: {data['copilot']['related']}",
        f"- Impacted commit: {data['copilot']['impacted_commit']}",
        f"- Confidence: {data['copilot']['confidence']}",
        f"- Rationale: {data['copilot']['rationale']}",
        "",
        "## Candidate Commits",
    ]
    for commit in data["commits"]:
        lines.append(f"- `{commit['commit']}` {commit['date']} {commit['subject']}")
    if data.get("jira_issue"):
        lines.extend(["", "## Jira", f"- {data['jira_issue']['key']}: {data['jira_issue']['url']}"])
    return "\n".join(lines).strip() + "\n"


def _jira_adf_document(report: dict[str, Any]) -> dict[str, Any]:
    markdown = report_to_markdown(report)
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": markdown[:32000]}],
            }
        ],
    }


def _run_git(repo: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _call_openai_json(payload: dict[str, Any], *, api_key: str, timeout_seconds: int) -> tuple[dict[str, Any], str, str | None]:
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        body = ""
        if getattr(exc, "response", None) is not None:
            body = exc.response.text[:2000]
        return {}, body, f"OpenAI API request failed: {exc}"

    raw_payload = response.json()
    content = raw_payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        return {}, json.dumps(raw_payload), "OpenAI API response did not contain message content."
    parsed = _extract_json_object(content)
    if parsed is None:
        return {}, content, "OpenAI API response was not valid JSON."
    return parsed, content, None
