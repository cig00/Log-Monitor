from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from app_core.log_regression_service import (
    ErrorAnalysisReport,
    ask_copilot_about_diffs,
    collect_commit_diffs,
    find_first_occurrence,
    post_report_to_jira,
    report_to_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the first log occurrence of an error, analyze nearby git diffs with OpenAI, and create a Jira report."
    )
    parser.add_argument("--log-file", required=True, help="Path to the log file to scan.")
    parser.add_argument("--error-message", help="Error message substring to find. Reads stdin when omitted.")
    parser.add_argument("--repo", default=".", help="Git repository path. Defaults to the current directory.")
    parser.add_argument("--days-before", type=int, default=3, help="Days before the first occurrence to include.")
    parser.add_argument("--days-after", type=int, default=1, help="Days after the first occurrence to include.")
    parser.add_argument("--ignore-case", action="store_true", help="Match the error message case-insensitively.")
    parser.add_argument("--copilot-command", help="Optional local command override that reads the analysis prompt from stdin.")
    parser.add_argument("--copilot-timeout", type=int, default=180, help="Seconds to wait for OpenAI or the command override.")
    parser.add_argument("--openai-model", default=None, help="OpenAI model for analysis. Defaults to OPENAI_MODEL or gpt-4.1-mini.")
    parser.add_argument("--max-diff-chars", type=int, default=250_000, help="Maximum diff characters sent to OpenAI.")
    parser.add_argument("--output-dir", default="outputs/error_reports", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--no-jira", action="store_true", help="Skip Jira issue creation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    error_message = args.error_message if args.error_message is not None else sys.stdin.read()
    error_message = error_message.strip()
    if not error_message:
        print("Error: provide --error-message or pipe one on stdin.", file=sys.stderr)
        return 2

    occurrence = find_first_occurrence(args.log_file, error_message, ignore_case=args.ignore_case)
    window, commits = collect_commit_diffs(
        args.repo,
        occurrence.timestamp or datetime.now(timezone.utc).isoformat(),
        days_before=args.days_before,
        days_after=args.days_after,
        max_diff_chars=args.max_diff_chars,
    )
    verdict = ask_copilot_about_diffs(
        error_message,
        occurrence,
        commits,
        command=args.copilot_command,
        openai_model=args.openai_model,
        timeout_seconds=args.copilot_timeout,
    )
    report = ErrorAnalysisReport(
        error_message=error_message,
        occurrence=occurrence,
        git_window=window,
        commits=commits,
        copilot=verdict,
        jira_issue=None,
    )

    jira_issue = None
    if not args.no_jira:
        try:
            jira_issue = post_report_to_jira(asdict(report))
            report = replace(report, jira_issue=jira_issue)
        except Exception as exc:
            print(f"Jira issue was not created: {exc}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"error_analysis_{stamp}.json"
    markdown_path = output_dir / f"error_analysis_{stamp}.md"
    json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")

    print(f"Report JSON: {json_path}")
    print(f"Report Markdown: {markdown_path}")
    if jira_issue:
        print(f"Jira issue: {jira_issue.key} {jira_issue.url}")
    if verdict.related is True:
        print(f"OpenAI says the error is related to these changes. Impacted commit: {verdict.impacted_commit}")
    elif verdict.related is False:
        print("OpenAI says the error is not related to these changes.")
    else:
        print(f"OpenAI verdict is inconclusive: {verdict.rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
