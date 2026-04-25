from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app_core.log_regression_service import ask_openai_for_fix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OpenAI, or an optional local fixer command, to patch an analyzed error and open a GitHub PR."
    )
    parser.add_argument("--report", required=True, help="JSON report created by analyze_log_error.py.")
    parser.add_argument("--repo", default=".", help="Git repository path. Defaults to the current directory.")
    parser.add_argument("--base-branch", help="Base branch for the PR. Defaults to current branch.")
    parser.add_argument("--branch-prefix", default="fix/log-error", help="Prefix for the generated branch name.")
    parser.add_argument("--fix-command", help="Command that reads the fix prompt from stdin and edits the repo.")
    parser.add_argument("--openai-model", default=None, help="OpenAI model for patch generation. Defaults to OPENAI_MODEL or gpt-4.1-mini.")
    parser.add_argument("--commit-message", help="Commit message. Defaults to the Jira issue or error summary.")
    parser.add_argument("--draft", action="store_true", help="Create the pull request as draft.")
    parser.add_argument("--skip-push", action="store_true", help="Commit locally but do not push or create a PR.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow starting from a dirty worktree.")
    parser.add_argument("--fix-timeout", type=int, default=900, help="Seconds to wait for the fix command.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fix_command = args.fix_command or os.getenv("COPILOT_FIX_COMMAND")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not fix_command and not openai_api_key:
        print("No fixer configured. Set OPENAI_API_KEY, or set COPILOT_FIX_COMMAND/pass --fix-command.")
        return 2

    if not args.allow_dirty and _git(repo, ["status", "--porcelain"]).strip():
        print("Worktree is dirty. Commit/stash changes or rerun with --allow-dirty.")
        return 2

    base_branch = args.base_branch or _git(repo, ["branch", "--show-current"]).strip()
    branch_name = _unique_branch_name(repo, args.branch_prefix, report)
    _git(repo, ["checkout", "-b", branch_name])

    if fix_command:
        prompt = _build_fix_prompt(report, base_branch)
        completed = subprocess.run(
            shlex.split(fix_command),
            cwd=str(repo),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=args.fix_timeout,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr)
            print(f"Fix command exited with status {completed.returncode}.")
            return completed.returncode
    else:
        context = _collect_repository_context(repo)
        fix_plan = ask_openai_for_fix(
            report,
            context,
            model=args.openai_model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            timeout_seconds=args.fix_timeout,
        )
        if not fix_plan.unified_diff:
            print("OpenAI returned an empty unified diff.")
            return 1
        apply_error = _apply_unified_diff(repo, fix_plan.unified_diff)
        if apply_error:
            failed_patch_path = _save_failed_patch(repo, branch_name, fix_plan.unified_diff, apply_error)
            print("OpenAI returned a patch, but git could not apply it.")
            print(f"git apply error: {apply_error}")
            print(f"Failed patch saved to: {failed_patch_path}")
            print("No commit or PR was created.")
            return 1

    status = _git(repo, ["status", "--porcelain"]).strip()
    if not status:
        print("Fix command completed, but it did not modify any tracked or untracked files.")
        return 1

    _git(repo, ["add", "--all"])
    commit_message = args.commit_message or _default_commit_message(report)
    _git(repo, ["commit", "-m", commit_message])

    if args.skip_push:
        print(f"Created local branch and commit: {branch_name}")
        return 0

    _git(repo, ["push", "-u", "origin", branch_name])
    pr_body_path = repo / ".git" / f"pr_body_{branch_name.replace('/', '_')}.md"
    pr_body_path.write_text(_build_pr_body(report), encoding="utf-8")
    gh_args = [
        "pr",
        "create",
        "--base",
        base_branch,
        "--head",
        branch_name,
        "--title",
        commit_message,
        "--body-file",
        str(pr_body_path),
    ]
    if args.draft:
        gh_args.append("--draft")
    output = _run(["gh", *gh_args], repo)
    print(output.strip())
    return 0


def _build_fix_prompt(report: dict, base_branch: str) -> str:
    return "\n\n".join(
        [
            "You are fixing a bug in this git repository.",
            "Use the analysis report below to identify and implement the smallest correct code change.",
            "After editing, run the most relevant tests if practical. Do not create the git commit or PR.",
            f"Base branch: {base_branch}",
            "Analysis report JSON:",
            json.dumps(report, indent=2),
        ]
    )


def _build_pr_body(report: dict) -> str:
    jira = report.get("jira_issue") or {}
    copilot = report.get("copilot") or {}
    occurrence = report.get("occurrence") or {}
    return "\n".join(
        [
            "## Error",
            report.get("error_message", ""),
            "",
            "## Analysis",
            f"- First occurrence: `{occurrence.get('log_file')}` line {occurrence.get('line_number')}",
            f"- Impacted commit: `{copilot.get('impacted_commit')}`",
            f"- Jira: {jira.get('url', 'not created')}",
            "",
            "## Verification",
            "- Implemented by OpenAI or configured local fix command.",
        ]
    )


def _default_commit_message(report: dict) -> str:
    jira = report.get("jira_issue") or {}
    key = jira.get("key")
    prefix = f"{key}: " if key else ""
    error = re.sub(r"\s+", " ", report.get("error_message", "")).strip()
    return f"{prefix}Fix log error: {error[:80]}"


def _unique_branch_name(repo: Path, prefix: str, report: dict) -> str:
    jira = report.get("jira_issue") or {}
    suffix_source = jira.get("key") or report.get("error_message", "error")
    suffix = re.sub(r"[^a-zA-Z0-9]+", "-", suffix_source.lower()).strip("-")[:40]
    suffix = suffix or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    candidate = f"{prefix}/{suffix}"
    existing = _git(repo, ["branch", "--list", candidate]).strip()
    if not existing:
        return candidate
    return f"{candidate}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def _collect_repository_context(repo: Path, *, max_files: int = 25, max_chars: int = 120_000) -> str:
    file_list = _git(repo, ["ls-files"]).splitlines()
    selected = []
    preferred_suffixes = (".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml")
    for relative_path in file_list:
        if len(selected) >= max_files:
            break
        path = repo / relative_path
        if not path.is_file() or path.suffix not in preferred_suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        selected.append(f"FILE: {relative_path}\n{content}")
    context = "\n\n---\n\n".join(selected)
    if len(context) > max_chars:
        return context[:max_chars] + "\n[Repository context truncated]\n"
    return context


def _apply_unified_diff(repo: Path, unified_diff: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "apply", "--whitespace=fix"],
        input=unified_diff,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return completed.stderr.strip() or completed.stdout.strip()
    return None


def _save_failed_patch(repo: Path, branch_name: str, unified_diff: str, apply_error: str) -> Path:
    output_dir = repo / "outputs" / "failed_patches"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_branch = branch_name.replace("/", "_")
    patch_path = output_dir / f"{safe_branch}_{stamp}.patch"
    patch_path.write_text(
        "\n".join(
            [
                f"# git apply error: {apply_error}",
                "",
                unified_diff,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return patch_path


def _git(repo: Path, args: list[str]) -> str:
    return _run(["git", "-C", str(repo), *args], repo)


def _run(command: list[str], repo: Path) -> str:
    completed = subprocess.run(command, cwd=str(repo), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
