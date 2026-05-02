from __future__ import annotations

from typing import Any

import requests

from mlops_utils import clean_optional_string, now_utc_iso


class GitHubService:
    def build_headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def fetch_repos(self, token: str) -> list[str]:
        headers = self.build_headers(token)
        response = requests.get("https://api.github.com/user/repos?per_page=100", headers=headers, timeout=60)
        response.raise_for_status()
        repos = response.json()
        return [repo["full_name"] for repo in repos]

    def fetch_branches(self, token: str, repo_name: str) -> list[str]:
        headers = self.build_headers(token)
        response = requests.get(f"https://api.github.com/repos/{repo_name}/branches", headers=headers, timeout=60)
        response.raise_for_status()
        branches = response.json()
        return [branch["name"] for branch in branches]

    def build_log_forwarding_copilot_prompt(
        self,
        *,
        repo_name: str,
        base_branch: str,
        endpoint_url: str,
        endpoint_name: str = "",
        endpoint_auth_mode: str = "",
        service_kind: str = "",
        hosting_mode: str = "",
        azure_studio_endpoint_url: str = "",
        copilot_prompt_version_label: str = "",
        copilot_prompt_version_id: str = "",
    ) -> str:
        clean_repo = clean_optional_string(repo_name)
        clean_branch = clean_optional_string(base_branch)
        clean_endpoint_url = clean_optional_string(endpoint_url)
        clean_endpoint_name = clean_optional_string(endpoint_name)
        clean_auth_mode = clean_optional_string(endpoint_auth_mode) or "not specified"
        clean_service_kind = clean_optional_string(service_kind) or "hosted endpoint"
        clean_hosting_mode = clean_optional_string(hosting_mode) or "hosting"
        clean_azure_studio_endpoint_url = clean_optional_string(azure_studio_endpoint_url)
        clean_prompt_version_label = clean_optional_string(copilot_prompt_version_label)
        clean_prompt_version_id = clean_optional_string(copilot_prompt_version_id)
        clean_auth_mode_key = clean_auth_mode.lower().replace("-", "_").replace(" ", "_")
        endpoint_has_embedded_key = "?code=" in clean_endpoint_url.lower() or "&code=" in clean_endpoint_url.lower()
        if endpoint_has_embedded_key:
            auth_guidance = (
                "Use the exact deployed Function URL above as the forwarding target. It already includes the required "
                "Function access code, so do not split it into a separate key setting and do not ask operators to set "
                "server environment variables for this endpoint."
            )
        elif clean_auth_mode_key == "key":
            auth_guidance = (
                "Use the exact deployed endpoint URL above as the committed forwarding target. If this direct "
                "key-authenticated endpoint cannot be called without a separate key in this repository, implement "
                "the forwarding code but leave a clear TODO for Log Monitor to provide a Function proxy URL instead "
                "of asking users to manually edit server environment variables."
            )
        elif clean_auth_mode_key in {"aad", "aad_token", "entra", "entra_id", "managed_identity"}:
            auth_guidance = (
                "Use the exact deployed endpoint URL above as the committed forwarding target. For identity-authenticated "
                "endpoints, use the repository's existing managed identity or app auth pattern if one already exists."
            )
        else:
            auth_guidance = (
                "Use the exact deployed endpoint URL above as the committed forwarding target. Do not invent extra "
                "operator configuration for Log Monitor unless the repository already has a matching app setting pattern."
            )
        return f"""You are GitHub Copilot coding agent working in `{clean_repo}` from base branch `{clean_branch}`.

Goal:
Integrate non-blocking log forwarding to the Log Monitor endpoint created by the desktop hosting workflow.

Endpoint:
- URL: {clean_endpoint_url}
- Endpoint name: {clean_endpoint_name or "not provided"}
- Hosting mode: {clean_hosting_mode}
- Service kind: {clean_service_kind}
- Authentication mode: {clean_auth_mode}
- Azure ML Studio endpoint page: {clean_azure_studio_endpoint_url or "not provided"}
- Copilot prompt version: {clean_prompt_version_label or "not archived yet"}{f" ({clean_prompt_version_id})" if clean_prompt_version_id else ""}

Primary requirement:
Send application logs and error events to the endpoint without degrading user experience. The application must never wait for the endpoint request to complete on the user-facing request/UI path.

Implementation guidance:
- First inspect the repository to find its language, framework, logging layer, error handlers, and app startup/configuration patterns.
- Add the smallest idiomatic integration for that stack.
- Hard-code the deployed Log Monitor endpoint URL above as the default target in the application code/config committed in this PR.
- Do not require the user to access the server or set environment variables before forwarding can work.
- Do not create documentation that says the Log Monitor endpoint URL is empty by default or must be filled manually on the server.
- If the repository already has a normal config file or constants module, place the URL there as the default. Environment variables may be supported only as optional overrides.
- {auth_guidance}
- Forward logs asynchronously:
  - enqueue logs and drain them in a background worker, task, thread, queue, or framework-native async mechanism;
  - for browser/frontend apps, prefer `navigator.sendBeacon`, `fetch(..., {{ keepalive: true }})`, or an existing telemetry queue;
  - enforce short timeouts, bounded queues, backoff, and drop behavior under pressure;
  - swallow/log forwarding failures locally so they never break the app flow.
- Preserve the existing user experience. No synchronous blocking, modal waits, slow startup path, or visible UI changes unless the repository already exposes telemetry settings.
- Redact obvious secrets and sensitive values before forwarding logs.
- Document the committed default endpoint URL location and how to disable forwarding without requiring server environment variable setup.
- Keep the PR focused on log forwarding only.

Endpoint request contract:
- If this repository can send plain app logs, send each log as JSON including at least `message`, `level`, `timestamp`, `source`, and optional structured metadata.
- If the endpoint is a Log Monitor prediction API that expects `{{"errorMessage": "..."}}`, adapt the sender to that request body.
- Follow the authentication guidance above, and do not block the user path while obtaining or sending endpoint credentials.
- Include the Azure ML Studio endpoint page in the PR description for traceability, but application code must call the deployed endpoint URL, not the Studio page.

Verification:
- Run the existing formatter/linter/tests when they are already available and cheap.
- No new unit tests are required for this task unless the repository has a very clear existing telemetry test pattern.

Created by Log Monitor hosting at {now_utc_iso()}.
"""

    def build_log_forwarding_issue_body(self, prompt_text: str, endpoint_url: str, azure_studio_endpoint_url: str = "") -> str:
        clean_studio_url = clean_optional_string(azure_studio_endpoint_url)
        studio_line = f"Azure ML Studio endpoint page: `{clean_studio_url}`\n\n" if clean_studio_url else ""
        return (
            "Log Monitor created a hosted endpoint and this task asks Copilot coding agent to open a PR "
            "that forwards application logs to it asynchronously.\n\n"
            f"Endpoint URL: `{clean_optional_string(endpoint_url)}`\n\n"
            f"{studio_line}"
            "Important: the PR must preserve user experience by never waiting for endpoint requests on the "
            "user-facing path.\n\n"
            "```text\n"
            f"{prompt_text.strip()}\n"
            "```\n"
        )

    def create_copilot_log_forwarding_pr_task(
        self,
        *,
        token: str,
        repo_name: str,
        base_branch: str,
        endpoint_url: str,
        endpoint_name: str = "",
        endpoint_auth_mode: str = "",
        service_kind: str = "",
        hosting_mode: str = "",
        copilot_model: str = "",
        prompt_text: str = "",
        azure_studio_endpoint_url: str = "",
    ) -> dict[str, Any]:
        clean_token = clean_optional_string(token)
        clean_repo = clean_optional_string(repo_name)
        clean_branch = clean_optional_string(base_branch)
        clean_endpoint_url = clean_optional_string(endpoint_url)
        if not clean_token:
            raise ValueError("GitHub PAT is required to create a Copilot PR task.")
        if not clean_repo:
            raise ValueError("Select a GitHub repository before creating a Copilot PR task.")
        if not clean_branch:
            raise ValueError("Select a GitHub branch before creating a Copilot PR task.")
        if not clean_endpoint_url:
            raise ValueError("Endpoint URL is required to create a Copilot PR task.")

        if not clean_optional_string(prompt_text):
            prompt_text = self.build_log_forwarding_copilot_prompt(
                repo_name=clean_repo,
                base_branch=clean_branch,
                endpoint_url=clean_endpoint_url,
                endpoint_name=endpoint_name,
                endpoint_auth_mode=endpoint_auth_mode,
                service_kind=service_kind,
                hosting_mode=hosting_mode,
                azure_studio_endpoint_url=azure_studio_endpoint_url,
            )
        title = "Integrate async Log Monitor log forwarding"
        payload = {
            "title": title,
            "body": self.build_log_forwarding_issue_body(prompt_text, clean_endpoint_url, azure_studio_endpoint_url),
            "assignees": ["copilot-swe-agent[bot]"],
            "agent_assignment": {
                "target_repo": clean_repo,
                "base_branch": clean_branch,
                "custom_instructions": prompt_text,
                "custom_agent": "",
                "model": clean_optional_string(copilot_model),
            },
        }
        response = requests.post(
            f"https://api.github.com/repos/{clean_repo}/issues",
            headers=self.build_headers(clean_token),
            json=payload,
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                "GitHub could not create the Copilot PR task. Confirm that the repository has Issues enabled, "
                "Copilot coding agent is available for the account/repository, and the token has repo/issues/pull request permissions.\n\n"
                f"GitHub response: {detail[:2000]}"
            ) from exc

        issue = response.json()
        return {
            "title": title,
            "repo_name": clean_repo,
            "base_branch": clean_branch,
            "issue_number": issue.get("number"),
            "issue_url": issue.get("url", ""),
            "html_url": issue.get("html_url", ""),
            "endpoint_url": clean_endpoint_url,
            "endpoint_name": clean_optional_string(endpoint_name),
            "azure_studio_endpoint_url": clean_optional_string(azure_studio_endpoint_url),
            "copilot_assignee": "copilot-swe-agent[bot]",
            "copilot_model": clean_optional_string(copilot_model) or "github-default-best-available",
            "prompt_text": prompt_text,
            "created_at": now_utc_iso(),
        }
