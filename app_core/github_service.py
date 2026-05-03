from __future__ import annotations

import hashlib
from pathlib import Path
from string import Template
from typing import Any

import requests

from mlops_utils import clean_optional_string, now_utc_iso


class GitHubService:
    LOG_FORWARDING_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "log_forwarding_copilot_prompt.txt"

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

    def load_log_forwarding_prompt_template(self) -> str:
        return self.LOG_FORWARDING_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    def get_log_forwarding_prompt_template_info(self) -> dict[str, str]:
        template_text = self.load_log_forwarding_prompt_template()
        template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
        return {
            "copilot_prompt_template_path": str(self.LOG_FORWARDING_PROMPT_TEMPLATE_PATH),
            "copilot_prompt_template_version_id": template_hash,
            "copilot_prompt_template_version_label": template_hash[:12],
        }

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
        prompt_template_info = self.get_log_forwarding_prompt_template_info()
        clean_template_version_label = clean_optional_string(prompt_template_info.get("copilot_prompt_template_version_label"))
        clean_template_version_id = clean_optional_string(prompt_template_info.get("copilot_prompt_template_version_id"))
        clean_rendered_prompt_version = clean_prompt_version_label or "not archived yet"
        if clean_prompt_version_id:
            clean_rendered_prompt_version = f"{clean_rendered_prompt_version} ({clean_prompt_version_id})"
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
        return Template(self.load_log_forwarding_prompt_template()).safe_substitute(
            repo_name=clean_repo,
            base_branch=clean_branch,
            endpoint_url=clean_endpoint_url,
            endpoint_name=clean_endpoint_name or "not provided",
            hosting_mode=clean_hosting_mode,
            service_kind=clean_service_kind,
            endpoint_auth_mode=clean_auth_mode,
            azure_studio_endpoint_url=clean_azure_studio_endpoint_url or "not provided",
            prompt_template_version_label=clean_template_version_label,
            prompt_template_version_id=clean_template_version_id,
            copilot_prompt_version_text=clean_rendered_prompt_version,
            auth_guidance=auth_guidance,
            created_at=now_utc_iso(),
        )

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
