import requests


class GitHubService:
    def fetch_repos(self, token: str) -> list[str]:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://api.github.com/user/repos?per_page=100", headers=headers, timeout=60)
        response.raise_for_status()
        repos = response.json()
        return [repo["full_name"] for repo in repos]

    def fetch_branches(self, token: str, repo_name: str) -> list[str]:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"https://api.github.com/repos/{repo_name}/branches", headers=headers, timeout=60)
        response.raise_for_status()
        branches = response.json()
        return [branch["name"] for branch in branches]
