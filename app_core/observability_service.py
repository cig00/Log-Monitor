from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from mlops_utils import clean_optional_string

from .runtime import ArtifactStore, StateStore


Reporter = Callable[[str], None]


class ObservabilityService:
    def __init__(
        self,
        project_dir: str,
        artifact_store: ArtifactStore,
        state_store: StateStore,
    ):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.artifact_store = artifact_store
        self.state_store = state_store
        self._lock = threading.Lock()
        self.hosting_process = None
        self.prometheus_process = None
        self.grafana_process = None

    def find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def terminate_process(self, process, timeout_seconds: int = 5) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def read_process_output(self, process, limit: int = 4000) -> str:
        if process is None or process.stdout is None or process.poll() is None:
            return ""
        try:
            output = process.stdout.read() or ""
        except Exception:
            return ""
        output = output.strip()
        if len(output) > limit:
            output = output[-limit:]
        return output

    def read_file_tail(self, path: str, limit: int = 6000) -> str:
        target = Path(clean_optional_string(path))
        if not target.exists() or not target.is_file():
            return ""
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        text = text.strip()
        if len(text) > limit:
            text = text[-limit:]
        return text

    def shutdown_local_hosting_stack(self) -> None:
        with self._lock:
            processes = [self.grafana_process, self.prometheus_process, self.hosting_process]
            self.grafana_process = None
            self.prometheus_process = None
            self.hosting_process = None
        for process in processes:
            self.terminate_process(process)
        self.state_store.set_value("active_local_hosting", {})

    def get_local_observability_root(self) -> Path:
        return Path(self.artifact_store.local_observability_root)

    def get_local_observability_tools_root(self) -> Path:
        return self.get_local_observability_root() / "tools" / self.get_local_observability_platform_name()

    def get_local_observability_downloads_root(self) -> Path:
        return self.get_local_observability_root() / "downloads" / self.get_local_observability_platform_name()

    def get_local_observability_platform_name(self) -> str:
        if sys.platform == "win32":
            return "windows"
        if sys.platform == "darwin":
            return "macos"
        return "linux"

    def get_observability_binary_names(self, tool_name: str) -> list[str]:
        if tool_name == "grafana-server":
            return ["grafana-server.exe", "grafana.exe"] if sys.platform == "win32" else ["grafana-server", "grafana"]
        if tool_name == "prometheus":
            return ["prometheus.exe"] if sys.platform == "win32" else ["prometheus"]
        return [tool_name + (".exe" if sys.platform == "win32" else "")]

    def find_local_executable(self, candidates: list[str]) -> str:
        for candidate in candidates:
            clean_candidate = clean_optional_string(candidate)
            if not clean_candidate:
                continue
            resolved = shutil.which(clean_candidate)
            if resolved:
                return resolved
            path_candidate = Path(clean_candidate).expanduser()
            if path_candidate.is_file():
                return str(path_candidate)
        return ""

    def find_vendored_observability_binary(self, tool_name: str) -> str:
        root = self.get_local_observability_tools_root()
        if not root.exists():
            return ""
        binary_names = self.get_observability_binary_names(tool_name)
        preferred_root = root / tool_name / "current"
        search_roots = [preferred_root, root]
        seen: set[str] = set()
        for search_root in search_roots:
            try:
                resolved_root = search_root.resolve()
            except Exception:
                resolved_root = search_root
            key = str(resolved_root)
            if key in seen or not search_root.exists():
                continue
            seen.add(key)
            for binary_name in binary_names:
                try:
                    matches = [path for path in search_root.rglob(binary_name) if path.is_file()]
                except Exception:
                    matches = []
                if matches:
                    return str(max(matches, key=lambda path: len(path.parts)))
        return ""

    def get_os_release_info(self) -> dict:
        target = Path("/etc/os-release")
        if not target.exists():
            return {}
        payload: dict[str, str] = {}
        try:
            for raw_line in target.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    payload[key.strip().lower()] = value.strip().strip('"')
        except Exception:
            return {}
        return payload

    def can_auto_install_local_observability(self) -> bool:
        if sys.platform == "win32":
            return platform.machine().lower() in {"amd64", "x86_64", "arm64"}
        if sys.platform == "darwin":
            return bool(shutil.which("brew"))
        if not sys.platform.startswith("linux") or not shutil.which("apt-get") or not shutil.which("pkexec"):
            return False
        release_info = self.get_os_release_info()
        distro_id = clean_optional_string(release_info.get("id", "")).lower()
        distro_like = clean_optional_string(release_info.get("id_like", "")).lower()
        supported_ids = {"ubuntu", "debian"}
        return distro_id in supported_ids or any(token in supported_ids for token in distro_like.split())

    def get_prometheus_binary(self) -> str:
        return self.find_local_executable(
            [
                os.environ.get("PROMETHEUS_BIN", ""),
                self.find_vendored_observability_binary("prometheus"),
                "prometheus",
                "/opt/homebrew/bin/prometheus",
                "/opt/homebrew/opt/prometheus/bin/prometheus",
                "/usr/bin/prometheus",
                "/usr/local/bin/prometheus",
                "/usr/local/opt/prometheus/bin/prometheus",
            ]
        )

    def get_grafana_server_binary(self) -> str:
        return self.find_local_executable(
            [
                os.environ.get("GRAFANA_SERVER_BIN", ""),
                self.find_vendored_observability_binary("grafana-server"),
                "grafana-server",
                "grafana",
                "/opt/homebrew/bin/grafana-server",
                "/opt/homebrew/bin/grafana",
                "/usr/sbin/grafana-server",
                "/usr/bin/grafana-server",
                "/usr/share/grafana/bin/grafana-server",
                "/opt/homebrew/opt/grafana/bin/grafana-server",
                "/opt/homebrew/opt/grafana/bin/grafana",
                "/usr/local/bin/grafana-server",
                "/usr/local/bin/grafana",
                "/usr/local/opt/grafana/bin/grafana-server",
                "/usr/local/opt/grafana/bin/grafana",
            ]
        )

    def get_missing_local_observability_tools(self) -> list[str]:
        missing: list[str] = []
        if not self.get_prometheus_binary():
            missing.append("prometheus")
        if not self.get_grafana_server_binary():
            missing.append("grafana-server")
        return missing

    def get_local_observability_install_script(self) -> str:
        script_path = self.project_dir / "scripts" / "install_local_observability.sh"
        if not script_path.exists():
            raise RuntimeError("The local observability install script is missing from this project.")
        return str(script_path)

    def fetch_text_url(self, url: str, timeout_seconds: int = 60) -> str:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        return response.text

    def download_url_to_file(self, url: str, destination_path: Path) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with open(destination_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    def clear_directory(self, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    def extract_zip_safely(self, archive_path: Path, destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        base_dir = destination_dir.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = (destination_dir / member.filename).resolve()
                if member.filename and member_path != base_dir and base_dir not in member_path.parents:
                    raise RuntimeError(f"Unsafe ZIP archive entry: {member.filename}")
            archive.extractall(destination_dir)

    def extract_tar_safely(self, archive_path: Path, destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        base_dir = destination_dir.resolve()
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                member_path = (destination_dir / member.name).resolve()
                if member.name and member_path != base_dir and base_dir not in member_path.parents:
                    raise RuntimeError(f"Unsafe TAR archive entry: {member.name}")
            archive.extractall(destination_dir)

    def fetch_grafana_windows_download_url(self) -> str:
        page_text = self.fetch_text_url("https://grafana.com/grafana/download?platform=windows")
        match = re.search(r"https://dl\.grafana\.com/[^\s\"']+windows_amd64\.(?:zip|tar\.gz)", page_text)
        if not match:
            raise RuntimeError("Could not find the official Grafana Windows standalone download URL.")
        return match.group(0)

    def fetch_prometheus_windows_download_url(self) -> str:
        page_text = self.fetch_text_url("https://prometheus.io/download/")
        match = re.search(
            r"https://github\.com/prometheus/prometheus/releases/download/[^\s\"']+/prometheus-[^\s\"']+windows-amd64\.zip",
            page_text,
        )
        if not match:
            raise RuntimeError("Could not find the official Prometheus Windows download URL.")
        return match.group(0)

    def download_and_extract_observability_package(self, url: str, tool_name: str) -> str:
        downloads_root = self.get_local_observability_downloads_root()
        tools_root = self.get_local_observability_tools_root()
        archive_name = Path(urlparse(url).path).name or f"{tool_name}.archive"
        archive_path = downloads_root / archive_name
        extract_root = tools_root / tool_name
        current_root = extract_root / "current"
        self.download_url_to_file(url, archive_path)
        self.clear_directory(current_root)
        if archive_name.endswith(".zip"):
            self.extract_zip_safely(archive_path, current_root)
        elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
            self.extract_tar_safely(archive_path, current_root)
        else:
            raise RuntimeError(f"Unsupported archive format for {tool_name}: {archive_name}")
        binary_path = self.find_vendored_observability_binary(tool_name)
        if not binary_path:
            raise RuntimeError(f"The {tool_name} archive was extracted, but the expected binary was not found.")
        return binary_path

    def install_windows_local_observability_dependencies(self, emit: Reporter | None = None) -> None:
        if emit:
            emit("Downloading Prometheus for Windows...")
        self.download_and_extract_observability_package(self.fetch_prometheus_windows_download_url(), "prometheus")
        if emit:
            emit("Downloading Grafana for Windows...")
        self.download_and_extract_observability_package(self.fetch_grafana_windows_download_url(), "grafana-server")

    def install_local_observability_dependencies(self, emit: Reporter | None = None) -> None:
        if sys.platform == "win32":
            self.install_windows_local_observability_dependencies(emit=emit)
            return
        script_path = self.get_local_observability_install_script()
        if sys.platform == "darwin":
            process = subprocess.Popen(["/bin/bash", script_path], cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            output, _ = process.communicate()
            clean_output = (output or "").strip()
            if process.returncode != 0:
                raise RuntimeError("Automatic installation of Grafana and Prometheus did not complete successfully.\n\n" + (clean_output[-4000:] or "Homebrew installation did not complete successfully."))
        else:
            process = subprocess.Popen(["pkexec", "/bin/bash", script_path], cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            output, _ = process.communicate()
            clean_output = (output or "").strip()
            if process.returncode != 0:
                raise RuntimeError("Automatic installation of Grafana and Prometheus did not complete successfully.\n\n" + (clean_output[-4000:] or "Installation was canceled or denied."))
        missing = self.get_missing_local_observability_tools()
        if missing:
            raise RuntimeError("The installation finished, but the required binaries are still missing: " + ", ".join(missing))

    def get_homebrew_prefix(self, formula_name: str) -> str:
        brew_binary = shutil.which("brew")
        if not brew_binary:
            return ""
        try:
            result = subprocess.run([brew_binary, "--prefix", formula_name], capture_output=True, text=True, check=True)
        except Exception:
            return ""
        return clean_optional_string(result.stdout)

    def get_grafana_home(self, grafana_binary: str) -> str:
        explicit_home = clean_optional_string(os.environ.get("GRAFANA_HOME", ""))
        homebrew_prefix = self.get_homebrew_prefix("grafana")
        candidates: list[Path] = []
        if explicit_home:
            candidates.append(Path(explicit_home).expanduser())
        binary_path = Path(grafana_binary).expanduser()
        candidates.extend(
            [
                binary_path.parent.parent,
                binary_path.parent.parent / "share" / "grafana",
                Path("/usr/share/grafana"),
                Path("/usr/local/share/grafana"),
                Path("/opt/homebrew/share/grafana"),
                Path("/opt/homebrew/opt/grafana/share/grafana"),
                Path("/usr/local/opt/grafana/share/grafana"),
            ]
        )
        if homebrew_prefix:
            candidates.append(Path(homebrew_prefix) / "share" / "grafana")
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if (resolved / "conf" / "defaults.ini").exists() and (resolved / "public").exists():
                return str(resolved)
        return ""

    def yaml_quote(self, value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def wait_for_http_endpoint(self, url: str, timeout_seconds: int = 60, ready_statuses: tuple[int, ...] = (200, 302, 401, 403)) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=2, allow_redirects=False)
                if response.status_code in ready_statuses:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def wait_for_process_http_endpoint(self, process, url: str, timeout_seconds: int = 60, ready_statuses: tuple[int, ...] = (200, 302, 401, 403)) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if process is not None and process.poll() is not None:
                return False
            try:
                response = requests.get(url, timeout=2, allow_redirects=False)
                if response.status_code in ready_statuses:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def wait_for_process_http_endpoints(self, process, urls: list[str], timeout_seconds: int = 60, ready_statuses: tuple[int, ...] = (200, 302, 401, 403)) -> bool:
        deadline = time.time() + timeout_seconds
        candidate_urls = [clean_optional_string(url) for url in urls if clean_optional_string(url)]
        while time.time() < deadline:
            if process is not None and process.poll() is not None:
                return False
            for url in candidate_urls:
                try:
                    response = requests.get(url, timeout=2, allow_redirects=False)
                    if response.status_code in ready_statuses:
                        return True
                except Exception:
                    pass
            time.sleep(1)
        return False

    def build_local_grafana_dashboard_json(self, hosting_meta: dict, training_meta: dict, tracking_console_url: str = "", tracking_console_note: str = "") -> str:
        datasource = {"type": "prometheus", "uid": "local-prometheus"}
        api_url = clean_optional_string(hosting_meta.get("api_url"))
        api_home_url = api_url[:-8] if api_url.endswith("/predict") else api_url
        health_url = clean_optional_string(hosting_meta.get("health_url"))
        metrics_url = clean_optional_string(hosting_meta.get("metrics_url"))
        prometheus_url = clean_optional_string(hosting_meta.get("prometheus_url"))
        grafana_url = clean_optional_string(hosting_meta.get("grafana_url"))
        model_version_id = clean_optional_string(hosting_meta.get("model_version_id")) or clean_optional_string(training_meta.get("model_version_id"))
        run_id = clean_optional_string(training_meta.get("run_id"))
        experiment_name = clean_optional_string(training_meta.get("experiment_name"))
        backend = clean_optional_string(training_meta.get("backend")) or "local"
        sample_request = json.dumps({"errorMessage": "timeout while opening socket"}, indent=2)
        details_lines = [
            "# Log Monitor Local Hosting",
            "",
            f"- Prediction endpoint: [{api_url}]({api_url})" if api_url else "- Prediction endpoint: not available",
            f"- API home: [{api_home_url}]({api_home_url})" if api_home_url else "- API home: not available",
            f"- Health check: [{health_url}]({health_url})" if health_url else "- Health check: not available",
            f"- Metrics: [{metrics_url}]({metrics_url})" if metrics_url else "- Metrics: not available",
            f"- Prometheus: [{prometheus_url}]({prometheus_url})" if prometheus_url else "- Prometheus: not available",
            f"- Grafana: [{grafana_url}]({grafana_url})" if grafana_url else "- Grafana: not available",
            f"- Model version: `{model_version_id or 'Not available'}`",
            f"- Training run: `{run_id or 'Not available'}`",
            f"- Experiment: `{experiment_name or 'Not available'}`",
            f"- Tracking backend: `{backend}`",
        ]
        if tracking_console_url:
            details_lines.append(f"- Tracking console: [{tracking_console_url}]({tracking_console_url})")
        elif tracking_console_note:
            details_lines.append(f"- Tracking console: {tracking_console_note}")
        details_lines.extend(["", "## Request Example", "", "```json", sample_request, "```"])
        dashboard = {
            "id": None,
            "uid": "log-monitor-local",
            "title": "Log Monitor Local Hosting",
            "tags": ["log-monitor", "local", "grafana"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "10s",
            "time": {"from": "now-6h", "to": "now"},
            "annotations": {"list": []},
            "editable": False,
            "graphTooltip": 0,
            "panels": [
                {"id": 1, "type": "stat", "title": "Prediction Requests", "datasource": datasource, "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0}, "targets": [{"expr": "sum(log_monitor_predictions_total)", "instant": True, "refId": "A"}]},
                {"id": 8, "type": "text", "title": "Service Notes", "gridPos": {"h": 8, "w": 14, "x": 10, "y": 12}, "options": {"mode": "markdown", "content": "\n".join(details_lines)}},
            ],
            "templating": {"list": []},
        }
        return json.dumps(dashboard, indent=2)

    def write_local_observability_files(self, hosting_meta: dict, training_meta: dict, tracking_console_url: str = "", tracking_console_note: str = "") -> dict:
        root = self.get_local_observability_root()
        prometheus_root = root / "prometheus"
        grafana_root = root / "grafana"
        provisioning_root = grafana_root / "provisioning"
        datasources_root = provisioning_root / "datasources"
        dashboards_root = provisioning_root / "dashboards"
        dashboard_files_root = grafana_root / "dashboards"
        grafana_data_root = grafana_root / "data"
        grafana_logs_root = grafana_root / "logs"
        grafana_plugins_root = grafana_root / "plugins"
        prometheus_data_root = prometheus_root / "data"
        for path in [prometheus_root, prometheus_data_root, datasources_root, dashboards_root, dashboard_files_root, grafana_data_root, grafana_logs_root, grafana_plugins_root]:
            path.mkdir(parents=True, exist_ok=True)
        metrics_url = clean_optional_string(hosting_meta.get("metrics_url"))
        metrics_target = urlparse(metrics_url).netloc or "127.0.0.1:8000"
        prometheus_url = clean_optional_string(hosting_meta.get("prometheus_url"))
        prometheus_config_path = prometheus_root / "prometheus.yml"
        prometheus_config_path.write_text(
            (
                "global:\n  scrape_interval: 5s\n  evaluation_interval: 5s\n\n"
                "scrape_configs:\n"
                "  - job_name: 'log-monitor-local-api'\n"
                "    metrics_path: /metrics\n"
                "    static_configs:\n"
                f"      - targets: [{self.yaml_quote(metrics_target)}]\n"
                "        labels:\n          service: 'log-monitor-local-api'\n"
            ),
            encoding="utf-8",
        )
        datasource_path = datasources_root / "local-prometheus.yml"
        datasource_path.write_text(
            (
                "apiVersion: 1\n"
                "datasources:\n"
                "  - name: Local Prometheus\n"
                "    uid: local-prometheus\n"
                "    type: prometheus\n"
                "    access: proxy\n"
                f"    url: {self.yaml_quote(prometheus_url)}\n"
                "    isDefault: true\n"
                "    editable: false\n"
            ),
            encoding="utf-8",
        )
        dashboard_provider_path = dashboards_root / "log-monitor-local.yml"
        dashboard_provider_path.write_text(
            (
                "apiVersion: 1\nproviders:\n"
                "  - name: Log Monitor Local\n    orgId: 1\n    type: file\n    disableDeletion: false\n"
                "    updateIntervalSeconds: 5\n    allowUiUpdates: false\n    options:\n"
                f"      path: {self.yaml_quote(str(dashboard_files_root.resolve()))}\n"
            ),
            encoding="utf-8",
        )
        dashboard_path = dashboard_files_root / "log-monitor-local-hosting.json"
        dashboard_path.write_text(
            self.build_local_grafana_dashboard_json(
                hosting_meta=hosting_meta,
                training_meta=training_meta,
                tracking_console_url=tracking_console_url,
                tracking_console_note=tracking_console_note,
            ),
            encoding="utf-8",
        )
        return {
            "prometheus_config_path": str(prometheus_config_path.resolve()),
            "prometheus_data_path": str(prometheus_data_root.resolve()),
            "prometheus_launch_log_path": str((prometheus_root / "prometheus.log").resolve()),
            "grafana_provisioning_path": str(provisioning_root.resolve()),
            "grafana_dashboard_path": str(dashboard_path.resolve()),
            "grafana_data_path": str(grafana_data_root.resolve()),
            "grafana_logs_path": str(grafana_logs_root.resolve()),
            "grafana_plugins_path": str(grafana_plugins_root.resolve()),
            "grafana_launch_log_path": str((grafana_logs_root / "grafana-startup.log").resolve()),
        }

    def start_local_prometheus(self, config_path: str, data_path: str, port: int, log_path: str = ""):
        prometheus_binary = self.get_prometheus_binary()
        if not prometheus_binary:
            raise RuntimeError("Prometheus is required for local Grafana hosting, but the `prometheus` binary was not found.")
        log_handle = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("", encoding="utf-8")
            log_handle = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            [prometheus_binary, f"--config.file={config_path}", f"--storage.tsdb.path={data_path}", f"--web.listen-address=127.0.0.1:{port}", "--web.enable-lifecycle"],
            cwd=self.project_dir,
            stdout=log_handle if log_handle is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=log_handle is None,
            bufsize=1 if log_handle is None else -1,
        )
        if log_handle is not None:
            log_handle.close()
        ready_url = f"http://127.0.0.1:{port}/-/ready"
        if not self.wait_for_process_http_endpoint(process, ready_url, timeout_seconds=60, ready_statuses=(200,)):
            output = self.read_file_tail(log_path) if log_path else self.read_process_output(process)
            self.terminate_process(process)
            raise RuntimeError("Prometheus failed to start.\n\n" + output)
        return process

    def start_local_grafana(self, provisioning_path: str, dashboard_path: str, data_path: str, logs_path: str, plugins_path: str, port: int, log_path: str = ""):
        grafana_binary = self.get_grafana_server_binary()
        if not grafana_binary:
            raise RuntimeError("Grafana is required for local hosting, but the `grafana-server` binary was not found.")
        grafana_home = self.get_grafana_home(grafana_binary)
        if not grafana_home:
            raise RuntimeError("The Grafana server binary was found, but the app could not determine Grafana's home directory.")
        env = os.environ.copy()
        env.update(
            {
                "GF_PATHS_DATA": data_path,
                "GF_PATHS_LOGS": logs_path,
                "GF_PATHS_PLUGINS": plugins_path,
                "GF_PATHS_PROVISIONING": provisioning_path,
                "GF_SERVER_HTTP_ADDR": "127.0.0.1",
                "GF_SERVER_HTTP_PORT": str(port),
                "GF_SERVER_ROOT_URL": f"http://127.0.0.1:{port}/",
                "GF_LOG_MODE": "console file",
                "GF_AUTH_ANONYMOUS_ENABLED": "true",
                "GF_AUTH_ANONYMOUS_ORG_NAME": "Main Org.",
                "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
                "GF_AUTH_BASIC_ENABLED": "false",
                "GF_AUTH_DISABLE_LOGIN_FORM": "true",
                "GF_USERS_ALLOW_SIGN_UP": "false",
                "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH": dashboard_path,
            }
        )
        command = [grafana_binary]
        if Path(grafana_binary).stem.lower() == "grafana":
            command.append("server")
        command.extend(["--homepath", grafana_home])
        log_handle = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("", encoding="utf-8")
            log_handle = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=grafana_home,
            env=env,
            stdout=log_handle if log_handle is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=log_handle is None,
            bufsize=1 if log_handle is None else -1,
        )
        if log_handle is not None:
            log_handle.close()
        readiness_urls = [f"http://127.0.0.1:{port}/api/health", f"http://127.0.0.1:{port}/login", f"http://127.0.0.1:{port}/"]
        if not self.wait_for_process_http_endpoints(process, readiness_urls, timeout_seconds=300, ready_statuses=(200, 302, 401, 403)):
            output = self.read_file_tail(log_path) if log_path else self.read_process_output(process)
            self.terminate_process(process)
            raise RuntimeError("Grafana failed to start.\n\n" + output)
        return process
