import shutil
from pathlib import Path

from mlops_utils import (
    clean_optional_string,
    compute_file_sha256,
    discover_model_dir,
    now_utc_iso,
    read_json,
    write_json,
)

from .contracts import ModelRecord
from .runtime import ArtifactStore


class ModelCatalogService:
    def __init__(self, project_dir: str, artifact_store: ArtifactStore):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.artifact_store = artifact_store

    def is_supported_model_dir(self, candidate: Path) -> bool:
        return candidate.is_dir() and (candidate / "config.json").exists() and (
            (candidate / "pytorch_model.bin").exists()
            or (candidate / "model.safetensors").exists()
            or (candidate / "tf_model.h5").exists()
        )

    def archive_data_version(self, csv_path: str, metadata: dict | None = None) -> dict:
        resolved_csv = Path(csv_path).expanduser().resolve()
        if not resolved_csv.exists():
            return {}

        dataset_hash = compute_file_sha256(str(resolved_csv))
        version_root = self.project_dir / "outputs" / "data_versions" / dataset_hash
        version_root.mkdir(parents=True, exist_ok=True)

        archived_dataset_path = version_root / "dataset.csv"
        try:
            same_target = archived_dataset_path.resolve() == resolved_csv
        except Exception:
            same_target = False
        if not same_target:
            shutil.copy2(str(resolved_csv), str(archived_dataset_path))

        metadata_path = version_root / "metadata.json"
        existing_metadata = read_json(str(metadata_path)) or {}
        payload = {
            "data_version_id": dataset_hash,
            "dataset_hash": dataset_hash,
            "source_dataset_path": str(resolved_csv),
            "archived_dataset_path": str(archived_dataset_path),
            "source_filename": resolved_csv.name,
            "created_at": clean_optional_string(existing_metadata.get("created_at")) or now_utc_iso(),
        }
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    cleaned = clean_optional_string(value)
                    if cleaned:
                        payload[key] = cleaned
                else:
                    payload[key] = value
        write_json(str(metadata_path), payload)
        return {
            "data_version_id": dataset_hash,
            "data_version_dir": str(version_root),
            "data_version_path": str(archived_dataset_path),
        }

    def iter_model_dirs_under(self, root: Path) -> list[Path]:
        discovered: list[Path] = []
        seen: set[str] = set()
        if not root.exists():
            return discovered

        def add_candidate(candidate: Path) -> None:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            key = str(resolved)
            if key in seen or not self.is_supported_model_dir(resolved):
                return
            seen.add(key)
            discovered.append(resolved)

        add_candidate(root)
        if not root.is_dir():
            return discovered

        try:
            config_matches = sorted(root.rglob("config.json"))
        except Exception:
            config_matches = []
        for config_path in config_matches:
            add_candidate(config_path.parent)
        return discovered

    def find_training_metadata_for_model_dir(self, model_dir: Path) -> dict:
        current = model_dir
        for _ in range(6):
            metadata_path = current / "last_training_mlflow.json"
            if metadata_path.exists():
                payload = read_json(str(metadata_path)) or {}
                if payload:
                    payload["_metadata_path"] = str(metadata_path)
                    return payload
            if current.parent == current:
                break
            current = current.parent
        return {}

    def build_model_inventory_label(self, source_label: str, model_dir: Path, metadata: dict) -> str:
        created_at = clean_optional_string(metadata.get("created_at"))
        created_label = created_at.replace("T", " ")[:19] if created_at else "no timestamp"
        version_id = clean_optional_string(metadata.get("model_version_id"))
        run_id = clean_optional_string(metadata.get("run_id"))
        accuracy = ""
        test_metrics = metadata.get("test_metrics")
        if isinstance(test_metrics, dict) and test_metrics.get("accuracy") is not None:
            try:
                accuracy = f"acc {float(test_metrics['accuracy']):.4f}"
            except Exception:
                accuracy = f"acc {test_metrics['accuracy']}"

        parts = [source_label, created_label]
        if version_id:
            parts.append(f"ver {version_id[:18]}")
        elif run_id:
            parts.append(f"run {run_id[:8]}")
        if accuracy:
            parts.append(accuracy)
        parent_name = model_dir.parent.name if model_dir.name == "final_model" else model_dir.name
        if parent_name:
            parts.append(parent_name)
        return " | ".join(parts)

    def discover_available_hosted_models(self, selected_path: str = "") -> list[ModelRecord]:
        inventory: list[ModelRecord] = []
        seen: set[str] = set()

        def add_entry(candidate_path: Path, source_label: str) -> None:
            try:
                resolved_model_dir = Path(discover_model_dir(str(candidate_path))).resolve()
            except Exception:
                return
            key = str(resolved_model_dir)
            if key in seen:
                return
            seen.add(key)
            metadata = self.find_training_metadata_for_model_dir(resolved_model_dir)
            inventory.append(
                ModelRecord(
                    path=str(resolved_model_dir),
                    source=source_label,
                    created_at=clean_optional_string(metadata.get("created_at")),
                    model_version_id=clean_optional_string(metadata.get("model_version_id")),
                    run_id=clean_optional_string(metadata.get("run_id")),
                    label=self.build_model_inventory_label(source_label, resolved_model_dir, metadata),
                )
            )

        add_entry(self.project_dir / "outputs" / "final_model", "Latest local")

        archived_models_root = self.project_dir / "outputs" / "model_versions"
        if archived_models_root.exists():
            for version_dir in sorted(archived_models_root.iterdir(), reverse=True):
                add_entry(version_dir / "final_model", "Archived")

        project_download_root = self.project_dir / "downloaded_model"
        if project_download_root.exists():
            for model_dir in self.iter_model_dirs_under(project_download_root):
                add_entry(model_dir, "Downloaded")

        if clean_optional_string(selected_path):
            add_entry(Path(selected_path), "Selected")

        return sorted(
            inventory,
            key=lambda entry: (clean_optional_string(entry.created_at), clean_optional_string(entry.model_version_id), entry.path),
            reverse=True,
        )

    def save_last_hosting_metadata(self, payload: dict) -> str:
        return self.artifact_store.write_last_hosting_metadata(payload).path

    def read_last_hosting_metadata(self) -> dict:
        return self.artifact_store.read_last_hosting_metadata()

    def get_training_metadata_search_roots(self, hosted_model_path: str = "") -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()

        def add_root(path: Path | None) -> None:
            if path is None:
                return
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                return
            key = str(resolved)
            if key in seen or not resolved.exists():
                return
            seen.add(key)
            roots.append(resolved)

        add_root(self.project_dir / "outputs")
        add_root(self.project_dir / "downloaded_model")

        hosted_model_value = clean_optional_string(hosted_model_path)
        if hosted_model_value:
            try:
                hosted_model_dir = Path(discover_model_dir(hosted_model_value))
            except Exception:
                hosted_model_dir = Path(hosted_model_value).expanduser()
            current = hosted_model_dir
            for _ in range(5):
                add_root(current)
                if current.parent == current:
                    break
                current = current.parent

        return roots

    def describe_training_metadata_search_roots(self, hosted_model_path: str = "") -> list[str]:
        return [str(path) for path in self.get_training_metadata_search_roots(hosted_model_path)]
