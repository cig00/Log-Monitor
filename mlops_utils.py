import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


MLOPS_ENV_VARS = [
    "MLOPS_ENABLED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_PIPELINE_ID",
    "MLFLOW_PARENT_RUN_ID",
    "MLFLOW_RUN_SOURCE",
    "MLFLOW_TAGS_JSON",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bool_from_env(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clean_optional_string(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text


def compute_file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def sidecar_path_for_csv(csv_path: str) -> str:
    return f"{csv_path}.mlmeta.json"


def read_sidecar_for_csv(csv_path: str) -> Optional[Dict[str, Any]]:
    return read_json(sidecar_path_for_csv(csv_path))


def write_sidecar_for_csv(csv_path: str, payload: Dict[str, Any]) -> str:
    path = sidecar_path_for_csv(csv_path)
    write_json(path, payload)
    return path


def local_mlflow_tracking_uri(project_dir: Optional[str] = None) -> str:
    base_dir = Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
    mlruns_dir = base_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return str(mlruns_dir)


def safe_prompt_preview(prompt_text: str, max_chars: int = 2000) -> str:
    text = prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    return text[:max_chars]


def prompt_sha256(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _sanitize_cell(value: Any, max_chars: int = 200) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


def dataframe_metadata(df: Any, label_col: str = "class") -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "row_count": int(len(df)),
        "columns": [str(col) for col in list(df.columns)],
    }
    if label_col in df.columns:
        counts = df[label_col].astype(str).value_counts().to_dict()
        metadata["label_distribution"] = {str(k): int(v) for k, v in counts.items()}
    return metadata


def dataframe_sample(df: Any, max_rows: int = 100, max_cell_chars: int = 200) -> Any:
    sample = df.head(max_rows).copy()
    for column in sample.columns:
        sample[column] = sample[column].apply(lambda value: _sanitize_cell(value, max_cell_chars))
    return sample


def parse_tags_json(raw: str) -> Dict[str, str]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return {}
    tags: Dict[str, str] = {}
    for key, value in parsed.items():
        text_key = str(key).strip()
        if not text_key:
            continue
        tags[text_key] = str(value)
    return tags
