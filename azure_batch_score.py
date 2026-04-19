import csv
import json
import os
from pathlib import Path
from typing import Iterator

from inference_utils import load_model_bundle, predict_error_message


MODEL_BUNDLE = None
MESSAGE_KEYS = (
    "errorMessage",
    "LogMessage",
    "logMessage",
    "message",
    "log",
    "msg",
    "text",
)


def init():
    global MODEL_BUNDLE
    model_root = os.getenv("AZUREML_MODEL_DIR", "")
    MODEL_BUNDLE = load_model_bundle(model_root)


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


def _iter_csv_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        has_header = True
        try:
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            has_header = True

        if has_header:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
            return

        plain_reader = csv.reader(handle)
        for row in plain_reader:
            if not row:
                continue
            yield {"errorMessage": row[0]}


def _iter_json_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            yield item if isinstance(item, dict) else {"errorMessage": item}
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            for item in payload["items"]:
                yield item if isinstance(item, dict) else {"errorMessage": item}
            return
        yield payload
        return
    yield {"errorMessage": payload}


def _iter_jsonl_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                payload = {"errorMessage": line}
            yield payload if isinstance(payload, dict) else {"errorMessage": payload}


def _iter_text_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield {"errorMessage": line}


def _iter_records(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _iter_csv_rows(path)
        return
    if suffix == ".json":
        yield from _iter_json_rows(path)
        return
    if suffix == ".jsonl":
        yield from _iter_jsonl_rows(path)
        return
    if suffix in {".txt", ".log"}:
        yield from _iter_text_rows(path)
        return

    with path.open("r", encoding="utf-8") as handle:
        body = handle.read().strip()
    if body:
        yield {"errorMessage": body}


def run(mini_batch):
    results = []
    for raw_path in mini_batch:
        path = Path(raw_path)
        try:
            for row_index, record in enumerate(_iter_records(path)):
                error_message = _extract_message(record)
                prediction = predict_error_message(MODEL_BUNDLE, error_message)
                results.append(
                    json.dumps(
                        {
                            "source_file": path.name,
                            "row_index": row_index,
                            "errorMessage": error_message,
                            "prediction": prediction,
                        },
                        ensure_ascii=True,
                    )
                )
        except Exception as exc:
            results.append(
                json.dumps(
                    {
                        "source_file": path.name,
                        "row_index": -1,
                        "errorMessage": "",
                        "prediction": "",
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                )
            )
    return results
