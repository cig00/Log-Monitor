from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mlops_utils import discover_model_dir


LABEL_NAMES = ["Error", "CONFIGURATION", "SYSTEM", "Noise"]


def load_model_bundle(model_path: str) -> dict[str, Any]:
    model_dir = discover_model_dir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return {
        "model_dir": model_dir,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }


def predict_error_message(bundle: dict[str, Any], error_message: str) -> str:
    text = str(error_message or "")
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        prediction_index = int(torch.argmax(outputs.logits, dim=-1).item())

    if 0 <= prediction_index < len(LABEL_NAMES):
        return LABEL_NAMES[prediction_index]
    return ""
