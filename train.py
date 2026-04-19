import argparse
import itertools
import os
import random
import shutil
import tempfile
from contextlib import nullcontext
from typing import Any, Dict

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mlops_utils import (
    bool_from_env,
    clean_optional_string,
    compute_file_sha256,
    dataframe_metadata,
    dataframe_sample,
    now_utc_iso,
    parse_tags_json,
    write_json,
)

try:
    import mlflow
except Exception:
    mlflow = None


LABEL_MAPPING = {"Error": 0, "CONFIGURATION": 1, "SYSTEM": 2, "Noise": 3}
LABEL_NAMES = [name for name, _ in sorted(LABEL_MAPPING.items(), key=lambda item: item[1])]
MODEL_NAME = "microsoft/deberta-v3-xsmall"


class LogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_mlflow_context() -> Dict[str, Any]:
    raw_tags = clean_optional_string(os.getenv("MLFLOW_TAGS_JSON", "{}"))
    try:
        tags = parse_tags_json(raw_tags)
    except Exception:
        tags = {}

    context: Dict[str, Any] = {
        "enabled": False,
        "tracking_uri": "",
        "experiment_name": "",
        "pipeline_id": clean_optional_string(os.getenv("MLFLOW_PIPELINE_ID", "")),
        "parent_run_id": clean_optional_string(os.getenv("MLFLOW_PARENT_RUN_ID", "")),
        "run_source": clean_optional_string(os.getenv("MLFLOW_RUN_SOURCE", "")) or "unknown",
        "tags": tags,
    }

    if not bool_from_env(os.getenv("MLOPS_ENABLED", "0")):
        return context

    if mlflow is None:
        print("[MLOPS] `mlflow` package unavailable; continuing without MLflow logging.")
        return context

    tracking_uri = clean_optional_string(os.getenv("MLFLOW_TRACKING_URI", ""))
    experiment_name = clean_optional_string(os.getenv("MLFLOW_EXPERIMENT_NAME", ""))
    if not tracking_uri or not experiment_name:
        print("[MLOPS] Missing tracking URI or experiment name; continuing without MLflow logging.")
        return context

    context["enabled"] = True
    context["tracking_uri"] = tracking_uri
    context["experiment_name"] = experiment_name
    context["tags"] = tags
    return context


def safe_mlflow_call(enabled: bool, fn, description: str) -> None:
    if not enabled:
        return
    try:
        fn()
    except Exception as exc:
        print(f"[MLOPS] {description} failed: {exc}")


def build_model_version_id(created_at: str, run_id: str, dataset_hash: str) -> str:
    timestamp_token = "".join(ch for ch in clean_optional_string(created_at) if ch.isalnum())
    suffix_source = clean_optional_string(run_id) or clean_optional_string(dataset_hash) or "local"
    suffix_token = "".join(ch for ch in suffix_source if ch.isalnum())[:12] or "local"
    return f"{timestamp_token[:24]}_{suffix_token}"


def parse_numeric_csv(raw: str, parser, field_name: str):
    cleaned = clean_optional_string(raw)
    if not cleaned:
        return []

    parsed = []
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed.append(parser(token))
        except Exception as exc:
            raise ValueError(f"Invalid token '{token}' in {field_name}: {exc}") from exc
    return parsed


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def split_with_optional_stratification(texts, labels, test_size, random_state, split_name: str):
    stratify_target = labels if len(set(labels)) > 1 else None
    try:
        return train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError as exc:
        if stratify_target is None:
            raise
        print(
            f"[WARN] {split_name}: stratified split failed ({exc}). "
            "Falling back to non-stratified split."
        )
        return train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def clone_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_eval_metrics(y_true, y_pred):
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES)))).tolist()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    return metrics, report, matrix


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += float(loss.item())

            predictions = torch.argmax(outputs.logits, dim=-1)
            y_true.extend(batch["labels"].detach().cpu().numpy().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics, report, matrix = build_eval_metrics(y_true, y_pred)
    metrics["loss"] = float(avg_loss)
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def train_with_validation(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    config: dict,
    device,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_NAMES))
    model.to(device)

    train_dataset = LogDataset(train_texts, train_labels, tokenizer, max_length=config["max_length"])
    val_dataset = LogDataset(val_texts, val_labels, tokenizer, max_length=config["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_state = clone_model_state(model)
    best_epoch = 1
    best_score = float("-inf")
    history = []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        eval_result = evaluate_model(model, val_loader, device)
        val_metrics = eval_result["metrics"]
        val_score = float(val_metrics["weighted_f1"])

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_weighted_precision": float(val_metrics["weighted_precision"]),
            "val_weighted_recall": float(val_metrics["weighted_recall"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(epoch_entry)
        print(
            "Selection epoch "
            f"{epoch + 1}/{config['epochs']} - "
            f"train_loss: {train_loss:.4f} - "
            f"val_loss: {val_metrics['loss']:.4f} - "
            f"val_accuracy: {val_metrics['accuracy']:.4f} - "
            f"val_weighted_f1: {val_metrics['weighted_f1']:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = clone_model_state(model)

    model.load_state_dict(best_state)
    best_eval = evaluate_model(model, val_loader, device)
    print(
        "Best selection metrics - "
        f"epoch: {best_epoch} - "
        f"accuracy: {best_eval['metrics']['accuracy']:.4f} - "
        f"weighted_f1: {best_eval['metrics']['weighted_f1']:.4f} - "
        f"loss: {best_eval['metrics']['loss']:.4f}"
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_eval": best_eval,
    }


def train_on_full_dev(dev_texts, dev_labels, config: dict, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_NAMES))
    model.to(device)

    dataset = LogDataset(dev_texts, dev_labels, tokenizer, max_length=config["max_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    history = []
    final_loss = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())

        epoch_loss = total_loss / max(len(dataloader), 1)
        final_loss = float(epoch_loss)
        history.append({"epoch": epoch + 1, "train_loss": float(epoch_loss)})
        print(f"Final-train epoch {epoch + 1}/{config['epochs']} - loss: {epoch_loss:.4f}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "final_loss": final_loss,
    }


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_candidate_configs(args):
    base_config = {
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "weight_decay": float(args.weight_decay),
        "max_length": int(args.max_length),
    }

    if args.train_mode == "default":
        return [base_config]

    lr_values = unique_preserve_order(
        [float(args.learning_rate)]
        + [value for value in parse_numeric_csv(args.tune_learning_rates, float, "tune_learning_rates") if value > 0]
    )
    batch_values = unique_preserve_order(
        [int(args.batch_size)]
        + [value for value in parse_numeric_csv(args.tune_batch_sizes, int, "tune_batch_sizes") if value > 0]
    )
    epoch_values = unique_preserve_order(
        [int(args.epochs)]
        + [value for value in parse_numeric_csv(args.tune_epochs, int, "tune_epochs") if value > 0]
    )
    wd_values = unique_preserve_order(
        [float(args.weight_decay)]
        + [value for value in parse_numeric_csv(args.tune_weight_decays, float, "tune_weight_decays") if value >= 0]
    )
    max_len_values = unique_preserve_order(
        [int(args.max_length)]
        + [value for value in parse_numeric_csv(args.tune_max_lengths, int, "tune_max_lengths") if value >= 16]
    )

    full_grid = []
    for lr, batch, epochs, wd, max_len in itertools.product(
        lr_values, batch_values, epoch_values, wd_values, max_len_values
    ):
        full_grid.append(
            {
                "learning_rate": float(lr),
                "batch_size": int(batch),
                "epochs": int(epochs),
                "weight_decay": float(wd),
                "max_length": int(max_len),
            }
        )

    deduped_configs = {}
    for cfg in full_grid:
        key = (
            cfg["learning_rate"],
            cfg["batch_size"],
            cfg["epochs"],
            cfg["weight_decay"],
            cfg["max_length"],
        )
        if key not in deduped_configs:
            deduped_configs[key] = cfg
    full_grid = list(deduped_configs.values())
    rng = random.Random(int(args.seed))
    rng.shuffle(full_grid)
    trial_count = min(max(int(args.max_trials), 1), len(full_grid))
    return full_grid[:trial_count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the labeled training CSV file")
    parser.add_argument("--train-mode", choices=["default", "tune", "tune_cv"], default="default")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-learning-rates", type=str, default="5e-5,3e-5,1e-4")
    parser.add_argument("--tune-batch-sizes", type=str, default="8,16")
    parser.add_argument("--tune-epochs", type=str, default="3,4")
    parser.add_argument("--tune-weight-decays", type=str, default="0.0,0.01")
    parser.add_argument("--tune-max-lengths", type=str, default="128")
    args = parser.parse_args()

    if not (0.0 < args.test_ratio < 0.5):
        raise ValueError("--test-ratio must be between 0 and 0.5.")
    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("--val-ratio must be between 0 and 0.5.")
    if args.test_ratio + args.val_ratio >= 0.9:
        raise ValueError("--val-ratio + --test-ratio is too large; keep enough data for training.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")
    if args.max_trials < 1:
        raise ValueError("--max-trials must be >= 1.")

    set_global_seed(int(args.seed))
    mlflow_context = build_mlflow_context()
    mlflow_enabled = bool(mlflow_context.get("enabled"))
    active_run = None
    run_id = ""

    # 1) Load + validate dataset
    df = pd.read_csv(args.data)
    input_data_hash = compute_file_sha256(args.data)
    df["label"] = df["class"].map(LABEL_MAPPING)
    df = df.dropna(subset=["LogMessage", "label"]).copy()
    if df.empty:
        raise ValueError("Training data contains no valid rows after label mapping.")
    df["label"] = df["label"].astype(int)
    df["LogMessage"] = df["LogMessage"].astype(str)

    dataset_meta = dataframe_metadata(df, label_col="class")
    texts = df["LogMessage"].tolist()
    labels = df["label"].tolist()

    dev_texts, test_texts, dev_labels, test_labels = split_with_optional_stratification(
        texts, labels, test_size=args.test_ratio, random_state=int(args.seed), split_name="dev/test"
    )
    if len(dev_texts) < 2:
        raise ValueError("Development split is too small after test split.")
    if len(test_texts) < 1:
        raise ValueError("Test split is empty after splitting.")

    split_metadata = {
        "total_rows": len(texts),
        "dev_rows": len(dev_texts),
        "test_rows": len(test_texts),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "train_mode": args.train_mode,
        "cv_folds": int(args.cv_folds),
        "max_trials": int(args.max_trials),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_device = str(device)
    runtime_mode = (
        "container" if "container" in clean_optional_string(mlflow_context.get("run_source", "")) else "host"
    )

    # 2) MLflow run initialization
    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(mlflow_context["tracking_uri"])
            mlflow.set_experiment(mlflow_context["experiment_name"])
            active_run = mlflow.start_run(run_name="training")
            run_id = active_run.info.run_id

            if clean_optional_string(mlflow_context.get("parent_run_id")):
                mlflow.set_tag("mlflow.parentRunId", mlflow_context["parent_run_id"])

            base_tags = {
                "run_type": "training",
                "pipeline_id": clean_optional_string(mlflow_context.get("pipeline_id")),
                "run_source": clean_optional_string(mlflow_context.get("run_source", "unknown")),
                "train_mode": args.train_mode,
            }
            base_tags.update(mlflow_context.get("tags", {}))
            safe_mlflow_call(mlflow_enabled, lambda: mlflow.set_tags(base_tags), "set base tags")
        except Exception as exc:
            print(f"[MLOPS] Failed to initialize MLflow run; proceeding without MLflow. Details: {exc}")
            mlflow_enabled = False
            active_run = None
            run_id = ""

    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("model_name", MODEL_NAME), "log model_name")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("train_mode", args.train_mode), "log train_mode")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("cv_folds", int(args.cv_folds)), "log cv_folds")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("max_trials", int(args.max_trials)), "log max_trials")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("resolved_device", resolved_device), "log resolved_device")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("runtime_mode", runtime_mode), "log runtime_mode")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_param("input_dataset_hash", input_data_hash), "log input_dataset_hash")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_dict(dataset_meta, "dataset_metadata.json"), "log dataset metadata")
    safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_dict(split_metadata, "split_metadata.json"), "log split metadata")

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_path = os.path.join(tmp_dir, "training_data_sample.csv")
        sample_df = dataframe_sample(df, max_rows=100, max_cell_chars=200)
        sample_df.to_csv(sample_path, index=False)
        safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_artifact(sample_path, "dataset"), "log dataset sample")

    # 3) Selection phase (default / tune / tune_cv)
    candidate_configs = build_candidate_configs(args)
    trial_summaries = []
    best_config = None
    best_score = float("-inf")
    best_selection_metrics = {}
    best_selection_report = {}
    best_selection_confusion = []

    if args.train_mode in {"default", "tune"}:
        val_ratio_within_dev = args.val_ratio / (1.0 - args.test_ratio)
        train_texts, val_texts, train_labels, val_labels = split_with_optional_stratification(
            dev_texts,
            dev_labels,
            test_size=val_ratio_within_dev,
            random_state=int(args.seed),
            split_name="train/validation",
        )
        if len(train_texts) < 1 or len(val_texts) < 1:
            raise ValueError("Train/validation split produced an empty subset; adjust ratios or provide more data.")

        for trial_index, config in enumerate(candidate_configs, start=1):
            trial_run_ctx = mlflow.start_run(run_name=f"trial-{trial_index}", nested=True) if mlflow_enabled else nullcontext()
            with trial_run_ctx:
                result = train_with_validation(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    config=config,
                    device=device,
                )
                trial_metrics = result["best_eval"]["metrics"]
                trial_score = float(trial_metrics["weighted_f1"])

                summary = {
                    "trial_index": trial_index,
                    "config": config,
                    "best_epoch": int(result["best_epoch"]),
                    "score_weighted_f1": trial_score,
                    "val_metrics": trial_metrics,
                }
                trial_summaries.append(summary)

                if trial_score > best_score:
                    best_score = trial_score
                    best_config = dict(config)
                    best_selection_metrics = dict(trial_metrics)
                    best_selection_report = result["best_eval"]["classification_report"]
                    best_selection_confusion = result["best_eval"]["confusion_matrix"]

                safe_mlflow_call(mlflow_enabled, lambda cfg=config: mlflow.log_params(cfg), "log trial params")
                safe_mlflow_call(
                    mlflow_enabled,
                    lambda m=trial_metrics: mlflow.log_metrics({f"val_{key}": float(value) for key, value in m.items()}),
                    "log trial val metrics",
                )
                safe_mlflow_call(
                    mlflow_enabled,
                    lambda s=trial_score: mlflow.log_metric("selection_score_weighted_f1", float(s)),
                    "log trial selection score",
                )
                safe_mlflow_call(
                    mlflow_enabled,
                    lambda e=result["best_epoch"]: mlflow.log_metric("best_epoch", int(e)),
                    "log trial best epoch",
                )
    else:
        class_counts = pd.Series(dev_labels).value_counts()
        if int(class_counts.min()) < int(args.cv_folds):
            raise ValueError(
                f"Not enough examples per class in dev split for cv_folds={args.cv_folds}. "
                "Reduce CV folds or provide more labeled data."
            )

        splitter = StratifiedKFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        dev_texts_np = np.array(dev_texts, dtype=object)
        dev_labels_np = np.array(dev_labels, dtype=int)

        for trial_index, config in enumerate(candidate_configs, start=1):
            trial_run_ctx = mlflow.start_run(run_name=f"trial-{trial_index}", nested=True) if mlflow_enabled else nullcontext()
            with trial_run_ctx:
                fold_scores = []
                fold_metrics = []
                fold_reports = {}
                fold_confusions = {}
                for fold_index, (train_idx, val_idx) in enumerate(splitter.split(dev_texts_np, dev_labels_np), start=1):
                    result = train_with_validation(
                        train_texts=dev_texts_np[train_idx].tolist(),
                        train_labels=dev_labels_np[train_idx].tolist(),
                        val_texts=dev_texts_np[val_idx].tolist(),
                        val_labels=dev_labels_np[val_idx].tolist(),
                        config=config,
                        device=device,
                    )
                    metrics = result["best_eval"]["metrics"]
                    fold_score = float(metrics["weighted_f1"])
                    fold_scores.append(fold_score)
                    fold_metrics.append(metrics)
                    fold_reports[f"fold_{fold_index}"] = result["best_eval"]["classification_report"]
                    fold_confusions[f"fold_{fold_index}"] = result["best_eval"]["confusion_matrix"]

                    safe_mlflow_call(
                        mlflow_enabled,
                        lambda idx=fold_index, score=fold_score: mlflow.log_metric(f"fold_{idx}_val_weighted_f1", score),
                        "log fold weighted_f1",
                    )

                avg_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
                avg_metrics = {
                    key: float(np.mean([entry[key] for entry in fold_metrics]))
                    for key in fold_metrics[0].keys()
                }

                summary = {
                    "trial_index": trial_index,
                    "config": config,
                    "score_weighted_f1": avg_score,
                    "cv_avg_metrics": avg_metrics,
                    "fold_weighted_f1": fold_scores,
                }
                trial_summaries.append(summary)

                if avg_score > best_score:
                    best_score = avg_score
                    best_config = dict(config)
                    best_selection_metrics = dict(avg_metrics)
                    best_selection_report = fold_reports
                    best_selection_confusion = fold_confusions

                safe_mlflow_call(mlflow_enabled, lambda cfg=config: mlflow.log_params(cfg), "log trial params")
                safe_mlflow_call(
                    mlflow_enabled,
                    lambda metrics=avg_metrics: mlflow.log_metrics(
                        {f"cv_avg_{key}": float(value) for key, value in metrics.items()}
                    ),
                    "log cv avg metrics",
                )
                safe_mlflow_call(
                    mlflow_enabled,
                    lambda s=avg_score: mlflow.log_metric("selection_score_weighted_f1", float(s)),
                    "log cv selection score",
                )

    if best_config is None:
        raise RuntimeError("No valid training trial produced a selectable configuration.")

    selection_summary = {
        "train_mode": args.train_mode,
        "best_score_weighted_f1": float(best_score),
        "best_config": best_config,
        "best_selection_metrics": best_selection_metrics,
        "trial_count": len(trial_summaries),
        "trials": trial_summaries,
    }

    # 4) Final training on full dev + test evaluation
    final_train_result = train_on_full_dev(
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        config=best_config,
        device=device,
    )
    model = final_train_result["model"]
    tokenizer = final_train_result["tokenizer"]

    for epoch_entry in final_train_result["history"]:
        epoch_idx = int(epoch_entry["epoch"])
        loss_value = float(epoch_entry["train_loss"])
        safe_mlflow_call(
            mlflow_enabled,
            lambda e=epoch_idx, l=loss_value: mlflow.log_metric("train_loss", l, step=e),
            "log final train_loss",
        )

    test_dataset = LogDataset(test_texts, test_labels, tokenizer, max_length=int(best_config["max_length"]))
    test_loader = DataLoader(test_dataset, batch_size=int(best_config["batch_size"]), shuffle=False)
    test_eval = evaluate_model(model, test_loader, device)
    test_metrics = test_eval["metrics"]
    test_report = test_eval["classification_report"]
    test_confusion = test_eval["confusion_matrix"]
    print(
        "Test metrics - "
        f"accuracy: {test_metrics['accuracy']:.4f} - "
        f"weighted_precision: {test_metrics['weighted_precision']:.4f} - "
        f"weighted_recall: {test_metrics['weighted_recall']:.4f} - "
        f"weighted_f1: {test_metrics['weighted_f1']:.4f} - "
        f"loss: {test_metrics['loss']:.4f}"
    )

    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_metrics({f"test_{key}": float(value) for key, value in test_metrics.items()}),
        "log test metrics",
    )
    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_metric("selection_best_weighted_f1", float(best_score)),
        "log best selection score",
    )
    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_metric("final_loss", float(final_train_result["final_loss"])),
        "log final_loss",
    )
    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_params({f"selected_{key}": value for key, value in best_config.items()}),
        "log selected config params",
    )

    # 5) Save artifacts
    output_root = "./outputs"
    output_dir = os.path.join(output_root, "final_model")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        selection_path = os.path.join(tmp_dir, "selection_summary.json")
        write_json(selection_path, selection_summary)
        test_report_path = os.path.join(tmp_dir, "test_classification_report.json")
        write_json(test_report_path, test_report)
        test_confusion_path = os.path.join(tmp_dir, "test_confusion_matrix.json")
        write_json(test_confusion_path, {"labels": LABEL_NAMES, "matrix": test_confusion})

        selection_report_path = os.path.join(tmp_dir, "selection_report.json")
        write_json(selection_report_path, best_selection_report)
        selection_confusion_path = os.path.join(tmp_dir, "selection_confusion.json")
        write_json(selection_confusion_path, best_selection_confusion)

        safe_mlflow_call(mlflow_enabled, lambda: mlflow.log_artifacts(tmp_dir, artifact_path="evaluation"), "log evaluation artifacts")

    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_artifacts(output_dir, artifact_path="final_model"),
        "log final_model artifacts",
    )

    model_uri = f"runs:/{run_id}/final_model" if run_id else ""
    lineage_tags = mlflow_context.get("tags", {})
    created_at = now_utc_iso()
    model_version_id = build_model_version_id(created_at, run_id, input_data_hash)
    model_versions_root = os.path.join(output_root, "model_versions")
    version_root = os.path.join(model_versions_root, model_version_id)
    while os.path.exists(version_root):
        model_version_id = f"{model_version_id}_{random.randint(1000, 9999)}"
        version_root = os.path.join(model_versions_root, model_version_id)
    archived_model_dir = os.path.join(version_root, "final_model")

    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_param("model_version_id", model_version_id),
        "log model_version_id",
    )
    if clean_optional_string(lineage_tags.get("data_version_id", "")):
        safe_mlflow_call(
            mlflow_enabled,
            lambda: mlflow.log_param("data_version_id", clean_optional_string(lineage_tags.get("data_version_id", ""))),
            "log data_version_id",
        )

    metadata_payload = {
        "run_id": run_id,
        "tracking_uri": clean_optional_string(mlflow_context.get("tracking_uri", "")),
        "experiment_name": clean_optional_string(mlflow_context.get("experiment_name", "")),
        "model_uri": model_uri,
        "pipeline_id": clean_optional_string(mlflow_context.get("pipeline_id", "")),
        "parent_run_id": clean_optional_string(mlflow_context.get("parent_run_id", "")),
        "run_source": clean_optional_string(mlflow_context.get("run_source", "")),
        "resolved_device": resolved_device,
        "runtime_mode": runtime_mode,
        "input_dataset_hash": input_data_hash,
        "data_prep_run_id": clean_optional_string(lineage_tags.get("data_prep_run_id", "")),
        "data_prep_tracking_uri": clean_optional_string(lineage_tags.get("data_prep_tracking_uri", "")),
        "data_prep_experiment_name": clean_optional_string(lineage_tags.get("data_prep_experiment_name", "")),
        "data_prep_input_dataset_hash": clean_optional_string(lineage_tags.get("data_prep_input_dataset_hash", "")),
        "data_prep_output_dataset_hash": clean_optional_string(lineage_tags.get("data_prep_output_dataset_hash", "")),
        "data_version_id": clean_optional_string(lineage_tags.get("data_version_id", "")),
        "data_version_dir": clean_optional_string(lineage_tags.get("data_version_dir", "")),
        "data_version_path": clean_optional_string(lineage_tags.get("data_version_path", "")),
        "prompt_hash": clean_optional_string(lineage_tags.get("prompt_hash", "")),
        "llm_model": clean_optional_string(lineage_tags.get("llm_model", "")),
        "dataset_metadata": dataset_meta,
        "split_metadata": split_metadata,
        "selection_summary": selection_summary,
        "test_metrics": test_metrics,
        "created_at": created_at,
        "model_version_id": model_version_id,
        "model_version_dir": os.path.abspath(version_root),
        "model_version_model_dir": os.path.abspath(archived_model_dir),
    }

    os.makedirs(output_root, exist_ok=True)
    top_level_metadata_path = os.path.join(output_root, "last_training_mlflow.json")
    write_json(top_level_metadata_path, metadata_payload)
    shutil.copy2(top_level_metadata_path, os.path.join(output_dir, "last_training_mlflow.json"))
    os.makedirs(version_root, exist_ok=True)
    shutil.copy2(top_level_metadata_path, os.path.join(version_root, "last_training_mlflow.json"))
    shutil.copytree(output_dir, archived_model_dir)
    safe_mlflow_call(
        mlflow_enabled,
        lambda: mlflow.log_dict(metadata_payload, "training_run_metadata.json"),
        "log training_run_metadata",
    )

    if mlflow_enabled and active_run is not None:
        try:
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
