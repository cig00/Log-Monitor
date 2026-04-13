import argparse
import hashlib
import json
import os
from numbers import Real
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

LABEL_MAPPING = {"Error": 0, "CONFIGURATION": 1, "SYSTEM": 2, "Noise": 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--experiment-name", type=str, default="deberta-log-classification")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--registry-model-name", type=str, default="log-monitor-deberta-classifier")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-split-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-data-artifact",
        action="store_true",
        help="Skip uploading the input CSV as an MLflow artifact.",
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_tracking_uri(explicit_tracking_uri, output_dir):
    if explicit_tracking_uri:
        return explicit_tracking_uri
    env_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_tracking_uri:
        return env_tracking_uri
    local_store = os.path.abspath(os.path.join(output_dir, "mlruns"))
    os.makedirs(local_store, exist_ok=True)
    return "file:" + local_store


def log_json_artifact(data, artifact_file):
    if hasattr(mlflow, "log_dict"):
        mlflow.log_dict(data, artifact_file)
        return
    temp_path = os.path.join("/tmp", os.path.basename(artifact_file))
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    mlflow.log_artifact(temp_path, artifact_path=os.path.dirname(artifact_file))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def flatten_numeric_metrics(metrics_dict, prefix):
    flattened = {}
    for key, value in metrics_dict.items():
        if isinstance(value, Real):
            flattened["{}/{}".format(prefix, key)] = float(value)
    return flattened


def register_model_version(registry_model_name, run_id, model_artifact_path, dataset_hash):
    model_uri = "runs:/{}/{}".format(run_id, model_artifact_path)
    try:
        version = mlflow.register_model(model_uri=model_uri, name=registry_model_name)
    except Exception as exc:
        print("WARNING: model registration failed: {}".format(exc))
        return None

    try:
        client = MlflowClient()
        client.set_model_version_tag(
            name=registry_model_name,
            version=version.version,
            key="dataset_sha256",
            value=dataset_hash,
        )
    except Exception as exc:
        print("WARNING: model version tag update failed: {}".format(exc))
    return version


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tracking_uri = resolve_tracking_uri(args.tracking_uri, args.output_dir)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print("MLflow tracking URI: {}".format(mlflow.get_tracking_uri()))
    print("MLflow experiment: {}".format(args.experiment_name))

    df = pd.read_csv(args.data)
    required_columns = {"LogMessage", "class"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: {}".format(", ".join(sorted(missing_columns)))
        )

    original_row_count = len(df)
    df["label"] = df["class"].map(LABEL_MAPPING)
    df = df.dropna(subset=["LogMessage", "label"]).copy()
    filtered_row_count = len(df)

    if filtered_row_count == 0:
        raise ValueError("No valid rows found after mapping labels from the 'class' column.")

    df["label"] = df["label"].astype(int)

    dataset_hash = file_sha256(args.data)
    dataset_profile = {
        "data_path": os.path.abspath(args.data),
        "data_sha256": dataset_hash,
        "rows_total": int(original_row_count),
        "rows_after_filter": int(filtered_row_count),
        "rows_dropped": int(original_row_count - filtered_row_count),
        "columns": list(df.columns),
        "label_mapping": LABEL_MAPPING,
        "class_distribution": {k: int(v) for k, v in df["class"].value_counts().to_dict().items()},
        "utc_created": datetime.now(timezone.utc).isoformat(),
    }

    dataset = Dataset.from_pandas(df[["LogMessage", "label"]], preserve_index=False)

    use_eval_split = len(dataset) >= 5 and 0 < args.eval_split_ratio < 0.5
    if use_eval_split:
        split_dataset = dataset.train_test_split(test_size=args.eval_split_ratio, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["LogMessage"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True) if eval_dataset else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABEL_MAPPING)
    )

    run_name = args.run_name if args.run_name else "deberta-{}".format(int(datetime.now().timestamp()))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "Log-Monitor",
                "task": "log-classification",
                "framework": "transformers",
                "base_model": args.model_name,
                "dataset_sha256": dataset_hash,
            }
        )

        mlflow.log_params(
            {
                "data_path": os.path.abspath(args.data),
                "model_name": args.model_name,
                "num_labels": len(LABEL_MAPPING),
                "num_train_epochs": args.num_train_epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "eval_split_ratio": args.eval_split_ratio if use_eval_split else 0.0,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_length": args.max_length,
                "seed": args.seed,
                "training_rows": len(tokenized_train),
                "evaluation_rows": len(tokenized_eval) if tokenized_eval else 0,
            }
        )
        log_json_artifact(dataset_profile, "lineage/dataset_profile.json")
        log_json_artifact(LABEL_MAPPING, "lineage/label_mapping.json")

        if not args.skip_data_artifact:
            mlflow.log_artifact(args.data, artifact_path="lineage/input_data")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_total_limit=1,
            evaluation_strategy="epoch" if tokenized_eval else "no",
            save_strategy="epoch" if tokenized_eval else "no",
            logging_strategy="epoch",
            load_best_model_at_end=bool(tokenized_eval),
            metric_for_best_model="f1_weighted" if tokenized_eval else None,
            greater_is_better=True if tokenized_eval else None,
            report_to="none",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=compute_metrics if tokenized_eval else None,
        )

        print("Starting DeBERTa training...")
        train_result = trainer.train()
        train_metrics = flatten_numeric_metrics(train_result.metrics, "train")
        if train_metrics:
            mlflow.log_metrics(train_metrics)

        if tokenized_eval:
            eval_metrics = trainer.evaluate()
            mlflow.log_metrics(flatten_numeric_metrics(eval_metrics, "eval"))

        final_model_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        mlflow.log_artifacts(final_model_dir, artifact_path="final_model_files")

        model_artifact_path = "model"
        model_logged = False
        try:
            mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                artifact_path=model_artifact_path,
                task="text-classification",
            )
            model_logged = True
        except Exception as exc:
            print("WARNING: transformers flavor logging failed: {}".format(exc))
            try:
                mlflow.pytorch.log_model(trainer.model, artifact_path=model_artifact_path)
                model_logged = True
                print("Fallback to PyTorch model logging succeeded.")
            except Exception as fallback_exc:
                print("WARNING: PyTorch model logging also failed: {}".format(fallback_exc))

        if args.register_model and model_logged:
            version = register_model_version(
                registry_model_name=args.registry_model_name,
                run_id=run.info.run_id,
                model_artifact_path=model_artifact_path,
                dataset_hash=dataset_hash,
            )
            if version:
                mlflow.set_tag("registered_model_name", args.registry_model_name)
                mlflow.set_tag("registered_model_version", str(version.version))
                print(
                    "Registered model '{}' version {}.".format(
                        args.registry_model_name, version.version
                    )
                )

        print("MLflow run completed: {}".format(run.info.run_id))
        print("Training complete. Model saved to {}".format(final_model_dir))


if __name__ == "__main__":
    main()
