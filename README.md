# Log-Monitor

Log-Monitor is a desktop-assisted workflow for log classification:

- `app.py` provides a Tkinter UI to prepare labeled log data and launch Azure training jobs.
- `train.py` fine-tunes DeBERTa and now includes MLflow-based MLOps tracking.

## MLflow MLOps (Implemented)

Every training run now tracks model lineage and performance through MLflow:

- Dataset lineage:
  - input CSV SHA-256 fingerprint
  - row counts before/after filtering
  - dropped row count
  - class distribution
  - optional raw dataset artifact upload
- Run configuration:
  - base model name
  - hyperparameters (epochs, batch size, LR, max length, split ratio, seed)
- Metrics:
  - train metrics from Hugging Face Trainer
  - eval metrics when split is enabled (`accuracy`, weighted `precision`, `recall`, `f1`)
- Artifacts:
  - final Hugging Face model directory (`outputs/final_model`)
  - MLflow model artifact (`model`) for reproducible versioning
- Registry:
  - optional automatic model registration (`log-monitor-deberta-classifier`)
  - registered versions tagged with the dataset hash

## Local Training With MLflow

Install dependencies (example):

```bash
pip install pandas datasets transformers torch scikit-learn mlflow
```

Run training:

```bash
python train.py --data /absolute/path/to/processed_logs.csv
```

Optional useful flags:

```bash
python train.py \
  --data /absolute/path/to/processed_logs.csv \
  --experiment-name deberta-log-classification \
  --run-name local-baseline-001 \
  --register-model \
  --registry-model-name log-monitor-deberta-classifier
```

If `--tracking-uri` is not provided and `MLFLOW_TRACKING_URI` is unset, runs are stored locally in `./outputs/mlruns`.

Launch MLflow UI locally:

```bash
mlflow ui --backend-store-uri file:./outputs/mlruns
```

## Azure Training Path

When started from `app.py`, Azure jobs now pass MLflow metadata automatically:

- experiment name: `deberta-log-classification`
- run name: generated from selected repo + branch + timestamp
- model registration: enabled by default to `log-monitor-deberta-classifier`

This gives consistent tracking of trained model versions and training lineage over time.
