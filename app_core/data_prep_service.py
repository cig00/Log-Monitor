import json
import time
from typing import Any

import pandas as pd
import requests

from mlops_utils import (
    clean_optional_string,
    compute_file_sha256,
    now_utc_iso,
    prompt_sha256,
    write_sidecar_for_csv,
)

from .contracts import DataPrepRequest
from .mlops_service import MlopsService
from .model_catalog_service import ModelCatalogService
from .runtime import JobContext, JobManager


class DataPrepService:
    def __init__(self, job_manager: JobManager, mlops_service: MlopsService, model_catalog_service: ModelCatalogService):
        self.job_manager = job_manager
        self.mlops_service = mlops_service
        self.model_catalog_service = model_catalog_service

    def submit_data_prep(self, request: DataPrepRequest):
        return self.job_manager.submit(
            "data_prep",
            lambda ctx: self._run(ctx, request),
            metadata={"operation": "data_prep", "input_path": request.input_path, "output_path": request.output_path},
        )

    def _run(self, ctx: JobContext, request: DataPrepRequest) -> dict[str, Any]:
        processed_logs = []
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        system_prompt = self.mlops_service.load_prompt()
        df = pd.read_csv(request.input_path)
        input_file_hash = compute_file_sha256(request.input_path)

        log_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["log", "message", "msg", "text"]):
                log_col = col
                break
        if not log_col:
            log_col = df.columns[0]

        total_rows = len(df)
        batch_size = 10
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {request.api_key}", "Content-Type": "application/json"}

        for i in range(0, total_rows, batch_size):
            ctx.check_cancelled()
            batch_df = df.iloc[i : i + batch_size]
            ctx.emit("progress", f"Processing rows {i + 1} to {min(i + batch_size, total_rows)} of {total_rows}...", percent=((i / max(total_rows, 1)) * 100.0))
            logs_batch = batch_df[log_col].astype(str).tolist()
            payload = {
                "model": request.model_name,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(logs_batch)}],
                "response_format": {"type": "json_object"},
            }
            success = False
            for attempt in range(5):
                ctx.check_cancelled()
                response = requests.post(api_url, json=payload, headers=headers, timeout=120)
                if response.status_code == 429:
                    wait_time = 5 * (2**attempt)
                    ctx.emit("rate_limit", f"Rate limit hit (429). Pausing for {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                success = True
                break
            if not success:
                raise RuntimeError("Max retries reached. API rejecting requests due to rate limits.")

            result_json = response.json()
            usage = result_json.get("usage", {}) if isinstance(result_json, dict) else {}
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key, 0) or 0)
            llm_content = result_json["choices"][0]["message"]["content"]
            try:
                parsed_data = json.loads(llm_content)
                results_array = parsed_data.get("results", [])
                for idx, log_text in enumerate(logs_batch):
                    log_class = results_array[idx].get("class", "Noise") if idx < len(results_array) else "Noise"
                    processed_logs.append({"LogMessage": log_text, "class": log_class})
            except json.JSONDecodeError:
                for log_text in logs_batch:
                    processed_logs.append({"LogMessage": log_text, "class": "Noise"})
            time.sleep(1.5)

        output_df = pd.DataFrame(processed_logs)
        output_df.to_csv(request.output_path, index=False)
        output_file_hash = compute_file_sha256(request.output_path)
        data_version_info = self.model_catalog_service.archive_data_version(
            request.output_path,
            {
                "llm_model": clean_optional_string(request.model_name),
                "prompt_hash": prompt_sha256(system_prompt),
                "input_dataset_hash": input_file_hash,
                "output_dataset_hash": output_file_hash,
                "source_input_path": request.input_path,
                "generated_output_path": request.output_path,
            },
        )
        mlops_lineage = self.mlops_service.log_data_prep_mlflow(
            mlflow_config=request.mlflow_config,
            input_path=request.input_path,
            output_path=request.output_path,
            system_prompt=system_prompt,
            llm_model=request.model_name,
            input_df=df,
            output_df=output_df,
            input_hash=input_file_hash,
            output_hash=output_file_hash,
            usage_totals=usage_totals,
            data_version_info=data_version_info,
        )
        sidecar_payload = {
            "pipeline_id": mlops_lineage.get("pipeline_id", ""),
            "parent_run_id": mlops_lineage.get("parent_run_id", ""),
            "data_prep_run_id": mlops_lineage.get("data_prep_run_id", ""),
            "prompt_hash": mlops_lineage.get("prompt_hash", ""),
            "llm_model": mlops_lineage.get("llm_model", ""),
            "input_dataset_hash": mlops_lineage.get("input_dataset_hash", ""),
            "output_dataset_hash": mlops_lineage.get("output_dataset_hash", ""),
            "created_at": mlops_lineage.get("created_at", now_utc_iso()),
            "tracking_uri": mlops_lineage.get("tracking_uri", ""),
            "experiment_name": mlops_lineage.get("experiment_name", ""),
            "data_prep_tracking_uri": mlops_lineage.get("data_prep_tracking_uri", ""),
            "data_prep_experiment_name": mlops_lineage.get("data_prep_experiment_name", ""),
            "data_version_id": data_version_info.get("data_version_id", ""),
            "data_version_dir": data_version_info.get("data_version_dir", ""),
            "data_version_path": data_version_info.get("data_version_path", ""),
        }
        write_sidecar_for_csv(request.output_path, sidecar_payload)
        return {
            "operation": "data_prep",
            "message": f"Data processed successfully! Saved to {request.output_path}",
            "output_path": request.output_path,
            "row_count": len(output_df),
        }
