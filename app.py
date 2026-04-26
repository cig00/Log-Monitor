from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
import traceback
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app_core.azure_platform_service import AZURE_AVAILABLE, AzurePlatformService
from app_core.contracts import DataPrepRequest, MlflowConfig, TrainingRequest, HostingRequest
from app_core.data_prep_service import DataPrepService
from app_core.github_service import GitHubService
from app_core.hosting_service import HostingService
from app_core.mlops_service import MlopsService
from app_core.model_catalog_service import ModelCatalogService
from app_core.observability_service import ObservabilityService
from app_core.runtime import ArtifactStore, JobManager, StateStore
from app_core.training_service import TrainingService

from mlops_utils import (
    clean_optional_string,
    discover_model_dir,
    local_mlflow_tracking_uri,
)

class LogProcessorApp:
    RESOURCE_GROUP = "LogClassifier-RG"
    WORKSPACE_NAME = "LogClassifier-Workspace"
    OPENAI_LABEL_MODELS = (
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5.2",
    )
    DEFAULT_PROMPT_TEST_CASES = [
        {
            "name": "Missing Environment Variable",
            "message": "Error: application startup failed because MAIL_HOST environment variable is missing",
            "expected": "CONFIGURATION",
        },
        {
            "name": "Disk Space",
            "message": "kernel: write failed on /var/log/app.log, no space left on device",
            "expected": "SYSTEM",
        },
        {
            "name": "Code Exception",
            "message": "Traceback: TypeError unsupported operand type for +: 'NoneType' and 'str'",
            "expected": "Error",
        },
        {
            "name": "Canceled Work",
            "message": "processed Canceled after user interrupted the background task",
            "expected": "Noise",
        },
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Log Classifier & DeBERTa Trainer")
        self.root.geometry("650x700")
        self.root.minsize(650, 620)
        self.root.resizable(True, True)
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_mlflow_tracking_uri = local_mlflow_tracking_uri()
        self.artifact_store = ArtifactStore(self.project_dir)
        self.state_store = StateStore(self.artifact_store.state_db_path)
        self.job_manager = JobManager(self.state_store)
        self.github_service = GitHubService()
        self.model_catalog_service = ModelCatalogService(self.project_dir, self.artifact_store)
        self.mlops_service = MlopsService(
            self.project_dir,
            self.artifact_store,
            self.model_catalog_service,
            resource_group=self.RESOURCE_GROUP,
            workspace_name=self.WORKSPACE_NAME,
            local_tracking_uri=self.local_mlflow_tracking_uri,
        )
        self.azure_platform_service = AzurePlatformService(
            self.project_dir,
            resource_group=self.RESOURCE_GROUP,
            workspace_name=self.WORKSPACE_NAME,
        )
        self.observability_service = ObservabilityService(
            self.project_dir,
            self.artifact_store,
            self.state_store,
        )
        self.data_prep_service = DataPrepService(self.job_manager, self.mlops_service, self.model_catalog_service)
        self.training_service = TrainingService(
            self.project_dir,
            self.job_manager,
            self.model_catalog_service,
            self.mlops_service,
            self.azure_platform_service,
        )
        self.hosting_service = HostingService(
            self.project_dir,
            self.job_manager,
            self.model_catalog_service,
            self.mlops_service,
            self.azure_platform_service,
            self.observability_service,
            self.github_service,
        )
        self.current_data_prep_job_id = ""
        self.current_training_job_id = ""
        self.current_hosting_job_id = ""
        self.current_repo_job_id = ""
        self.current_branch_job_id = ""
        self.current_prompt_test_job_id = ""
        self.training_state_lock = threading.Lock()
        self.training_active = False
        self.training_config_visible = False
        self.hosting_active = False
        self.hosting_state_lock = threading.Lock()
        self.hosting_process = None
        self.prometheus_process = None
        self.grafana_process = None
        self.hosting_mode_var = tk.StringVar(value="local")
        default_model_dir = Path(self.project_dir) / "outputs" / "final_model"
        self.hosted_model_path_var = tk.StringVar(value=str(default_model_dir) if default_model_dir.exists() else "")
        self.hosted_model_inventory = []
        self.available_model_choice_var = tk.StringVar(value="")
        self.hosting_api_url_var = tk.StringVar(value="")
        self.hosting_mode_summary_var = tk.StringVar(value="")
        self.azure_host_sub_var = tk.StringVar(value="")
        self.azure_host_tenant_var = tk.StringVar(value="")
        self.azure_host_compute_var = tk.StringVar(value="cpu")
        self.azure_host_service_var = tk.StringVar(value="queued_batch")
        default_serverless_model_id = self.azure_platform_service.get_default_serverless_model_id()
        default_serverless_endpoint_name = self.azure_platform_service.build_default_serverless_endpoint_name(
            default_serverless_model_id,
            suffix=str(int(time.time())),
        )
        self.azure_serverless_endpoint_name_is_auto = True
        self.azure_serverless_endpoint_name_setting = False
        self.azure_serverless_last_auto_endpoint_name = default_serverless_endpoint_name
        self.azure_serverless_model_id_var = tk.StringVar(value=default_serverless_model_id)
        self.azure_serverless_endpoint_name_var = tk.StringVar(value=default_serverless_endpoint_name)
        self.azure_batch_input_var = tk.StringVar(value="")
        self.azure_batch_time_var = tk.StringVar(value="02:00")
        self.azure_batch_timezone_var = tk.StringVar(value="UTC")
        self.azure_training_instance_var = tk.StringVar(value="Standard_D2as_v4")
        self.azure_host_instance_var = tk.StringVar(value="Standard_D2as_v4")
        self.azure_mlops_url_var = tk.StringVar(value="")
        self.azure_llmops_url_var = tk.StringVar(value="")
        self.azure_hosted_endpoint_name_var = tk.StringVar(value="")
        self.create_pr_var = tk.BooleanVar(value=False)
        self.github_pr_url_var = tk.StringVar(value="")
        self.mlflow_enabled_var = tk.BooleanVar(value=False)
        self.mlflow_backend_var = tk.StringVar(value="local")
        self.mlflow_tracking_uri_var = tk.StringVar(value=self.local_mlflow_tracking_uri)
        self.mlflow_experiment_var = tk.StringVar(value="")
        self.mlflow_registered_model_var = tk.StringVar(value="")
        self.azure_host_sub_var.trace_add("write", lambda *_: self.refresh_azure_dashboard_links())
        self.azure_serverless_endpoint_name_var.trace_add("write", self.on_azure_serverless_endpoint_name_changed)
        self.prompt_test_window = None
        self.prompt_test_tree = None
        self.prompt_test_run_btn = None
        self.prompt_test_cases = []
        self.prompt_test_row_ids = []
        self.prompt_version_choice_var = tk.StringVar(value="default")
        self.prompt_version_choices: list[dict[str, str]] = []

        # UI Styling
        style = ttk.Style()
        style.theme_use('clam')

        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True)

        self.main_canvas = tk.Canvas(self.main_container, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(
            self.main_container,
            orient="vertical",
            command=self.main_canvas.yview,
        )
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar.pack(side="right", fill="y")

        self.content_frame = ttk.Frame(self.main_canvas)
        self.content_window_id = self.main_canvas.create_window(
            (0, 0),
            window=self.content_frame,
            anchor="nw",
        )
        self.content_frame.bind("<Configure>", self._on_content_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_canvas_configure)
        self.main_canvas.bind("<Enter>", self._bind_mousewheel)
        self.main_canvas.bind("<Leave>", self._unbind_mousewheel)

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_app_close)
        self.root.after(250, self.poll_job_events)

    def create_widgets(self):
        # --- File Upload Section ---
        file_frame = ttk.LabelFrame(self.content_frame, text="Data Processing", padding=(10, 10))
        file_frame.pack(fill="x", padx=10, pady=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Log File (CSV):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.filepath_entry = ttk.Entry(file_frame, width=40)
        self.filepath_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_btn.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(file_frame, text="OpenAI API Key:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.openai_api_key_entry = ttk.Entry(file_frame, width=40, show="*")
        self.openai_api_key_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        ttk.Label(file_frame, text="OpenAI Model:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.openai_model_var = tk.StringVar(value=self.OPENAI_LABEL_MODELS[0])
        self.openai_model_combo = ttk.Combobox(
            file_frame,
            textvariable=self.openai_model_var,
            values=self.OPENAI_LABEL_MODELS,
            state="normal",
            width=37,
        )
        self.openai_model_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        prompt_frame = ttk.LabelFrame(file_frame, text="Prompt Lab", padding=(8, 8))
        prompt_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=(8, 5))
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(1, weight=1)

        prompt_picker = ttk.Frame(prompt_frame)
        prompt_picker.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        prompt_picker.columnconfigure(1, weight=1)
        ttk.Label(prompt_picker, text="Version:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.prompt_version_combo = ttk.Combobox(
            prompt_picker,
            textvariable=self.prompt_version_choice_var,
            state="readonly",
            width=42,
        )
        self.prompt_version_combo.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.prompt_version_combo.bind("<<ComboboxSelected>>", self.on_prompt_version_selected)
        self.refresh_prompt_version_choices()

        self.prompt_text_widget = tk.Text(
            prompt_frame,
            height=10,
            wrap="word",
            undo=True,
            font=("Consolas", 9),
        )
        self.prompt_text_widget.grid(row=1, column=0, sticky="nsew")
        prompt_scroll = ttk.Scrollbar(prompt_frame, orient="vertical", command=self.prompt_text_widget.yview)
        prompt_scroll.grid(row=1, column=1, sticky="ns")
        self.prompt_text_widget.configure(yscrollcommand=prompt_scroll.set)
        self.reload_prompt_text()

        prompt_actions = ttk.Frame(prompt_frame)
        prompt_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        prompt_actions.columnconfigure(4, weight=1)
        self.reload_prompt_btn = ttk.Button(prompt_actions, text="Reload Prompt", command=self.reload_prompt_text)
        self.reload_prompt_btn.grid(row=0, column=0, padx=(0, 6))
        self.refresh_prompt_versions_btn = ttk.Button(prompt_actions, text="Refresh Versions", command=self.refresh_prompt_version_choices)
        self.refresh_prompt_versions_btn.grid(row=0, column=1, padx=(0, 6))
        self.prompt_tests_btn = ttk.Button(prompt_actions, text="Run Prompt Tests", command=lambda: self.show_prompt_test_window(run_immediately=True))
        self.prompt_tests_btn.grid(row=0, column=2, padx=(0, 6))
        self.compare_prompts_btn = ttk.Button(prompt_actions, text="Compare Prompt Versions", command=self.show_prompt_comparison)
        self.compare_prompts_btn.grid(row=0, column=3, padx=(0, 6))

        self.prepare_btn = ttk.Button(file_frame, text="Prepare Data (OpenAI)", command=self.prepare_data)
        self.prepare_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # --- Model Training Section ---
        train_frame = ttk.LabelFrame(self.content_frame, text="Model Training (DeBERTa)", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)
        train_frame.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="Labeled Data (CSV):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.training_filepath_entry = ttk.Entry(train_frame, width=40)
        self.training_filepath_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.training_browse_btn = ttk.Button(train_frame, text="Browse", command=self.browse_training_file)
        self.training_browse_btn.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(train_frame, text="Environment:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.train_mode = tk.StringVar(value="azure")
        self.azure_radio = ttk.Radiobutton(
            train_frame,
            text="Azure Cloud",
            variable=self.train_mode,
            value="azure",
            command=self.on_train_mode_change,
        )
        self.azure_radio.grid(row=1, column=1, sticky="w", padx=5)
        self.local_radio = ttk.Radiobutton(
            train_frame,
            text="Local Device (CPU/GPU)",
            variable=self.train_mode,
            value="local",
            command=self.on_train_mode_change,
        )
        self.local_radio.grid(row=1, column=2, sticky="w", padx=5)

        self.azure_sub_label = ttk.Label(train_frame, text="Azure Sub ID:")
        self.azure_sub_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.azure_sub_entry = ttk.Entry(train_frame, width=40)
        self.azure_sub_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        self.azure_tenant_label = ttk.Label(train_frame, text="Tenant ID:")
        self.azure_tenant_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.azure_tenant_entry = ttk.Entry(train_frame, width=40)
        self.azure_tenant_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        self.azure_compute_label = ttk.Label(train_frame, text="Azure Compute:")
        self.azure_compute_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.azure_compute_var = tk.StringVar(value="cpu")
        self.azure_compute_combo = ttk.Combobox(
            train_frame,
            textvariable=self.azure_compute_var,
            state="readonly",
            values=["cpu", "gpu"],
            width=16,
        )
        self.azure_compute_combo.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.azure_compute_combo.bind("<<ComboboxSelected>>", self.on_azure_training_compute_change)

        self.azure_instance_label = ttk.Label(train_frame, text="Azure VM Size:")
        self.azure_instance_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.azure_instance_combo = ttk.Combobox(
            train_frame,
            textvariable=self.azure_training_instance_var,
            state="readonly",
            width=24,
        )
        self.azure_instance_combo.grid(row=5, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        self.local_device_label = ttk.Label(train_frame, text="Local Device:")
        self.local_device_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.local_device_var = tk.StringVar(value="auto")
        self.local_device_combo = ttk.Combobox(
            train_frame,
            textvariable=self.local_device_var,
            state="readonly",
            values=["auto", "cpu", "cuda"],
            width=16,
        )
        self.local_device_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.local_device_combo.bind("<<ComboboxSelected>>", self.on_local_device_change)

        self.local_runtime_label = ttk.Label(train_frame, text="Local Runtime:")
        self.local_runtime_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.local_runtime_var = tk.StringVar(value="host")
        self.local_runtime_combo = ttk.Combobox(
            train_frame,
            textvariable=self.local_runtime_var,
            state="disabled",
            values=["host", "container"],
            width=16,
        )
        self.local_runtime_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        self.training_config_toggle_btn = ttk.Button(
            train_frame,
            text="Show Model Parameters",
            command=self.toggle_training_config_panel,
        )
        self.training_config_toggle_btn.grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 2))

        self.training_config_frame = ttk.LabelFrame(train_frame, text="Training Config", padding=(8, 6))
        self.training_config_frame.grid(row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=(2, 6))
        self.training_config_frame.columnconfigure(1, weight=1)
        self.training_config_frame.columnconfigure(3, weight=1)

        ttk.Label(self.training_config_frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.training_strategy_var = tk.StringVar(value="default")
        self.training_strategy_combo = ttk.Combobox(
            self.training_config_frame,
            textvariable=self.training_strategy_var,
            state="readonly",
            values=["default", "tune", "tune_cv"],
            width=14,
        )
        self.training_strategy_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.training_strategy_combo.bind("<<ComboboxSelected>>", self.on_training_strategy_change)

        ttk.Label(self.training_config_frame, text="CV Folds:").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.cv_folds_var = tk.StringVar(value="3")
        self.cv_folds_entry = ttk.Entry(self.training_config_frame, textvariable=self.cv_folds_var, width=8, state="disabled")
        self.cv_folds_entry.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Epochs:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.epochs_var = tk.StringVar(value="3")
        self.epochs_entry = ttk.Entry(self.training_config_frame, textvariable=self.epochs_var, width=10)
        self.epochs_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Batch Size:").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        self.batch_size_var = tk.StringVar(value="8")
        self.batch_size_entry = ttk.Entry(self.training_config_frame, textvariable=self.batch_size_var, width=10)
        self.batch_size_entry.grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.learning_rate_var = tk.StringVar(value="5e-5")
        self.learning_rate_entry = ttk.Entry(self.training_config_frame, textvariable=self.learning_rate_var, width=14)
        self.learning_rate_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Weight Decay:").grid(row=2, column=2, sticky="w", padx=4, pady=4)
        self.weight_decay_var = tk.StringVar(value="0.01")
        self.weight_decay_entry = ttk.Entry(self.training_config_frame, textvariable=self.weight_decay_var, width=10)
        self.weight_decay_entry.grid(row=2, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Max Length:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        self.max_length_var = tk.StringVar(value="128")
        self.max_length_entry = ttk.Entry(self.training_config_frame, textvariable=self.max_length_var, width=10)
        self.max_length_entry.grid(row=3, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Max Trials:").grid(row=3, column=2, sticky="w", padx=4, pady=4)
        self.max_trials_var = tk.StringVar(value="8")
        self.max_trials_entry = ttk.Entry(self.training_config_frame, textvariable=self.max_trials_var, width=10)
        self.max_trials_entry.grid(row=3, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Tune LRs:").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        self.tune_lrs_var = tk.StringVar(value="5e-5,3e-5,1e-4")
        self.tune_lrs_entry = ttk.Entry(self.training_config_frame, textvariable=self.tune_lrs_var)
        self.tune_lrs_entry.grid(row=4, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Tune Batch Sizes:").grid(row=5, column=0, sticky="w", padx=4, pady=4)
        self.tune_batch_sizes_var = tk.StringVar(value="8,16")
        self.tune_batch_sizes_entry = ttk.Entry(self.training_config_frame, textvariable=self.tune_batch_sizes_var)
        self.tune_batch_sizes_entry.grid(row=5, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Tune Epochs:").grid(row=6, column=0, sticky="w", padx=4, pady=4)
        self.tune_epochs_var = tk.StringVar(value="3,4")
        self.tune_epochs_entry = ttk.Entry(self.training_config_frame, textvariable=self.tune_epochs_var)
        self.tune_epochs_entry.grid(row=6, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Tune Weight Decays:").grid(row=7, column=0, sticky="w", padx=4, pady=4)
        self.tune_weight_decays_var = tk.StringVar(value="0.0,0.01")
        self.tune_weight_decays_entry = ttk.Entry(self.training_config_frame, textvariable=self.tune_weight_decays_var)
        self.tune_weight_decays_entry.grid(row=7, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        ttk.Label(self.training_config_frame, text="Tune Max Lengths:").grid(row=8, column=0, sticky="w", padx=4, pady=4)
        self.tune_max_lengths_var = tk.StringVar(value="128")
        self.tune_max_lengths_entry = ttk.Entry(self.training_config_frame, textvariable=self.tune_max_lengths_var)
        self.tune_max_lengths_entry.grid(row=8, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        self.get_model_btn = ttk.Button(train_frame, text="Get Model (Train)", command=self.start_training_thread)
        self.get_model_btn.grid(row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        self.stop_training_btn = ttk.Button(
            train_frame,
            text="Interrupt Training",
            command=self.stop_training,
            state="disabled",
        )
        self.stop_training_btn.grid(row=8, column=2, sticky="ew", padx=5, pady=10)

        self.training_config_frame.grid_remove()

        # --- Hosting Section ---
        hosting_frame = ttk.LabelFrame(self.content_frame, text="Hosting", padding=(10, 10))
        hosting_frame.pack(fill="x", padx=10, pady=5)
        hosting_frame.columnconfigure(1, weight=1)
        hosting_frame.columnconfigure(3, weight=1)

        ttk.Label(hosting_frame, text="GitHub PAT:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.github_key_entry = ttk.Entry(hosting_frame, width=30, show="*")
        self.github_key_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.load_repos_btn = ttk.Button(hosting_frame, text="Load Repos", command=self.start_repo_thread)
        self.load_repos_btn.grid(row=0, column=2, padx=5, pady=5)
        self.create_pr_check = ttk.Checkbutton(
            hosting_frame,
            text="Create PR",
            variable=self.create_pr_var,
        )
        self.create_pr_check.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Repository:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.repo_combo = ttk.Combobox(hosting_frame, state="readonly", width=28)
        self.repo_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.repo_combo.bind("<<ComboboxSelected>>", self.start_branch_thread)

        ttk.Label(hosting_frame, text="Branch:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.branch_combo = ttk.Combobox(hosting_frame, state="readonly", width=15)
        self.branch_combo.grid(row=1, column=3, sticky="ew", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Generated Model:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.hosted_model_entry = ttk.Entry(hosting_frame, textvariable=self.hosted_model_path_var, width=30)
        self.hosted_model_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.hosted_model_browse_btn = ttk.Button(hosting_frame, text="Browse", command=self.browse_hosted_model)
        self.hosted_model_browse_btn.grid(row=2, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="Available Models:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.available_models_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.available_model_choice_var,
            state="readonly",
            width=30,
        )
        self.available_models_combo.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.available_models_combo.bind("<<ComboboxSelected>>", self.on_available_model_selected)
        self.refresh_models_btn = ttk.Button(
            hosting_frame,
            text="Refresh Models",
            command=self.refresh_hosted_model_inventory,
        )
        self.refresh_models_btn.grid(row=3, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="Host Target:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.host_local_radio = ttk.Radiobutton(
            hosting_frame,
            text="Local",
            variable=self.hosting_mode_var,
            value="local",
            command=self.on_hosting_mode_change,
        )
        self.host_local_radio.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.host_azure_radio = ttk.Radiobutton(
            hosting_frame,
            text="Azure",
            variable=self.hosting_mode_var,
            value="azure",
            command=self.on_hosting_mode_change,
        )
        self.host_azure_radio.grid(row=4, column=2, sticky="w", padx=5, pady=5)

        self.azure_host_service_label = ttk.Label(hosting_frame, text="Azure Service:")
        self.azure_host_service_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.azure_host_online_radio = ttk.Radiobutton(
            hosting_frame,
            text="Real-time endpoint",
            variable=self.azure_host_service_var,
            value="online",
            command=self.on_hosting_mode_change,
        )
        self.azure_host_online_radio.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.azure_host_queue_batch_radio = ttk.Radiobutton(
            hosting_frame,
            text="Queued batch API (daily)",
            variable=self.azure_host_service_var,
            value="queued_batch",
            command=self.on_hosting_mode_change,
        )
        self.azure_host_queue_batch_radio.grid(row=5, column=2, sticky="w", padx=5, pady=5)
        self.azure_host_serverless_radio = ttk.Radiobutton(
            hosting_frame,
            text="Serverless endpoint",
            variable=self.azure_host_service_var,
            value="serverless",
            command=self.on_hosting_mode_change,
        )
        self.azure_host_serverless_radio.grid(row=5, column=3, sticky="w", padx=5, pady=5)

        self.azure_serverless_model_id_label = ttk.Label(hosting_frame, text="Serverless Model ID:")
        self.azure_serverless_model_id_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.azure_serverless_model_id_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_serverless_model_id_var,
            width=30,
        )
        self.azure_serverless_model_id_entry.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
        self.azure_serverless_model_id_entry.bind("<FocusOut>", self.on_azure_serverless_model_id_focus_out)

        self.azure_serverless_endpoint_name_label = ttk.Label(hosting_frame, text="Endpoint Name:")
        self.azure_serverless_endpoint_name_label.grid(row=6, column=2, sticky="w", padx=5, pady=5)
        self.azure_serverless_endpoint_name_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_serverless_endpoint_name_var,
            width=20,
        )
        self.azure_serverless_endpoint_name_entry.grid(row=6, column=3, sticky="ew", padx=5, pady=5)

        self.azure_host_sub_label = ttk.Label(hosting_frame, text="Azure Host Sub ID:")
        self.azure_host_sub_label.grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.azure_host_sub_entry = ttk.Entry(hosting_frame, textvariable=self.azure_host_sub_var, width=30)
        self.azure_host_sub_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=5)

        self.azure_host_tenant_label = ttk.Label(hosting_frame, text="Azure Host Tenant:")
        self.azure_host_tenant_label.grid(row=7, column=2, sticky="w", padx=5, pady=5)
        self.azure_host_tenant_entry = ttk.Entry(hosting_frame, textvariable=self.azure_host_tenant_var, width=20)
        self.azure_host_tenant_entry.grid(row=7, column=3, sticky="ew", padx=5, pady=5)

        self.azure_host_compute_label = ttk.Label(hosting_frame, text="Azure Host Compute:")
        self.azure_host_compute_label.grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.azure_host_compute_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.azure_host_compute_var,
            state="readonly",
            values=["cpu", "gpu"],
            width=16,
        )
        self.azure_host_compute_combo.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        self.azure_host_compute_combo.bind("<<ComboboxSelected>>", self.on_azure_host_compute_change)

        self.azure_host_instance_label = ttk.Label(hosting_frame, text="Azure VM Size:")
        self.azure_host_instance_label.grid(row=8, column=2, sticky="w", padx=5, pady=5)
        self.azure_host_instance_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.azure_host_instance_var,
            state="readonly",
            width=20,
        )
        self.azure_host_instance_combo.grid(row=8, column=3, sticky="ew", padx=5, pady=5)

        self.azure_batch_time_label = ttk.Label(hosting_frame, text="Daily Time (HH:MM):")
        self.azure_batch_time_label.grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.azure_batch_time_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_batch_time_var,
            width=12,
        )
        self.azure_batch_time_entry.grid(row=9, column=1, sticky="w", padx=5, pady=5)

        self.azure_batch_timezone_label = ttk.Label(hosting_frame, text="Time Zone:")
        self.azure_batch_timezone_label.grid(row=9, column=2, sticky="w", padx=5, pady=5)
        self.azure_batch_timezone_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.azure_batch_timezone_var,
            state="readonly",
            values=self.get_azure_batch_timezone_options(),
            width=20,
        )
        self.azure_batch_timezone_combo.grid(row=9, column=3, sticky="ew", padx=5, pady=5)

        self.azure_batch_note_label = ttk.Label(
            hosting_frame,
            text=(
                "Queued batch hosting deploys an Azure Function log API, Service Bus queue, Blob Storage, "
                "and a daily Azure ML batch launcher. Logs can arrive anytime and are processed once per day."
            ),
            wraplength=460,
            justify="left",
        )
        self.azure_batch_note_label.grid(row=10, column=1, columnspan=3, sticky="w", padx=5, pady=(0, 6))

        self.azure_serverless_note_label = ttk.Label(
            hosting_frame,
            text=(
                "Serverless hosting uses an auto-filled Azure ML catalog model ID and endpoint name. "
                "You can edit either field; the local generated model folder is not uploaded."
            ),
            wraplength=460,
            justify="left",
        )
        self.azure_serverless_note_label.grid(row=10, column=1, columnspan=3, sticky="w", padx=5, pady=(0, 6))

        self.host_service_btn = ttk.Button(
            hosting_frame,
            text="Host Service",
            command=self.start_hosting_thread,
        )
        self.host_service_btn.grid(row=11, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        self.stop_hosting_btn = ttk.Button(
            hosting_frame,
            text="Stop Local Stack",
            command=self.stop_hosting,
            state="disabled",
        )
        self.stop_hosting_btn.grid(row=11, column=2, columnspan=2, sticky="ew", padx=5, pady=10)

        ttk.Label(hosting_frame, text="Endpoint URL:").grid(row=12, column=0, sticky="w", padx=5, pady=5)
        self.hosting_api_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.hosting_api_url_var,
            width=30,
            state="readonly",
        )
        self.hosting_api_url_entry.grid(row=12, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_hosting_api_btn = ttk.Button(
            hosting_frame,
            text="Open Endpoint",
            command=lambda: self.open_url_value(self.hosting_api_url_var.get(), "Endpoint"),
        )
        self.open_hosting_api_btn.grid(row=12, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="GitHub PR Task:").grid(row=13, column=0, sticky="w", padx=5, pady=5)
        self.github_pr_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.github_pr_url_var,
            width=30,
            state="readonly",
        )
        self.github_pr_url_entry.grid(row=13, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_github_pr_btn = ttk.Button(
            hosting_frame,
            text="Open PR Task",
            command=lambda: self.open_url_value(self.github_pr_url_var.get(), "GitHub PR task"),
        )
        self.open_github_pr_btn.grid(row=13, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="Hosting Status:").grid(row=14, column=0, sticky="nw", padx=5, pady=5)
        self.hosting_status_label = ttk.Label(
            hosting_frame,
            textvariable=self.hosting_mode_summary_var,
            wraplength=460,
            justify="left",
        )
        self.hosting_status_label.grid(row=14, column=1, columnspan=3, sticky="w", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Azure MLOps URL:").grid(row=15, column=0, sticky="w", padx=5, pady=5)
        self.azure_mlops_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_mlops_url_var,
            width=30,
            state="readonly",
        )
        self.azure_mlops_url_entry.grid(row=15, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_azure_mlops_btn = ttk.Button(
            hosting_frame,
            text="Open MLOps",
            command=lambda: self.open_url_value(self.azure_mlops_url_var.get(), "Azure MLOps dashboard"),
        )
        self.open_azure_mlops_btn.grid(row=15, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="Azure LLMOps URL:").grid(row=16, column=0, sticky="w", padx=5, pady=5)
        self.azure_llmops_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_llmops_url_var,
            width=30,
            state="readonly",
        )
        self.azure_llmops_url_entry.grid(row=16, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_azure_llmops_btn = ttk.Button(
            hosting_frame,
            text="Open LLMOps",
            command=lambda: self.open_url_value(self.azure_llmops_url_var.get(), "Azure LLMOps dashboard"),
        )
        self.open_azure_llmops_btn.grid(row=16, column=3, padx=5, pady=5)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        self._last_local_device = self.local_device_var.get()
        self.on_training_strategy_change()
        self.on_train_mode_change()
        self.on_hosting_mode_change()
        self.refresh_hosted_model_inventory()

    def _on_content_frame_configure(self, event=None):
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.main_canvas.itemconfigure(self.content_window_id, width=event.width)

    def _bind_mousewheel(self, event=None):
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event=None):
        self.main_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        if event.delta == 0:
            return
        if sys.platform == "darwin":
            step = -1 * int(event.delta)
        else:
            step = -1 * int(event.delta / 120)
        if step != 0:
            self.main_canvas.yview_scroll(step, "units")

    def _refresh_scroll_region(self):
        self.root.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def build_mlflow_config_from_ui(
        self,
        require_tracking_uri: bool,
        soft_disable: bool = False,
        ml_client: Any | None = None,
    ) -> tuple[MlflowConfig, str | None]:
        return self.mlops_service.resolve_mlflow_config(
            enabled=bool(self.mlflow_enabled_var.get()),
            backend=self.mlflow_backend_var.get().strip() or "local",
            experiment_name=self.mlflow_experiment_var.get(),
            registered_model_name=self.mlflow_registered_model_var.get(),
            tracking_uri=self.mlflow_tracking_uri_var.get(),
            require_tracking_uri=require_tracking_uri,
            ml_client=ml_client,
            soft_disable=soft_disable,
        )

    def poll_job_events(self):
        for event in self.job_manager.drain_events():
            self.handle_job_event(event)
        self.root.after(250, self.poll_job_events)

    def handle_job_event(self, event):
        if event.message:
            self.status_var.set(event.message)
        if event.job_id == self.current_data_prep_job_id:
            self._handle_data_prep_event(event)
        elif event.job_id == self.current_training_job_id:
            self._handle_training_event(event)
        elif event.job_id == self.current_hosting_job_id:
            self._handle_hosting_event(event)
        elif event.job_id == self.current_repo_job_id:
            self._handle_repo_event(event)
        elif event.job_id == self.current_branch_job_id:
            self._handle_branch_event(event)
        elif event.job_id == self.current_prompt_test_job_id:
            self._handle_prompt_test_event(event)

    def _handle_data_prep_event(self, event):
        if event.status not in {"succeeded", "failed", "canceled"}:
            return
        self.current_data_prep_job_id = ""
        self.prepare_btn.config(state="normal")
        if event.status == "succeeded":
            self.status_var.set("Data processed successfully!")
            self.refresh_prompt_version_choices()
            messagebox.showinfo("Success", event.payload.get("message", "Data processed successfully."))
            return
        if event.status == "canceled":
            self.status_var.set("Data preparation was interrupted.")
            messagebox.showinfo("Canceled", event.payload.get("message", "Data preparation was interrupted."))
            return
        self.show_error(event.payload.get("message", "Data preparation failed."))

    def _handle_training_event(self, event):
        if event.status not in {"succeeded", "failed", "canceled"}:
            return
        self.current_training_job_id = ""
        self.finish_training_session()
        if event.status == "succeeded":
            selected_model_dir = clean_optional_string(event.payload.get("selected_model_dir"))
            if selected_model_dir:
                self.refresh_hosted_model_inventory(selected_model_dir)
            self.status_var.set("Training completed successfully.")
            messagebox.showinfo("Success", event.payload.get("message", "Training completed successfully."))
            return
        if event.status == "canceled":
            self.status_var.set("Training was interrupted.")
            messagebox.showinfo("Training Interrupted", event.payload.get("message", "Training was interrupted before completion."))
            return
        self.status_var.set("Training failed.")
        messagebox.showerror("Training Error", event.payload.get("message", "Training failed."))

    def _handle_hosting_event(self, event):
        if event.status not in {"succeeded", "failed", "canceled"}:
            return
        self.current_hosting_job_id = ""
        self.hosting_process = self.observability_service.hosting_process
        self.prometheus_process = self.observability_service.prometheus_process
        self.grafana_process = self.observability_service.grafana_process
        if event.status == "succeeded":
            self.hosting_api_url_var.set(clean_optional_string(event.payload.get("api_url")))
            self.hosting_mode_summary_var.set(clean_optional_string(event.payload.get("summary")))
            self.azure_mlops_url_var.set(clean_optional_string(event.payload.get("mlops_url")))
            self.azure_llmops_url_var.set(clean_optional_string(event.payload.get("llmops_url")))
            self.azure_hosted_endpoint_name_var.set(clean_optional_string(event.payload.get("endpoint_name")))
            self.github_pr_url_var.set(clean_optional_string(event.payload.get("github_pr_url")))
            self.finish_hosting_action()
            self.status_var.set(clean_optional_string(event.payload.get("message")) or "Hosting is ready.")
            messagebox.showinfo("Hosting Ready", event.payload.get("message", "Hosting is ready."))
            self.open_hosting_dashboard_on_success(self.hosting_mode_var.get(), clean_optional_string(event.payload.get("mlops_url")))
            return
        self.finish_hosting_action()
        if event.status == "canceled":
            self.status_var.set("Hosting was interrupted.")
            messagebox.showinfo("Hosting Interrupted", event.payload.get("message", "Hosting was interrupted."))
            return
        self.status_var.set("Hosting failed.")
        messagebox.showerror("Hosting Error", event.payload.get("message", "Hosting failed."))

    def _handle_repo_event(self, event):
        if event.status not in {"succeeded", "failed", "canceled"}:
            return
        self.current_repo_job_id = ""
        self.load_repos_btn.config(state="normal")
        if event.status == "succeeded":
            self.update_repo_combo(event.payload.get("repo_names", []))
            return
        if event.status == "canceled":
            self.status_var.set("Repository load canceled.")
            return
        self.show_error(event.payload.get("message", "Failed to load repositories."))

    def _handle_branch_event(self, event):
        if event.status not in {"succeeded", "failed", "canceled"}:
            return
        self.current_branch_job_id = ""
        if event.status == "succeeded":
            self.update_branch_combo(event.payload.get("branch_names", []))
            return
        if event.status == "canceled":
            self.status_var.set("Branch load canceled.")
            return
        self.show_error(event.payload.get("message", "Failed to load branches."))

    def _handle_prompt_test_event(self, event):
        if event.status not in {"succeeded", "failed", "canceled"}:
            return
        self.current_prompt_test_job_id = ""
        if self.prompt_test_run_btn is not None:
            self.prompt_test_run_btn.config(state="normal")
        if event.status == "succeeded":
            cases = event.payload.get("cases", [])
            self.update_prompt_test_results(cases)
            passed = sum(1 for case in cases if case.get("match"))
            self.status_var.set(f"Prompt tests completed: {passed}/{len(cases)} passed.")
            return
        message = event.payload.get("message", "Prompt tests failed.")
        self.status_var.set(message)
        messagebox.showerror("Prompt Tests", message)

    # --- GitHub Repos/Branches Methods ---
    def start_repo_thread(self):
        token = self.github_key_entry.get().strip()
        if not token:
            messagebox.showwarning("Warning", "Please enter a GitHub Personal Access Token.")
            return

        self.status_var.set("Loading repositories...")
        self.load_repos_btn.config(state="disabled")
        job = self.job_manager.submit(
            "github_fetch_repos",
            lambda ctx: {"repo_names": self.github_service.fetch_repos(token)},
            metadata={"operation": "github_fetch_repos"},
        )
        self.current_repo_job_id = job.job_id

    def update_repo_combo(self, repo_names):
        self.repo_combo['values'] = repo_names
        if repo_names:
            self.repo_combo.set(repo_names[0])
            self.status_var.set("Repositories loaded.")
            self.start_branch_thread()
        else:
            self.status_var.set("No repositories found.")

    def start_branch_thread(self, event=None):
        token = self.github_key_entry.get().strip()
        repo_name = self.repo_combo.get()
        if not token or not repo_name:
            return

        self.status_var.set("Loading branches...")
        self.branch_combo.set('')
        job = self.job_manager.submit(
            "github_fetch_branches",
            lambda ctx: {"branch_names": self.github_service.fetch_branches(token, repo_name)},
            metadata={"operation": "github_fetch_branches", "repo_name": repo_name},
        )
        self.current_branch_job_id = job.job_id

    def update_branch_combo(self, branch_names):
        self.branch_combo['values'] = branch_names
        if branch_names:
            self.branch_combo.set(branch_names[0])
        self.status_var.set("Branches loaded.")

    # --- Hosting Methods ---
    def browse_hosted_model(self):
        initial_dir = clean_optional_string(self.hosted_model_path_var.get())
        if not initial_dir:
            initial_dir = str((Path(self.project_dir) / "outputs").resolve())
        if not os.path.isdir(initial_dir):
            initial_dir = self.project_dir

        selected_dir = filedialog.askdirectory(
            title="Select Generated Model Directory",
            initialdir=initial_dir,
        )
        if selected_dir:
            self.hosted_model_path_var.set(selected_dir)
            self.refresh_hosted_model_inventory(preferred_path=selected_dir)

    def refresh_azure_training_instance_options(self, preferred: str = ""):
        candidates = self.azure_platform_service.get_azure_training_instance_candidates(self.azure_compute_var.get())
        self.azure_instance_combo["values"] = candidates
        preferred_value = clean_optional_string(preferred) or clean_optional_string(self.azure_training_instance_var.get())
        if preferred_value in candidates:
            self.azure_training_instance_var.set(preferred_value)
        elif candidates:
            self.azure_training_instance_var.set(candidates[0])
        else:
            self.azure_training_instance_var.set("")

    def on_azure_training_compute_change(self, event=None):
        self.refresh_azure_training_instance_options()

    def refresh_azure_host_instance_options(self, preferred: str = ""):
        candidates = self.azure_platform_service.get_azure_host_instance_candidates(self.azure_host_compute_var.get())
        self.azure_host_instance_combo["values"] = candidates
        preferred_value = clean_optional_string(preferred) or clean_optional_string(self.azure_host_instance_var.get())
        if preferred_value in candidates:
            self.azure_host_instance_var.set(preferred_value)
        elif candidates:
            self.azure_host_instance_var.set(candidates[0])
        else:
            self.azure_host_instance_var.set("")

    def on_azure_host_compute_change(self, event=None):
        self.refresh_azure_host_instance_options()

    def on_azure_serverless_endpoint_name_changed(self, *_):
        if self.azure_serverless_endpoint_name_setting:
            return
        current_endpoint_name = clean_optional_string(self.azure_serverless_endpoint_name_var.get())
        if current_endpoint_name and current_endpoint_name != self.azure_serverless_last_auto_endpoint_name:
            self.azure_serverless_endpoint_name_is_auto = False

    def on_azure_serverless_model_id_focus_out(self, event=None):
        self.ensure_azure_serverless_defaults(refresh_endpoint=False)

    def set_azure_serverless_endpoint_name(self, endpoint_name: str, *, auto: bool):
        clean_endpoint_name = clean_optional_string(endpoint_name)
        self.azure_serverless_endpoint_name_setting = True
        try:
            self.azure_serverless_endpoint_name_var.set(clean_endpoint_name)
        finally:
            self.azure_serverless_endpoint_name_setting = False
        if auto:
            self.azure_serverless_last_auto_endpoint_name = clean_endpoint_name
            self.azure_serverless_endpoint_name_is_auto = True

    def ensure_azure_serverless_defaults(self, refresh_endpoint: bool = False):
        model_id = clean_optional_string(self.azure_serverless_model_id_var.get())
        if not model_id:
            model_id = self.azure_platform_service.get_default_serverless_model_id()
            self.azure_serverless_model_id_var.set(model_id)
        normalized_model_id = self.azure_platform_service.normalize_serverless_model_id(model_id)
        if normalized_model_id and normalized_model_id != model_id:
            self.azure_serverless_model_id_var.set(normalized_model_id)
            model_id = normalized_model_id

        endpoint_name = clean_optional_string(self.azure_serverless_endpoint_name_var.get())
        should_generate_endpoint = not endpoint_name or self.azure_serverless_endpoint_name_is_auto
        if should_generate_endpoint:
            generated_endpoint_name = self.azure_platform_service.build_default_serverless_endpoint_name(
                model_id,
                suffix=str(int(time.time())),
            )
            self.set_azure_serverless_endpoint_name(generated_endpoint_name, auto=True)


    def discover_available_hosted_models(self) -> list[dict]:
        return [asdict(entry) for entry in self.model_catalog_service.discover_available_hosted_models(self.hosted_model_path_var.get())]

    def refresh_hosted_model_inventory(self, preferred_path: str = ""):
        inventory = self.discover_available_hosted_models()
        self.hosted_model_inventory = inventory
        values = [entry.get("label", "") for entry in inventory]
        self.available_models_combo["values"] = values

        preferred_model_path = clean_optional_string(preferred_path) or clean_optional_string(self.hosted_model_path_var.get())
        resolved_preferred = ""
        if preferred_model_path:
            try:
                resolved_preferred = str(Path(discover_model_dir(preferred_model_path)).resolve())
            except Exception:
                resolved_preferred = ""

        selected_index = -1
        if resolved_preferred:
            for index, entry in enumerate(inventory):
                if clean_optional_string(entry.get("path")) == resolved_preferred:
                    selected_index = index
                    break

        if selected_index < 0 and inventory:
            selected_index = 0

        if selected_index >= 0:
            self.available_models_combo.current(selected_index)
            self.on_available_model_selected()
        else:
            self.available_model_choice_var.set("")

    def on_available_model_selected(self, event=None):
        current_index = self.available_models_combo.current()
        if current_index < 0 or current_index >= len(self.hosted_model_inventory):
            return
        selected_entry = self.hosted_model_inventory[current_index]
        selected_path = clean_optional_string(selected_entry.get("path"))
        if selected_path:
            self.hosted_model_path_var.set(selected_path)

    def on_hosting_mode_change(self):
        if not clean_optional_string(self.azure_host_sub_var.get()):
            self.azure_host_sub_var.set(clean_optional_string(self.azure_sub_entry.get()))
        if not clean_optional_string(self.azure_host_tenant_var.get()):
            self.azure_host_tenant_var.set(clean_optional_string(self.azure_tenant_entry.get()))
        if not clean_optional_string(self.azure_host_compute_var.get()):
            self.azure_host_compute_var.set(clean_optional_string(self.azure_compute_var.get()) or "cpu")
        if not clean_optional_string(self.azure_host_instance_var.get()):
            inherited_instance = clean_optional_string(self.azure_training_instance_var.get())
            if inherited_instance:
                self.azure_host_instance_var.set(inherited_instance)
        if not clean_optional_string(self.azure_batch_timezone_var.get()):
            self.azure_batch_timezone_var.set("UTC")

        is_azure = self.hosting_mode_var.get().strip() == "azure"
        azure_service = self.azure_host_service_var.get().strip()
        is_queued_batch = is_azure and azure_service == "queued_batch"
        is_serverless = is_azure and azure_service == "serverless"
        if is_serverless:
            self.ensure_azure_serverless_defaults(refresh_endpoint=False)
        show_batch_schedule = is_queued_batch
        azure_widgets = [
            self.azure_host_service_label,
            self.azure_host_online_radio,
            self.azure_host_queue_batch_radio,
            self.azure_host_serverless_radio,
            self.azure_host_sub_label,
            self.azure_host_sub_entry,
            self.azure_host_tenant_label,
            self.azure_host_tenant_entry,
        ]
        azure_compute_widgets = [
            self.azure_host_compute_label,
            self.azure_host_compute_combo,
            self.azure_host_instance_label,
            self.azure_host_instance_combo,
        ]
        for widget in azure_widgets:
            if is_azure:
                widget.grid()
            else:
                widget.grid_remove()
        for widget in azure_compute_widgets:
            if is_azure and not is_serverless:
                widget.grid()
            else:
                widget.grid_remove()
        for widget in (self.azure_batch_time_label, self.azure_batch_time_entry, self.azure_batch_timezone_label, self.azure_batch_timezone_combo, self.azure_batch_note_label):
            if show_batch_schedule:
                widget.grid()
            else:
                widget.grid_remove()
        for widget in (
            self.azure_serverless_model_id_label,
            self.azure_serverless_model_id_entry,
            self.azure_serverless_endpoint_name_label,
            self.azure_serverless_endpoint_name_entry,
            self.azure_serverless_note_label,
        ):
            if is_serverless:
                widget.grid()
            else:
                widget.grid_remove()

        if is_queued_batch:
            self.azure_batch_note_label.config(
                text=(
                    "Queued batch hosting deploys an Azure Function log API, Service Bus queue, "
                    "Blob Storage, and a daily Azure ML batch launcher. Logs can arrive anytime; "
                    "they are flushed and classified once per day at the scheduled time."
                )
            )

        self.refresh_azure_host_instance_options()
        self.refresh_azure_dashboard_links()
        self._refresh_scroll_region()

    def open_url_value(self, url: str, label: str):
        clean_url = clean_optional_string(url)
        if not clean_url:
            messagebox.showwarning("Open URL", f"No {label} URL is available yet.")
            return
        webbrowser.open(clean_url)

    def begin_hosting_action(self) -> bool:
        with self.hosting_state_lock:
            if self.hosting_active:
                return False
            self.hosting_active = True
        self.host_service_btn.config(state="disabled")
        self.stop_hosting_btn.config(state="disabled")
        return True

    def finish_hosting_action(self):
        with self.hosting_state_lock:
            self.hosting_active = False
            processes = [
                self.observability_service.hosting_process,
                self.observability_service.prometheus_process,
                self.observability_service.grafana_process,
            ]
        self.host_service_btn.config(state="normal")
        local_running = any(process is not None and process.poll() is None for process in processes)
        self.stop_hosting_btn.config(state="normal" if local_running else "disabled")


    def get_azure_batch_timezone_options(self) -> list[str]:
        return self.azure_platform_service.get_azure_batch_timezone_options()

    def build_azure_studio_url(self, sub_id: str, tenant_id: str = "") -> str:
        return self.azure_platform_service.build_azure_studio_url(sub_id, tenant_id)

    def refresh_azure_dashboard_links(self):
        sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
        tenant_id = clean_optional_string(self.azure_host_tenant_var.get()) or clean_optional_string(self.azure_tenant_entry.get())
        mlops_url, llmops_url = self.azure_platform_service.build_azure_dashboard_urls(sub_id, tenant_id)
        self.azure_mlops_url_var.set(mlops_url)
        self.azure_llmops_url_var.set(llmops_url)

    def open_local_dashboard_page(self, launch_live_console: bool = True) -> str:
        hosting_meta = self.model_catalog_service.read_last_hosting_metadata()
        if clean_optional_string(hosting_meta.get("mode")) != "local":
            raise RuntimeError("Grafana dashboard URL is not available yet. Start local hosting first.")

        hosted_model_path = clean_optional_string(hosting_meta.get("model_dir")) or clean_optional_string(self.hosted_model_path_var.get())
        sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
        tenant_id = clean_optional_string(self.azure_host_tenant_var.get()) or clean_optional_string(self.azure_tenant_entry.get())
        tracking_console_url, tracking_console_note = self.mlops_service.resolve_dashboard_tracking_console(
            backend=self.mlflow_backend_var.get().strip() or "local",
            tracking_uri=clean_optional_string(self.mlflow_tracking_uri_var.get()),
            azure_studio_url=self.azure_platform_service.build_azure_studio_url(sub_id, tenant_id),
            launch_live_console=launch_live_console,
            hosted_model_path=hosted_model_path,
        )
        training_meta = self.mlops_service.find_latest_training_mlflow_metadata(hosted_model_path) or {}
        self.observability_service.write_local_observability_files(
            hosting_meta=hosting_meta,
            training_meta=training_meta,
            tracking_console_url=tracking_console_url,
            tracking_console_note=tracking_console_note,
        )
        dashboard_url = clean_optional_string(hosting_meta.get("dashboard_url")) or clean_optional_string(
            hosting_meta.get("grafana_url")
        )
        if not dashboard_url:
            raise RuntimeError("Grafana dashboard URL is not available yet. Start local hosting first.")
        self.mlops_service.open_dashboard_url(dashboard_url)
        return dashboard_url

    def open_hosting_dashboard_on_success(self, hosting_mode: str, mlops_url: str = "") -> bool:
        try:
            mode = clean_optional_string(hosting_mode) or clean_optional_string(self.hosting_mode_var.get())
            if mode in {"azure", "azure_batch", "azure_queue_batch"}:
                dashboard_url = clean_optional_string(mlops_url) or clean_optional_string(self.azure_mlops_url_var.get())
                if dashboard_url:
                    webbrowser.open(dashboard_url)
                    return True
                sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
                tenant_id = clean_optional_string(self.azure_host_tenant_var.get()) or clean_optional_string(self.azure_tenant_entry.get())
                studio_url = self.build_azure_studio_url(sub_id, tenant_id)
                if studio_url:
                    webbrowser.open(studio_url)
                    return True
                return False

            self.open_local_dashboard_page(launch_live_console=True)
            return True
        except Exception as exc:
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set("Hosting completed, but the dashboard could not be opened."))
            self.root.after(
                0,
                lambda err=str(exc): messagebox.showwarning(
                    "Dashboard",
                    f"Hosting completed, but the dashboard could not be opened automatically.\n\n{err}",
                ),
            )
        return False


    def start_hosting_thread(self):
        hosting_mode = self.hosting_mode_var.get().strip() or "local"
        azure_service = clean_optional_string(self.azure_host_service_var.get()) or "queued_batch"
        is_serverless = hosting_mode == "azure" and azure_service == "serverless"
        model_path = clean_optional_string(self.hosted_model_path_var.get())
        resolved_model_dir = ""
        if not model_path and not is_serverless:
            messagebox.showwarning("Hosting", "Please select the generated model directory first.")
            return

        if model_path:
            try:
                resolved_model_dir = discover_model_dir(model_path)
            except Exception as exc:
                if not is_serverless:
                    messagebox.showerror("Hosting", f"Could not locate a saved model in that path.\n\n{exc}")
                    return
                resolved_model_dir = model_path

        if resolved_model_dir:
            self.hosted_model_path_var.set(resolved_model_dir)
            self.refresh_hosted_model_inventory(preferred_path=resolved_model_dir)
        self.hosting_api_url_var.set("")
        self.hosting_mode_summary_var.set("")
        self.github_pr_url_var.set("")

        if not self.begin_hosting_action():
            messagebox.showwarning("Hosting In Progress", "A hosting workflow is already running.")
            return

        request_kwargs = {
            "model_dir": resolved_model_dir,
            "mode": hosting_mode,
        }
        create_github_pr = bool(self.create_pr_var.get())
        github_token = clean_optional_string(self.github_key_entry.get())
        github_repo = clean_optional_string(self.repo_combo.get())
        github_branch = clean_optional_string(self.branch_combo.get())
        if create_github_pr:
            if hosting_mode != "azure":
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Create PR needs an Azure-hosted endpoint that the GitHub app can reach.")
                return
            if not github_token:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please enter a GitHub PAT before creating a PR task.")
                return
            if not github_repo:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please select a GitHub repository before creating a PR task.")
                return
            if not github_branch:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please select a GitHub branch before creating a PR task.")
                return
        if hosting_mode == "azure":
            if not AZURE_AVAILABLE:
                self.finish_hosting_action()
                messagebox.showerror("Azure Dependencies Missing", "Azure SDK dependencies are not installed in this Python environment.")
                return
            sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
            tenant_id = clean_optional_string(self.azure_host_tenant_var.get()) or clean_optional_string(self.azure_tenant_entry.get())
            azure_compute = clean_optional_string(self.azure_host_compute_var.get()) or "cpu"
            azure_instance_type = clean_optional_string(self.azure_host_instance_var.get())
            if azure_service == "serverless":
                self.ensure_azure_serverless_defaults(refresh_endpoint=True)
            serverless_model_id = clean_optional_string(self.azure_serverless_model_id_var.get())
            serverless_endpoint_name = clean_optional_string(self.azure_serverless_endpoint_name_var.get())
            batch_input_uri = clean_optional_string(self.azure_batch_input_var.get())
            batch_time = clean_optional_string(self.azure_batch_time_var.get())
            batch_timezone = clean_optional_string(self.azure_batch_timezone_var.get()) or "UTC"
            self.azure_host_sub_var.set(sub_id)
            self.azure_host_tenant_var.set(tenant_id)
            self.azure_host_compute_var.set(azure_compute)
            if azure_service != "serverless":
                valid_host_sizes = self.azure_platform_service.get_azure_host_instance_candidates(azure_compute)
                if not azure_instance_type:
                    azure_instance_type = valid_host_sizes[0] if valid_host_sizes else ""
                    self.azure_host_instance_var.set(azure_instance_type)
                if valid_host_sizes and azure_instance_type not in valid_host_sizes:
                    self.finish_hosting_action()
                    messagebox.showwarning("Hosting", "Please select a supported Azure VM size for hosting.")
                    return
            if azure_service == "serverless":
                batch_hour = 0
                batch_minute = 0
                if not serverless_model_id:
                    self.finish_hosting_action()
                    messagebox.showwarning("Hosting", "Please provide a Serverless Model ID from the Azure ML model catalog.")
                    return
            elif azure_service == "batch":
                if not batch_input_uri:
                    self.finish_hosting_action()
                    messagebox.showwarning("Hosting", "Please provide a batch input URI for the daily batch schedule.")
                    return
                if not self.azure_platform_service.is_cloud_accessible_batch_input(batch_input_uri):
                    self.finish_hosting_action()
                    messagebox.showwarning(
                        "Hosting",
                        "Batch schedules need an Azure-accessible URI such as azureml://..., https://..., or a workspace data asset ID.",
                    )
                    return
                try:
                    batch_hour, batch_minute = self.azure_platform_service.parse_daily_time(batch_time)
                except Exception as exc:
                    self.finish_hosting_action()
                    messagebox.showwarning("Hosting", str(exc))
                    return
            elif azure_service == "queued_batch":
                try:
                    batch_hour, batch_minute = self.azure_platform_service.parse_daily_time(batch_time)
                except Exception as exc:
                    self.finish_hosting_action()
                    messagebox.showwarning("Hosting", str(exc))
                    return
            else:
                batch_hour = 0
                batch_minute = 0
            self.refresh_azure_dashboard_links()

            if not sub_id:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please provide your Azure Subscription ID for hosting.")
                return
            if not tenant_id:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please provide your Azure Tenant ID for hosting.")
                return

            request_kwargs.update(
                {
                    "azure_sub_id": sub_id,
                    "azure_tenant_id": tenant_id,
                    "azure_compute": azure_compute,
                    "azure_instance_type": azure_instance_type,
                    "azure_service": azure_service,
                    "azure_serverless_model_id": serverless_model_id,
                    "azure_serverless_endpoint_name": serverless_endpoint_name,
                    "batch_input_uri": batch_input_uri,
                    "batch_hour": batch_hour,
                    "batch_minute": batch_minute,
                    "batch_timezone": batch_timezone,
                    "create_github_pr": create_github_pr,
                    "github_token": github_token,
                    "github_repo": github_repo,
                    "github_branch": github_branch,
                }
            )
            job = self.hosting_service.submit_hosting(HostingRequest(**request_kwargs))
            self.current_hosting_job_id = job.job_id
            return

        auto_install_missing_tools = False
        missing_tools = self.observability_service.get_missing_local_observability_tools()
        if missing_tools:
            if not self.observability_service.can_auto_install_local_observability():
                self.finish_hosting_action()
                messagebox.showerror(
                    "Local Hosting",
                    (
                        "Local Grafana hosting needs extra dependencies, but automatic installation is only "
                        "supported on 64-bit Windows, macOS with Homebrew, or Debian/Ubuntu-style Linux systems with `pkexec`.\n\n"
                        "Missing tools:\n- "
                        + "\n- ".join(missing_tools)
                    ),
                )
                return

            install_message = (
                "Local hosting now uses Grafana and Prometheus.\n\n"
                "The required dependencies are missing:\n- "
                + "\n- ".join(missing_tools)
            )
            if sys.platform == "win32":
                install_message += (
                    "\n\nThe app can download the official Windows portable builds into this project automatically. Continue?"
                )
            elif sys.platform == "darwin":
                install_message += (
                    "\n\nThe app can install them automatically with Homebrew. Continue?"
                )
            else:
                install_message += (
                    "\n\nThe app can install them automatically using your system administrator password. Continue?"
                )

            should_install = messagebox.askyesno(
                "Install Grafana And Prometheus",
                install_message,
            )
            if not should_install:
                self.finish_hosting_action()
                self.status_var.set("Local hosting canceled.")
                return
            auto_install_missing_tools = True

        request_kwargs["auto_install_missing_tools"] = auto_install_missing_tools
        request_kwargs["create_github_pr"] = create_github_pr
        request_kwargs["github_token"] = github_token
        request_kwargs["github_repo"] = github_repo
        request_kwargs["github_branch"] = github_branch
        job = self.hosting_service.submit_hosting(HostingRequest(**request_kwargs))
        self.current_hosting_job_id = job.job_id

    def stop_hosting(self):
        if self.current_hosting_job_id:
            if self.hosting_service.cancel(self.current_hosting_job_id):
                self.status_var.set("Interrupt requested. Stopping hosting...")
                return

        processes = [
            self.observability_service.hosting_process,
            self.observability_service.prometheus_process,
            self.observability_service.grafana_process,
        ]
        if not any(process is not None and process.poll() is None for process in processes):
            self.stop_hosting_btn.config(state="disabled")
            self.status_var.set("No local hosted stack is currently running.")
            return

        try:
            self.hosting_service.stop_local_stack()
        finally:
            self.hosting_process = None
            self.prometheus_process = None
            self.grafana_process = None
            self.hosting_api_url_var.set("")
            self.hosting_mode_summary_var.set("Local Grafana hosting stack stopped.")
            self.stop_hosting_btn.config(state="disabled")
            self.status_var.set("Local hosted stack stopped.")


    def on_app_close(self):
        self.hosting_service.stop_local_stack()
        self.mlops_service.stop_local_mlflow_ui()
        self.root.destroy()

    # --- File Processing & LLM Methods ---
    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.filepath_entry.delete(0, tk.END)
            self.filepath_entry.insert(0, filepath)

    def browse_training_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Labeled Data File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.training_filepath_entry.delete(0, tk.END)
            self.training_filepath_entry.insert(0, filepath)

    def on_train_mode_change(self):
        is_azure = self.train_mode.get() == "azure"
        azure_state = "normal" if is_azure else "disabled"
        azure_compute_state = "readonly" if is_azure else "disabled"
        azure_instance_state = "readonly" if is_azure else "disabled"
        local_state = "disabled" if is_azure else "readonly"

        if is_azure:
            self.azure_sub_label.grid()
            self.azure_sub_entry.grid()
            self.azure_tenant_label.grid()
            self.azure_tenant_entry.grid()
            self.azure_compute_label.grid()
            self.azure_compute_combo.grid()
            self.azure_instance_label.grid()
            self.azure_instance_combo.grid()
        else:
            self.azure_sub_label.grid_remove()
            self.azure_sub_entry.grid_remove()
            self.azure_tenant_label.grid_remove()
            self.azure_tenant_entry.grid_remove()
            self.azure_compute_label.grid_remove()
            self.azure_compute_combo.grid_remove()
            self.azure_instance_label.grid_remove()
            self.azure_instance_combo.grid_remove()

        self.azure_sub_entry.config(state=azure_state)
        self.azure_tenant_entry.config(state=azure_state)
        self.azure_compute_combo.config(state=azure_compute_state)
        self.azure_instance_combo.config(state=azure_instance_state)
        self.local_device_combo.config(state=local_state)
        self.refresh_azure_training_instance_options()

        if is_azure:
            self.local_device_label.grid_remove()
            self.local_device_combo.grid_remove()
            self.local_runtime_label.grid_remove()
            self.local_runtime_combo.grid_remove()
            self.local_runtime_var.set("host")
            self.local_runtime_combo.config(state="disabled")
        else:
            self.local_device_label.grid()
            self.local_device_combo.grid()
            self.local_runtime_label.grid()
            self.local_runtime_combo.grid()
            self.on_local_device_change()
        self._refresh_scroll_region()

    def on_local_device_change(self, event=None):
        if self.train_mode.get() != "local":
            self.local_runtime_var.set("host")
            self.local_runtime_combo.config(state="disabled")
            return

        current_device = self.local_device_var.get().strip() or "auto"
        if current_device == "cuda":
            if self._last_local_device != "cuda":
                self.local_runtime_var.set("container")
            self.local_runtime_combo.config(state="readonly")
        else:
            self.local_runtime_var.set("host")
            self.local_runtime_combo.config(state="disabled")

        self._last_local_device = current_device

    def on_training_strategy_change(self, event=None):
        strategy = (self.training_strategy_var.get().strip() or "default").lower()
        is_tuning_mode = strategy in {"tune", "tune_cv"}
        is_cv_mode = strategy == "tune_cv"

        tune_state = "normal" if is_tuning_mode else "disabled"
        cv_state = "normal" if is_cv_mode else "disabled"
        self.max_trials_entry.config(state=tune_state)
        self.tune_lrs_entry.config(state=tune_state)
        self.tune_batch_sizes_entry.config(state=tune_state)
        self.tune_epochs_entry.config(state=tune_state)
        self.tune_weight_decays_entry.config(state=tune_state)
        self.tune_max_lengths_entry.config(state=tune_state)
        self.cv_folds_entry.config(state=cv_state)
        self._refresh_scroll_region()

    def toggle_training_config_panel(self):
        if self.training_config_visible:
            self.training_config_frame.grid_remove()
            self.training_config_toggle_btn.config(text="Show Model Parameters")
            self.training_config_visible = False
        else:
            self.training_config_frame.grid()
            self.training_config_toggle_btn.config(text="Hide Model Parameters")
            self.training_config_visible = True
        self._refresh_scroll_region()


    def start_training_session(self) -> bool:
        with self.training_state_lock:
            if self.training_active:
                return False
            self.training_active = True

        self.get_model_btn.config(state="disabled")
        self.stop_training_btn.config(state="normal")
        return True

    def finish_training_session(self):
        with self.training_state_lock:
            self.training_active = False

        self.get_model_btn.config(state="normal")
        self.stop_training_btn.config(state="disabled")
        self.status_var.set("Ready")


    def stop_training(self):
        if not self.current_training_job_id:
            self.status_var.set("No training is currently running.")
            return
        if self.training_service.cancel(self.current_training_job_id):
            self.status_var.set("Interrupt requested. Stopping training...")

    def reload_prompt_text(self):
        if not hasattr(self, "prompt_text_widget"):
            return
        try:
            prompt_text = self.mlops_service.load_prompt()
        except Exception as exc:
            messagebox.showerror("Prompt", str(exc))
            return
        self.prompt_text_widget.delete("1.0", tk.END)
        self.prompt_text_widget.insert("1.0", prompt_text)
        if hasattr(self, "prompt_version_choice_var"):
            self.prompt_version_choice_var.set("default")

    def refresh_prompt_version_choices(self):
        choices = [{"label": "default", "version_id": "", "source": "default"}]
        try:
            versions = list(reversed(self.mlops_service.list_prompt_versions()))
        except Exception:
            versions = []
        for version in versions:
            version_id = clean_optional_string(version.get("prompt_version_id"))
            label = clean_optional_string(version.get("prompt_version_label")) or version_id[:12]
            created_at = clean_optional_string(version.get("created_at")).replace("T", " ")[:19]
            source = clean_optional_string(version.get("prompt_source"))
            display_parts = [label]
            if created_at:
                display_parts.append(created_at)
            if source:
                display_parts.append(source)
            choices.append({"label": " | ".join(display_parts), "version_id": version_id, "source": source or "archived"})
        self.prompt_version_choices = choices
        if hasattr(self, "prompt_version_combo"):
            labels = [choice["label"] for choice in choices]
            self.prompt_version_combo["values"] = labels
            current = clean_optional_string(self.prompt_version_choice_var.get()) or "default"
            if current not in labels:
                self.prompt_version_choice_var.set("default")

    def get_selected_prompt_version_choice(self) -> dict[str, str]:
        selected = clean_optional_string(self.prompt_version_choice_var.get()) or "default"
        for choice in self.prompt_version_choices:
            if choice.get("label") == selected:
                return dict(choice)
        return {"label": "default", "version_id": "", "source": "default"}

    def on_prompt_version_selected(self, event=None):
        del event
        choice = self.get_selected_prompt_version_choice()
        version_id = clean_optional_string(choice.get("version_id"))
        try:
            if version_id:
                prompt_text = self.mlops_service.read_prompt_version_text(version_id)
            else:
                prompt_text = self.mlops_service.load_prompt()
        except Exception as exc:
            messagebox.showerror("Prompt Versions", str(exc))
            return
        self.prompt_text_widget.delete("1.0", tk.END)
        self.prompt_text_widget.insert("1.0", prompt_text)

    def get_prompt_text_from_ui(self) -> str:
        if not hasattr(self, "prompt_text_widget"):
            return self.mlops_service.load_prompt()
        return self.prompt_text_widget.get("1.0", "end-1c").strip()

    def get_prompt_source_from_ui(self) -> str:
        choice = self.get_selected_prompt_version_choice()
        version_id = clean_optional_string(choice.get("version_id"))
        if version_id:
            return f"derived_from:{version_id}"
        return "default"

    def get_prompt_test_cases(self) -> list[dict]:
        if not self.prompt_test_cases:
            self.prompt_test_cases = [dict(case) for case in self.DEFAULT_PROMPT_TEST_CASES]
        return self.prompt_test_cases

    def show_prompt_test_window(self, run_immediately: bool = False):
        if self.prompt_test_window is not None and self.prompt_test_window.winfo_exists():
            self.prompt_test_window.lift()
            if run_immediately:
                self.run_prompt_tests()
            return

        self.prompt_test_cases = [dict(case) for case in self.get_prompt_test_cases()]
        self.prompt_test_row_ids = []
        window = tk.Toplevel(self.root)
        self.prompt_test_window = window
        window.title("Prompt Unit Tests")
        window.geometry("980x620")
        window.minsize(760, 440)
        window.protocol("WM_DELETE_WINDOW", self.close_prompt_test_window)

        outer = ttk.Frame(window, padding=(10, 10))
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        columns = ("name", "message", "expected", "got", "result")
        tree = ttk.Treeview(outer, columns=columns, show="headings", height=12)
        self.prompt_test_tree = tree
        tree.heading("name", text="Test Case")
        tree.heading("message", text="Log Message")
        tree.heading("expected", text="Expected")
        tree.heading("got", text="Got")
        tree.heading("result", text="Result")
        tree.column("name", width=170, anchor="w")
        tree.column("message", width=420, anchor="w")
        tree.column("expected", width=120, anchor="center")
        tree.column("got", width=120, anchor="center")
        tree.column("result", width=90, anchor="center")
        tree.tag_configure("pass", background="#d9f7df", foreground="#14532d")
        tree.tag_configure("fail", background="#ffe1e1", foreground="#7f1d1d")
        tree.tag_configure("pending", background="#eef2ff", foreground="#1e3a8a")
        tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(outer, orient="vertical", command=tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=tree_scroll.set)

        editor = ttk.LabelFrame(outer, text="Add Custom Test Case", padding=(8, 8))
        editor.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        editor.columnconfigure(1, weight=1)
        ttk.Label(editor, text="Name:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.prompt_test_name_var = tk.StringVar(value="Custom Case")
        ttk.Entry(editor, textvariable=self.prompt_test_name_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(editor, text="Expected:").grid(row=0, column=2, sticky="w", padx=(10, 6), pady=3)
        self.prompt_test_expected_var = tk.StringVar(value="Error")
        ttk.Combobox(
            editor,
            textvariable=self.prompt_test_expected_var,
            values=["Error", "CONFIGURATION", "SYSTEM", "Noise"],
            state="readonly",
            width=18,
        ).grid(row=0, column=3, sticky="ew", pady=3)
        ttk.Label(editor, text="Log Message:").grid(row=1, column=0, sticky="nw", padx=(0, 6), pady=3)
        self.prompt_test_message_text = tk.Text(editor, height=3, wrap="word")
        self.prompt_test_message_text.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)

        actions = ttk.Frame(outer)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        actions.columnconfigure(3, weight=1)
        ttk.Button(actions, text="Add Test Case", command=self.add_prompt_test_case).grid(row=0, column=0, padx=(0, 6))
        self.prompt_test_run_btn = ttk.Button(actions, text="Run Tests", command=self.run_prompt_tests)
        self.prompt_test_run_btn.grid(row=0, column=1, padx=(0, 6))
        ttk.Button(actions, text="Reset Built-ins", command=self.reset_prompt_test_cases).grid(row=0, column=2, padx=(0, 6))

        self.refresh_prompt_test_tree()
        if run_immediately:
            self.run_prompt_tests()

    def close_prompt_test_window(self):
        if self.prompt_test_window is not None:
            self.prompt_test_window.destroy()
        self.prompt_test_window = None
        self.prompt_test_tree = None
        self.prompt_test_run_btn = None
        self.prompt_test_row_ids = []

    def refresh_prompt_test_tree(self):
        if self.prompt_test_tree is None:
            return
        self.prompt_test_tree.delete(*self.prompt_test_tree.get_children())
        self.prompt_test_row_ids = []
        for case in self.get_prompt_test_cases():
            row_id = self.prompt_test_tree.insert(
                "",
                "end",
                values=(
                    clean_optional_string(case.get("name")),
                    clean_optional_string(case.get("message")),
                    clean_optional_string(case.get("expected")),
                    clean_optional_string(case.get("got")),
                    clean_optional_string(case.get("result")) or "Not run",
                ),
                tags=("pending",),
            )
            self.prompt_test_row_ids.append(row_id)

    def add_prompt_test_case(self):
        if self.prompt_test_window is None:
            return
        message = self.prompt_test_message_text.get("1.0", "end-1c").strip()
        expected = self.prompt_test_expected_var.get().strip()
        name = self.prompt_test_name_var.get().strip() or "Custom Case"
        if not message:
            messagebox.showwarning("Prompt Tests", "Enter a log message for the custom test case.")
            return
        self.get_prompt_test_cases().append({"name": name, "message": message, "expected": expected})
        self.prompt_test_message_text.delete("1.0", tk.END)
        self.refresh_prompt_test_tree()

    def reset_prompt_test_cases(self):
        self.prompt_test_cases = [dict(case) for case in self.DEFAULT_PROMPT_TEST_CASES]
        self.refresh_prompt_test_tree()

    def run_prompt_tests(self):
        api_key = self.openai_api_key_entry.get().strip()
        model_name = self.openai_model_var.get().strip()
        prompt_text = self.get_prompt_text_from_ui()
        if not api_key:
            messagebox.showwarning("Prompt Tests", "Please enter your OpenAI API key.")
            return
        if not model_name:
            messagebox.showwarning("Prompt Tests", "Please select or enter an OpenAI model name.")
            return
        if not prompt_text:
            messagebox.showwarning("Prompt Tests", "Prompt text is empty.")
            return
        if self.current_prompt_test_job_id:
            messagebox.showwarning("Prompt Tests", "Prompt tests are already running.")
            return
        cases = [dict(case) for case in self.get_prompt_test_cases()]
        if not cases:
            messagebox.showwarning("Prompt Tests", "No prompt test cases are available.")
            return
        self.show_prompt_test_window(run_immediately=False)
        for row_id in self.prompt_test_row_ids:
            self.prompt_test_tree.item(row_id, tags=("pending",))
        if self.prompt_test_run_btn is not None:
            self.prompt_test_run_btn.config(state="disabled")
        self.status_var.set("Running prompt tests...")
        job = self.job_manager.submit(
            "prompt_tests",
            lambda ctx: self.data_prep_service.evaluate_prompt_test_cases(
                api_key=api_key,
                model_name=model_name,
                prompt_text=prompt_text,
                cases=cases,
            ),
            metadata={"operation": "prompt_tests", "case_count": len(cases)},
        )
        self.current_prompt_test_job_id = job.job_id

    def update_prompt_test_results(self, cases: list[dict]):
        if self.prompt_test_tree is None:
            return
        for index, case in enumerate(cases):
            if index >= len(self.prompt_test_row_ids):
                continue
            got = clean_optional_string(case.get("got"))
            expected = clean_optional_string(case.get("expected"))
            is_match = bool(case.get("match"))
            row_id = self.prompt_test_row_ids[index]
            values = (
                clean_optional_string(case.get("name")),
                clean_optional_string(case.get("message")),
                expected,
                got,
                "Pass" if is_match else "Fail",
            )
            self.prompt_test_tree.item(row_id, values=values, tags=("pass" if is_match else "fail",))

    def show_prompt_comparison(self):
        try:
            comparison = self.mlops_service.compare_prompt_versions()
        except Exception as exc:
            messagebox.showerror("Prompt Versions", str(exc))
            return

        diff_text = clean_optional_string(comparison.get("diff"))
        message = clean_optional_string(comparison.get("message"))
        old_meta = comparison.get("old") if isinstance(comparison.get("old"), dict) else {}
        new_meta = comparison.get("new") if isinstance(comparison.get("new"), dict) else {}

        if not diff_text:
            if message:
                messagebox.showinfo("Prompt Versions", message)
                return
            diff_text = "No textual prompt changes."

        old_label = clean_optional_string(old_meta.get("prompt_version_label")) or clean_optional_string(old_meta.get("prompt_version_id"))[:12]
        new_label = clean_optional_string(new_meta.get("prompt_version_label")) or clean_optional_string(new_meta.get("prompt_version_id"))[:12]
        header = (
            f"Old prompt: {old_label or 'not available'}\n"
            f"New prompt: {new_label or 'not available'}\n\n"
        )

        window = tk.Toplevel(self.root)
        window.title("Prompt Version Comparison")
        window.geometry("900x600")
        window.minsize(640, 420)

        frame = ttk.Frame(window, padding=(8, 8))
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        text_widget = tk.Text(frame, wrap="none", font=("Consolas", 10))
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_widget.insert("1.0", header + diff_text)
        text_widget.configure(state="disabled")

    def prepare_data(self):
        api_key = self.openai_api_key_entry.get().strip()
        model_name = self.openai_model_var.get().strip()
        input_path = self.filepath_entry.get().strip()
        prompt_text = self.get_prompt_text_from_ui()
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter your OpenAI API key.")
            return
        if not model_name:
            messagebox.showwarning("Warning", "Please select or enter an OpenAI model name.")
            return
        if not input_path:
            messagebox.showwarning("Warning", "Please select a log file first.")
            return
        if not prompt_text:
            messagebox.showwarning("Warning", "Prompt text is empty.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Prepared Data",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not save_path:
            return 

        mlflow_config, mlflow_error = self.build_mlflow_config_from_ui(require_tracking_uri=False, soft_disable=True)
        if mlflow_error:
            messagebox.showerror("MLflow Configuration Error", mlflow_error)
            return

        self.status_var.set(f"Processing data with OpenAI ({model_name})...")
        self.prepare_btn.config(state="disabled")
        job = self.data_prep_service.submit_data_prep(
            DataPrepRequest(
                input_path=input_path,
                output_path=save_path,
                api_key=api_key,
                model_name=model_name,
                mlflow_config=mlflow_config,
                prompt_text=prompt_text,
                prompt_source=self.get_prompt_source_from_ui(),
            )
        )
        self.current_data_prep_job_id = job.job_id


    # --- HEAVILY LOGGED TRAINING METHODS ---
    def start_training_thread(self):
        csv_path = self.training_filepath_entry.get().strip()
        sub_id = self.azure_sub_entry.get().strip()
        tenant_id = self.azure_tenant_entry.get().strip()
        azure_compute = self.azure_compute_var.get().strip().lower() or "cpu"
        azure_instance_type = clean_optional_string(self.azure_training_instance_var.get())
        local_device = self.local_device_var.get().strip() or "auto"
        local_runtime = self.local_runtime_var.get().strip() or "host"
        train_mode = self.train_mode.get().strip()
        training_options, training_options_error = self.training_service.collect_training_options(
            {
                "strategy": self.training_strategy_var.get(),
                "epochs": self.epochs_var.get(),
                "batch_size": self.batch_size_var.get(),
                "learning_rate": self.learning_rate_var.get(),
                "weight_decay": self.weight_decay_var.get(),
                "max_length": self.max_length_var.get(),
                "cv_folds": self.cv_folds_var.get(),
                "max_trials": self.max_trials_var.get(),
                "tune_lrs": self.tune_lrs_var.get(),
                "tune_batch_sizes": self.tune_batch_sizes_var.get(),
                "tune_epochs": self.tune_epochs_var.get(),
                "tune_weight_decays": self.tune_weight_decays_var.get(),
                "tune_max_lengths": self.tune_max_lengths_var.get(),
            }
        )

        print("\n--- [DEBUG] STARTING TRAINING WORKFLOW ---")
        print(f"[DEBUG] CSV Path: {csv_path}")
        print(f"[DEBUG] Sub ID: {sub_id}")
        print(f"[DEBUG] Tenant ID: {tenant_id}")
        print(f"[DEBUG] Azure Compute: {azure_compute}")
        print(f"[DEBUG] Azure VM Size: {azure_instance_type}")
        print(f"[DEBUG] Train Mode: {self.train_mode.get()}")
        print(f"[DEBUG] Local Device: {local_device}")
        print(f"[DEBUG] Local Runtime: {local_runtime}")
        print(f"[DEBUG] Training Strategy: {(training_options or {}).get('train_mode', '')}")

        if not csv_path:
            messagebox.showwarning("Warning", "Please select a labeled CSV file for training.")
            return
        if not os.path.exists(csv_path):
            messagebox.showwarning("Warning", "The selected labeled CSV file does not exist.")
            return
        if train_mode == "azure" and not sub_id:
            messagebox.showwarning("Warning", "Please provide your Azure Subscription ID.")
            return
        if train_mode == "azure" and not tenant_id:
            messagebox.showwarning("Warning", "Please provide your Azure Tenant ID.")
            return
        if train_mode == "azure" and not AZURE_AVAILABLE:
            messagebox.showerror("Azure Dependencies Missing", "Azure SDK dependencies are not installed in this Python environment.")
            return
        if not os.path.exists("train.py"):
            messagebox.showerror("Error", "Could not find 'train.py' in the app directory.")
            return
        if training_options_error:
            messagebox.showerror("Training Configuration Error", training_options_error)
            return
        if training_options is None:
            messagebox.showerror("Training Configuration Error", "Training configuration could not be resolved.")
            return
        if train_mode == "azure":
            valid_training_sizes = self.azure_platform_service.get_azure_training_instance_candidates(azure_compute)
            if not azure_instance_type:
                azure_instance_type = valid_training_sizes[0] if valid_training_sizes else ""
                self.azure_training_instance_var.set(azure_instance_type)
            if valid_training_sizes and azure_instance_type not in valid_training_sizes:
                messagebox.showerror(
                    "Azure Configuration Error",
                    "Please select a supported Azure VM size for training.",
                )
                return

        mlflow_config, mlflow_error = self.build_mlflow_config_from_ui(require_tracking_uri=False, soft_disable=True)
        if mlflow_error:
            messagebox.showerror("MLflow Configuration Error", mlflow_error)
            return

        if mlflow_config.enabled:
            if train_mode == "azure" and mlflow_config.backend == "local":
                messagebox.showerror(
                    "MLflow Configuration Error",
                    (
                        "Azure training cannot use local MLflow backend.\n\n"
                        "Switch MLflow Backend to `azure` or `custom_uri` before starting Azure training."
                    ),
                )
                return

            if mlflow_config.backend == "custom_uri" and not clean_optional_string(mlflow_config.tracking_uri):
                messagebox.showerror(
                    "MLflow Configuration Error",
                    "MLflow backend is `custom_uri` but Tracking URI is empty.",
                )
                return

            if (
                train_mode == "local"
                and mlflow_config.backend == "azure"
                and not clean_optional_string(mlflow_config.tracking_uri)
            ):
                messagebox.showerror(
                    "MLflow Configuration Error",
                    (
                        "MLflow backend is `azure` but tracking URI is unresolved.\n\n"
                        "Run one Azure training first or use `custom_uri`."
                    ),
                )
                return

        if train_mode != "azure":
            if local_device != "cuda":
                local_runtime = "host"
            if local_device == "cuda" and local_runtime == "host":
                available, check_error = self.training_service.check_host_cuda_available()
                if not available:
                    hint = (
                        "CUDA is not available in the host Python environment.\n\n"
                        "Switch Local Runtime to 'container' or install CUDA-enabled PyTorch locally."
                    )
                    if check_error:
                        hint += f"\n\nDetails:\n{check_error}"
                    messagebox.showerror("CUDA Not Available", hint)
                    return

        if not self.start_training_session():
            messagebox.showwarning("Training In Progress", "A training workflow is already running.")
            return
        self.status_var.set("Preparing training...")
        job = self.training_service.submit_training(
            TrainingRequest(
                csv_path=csv_path,
                environment_mode=train_mode,
                local_device=local_device,
                local_runtime=local_runtime,
                azure_sub_id=sub_id,
                azure_tenant_id=tenant_id,
                azure_compute=azure_compute,
                azure_instance_type=azure_instance_type,
                training_options=training_options,
                mlflow_config=mlflow_config,
            )
        )
        self.current_training_job_id = job.job_id


    def show_error(self, message):
        self.status_var.set("Process halted.")
        messagebox.showerror("Notice", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = LogProcessorApp(root)
    root.mainloop()
