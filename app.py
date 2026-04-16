import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import requests
import pandas as pd
import json
import os
import subprocess
import sys
import time
import traceback
import tempfile
import uuid
import webbrowser
import shutil
import shlex
import socket
import re
import html
from pathlib import Path
from urllib.parse import quote as url_quote

# Azure ML Imports
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    Workspace,
    AmlCompute,
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.mgmt.resource import ResourceManagementClient

from mlops_utils import (
    MLOPS_ENV_VARS,
    bool_from_env,
    clean_optional_string,
    compute_file_sha256,
    dataframe_metadata,
    dataframe_sample,
    discover_model_dir,
    local_mlflow_tracking_uri,
    now_utc_iso,
    prompt_sha256,
    read_json,
    read_sidecar_for_csv,
    safe_prompt_preview,
    write_json,
    write_sidecar_for_csv,
)

try:
    import mlflow
except Exception:
    mlflow = None


class TrainingInterrupted(Exception):
    """Raised when the user interrupts an active training workflow."""


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

    def __init__(self, root):
        self.root = root
        self.root.title("Log Classifier & DeBERTa Trainer")
        self.root.geometry("650x700")
        self.root.minsize(650, 620)
        self.root.resizable(True, True)
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.azure_mlflow_tracking_uri = ""
        self.local_mlflow_tracking_uri = local_mlflow_tracking_uri()
        self.mlflow_ui_process = None
        self.training_cancel_event = threading.Event()
        self.training_state_lock = threading.Lock()
        self.training_active = False
        self.local_training_process = None
        self.azure_ml_client_for_cancel = None
        self.azure_active_job_name = ""
        self.azure_cancel_requested = False
        self.training_config_visible = False
        self.hosting_active = False
        self.hosting_state_lock = threading.Lock()
        self.hosting_process = None
        self.hosting_mode_var = tk.StringVar(value="local")
        default_model_dir = Path(self.project_dir) / "outputs" / "final_model"
        self.hosted_model_path_var = tk.StringVar(value=str(default_model_dir) if default_model_dir.exists() else "")
        self.hosting_api_url_var = tk.StringVar(value="")
        self.hosting_mode_summary_var = tk.StringVar(value="")
        self.azure_host_sub_var = tk.StringVar(value="")
        self.azure_host_tenant_var = tk.StringVar(value="")
        self.azure_host_compute_var = tk.StringVar(value="cpu")
        self.azure_mlops_url_var = tk.StringVar(value="")
        self.azure_llmops_url_var = tk.StringVar(value="")
        self.azure_hosted_endpoint_name_var = tk.StringVar(value="")
        self.azure_host_sub_var.trace_add("write", lambda *_: self.refresh_azure_dashboard_links())

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

        self.prepare_btn = ttk.Button(file_frame, text="Prepare Data (OpenAI)", command=self.prepare_data)
        self.prepare_btn.grid(row=3, column=0, columnspan=3, pady=10)

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
        self.training_config_toggle_btn.grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 2))

        self.training_config_frame = ttk.LabelFrame(train_frame, text="Training Config", padding=(8, 6))
        self.training_config_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=(2, 6))
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
        self.get_model_btn.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        self.stop_training_btn = ttk.Button(
            train_frame,
            text="Interrupt Training",
            command=self.stop_training,
            state="disabled",
        )
        self.stop_training_btn.grid(row=7, column=2, sticky="ew", padx=5, pady=10)

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

        ttk.Label(hosting_frame, text="Host Target:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.host_local_radio = ttk.Radiobutton(
            hosting_frame,
            text="Local",
            variable=self.hosting_mode_var,
            value="local",
            command=self.on_hosting_mode_change,
        )
        self.host_local_radio.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.host_azure_radio = ttk.Radiobutton(
            hosting_frame,
            text="Azure",
            variable=self.hosting_mode_var,
            value="azure",
            command=self.on_hosting_mode_change,
        )
        self.host_azure_radio.grid(row=3, column=2, sticky="w", padx=5, pady=5)

        self.azure_host_sub_label = ttk.Label(hosting_frame, text="Azure Host Sub ID:")
        self.azure_host_sub_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.azure_host_sub_entry = ttk.Entry(hosting_frame, textvariable=self.azure_host_sub_var, width=30)
        self.azure_host_sub_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        self.azure_host_tenant_label = ttk.Label(hosting_frame, text="Azure Host Tenant:")
        self.azure_host_tenant_label.grid(row=4, column=2, sticky="w", padx=5, pady=5)
        self.azure_host_tenant_entry = ttk.Entry(hosting_frame, textvariable=self.azure_host_tenant_var, width=20)
        self.azure_host_tenant_entry.grid(row=4, column=3, sticky="ew", padx=5, pady=5)

        self.azure_host_compute_label = ttk.Label(hosting_frame, text="Azure Host Compute:")
        self.azure_host_compute_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.azure_host_compute_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.azure_host_compute_var,
            state="readonly",
            values=["cpu", "gpu"],
            width=16,
        )
        self.azure_host_compute_combo.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        self.host_service_btn = ttk.Button(
            hosting_frame,
            text="Host Service",
            command=self.start_hosting_thread,
        )
        self.host_service_btn.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        self.stop_hosting_btn = ttk.Button(
            hosting_frame,
            text="Stop Hosted API",
            command=self.stop_hosting,
            state="disabled",
        )
        self.stop_hosting_btn.grid(row=6, column=2, columnspan=2, sticky="ew", padx=5, pady=10)

        ttk.Label(hosting_frame, text="POST API URL:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.hosting_api_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.hosting_api_url_var,
            width=30,
            state="readonly",
        )
        self.hosting_api_url_entry.grid(row=7, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_hosting_api_btn = ttk.Button(
            hosting_frame,
            text="Open API URL",
            command=lambda: self.open_url_value(self.hosting_api_url_var.get(), "Prediction API"),
        )
        self.open_hosting_api_btn.grid(row=7, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="Hosting Status:").grid(row=8, column=0, sticky="nw", padx=5, pady=5)
        self.hosting_status_label = ttk.Label(
            hosting_frame,
            textvariable=self.hosting_mode_summary_var,
            wraplength=460,
            justify="left",
        )
        self.hosting_status_label.grid(row=8, column=1, columnspan=3, sticky="w", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Azure MLOps URL:").grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.azure_mlops_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_mlops_url_var,
            width=30,
            state="readonly",
        )
        self.azure_mlops_url_entry.grid(row=9, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_azure_mlops_btn = ttk.Button(
            hosting_frame,
            text="Open MLOps",
            command=lambda: self.open_url_value(self.azure_mlops_url_var.get(), "Azure MLOps dashboard"),
        )
        self.open_azure_mlops_btn.grid(row=9, column=3, padx=5, pady=5)

        ttk.Label(hosting_frame, text="Azure LLMOps URL:").grid(row=10, column=0, sticky="w", padx=5, pady=5)
        self.azure_llmops_url_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.azure_llmops_url_var,
            width=30,
            state="readonly",
        )
        self.azure_llmops_url_entry.grid(row=10, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.open_azure_llmops_btn = ttk.Button(
            hosting_frame,
            text="Open LLMOps",
            command=lambda: self.open_url_value(self.azure_llmops_url_var.get(), "Azure LLMOps dashboard"),
        )
        self.open_azure_llmops_btn.grid(row=10, column=3, padx=5, pady=5)

        ttk.Separator(hosting_frame, orient="horizontal").grid(
            row=11,
            column=0,
            columnspan=4,
            sticky="ew",
            padx=5,
            pady=(4, 8),
        )

        ttk.Label(hosting_frame, text="MLflow Enabled:").grid(row=12, column=0, sticky="w", padx=5, pady=5)
        self.mlflow_enabled_var = tk.BooleanVar(value=True)
        self.mlflow_enabled_check = ttk.Checkbutton(
            hosting_frame,
            variable=self.mlflow_enabled_var,
            command=self.on_mlflow_enabled_change,
        )
        self.mlflow_enabled_check.grid(row=12, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(hosting_frame, text="MLflow Backend:").grid(row=12, column=2, sticky="w", padx=5, pady=5)
        self.mlflow_backend_var = tk.StringVar(value="local")
        self.mlflow_backend_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.mlflow_backend_var,
            state="readonly",
            values=["local", "azure", "custom_uri"],
            width=15,
        )
        self.mlflow_backend_combo.grid(row=12, column=3, sticky="ew", padx=5, pady=5)
        self.mlflow_backend_combo.bind("<<ComboboxSelected>>", self.on_mlflow_backend_change)

        ttk.Label(hosting_frame, text="Tracking URI:").grid(row=13, column=0, sticky="w", padx=5, pady=5)
        self.mlflow_tracking_uri_var = tk.StringVar(value=self.local_mlflow_tracking_uri)
        self.mlflow_tracking_uri_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.mlflow_tracking_uri_var,
            width=30,
            state="readonly",
        )
        self.mlflow_tracking_uri_entry.grid(row=13, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Experiment Name:").grid(row=14, column=0, sticky="w", padx=5, pady=5)
        self.mlflow_experiment_var = tk.StringVar(value="")
        self.mlflow_experiment_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.mlflow_experiment_var,
            width=24,
        )
        self.mlflow_experiment_entry.grid(row=14, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Registered Model:").grid(row=14, column=2, sticky="w", padx=5, pady=5)
        self.mlflow_registered_model_var = tk.StringVar(value="")
        self.mlflow_registered_model_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.mlflow_registered_model_var,
            width=24,
        )
        self.mlflow_registered_model_entry.grid(row=14, column=3, sticky="ew", padx=5, pady=5)

        self.open_mlflow_btn = ttk.Button(
            hosting_frame,
            text="Open Dashboard",
            command=self.open_mlflow_console,
        )
        self.open_mlflow_btn.grid(row=15, column=0, columnspan=2, sticky="ew", padx=5, pady=8)

        self.register_model_btn = ttk.Button(
            hosting_frame,
            text="Register Last Model",
            command=self.register_last_model_version,
        )
        self.register_model_btn.grid(row=15, column=2, columnspan=2, sticky="ew", padx=5, pady=8)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        self._last_local_device = self.local_device_var.get()
        self.on_mlflow_backend_change()
        self.on_mlflow_enabled_change()
        self.on_training_strategy_change()
        self.on_train_mode_change()
        self.on_hosting_mode_change()

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

    # --- GitHub Repos/Branches Methods ---
    def start_repo_thread(self):
        token = self.github_key_entry.get().strip()
        if not token:
            messagebox.showwarning("Warning", "Please enter a GitHub Personal Access Token.")
            return
        
        self.status_var.set("Loading repositories...")
        self.load_repos_btn.config(state="disabled")
        threading.Thread(target=self.fetch_repos, args=(token,), daemon=True).start()

    def fetch_repos(self, token):
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get("https://api.github.com/user/repos?per_page=100", headers=headers)
            response.raise_for_status()
            repos = response.json()
            repo_names = [repo['full_name'] for repo in repos]
            self.root.after(0, self.update_repo_combo, repo_names)
        except Exception as e:
            self.root.after(0, self.show_error, f"Failed to load repos: {e}")
        finally:
            self.root.after(0, lambda: self.load_repos_btn.config(state="normal"))

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
        threading.Thread(target=self.fetch_branches, args=(token, repo_name), daemon=True).start()

    def fetch_branches(self, token, repo_name):
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(f"https://api.github.com/repos/{repo_name}/branches", headers=headers)
            response.raise_for_status()
            branches = response.json()
            branch_names = [branch['name'] for branch in branches]
            self.root.after(0, self.update_branch_combo, branch_names)
        except Exception as e:
            self.root.after(0, self.show_error, f"Failed to load branches: {e}")

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

    def on_hosting_mode_change(self):
        if not clean_optional_string(self.azure_host_sub_var.get()):
            self.azure_host_sub_var.set(clean_optional_string(self.azure_sub_entry.get()))
        if not clean_optional_string(self.azure_host_tenant_var.get()):
            self.azure_host_tenant_var.set(clean_optional_string(self.azure_tenant_entry.get()))
        if not clean_optional_string(self.azure_host_compute_var.get()):
            self.azure_host_compute_var.set(clean_optional_string(self.azure_compute_var.get()) or "cpu")

        is_azure = self.hosting_mode_var.get().strip() == "azure"
        azure_widgets = [
            self.azure_host_sub_label,
            self.azure_host_sub_entry,
            self.azure_host_tenant_label,
            self.azure_host_tenant_entry,
            self.azure_host_compute_label,
            self.azure_host_compute_combo,
        ]
        for widget in azure_widgets:
            if is_azure:
                widget.grid()
            else:
                widget.grid_remove()

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
            process = self.hosting_process
        self.host_service_btn.config(state="normal")
        local_running = process is not None and process.poll() is None
        self.stop_hosting_btn.config(state="normal" if local_running else "disabled")

    def is_iis_available(self) -> bool:
        if os.name != "nt":
            return False
        windir = os.environ.get("WINDIR", r"C:\Windows")
        return Path(windir, "System32", "inetsrv", "appcmd.exe").exists()

    def find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def sanitize_azure_name(self, raw_value: str, max_length: int = 32) -> str:
        cleaned = re.sub(r"[^a-z0-9-]", "-", clean_optional_string(raw_value).lower())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        cleaned = cleaned[:max_length].strip("-")
        return cleaned or "log-monitor"

    def build_azure_workspace_id(self, sub_id: str) -> str:
        return (
            f"/subscriptions/{sub_id}/resourceGroups/{self.RESOURCE_GROUP}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{self.WORKSPACE_NAME}"
        )

    def build_azure_studio_url(self, sub_id: str) -> str:
        clean_sub_id = clean_optional_string(sub_id)
        if not clean_sub_id:
            return ""
        wsid = self.build_azure_workspace_id(clean_sub_id)
        return f"https://ml.azure.com/experiments?wsid={url_quote(wsid, safe='')}"

    def build_azure_dashboard_urls(self, sub_id: str) -> tuple[str, str]:
        clean_sub_id = clean_optional_string(sub_id)
        if not clean_sub_id:
            return "", ""
        wsid = self.build_azure_workspace_id(clean_sub_id)
        encoded_wsid = url_quote(wsid, safe="")
        mlops_url = f"https://ml.azure.com/models?wsid={encoded_wsid}"
        llmops_url = f"https://ml.azure.com/experiments?wsid={encoded_wsid}"
        return mlops_url, llmops_url

    def get_azure_host_instance_candidates(self, azure_compute: str) -> list[str]:
        compute_mode = clean_optional_string(azure_compute).lower() or "cpu"
        if compute_mode == "gpu":
            candidates = [
                "Standard_NC4as_T4_v3",
                "Standard_NC6s_v3",
            ]
        else:
            candidates = [
                "Standard_D2as_v4",
                "Standard_DS2_v2",
                "Standard_DS1_v2",
                "Standard_F2s_v2",
                "Standard_DS3_v2",
            ]

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
        return unique_candidates

    def is_azure_quota_error(self, exc: Exception) -> bool:
        message = clean_optional_string(str(exc)).lower()
        quota_signals = [
            "not enough quota available",
            "quota",
            "additional needed",
            "vmsize",
        ]
        return any(signal in message for signal in quota_signals)

    def format_azure_hosting_error(self, exc: Exception, attempted_instance_types: list[str] | None = None) -> str:
        message = clean_optional_string(str(exc))
        if self.is_azure_quota_error(exc):
            attempted_text = ", ".join(attempted_instance_types or [])
            guidance = (
                "Azure hosting failed because this subscription does not have enough quota in "
                f"`eastus` for the online endpoint VM sizes we tried.\n\n"
            )
            if attempted_text:
                guidance += f"Tried instance types:\n- {attempted_text.replace(', ', chr(10) + '- ')}\n\n"
            guidance += (
                "What you can do next:\n"
                "- Request a quota increase in Azure for one of those VM families.\n"
                "- Use a different Azure subscription with available quota.\n"
                "- Change the workspace region from `eastus` if you want to target a region with quota.\n"
            )
            return guidance
        return message

    def refresh_azure_dashboard_links(self):
        sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
        mlops_url, llmops_url = self.build_azure_dashboard_urls(sub_id)
        self.azure_mlops_url_var.set(mlops_url)
        self.azure_llmops_url_var.set(llmops_url)

    def start_local_mlflow_ui(self, tracking_uri: str = "") -> str:
        if mlflow is None:
            raise RuntimeError("The `mlflow` package is not available in this Python environment.")
        backend_store_uri = clean_optional_string(tracking_uri) or self.local_mlflow_tracking_uri
        if backend_store_uri.startswith("file://"):
            backend_store_uri = backend_store_uri[7:]
        if self.mlflow_ui_process is None or self.mlflow_ui_process.poll() is not None:
            self.mlflow_ui_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "ui",
                    "--backend-store-uri",
                    backend_store_uri,
                    "--port",
                    "5001",
                ],
                cwd=self.project_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            if self.mlflow_ui_process.poll() is not None:
                raise RuntimeError("Failed to start the local MLflow UI process.")
        return "http://127.0.0.1:5001"

    def open_hosting_dashboard_on_success(self, hosting_mode: str, mlops_url: str = "") -> bool:
        try:
            mode = clean_optional_string(hosting_mode) or clean_optional_string(self.hosting_mode_var.get())
            if mode == "azure":
                dashboard_url = clean_optional_string(mlops_url) or clean_optional_string(self.azure_mlops_url_var.get())
                if dashboard_url:
                    webbrowser.open(dashboard_url)
                    return True
                sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
                studio_url = self.build_azure_studio_url(sub_id)
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

    def save_last_hosting_metadata(self, payload: dict) -> str:
        target = Path(self.project_dir) / "outputs" / "last_hosting.json"
        write_json(str(target), payload)
        return str(target)

    def read_last_hosting_metadata(self) -> dict:
        target = Path(self.project_dir) / "outputs" / "last_hosting.json"
        if not target.exists():
            return {}
        return read_json(str(target)) or {}

    def get_training_metadata_search_roots(self) -> list[Path]:
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

        add_root(Path(self.project_dir) / "outputs")
        add_root(Path(self.project_dir) / "downloaded_model")

        hosted_model_value = clean_optional_string(self.hosted_model_path_var.get())
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

    def describe_training_metadata_search_roots(self) -> list[str]:
        return [str(path) for path in self.get_training_metadata_search_roots()]

    def _dashboard_value_html(self, value) -> str:
        text = clean_optional_string(value)
        return html.escape(text) if text else "<span class='muted'>Not available</span>"

    def _dashboard_link_html(self, url: str, label: str | None = None) -> str:
        clean_url = clean_optional_string(url)
        if not clean_url:
            return "<span class='muted'>Not available</span>"
        link_label = clean_optional_string(label) or clean_url
        return (
            f"<a href=\"{html.escape(clean_url, quote=True)}\" target=\"_blank\" rel=\"noreferrer\">"
            f"{html.escape(link_label)}</a>"
        )

    def build_local_dashboard_html(self, launch_live_console: bool = True) -> str:
        hosting_meta = self.read_last_hosting_metadata()
        training_meta = self.find_latest_training_mlflow_metadata() or {}

        api_url = clean_optional_string(hosting_meta.get("api_url")) or clean_optional_string(self.hosting_api_url_var.get())
        health_url = clean_optional_string(hosting_meta.get("health_url"))
        api_browser_url = api_url[:-8] if api_url.endswith("/predict") else api_url

        backend = self.mlflow_backend_var.get().strip() or "local"
        tracking_uri = clean_optional_string(training_meta.get("tracking_uri")) or clean_optional_string(self.mlflow_tracking_uri_var.get())
        experiment_name = clean_optional_string(training_meta.get("experiment_name")) or clean_optional_string(self.mlflow_experiment_var.get())
        run_id = clean_optional_string(training_meta.get("run_id"))
        model_uri = clean_optional_string(training_meta.get("model_uri"))
        created_at = clean_optional_string(training_meta.get("created_at")) or clean_optional_string(hosting_meta.get("created_at"))
        test_metrics = training_meta.get("test_metrics") if isinstance(training_meta.get("test_metrics"), dict) else {}
        selection_summary = training_meta.get("selection_summary") if isinstance(training_meta.get("selection_summary"), dict) else {}
        metadata_path = clean_optional_string(training_meta.get("_metadata_path"))
        metadata_search_roots = self.describe_training_metadata_search_roots()
        metadata_note = ""
        if not training_meta:
            metadata_note = (
                "Training metadata was not found. The dashboard looked under: "
                + ", ".join(metadata_search_roots)
            )

        console_url = ""
        console_note = ""
        if backend == "local":
            if mlflow is None:
                console_note = "The local MLflow UI is unavailable because the `mlflow` package is not installed in this Python environment."
            else:
                console_url = "http://127.0.0.1:5001"
                if launch_live_console:
                    try:
                        console_url = self.start_local_mlflow_ui(self.local_mlflow_tracking_uri)
                    except Exception as exc:
                        console_url = ""
                        console_note = str(exc)
        elif backend == "custom_uri":
            if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
                console_url = tracking_uri
            else:
                console_note = "The custom tracking URI is not an HTTP URL, so it cannot be opened in the browser."
        else:
            sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
            console_url = self.build_azure_studio_url(sub_id)
            if not console_url:
                console_note = "Azure dashboard URL is unavailable because the Azure subscription ID is empty."

        metrics_html = ""
        for key, label in [
            ("accuracy", "Accuracy"),
            ("weighted_precision", "Weighted Precision"),
            ("weighted_recall", "Weighted Recall"),
            ("weighted_f1", "Weighted F1"),
            ("loss", "Loss"),
        ]:
            metric_value = test_metrics.get(key)
            if metric_value is None:
                rendered = "Not available"
            else:
                try:
                    rendered = f"{float(metric_value):.4f}"
                except Exception:
                    rendered = str(metric_value)
            metrics_html += (
                "<div class='metric'>"
                f"<span>{html.escape(label)}</span>"
                f"<strong>{html.escape(rendered)}</strong>"
                "</div>"
            )

        best_config = selection_summary.get("best_config") if isinstance(selection_summary.get("best_config"), dict) else {}
        selection_html = ""
        for key in ("train_mode", "epochs", "batch_size", "learning_rate", "weight_decay", "max_length"):
            value = best_config.get(key)
            if value not in (None, ""):
                selection_html += (
                    "<div class='kv'>"
                    f"<span>{html.escape(key.replace('_', ' ').title())}</span>"
                    f"<strong>{html.escape(str(value))}</strong>"
                    "</div>"
                )
        if not selection_html:
            selection_html = "<p class='muted'>No selected training configuration was recorded.</p>"

        payload_example = json.dumps({"errorMessage": ""}, indent=2)
        response_example = json.dumps({"prediction": ""}, indent=2)
        training_json = json.dumps(training_meta, indent=2, sort_keys=True)
        hosting_json = json.dumps(hosting_meta, indent=2, sort_keys=True)
        tracking_action = (
            f"<a class='action' href='{html.escape(console_url, quote=True)}' target='_blank' rel='noreferrer'>Open Tracking Console</a>"
            if console_url else ""
        )
        api_action = (
            f"<a class='action' href='{html.escape(api_browser_url, quote=True)}' target='_blank' rel='noreferrer'>Open Local API Home</a>"
            if api_browser_url else ""
        )
        health_action = (
            f"<a class='action' href='{html.escape(health_url, quote=True)}' target='_blank' rel='noreferrer'>Open Health Check</a>"
            if health_url else ""
        )
        console_note_html = f"<p class='muted' style='margin-top:16px'>{html.escape(console_note)}</p>" if console_note else ""
        metadata_note_html = f"<p class='muted' style='margin-top:12px'>{html.escape(metadata_note)}</p>" if metadata_note else ""

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log Monitor Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --ink: #1f2328;
      --muted: #6b7280;
      --line: #dfd4c4;
      --accent: #0f766e;
      --accent-2: #a16207;
      --shadow: 0 20px 45px rgba(69, 52, 30, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 32%),
        radial-gradient(circle at top right, rgba(161, 98, 7, 0.16), transparent 28%),
        linear-gradient(180deg, #f7f2eb 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,250,243,0.88));
      border: 1px solid rgba(223, 212, 196, 0.9);
      border-radius: 24px;
      padding: 28px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent);
      margin: 0 0 10px;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-weight: 600;
    }}
    h1 {{
      font-size: clamp(32px, 5vw, 52px);
      line-height: 1.02;
      margin-bottom: 12px;
    }}
    p {{
      margin: 0;
      line-height: 1.6;
    }}
    .hero p {{
      max-width: 740px;
      color: #394150;
      font-size: 18px;
    }}
    .stack {{
      display: grid;
      gap: 18px;
      margin-top: 22px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
      margin-top: 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      font-size: 24px;
      margin-bottom: 14px;
    }}
    .panel h3 {{
      font-size: 18px;
      margin-bottom: 10px;
      margin-top: 16px;
    }}
    .kv {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 10px 0;
      border-bottom: 1px solid rgba(223, 212, 196, 0.8);
    }}
    .kv:last-child {{
      border-bottom: 0;
      padding-bottom: 0;
    }}
    .kv span {{
      color: var(--muted);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .metric {{
      border: 1px solid rgba(15, 118, 110, 0.18);
      background: rgba(15, 118, 110, 0.05);
      border-radius: 16px;
      padding: 14px;
      display: grid;
      gap: 6px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .metric strong {{
      font-size: 24px;
      font-weight: 600;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .action {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 11px 16px;
      border-radius: 999px;
      border: 1px solid rgba(15, 118, 110, 0.22);
      background: white;
      color: var(--ink);
      text-decoration: none;
      font-size: 14px;
    }}
    .action:hover {{
      border-color: var(--accent);
    }}
    .muted {{
      color: var(--muted);
    }}
    pre {{
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 16px;
      background: #201c1a;
      color: #f5efe6;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.55;
    }}
    details {{
      margin-top: 14px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent-2);
      font-weight: 600;
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">Log Monitor</p>
      <h1>Local MLOps and LLMOps Dashboard</h1>
      <p>This page summarizes the most recent hosted API, training run, and LLM data-preparation lineage available on this machine.</p>
      <div class="actions">
        {tracking_action}
        {api_action}
        {health_action}
      </div>
      {console_note_html}
    </section>

    <div class="grid">
      <section class="panel">
        <h2>Hosted API</h2>
        <div class="kv"><span>Prediction Endpoint</span><strong>{self._dashboard_link_html(api_url)}</strong></div>
        <div class="kv"><span>Browser URL</span><strong>{self._dashboard_link_html(api_browser_url)}</strong></div>
        <div class="kv"><span>Health Check</span><strong>{self._dashboard_link_html(health_url)}</strong></div>
        <div class="kv"><span>Hosting Mode</span><strong>{self._dashboard_value_html(hosting_meta.get("mode"))}</strong></div>
        <div class="kv"><span>Generated</span><strong>{self._dashboard_value_html(created_at)}</strong></div>
        <h3>Prediction Contract</h3>
        <pre>{html.escape(payload_example)}</pre>
        <pre>{html.escape(response_example)}</pre>
      </section>

      <section class="panel">
        <h2>MLOps</h2>
        <div class="kv"><span>Backend</span><strong>{self._dashboard_value_html(backend)}</strong></div>
        <div class="kv"><span>Experiment</span><strong>{self._dashboard_value_html(experiment_name)}</strong></div>
        <div class="kv"><span>Tracking URI</span><strong>{self._dashboard_value_html(tracking_uri)}</strong></div>
        <div class="kv"><span>Run ID</span><strong>{self._dashboard_value_html(run_id)}</strong></div>
        <div class="kv"><span>Model URI</span><strong>{self._dashboard_value_html(model_uri)}</strong></div>
        <div class="kv"><span>Metadata File</span><strong>{self._dashboard_value_html(metadata_path)}</strong></div>
        <div class="metric-grid">{metrics_html}</div>
        {metadata_note_html}
      </section>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>Selected Training Config</h2>
        {selection_html}
      </section>

      <section class="panel">
        <h2>LLMOps</h2>
        <div class="kv"><span>LLM Model</span><strong>{self._dashboard_value_html(training_meta.get("llm_model"))}</strong></div>
        <div class="kv"><span>Prompt Hash</span><strong>{self._dashboard_value_html(training_meta.get("prompt_hash"))}</strong></div>
        <div class="kv"><span>Pipeline ID</span><strong>{self._dashboard_value_html(training_meta.get("pipeline_id"))}</strong></div>
        <div class="kv"><span>Parent Run ID</span><strong>{self._dashboard_value_html(training_meta.get("parent_run_id"))}</strong></div>
        <div class="kv"><span>Data Prep Run ID</span><strong>{self._dashboard_value_html(training_meta.get("data_prep_run_id"))}</strong></div>
        <div class="kv"><span>Data Prep Experiment</span><strong>{self._dashboard_value_html(training_meta.get("data_prep_experiment_name"))}</strong></div>
      </section>
    </div>

    <div class="stack">
      <section class="panel">
        <h2>Training Metadata</h2>
        <details open>
          <summary>Show JSON</summary>
          <pre>{html.escape(training_json)}</pre>
        </details>
      </section>

      <section class="panel">
        <h2>Hosting Metadata</h2>
        <details>
          <summary>Show JSON</summary>
          <pre>{html.escape(hosting_json)}</pre>
        </details>
      </section>
    </div>
  </main>
</body>
</html>
"""

    def open_local_dashboard_page(self, launch_live_console: bool = True) -> str:
        output_dir = Path(self.project_dir) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = output_dir / "local_dashboard.html"
        dashboard_path.write_text(
            self.build_local_dashboard_html(launch_live_console=launch_live_console),
            encoding="utf-8",
        )
        dashboard_uri = dashboard_path.resolve().as_uri()
        webbrowser.open(dashboard_uri)
        return dashboard_uri

    def ensure_azure_workspace(self, sub_id: str, tenant_id: str) -> MLClient:
        resource_group = self.RESOURCE_GROUP
        workspace_name = self.WORKSPACE_NAME

        self.root.after(0, lambda: self.status_var.set("Please log in to Azure in your web browser..."))
        credential = InteractiveBrowserCredential(tenant_id=tenant_id)
        ml_client = MLClient(credential, sub_id, resource_group, workspace_name)
        resource_client = ResourceManagementClient(credential, sub_id)

        self.root.after(0, lambda: self.status_var.set("Checking Azure Resource Group..."))
        try:
            resource_client.resource_groups.get(resource_group)
        except Exception:
            self.root.after(0, lambda: self.status_var.set("Creating Azure Resource Group..."))
            resource_client.resource_groups.create_or_update(resource_group, {"location": "eastus"})

        self.root.after(0, lambda: self.status_var.set("Verifying Azure ML registration..."))
        resource_client.providers.register("Microsoft.MachineLearningServices")
        while True:
            provider_info = resource_client.providers.get("Microsoft.MachineLearningServices")
            if provider_info.registration_state == "Registered":
                break
            self.root.after(0, lambda: self.status_var.set("Activating Azure ML services..."))
            time.sleep(10)

        self.root.after(0, lambda: self.status_var.set("Ensuring Azure ML Workspace exists..."))
        try:
            ml_client.workspaces.get(workspace_name)
        except Exception:
            self.root.after(0, lambda: self.status_var.set("Creating Azure ML Workspace..."))
            workspace = Workspace(name=workspace_name, location="eastus")
            ml_client.workspaces.begin_create(workspace).result()

        return ml_client

    def start_hosting_thread(self):
        model_path = clean_optional_string(self.hosted_model_path_var.get())
        if not model_path:
            messagebox.showwarning("Hosting", "Please select the generated model directory first.")
            return

        try:
            resolved_model_dir = discover_model_dir(model_path)
        except Exception as exc:
            messagebox.showerror("Hosting", f"Could not locate a saved model in that path.\n\n{exc}")
            return

        self.hosted_model_path_var.set(resolved_model_dir)
        self.hosting_api_url_var.set("")
        self.hosting_mode_summary_var.set("")

        if not self.begin_hosting_action():
            messagebox.showwarning("Hosting In Progress", "A hosting workflow is already running.")
            return

        hosting_mode = self.hosting_mode_var.get().strip() or "local"
        if hosting_mode == "azure":
            sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
            tenant_id = clean_optional_string(self.azure_host_tenant_var.get()) or clean_optional_string(self.azure_tenant_entry.get())
            azure_compute = clean_optional_string(self.azure_host_compute_var.get()) or "cpu"
            self.azure_host_sub_var.set(sub_id)
            self.azure_host_tenant_var.set(tenant_id)
            self.azure_host_compute_var.set(azure_compute)
            self.refresh_azure_dashboard_links()

            if not sub_id:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please provide your Azure Subscription ID for hosting.")
                return
            if not tenant_id:
                self.finish_hosting_action()
                messagebox.showwarning("Hosting", "Please provide your Azure Tenant ID for hosting.")
                return

            threading.Thread(
                target=self.run_azure_hosting,
                args=(resolved_model_dir, sub_id, tenant_id, azure_compute),
                daemon=True,
            ).start()
            return

        threading.Thread(
            target=self.run_local_hosting,
            args=(resolved_model_dir,),
            daemon=True,
        ).start()

    def stop_hosting(self):
        process = None
        with self.hosting_state_lock:
            process = self.hosting_process

        if process is None or process.poll() is not None:
            self.stop_hosting_btn.config(state="disabled")
            self.status_var.set("No local hosted API is currently running.")
            return

        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            with self.hosting_state_lock:
                self.hosting_process = None
            self.hosting_api_url_var.set("")
            self.hosting_mode_summary_var.set("Local prediction API stopped.")
            self.stop_hosting_btn.config(state="disabled")
            self.status_var.set("Local hosted API stopped.")

    def run_local_hosting(self, model_dir: str):
        process = None
        try:
            serve_script = os.path.join(self.project_dir, "serve_model.py")
            if not os.path.exists(serve_script):
                raise FileNotFoundError("Could not find 'serve_model.py' in the app directory.")

            existing_process = None
            with self.hosting_state_lock:
                existing_process = self.hosting_process
            if existing_process and existing_process.poll() is None:
                existing_process.terminate()
                try:
                    existing_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    existing_process.kill()

            port = self.find_free_port()
            host = "127.0.0.1"
            api_url = f"http://{host}:{port}/predict"
            health_url = f"http://{host}:{port}/health"
            self.root.after(0, lambda: self.status_var.set("Starting local prediction API..."))

            process = subprocess.Popen(
                [sys.executable, serve_script, "--model-dir", model_dir, "--host", host, "--port", str(port)],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self.hosting_state_lock:
                self.hosting_process = process

            ready = False
            deadline = time.time() + 60
            while time.time() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else ""
                    raise RuntimeError(f"Local hosting exited unexpectedly.\n\n{output}")
                try:
                    response = requests.get(health_url, timeout=2)
                    if response.ok:
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(1)

            if not ready:
                raise TimeoutError("Timed out waiting for the local prediction API to become ready.")

            host_note = (
                "IIS was detected on this machine; this implementation is serving the model directly "
                "from the app host for a simpler local API workflow."
                if self.is_iis_available()
                else "Serving the model directly from this machine."
            )
            summary = (
                f"Local prediction API is running.\n"
                f"POST {api_url}\n"
                f"Body: {{\"errorMessage\": \"...\"}}\n"
                f"Response: {{\"prediction\": \"...\"}}\n"
                f"{host_note}"
            )
            metadata_path = self.save_last_hosting_metadata(
                {
                    "mode": "local",
                    "model_dir": model_dir,
                    "api_url": api_url,
                    "health_url": health_url,
                    "created_at": now_utc_iso(),
                }
            )
            self.root.after(0, lambda: self.hosting_api_url_var.set(api_url))
            self.root.after(0, lambda: self.hosting_mode_summary_var.set(summary))
            self.root.after(0, lambda: self.status_var.set("Local prediction API is ready."))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Hosting Ready",
                    (
                        f"Local prediction API is ready.\n\n"
                        f"Prediction URL:\n{api_url}\n\n"
                        "The app will try to open the dashboard in your browser.\n\n"
                        f"Hosting metadata saved to:\n{metadata_path}"
                    ),
                ),
            )
            self.root.after(100, lambda: self.open_hosting_dashboard_on_success("local"))
        except Exception as exc:
            if process and process.poll() is None:
                process.terminate()
            with self.hosting_state_lock:
                if self.hosting_process is process:
                    self.hosting_process = None
            self.root.after(0, lambda err=str(exc): messagebox.showerror("Hosting Error", err))
            self.root.after(0, lambda: self.status_var.set("Local hosting failed."))
        finally:
            self.root.after(0, self.finish_hosting_action)

    def run_azure_hosting(self, model_dir: str, sub_id: str, tenant_id: str, azure_compute: str):
        attempted_instance_types: list[str] = []
        try:
            self.root.after(0, lambda: self.status_var.set("Preparing Azure hosting..."))
            ml_client = self.ensure_azure_workspace(sub_id, tenant_id)

            timestamp = int(time.time())
            model_name = self.sanitize_azure_name(f"log-monitor-model-{Path(model_dir).name}-{timestamp}")
            endpoint_name = self.sanitize_azure_name(f"log-monitor-endpoint-{timestamp}")
            deployment_name = "blue"
            env_name = self.sanitize_azure_name(f"log-monitor-inference-env-{timestamp}")
            instance_candidates = self.get_azure_host_instance_candidates(azure_compute)
            selected_instance_type = ""

            self.root.after(0, lambda: self.status_var.set("Registering model in Azure ML..."))
            model_asset = Model(
                path=model_dir.replace("\\", "/"),
                name=model_name,
                type=AssetTypes.CUSTOM_MODEL,
                description="Log Monitor generated DeBERTa model",
            )
            registered_model = ml_client.models.create_or_update(model_asset)

            self.root.after(0, lambda: self.status_var.set("Creating Azure inference environment..."))
            environment = Environment(
                name=env_name,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
                conda_file=os.path.join(self.project_dir, "azure_inference_conda.yml"),
                description="Inference environment for Log Monitor hosted API",
            )
            environment = ml_client.environments.create_or_update(environment)

            self.root.after(0, lambda: self.status_var.set("Creating Azure online endpoint..."))
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                auth_mode="key",
                description="Hosted prediction API for Log Monitor",
            )
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()

            last_deployment_error = None
            for instance_type in instance_candidates:
                attempted_instance_types.append(instance_type)
                self.root.after(
                    0,
                    lambda size=instance_type: self.status_var.set(
                        f"Deploying model to Azure endpoint ({size})..."
                    ),
                )
                try:
                    deployment = ManagedOnlineDeployment(
                        name=deployment_name,
                        endpoint_name=endpoint_name,
                        model=registered_model,
                        environment=environment,
                        code_configuration=CodeConfiguration(
                            code=self.project_dir,
                            scoring_script="azure_score.py",
                        ),
                        instance_type=instance_type,
                        instance_count=1,
                    )
                    ml_client.online_deployments.begin_create_or_update(deployment).result()
                    selected_instance_type = instance_type
                    break
                except Exception as exc:
                    last_deployment_error = exc
                    if self.is_azure_quota_error(exc) and instance_type != instance_candidates[-1]:
                        print(
                            f"[AZURE-HOST] Instance type {instance_type} is unavailable due to quota. "
                            "Trying the next fallback size."
                        )
                        continue
                    raise

            if not selected_instance_type:
                if last_deployment_error is not None:
                    raise last_deployment_error
                raise RuntimeError("Azure deployment did not complete and no scoring instance was selected.")

            endpoint.traffic = {deployment_name: 100}
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            endpoint_details = ml_client.online_endpoints.get(endpoint_name)
            scoring_uri = clean_optional_string(getattr(endpoint_details, "scoring_uri", ""))
            if not scoring_uri:
                raise RuntimeError("Azure deployment completed but no scoring URI was returned.")

            mlops_url, llmops_url = self.build_azure_dashboard_urls(sub_id)
            metadata_path = self.save_last_hosting_metadata(
                {
                    "mode": "azure",
                    "model_dir": model_dir,
                    "endpoint_name": endpoint_name,
                    "deployment_name": deployment_name,
                    "instance_type": selected_instance_type,
                    "api_url": scoring_uri,
                    "azure_subscription_id": sub_id,
                    "azure_tenant_id": tenant_id,
                    "azure_compute": azure_compute,
                    "mlops_url": mlops_url,
                    "llmops_url": llmops_url,
                    "created_at": now_utc_iso(),
                }
            )

            summary = (
                f"Azure prediction endpoint is ready.\n"
                f"POST {scoring_uri}\n"
                f"Instance Type: {selected_instance_type}\n"
                f"Body: {{\"errorMessage\": \"...\"}}\n"
                f"Response: {{\"prediction\": \"...\"}}\n"
                f"Azure ML online endpoints typically require endpoint key authentication."
            )

            self.root.after(0, lambda: self.azure_hosted_endpoint_name_var.set(endpoint_name))
            self.root.after(0, lambda: self.hosting_api_url_var.set(scoring_uri))
            self.root.after(0, lambda: self.azure_mlops_url_var.set(mlops_url))
            self.root.after(0, lambda: self.azure_llmops_url_var.set(llmops_url))
            self.root.after(0, lambda: self.hosting_mode_summary_var.set(summary))
            self.root.after(0, lambda: self.status_var.set("Azure prediction endpoint is ready."))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Hosting Ready",
                    (
                        f"Azure prediction endpoint is ready.\n\n"
                        f"Prediction URL:\n{scoring_uri}\n\n"
                        f"Instance Type:\n{selected_instance_type}\n\n"
                        f"MLOps Dashboard:\n{mlops_url}\n\n"
                        f"LLMOps Dashboard:\n{llmops_url}\n\n"
                        "The app will try to open the MLOps dashboard in your browser.\n\n"
                        f"Hosting metadata saved to:\n{metadata_path}"
                    ),
                ),
            )
            self.root.after(100, lambda: self.open_hosting_dashboard_on_success("azure", mlops_url))
        except Exception as exc:
            error_message = self.format_azure_hosting_error(exc, attempted_instance_types)
            self.root.after(0, lambda err=error_message: messagebox.showerror("Hosting Error", err))
            self.root.after(0, lambda: self.status_var.set("Azure hosting failed."))
        finally:
            self.root.after(0, self.finish_hosting_action)

    def on_app_close(self):
        with self.hosting_state_lock:
            process = self.hosting_process
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
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

    def load_prompt(self):
        prompt_path = os.path.join(self.project_dir, "prompt.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError("Could not find 'prompt.txt' in the application directory.")
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()

    def on_mlflow_enabled_change(self):
        enabled = bool(self.mlflow_enabled_var.get())
        self.mlflow_backend_combo.config(state="readonly" if enabled else "disabled")
        self.mlflow_experiment_entry.config(state="normal" if enabled else "disabled")
        self.mlflow_registered_model_entry.config(state="normal" if enabled else "disabled")
        self.open_mlflow_btn.config(state="normal")
        self.register_model_btn.config(state="normal" if enabled else "disabled")

        if enabled:
            self.on_mlflow_backend_change()
        else:
            self.mlflow_tracking_uri_entry.config(state="disabled")

    def on_mlflow_backend_change(self, event=None):
        if not bool(self.mlflow_enabled_var.get()):
            self.mlflow_tracking_uri_entry.config(state="disabled")
            return

        backend = self.mlflow_backend_var.get().strip() or "local"
        current_uri = clean_optional_string(self.mlflow_tracking_uri_var.get())
        unresolved_azure_placeholder = "(resolved during Azure run)"

        if backend == "local":
            self.mlflow_tracking_uri_var.set(self.local_mlflow_tracking_uri)
            self.mlflow_tracking_uri_entry.config(state="readonly")
            return

        if backend == "azure":
            if self.azure_mlflow_tracking_uri:
                self.mlflow_tracking_uri_var.set(self.azure_mlflow_tracking_uri)
            else:
                self.mlflow_tracking_uri_var.set(unresolved_azure_placeholder)
            self.mlflow_tracking_uri_entry.config(state="readonly")
            return

        if current_uri == unresolved_azure_placeholder:
            self.mlflow_tracking_uri_var.set("")
        self.mlflow_tracking_uri_entry.config(state="normal")

    def _resolve_azure_mlflow_tracking_uri(self, ml_client: MLClient | None) -> str:
        if self.azure_mlflow_tracking_uri:
            return self.azure_mlflow_tracking_uri

        if ml_client is None:
            return ""

        try:
            workspace = ml_client.workspaces.get(self.WORKSPACE_NAME)
            uri = clean_optional_string(getattr(workspace, "mlflow_tracking_uri", ""))
            if uri:
                self.azure_mlflow_tracking_uri = uri
                self.root.after(0, lambda value=uri: self.mlflow_tracking_uri_var.set(value))
            return uri
        except Exception:
            print("[MLOPS] Failed to resolve Azure MLflow tracking URI.")
            traceback.print_exc()
            return ""

    def resolve_mlflow_config(
        self,
        require_tracking_uri: bool,
        ml_client: MLClient | None = None,
        soft_disable: bool = False,
    ):
        enabled = bool(self.mlflow_enabled_var.get())
        backend = self.mlflow_backend_var.get().strip() or "local"
        experiment_name = clean_optional_string(self.mlflow_experiment_var.get())
        registered_model_name = clean_optional_string(self.mlflow_registered_model_var.get())

        config = {
            "enabled": enabled,
            "backend": backend,
            "tracking_uri": "",
            "experiment_name": experiment_name,
            "registered_model_name": registered_model_name,
        }

        if not enabled:
            return config, None

        def soft_disabled(reason: str):
            fallback = dict(config)
            fallback["enabled"] = False
            fallback["tracking_uri"] = ""
            fallback["disabled_reason"] = reason
            print(f"[MLOPS] {reason} Proceeding with MLflow disabled.")
            return fallback, None

        if mlflow is None:
            if soft_disable:
                return soft_disabled("MLflow is enabled in UI, but the `mlflow` package is not available.")
            return config, "MLflow is enabled in UI, but the `mlflow` package is not available."

        if not experiment_name:
            if soft_disable:
                return soft_disabled(
                    (
                        "MLflow is enabled, but `Experiment Name` is empty.\n\n"
                        "Set a non-empty value in Hosting:\n"
                        "- Experiment Name (example: `log-monitor`)"
                    )
                )
            return (
                config,
                (
                    "MLflow is enabled, but `Experiment Name` is empty.\n\n"
                    "Set a non-empty value in Hosting:\n"
                    "- Experiment Name (example: `log-monitor`)"
                ),
            )

        if backend == "local":
            config["tracking_uri"] = self.local_mlflow_tracking_uri
            self.mlflow_tracking_uri_var.set(self.local_mlflow_tracking_uri)
        elif backend == "custom_uri":
            config["tracking_uri"] = clean_optional_string(self.mlflow_tracking_uri_var.get())
            if require_tracking_uri and not config["tracking_uri"]:
                if soft_disable:
                    return soft_disabled("MLflow backend is `custom_uri` but Tracking URI is empty.")
                return config, "MLflow backend is `custom_uri` but Tracking URI is empty."
        elif backend == "azure":
            config["tracking_uri"] = self._resolve_azure_mlflow_tracking_uri(ml_client)
            if not config["tracking_uri"]:
                current_value = clean_optional_string(self.mlflow_tracking_uri_var.get())
                if current_value and "resolved during Azure run" not in current_value:
                    config["tracking_uri"] = current_value
            if require_tracking_uri and not config["tracking_uri"]:
                if soft_disable:
                    return soft_disabled("Azure MLflow URI is not resolved yet. Run Azure auth or provide `custom_uri` backend.")
                return config, "Azure MLflow URI is not resolved yet. Run Azure auth or provide `custom_uri` backend."
        else:
            if soft_disable:
                return soft_disabled(f"Unsupported MLflow backend: {backend}")
            return config, f"Unsupported MLflow backend: {backend}"

        return config, None

    def sidecar_matches_mlflow_target(self, sidecar: dict, mlflow_config: dict) -> bool:
        if not bool(mlflow_config.get("enabled")):
            return True

        current_tracking_uri = clean_optional_string(mlflow_config.get("tracking_uri", ""))
        current_experiment = clean_optional_string(mlflow_config.get("experiment_name", ""))
        recorded_tracking_uri = clean_optional_string(
            sidecar.get("training_tracking_uri")
            or sidecar.get("tracking_uri")
            or sidecar.get("data_prep_tracking_uri")
        )
        recorded_experiment = clean_optional_string(
            sidecar.get("training_experiment_name")
            or sidecar.get("experiment_name")
            or sidecar.get("data_prep_experiment_name")
        )

        if recorded_tracking_uri and current_tracking_uri and recorded_tracking_uri != current_tracking_uri:
            return False
        if recorded_experiment and current_experiment and recorded_experiment != current_experiment:
            return False
        return True

    def build_training_mlflow_env(self, mlflow_config: dict, pipeline_context: dict, run_source: str) -> dict[str, str]:
        env = {key: "" for key in MLOPS_ENV_VARS}

        tags = {
            "run_type": "training",
            "pipeline_id": str(pipeline_context.get("pipeline_id", "")),
            "run_source": run_source,
            "environment_mode": self.train_mode.get().strip() or "unknown",
        }
        data_prep_run_id = clean_optional_string(pipeline_context.get("data_prep_run_id", ""))
        if data_prep_run_id:
            tags["data_prep_run_id"] = data_prep_run_id
        prompt_hash = clean_optional_string(pipeline_context.get("prompt_hash", ""))
        if prompt_hash:
            tags["prompt_hash"] = prompt_hash
        llm_model = clean_optional_string(pipeline_context.get("llm_model", ""))
        if llm_model:
            tags["llm_model"] = llm_model
        data_prep_input_hash = clean_optional_string(pipeline_context.get("input_dataset_hash", ""))
        if data_prep_input_hash:
            tags["data_prep_input_dataset_hash"] = data_prep_input_hash
        data_prep_output_hash = clean_optional_string(pipeline_context.get("output_dataset_hash", ""))
        if data_prep_output_hash:
            tags["data_prep_output_dataset_hash"] = data_prep_output_hash
        data_prep_tracking_uri = clean_optional_string(pipeline_context.get("data_prep_tracking_uri", ""))
        if data_prep_tracking_uri:
            tags["data_prep_tracking_uri"] = data_prep_tracking_uri
        data_prep_experiment_name = clean_optional_string(pipeline_context.get("data_prep_experiment_name", ""))
        if data_prep_experiment_name:
            tags["data_prep_experiment_name"] = data_prep_experiment_name

        env["MLFLOW_PIPELINE_ID"] = str(pipeline_context.get("pipeline_id", ""))
        env["MLFLOW_PARENT_RUN_ID"] = str(pipeline_context.get("parent_run_id", ""))
        env["MLFLOW_RUN_SOURCE"] = str(run_source)
        env["MLFLOW_TAGS_JSON"] = json.dumps(tags)

        enabled = (
            bool(mlflow_config.get("enabled"))
            and mlflow is not None
            and bool(clean_optional_string(mlflow_config.get("tracking_uri", "")))
        )
        env["MLOPS_ENABLED"] = "1" if enabled else "0"
        if not enabled:
            return env

        env["MLFLOW_TRACKING_URI"] = str(mlflow_config["tracking_uri"])
        env["MLFLOW_EXPERIMENT_NAME"] = str(mlflow_config["experiment_name"])
        return env

    def build_shell_export_segment(self, env_map: dict[str, str]) -> str:
        exports: list[str] = []
        for key, value in env_map.items():
            safe_key = clean_optional_string(key)
            if not safe_key:
                continue
            exports.append(f"export {safe_key}={shlex.quote(str(value))}")
        return " && ".join(exports)

    def create_pipeline_parent_run(self, mlflow_config: dict, pipeline_id: str, run_source: str) -> str:
        if not bool(mlflow_config.get("enabled")) or mlflow is None:
            return ""
        tracking_uri = clean_optional_string(mlflow_config.get("tracking_uri"))
        if not tracking_uri:
            return ""

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(mlflow_config["experiment_name"])
            tags = {
                "run_type": "pipeline",
                "pipeline_id": pipeline_id,
                "run_source": run_source,
                "created_by": "app.py",
            }
            with mlflow.start_run(run_name=f"pipeline-{pipeline_id}", tags=tags) as run:
                mlflow.log_dict(
                    {
                        "pipeline_id": pipeline_id,
                        "run_source": run_source,
                        "created_at": now_utc_iso(),
                    },
                    "pipeline_context.json",
                )
                return run.info.run_id
        except Exception:
            print("[MLOPS] Failed to create pipeline parent run.")
            traceback.print_exc()
            return ""

    def prepare_training_pipeline_context(self, csv_path: str, mlflow_config: dict, run_source: str) -> dict:
        sidecar = read_sidecar_for_csv(csv_path) or {}
        pipeline_id = clean_optional_string(sidecar.get("pipeline_id")) or str(uuid.uuid4())
        parent_run_id = clean_optional_string(sidecar.get("parent_run_id"))

        if not self.sidecar_matches_mlflow_target(sidecar, mlflow_config):
            print(
                "[MLOPS] Existing sidecar lineage points to a different MLflow target. "
                "Creating a new pipeline parent run for the current target."
            )
            parent_run_id = ""

        if bool(mlflow_config.get("enabled")) and not parent_run_id:
            parent_run_id = self.create_pipeline_parent_run(mlflow_config, pipeline_id, run_source)

        context = {
            "pipeline_id": pipeline_id,
            "parent_run_id": parent_run_id,
            "data_prep_run_id": clean_optional_string(sidecar.get("data_prep_run_id")),
            "prompt_hash": clean_optional_string(sidecar.get("prompt_hash")),
            "input_dataset_hash": clean_optional_string(sidecar.get("input_dataset_hash")),
            "output_dataset_hash": clean_optional_string(sidecar.get("output_dataset_hash")),
            "llm_model": clean_optional_string(sidecar.get("llm_model")),
            "data_prep_tracking_uri": clean_optional_string(
                sidecar.get("data_prep_tracking_uri") or sidecar.get("tracking_uri")
            ),
            "data_prep_experiment_name": clean_optional_string(
                sidecar.get("data_prep_experiment_name") or sidecar.get("experiment_name")
            ),
        }

        sidecar_payload = {
            "pipeline_id": pipeline_id,
            "parent_run_id": parent_run_id,
            "data_prep_run_id": context["data_prep_run_id"],
            "prompt_hash": context["prompt_hash"],
            "input_dataset_hash": context["input_dataset_hash"],
            "output_dataset_hash": context["output_dataset_hash"],
            "llm_model": context["llm_model"],
            "created_at": clean_optional_string(sidecar.get("created_at")) or now_utc_iso(),
            "tracking_uri": clean_optional_string(sidecar.get("tracking_uri"))
            or clean_optional_string(mlflow_config.get("tracking_uri", "")),
            "experiment_name": clean_optional_string(sidecar.get("experiment_name"))
            or clean_optional_string(mlflow_config.get("experiment_name", "")),
            "data_prep_tracking_uri": context["data_prep_tracking_uri"],
            "data_prep_experiment_name": context["data_prep_experiment_name"],
            "training_tracking_uri": clean_optional_string(mlflow_config.get("tracking_uri", ""))
            or clean_optional_string(sidecar.get("training_tracking_uri", "")),
            "training_experiment_name": clean_optional_string(mlflow_config.get("experiment_name", ""))
            or clean_optional_string(sidecar.get("training_experiment_name", "")),
        }
        try:
            write_sidecar_for_csv(csv_path, sidecar_payload)
        except Exception:
            print("[MLOPS] Failed to persist training sidecar context.")
            traceback.print_exc()
        return context

    def log_data_prep_mlflow(
        self,
        mlflow_config: dict,
        input_path: str,
        output_path: str,
        system_prompt: str,
        llm_model: str,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        input_hash: str,
        output_hash: str,
        usage_totals: dict[str, int],
    ) -> dict:
        payload = {
            "pipeline_id": str(uuid.uuid4()),
            "parent_run_id": "",
            "data_prep_run_id": "",
            "prompt_hash": prompt_sha256(system_prompt),
            "llm_model": clean_optional_string(llm_model),
            "input_dataset_hash": input_hash,
            "output_dataset_hash": output_hash,
            "created_at": now_utc_iso(),
            "tracking_uri": clean_optional_string(mlflow_config.get("tracking_uri", "")),
            "experiment_name": clean_optional_string(mlflow_config.get("experiment_name", "")),
            "data_prep_tracking_uri": clean_optional_string(mlflow_config.get("tracking_uri", "")),
            "data_prep_experiment_name": clean_optional_string(mlflow_config.get("experiment_name", "")),
        }

        if not bool(mlflow_config.get("enabled")):
            return payload
        if mlflow is None:
            print("[MLOPS] Skipping data-prep tracking because `mlflow` is unavailable.")
            return payload
        if not payload["tracking_uri"]:
            print("[MLOPS] Skipping data-prep tracking because tracking URI is unresolved.")
            return payload

        try:
            mlflow.set_tracking_uri(payload["tracking_uri"])
            mlflow.set_experiment(payload["experiment_name"])
            with mlflow.start_run(
                run_name=f"pipeline-{payload['pipeline_id']}",
                tags={
                    "run_type": "pipeline",
                    "pipeline_id": payload["pipeline_id"],
                    "run_source": "data_prep",
                },
            ) as parent_run:
                payload["parent_run_id"] = parent_run.info.run_id

                with mlflow.start_run(
                    run_name="data-prep",
                    nested=True,
                    tags={
                        "run_type": "data_prep",
                        "pipeline_id": payload["pipeline_id"],
                    },
                ) as child_run:
                    payload["data_prep_run_id"] = child_run.info.run_id
                    input_meta = dataframe_metadata(input_df, label_col="class")
                    output_meta = dataframe_metadata(output_df, label_col="class")

                    mlflow.log_param("llm_model", clean_optional_string(llm_model) or "unknown")
                    mlflow.log_param("input_filename", os.path.basename(input_path))
                    mlflow.log_param("output_filename", os.path.basename(output_path))
                    mlflow.log_param("prompt_hash", payload["prompt_hash"])
                    mlflow.log_param("input_dataset_hash", payload["input_dataset_hash"])
                    mlflow.log_param("output_dataset_hash", payload["output_dataset_hash"])

                    mlflow.log_metric("input_rows", int(input_meta.get("row_count", 0)))
                    mlflow.log_metric("output_rows", int(output_meta.get("row_count", 0)))
                    for metric_name, metric_value in usage_totals.items():
                        mlflow.log_metric(metric_name, int(metric_value))

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        prompt_preview_path = os.path.join(tmp_dir, "prompt_preview.txt")
                        with open(prompt_preview_path, "w", encoding="utf-8") as file_obj:
                            file_obj.write(safe_prompt_preview(system_prompt, max_chars=2000))

                        metadata_path = os.path.join(tmp_dir, "data_prep_metadata.json")
                        write_json(
                            metadata_path,
                            {
                                "pipeline_id": payload["pipeline_id"],
                                "input_path": input_path,
                                "output_path": output_path,
                                "prompt_hash": payload["prompt_hash"],
                                "input_dataset_hash": payload["input_dataset_hash"],
                                "output_dataset_hash": payload["output_dataset_hash"],
                                "usage_totals": usage_totals,
                                "input_metadata": input_meta,
                                "output_metadata": output_meta,
                                "created_at": payload["created_at"],
                            },
                        )

                        sample_path = os.path.join(tmp_dir, "output_sample.csv")
                        output_sample_df = dataframe_sample(output_df, max_rows=100, max_cell_chars=200)
                        output_sample_df.to_csv(sample_path, index=False)

                        mlflow.log_artifacts(tmp_dir, artifact_path="data_prep")
        except Exception:
            print("[MLOPS] Data-prep MLflow logging failed.")
            traceback.print_exc()
        return payload

    def find_latest_training_mlflow_metadata(self) -> dict | None:
        candidate_paths: list[Path] = []
        seen: set[str] = set()
        recursive_roots: list[Path] = []

        hosted_model_value = clean_optional_string(self.hosted_model_path_var.get())
        if hosted_model_value:
            try:
                recursive_roots.append(Path(discover_model_dir(hosted_model_value)).resolve())
            except Exception:
                pass

        project_download_root = (Path(self.project_dir) / "downloaded_model").resolve()
        if project_download_root.exists():
            recursive_roots.append(project_download_root)

        for root in self.get_training_metadata_search_roots():
            direct_meta = root / "last_training_mlflow.json"
            if direct_meta.exists():
                key = str(direct_meta.resolve())
                if key not in seen:
                    seen.add(key)
                    candidate_paths.append(direct_meta)

            should_search_recursively = any(root == recursive_root for recursive_root in recursive_roots)
            if root.is_dir() and should_search_recursively:
                try:
                    nested_matches = root.rglob("last_training_mlflow.json")
                except Exception:
                    nested_matches = []
                for match in nested_matches:
                    try:
                        key = str(match.resolve())
                    except Exception:
                        key = str(match)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_paths.append(match)

        if not candidate_paths:
            return None

        latest = max(candidate_paths, key=lambda path: path.stat().st_mtime)
        payload = read_json(str(latest)) or {}
        if payload:
            payload["_metadata_path"] = str(latest)
        return payload if payload else None

    def cache_downloaded_training_mlflow_metadata(self, download_path: str) -> None:
        search_root = Path(download_path)
        if not search_root.exists():
            return
        candidates = list(search_root.rglob("last_training_mlflow.json"))
        if not candidates:
            return

        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        target = Path(self.project_dir) / "outputs" / "last_training_mlflow.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(latest), str(target))
        print(f"[MLOPS] Cached training metadata to {target}")

    def open_mlflow_console(self):
        backend = self.mlflow_backend_var.get().strip() or "local"
        tracking_uri = clean_optional_string(self.mlflow_tracking_uri_var.get())

        if backend == "local":
            try:
                self.open_local_dashboard_page(launch_live_console=True)
            except Exception as exc:
                messagebox.showerror("Dashboard", f"Failed to open the local dashboard.\n\n{exc}")
            return

        if backend == "custom_uri":
            if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
                webbrowser.open(tracking_uri)
            else:
                messagebox.showinfo(
                    "MLflow",
                    "Custom tracking URI is not an HTTP URL. Use your own MLflow UI endpoint or switch backend.",
                )
            return

        sub_id = clean_optional_string(self.azure_host_sub_var.get()) or clean_optional_string(self.azure_sub_entry.get())
        if not sub_id:
            messagebox.showwarning("MLflow", "Azure Subscription ID is required to open Azure ML Studio.")
            return
        studio_url = self.build_azure_studio_url(sub_id)
        webbrowser.open(studio_url)

    def register_last_model_version(self):
        if mlflow is None:
            messagebox.showerror("MLflow", "Model registration requires `mlflow` package availability.")
            return

        metadata = self.find_latest_training_mlflow_metadata()
        if not metadata:
            messagebox.showwarning("MLflow", "No training MLflow metadata file found.")
            return

        run_id = clean_optional_string(metadata.get("run_id"))
        if not run_id:
            messagebox.showerror("MLflow", "Training metadata is missing run_id; cannot register model.")
            return

        tracking_uri = clean_optional_string(metadata.get("tracking_uri"))
        if not tracking_uri:
            config, error = self.resolve_mlflow_config(require_tracking_uri=True)
            if error:
                messagebox.showerror("MLflow", error)
                return
            tracking_uri = clean_optional_string(config.get("tracking_uri"))

        model_name = clean_optional_string(self.mlflow_registered_model_var.get())
        if not model_name:
            messagebox.showerror(
                "MLflow",
                (
                    "`Registered Model` is empty.\n\n"
                    "Set a non-empty model name in Hosting > Registered Model "
                    "before registering a version."
                ),
            )
            return
        model_uri = clean_optional_string(metadata.get("model_uri")) or f"runs:/{run_id}/final_model"

        try:
            mlflow.set_tracking_uri(tracking_uri)
            version = mlflow.register_model(model_uri=model_uri, name=model_name)
            messagebox.showinfo(
                "MLflow",
                (
                    f"Registered model version successfully.\n\n"
                    f"Model: {model_name}\n"
                    f"Version: {version.version}\n"
                    f"Run ID: {run_id}"
                ),
            )
        except Exception as exc:
            messagebox.showerror("MLflow", f"Failed to register model version.\n\n{exc}")

    def on_train_mode_change(self):
        is_azure = self.train_mode.get() == "azure"
        azure_state = "normal" if is_azure else "disabled"
        azure_compute_state = "readonly" if is_azure else "disabled"
        local_state = "disabled" if is_azure else "readonly"

        if is_azure:
            self.azure_sub_label.grid()
            self.azure_sub_entry.grid()
            self.azure_tenant_label.grid()
            self.azure_tenant_entry.grid()
            self.azure_compute_label.grid()
            self.azure_compute_combo.grid()
        else:
            self.azure_sub_label.grid_remove()
            self.azure_sub_entry.grid_remove()
            self.azure_tenant_label.grid_remove()
            self.azure_tenant_entry.grid_remove()
            self.azure_compute_label.grid_remove()
            self.azure_compute_combo.grid_remove()

        self.azure_sub_entry.config(state=azure_state)
        self.azure_tenant_entry.config(state=azure_state)
        self.azure_compute_combo.config(state=azure_compute_state)
        self.local_device_combo.config(state=local_state)

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

    def _parse_numeric_list(self, raw_value: str, parser, field_name: str):
        cleaned = raw_value.strip()
        if not cleaned:
            return []
        parsed_values = []
        for part in cleaned.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                parsed_values.append(parser(token))
            except Exception as exc:
                raise ValueError(f"Invalid value '{token}' in {field_name}: {exc}") from exc
        return parsed_values

    def collect_training_options(self):
        try:
            strategy = (self.training_strategy_var.get().strip() or "default").lower()
            if strategy not in {"default", "tune", "tune_cv"}:
                return None, f"Unsupported training mode: {strategy}"

            epochs = int(self.epochs_var.get().strip())
            batch_size = int(self.batch_size_var.get().strip())
            learning_rate = float(self.learning_rate_var.get().strip())
            weight_decay = float(self.weight_decay_var.get().strip())
            max_length = int(self.max_length_var.get().strip())
            cv_folds = int(self.cv_folds_var.get().strip() or "3")
            max_trials = int(self.max_trials_var.get().strip() or "8")
        except Exception as exc:
            return None, f"Invalid training configuration values: {exc}"

        if epochs < 1:
            return None, "Epochs must be >= 1."
        if batch_size < 1:
            return None, "Batch Size must be >= 1."
        if learning_rate <= 0:
            return None, "Learning Rate must be > 0."
        if weight_decay < 0:
            return None, "Weight Decay must be >= 0."
        if max_length < 16:
            return None, "Max Length must be >= 16."
        if cv_folds < 2:
            return None, "CV Folds must be >= 2."
        if max_trials < 1:
            return None, "Max Trials must be >= 1."

        try:
            tune_lrs = self._parse_numeric_list(self.tune_lrs_var.get(), float, "Tune LRs")
            tune_batch_sizes = self._parse_numeric_list(self.tune_batch_sizes_var.get(), int, "Tune Batch Sizes")
            tune_epochs = self._parse_numeric_list(self.tune_epochs_var.get(), int, "Tune Epochs")
            tune_weight_decays = self._parse_numeric_list(self.tune_weight_decays_var.get(), float, "Tune Weight Decays")
            tune_max_lengths = self._parse_numeric_list(self.tune_max_lengths_var.get(), int, "Tune Max Lengths")
        except ValueError as exc:
            return None, str(exc)

        tune_lrs = [value for value in tune_lrs if value > 0]
        tune_batch_sizes = [value for value in tune_batch_sizes if value > 0]
        tune_epochs = [value for value in tune_epochs if value > 0]
        tune_weight_decays = [value for value in tune_weight_decays if value >= 0]
        tune_max_lengths = [value for value in tune_max_lengths if value >= 16]

        if not tune_lrs:
            tune_lrs = [learning_rate]
        if not tune_batch_sizes:
            tune_batch_sizes = [batch_size]
        if not tune_epochs:
            tune_epochs = [epochs]
        if not tune_weight_decays:
            tune_weight_decays = [weight_decay]
        if not tune_max_lengths:
            tune_max_lengths = [max_length]

        if strategy == "default":
            max_trials = 1
        if strategy != "tune_cv":
            cv_folds = 3

        options = {
            "train_mode": strategy,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "cv_folds": cv_folds,
            "max_trials": max_trials,
            "tune_learning_rates": tune_lrs,
            "tune_batch_sizes": tune_batch_sizes,
            "tune_epochs": tune_epochs,
            "tune_weight_decays": tune_weight_decays,
            "tune_max_lengths": tune_max_lengths,
        }
        return options, None

    def build_train_cli_args(self, training_options: dict) -> list[str]:
        def list_to_csv(values):
            return ",".join(str(value) for value in values)

        return [
            "--train-mode",
            str(training_options["train_mode"]),
            "--epochs",
            str(training_options["epochs"]),
            "--batch-size",
            str(training_options["batch_size"]),
            "--learning-rate",
            str(training_options["learning_rate"]),
            "--weight-decay",
            str(training_options["weight_decay"]),
            "--max-length",
            str(training_options["max_length"]),
            "--val-ratio",
            str(training_options["val_ratio"]),
            "--test-ratio",
            str(training_options["test_ratio"]),
            "--cv-folds",
            str(training_options["cv_folds"]),
            "--max-trials",
            str(training_options["max_trials"]),
            "--tune-learning-rates",
            list_to_csv(training_options["tune_learning_rates"]),
            "--tune-batch-sizes",
            list_to_csv(training_options["tune_batch_sizes"]),
            "--tune-epochs",
            list_to_csv(training_options["tune_epochs"]),
            "--tune-weight-decays",
            list_to_csv(training_options["tune_weight_decays"]),
            "--tune-max-lengths",
            list_to_csv(training_options["tune_max_lengths"]),
        ]

    def build_train_cli_segment(self, training_options: dict) -> str:
        train_args = self.build_train_cli_args(training_options)
        if not train_args:
            return ""
        return " ".join(shlex.quote(arg) for arg in train_args)

    def start_training_session(self) -> bool:
        with self.training_state_lock:
            if self.training_active:
                return False
            self.training_active = True
            self.training_cancel_event.clear()
            self.local_training_process = None
            self.azure_ml_client_for_cancel = None
            self.azure_active_job_name = ""
            self.azure_cancel_requested = False

        self.get_model_btn.config(state="disabled")
        self.stop_training_btn.config(state="normal")
        return True

    def finish_training_session(self):
        with self.training_state_lock:
            self.training_active = False
            self.training_cancel_event.clear()
            self.local_training_process = None
            self.azure_ml_client_for_cancel = None
            self.azure_active_job_name = ""
            self.azure_cancel_requested = False

        self.get_model_btn.config(state="normal")
        self.stop_training_btn.config(state="disabled")
        self.status_var.set("Ready")

    def _raise_if_training_cancelled(self):
        if self.training_cancel_event.is_set():
            raise TrainingInterrupted("Training was interrupted by the user.")

    def request_azure_job_cancel(self, ml_client: MLClient, job_name: str):
        jobs_client = ml_client.jobs
        if hasattr(jobs_client, "begin_cancel"):
            jobs_client.begin_cancel(job_name)
            return
        if hasattr(jobs_client, "cancel"):
            jobs_client.cancel(job_name)
            return
        raise RuntimeError("Azure ML SDK does not expose a job cancellation method on this client version.")

    def _request_azure_cancel_once(self, ml_client: MLClient, job_name: str):
        with self.training_state_lock:
            if self.azure_cancel_requested:
                return
            self.azure_cancel_requested = True

        print(f"[DEBUG] Issuing cancellation for Azure job: {job_name}")
        self.root.after(
            0,
            lambda: self.status_var.set("Cancellation requested. Waiting for Azure to stop the job..."),
        )
        try:
            self.request_azure_job_cancel(ml_client, job_name)
        except Exception:
            with self.training_state_lock:
                self.azure_cancel_requested = False
            raise

    def stop_training(self):
        local_process = None
        azure_client = None
        azure_job_name = ""
        with self.training_state_lock:
            if not self.training_active:
                self.status_var.set("No training is currently running.")
                return
            self.training_cancel_event.set()
            local_process = self.local_training_process
            azure_client = self.azure_ml_client_for_cancel
            azure_job_name = self.azure_active_job_name

        self.status_var.set("Interrupt requested. Stopping training...")

        if local_process and local_process.poll() is None:
            try:
                local_process.terminate()
            except Exception:
                print("[DEBUG] Failed to terminate local training process cleanly. Attempting kill...")
                try:
                    local_process.kill()
                except Exception:
                    traceback.print_exc()

        if azure_client and azure_job_name:
            try:
                self._request_azure_cancel_once(azure_client, azure_job_name)
            except Exception:
                print("[DEBUG] Failed to request Azure job cancellation.")
                traceback.print_exc()

    def prepare_data(self):
        api_key = self.openai_api_key_entry.get().strip()
        model_name = self.openai_model_var.get().strip()
        input_path = self.filepath_entry.get().strip()
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter your OpenAI API key.")
            return
        if not model_name:
            messagebox.showwarning("Warning", "Please select or enter an OpenAI model name.")
            return
        if not input_path:
            messagebox.showwarning("Warning", "Please select a log file first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Prepared Data",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not save_path:
            return 

        mlflow_config, mlflow_error = self.resolve_mlflow_config(
            require_tracking_uri=False,
            soft_disable=True,
        )
        if mlflow_error:
            messagebox.showerror("MLflow Configuration Error", mlflow_error)
            return

        self.status_var.set(f"Processing data with OpenAI ({model_name})...")
        self.prepare_btn.config(state="disabled")
        threading.Thread(
            target=self.process_logs_llm,
            args=(input_path, save_path, api_key, model_name, mlflow_config),
            daemon=True,
        ).start()

    def process_logs_llm(self, input_path, save_path, api_key, model_name, mlflow_config):
        processed_logs = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        try:
            system_prompt = self.load_prompt()
            df = pd.read_csv(input_path)
            input_file_hash = compute_file_sha256(input_path)
            
            log_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['log', 'message', 'msg', 'text']):
                    log_col = col
                    break
            if not log_col:
                log_col = df.columns[0]

            total_rows = len(df)
            batch_size = 10 

            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                self.root.after(0, lambda current=i, total=total_rows: self.status_var.set(
                    f"Processing rows {current + 1} to {min(current + batch_size, total)} of {total}..."
                ))

                logs_batch = batch_df[log_col].astype(str).tolist()

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(logs_batch)}
                    ],
                    "response_format": {"type": "json_object"},
                }

                max_retries = 5
                base_delay = 5 
                success = False

                for attempt in range(max_retries):
                    response = requests.post(api_url, json=payload, headers=headers, timeout=120)
                    
                    if response.status_code == 429:
                        wait_time = base_delay * (2 ** attempt)
                        self.root.after(0, lambda w=wait_time: self.status_var.set(
                            f"Rate limit hit (429). Pausing for {w} seconds..."
                        ))
                        time.sleep(wait_time)
                        continue 
                    
                    response.raise_for_status()
                    success = True
                    break 

                if not success:
                    raise Exception("Max retries reached. API rejecting requests due to rate limits.")

                result_json = response.json()
                usage = result_json.get("usage", {}) if isinstance(result_json, dict) else {}
                for key in usage_totals:
                    usage_totals[key] += int(usage.get(key, 0) or 0)
                llm_content = result_json['choices'][0]['message']['content']
                
                try:
                    parsed_data = json.loads(llm_content)
                    results_array = parsed_data.get("results", [])
                    
                    for idx, log_text in enumerate(logs_batch):
                        if idx < len(results_array):
                            log_class = results_array[idx].get("class", "Noise")
                        else:
                            log_class = "Noise"

                        processed_logs.append({
                            "LogMessage": log_text,
                            "class": log_class
                        })
                except json.JSONDecodeError:
                    for log_text in logs_batch:
                        processed_logs.append({
                            "LogMessage": log_text,
                            "class": "Noise"
                        })
                        
                time.sleep(1.5) 

            output_df = pd.DataFrame(processed_logs)
            output_df.to_csv(save_path, index=False)

            output_file_hash = compute_file_sha256(save_path)
            mlops_lineage = self.log_data_prep_mlflow(
                mlflow_config=mlflow_config,
                input_path=input_path,
                output_path=save_path,
                system_prompt=system_prompt,
                llm_model=model_name,
                input_df=df,
                output_df=output_df,
                input_hash=input_file_hash,
                output_hash=output_file_hash,
                usage_totals=usage_totals,
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
            }
            write_sidecar_for_csv(save_path, sidecar_payload)
            
            self.root.after(0, lambda: self.status_var.set("Data processed successfully!"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Data saved to:\n{save_path}"))

        except Exception as e:
            if processed_logs:
                output_df = pd.DataFrame(processed_logs)
                output_df.to_csv(save_path, index=False)
                error_msg = f"An error stopped the process: {e}\n\nHowever, partial data ({len(processed_logs)} rows) was successfully saved to:\n{save_path}"
            else:
                error_msg = f"Error processing file: {e}\n\nNo data was saved."
            
            self.root.after(0, self.show_error, error_msg)

        finally:
            self.root.after(0, lambda: self.prepare_btn.config(state="normal"))

    # --- HEAVILY LOGGED TRAINING METHODS ---
    def start_training_thread(self):
        csv_path = self.training_filepath_entry.get().strip()
        sub_id = self.azure_sub_entry.get().strip()
        tenant_id = self.azure_tenant_entry.get().strip()
        azure_compute = self.azure_compute_var.get().strip().lower() or "cpu"
        local_device = self.local_device_var.get().strip() or "auto"
        local_runtime = self.local_runtime_var.get().strip() or "host"
        train_mode = self.train_mode.get().strip()
        training_options, training_options_error = self.collect_training_options()

        print("\n--- [DEBUG] STARTING TRAINING WORKFLOW ---")
        print(f"[DEBUG] CSV Path: {csv_path}")
        print(f"[DEBUG] Sub ID: {sub_id}")
        print(f"[DEBUG] Tenant ID: {tenant_id}")
        print(f"[DEBUG] Azure Compute: {azure_compute}")
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
        if not os.path.exists("train.py"):
            messagebox.showerror("Error", "Could not find 'train.py' in the app directory.")
            return
        if training_options_error:
            messagebox.showerror("Training Configuration Error", training_options_error)
            return
        if training_options is None:
            messagebox.showerror("Training Configuration Error", "Training configuration could not be resolved.")
            return

        mlflow_config, mlflow_error = self.resolve_mlflow_config(
            require_tracking_uri=False,
            soft_disable=True,
        )
        if mlflow_error:
            messagebox.showerror("MLflow Configuration Error", mlflow_error)
            return

        if bool(mlflow_config.get("enabled")):
            if train_mode == "azure" and mlflow_config.get("backend") == "local":
                messagebox.showerror(
                    "MLflow Configuration Error",
                    (
                        "Azure training cannot use local MLflow backend.\n\n"
                        "Switch MLflow Backend to `azure` or `custom_uri` before starting Azure training."
                    ),
                )
                return

            if mlflow_config.get("backend") == "custom_uri" and not clean_optional_string(mlflow_config.get("tracking_uri")):
                messagebox.showerror(
                    "MLflow Configuration Error",
                    "MLflow backend is `custom_uri` but Tracking URI is empty.",
                )
                return

            if (
                train_mode == "local"
                and mlflow_config.get("backend") == "azure"
                and not clean_optional_string(mlflow_config.get("tracking_uri"))
            ):
                messagebox.showerror(
                    "MLflow Configuration Error",
                    (
                        "MLflow backend is `azure` but tracking URI is unresolved.\n\n"
                        "Run one Azure training first or use `custom_uri`."
                    ),
                )
                return

        if train_mode == "azure":
            run_source = "azure_gpu" if azure_compute == "gpu" else "azure_cpu"
            pipeline_context = self.prepare_training_pipeline_context(csv_path, mlflow_config, run_source)
            if not self.start_training_session():
                messagebox.showwarning("Training In Progress", "A training workflow is already running.")
                return
            self.status_var.set("Initializing authentication...")
            try:
                threading.Thread(
                    target=self.run_azure_training,
                    args=(csv_path, sub_id, tenant_id, azure_compute, mlflow_config, pipeline_context, training_options),
                    daemon=True,
                ).start()
            except Exception:
                self.finish_training_session()
                raise
        else:
            if local_device != "cuda":
                local_runtime = "host"

            if local_device == "cuda" and local_runtime == "host":
                available, check_error = self.check_host_cuda_available()
                if not available:
                    hint = (
                        "CUDA is not available in the host Python environment.\n\n"
                        "Switch Local Runtime to 'container' or install CUDA-enabled PyTorch locally."
                    )
                    if check_error:
                        hint += f"\n\nDetails:\n{check_error}"
                    messagebox.showerror("CUDA Not Available", hint)
                    return

            run_source = "local_container" if local_runtime == "container" else "local_host"
            pipeline_context = self.prepare_training_pipeline_context(csv_path, mlflow_config, run_source)
            mlflow_env = self.build_training_mlflow_env(mlflow_config, pipeline_context, run_source)

            if not self.start_training_session():
                messagebox.showwarning("Training In Progress", "A training workflow is already running.")
                return
            self.status_var.set("Preparing local training...")
            try:
                threading.Thread(
                    target=self.run_local_training,
                    args=(csv_path, local_device, local_runtime, mlflow_env, training_options),
                    daemon=True,
                ).start()
            except Exception:
                self.finish_training_session()
                raise

    def check_host_cuda_available(self):
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() == "1", None
        except Exception as exc:
            return False, str(exc)

    def run_local_training(self, csv_path, device, runtime, mlflow_env, training_options):
        process = None
        backend = "container" if runtime == "container" else "host"
        try:
            self._raise_if_training_cancelled()
            project_dir = os.path.dirname(os.path.abspath(__file__))
            extra_train_args = self.build_train_cli_args(training_options)

            if backend == "container":
                docker_script = os.path.join(project_dir, "scripts", "train_docker.sh")
                if not os.path.exists(docker_script):
                    raise FileNotFoundError("Could not find scripts/train_docker.sh in the app directory.")
                command_args = ["bash", docker_script, csv_path, *extra_train_args]
                process_env = os.environ.copy()
                process_env["DEVICE"] = device
                process_env.update(mlflow_env)
            else:
                train_script = os.path.join(project_dir, "train.py")
                if not os.path.exists(train_script):
                    raise FileNotFoundError("Could not find 'train.py' in the app directory.")
                command_args = [sys.executable, train_script, "--data", csv_path, *extra_train_args]
                process_env = os.environ.copy()
                if device == "cpu":
                    process_env["CUDA_VISIBLE_DEVICES"] = "-1"
                process_env.update(mlflow_env)

            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Starting local {backend} training on device: {device}..."
                ),
            )

            process = subprocess.Popen(
                command_args,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_env,
            )
            with self.training_state_lock:
                self.local_training_process = process

            if process.stdout:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                        continue
                    clean_line = line.strip()
                    if clean_line:
                        print(f"[LOCAL-TRAIN] {clean_line}")
                        self.root.after(
                            0,
                            lambda msg=clean_line: self.status_var.set(msg[:150]),
                        )
                    if self.training_cancel_event.is_set() and process.poll() is None:
                        try:
                            process.terminate()
                        except Exception:
                            pass

            return_code = process.wait()
            if self.training_cancel_event.is_set():
                self.root.after(0, lambda: self.status_var.set("Local training was interrupted."))
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Training Interrupted",
                        "Local training was interrupted before completion.",
                    ),
                )
                return

            if return_code != 0:
                raise RuntimeError(f"Local training failed with exit code {return_code}.")

            model_path = os.path.abspath(os.path.join(project_dir, "outputs", "final_model"))
            self.root.after(0, lambda: self.status_var.set("Local training completed successfully."))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Local {backend} model training completed.\n\nModel saved to:\n{model_path}",
                ),
            )
        except TrainingInterrupted:
            self.root.after(0, lambda: self.status_var.set("Local training was interrupted."))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Training Interrupted",
                    "Local training was interrupted before completion.",
                ),
            )
        except Exception as e:
            if self.training_cancel_event.is_set():
                self.root.after(0, lambda: self.status_var.set("Local training was interrupted."))
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Training Interrupted",
                        "Local training was interrupted before completion.",
                    ),
                )
            else:
                print("\n--- [DEBUG] LOCAL TRAINING EXCEPTION ---")
                traceback.print_exc()
                print("----------------------------------------\n")
                self.root.after(
                    0,
                    lambda err=str(e): messagebox.showerror(
                        "Training Error",
                        f"An error occurred during local training.\n\n{err}",
                    ),
                )
                self.root.after(0, lambda: self.status_var.set("Local training failed."))
        finally:
            with self.training_state_lock:
                self.local_training_process = None
            self.root.after(0, self.finish_training_session)

    def run_azure_training(self, csv_path, sub_id, tenant_id, azure_compute, mlflow_config, pipeline_context, training_options):
        ml_client = None
        returned_job_name = ""
        job_status = ""
        compute_mode = "gpu" if azure_compute == "gpu" else "cpu"
        if compute_mode == "gpu":
            compute_name = "gpu-cluster-temp"
            compute_size = "Standard_NC4as_T4_v3"
        else:
            compute_name = "cpu-cluster-temp"
            compute_size = "Standard_D2as_v4"
        compute_created = False
        resource_group = self.RESOURCE_GROUP
        workspace_name = self.WORKSPACE_NAME

        try:
            self._raise_if_training_cancelled()
            print(f"[DEBUG] Attempting Interactive Browser Login to Tenant: {tenant_id}")
            self.root.after(0, lambda: self.status_var.set("Please log in to Azure in your web browser..."))
            
            # Authenticate directly to the correct Tenant
            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
            ml_client = MLClient(credential, sub_id, resource_group, workspace_name)
            print("[DEBUG] MLClient instantiated successfully.")
            with self.training_state_lock:
                self.azure_ml_client_for_cancel = ml_client

            # --- Check / Create Resource Group ---
            self._raise_if_training_cancelled()
            print(f"[DEBUG] Checking existence of Resource Group: {resource_group}")
            self.root.after(0, lambda: self.status_var.set("Checking Resource Group..."))
            resource_client = ResourceManagementClient(credential, sub_id)
            
            try:
                resource_client.resource_groups.get(resource_group)
                print(f"[DEBUG] SUCCESS: Resource Group '{resource_group}' already exists. Skipping creation.")
            except Exception as e:
                print(f"[DEBUG] Resource group not found. Reason: {e}")
                print(f"[DEBUG] Attempting to create Resource Group '{resource_group}'...")
                self.root.after(0, lambda: self.status_var.set("Creating Resource Group..."))
                resource_client.resource_groups.create_or_update(
                    resource_group, 
                    {"location": "eastus"}
                )
                print(f"[DEBUG] SUCCESS: Resource Group '{resource_group}' created.")

            # --- Auto-Register Machine Learning Services ---
            self._raise_if_training_cancelled()
            print("[DEBUG] Verifying Microsoft.MachineLearningServices provider registration...")
            self.root.after(0, lambda: self.status_var.set("Verifying Azure ML registration..."))
            resource_client.providers.register('Microsoft.MachineLearningServices')
            
            while True:
                self._raise_if_training_cancelled()
                provider_info = resource_client.providers.get('Microsoft.MachineLearningServices')
                print(f"[DEBUG] Provider State: {provider_info.registration_state}")
                if provider_info.registration_state == 'Registered':
                    print("[DEBUG] SUCCESS: Provider is fully Registered.")
                    break
                self.root.after(0, lambda: self.status_var.set("Activating ML Services on Azure (takes 1-2 mins)..."))
                time.sleep(10)

            # --- Check / Create Workspace ---
            self._raise_if_training_cancelled()
            print(f"[DEBUG] Checking existence of ML Workspace: {workspace_name}")
            self.root.after(0, lambda: self.status_var.set("Ensuring Azure ML Workspace exists..."))
            try:
                ml_client.workspaces.get(workspace_name)
                print(f"[DEBUG] SUCCESS: Workspace '{workspace_name}' already exists. Skipping creation.")
            except Exception as e:
                print(f"[DEBUG] Workspace not found or inaccessible. Reason: {e}")
                print(f"[DEBUG] Attempting to create ML Workspace '{workspace_name}' (This takes a minute)...")
                self.root.after(0, lambda: self.status_var.set("Creating ML Workspace..."))
                ws = Workspace(name=workspace_name, location="eastus")
                ml_client.workspaces.begin_create(ws).result()
                print(f"[DEBUG] SUCCESS: Workspace '{workspace_name}' created.")

            resolved_mlflow_config = dict(mlflow_config)
            run_source = "azure_gpu" if compute_mode == "gpu" else "azure_cpu"
            tracking_uri = clean_optional_string(resolved_mlflow_config.get("tracking_uri"))

            if bool(resolved_mlflow_config.get("enabled")) and resolved_mlflow_config.get("backend") == "azure":
                tracking_uri = self._resolve_azure_mlflow_tracking_uri(ml_client)
                resolved_mlflow_config["tracking_uri"] = tracking_uri

            if bool(resolved_mlflow_config.get("enabled")) and not tracking_uri:
                raise ValueError(
                    "MLflow is enabled for Azure training but tracking URI is unresolved. "
                    "Use backend `azure` (with resolvable workspace URI) or `custom_uri`."
                )

            pipeline_context_local = dict(pipeline_context)
            if (
                bool(resolved_mlflow_config.get("enabled"))
                and not clean_optional_string(pipeline_context_local.get("parent_run_id"))
            ):
                parent_run_id = self.create_pipeline_parent_run(
                    resolved_mlflow_config,
                    str(pipeline_context_local.get("pipeline_id", "")),
                    run_source,
                )
                pipeline_context_local["parent_run_id"] = parent_run_id
                sidecar_existing = read_sidecar_for_csv(csv_path) or {}
                sidecar_existing["parent_run_id"] = parent_run_id
                sidecar_existing["tracking_uri"] = clean_optional_string(sidecar_existing.get("tracking_uri")) or clean_optional_string(
                    resolved_mlflow_config.get("tracking_uri", "")
                )
                sidecar_existing["experiment_name"] = clean_optional_string(sidecar_existing.get("experiment_name")) or clean_optional_string(
                    resolved_mlflow_config.get("experiment_name", "")
                )
                sidecar_existing["training_tracking_uri"] = clean_optional_string(resolved_mlflow_config.get("tracking_uri", ""))
                sidecar_existing["training_experiment_name"] = clean_optional_string(resolved_mlflow_config.get("experiment_name", ""))
                if "created_at" not in sidecar_existing:
                    sidecar_existing["created_at"] = now_utc_iso()
                write_sidecar_for_csv(csv_path, sidecar_existing)

            mlflow_env = self.build_training_mlflow_env(resolved_mlflow_config, pipeline_context_local, run_source)
            export_segment = self.build_shell_export_segment(mlflow_env)
            mlflow_install_fragment = "mlflow==2.9.2 " if bool_from_env(mlflow_env.get("MLOPS_ENABLED", "0")) else ""
            train_cli_segment = self.build_train_cli_segment(training_options)
            train_cli_segment = f" {train_cli_segment}" if train_cli_segment else ""

            # --- Create Compute Cluster ---
            self._raise_if_training_cancelled()
            print(f"[DEBUG] Checking/Provisioning Compute Cluster: {compute_name}...")
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Provisioning {compute_mode.upper()} cluster ({compute_size})..."
                ),
            )
            compute = AmlCompute(
                name=compute_name,
                type="amlcompute",
                size=compute_size,
                min_instances=0,
                max_instances=1,
                idle_time_before_scale_down=120
            )
            ml_client.compute.begin_create_or_update(compute).result()
            compute_created = True
            print(f"[DEBUG] SUCCESS: Compute cluster '{compute_name}' is ready.")
            self._raise_if_training_cancelled()

            # --- Define and Submit Job ---
            print(f"[DEBUG] Defining training job using target CSV: {csv_path}")
            self.root.after(0, lambda: self.status_var.set("Uploading data and starting DeBERTa training..."))
            
            # THE FIX: Convert Windows backslashes to forward slashes for Azure URI compatibility
            safe_csv_path = csv_path.replace("\\", "/")
            print(f"[DEBUG] Normalized safe path for Azure: {safe_csv_path}")

            job = command(
                inputs={
                    "training_data": Input(type="uri_file", path=safe_csv_path, mode="download")
                },
                compute=compute_name,
                environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest", 
                code=".", 
                # Install versions that remain compatible with the curated Torch 1.10 Azure image.
                command=(
                    "pip install --upgrade "
                    "numpy==1.23.5 "
                    "pandas==1.5.3 "
                    "transformers==4.24.0 "
                    "sentencepiece==0.1.99 "
                    "protobuf==3.20.3 "
                    "scikit-learn==1.1.3 "
                    f"{mlflow_install_fragment}"
                    f"&& {export_segment} "
                    f"&& USE_TF=0 python train.py --data ${{inputs.training_data}}{train_cli_segment}"
                ),
                experiment_name="deberta-log-classification",
            )

            print("[DEBUG] Submitting job to Azure...")
            returned_job = ml_client.jobs.create_or_update(job)
            print(f"[DEBUG] SUCCESS: Job submitted! Job Name: {returned_job.name}")
            returned_job_name = returned_job.name
            with self.training_state_lock:
                self.azure_active_job_name = returned_job_name

            # --- Polling Loop ---
            print("[DEBUG] Beginning polling loop for job completion...")
            while True:
                if self.training_cancel_event.is_set() and returned_job_name:
                    try:
                        self._request_azure_cancel_once(ml_client, returned_job_name)
                    except Exception:
                        print("[DEBUG] Azure cancellation request failed during polling.")
                        traceback.print_exc()

                job_status = ml_client.jobs.get(returned_job_name).status
                print(f"[DEBUG] Azure Job Status: {job_status}")
                self.root.after(0, lambda s=job_status: self.status_var.set(f"Training in progress. Azure Status: {s}"))
                if job_status in ["Completed", "Failed", "Canceled"]:
                    break
                time.sleep(15)

            if job_status == "Failed":
                print("[DEBUG] FATAL: Job failed on the Azure side.")
                raise Exception("The training job failed on the Azure machine. Check Azure Portal logs.")
            if job_status == "Canceled" or self.training_cancel_event.is_set():
                self.root.after(0, lambda: self.status_var.set("Azure training was interrupted."))
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Training Interrupted",
                        "Azure training was interrupted before completion.",
                    ),
                )
                return

            # --- Download Model ---
            print("[DEBUG] Job Completed! Attempting to download outputs...")
            self.root.after(0, lambda: self.status_var.set("Training Complete! Downloading model..."))
            download_path = "./downloaded_model"
            ml_client.jobs.download(name=returned_job_name, download_path=download_path, all=False)
            print(f"[DEBUG] SUCCESS: Files downloaded to {download_path}")
            self.cache_downloaded_training_mlflow_metadata(download_path)
            
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Model trained and downloaded to:\n{os.path.abspath(download_path)}"))

        except TrainingInterrupted:
            if ml_client and returned_job_name:
                try:
                    self._request_azure_cancel_once(ml_client, returned_job_name)
                except Exception:
                    print("[DEBUG] Failed to request Azure cancellation while handling interrupt.")
                    traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set("Azure training was interrupted."))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Training Interrupted",
                    "Azure training was interrupted before completion.",
                ),
            )
        except Exception as e:
            if self.training_cancel_event.is_set():
                self.root.after(0, lambda: self.status_var.set("Azure training was interrupted."))
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Training Interrupted",
                        "Azure training was interrupted before completion.",
                    ),
                )
            else:
                print("\n--- [DEBUG] AN EXCEPTION OCCURRED ---")
                traceback.print_exc() 
                print("--------------------------------------\n")
                
                self.root.after(0, lambda err=str(e): messagebox.showerror("Training Error", f"An error occurred. Check the terminal for full details.\n\n{err}"))
                self.root.after(0, lambda: self.status_var.set("Process halted due to error."))

        finally:
            # --- THE CRITICAL CLEANUP BLOCK ---
            print("[DEBUG] Entering cleanup block...")
            if ml_client and returned_job_name and self.training_cancel_event.is_set():
                try:
                    self._request_azure_cancel_once(ml_client, returned_job_name)
                except Exception:
                    print("[DEBUG] Azure cancellation re-check failed in cleanup.")
                    traceback.print_exc()

            if ml_client and compute_created:
                self.root.after(0, lambda: self.status_var.set("Destroying compute cluster to prevent charges..."))
                print(f"[DEBUG] Issuing delete command for compute cluster '{compute_name}'...")
                try:
                    ml_client.compute.begin_delete(compute_name).result()
                    print(f"[DEBUG] SUCCESS: Compute cluster '{compute_name}' permanently deleted.")
                except Exception as cleanup_error:
                    print("\n--- [DEBUG] FATAL CLEANUP ERROR ---")
                    print(f"FAILED TO DELETE CLUSTER {compute_name}. You may be charged!")
                    traceback.print_exc()
                    print("--------------------------------------\n")
            else:
                print("[DEBUG] Cleanup bypassed: Compute cluster was never fully created.")
            
            with self.training_state_lock:
                self.azure_ml_client_for_cancel = None
                self.azure_active_job_name = ""

            print("[DEBUG] WORKFLOW FINISHED.\n")
            self.root.after(0, self.finish_training_session)

    def show_error(self, message):
        self.status_var.set("Process halted.")
        messagebox.showerror("Notice", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = LogProcessorApp(root)
    root.mainloop()
