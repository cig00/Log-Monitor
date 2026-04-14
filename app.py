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
from pathlib import Path
from urllib.parse import quote as url_quote

# Azure ML Imports
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.mgmt.resource import ResourceManagementClient

from mlops_utils import (
    MLOPS_ENV_VARS,
    bool_from_env,
    clean_optional_string,
    compute_file_sha256,
    dataframe_metadata,
    dataframe_sample,
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

        self.prepare_btn = ttk.Button(file_frame, text="Prepare Data (GPT-4o)", command=self.prepare_data)
        self.prepare_btn.grid(row=1, column=0, columnspan=3, pady=10)

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

        self.host_service_btn = ttk.Button(
            hosting_frame,
            text="Host Service",
            command=lambda: messagebox.showinfo("Info", "Host Service feature coming soon.")
        )
        self.host_service_btn.grid(row=2, column=0, columnspan=4, pady=10)

        ttk.Separator(hosting_frame, orient="horizontal").grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="ew",
            padx=5,
            pady=(4, 8),
        )

        ttk.Label(hosting_frame, text="MLflow Enabled:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.mlflow_enabled_var = tk.BooleanVar(value=True)
        self.mlflow_enabled_check = ttk.Checkbutton(
            hosting_frame,
            variable=self.mlflow_enabled_var,
            command=self.on_mlflow_enabled_change,
        )
        self.mlflow_enabled_check.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(hosting_frame, text="MLflow Backend:").grid(row=4, column=2, sticky="w", padx=5, pady=5)
        self.mlflow_backend_var = tk.StringVar(value="local")
        self.mlflow_backend_combo = ttk.Combobox(
            hosting_frame,
            textvariable=self.mlflow_backend_var,
            state="readonly",
            values=["local", "azure", "custom_uri"],
            width=15,
        )
        self.mlflow_backend_combo.grid(row=4, column=3, sticky="ew", padx=5, pady=5)
        self.mlflow_backend_combo.bind("<<ComboboxSelected>>", self.on_mlflow_backend_change)

        ttk.Label(hosting_frame, text="Tracking URI:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.mlflow_tracking_uri_var = tk.StringVar(value=self.local_mlflow_tracking_uri)
        self.mlflow_tracking_uri_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.mlflow_tracking_uri_var,
            width=30,
            state="readonly",
        )
        self.mlflow_tracking_uri_entry.grid(row=5, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Experiment Name:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.mlflow_experiment_var = tk.StringVar(value="")
        self.mlflow_experiment_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.mlflow_experiment_var,
            width=24,
        )
        self.mlflow_experiment_entry.grid(row=6, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(hosting_frame, text="Registered Model:").grid(row=6, column=2, sticky="w", padx=5, pady=5)
        self.mlflow_registered_model_var = tk.StringVar(value="")
        self.mlflow_registered_model_entry = ttk.Entry(
            hosting_frame,
            textvariable=self.mlflow_registered_model_var,
            width=24,
        )
        self.mlflow_registered_model_entry.grid(row=6, column=3, sticky="ew", padx=5, pady=5)

        self.open_mlflow_btn = ttk.Button(
            hosting_frame,
            text="Open MLflow Console",
            command=self.open_mlflow_console,
        )
        self.open_mlflow_btn.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=8)

        self.register_model_btn = ttk.Button(
            hosting_frame,
            text="Register Last Model",
            command=self.register_last_model_version,
        )
        self.register_model_btn.grid(row=7, column=2, columnspan=2, sticky="ew", padx=5, pady=8)

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
        if not os.path.exists("prompt.txt"):
            raise FileNotFoundError("Could not find 'prompt.txt' in the application directory.")
        with open("prompt.txt", "r") as file:
            return file.read()

    def on_mlflow_enabled_change(self):
        enabled = bool(self.mlflow_enabled_var.get())
        self.mlflow_backend_combo.config(state="readonly" if enabled else "disabled")
        self.mlflow_experiment_entry.config(state="normal" if enabled else "disabled")
        self.mlflow_registered_model_entry.config(state="normal" if enabled else "disabled")
        self.open_mlflow_btn.config(state="normal" if enabled else "disabled")
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

    def resolve_mlflow_config(self, require_tracking_uri: bool, ml_client: MLClient | None = None):
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

        if mlflow is None:
            return config, "MLflow is enabled in UI, but the `mlflow` package is not available."

        missing_fields: list[str] = []
        if not experiment_name:
            missing_fields.append("Experiment Name")
        if not registered_model_name:
            missing_fields.append("Registered Model")
        if missing_fields:
            fields_text = ", ".join(missing_fields)
            return (
                config,
                (
                    f"MLflow is enabled, but required field(s) are empty: {fields_text}.\n\n"
                    "Set non-empty values in Hosting:\n"
                    "- Experiment Name (example: `log-monitor`)\n"
                    "- Registered Model (example: `deberta-log-classifier`)"
                ),
            )

        if backend == "local":
            config["tracking_uri"] = self.local_mlflow_tracking_uri
            self.mlflow_tracking_uri_var.set(self.local_mlflow_tracking_uri)
        elif backend == "custom_uri":
            config["tracking_uri"] = clean_optional_string(self.mlflow_tracking_uri_var.get())
            if require_tracking_uri and not config["tracking_uri"]:
                return config, "MLflow backend is `custom_uri` but Tracking URI is empty."
        elif backend == "azure":
            config["tracking_uri"] = self._resolve_azure_mlflow_tracking_uri(ml_client)
            if not config["tracking_uri"]:
                current_value = clean_optional_string(self.mlflow_tracking_uri_var.get())
                if current_value and "resolved during Azure run" not in current_value:
                    config["tracking_uri"] = current_value
            if require_tracking_uri and not config["tracking_uri"]:
                return config, "Azure MLflow URI is not resolved yet. Run Azure auth or provide `custom_uri` backend."
        else:
            return config, f"Unsupported MLflow backend: {backend}"

        return config, None

    def build_training_mlflow_env(self, mlflow_config: dict, pipeline_context: dict, run_source: str) -> dict[str, str]:
        env = {key: "" for key in MLOPS_ENV_VARS}
        enabled = (
            bool(mlflow_config.get("enabled"))
            and mlflow is not None
            and bool(clean_optional_string(mlflow_config.get("tracking_uri", "")))
        )
        env["MLOPS_ENABLED"] = "1" if enabled else "0"
        if not enabled:
            return env

        tags = {
            "run_type": "training",
            "pipeline_id": str(pipeline_context.get("pipeline_id", "")),
            "run_source": run_source,
            "environment_mode": self.train_mode.get().strip() or "unknown",
        }
        data_prep_run_id = clean_optional_string(pipeline_context.get("data_prep_run_id", ""))
        if data_prep_run_id:
            tags["data_prep_run_id"] = data_prep_run_id

        env["MLFLOW_TRACKING_URI"] = str(mlflow_config["tracking_uri"])
        env["MLFLOW_EXPERIMENT_NAME"] = str(mlflow_config["experiment_name"])
        env["MLFLOW_PIPELINE_ID"] = str(pipeline_context.get("pipeline_id", ""))
        env["MLFLOW_PARENT_RUN_ID"] = str(pipeline_context.get("parent_run_id", ""))
        env["MLFLOW_RUN_SOURCE"] = str(run_source)
        env["MLFLOW_TAGS_JSON"] = json.dumps(tags)
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

        if bool(mlflow_config.get("enabled")) and not parent_run_id:
            parent_run_id = self.create_pipeline_parent_run(mlflow_config, pipeline_id, run_source)

        context = {
            "pipeline_id": pipeline_id,
            "parent_run_id": parent_run_id,
            "data_prep_run_id": clean_optional_string(sidecar.get("data_prep_run_id")),
            "prompt_hash": clean_optional_string(sidecar.get("prompt_hash")),
            "input_dataset_hash": clean_optional_string(sidecar.get("input_dataset_hash")),
            "output_dataset_hash": clean_optional_string(sidecar.get("output_dataset_hash")),
        }

        sidecar_payload = {
            "pipeline_id": pipeline_id,
            "parent_run_id": parent_run_id,
            "data_prep_run_id": context["data_prep_run_id"],
            "prompt_hash": context["prompt_hash"],
            "input_dataset_hash": context["input_dataset_hash"],
            "output_dataset_hash": context["output_dataset_hash"],
            "created_at": clean_optional_string(sidecar.get("created_at")) or now_utc_iso(),
            "tracking_uri": clean_optional_string(mlflow_config.get("tracking_uri", "")),
            "experiment_name": clean_optional_string(mlflow_config.get("experiment_name", "")),
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
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        input_hash: str,
        output_hash: str,
    ) -> dict:
        payload = {
            "pipeline_id": str(uuid.uuid4()),
            "parent_run_id": "",
            "data_prep_run_id": "",
            "prompt_hash": prompt_sha256(system_prompt),
            "input_dataset_hash": input_hash,
            "output_dataset_hash": output_hash,
            "created_at": now_utc_iso(),
            "tracking_uri": clean_optional_string(mlflow_config.get("tracking_uri", "")),
            "experiment_name": clean_optional_string(mlflow_config.get("experiment_name", "")),
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

                    mlflow.log_param("llm_model", "gpt-4o")
                    mlflow.log_param("input_filename", os.path.basename(input_path))
                    mlflow.log_param("output_filename", os.path.basename(output_path))
                    mlflow.log_param("prompt_hash", payload["prompt_hash"])
                    mlflow.log_param("input_dataset_hash", payload["input_dataset_hash"])
                    mlflow.log_param("output_dataset_hash", payload["output_dataset_hash"])

                    mlflow.log_metric("input_rows", int(input_meta.get("row_count", 0)))
                    mlflow.log_metric("output_rows", int(output_meta.get("row_count", 0)))

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
        direct_meta = Path(self.project_dir) / "outputs" / "last_training_mlflow.json"
        if direct_meta.exists():
            candidate_paths.append(direct_meta)

        download_root = Path(self.project_dir) / "downloaded_model"
        if download_root.exists():
            candidate_paths.extend(download_root.rglob("last_training_mlflow.json"))

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
        if not bool(self.mlflow_enabled_var.get()):
            messagebox.showwarning("MLflow", "Enable MLflow first to open the tracking console.")
            return

        config, error = self.resolve_mlflow_config(require_tracking_uri=False)
        if error:
            is_azure_unresolved = (
                config.get("backend") == "azure"
                and "Azure MLflow URI is not resolved yet" in str(error)
            )
            if not is_azure_unresolved:
                messagebox.showerror("MLflow", error)
                return

        backend = config["backend"]
        tracking_uri = clean_optional_string(config.get("tracking_uri"))

        if backend == "local":
            try:
                backend_store_uri = tracking_uri
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
                webbrowser.open("http://127.0.0.1:5001")
            except Exception as exc:
                messagebox.showerror("MLflow", f"Failed to start local MLflow UI.\n\n{exc}")
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

        sub_id = clean_optional_string(self.azure_sub_entry.get())
        if not sub_id:
            messagebox.showwarning("MLflow", "Azure Subscription ID is required to open Azure ML Studio.")
            return
        wsid = (
            f"/subscriptions/{sub_id}/resourceGroups/{self.RESOURCE_GROUP}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/{self.WORKSPACE_NAME}"
        )
        studio_url = f"https://ml.azure.com/experiments?wsid={url_quote(wsid, safe='')}"
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
        token = self.github_key_entry.get().strip()
        input_path = self.filepath_entry.get().strip()
        
        if not token:
            messagebox.showwarning("Warning", "Please enter your GitHub PAT to use the LLM.")
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

        mlflow_config, mlflow_error = self.resolve_mlflow_config(require_tracking_uri=False)
        if mlflow_error:
            messagebox.showerror("MLflow Configuration Error", mlflow_error)
            return

        self.status_var.set("Processing data with GPT-4o...")
        self.prepare_btn.config(state="disabled")
        threading.Thread(
            target=self.process_logs_llm,
            args=(input_path, save_path, token, mlflow_config),
            daemon=True,
        ).start()

    def process_logs_llm(self, input_path, save_path, token, mlflow_config):
        processed_logs = []
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

            api_url = "https://models.inference.ai.azure.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                self.root.after(0, lambda current=i, total=total_rows: self.status_var.set(
                    f"Processing rows {current + 1} to {min(current + batch_size, total)} of {total}..."
                ))

                logs_batch = batch_df[log_col].astype(str).tolist()

                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(logs_batch)}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1
                }

                max_retries = 5
                base_delay = 5 
                success = False

                for attempt in range(max_retries):
                    response = requests.post(api_url, json=payload, headers=headers)
                    
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
                input_df=df,
                output_df=output_df,
                input_hash=input_file_hash,
                output_hash=output_file_hash,
            )

            sidecar_payload = {
                "pipeline_id": mlops_lineage.get("pipeline_id", ""),
                "parent_run_id": mlops_lineage.get("parent_run_id", ""),
                "data_prep_run_id": mlops_lineage.get("data_prep_run_id", ""),
                "prompt_hash": mlops_lineage.get("prompt_hash", ""),
                "input_dataset_hash": mlops_lineage.get("input_dataset_hash", ""),
                "output_dataset_hash": mlops_lineage.get("output_dataset_hash", ""),
                "created_at": mlops_lineage.get("created_at", now_utc_iso()),
                "tracking_uri": mlops_lineage.get("tracking_uri", ""),
                "experiment_name": mlops_lineage.get("experiment_name", ""),
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

        mlflow_config, mlflow_error = self.resolve_mlflow_config(require_tracking_uri=False)
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
                sidecar_existing["tracking_uri"] = clean_optional_string(resolved_mlflow_config.get("tracking_uri", ""))
                sidecar_existing["experiment_name"] = clean_optional_string(resolved_mlflow_config.get("experiment_name", ""))
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
