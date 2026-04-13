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

# Azure ML Imports
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.mgmt.resource import ResourceManagementClient

class LogProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Log Classifier & DeBERTa Trainer")
        self.root.geometry("650x700")
        self.root.minsize(650, 620)
        self.root.resizable(True, True)

        # UI Styling
        style = ttk.Style()
        style.theme_use('clam')

        self.create_widgets()

    def create_widgets(self):
        # --- File Upload Section ---
        file_frame = ttk.LabelFrame(self.root, text="Data Processing", padding=(10, 10))
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
        train_frame = ttk.LabelFrame(self.root, text="Model Training (DeBERTa)", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)
        train_frame.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="Azure Sub ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.azure_sub_entry = ttk.Entry(train_frame, width=40)
        self.azure_sub_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        ttk.Label(train_frame, text="Tenant ID:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.azure_tenant_entry = ttk.Entry(train_frame, width=40)
        self.azure_tenant_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        ttk.Label(train_frame, text="Azure Compute:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.azure_compute_var = tk.StringVar(value="cpu")
        self.azure_compute_combo = ttk.Combobox(
            train_frame,
            textvariable=self.azure_compute_var,
            state="readonly",
            values=["cpu", "gpu"],
            width=16,
        )
        self.azure_compute_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(train_frame, text="Labeled Data (CSV):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.training_filepath_entry = ttk.Entry(train_frame, width=40)
        self.training_filepath_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        self.training_browse_btn = ttk.Button(train_frame, text="Browse", command=self.browse_training_file)
        self.training_browse_btn.grid(row=3, column=2, padx=5, pady=5)

        ttk.Label(train_frame, text="Environment:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        
        self.train_mode = tk.StringVar(value="azure")
        self.azure_radio = ttk.Radiobutton(
            train_frame,
            text="Azure Cloud",
            variable=self.train_mode,
            value="azure",
            command=self.on_train_mode_change,
        )
        self.azure_radio.grid(row=4, column=1, sticky="w", padx=5)
        self.local_radio = ttk.Radiobutton(
            train_frame,
            text="Local Device (CPU/GPU)",
            variable=self.train_mode,
            value="local",
            command=self.on_train_mode_change,
        )
        self.local_radio.grid(row=4, column=2, sticky="w", padx=5)

        ttk.Label(train_frame, text="Local Device:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.local_device_var = tk.StringVar(value="auto")
        self.local_device_combo = ttk.Combobox(
            train_frame,
            textvariable=self.local_device_var,
            state="readonly",
            values=["auto", "cpu", "cuda"],
            width=16,
        )
        self.local_device_combo.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.local_device_combo.bind("<<ComboboxSelected>>", self.on_local_device_change)

        self.local_runtime_label = ttk.Label(train_frame, text="Local Runtime:")
        self.local_runtime_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.local_runtime_var = tk.StringVar(value="host")
        self.local_runtime_combo = ttk.Combobox(
            train_frame,
            textvariable=self.local_runtime_var,
            state="disabled",
            values=["host", "container"],
            width=16,
        )
        self.local_runtime_combo.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        self.get_model_btn = ttk.Button(train_frame, text="Get Model (Train)", command=self.start_training_thread)
        self.get_model_btn.grid(row=7, column=0, columnspan=3, pady=10)

        # --- Hosting Section ---
        hosting_frame = ttk.LabelFrame(self.root, text="Hosting", padding=(10, 10))
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

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        self._last_local_device = self.local_device_var.get()
        self.on_train_mode_change()

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

    def on_train_mode_change(self):
        is_azure = self.train_mode.get() == "azure"
        azure_state = "normal" if is_azure else "disabled"
        azure_compute_state = "readonly" if is_azure else "disabled"
        local_state = "disabled" if is_azure else "readonly"

        self.azure_sub_entry.config(state=azure_state)
        self.azure_tenant_entry.config(state=azure_state)
        self.azure_compute_combo.config(state=azure_compute_state)
        self.local_device_combo.config(state=local_state)

        if is_azure:
            self.local_runtime_label.grid_remove()
            self.local_runtime_combo.grid_remove()
            self.local_runtime_var.set("host")
            self.local_runtime_combo.config(state="disabled")
        else:
            self.local_runtime_label.grid()
            self.local_runtime_combo.grid()
            self.on_local_device_change()

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

        self.status_var.set("Processing data with GPT-4o...")
        self.prepare_btn.config(state="disabled")
        threading.Thread(target=self.process_logs_llm, args=(input_path, save_path, token), daemon=True).start()

    def process_logs_llm(self, input_path, save_path, token):
        processed_logs = []
        try:
            system_prompt = self.load_prompt()
            df = pd.read_csv(input_path)
            
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

        print("\n--- [DEBUG] STARTING TRAINING WORKFLOW ---")
        print(f"[DEBUG] CSV Path: {csv_path}")
        print(f"[DEBUG] Sub ID: {sub_id}")
        print(f"[DEBUG] Tenant ID: {tenant_id}")
        print(f"[DEBUG] Azure Compute: {azure_compute}")
        print(f"[DEBUG] Train Mode: {self.train_mode.get()}")
        print(f"[DEBUG] Local Device: {local_device}")
        print(f"[DEBUG] Local Runtime: {local_runtime}")

        if not csv_path:
            messagebox.showwarning("Warning", "Please select a labeled CSV file for training.")
            return
        if not os.path.exists(csv_path):
            messagebox.showwarning("Warning", "The selected labeled CSV file does not exist.")
            return
        if self.train_mode.get() == "azure" and not sub_id:
            messagebox.showwarning("Warning", "Please provide your Azure Subscription ID.")
            return
        if self.train_mode.get() == "azure" and not tenant_id:
            messagebox.showwarning("Warning", "Please provide your Azure Tenant ID.")
            return
        if not os.path.exists("train.py"):
            messagebox.showerror("Error", "Could not find 'train.py' in the app directory.")
            return

        if self.train_mode.get() == "azure":
            self.get_model_btn.config(state="disabled")
            self.status_var.set("Initializing authentication...")
            threading.Thread(
                target=self.run_azure_training,
                args=(csv_path, sub_id, tenant_id, azure_compute),
                daemon=True,
            ).start()
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

            self.get_model_btn.config(state="disabled")
            self.status_var.set("Preparing local training...")
            threading.Thread(
                target=self.run_local_training,
                args=(csv_path, local_device, local_runtime),
                daemon=True,
            ).start()

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

    def run_local_training(self, csv_path, device, runtime):
        try:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            backend = "container" if runtime == "container" else "host"

            if backend == "container":
                docker_script = os.path.join(project_dir, "scripts", "train_docker.sh")
                if not os.path.exists(docker_script):
                    raise FileNotFoundError("Could not find scripts/train_docker.sh in the app directory.")
                command_args = ["bash", docker_script, csv_path]
                process_env = os.environ.copy()
                process_env["DEVICE"] = device
            else:
                train_script = os.path.join(project_dir, "train.py")
                if not os.path.exists(train_script):
                    raise FileNotFoundError("Could not find 'train.py' in the app directory.")
                command_args = [sys.executable, train_script, "--data", csv_path]
                process_env = os.environ.copy()
                if device == "cpu":
                    process_env["CUDA_VISIBLE_DEVICES"] = "-1"

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

            if process.stdout:
                for line in process.stdout:
                    clean_line = line.strip()
                    if clean_line:
                        print(f"[LOCAL-TRAIN] {clean_line}")
                        self.root.after(
                            0,
                            lambda msg=clean_line: self.status_var.set(msg[:150]),
                        )

            return_code = process.wait()
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
        except Exception as e:
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
            self.root.after(0, lambda: self.get_model_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def run_azure_training(self, csv_path, sub_id, tenant_id, azure_compute):
        ml_client = None
        compute_mode = "gpu" if azure_compute == "gpu" else "cpu"
        if compute_mode == "gpu":
            compute_name = "gpu-cluster-temp"
            compute_size = "Standard_NC4as_T4_v3"
        else:
            compute_name = "cpu-cluster-temp"
            compute_size = "Standard_D2as_v4"
        compute_created = False
        resource_group = "LogClassifier-RG"
        workspace_name = "LogClassifier-Workspace"

        try:
            print(f"[DEBUG] Attempting Interactive Browser Login to Tenant: {tenant_id}")
            self.root.after(0, lambda: self.status_var.set("Please log in to Azure in your web browser..."))
            
            # Authenticate directly to the correct Tenant
            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
            ml_client = MLClient(credential, sub_id, resource_group, workspace_name)
            print("[DEBUG] MLClient instantiated successfully.")

            # --- Check / Create Resource Group ---
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
            print("[DEBUG] Verifying Microsoft.MachineLearningServices provider registration...")
            self.root.after(0, lambda: self.status_var.set("Verifying Azure ML registration..."))
            resource_client.providers.register('Microsoft.MachineLearningServices')
            
            while True:
                provider_info = resource_client.providers.get('Microsoft.MachineLearningServices')
                print(f"[DEBUG] Provider State: {provider_info.registration_state}")
                if provider_info.registration_state == 'Registered':
                    print("[DEBUG] SUCCESS: Provider is fully Registered.")
                    break
                self.root.after(0, lambda: self.status_var.set("Activating ML Services on Azure (takes 1-2 mins)..."))
                time.sleep(10)

            # --- Check / Create Workspace ---
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

            # --- Create Compute Cluster ---
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
                    "&& USE_TF=0 python train.py --data ${{inputs.training_data}}"
                ),
                experiment_name="deberta-log-classification",
            )

            print("[DEBUG] Submitting job to Azure...")
            returned_job = ml_client.jobs.create_or_update(job)
            print(f"[DEBUG] SUCCESS: Job submitted! Job Name: {returned_job.name}")

            # --- Polling Loop ---
            print("[DEBUG] Beginning polling loop for job completion...")
            while True:
                job_status = ml_client.jobs.get(returned_job.name).status
                print(f"[DEBUG] Azure Job Status: {job_status}")
                self.root.after(0, lambda s=job_status: self.status_var.set(f"Training in progress. Azure Status: {s}"))
                if job_status in ["Completed", "Failed", "Canceled"]:
                    break
                time.sleep(30)

            if job_status == "Failed":
                print("[DEBUG] FATAL: Job failed on the Azure side.")
                raise Exception("The training job failed on the Azure machine. Check Azure Portal logs.")

            # --- Download Model ---
            print("[DEBUG] Job Completed! Attempting to download outputs...")
            self.root.after(0, lambda: self.status_var.set("Training Complete! Downloading model..."))
            download_path = "./downloaded_model"
            ml_client.jobs.download(name=returned_job.name, download_path=download_path, all=False)
            print(f"[DEBUG] SUCCESS: Files downloaded to {download_path}")
            
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Model trained and downloaded to:\n{os.path.abspath(download_path)}"))

        except Exception as e:
            print("\n--- [DEBUG] AN EXCEPTION OCCURRED ---")
            traceback.print_exc() 
            print("--------------------------------------\n")
            
            self.root.after(0, lambda err=str(e): messagebox.showerror("Training Error", f"An error occurred. Check the terminal for full details.\n\n{err}"))
            self.root.after(0, lambda: self.status_var.set("Process halted due to error."))

        finally:
            # --- THE CRITICAL CLEANUP BLOCK ---
            print("[DEBUG] Entering cleanup block...")
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
            
            print("[DEBUG] WORKFLOW FINISHED.\n")
            self.root.after(0, lambda: self.status_var.set("Ready"))
            self.root.after(0, lambda: self.get_model_btn.config(state="normal"))

    def show_error(self, message):
        self.status_var.set("Process halted.")
        messagebox.showerror("Notice", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = LogProcessorApp(root)
    root.mainloop()
