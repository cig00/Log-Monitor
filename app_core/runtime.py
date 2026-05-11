from __future__ import annotations

import json
import queue
import sqlite3
import threading
import traceback
import uuid
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any, Callable

from mlops_utils import clean_optional_string, now_utc_iso, read_json, write_json

from .contracts import (
    JOB_STATUS_CANCELED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    ArtifactRef,
    JobRecord,
    ProgressEvent,
)


class JobCancelled(Exception):
    """Raised when a managed background job is canceled."""


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    error TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_job(self, record: JobRecord) -> JobRecord:
        payload = asdict(record)
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, status, submitted_at, started_at, finished_at,
                    error, result_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type=excluded.job_type,
                    status=excluded.status,
                    submitted_at=excluded.submitted_at,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    error=excluded.error,
                    result_json=excluded.result_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    payload["job_id"],
                    payload["job_type"],
                    payload["status"],
                    payload["submitted_at"],
                    payload["started_at"],
                    payload["finished_at"],
                    payload["error"],
                    json.dumps(payload["result"], sort_keys=True),
                    json.dumps(payload["metadata"], sort_keys=True),
                ),
            )
            conn.commit()
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord(
            job_id=row["job_id"],
            job_type=row["job_type"],
            status=row["status"],
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error=row["error"],
            result=json.loads(row["result_json"] or "{}"),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def set_value(self, key: str, value: Any) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO kv_state(key, value_json) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
                """,
                (key, json.dumps(value, sort_keys=True)),
            )
            conn.commit()

    def get_value(self, key: str, default: Any = None) -> Any:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT value_json FROM kv_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default


class ArtifactStore:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.outputs_dir = self.project_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_db_path(self) -> str:
        return str((self.outputs_dir / "runtime_state.sqlite3").resolve())

    @property
    def last_hosting_path(self) -> str:
        return str((self.outputs_dir / "last_hosting.json").resolve())

    @property
    def last_training_metadata_path(self) -> str:
        return str((self.outputs_dir / "last_training_mlflow.json").resolve())

    @property
    def local_observability_root(self) -> str:
        return str((self.outputs_dir / "local_observability").resolve())

    def write_last_hosting_metadata(self, payload: dict[str, Any]) -> ArtifactRef:
        write_json(self.last_hosting_path, payload)
        return ArtifactRef(
            artifact_id="last_hosting",
            kind="hosting_metadata",
            path=self.last_hosting_path,
            metadata=dict(payload),
        )

    def read_last_hosting_metadata(self) -> dict[str, Any]:
        return read_json(self.last_hosting_path) or {}


class JobContext:
    def __init__(
        self,
        manager: "JobManager",
        job_id: str,
        job_type: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.manager = manager
        self.job_id = job_id
        self.job_type = job_type
        self.metadata = dict(metadata or {})
        self.cancel_event = threading.Event()
        self._subprocesses: dict[str, Any] = {}
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def emit(
        self,
        stage: str,
        message: str,
        percent: float | None = None,
        status: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = ProgressEvent(
            job_id=self.job_id,
            stage=stage,
            message=message,
            percent=percent,
            timestamp=now_utc_iso(),
            status=status,
            payload=dict(payload or {}),
        )
        self.manager._events.put(event)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("Job cancellation was requested.")

    def register_subprocess(self, name: str, process: Any) -> None:
        with self._lock:
            self._subprocesses[name] = process

    def clear_subprocess(self, name: str) -> None:
        with self._lock:
            self._subprocesses.pop(name, None)

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            subprocesses = list(self._subprocesses.values())
            callbacks = list(self._cleanup_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                traceback.print_exc()
        for process in subprocesses:
            if process is None:
                continue
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except TimeoutExpired:
                        process.kill()
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


class JobManager:
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self._events: "queue.Queue[ProgressEvent]" = queue.Queue()
        self._contexts: dict[str, JobContext] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        job_type: str,
        handler: Callable[[JobContext], dict[str, Any] | None],
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
        submitted_at = now_utc_iso()
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            submitted_at=submitted_at,
            metadata=dict(metadata or {}),
        )
        self.state_store.save_job(record)
        context = JobContext(self, job_id, job_type, metadata=metadata)
        with self._lock:
            self._contexts[job_id] = context

        def runner() -> None:
            started_record = JobRecord(
                job_id=job_id,
                job_type=job_type,
                status=JOB_STATUS_RUNNING,
                submitted_at=submitted_at,
                started_at=now_utc_iso(),
                metadata=dict(metadata or {}),
            )
            self.state_store.save_job(started_record)
            context.emit("started", f"{job_type.replace('_', ' ').title()} started.", status=JOB_STATUS_RUNNING)
            try:
                result = dict(handler(context) or {})
                finished_record = JobRecord(
                    job_id=job_id,
                    job_type=job_type,
                    status=JOB_STATUS_SUCCEEDED,
                    submitted_at=submitted_at,
                    started_at=started_record.started_at,
                    finished_at=now_utc_iso(),
                    result=result,
                    metadata=dict(metadata or {}),
                )
                self.state_store.save_job(finished_record)
                context.emit(
                    "finished",
                    clean_optional_string(result.get("message")) or f"{job_type.replace('_', ' ').title()} completed.",
                    status=JOB_STATUS_SUCCEEDED,
                    payload=result,
                )
            except JobCancelled as exc:
                context.cancel()
                canceled_record = JobRecord(
                    job_id=job_id,
                    job_type=job_type,
                    status=JOB_STATUS_CANCELED,
                    submitted_at=submitted_at,
                    started_at=started_record.started_at,
                    finished_at=now_utc_iso(),
                    error=str(exc),
                    metadata=dict(metadata or {}),
                )
                self.state_store.save_job(canceled_record)
                context.emit(
                    "finished",
                    str(exc) or "Operation canceled.",
                    status=JOB_STATUS_CANCELED,
                    payload={"message": str(exc)},
                )
            except Exception as exc:
                context.cancel()
                failed_record = JobRecord(
                    job_id=job_id,
                    job_type=job_type,
                    status=JOB_STATUS_FAILED,
                    submitted_at=submitted_at,
                    started_at=started_record.started_at,
                    finished_at=now_utc_iso(),
                    error=str(exc),
                    metadata=dict(metadata or {}),
                )
                self.state_store.save_job(failed_record)
                context.emit(
                    "finished",
                    str(exc),
                    status=JOB_STATUS_FAILED,
                    payload={"message": str(exc), "traceback": traceback.format_exc()},
                )
            finally:
                with self._lock:
                    self._threads.pop(job_id, None)
                    self._contexts.pop(job_id, None)

        thread = threading.Thread(target=runner, daemon=True)
        with self._lock:
            self._threads[job_id] = thread
        thread.start()
        return record

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            context = self._contexts.get(job_id)
        if context is None:
            return False
        context.cancel()
        context.emit("cancel_requested", "Cancellation requested.", status=JOB_STATUS_RUNNING)
        return True

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.state_store.get_job(job_id)

    def drain_events(self) -> list[ProgressEvent]:
        events: list[ProgressEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events
