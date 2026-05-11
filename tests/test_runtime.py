from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from app_core.contracts import JOB_STATUS_CANCELED, JOB_STATUS_FAILED, JOB_STATUS_SUCCEEDED
from app_core.runtime import ArtifactStore, JobCancelled, JobManager, StateStore


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.artifact_store = ArtifactStore(str(self.project_dir))
        self.state_store = StateStore(self.artifact_store.state_db_path)
        self.job_manager = JobManager(self.state_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def wait_for_terminal_event(self, job_id: str, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for event in self.job_manager.drain_events():
                if event.job_id == job_id and event.status in {JOB_STATUS_SUCCEEDED, JOB_STATUS_CANCELED, "failed"}:
                    return event
            time.sleep(0.05)
        self.fail(f"Timed out waiting for terminal event for job {job_id}")

    def test_state_store_round_trip(self):
        self.state_store.set_value("active", {"mode": "local"})
        self.assertEqual(self.state_store.get_value("active"), {"mode": "local"})

    def test_job_manager_success_path(self):
        record = self.job_manager.submit("unit_test", lambda ctx: {"message": "done", "answer": 42})
        event = self.wait_for_terminal_event(record.job_id)
        self.assertEqual(event.status, JOB_STATUS_SUCCEEDED)
        job = self.state_store.get_job(record.job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JOB_STATUS_SUCCEEDED)
        self.assertEqual(job.result["answer"], 42)

    def test_job_manager_cancel_path(self):
        def long_running(ctx):
            while True:
                ctx.check_cancelled()
                time.sleep(0.05)

        record = self.job_manager.submit("cancel_test", long_running)
        time.sleep(0.1)
        self.assertTrue(self.job_manager.cancel_job(record.job_id))
        event = self.wait_for_terminal_event(record.job_id)
        self.assertEqual(event.status, JOB_STATUS_CANCELED)
        job = self.state_store.get_job(record.job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JOB_STATUS_CANCELED)

    def test_job_manager_terminates_registered_subprocesses_on_failure(self):
        class FakeProcess:
            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return 0 if self.terminated or self.killed else None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.killed = True

        seen = {}

        def failing_handler(ctx):
            process = FakeProcess()
            seen["process"] = process
            ctx.register_subprocess("child", process)
            raise RuntimeError("boom")

        record = self.job_manager.submit("failure_cleanup_test", failing_handler)
        event = self.wait_for_terminal_event(record.job_id)

        self.assertEqual(event.status, JOB_STATUS_FAILED)
        self.assertTrue(seen["process"].terminated)
        job = self.state_store.get_job(record.job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JOB_STATUS_FAILED)


if __name__ == "__main__":
    unittest.main()
