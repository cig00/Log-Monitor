from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_core.log_regression_service import (
    CommitDiff,
    LogOccurrence,
    ask_copilot_about_diffs,
    extract_timestamp,
    find_first_occurrence,
)


class LogRegressionServiceTests(unittest.TestCase):
    def test_extract_timestamp_supports_common_log_formats(self):
        self.assertEqual(
            extract_timestamp("2026-04-24T10:11:12Z ERROR failed").isoformat(),
            "2026-04-24T10:11:12+00:00",
        )
        self.assertEqual(
            extract_timestamp('[10/Oct/2025:13:55:36 -0700] "GET /" 500').isoformat(),
            "2025-10-10T13:55:36-07:00",
        )

    def test_find_first_occurrence_returns_earliest_matching_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "app.log"
            log_file.write_text(
                "\n".join(
                    [
                        "2026-04-24T10:00:00Z INFO started",
                        "2026-04-24T10:01:00Z ERROR database timeout",
                        "2026-04-24T10:02:00Z ERROR database timeout",
                    ]
                ),
                encoding="utf-8",
            )

            occurrence = find_first_occurrence(log_file, "database timeout")

        self.assertEqual(occurrence.line_number, 2)
        self.assertEqual(occurrence.timestamp, "2026-04-24T10:01:00+00:00")
        self.assertEqual(occurrence.timestamp_source, "log_line")

    def test_copilot_verdict_parses_json_from_command_output(self):
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = "Result:\n" + json.dumps(
            {
                "related": True,
                "impacted_commit": "abc123",
                "confidence": "high",
                "rationale": "The diff changed the failing path.",
            }
        )
        completed.stderr = ""
        occurrence = LogOccurrence("app.log", 12, "ERROR failed", "2026-04-24T10:00:00+00:00", "log_line")
        with mock.patch("app_core.log_regression_service.subprocess.run", return_value=completed):
            verdict = ask_copilot_about_diffs(
                "ERROR failed",
                occurrence,
                [CommitDiff("abc123", "A", "2026-04-24T10:00:00Z", "change", "diff")],
                command="copilot-test",
            )

        self.assertTrue(verdict.related)
        self.assertEqual(verdict.impacted_commit, "abc123")
        self.assertIsNone(verdict.error)


if __name__ == "__main__":
    unittest.main()
