import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from inference_utils import load_model_bundle, predict_error_message


class MetricsStore:
    def __init__(self):
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._request_counts = {}
        self._request_duration_sums = {}
        self._request_duration_counts = {}
        self._prediction_counts = {}

    @staticmethod
    def _escape_label(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    def _format_labels(self, labels: dict) -> str:
        if not labels:
            return ""
        parts = []
        for key, value in labels.items():
            parts.append(f'{key}="{self._escape_label(value)}"')
        return "{" + ",".join(parts) + "}"

    def record_request(self, method: str, path: str, status: int, duration_seconds: float, prediction: str = "") -> None:
        request_key = (method, path, str(status))
        duration_key = (method, path)
        clean_prediction = str(prediction).strip()
        with self._lock:
            self._request_counts[request_key] = self._request_counts.get(request_key, 0) + 1
            self._request_duration_sums[duration_key] = self._request_duration_sums.get(duration_key, 0.0) + max(
                float(duration_seconds),
                0.0,
            )
            self._request_duration_counts[duration_key] = self._request_duration_counts.get(duration_key, 0) + 1
            if clean_prediction:
                prediction_key = (clean_prediction,)
                self._prediction_counts[prediction_key] = self._prediction_counts.get(prediction_key, 0) + 1

    def render(self, bundle_loaded: bool) -> str:
        with self._lock:
            request_counts = dict(self._request_counts)
            request_duration_sums = dict(self._request_duration_sums)
            request_duration_counts = dict(self._request_duration_counts)
            prediction_counts = dict(self._prediction_counts)

        lines = [
            "# HELP log_monitor_http_requests_total Total HTTP requests served by the local prediction API.",
            "# TYPE log_monitor_http_requests_total counter",
        ]
        for (method, path, status), count in sorted(request_counts.items()):
            lines.append(
                "log_monitor_http_requests_total"
                f'{self._format_labels({"method": method, "path": path, "status": status})} {count}'
            )

        lines.extend(
            [
                "# HELP log_monitor_request_duration_seconds Request duration summary by method and path.",
                "# TYPE log_monitor_request_duration_seconds summary",
            ]
        )
        for key in sorted(request_duration_sums.keys()):
            method, path = key
            labels = self._format_labels({"method": method, "path": path})
            lines.append(f"log_monitor_request_duration_seconds_sum{labels} {request_duration_sums[key]:.6f}")
            lines.append(f"log_monitor_request_duration_seconds_count{labels} {request_duration_counts.get(key, 0)}")

        lines.extend(
            [
                "# HELP log_monitor_predictions_total Total prediction responses returned by label.",
                "# TYPE log_monitor_predictions_total counter",
            ]
        )
        for (prediction,), count in sorted(prediction_counts.items()):
            lines.append(
                "log_monitor_predictions_total"
                f'{self._format_labels({"prediction": prediction})} {count}'
            )

        uptime_seconds = max(time.time() - self.started_at, 0.0)
        lines.extend(
            [
                "# HELP log_monitor_api_up Whether the local prediction API process is running.",
                "# TYPE log_monitor_api_up gauge",
                "log_monitor_api_up 1",
                "# HELP log_monitor_model_loaded Whether the model bundle is loaded in memory.",
                "# TYPE log_monitor_model_loaded gauge",
                f"log_monitor_model_loaded {1 if bundle_loaded else 0}",
                "# HELP log_monitor_process_uptime_seconds Seconds since the local prediction API started.",
                "# TYPE log_monitor_process_uptime_seconds gauge",
                f"log_monitor_process_uptime_seconds {uptime_seconds:.3f}",
            ]
        )
        return "\n".join(lines) + "\n"


METRICS = MetricsStore()


class PredictionHandler(BaseHTTPRequestHandler):
    bundle = None
    bundle_error = ""
    model_dir = ""
    active_model_dir_file = ""
    bundle_lock = threading.Lock()

    @classmethod
    def resolve_model_dir(cls) -> str:
        active_file = str(cls.active_model_dir_file or "").strip()
        if active_file:
            try:
                candidate = Path(active_file)
                if candidate.exists():
                    value = candidate.read_text(encoding="utf-8").strip()
                    if value:
                        return value
            except Exception:
                pass
        return str(cls.model_dir or "").strip()

    @classmethod
    def reload_bundle(cls) -> tuple[bool, str]:
        model_dir = cls.resolve_model_dir()
        if not model_dir:
            with cls.bundle_lock:
                cls.bundle = None
                cls.bundle_error = "No model directory is configured."
            return False, cls.bundle_error
        try:
            bundle = load_model_bundle(model_dir)
            with cls.bundle_lock:
                cls.bundle = bundle
                cls.bundle_error = ""
                cls.model_dir = model_dir
            return True, ""
        except Exception as exc:
            with cls.bundle_lock:
                cls.bundle = None
                cls.bundle_error = str(exc)
            return False, str(exc)

    @classmethod
    def unload_bundle(cls) -> None:
        with cls.bundle_lock:
            cls.bundle = None
            cls.bundle_error = "Model unloaded."

    def _normalized_path(self) -> str:
        parsed = urlparse(self.path or "/")
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
        return path or "/"

    def _write_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, status_code: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_text(self, status_code: int, body: str, content_type: str = "text/plain; version=0.0.4; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_method_error(self) -> None:
        self._write_json(
            405,
            {
                "prediction": "",
                "message": "Use POST /predict with JSON body {'errorMessage': ''}.",
            },
        )

    def _record_request(self, started_at: float, status_code: int, prediction: str = "") -> None:
        METRICS.record_request(
            method=self.command,
            path=self._normalized_path(),
            status=status_code,
            duration_seconds=time.time() - started_at,
            prediction=prediction,
        )

    def log_message(self, format, *args):
        return

    def do_GET(self):
        started_at = time.time()
        path = self._normalized_path()
        if path == "/":
            host = self.headers.get("Host", "127.0.0.1")
            self._write_html(
                200,
                f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Prediction API</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f7f1e7 0%, #f1ece2 100%);
      color: #1f2937;
    }}
    main {{
      max-width: 860px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }}
    .card {{
      background: rgba(255,255,255,0.88);
      border: 1px solid #ddd2c0;
      border-radius: 22px;
      padding: 24px;
      box-shadow: 0 16px 36px rgba(62, 49, 33, 0.08);
    }}
    h1 {{ margin-top: 0; font-size: 42px; }}
    p {{ line-height: 1.6; }}
    pre {{
      background: #201c1a;
      color: #f5efe6;
      border-radius: 16px;
      padding: 14px;
      overflow-x: auto;
    }}
    a {{ color: #0f766e; }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Local Prediction API</h1>
      <p>This service accepts a JSON <code>POST</code> request at <a href="http://{host}/predict">http://{host}/predict</a>.</p>
      <p>Health check: <a href="http://{host}/health">http://{host}/health</a></p>
      <p>Prometheus metrics: <a href="http://{host}/metrics">http://{host}/metrics</a></p>
      <pre>{{
  "errorMessage": ""
}}</pre>
      <pre>{{
  "prediction": ""
}}</pre>
    </section>
  </main>
</body>
</html>""",
            )
            self._record_request(started_at, 200)
            return
        if path == "/health":
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model_loaded": self.bundle is not None,
                    "model_dir": self.resolve_model_dir(),
                    "message": self.bundle_error,
                },
            )
            self._record_request(started_at, 200)
            return
        if path == "/metrics":
            self._write_text(200, METRICS.render(bundle_loaded=self.bundle is not None))
            self._record_request(started_at, 200)
            return
        if path == "/predict":
            self._write_method_error()
            self._record_request(started_at, 405)
            return
        self._write_json(404, {"prediction": ""})
        self._record_request(started_at, 404)

    def do_POST(self):
        started_at = time.time()
        path = self._normalized_path()
        if path == "/reload":
            ok, error = self.reload_bundle()
            status = 200 if ok else 500
            self._write_json(
                status,
                {
                    "status": "loaded" if ok else "failed",
                    "model_loaded": ok,
                    "model_dir": self.resolve_model_dir(),
                    "message": error,
                },
            )
            self._record_request(started_at, status)
            return
        if path == "/unload":
            self.unload_bundle()
            self._write_json(200, {"status": "unloaded", "model_loaded": False})
            self._record_request(started_at, 200)
            return

        self._handle_predict_post(started_at, path)

    def _handle_predict_post(self, started_at: float, path: str) -> None:
        if path != "/predict":
            self._write_json(404, {"prediction": ""})
            self._record_request(started_at, 404)
            return

        with self.bundle_lock:
            bundle = self.bundle
            bundle_error = self.bundle_error
        if bundle is None:
            self._write_json(
                503,
                {
                    "prediction": "",
                    "message": bundle_error or "Model is not loaded yet. Train or select a model, then host it.",
                },
            )
            self._record_request(started_at, 503)
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._write_json(400, {"prediction": ""})
            self._record_request(started_at, 400)
            return

        if not isinstance(payload, dict):
            self._write_json(400, {"prediction": ""})
            self._record_request(started_at, 400)
            return

        try:
            error_message = str(payload.get("errorMessage", ""))
            prediction = predict_error_message(bundle, error_message)
            self._write_json(200, {"prediction": prediction})
            self._record_request(started_at, 200, prediction=prediction)
        except Exception:
            self._write_json(500, {"prediction": "", "message": "Prediction failed."})
            self._record_request(started_at, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/workspace/outputs/final_model"))
    parser.add_argument("--active-model-dir-file", default=os.environ.get("ACTIVE_MODEL_DIR_FILE", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    PredictionHandler.model_dir = args.model_dir
    PredictionHandler.active_model_dir_file = args.active_model_dir_file
    ok, error = PredictionHandler.reload_bundle()
    if ok:
        print(f"Loaded model from {PredictionHandler.resolve_model_dir()}", flush=True)
    else:
        print(f"Prediction API starting without a loaded model: {error}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), PredictionHandler)
    print(f"Prediction API ready at http://{args.host}:{args.port}/predict", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
