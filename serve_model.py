import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inference_utils import load_model_bundle, predict_error_message


class PredictionHandler(BaseHTTPRequestHandler):
    bundle = None

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

    def _write_method_error(self) -> None:
        self._write_json(
            405,
            {
                "prediction": "",
                "message": "Use POST /predict with JSON body {'errorMessage': ''}.",
            },
        )

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path.rstrip("/") == "":
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
            return
        if self.path.rstrip("/") == "/health":
            self._write_json(200, {"status": "ok"})
            return
        if self.path.rstrip("/") == "/predict":
            self._write_method_error()
            return
        self._write_json(404, {"prediction": ""})

    def do_POST(self):
        if self.path.rstrip("/") != "/predict":
            self._write_json(404, {"prediction": ""})
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._write_json(400, {"prediction": ""})
            return

        if not isinstance(payload, dict):
            self._write_json(400, {"prediction": ""})
            return

        error_message = str(payload.get("errorMessage", ""))
        prediction = predict_error_message(self.bundle, error_message)
        self._write_json(200, {"prediction": prediction})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    PredictionHandler.bundle = load_model_bundle(args.model_dir)
    server = ThreadingHTTPServer((args.host, args.port), PredictionHandler)
    print(f"Prediction API ready at http://{args.host}:{args.port}/predict", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
