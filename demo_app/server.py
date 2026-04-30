from base64 import b64encode
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from json import dumps, loads
from os import environ
from pathlib import Path
from re import match
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        environ.setdefault(key.strip(), value.strip())


class LogFinderHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/jira/comment":
            self.send_json({"error": "Not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = loads(self.rfile.read(content_length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_json({"error": "Invalid JSON body"}, status=400)
            return

        issue_key = str(payload.get("issueKey", "")).strip().upper()
        comment = str(payload.get("comment", "")).strip()

        if not match(r"^[A-Z][A-Z0-9]+-\d+$", issue_key):
            self.send_json(
                {"error": "Use a full Jira issue key, for example KAN-123."},
                status=400,
            )
            return

        if not comment:
            self.send_json({"error": "Report text is empty."}, status=400)
            return

        base_url = environ.get("JIRA_BASE_URL", "").rstrip("/")
        email = environ.get("JIRA_EMAIL", "")
        token = environ.get("JIRA_API_TOKEN", "")

        if not base_url or not email or not token:
            self.send_json({"error": "Jira credentials are not configured."}, status=500)
            return

        response = post_jira_comment(base_url, email, token, issue_key, comment)
        self.send_json(response, status=response.pop("status", 200))

    def send_json(self, payload, status=200):
        body = dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def post_jira_comment(base_url, email, token, issue_key, comment):
    auth = b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    body = dumps({"body": to_adf(comment)}).encode("utf-8")
    request = Request(
        f"{base_url}/rest/api/3/issue/{issue_key}/comment",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Basic {auth}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            data = loads(response.read().decode("utf-8"))
            return {
                "status": response.status,
                "id": data.get("id"),
                "self": data.get("self"),
                "message": "Jira comment added.",
            }
    except HTTPError as error:
        return {
            "status": error.code,
            "error": jira_error_message(error),
        }
    except URLError as error:
        return {
            "status": 502,
            "error": f"Could not reach Jira: {error.reason}",
        }


def jira_error_message(error):
    try:
        data = loads(error.read().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return f"Jira returned HTTP {error.code}."

    messages = data.get("errorMessages") or []
    field_errors = data.get("errors") or {}
    if messages:
        return " ".join(messages)
    if field_errors:
        return " ".join(f"{key}: {value}" for key, value in field_errors.items())
    return f"Jira returned HTTP {error.code}."


def to_adf(text):
    content = []
    for line in text.splitlines():
        content.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": line}] if line else [],
            }
        )

    return {
        "type": "doc",
        "version": 1,
        "content": content or [{"type": "paragraph", "content": []}],
    }


if __name__ == "__main__":
    load_env()
    server = ThreadingHTTPServer(("", 8000), LogFinderHandler)
    print("Serving Log Finder on http://localhost:8000")
    server.serve_forever()
