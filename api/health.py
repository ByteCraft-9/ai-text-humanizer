"""GET /api/py/health — which model artefacts this deployment actually has.

The app is designed to deploy before the models are trained, so "is anything
missing?" is a real question with a useful answer rather than a formality.
The UI reads this to decide whether to show the degraded-mode notice.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.runtime import MODELS_DIR, available  # noqa: E402


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel requires this name
    def do_GET(self) -> None:  # noqa: N802
        models = available()
        payload = {
            "ok": True,
            "models": models,
            "models_dir": str(MODELS_DIR),
            "fully_provisioned": all(models.values()),
            "runtime": sys.version.split()[0],
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Silence the default access log."""
