#!/usr/bin/env python3
"""Serve the `api/*.py` functions locally for development.

`next dev` only serves the TypeScript routes, and `vercel dev` needs a login.
This runs the same handler classes Vercel runs, on one port, so the whole app
works locally with `npm run dev`.

In production Vercel routes `/api/py/*` to these files directly; in
development `next.config.mjs` rewrites to this server. The handler code is
identical either way — this file only provides the socket.

    python scripts/dev_python.py --port 8000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
sys.path.insert(0, str(API_DIR))

_handlers: dict[str, type[BaseHTTPRequestHandler]] = {}


def load_handler(name: str) -> type[BaseHTTPRequestHandler] | None:
    """Import `api/<name>.py` and return its `handler` class."""
    if name in _handlers:
        return _handlers[name]

    path = API_DIR / f"{name}.py"
    if not path.is_file() or name.startswith("_"):
        return None

    spec = importlib.util.spec_from_file_location(f"api_{name}", path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        traceback.print_exc()
        return None

    handler = getattr(module, "handler", None)
    if handler is None:
        return None

    _handlers[name] = handler
    return handler


class Router(BaseHTTPRequestHandler):
    """Dispatch `/api/<name>` to that file's handler.

    The handler classes are plain ``BaseHTTPRequestHandler`` subclasses, so
    their ``do_POST`` / ``do_GET`` can be called against this instance
    directly — same code path, no adapter shims that could mask a bug that
    would then only appear in production.
    """

    server_version = "aidh-dev/1.0"

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?", 1)[0].strip("/")
        parts = path.split("/")

        # Accept both /api/<name> and /api/py/<name> so the same URL works
        # whether or not the caller went through the rewrite.
        if parts and parts[0] == "api":
            parts = parts[1:]
        if parts and parts[0] == "py":
            parts = parts[1:]

        name = parts[0] if parts else ""
        handler_class = load_handler(name)

        if handler_class is None:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"no such function"}')
            return

        implementation = getattr(handler_class, f"do_{method}", None)
        if implementation is None:
            self.send_response(405)
            self.end_headers()
            return

        try:
            implementation(self)
        except Exception:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"handler raised"}')
            except OSError:
                pass

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def log_message(self, fmt: str, *args) -> None:
        # One concise line per request; the default logs the full path twice.
        sys.stderr.write(f"  py  {self.path} {args[1] if len(args) > 1 else ''}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    available = sorted(p.stem for p in API_DIR.glob("*.py") if not p.stem.startswith("_"))
    print(f"Python functions on http://{args.host}:{args.port}")
    for name in available:
        print(f"  /api/py/{name}")

    from _lib.runtime import available as models_available  # noqa: PLC0415

    present = models_available()
    missing = [k for k, v in present.items() if not v]
    if missing:
        print(f"\nMissing model weights: {', '.join(missing)}")
        print("Scores will be degraded and the UI will say so.")
        print("Run: python scripts/fetch_models.py --base-models\n")

    ThreadingHTTPServer((args.host, args.port), Router).serve_forever()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
