"""Small helpers shared by the Vercel Python functions.

Vercel's Python runtime routes to a ``BaseHTTPRequestHandler`` subclass named
``handler``. These helpers keep each function file down to its actual logic.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable

# Vercel caps request bodies at 4.5 MB, but 5,000 words is ~35 KB. Anything
# far above that is either a mistake or an attempt to burn our compute, so it
# is refused before parsing (PRD 15.2).
MAX_BODY_BYTES = 512 * 1024


class BadRequest(Exception):
    """Raised for malformed input; becomes a 400 with a readable message."""


def read_json(request: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(request.headers.get("content-length") or 0)
    if length <= 0:
        raise BadRequest("Request body is empty.")
    if length > MAX_BODY_BYTES:
        raise BadRequest(
            f"Request body is {length // 1024} KB; the limit is "
            f"{MAX_BODY_BYTES // 1024} KB. Split the document into smaller parts."
        )

    raw = request.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadRequest(f"Body is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise BadRequest("Body must be a JSON object.")
    return payload


def send_json(request: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request.send_response(status)
    request.send_header("Content-Type", "application/json; charset=utf-8")
    request.send_header("Content-Length", str(len(body)))
    # No document content is stored or cached anywhere (P3).
    request.send_header("Cache-Control", "no-store")
    request.end_headers()
    request.wfile.write(body)


def json_endpoint(
    fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> type[BaseHTTPRequestHandler]:
    """Wrap a ``payload -> response`` function as a Vercel handler class."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            try:
                payload = read_json(self)
                send_json(self, 200, fn(payload))
            except BadRequest as exc:
                send_json(self, 400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - last-resort guard
                # Never leak a stack trace or any fragment of the submitted
                # document into a response body.
                send_json(
                    self,
                    500,
                    {"error": f"Inference failed: {type(exc).__name__}"},
                )

        def do_GET(self) -> None:  # noqa: N802
            send_json(self, 405, {"error": "Use POST."})

        def log_message(self, *_args: Any) -> None:
            """Silence the default access log — it would echo request paths."""

    return Handler


def require_text(payload: dict[str, Any], key: str = "text", max_chars: int = 200_000) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"'{key}' must be a non-empty string.")
    if len(value) > max_chars:
        raise BadRequest(f"'{key}' is longer than {max_chars} characters.")
    return value


def optional_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise BadRequest(f"'{key}' must be an array of strings.")
    return value
