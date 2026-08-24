"""Internal call from `detect.py` to `perplexity.py`.

Why this exists: PRD 9.1 budgets `detect.py` for one DeBERTa (265 MB), but
PRD 8.1 requires two — Model A and Model B. Two INT8 DeBERTas plus the runtime
is 448 MB, and adding the Binoculars LMs on top would be 558 MB, over the
500 MB per-function limit. Since each `api/*.py` file gets its own bundle,
the fix is to keep the LMs in `perplexity.py` and have `detect.py` ask for
the statistics over the loopback.

The call is best-effort. If the sibling function is cold, slow or absent the
Binoculars and burstiness signals are simply unavailable, the fuser
renormalises, and the response is marked degraded — the same path a missing
model file takes. Detection never fails because one signal did.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# Vercel exposes the deployment host to the function. Locally there is none,
# which is exactly when the in-process import below is the right answer.
_HOST = os.environ.get("VERCEL_URL") or os.environ.get("VERCEL_BRANCH_URL") or ""
_TIMEOUT_SECONDS = float(os.environ.get("PERPLEXITY_TIMEOUT", "20"))


def _endpoint() -> str | None:
    if not _HOST:
        return None
    host = _HOST if _HOST.startswith("http") else f"https://{_HOST}"
    return f"{host}/api/py/perplexity"


def fetch_perplexity(text: str, sentences: list[str]) -> dict | None:
    """Return the perplexity statistics, or ``None`` if unavailable.

    Tries the in-process import first: on a single machine — local dev, tests,
    or a deployment where the weights happen to be present in this bundle —
    that avoids a network hop entirely and is strictly faster.
    """
    try:
        from .stats import compute_perplexity

        result = compute_perplexity(text, sentences)
        if result.available:
            return {
                "log_perplexity": result.log_perplexity,
                "cross_perplexity": result.cross_perplexity,
                "binoculars": result.binoculars,
                "burstiness": result.burstiness,
                "sentence_perplexity": result.sentence_perplexity,
                "available": True,
            }
    except Exception:
        pass

    endpoint = _endpoint()
    if endpoint is None:
        return None

    payload = json.dumps({"text": text, "sentences": sentences}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None

    return data if data.get("available") else None
