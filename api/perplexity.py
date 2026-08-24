"""POST /api/py/perplexity — the zero-shot statistical signal (PRD 8.5).

Hosts the two small causal LMs (distilgpt2 as observer, gpt2 as performer) in
their own function so their weights sit in a separate 500 MB bundle from the
detector's. Returns the raw statistics rather than a probability: the caller
fuses and calibrates, and exposing the raw numbers keeps the signal
inspectable when a score looks wrong.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.http import json_endpoint, optional_str_list, require_text  # noqa: E402
from _lib.segment import split_sentences  # noqa: E402
from _lib.stats import compute_perplexity  # noqa: E402


def perplexity(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()

    text = require_text(payload)
    sentences = optional_str_list(payload, "sentences") or split_sentences(text)

    result = compute_perplexity(text, sentences)

    return {
        "log_perplexity": result.log_perplexity,
        "cross_perplexity": result.cross_perplexity,
        "binoculars": result.binoculars,
        "burstiness": result.burstiness,
        "sentence_perplexity": result.sentence_perplexity,
        "available": result.available,
        "meta": {
            "tokens": result.token_count,
            "ms": int((time.perf_counter() - started) * 1000),
        },
    }


handler = json_endpoint(perplexity)
