"""POST /api/py/score — rank humanizer candidates locally.

This is the step that makes the loop affordable (PRD 10.1 step 4): every
paraphrase candidate is scored here, on our own hardware, for zero API tokens.
A pass therefore costs roughly 3,000 provider tokens instead of the ~13,400 a
whole-document rewrite would burn.

Only the **surrogate** panel runs here. Model A is deliberately not reachable
from this endpoint — H5 requires that the strict score is never the thing the
rewrite optimises against, and the cleanest way to guarantee that is to make
it unavailable at the point of optimisation.

Request:
    {"originals": ["..."], "candidates": [["...", "..."]]}
Response:
    {"scores": [[0.2, 0.6]], "similarity": [[0.93, 0.88]],
     "similarity_mode": "semantic" | "lexical"}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.classifier import classify_batch  # noqa: E402
from _lib.embed import similarity_matrix  # noqa: E402
from _lib.features import extract_features  # noqa: E402
from _lib.fuse import fuse, score_features  # noqa: E402
from _lib.http import BadRequest, json_endpoint  # noqa: E402
from _lib.runtime import DETECTOR_B  # noqa: E402
from _lib.stats import (  # noqa: E402
    binoculars_to_probability,
    burstiness_to_probability,
    compute_perplexity,
)

# One call may carry SENTENCES_PER_CALL x CANDIDATES_PER_SENTENCE candidates.
# The ceiling is generous but bounded so a malformed request cannot pin the
# single vCPU for the full 300 s.
MAX_CANDIDATES = 128
MAX_CANDIDATE_CHARS = 2_000


def _parse(payload: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    originals = payload.get("originals")
    candidates = payload.get("candidates")

    if not isinstance(originals, list) or not all(isinstance(o, str) for o in originals):
        raise BadRequest("'originals' must be an array of strings.")
    if not isinstance(candidates, list):
        raise BadRequest("'candidates' must be an array of arrays of strings.")
    if len(originals) != len(candidates):
        raise BadRequest("'originals' and 'candidates' must be the same length.")

    cleaned: list[list[str]] = []
    total = 0
    for group in candidates:
        if not isinstance(group, list) or not all(isinstance(c, str) for c in group):
            raise BadRequest("Each entry in 'candidates' must be an array of strings.")
        trimmed = [c[:MAX_CANDIDATE_CHARS] for c in group if c.strip()]
        total += len(trimmed)
        cleaned.append(trimmed)

    if total == 0:
        raise BadRequest("There are no candidates to score.")
    if total > MAX_CANDIDATES:
        raise BadRequest(f"{total} candidates exceeds the limit of {MAX_CANDIDATES}.")

    return originals, cleaned


def _score_one(text: str, classifier_probability: float | None) -> float:
    """Fuse the available signals for a single candidate sentence."""
    perplexity = compute_perplexity(text, [text])
    if perplexity.available:
        binoculars = binoculars_to_probability(perplexity.binoculars)
        burstiness = burstiness_to_probability(perplexity.burstiness)
    else:
        binoculars = None
        burstiness = None

    fused = fuse(
        {
            "classifier": classifier_probability,
            "features": score_features(extract_features(text, [text])),
            "binoculars_ratio": binoculars,
            "burstiness": burstiness,
        },
        name="b",
    )
    return fused.probability


def score(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    originals, candidates = _parse(payload)

    flat = [c for group in candidates for c in group]

    # One batched classifier pass over every candidate, then the cheap
    # statistical signals per candidate.
    classifier_scores = classify_batch(DETECTOR_B, flat)

    fused_flat: list[float] = []
    for i, text in enumerate(flat):
        probability = classifier_scores[i] if classifier_scores is not None else None
        fused_flat.append(_score_one(text, probability))

    scores: list[list[float]] = []
    cursor = 0
    for group in candidates:
        scores.append(fused_flat[cursor : cursor + len(group)])
        cursor += len(group)

    similarity = similarity_matrix(originals, candidates)

    return {
        "scores": scores,
        "similarity": similarity.values,
        # The caller applies a different floor for the lexical proxy, and the
        # UI says which check actually ran. Silently enforcing 0.85 against a
        # measure that is not cosine similarity would reject every rewrite.
        "similarity_mode": "semantic" if similarity.semantic else "lexical",
        "degraded": classifier_scores is None or not similarity.semantic,
        "ms": int((time.perf_counter() - started) * 1000),
    }


handler = json_endpoint(score)
