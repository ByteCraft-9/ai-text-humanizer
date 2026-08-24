"""POST /api/py/detect — score one chunk with both models.

Returns Model A's strict score, Model B's surrogate score, the four ensemble
signals and a per-sentence heatmap. Detection is entirely local to this
function: the submitted text never reaches a third party (P1), which is the
product's main privacy advantage over every free detector that uploads your
document to score it.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.classifier import classify_strict, classify_surrogate  # noqa: E402
from _lib.features import extract_features  # noqa: E402
from _lib.fuse import fuse, score_features  # noqa: E402
from _lib.http import json_endpoint, optional_str_list, require_text  # noqa: E402
from _lib.remote import fetch_perplexity  # noqa: E402
from _lib.segment import split_sentences  # noqa: E402
from _lib.stats import (  # noqa: E402
    binoculars_to_probability,
    burstiness_to_probability,
)


def _aggregate_feature_score(
    text: str, sentences: list[str], document_vector
) -> float:
    """The ``features`` signal for a whole chunk.

    Document-level features alone under-read mixed writing: a few paragraphs
    of obvious machine prose inside otherwise human text average out to
    nothing, because readability and sentence-length variance are computed
    over the whole span. So the chunk score also carries the sentence
    evidence, aggregated the same way PRD 16.3 aggregates chunks — a
    length-weighted mean blended toward the maximum, so one heavily flagged
    passage is not averaged away.
    """
    document = score_features(document_vector)

    long_enough = [s for s in sentences if len(s.split()) >= 6]
    if len(long_enough) < 2:
        return document

    scores = [score_features(extract_features(s, [s])) for s in long_enough]
    weights = [len(s.split()) for s in long_enough]
    total = sum(weights) or len(scores)

    weighted = sum(s * w for s, w in zip(scores, weights)) / total
    highest = max(scores)
    sentence_level = weighted * 0.7 + highest * 0.3

    # The document view sees structure a sentence cannot; the sentence view
    # sees passages the document average hides. Neither dominates.
    return document * 0.45 + sentence_level * 0.55


def _blend_sentence_scores(
    sentences: list[str],
    classifier_available: bool,
    classifier_scores: list[float],
    perplexity_scores: list[float],
) -> list[float]:
    """Per-sentence P(AI) for the heatmap.

    The classifier leads; per-sentence perplexity adds the local texture that
    makes the heatmap useful as evidence rather than a smear of the document
    average. When neither model is present the features are computed per
    sentence instead — noisy at this length, but a heatmap flat at 0.5 tells
    the user nothing at all, which is worse.
    """
    out: list[float] = []
    for i, sentence in enumerate(sentences):
        parts: list[tuple[float, float]] = []  # (weight, probability)

        if classifier_available and i < len(classifier_scores):
            parts.append((0.7, classifier_scores[i]))

        if i < len(perplexity_scores) and perplexity_scores[i] > 0:
            # Surprise around 3.0 nats is typical of generated text; above
            # ~5.0 the sentence is doing something a model would not predict.
            surprise = perplexity_scores[i]
            parts.append((0.3, 1.0 / (1.0 + math.exp(1.6 * (surprise - 4.0)))))

        if not parts:
            parts.append((1.0, score_features(extract_features(sentence, [sentence]))))

        total = sum(weight for weight, _ in parts)
        out.append(sum(weight * value for weight, value in parts) / total)

    return out


def detect(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()

    text = require_text(payload)
    sentences = optional_str_list(payload, "sentences") or split_sentences(text)
    include_strict = payload.get("include_strict", True) is not False

    # --- Signals shared by both models -----------------------------------
    feature_vector = extract_features(text, sentences)
    feature_score = _aggregate_feature_score(text, sentences, feature_vector)

    # The Binoculars LMs live in perplexity.py's bundle, not this one — two
    # DeBERTas plus the runtime already fills 448 MB of the 500 MB limit.
    # See api/_lib/remote.py. A miss here degrades the score, never fails it.
    perplexity = fetch_perplexity(text, sentences)
    if perplexity:
        binoculars_score = binoculars_to_probability(perplexity["binoculars"])
        burstiness_score = burstiness_to_probability(perplexity["burstiness"])
        sentence_perplexity = perplexity.get("sentence_perplexity") or []
    else:
        binoculars_score = None
        burstiness_score = None
        sentence_perplexity = []

    # --- Model B: the surrogate the humanizer optimises against ----------
    surrogate = classify_surrogate(text, sentences)
    surrogate_fused = fuse(
        {
            "classifier": surrogate.probability if surrogate.available else None,
            "features": feature_score,
            "binoculars_ratio": binoculars_score,
            "burstiness": burstiness_score,
        },
        name="b",
    )

    # --- Model A: the honest score, read but never optimised against -----
    if include_strict:
        strict = classify_strict(text, sentences)
        strict_fused = fuse(
            {
                "classifier": strict.probability if strict.available else None,
                "features": feature_score,
                "binoculars_ratio": binoculars_score,
                "burstiness": burstiness_score,
            },
            name="a",
        )
        strict_score = strict_fused.probability
        degraded = strict_fused.degraded or surrogate_fused.degraded
        model_version = strict.model_version if strict.available else "degraded"
    else:
        strict_score = surrogate_fused.probability
        degraded = surrogate_fused.degraded
        model_version = surrogate.model_version if surrogate.available else "degraded"

    # The heatmap comes from the surrogate: it is what the humanizer targets,
    # so the sentences the user sees highlighted are the ones that will be
    # rewritten.
    sentence_scores = _blend_sentence_scores(
        sentences,
        surrogate.available,
        surrogate.sentence_probabilities,
        sentence_perplexity,
    )

    return {
        "strict_score": strict_score,
        "surrogate_score": surrogate_fused.probability,
        "signals": surrogate_fused.signals,
        "sentence_scores": sentence_scores,
        "model_version": model_version,
        "degraded": degraded,
        "missing_signals": surrogate_fused.missing,
        "ms": int((time.perf_counter() - started) * 1000),
    }


handler = json_endpoint(detect)
