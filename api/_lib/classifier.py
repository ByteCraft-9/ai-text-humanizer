"""DeBERTa-v3-base + feature-attention classifier inference.

The exported graph takes three inputs — ``input_ids``, ``attention_mask`` and
the standardised 30-dimensional ``features`` vector — and returns a single
logit. The feature branch is what buys cross-domain robustness (E4: +7.2
points over Fast-DetectGPT on M4), so it is not optional at inference: a graph
exported without it is loaded as a plain text classifier and the fuser
compensates by leaning on the standalone feature score instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import extract_features, standardize
from .runtime import DETECTOR_A, DETECTOR_B, ModelSpec, get_session, get_tokenizer

MAX_LENGTH = 768


@dataclass
class ClassifierResult:
    probability: float
    sentence_probabilities: list[float]
    available: bool
    model_version: str


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _encode(tokenizer, text: str) -> tuple[np.ndarray, np.ndarray]:
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    encoding = tokenizer.encode(text)
    ids = np.asarray([encoding.ids], dtype=np.int64)
    mask = np.asarray([encoding.attention_mask], dtype=np.int64)
    return ids, mask


def _encode_batch(tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding()
    encodings = tokenizer.encode_batch(texts)
    ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
    mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
    tokenizer.no_padding()
    return ids, mask


def _run(spec: ModelSpec, ids: np.ndarray, mask: np.ndarray, feats: np.ndarray | None):
    session = get_session(spec)
    if session is None:
        return None

    expected = {inp.name for inp in session.get_inputs()}
    feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in expected:
        feeds["token_type_ids"] = np.zeros_like(ids)
    if "features" in expected:
        if feats is None:
            return None
        feeds["features"] = feats.astype(np.float32)

    feeds = {k: v for k, v in feeds.items() if k in expected}
    if expected - set(feeds):
        # The graph wants something we cannot supply; treat it as unavailable
        # rather than guessing and returning a meaningless score.
        return None

    try:
        outputs = session.run(None, feeds)
    except Exception:
        return None

    logits = np.asarray(outputs[0], dtype=np.float32)
    # Accept either a single logit or a two-class head.
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, 1] - logits[:, 0]
    return logits.reshape(logits.shape[0], -1)[:, 0]


def classify(
    spec: ModelSpec,
    text: str,
    sentences: list[str],
    model_version: str,
) -> ClassifierResult:
    """Score a chunk and each of its sentences with one classifier."""
    tokenizer = get_tokenizer(spec)
    if tokenizer is None or get_session(spec) is None:
        return ClassifierResult(0.5, [0.5] * len(sentences), False, "unavailable")

    doc_features = standardize(extract_features(text, sentences))[None, :]
    ids, mask = _encode(tokenizer, text)
    doc_logit = _run(spec, ids, mask, doc_features)
    if doc_logit is None:
        return ClassifierResult(0.5, [0.5] * len(sentences), False, "unavailable")

    probability = _sigmoid(float(doc_logit[0]))

    sentence_probabilities: list[float] = []
    if sentences:
        # Sentence features are noisy on their own — a single sentence is too
        # short for readability statistics to mean much — so each sentence is
        # scored with the document's feature context. That keeps the branch
        # meaningful while still letting the text encoder discriminate.
        batch_features = np.repeat(doc_features, len(sentences), axis=0)
        s_ids, s_mask = _encode_batch(tokenizer, sentences)
        logits = _run(spec, s_ids, s_mask, batch_features)
        if logits is not None:
            sentence_probabilities = [_sigmoid(float(v)) for v in logits]

    if not sentence_probabilities:
        sentence_probabilities = [probability] * len(sentences)

    return ClassifierResult(probability, sentence_probabilities, True, model_version)


def classify_strict(text: str, sentences: list[str]) -> ClassifierResult:
    """Model A — the honest score. Never the humanizer's optimisation target."""
    return classify(DETECTOR_A, text, sentences, "a-1.0.0")


def classify_surrogate(text: str, sentences: list[str]) -> ClassifierResult:
    """Model B — the third-party surrogate the humanizer optimises against."""
    return classify(DETECTOR_B, text, sentences, "b-1.0.0")


def classify_batch(spec: ModelSpec, texts: list[str]) -> list[float] | None:
    """Score many short texts at once. Used to rank humanizer candidates."""
    tokenizer = get_tokenizer(spec)
    if tokenizer is None or get_session(spec) is None or not texts:
        return None

    features = np.stack([standardize(extract_features(t, [t])) for t in texts])
    ids, mask = _encode_batch(tokenizer, texts)
    logits = _run(spec, ids, mask, features)
    if logits is None:
        return None
    return [_sigmoid(float(v)) for v in logits]
