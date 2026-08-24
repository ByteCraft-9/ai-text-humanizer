"""Sentence embeddings for the meaning-preservation gate (PRD 10.2).

Without this check the loop "humanizes" by quietly degrading content — the
dominant failure mode of existing tools, and the reason A8 sets a hard 0.85
cosine floor.

Uses an INT8 MiniLM encoder (~23 MB) with mean pooling. When the encoder is
absent the module falls back to a lexical measure and says so, because
rejecting a rewrite on a bad similarity estimate is better than accepting one
that changed the facts — but a caller that cannot tell the two apart would be
enforcing a threshold it does not understand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import tokenize_words
from .runtime import ENCODER, get_session, get_tokenizer
from .wordlists import FUNCTION_WORDS

MAX_LENGTH = 256


@dataclass
class SimilarityResult:
    """Per-original rows of per-candidate cosine similarities."""

    values: list[list[float]]
    #: True when a real encoder produced these; False for the lexical proxy.
    semantic: bool


def _mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = mask[:, :, None].astype(np.float32)
    summed = np.sum(hidden * expanded, axis=1)
    counts = np.clip(np.sum(expanded, axis=1), 1e-9, None)
    return summed / counts


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def embed(texts: list[str]) -> np.ndarray | None:
    """L2-normalised embeddings, or ``None`` when the encoder is unavailable."""
    if not texts:
        return None

    session = get_session(ENCODER)
    tokenizer = get_tokenizer(ENCODER)
    if session is None or tokenizer is None:
        return None

    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding()
    encodings = tokenizer.encode_batch(texts)
    tokenizer.no_padding()

    ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
    mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

    expected = {inp.name for inp in session.get_inputs()}
    feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in expected:
        feeds["token_type_ids"] = np.zeros_like(ids)
    feeds = {k: v for k, v in feeds.items() if k in expected}

    try:
        outputs = session.run(None, feeds)
    except Exception:
        return None

    hidden = np.asarray(outputs[0], dtype=np.float32)
    if hidden.ndim != 3:
        return None
    return _normalize(_mean_pool(hidden, mask))


def _content_tokens(text: str) -> set[str]:
    return {w.lower() for w in tokenize_words(text) if w.lower() not in FUNCTION_WORDS}


def _lexical_similarity(a: str, b: str) -> float:
    """Jaccard over content words, as a conservative stand-in.

    Systematically *lower* than semantic similarity for a good paraphrase —
    which is the safe direction. It rejects some valid rewrites rather than
    admitting one that changed the meaning.
    """
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def similarity_matrix(originals: list[str], candidates: list[list[str]]) -> SimilarityResult:
    """Cosine similarity of each candidate to its original sentence."""
    flat = [c for group in candidates for c in group]
    if not flat:
        return SimilarityResult([], True)

    vectors = embed(originals + flat)
    if vectors is None:
        values = [
            [_lexical_similarity(originals[i], c) for c in candidates[i]]
            for i in range(len(originals))
        ]
        return SimilarityResult(values, semantic=False)

    original_vectors = vectors[: len(originals)]
    candidate_vectors = vectors[len(originals) :]

    values: list[list[float]] = []
    cursor = 0
    for i, group in enumerate(candidates):
        row = []
        for _ in group:
            row.append(float(np.dot(original_vectors[i], candidate_vectors[cursor])))
            cursor += 1
        values.append(row)

    return SimilarityResult(values, semantic=True)
