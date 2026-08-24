"""Zero-shot statistical signals: perplexity, cross-perplexity, burstiness.

Implements the Binoculars ratio (E5) with GPT-2-scale models. The original
method uses two 7B models, which will not fit a free function; PAN 2025 showed
GPT-2-scale observers still reach 0.978 ROC-AUC once fused with stylometry,
which is exactly the arrangement here.

    Binoculars(x) = log-perplexity(x | performer)
                    ─────────────────────────────
                    cross-perplexity(observer ‖ performer)

The numerator asks "how surprised is the model by this text?". The denominator
normalises by how surprised *models in general* are by text like this, which
is what stops the score collapsing on unusual-but-human writing — the failure
mode that makes raw perplexity detectors flag non-native speakers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .runtime import OBSERVER, PERFORMER, ModelSpec, get_session, get_tokenizer

# Binoculars thresholds. The published low-FPR operating point is ~0.9;
# below it, text is machine-like. Re-fitted by 05_calibrate_eval.ipynb.
BINOCULARS_MACHINE = 0.85
BINOCULARS_HUMAN = 1.05

MAX_TOKENS = 512


@dataclass
class PerplexityResult:
    log_perplexity: float
    cross_perplexity: float
    binoculars: float
    burstiness: float
    sentence_perplexity: list[float]
    token_count: int
    available: bool


def _softmax_log(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _run_causal_lm(spec: ModelSpec, input_ids: np.ndarray) -> np.ndarray | None:
    """Return logits ``[1, seq, vocab]`` for a causal LM, or ``None``."""
    session = get_session(spec)
    if session is None:
        return None

    seq_len = input_ids.shape[1]
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            feeds["input_ids"] = input_ids.astype(np.int64)
        elif inp.name == "attention_mask":
            feeds["attention_mask"] = np.ones((1, seq_len), dtype=np.int64)
        elif inp.name == "position_ids":
            feeds["position_ids"] = np.arange(seq_len, dtype=np.int64)[None, :]
        elif inp.name.startswith("past_key_values"):
            # Exported with KV cache inputs; feed empty tensors for a fresh
            # forward pass. Shape is [batch, heads, 0, head_dim].
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            shape[2] = 0
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)

    try:
        outputs = session.run(None, feeds)
    except Exception:
        return None
    return np.asarray(outputs[0], dtype=np.float32)


def _token_log_probs(logits: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
    """Log P(token_t | tokens_<t) for every position after the first."""
    log_probs = _softmax_log(logits[0, :-1, :])
    targets = input_ids[0, 1:]
    return log_probs[np.arange(len(targets)), targets]


def _cross_entropy_between(
    observer_logits: np.ndarray, performer_logits: np.ndarray
) -> float:
    """H(observer ‖ performer): the Binoculars denominator.

    Expectation is taken under the *observer's* distribution of the
    performer's log-probabilities, averaged over positions.
    """
    observer_log = _softmax_log(observer_logits[0, :-1, :])
    performer_log = _softmax_log(performer_logits[0, :-1, :])
    observer_probs = np.exp(observer_log)
    per_position = -np.sum(observer_probs * performer_log, axis=-1)
    return float(np.mean(per_position))


def compute_perplexity(text: str, sentences: list[str] | None = None) -> PerplexityResult:
    tokenizer = get_tokenizer(PERFORMER)
    if tokenizer is None:
        return _unavailable(sentences)

    encoding = tokenizer.encode(text)
    ids = encoding.ids[:MAX_TOKENS]
    if len(ids) < 8:
        return _unavailable(sentences)

    input_ids = np.asarray([ids], dtype=np.int64)

    performer_logits = _run_causal_lm(PERFORMER, input_ids)
    observer_logits = _run_causal_lm(OBSERVER, input_ids)
    if performer_logits is None or observer_logits is None:
        return _unavailable(sentences)

    token_log_probs = _token_log_probs(performer_logits, input_ids)
    log_perplexity = float(-np.mean(token_log_probs))
    cross_perplexity = _cross_entropy_between(observer_logits, performer_logits)

    binoculars = log_perplexity / cross_perplexity if cross_perplexity > 1e-6 else 1.0

    # Burstiness: how unevenly surprise is distributed. Human writing spikes;
    # generated text stays flat. Normalised by the mean so it does not simply
    # restate perplexity.
    surprise = -token_log_probs
    burstiness = float(np.std(surprise) / (np.mean(surprise) + 1e-6))

    sentence_ppl = _per_sentence_perplexity(text, ids, surprise, tokenizer, sentences)

    return PerplexityResult(
        log_perplexity=log_perplexity,
        cross_perplexity=cross_perplexity,
        binoculars=binoculars,
        burstiness=burstiness,
        sentence_perplexity=sentence_ppl,
        token_count=len(ids),
        available=True,
    )


def _per_sentence_perplexity(
    text: str,
    ids: list[int],
    surprise: np.ndarray,
    tokenizer,
    sentences: list[str] | None,
) -> list[float]:
    """Mean surprise per sentence, for the heatmap.

    Sentences are located by re-encoding each one and walking the token
    stream, which avoids depending on offset mappings that differ between
    tokenizer versions.
    """
    if not sentences:
        return []

    result: list[float] = []
    cursor = 0
    for sentence in sentences:
        length = max(1, len(tokenizer.encode(sentence).ids))
        # surprise[i] scores ids[i+1], so shift the window by one.
        start = max(0, cursor - 1)
        end = min(len(surprise), start + length)
        window = surprise[start:end]
        result.append(float(np.mean(window)) if len(window) else 0.0)
        cursor += length
        if cursor >= len(ids):
            cursor = len(ids) - 1

    return result


def _unavailable(sentences: list[str] | None) -> PerplexityResult:
    return PerplexityResult(
        log_perplexity=0.0,
        cross_perplexity=0.0,
        binoculars=1.0,
        burstiness=0.0,
        sentence_perplexity=[0.0] * len(sentences or []),
        token_count=0,
        available=False,
    )


# ---------------------------------------------------------------------------
# Mapping raw statistics to probabilities
# ---------------------------------------------------------------------------


def binoculars_to_probability(binoculars: float) -> float:
    """Map the Binoculars ratio to P(AI).

    A logistic centred between the published machine and human operating
    points. Lower ratio means more machine-like, so the slope is negative.
    """
    midpoint = (BINOCULARS_MACHINE + BINOCULARS_HUMAN) / 2
    steepness = 24.0
    return float(1.0 / (1.0 + math.exp(steepness * (binoculars - midpoint))))


def burstiness_to_probability(burstiness: float) -> float:
    """Map normalised surprise variance to P(AI).

    Human writing typically lands around 0.55–0.85 on this measure; generated
    text sits lower because its surprise is flat. Low burstiness therefore
    raises P(AI).
    """
    midpoint = 0.55
    steepness = 9.0
    return float(1.0 / (1.0 + math.exp(steepness * (burstiness - midpoint))))
