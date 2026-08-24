"""Thirty handcrafted linguistic features (PRD 8.4).

These feed the feature-attention branch of the classifier and, when the
trained weights are unavailable, stand on their own as a degraded-mode
detector. E4 found readability and vocabulary features contributed most, so
those two families lead the ordering.

This module is the *canonical* definition. `training/lib/features.py` imports
from here so the features computed during training are byte-for-byte the
features computed at inference. Any drift between the two silently destroys
accuracy, which is exactly the failure this arrangement prevents.

Zero dependencies beyond the standard library and NumPy: it has to fit inside
a 500 MB function bundle alongside ONNX Runtime.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from .wordlists import COMMON_WORDS, DISCOURSE_MARKERS, FUNCTION_WORDS, SUBORDINATORS

# Order is part of the contract: the model's input vector is indexed by it.
FEATURE_NAMES: tuple[str, ...] = (
    # Readability (7)
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog",
    "coleman_liau",
    "automated_readability",
    "dale_chall",
    # Vocabulary (7)
    "type_token_ratio",
    "root_type_token_ratio",
    "hapax_ratio",
    "mean_word_length",
    "long_word_ratio",
    "rare_word_ratio",
    "lexical_density",
    # Syntax (6)
    "mean_sentence_length",
    "stdev_sentence_length",
    "clause_depth_proxy",
    "punctuation_diversity",
    "comma_rate",
    "subordinator_rate",
    # Repetition (5)
    "bigram_repeat_rate",
    "trigram_repeat_rate",
    "opener_diversity",
    "discourse_marker_density",
    "parallel_structure_rate",
    # Coherence (5)
    "adjacent_similarity_mean",
    "adjacent_similarity_stdev",
    "pronoun_density",
    "topic_drift",
    "transition_word_rate",
)

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 30, f"expected 30 features, defined {N_FEATURES}"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_PRONOUNS = frozenset(
    """i me my mine myself you your yours yourself he him his himself she her hers
    herself it its itself we us our ours ourselves they them their theirs
    themselves this that these those""".split()
)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def count_syllables(word: str) -> int:
    """Approximate English syllable count.

    A heuristic, not a dictionary lookup — a pronunciation dictionary would
    cost tens of megabytes for a feature that only needs to be *consistent*
    between training and inference, not phonetically correct.
    """
    word = word.lower().strip("'-")
    if not word:
        return 0

    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)

    # Silent terminal 'e', but not in "the" or "-le" endings like "table".
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    if word.endswith(("es", "ed")) and count > 1 and not word.endswith(("ies", "ted", "ded")):
        count -= 1

    return max(1, count)


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Readability (7)
# ---------------------------------------------------------------------------


def _readability(words: list[str], sentences: list[str], syllables: list[int]) -> list[float]:
    n_words = len(words)
    n_sentences = max(1, len(sentences))
    n_syllables = sum(syllables)
    n_chars = sum(len(w) for w in words)

    words_per_sentence = _safe_div(n_words, n_sentences)
    syllables_per_word = _safe_div(n_syllables, n_words)

    complex_words = sum(1 for s in syllables if s >= 3)
    complex_ratio = _safe_div(complex_words, n_words)

    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    fog = 0.4 * (words_per_sentence + 100 * complex_ratio)
    smog = 1.0430 * math.sqrt(complex_words * _safe_div(30, n_sentences)) + 3.1291

    letters_per_100 = _safe_div(n_chars, n_words) * 100
    sentences_per_100 = _safe_div(n_sentences, n_words) * 100
    coleman_liau = 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8

    ari = 4.71 * _safe_div(n_chars, n_words) + 0.5 * words_per_sentence - 21.43

    # Dale-Chall, using the bundled high-frequency list as the "easy word"
    # set. The published 3,000-word list would add weight for no gain: the
    # feature only needs to be consistent, and it is computed identically
    # during training.
    hard = sum(1 for w in words if w.lower() not in COMMON_WORDS)
    hard_ratio = _safe_div(hard, n_words)
    dale_chall = 0.1579 * (hard_ratio * 100) + 0.0496 * words_per_sentence
    if hard_ratio > 0.05:
        dale_chall += 3.6365

    return [flesch, fk_grade, fog, smog, coleman_liau, ari, dale_chall]


# ---------------------------------------------------------------------------
# Vocabulary (7)
# ---------------------------------------------------------------------------


def _vocabulary(words: list[str]) -> list[float]:
    n_words = len(words)
    lowered = [w.lower() for w in words]
    counts = Counter(lowered)
    n_types = len(counts)

    ttr = _safe_div(n_types, n_words)
    # Guiraud's root TTR: far less length-dependent than plain TTR, which
    # matters because documents here range from 100 to 5,000 words.
    root_ttr = _safe_div(n_types, math.sqrt(n_words)) if n_words else 0.0
    hapax = _safe_div(sum(1 for c in counts.values() if c == 1), n_types)
    mean_word_length = _safe_div(sum(len(w) for w in words), n_words)
    long_word_ratio = _safe_div(sum(1 for w in words if len(w) >= 7), n_words)
    rare_word_ratio = _safe_div(sum(1 for w in lowered if w not in COMMON_WORDS), n_words)
    lexical_density = _safe_div(
        sum(1 for w in lowered if w not in FUNCTION_WORDS), n_words
    )

    return [
        ttr,
        root_ttr,
        hapax,
        mean_word_length,
        long_word_ratio,
        rare_word_ratio,
        lexical_density,
    ]


# ---------------------------------------------------------------------------
# Syntax (6)
# ---------------------------------------------------------------------------


def _syntax(text: str, words: list[str], sentences: list[str]) -> list[float]:
    lengths = [len(tokenize_words(s)) for s in sentences] or [0]
    mean_length = sum(lengths) / len(lengths)
    stdev_length = _stdev(lengths)

    # Clause depth proxy: commas plus subordinators per sentence. A real
    # parse tree would need spaCy (~500 MB with its model), which the bundle
    # budget cannot carry.
    subordinator_count = sum(1 for w in words if w.lower() in SUBORDINATORS)
    clause_depth = _safe_div(text.count(",") + subordinator_count, len(sentences) or 1)

    punctuation = [c for c in text if c in ".,;:!?—-()\"'[]"]
    punctuation_diversity = _safe_div(len(set(punctuation)), 14)

    comma_rate = _safe_div(text.count(","), len(words))
    subordinator_rate = _safe_div(subordinator_count, len(words))

    return [
        mean_length,
        stdev_length,
        clause_depth,
        punctuation_diversity,
        comma_rate,
        subordinator_rate,
    ]


# ---------------------------------------------------------------------------
# Repetition (5)
# ---------------------------------------------------------------------------


def _ngram_repeat_rate(tokens: list[str], n: int) -> float:
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return _safe_div(repeated, len(grams))


def _repetition(words: list[str], sentences: list[str]) -> list[float]:
    lowered = [w.lower() for w in words]

    bigram = _ngram_repeat_rate(lowered, 2)
    trigram = _ngram_repeat_rate(lowered, 3)

    # Sentence openers: LLM prose reuses the same handful of first words.
    openers = [tokenize_words(s)[:1] for s in sentences]
    first_words = [o[0].lower() for o in openers if o]
    opener_diversity = _safe_div(len(set(first_words)), len(first_words), 1.0)

    marker_count = 0
    lowered_text = " ".join(lowered)
    for marker in DISCOURSE_MARKERS:
        marker_count += lowered_text.count(marker)
    marker_density = _safe_div(marker_count, len(sentences) or 1)

    # Parallel structure: consecutive sentences opening with the same
    # part-of-speech-ish token, which LLM listing style produces heavily.
    parallel = 0
    for i in range(1, len(first_words)):
        if first_words[i] == first_words[i - 1]:
            parallel += 1
    parallel_rate = _safe_div(parallel, max(1, len(first_words) - 1))

    return [bigram, trigram, opener_diversity, marker_density, parallel_rate]


# ---------------------------------------------------------------------------
# Coherence (5)
# ---------------------------------------------------------------------------


def _bag_of_words_vector(sentence: str) -> Counter:
    return Counter(w.lower() for w in tokenize_words(sentence) if w.lower() not in FUNCTION_WORDS)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return _safe_div(dot, norm_a * norm_b)


def _coherence(words: list[str], sentences: list[str]) -> list[float]:
    # Content-word cosine stands in for sentence embeddings. A MiniLM encoder
    # is available in api/score.py where meaning preservation genuinely needs
    # semantics; here the cheap proxy is enough and keeps detect.py's bundle
    # inside budget.
    vectors = [_bag_of_words_vector(s) for s in sentences]
    similarities = [_cosine(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))]

    similarity_mean = sum(similarities) / len(similarities) if similarities else 0.0
    similarity_stdev = _stdev(similarities)

    pronoun_density = _safe_div(
        sum(1 for w in words if w.lower() in _PRONOUNS), len(words)
    )

    # Topic drift: how far the closing third has moved from the opening third.
    third = max(1, len(sentences) // 3)
    opening = Counter()
    closing = Counter()
    for v in vectors[:third]:
        opening.update(v)
    for v in vectors[-third:]:
        closing.update(v)
    topic_drift = 1.0 - _cosine(opening, closing)

    transition_count = sum(
        1 for w in (w.lower() for w in words) if w in DISCOURSE_MARKERS
    )
    transition_rate = _safe_div(transition_count, len(words))

    return [
        similarity_mean,
        similarity_stdev,
        pronoun_density,
        topic_drift,
        transition_rate,
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def extract_features(text: str, sentences: Sequence[str] | None = None) -> np.ndarray:
    """Return the 30-dimensional feature vector for `text`.

    Always returns a finite float32 vector of length 30, even for degenerate
    input — a NaN here would poison the fused score downstream.
    """
    from .segment import split_sentences  # local import avoids a cycle

    sentence_list = list(sentences) if sentences is not None else split_sentences(text)
    sentence_list = [s for s in sentence_list if s.strip()]
    if not sentence_list:
        sentence_list = [text] if text.strip() else []

    words = tokenize_words(text)
    if not words:
        return np.zeros(N_FEATURES, dtype=np.float32)

    syllables = [count_syllables(w) for w in words]

    values: list[float] = []
    values += _readability(words, sentence_list, syllables)
    values += _vocabulary(words)
    values += _syntax(text, words, sentence_list)
    values += _repetition(words, sentence_list)
    values += _coherence(words, sentence_list)

    vector = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


# Empirical means and standard deviations over the training mix, used to
# standardise the vector before it reaches the model. Replaced by the real
# values written by training/05_calibrate_eval.ipynb; these defaults keep
# degraded mode sane before the first training run.
FEATURE_MEAN = np.array(
    [
        52.0, 11.0, 13.0, 12.0, 11.0, 12.0, 8.0,          # readability
        0.52, 6.5, 0.62, 4.9, 0.24, 0.36, 0.55,           # vocabulary
        20.0, 7.0, 2.2, 0.55, 0.055, 0.030,               # syntax
        0.020, 0.004, 0.80, 0.55, 0.06,                   # repetition
        0.14, 0.12, 0.055, 0.85, 0.010,                   # coherence
    ],
    dtype=np.float32,
)

FEATURE_STD = np.array(
    [
        16.0, 3.2, 3.6, 2.8, 2.6, 3.4, 1.6,
        0.10, 1.4, 0.10, 0.55, 0.07, 0.09, 0.07,
        6.5, 3.6, 0.9, 0.16, 0.020, 0.014,
        0.020, 0.007, 0.14, 0.35, 0.09,
        0.09, 0.07, 0.026, 0.12, 0.009,
    ],
    dtype=np.float32,
)


def standardize(vector: np.ndarray) -> np.ndarray:
    return np.nan_to_num((vector - FEATURE_MEAN) / FEATURE_STD, nan=0.0, posinf=0.0, neginf=0.0)
