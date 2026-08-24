"""Tests for the inference library.

Run with:  python -m pytest tests/test_python.py
(or plain `python tests/test_python.py` — it falls back to a tiny runner so a
pytest install is not required just to check the maths.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import numpy as np  # noqa: E402

from _lib.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    count_syllables,
    extract_features,
    standardize,
)
from _lib.fuse import fuse, score_features  # noqa: E402
from _lib.segment import split_sentences, split_sentences_with_offsets  # noqa: E402
from _lib.stats import binoculars_to_probability, burstiness_to_probability  # noqa: E402

AI_TEXT = (
    "Artificial intelligence has become an increasingly important topic in "
    "today's world. Furthermore, it plays a crucial role in a wide range of "
    "industries. Moreover, organizations are utilizing AI to facilitate the "
    "optimization of their processes. Additionally, it is important to note "
    "that the implementation of these systems requires careful consideration. "
    "Consequently, businesses must delve into the various factors that "
    "influence successful adoption."
)

HUMAN_TEXT = (
    "I spent most of Tuesday trying to get the printer to work. It jammed "
    "twice, then decided the cyan cartridge was empty - it wasn't. My "
    "neighbour Dave, who fixes these things for a living, laughed at me for "
    "about a minute straight before pointing out the paper guide was set to "
    "A5. Anyway. Two hours gone."
)

# Human academic prose. A4 is a release blocker: this must not score high.
ESL_ACADEMIC = (
    "In this paper we are investigate the effect of temperature on the "
    "reaction rate. The experiment was conducted three times for confirm the "
    "result. We find that increase of temperature is lead to faster reaction, "
    "which agree with the theory. However the third trial show some deviation "
    "because the equipment was not calibrate properly."
)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_offsets_index_back_into_the_source():
    text = "First sentence here. Second one follows! And a third?"
    for sentence, start, end in split_sentences_with_offsets(text):
        assert text[start:end] == sentence


def test_abbreviations_do_not_break_sentences():
    text = "Dr. Smith met Prof. J. Chen at 3.14 p.m. to discuss the U.S. results. They agreed."
    assert len(split_sentences(text)) == 2


def test_blank_line_is_a_hard_break():
    sentences = split_sentences("Methods\n\nWe ran the experiment twice.")
    assert len(sentences) == 2
    assert sentences[0] == "Methods"


def test_closing_quote_stays_with_its_sentence():
    sentences = split_sentences('He said "we are done." Then he left.')
    assert sentences[0] == 'He said "we are done."'


def test_empty_input():
    assert split_sentences("   \n\n  ") == []


def test_matches_the_typescript_splitter():
    """The heatmap breaks silently if the two splitters disagree.

    These cases are the same fixtures asserted in tests/chunk.test.ts. Any
    divergence puts sentence scores on the wrong spans in the UI.
    """
    cases = [
        ("First sentence here. Second one follows! And a third?", 3),
        ("Dr. Smith met Prof. J. Chen at 3.14 p.m. to discuss the U.S. results. They agreed.", 2),
        ("Methods\n\nWe ran the experiment twice.", 2),
        ('He said "we are done." Then he left.', 2),
        ("no punctuation here", 1),
    ]
    for text, expected in cases:
        assert len(split_sentences(text)) == expected, text


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_feature_vector_shape_and_finiteness():
    for text in (AI_TEXT, HUMAN_TEXT, ESL_ACADEMIC, "a", ""):
        vector = extract_features(text)
        assert vector.shape == (N_FEATURES,)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector)), text[:40]


def test_feature_names_are_unique_and_thirty():
    assert len(set(FEATURE_NAMES)) == 30


def test_standardize_is_finite_even_for_degenerate_input():
    assert np.all(np.isfinite(standardize(extract_features(""))))


def test_syllable_counting():
    for word, expected in [
        ("cat", 1),
        ("running", 2),
        ("beautiful", 3),
        ("the", 1),
        ("table", 2),
        ("", 0),
    ]:
        assert count_syllables(word) == expected, word


def test_discourse_markers_raise_the_feature_score():
    """The connective density feature is the strongest surface tell."""
    assert score_features(extract_features(AI_TEXT)) > score_features(
        extract_features(HUMAN_TEXT)
    )


def test_human_academic_writing_is_not_flagged():
    """A4 is a release blocker: non-native human prose must stay low."""
    assert score_features(extract_features(ESL_ACADEMIC)) < 0.5


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_fusion_with_all_signals_is_monotonic():
    low = fuse(
        {"classifier": 0.1, "features": 0.1, "binoculars_ratio": 0.1, "burstiness": 0.1}
    )
    high = fuse(
        {"classifier": 0.9, "features": 0.9, "binoculars_ratio": 0.9, "burstiness": 0.9}
    )
    assert high.probability > low.probability
    assert not low.degraded and not high.degraded


def test_missing_signals_are_reported_and_do_not_saturate():
    """A degraded reading must stay near its surviving signal.

    Redistributing the whole ensemble's weight onto one signal drives the
    output to 0 or 1, which looks like confidence the deployment has no basis
    for.
    """
    result = fuse(
        {"classifier": None, "features": 0.62, "binoculars_ratio": None, "burstiness": None}
    )
    assert result.degraded
    assert set(result.missing) == {"classifier", "binoculars_ratio", "burstiness"}
    assert 0.15 < result.probability < 0.85


def test_fusion_with_no_signals_returns_the_prior():
    result = fuse(
        {"classifier": None, "features": None, "binoculars_ratio": None, "burstiness": None}
    )
    assert result.probability == 0.5
    assert result.degraded


def test_fusion_output_is_bounded():
    for value in (0.0, 1.0, 0.5):
        result = fuse(
            {
                "classifier": value,
                "features": value,
                "binoculars_ratio": value,
                "burstiness": value,
            }
        )
        assert 0.0 <= result.probability <= 1.0


# ---------------------------------------------------------------------------
# Statistical signal mapping
# ---------------------------------------------------------------------------


def test_binoculars_direction():
    """Lower Binoculars ratio means more machine-like, so higher P(AI)."""
    assert binoculars_to_probability(0.6) > binoculars_to_probability(1.3)
    assert 0.0 <= binoculars_to_probability(0.95) <= 1.0


def test_burstiness_direction():
    """Flat surprise means machine-like, so lower burstiness raises P(AI)."""
    assert burstiness_to_probability(0.2) > burstiness_to_probability(0.9)


# ---------------------------------------------------------------------------
# End-to-end through the handler function
# ---------------------------------------------------------------------------


def test_detect_ranks_ai_above_human():
    from detect import detect  # noqa: PLC0415 - needs sys.path set above

    ai = detect({"text": AI_TEXT})
    human = detect({"text": HUMAN_TEXT})
    esl = detect({"text": ESL_ACADEMIC})

    assert ai["surrogate_score"] > human["surrogate_score"]
    assert ai["surrogate_score"] > esl["surrogate_score"]
    # A4: human writing by a non-native speaker must not be flagged.
    assert esl["surrogate_score"] < 0.5


def test_detect_returns_one_score_per_sentence():
    from detect import detect  # noqa: PLC0415

    result = detect({"text": AI_TEXT})
    assert len(result["sentence_scores"]) == len(split_sentences(AI_TEXT))
    assert all(0.0 <= s <= 1.0 for s in result["sentence_scores"])


def test_detect_heatmap_is_not_flat_in_degraded_mode():
    """A heatmap pinned at 0.5 is worse than no heatmap: it looks like data."""
    from detect import detect  # noqa: PLC0415

    scores = detect({"text": AI_TEXT + " " + HUMAN_TEXT})["sentence_scores"]
    assert max(scores) - min(scores) > 0.2


def test_mixed_document_scores_above_pure_human():
    """PRD 16.3: a flagged passage must not be averaged away."""
    from detect import detect  # noqa: PLC0415

    mixed = detect({"text": HUMAN_TEXT + " " + AI_TEXT})["surrogate_score"]
    human = detect({"text": HUMAN_TEXT})["surrogate_score"]
    assert mixed > human


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
