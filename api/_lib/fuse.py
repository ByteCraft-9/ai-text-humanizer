"""Calibrated ensemble fusion (PRD 8.3, 8.7).

Four signals become one probability by logistic regression, then that
probability is passed through an isotonic calibrator so a reported 82%
corresponds to an observed 82% positive rate within ±5 points (A6).
Uncalibrated confidence is the most common failure of tools in this category,
so the calibrator is a first-class artefact, not an afterthought.

Degraded mode matters as much as the happy path. When the trained weights are
absent — a fresh deploy, a failed model download — the fuser renormalises over
whichever signals *are* available and marks the result degraded, so the UI can
say the score is partial rather than quietly presenting a feature-only guess
as a full ensemble reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES, standardize
from .runtime import load_json

SIGNAL_NAMES = ("classifier", "features", "binoculars_ratio", "burstiness")

# Fusion weights over [classifier, features, binoculars, burstiness].
# Fitted by training/05_calibrate_eval.ipynb and shipped as fusion.json;
# these defaults keep the ensemble sane before the first training run and are
# weighted toward the classifier, per E3's finding that no single signal
# should carry the decision alone.
DEFAULT_WEIGHTS = np.array([2.6, 1.1, 1.4, 0.7], dtype=np.float64)
DEFAULT_BIAS = -2.6


@dataclass
class FusionConfig:
    weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    bias: float = DEFAULT_BIAS
    # Isotonic calibrator as parallel (x, y) knots; empty means identity.
    calibration_x: list[float] = field(default_factory=list)
    calibration_y: list[float] = field(default_factory=list)
    version: str = "default"


_config_cache: dict[str, FusionConfig] = {}


def load_fusion(name: str) -> FusionConfig:
    """Load ``fusion_<name>.json``, falling back to the built-in defaults."""
    cached = _config_cache.get(name)
    if cached is not None:
        return cached

    data = load_json(f"fusion_{name}.json")
    if not data:
        config = FusionConfig()
    else:
        weights = np.asarray(data.get("weights", DEFAULT_WEIGHTS), dtype=np.float64)
        if weights.shape != (len(SIGNAL_NAMES),):
            weights = DEFAULT_WEIGHTS.copy()
        config = FusionConfig(
            weights=weights,
            bias=float(data.get("bias", DEFAULT_BIAS)),
            calibration_x=[float(v) for v in data.get("calibration_x", [])],
            calibration_y=[float(v) for v in data.get("calibration_y", [])],
            version=str(data.get("version", "unknown")),
        )

    _config_cache[name] = config
    return config


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def apply_calibration(probability: float, config: FusionConfig) -> float:
    """Piecewise-linear interpolation over the isotonic knots."""
    if len(config.calibration_x) < 2:
        return probability
    return float(
        np.interp(probability, config.calibration_x, config.calibration_y)
    )


@dataclass
class FusedScore:
    probability: float
    signals: dict[str, float]
    degraded: bool
    missing: list[str]


def fuse(
    signals: dict[str, float | None],
    name: str = "a",
) -> FusedScore:
    """Combine available signals into one calibrated probability.

    ``None`` marks a signal whose model was unavailable. Its weight is *not*
    redistributed onto the survivors: multiplying one remaining signal by the
    whole ensemble's weight drives it straight to 0 or 1, which is how a
    degraded deployment ends up reporting a confident-looking number it has no
    basis for. Instead both the gain and the bias shrink to the fraction of
    weight that is actually backed by evidence, so a partial reading stays
    near the signals that produced it.
    """
    config = load_fusion(name)

    present: list[int] = []
    missing: list[str] = []
    for i, key in enumerate(SIGNAL_NAMES):
        value = signals.get(key)
        if value is None or not math.isfinite(value):
            missing.append(key)
        else:
            present.append(i)

    reported = {key: float(signals.get(key) or 0.5) for key in SIGNAL_NAMES}

    if not present:
        # Nothing to go on. Return the uninformative prior rather than a
        # number that looks like a measurement.
        return FusedScore(0.5, reported, True, missing)

    logits = {i: _logit(reported[SIGNAL_NAMES[i]]) for i in present}
    present_weight = float(np.sum(config.weights[present]))
    total_weight = float(np.sum(config.weights))
    if present_weight <= 1e-9 or total_weight <= 1e-9:
        return FusedScore(0.5, reported, True, missing)

    mean_logit = sum(config.weights[i] * logits[i] for i in present) / present_weight

    # Gain is the weight actually backed by evidence, and the bias is scaled
    # by the same fraction so the decision boundary stays put. With all four
    # signals this reduces exactly to the fitted logistic; with fewer, the
    # output stays near the surviving signal instead of being amplified into
    # saturation by weight that has nothing behind it.
    evidence = present_weight / total_weight
    z = config.bias * evidence + present_weight * mean_logit

    probability = apply_calibration(_sigmoid(z), config)
    return FusedScore(
        probability=float(min(1.0, max(0.0, probability))),
        signals=reported,
        degraded=bool(missing),
        missing=missing,
    )


# ---------------------------------------------------------------------------
# Standalone feature scorer — the degraded-mode backstop
# ---------------------------------------------------------------------------

# Direction and magnitude of each feature's contribution to P(AI), on the
# standardised scale. Positive means "higher value looks more machine-written".
# Replaced wholesale by the coefficients fitted in 05_calibrate_eval.ipynb;
# the signs below follow the published findings (E4) and are what degraded
# mode runs on before that fit exists.
_FEATURE_DIRECTION = {
    # Generated prose is smooth and even: high readability grade, low variance.
    "stdev_sentence_length": -0.55,
    "mean_sentence_length": 0.10,
    "flesch_reading_ease": -0.20,
    "gunning_fog": 0.12,
    # Vocabulary: narrower range, fewer one-off words, less rare vocabulary.
    "hapax_ratio": -0.35,
    "root_type_token_ratio": -0.25,
    "rare_word_ratio": -0.30,
    "type_token_ratio": -0.15,
    # Repetition and connective density are the strongest surface tells.
    "discourse_marker_density": 0.60,
    "transition_word_rate": 0.40,
    "opener_diversity": -0.30,
    "parallel_structure_rate": 0.35,
    "bigram_repeat_rate": 0.15,
    # Machine text is unusually cohesive between adjacent sentences.
    "adjacent_similarity_mean": 0.30,
    "adjacent_similarity_stdev": -0.20,
    "topic_drift": -0.15,
    "pronoun_density": -0.20,
    "comma_rate": 0.10,
    "clause_depth_proxy": 0.10,
}

_DIRECTION_VECTOR = np.array(
    [_FEATURE_DIRECTION.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32
)
# Chosen so a document one standard deviation "machine-like" on every weighted
# feature lands near 0.8 rather than saturating at 1.0.
_FEATURE_SCALE = 0.62
_FEATURE_BIAS = -0.15


def score_features(raw_features: np.ndarray) -> float:
    """P(AI) from the handcrafted features alone.

    Used as the ``features`` ensemble signal, and as the only signal when no
    model weights are present at all.
    """
    standardized = standardize(raw_features)
    z = float(np.dot(standardized, _DIRECTION_VECTOR)) * _FEATURE_SCALE + _FEATURE_BIAS
    return _sigmoid(z)
