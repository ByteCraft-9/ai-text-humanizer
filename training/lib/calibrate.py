"""Fusion fitting, isotonic calibration and the acceptance gate (PRD 8.7, 14).

Two artefacts come out of here, both tiny and both shipped alongside the ONNX
weights:

  * ``fusion_a.json`` / ``fusion_b.json`` — logistic weights over the four
    signals plus the isotonic calibrator knots.
  * ``eval_report.json`` — the real numbers for every criterion in PRD 14.

A6 requires that a reported "82% AI" corresponds to an observed 82% positive
rate within 5 points. Uncalibrated confidence is the most common failure of
tools in this category, so the calibrator is fitted here rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIGNAL_NAMES = ("classifier", "features", "binoculars_ratio", "burstiness")


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fit_fusion(
    signals: np.ndarray, labels: np.ndarray, max_iterations: int = 500
) -> tuple[np.ndarray, float]:
    """Fit the logistic fuser over signal logits.

    Plain IRLS-free gradient descent — four parameters and a bias do not
    justify a scikit-learn dependency inside the notebook, and keeping it
    explicit means the weights written to JSON are obviously the weights the
    inference code applies.

    Weights are constrained non-negative: a signal that argues *against* AI
    when it fires would be a sign the signal is broken, not something to fit
    around.
    """
    if signals.shape[1] != len(SIGNAL_NAMES):
        raise ValueError(f"expected {len(SIGNAL_NAMES)} signals, got {signals.shape[1]}")

    x = _logit(signals)
    y = labels.astype(np.float64)

    weights = np.ones(len(SIGNAL_NAMES), dtype=np.float64)
    bias = 0.0
    learning_rate = 0.05
    n = len(y)

    for _ in range(max_iterations):
        predictions = _sigmoid(x @ weights + bias)
        error = predictions - y
        weights -= learning_rate * (x.T @ error) / n
        bias -= learning_rate * error.mean()
        weights = np.maximum(weights, 0.0)

    return weights, float(bias)


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------


def fit_isotonic(
    probabilities: np.ndarray, labels: np.ndarray, n_knots: int = 32
) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators isotonic regression, thinned to `n_knots`.

    Returns parallel x/y arrays that `api/_lib/fuse.py` interpolates. Thinning
    keeps the shipped JSON to a couple of kilobytes without measurably
    changing the mapping.
    """
    order = np.argsort(probabilities)
    x = probabilities[order].astype(np.float64)
    y = labels[order].astype(np.float64)

    # PAVA.
    values = y.copy()
    weights = np.ones_like(values)
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total_weight = weights[i] + weights[i + 1]
        pooled = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_weight
        values[i] = pooled
        weights[i] = total_weight
        values = np.delete(values, i + 1)
        weights = np.delete(weights, i + 1)
        x = np.delete(x, i + 1)
        if i > 0:
            i -= 1

    # Thin to n_knots, always keeping the endpoints.
    if len(x) > n_knots:
        indices = np.unique(np.linspace(0, len(x) - 1, n_knots).astype(int))
        x = x[indices]
        values = values[indices]

    return x.tolist(), np.clip(values, 0.0, 1.0).tolist()


def calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict[str, float | list]:
    """Per-decile calibration error. A6 requires each bin within 5 points."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    worst = 0.0

    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (probabilities >= lower) & (
            probabilities < upper if i < n_bins - 1 else probabilities <= upper
        )
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": f"{lower:.1f}-{upper:.1f}", "n": 0})
            continue

        predicted = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        gap = abs(predicted - observed)
        worst = max(worst, gap)
        rows.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "n": count,
                "predicted": round(predicted, 4),
                "observed": round(observed, 4),
                "gap_points": round(gap * 100, 2),
            }
        )

    return {"max_gap_points": round(worst * 100, 2), "bins": rows}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    # Rank-based, so it stays O(n log n) on a full evaluation set.
    order = np.argsort(np.concatenate([positives, negatives]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum = ranks[: positives.size].sum()
    return float(
        (rank_sum - positives.size * (positives.size + 1) / 2)
        / (positives.size * negatives.size)
    )


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    threshold = np.quantile(negatives, 1.0 - target_fpr)
    return float((positives >= threshold).mean())


def fpr_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    negatives = scores[labels <= 0.5]
    if negatives.size == 0:
        return float("nan")
    return float((negatives >= threshold).mean())


def balanced_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    predictions = scores >= threshold
    positives = labels > 0.5
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    sensitivity = float(predictions[positives].mean())
    specificity = float((~predictions[negatives]).mean())
    return (sensitivity + specificity) / 2


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------


@dataclass
class Criterion:
    id: str
    description: str
    threshold: float
    #: True when higher is better.
    higher_is_better: bool = True
    blocker: bool = False
    value: float = float("nan")

    @property
    def passed(self) -> bool:
        if not np.isfinite(self.value):
            return False
        return (
            self.value >= self.threshold
            if self.higher_is_better
            else self.value <= self.threshold
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "threshold": self.threshold,
            "value": None if not np.isfinite(self.value) else round(float(self.value), 4),
            "passed": self.passed,
            "blocker": self.blocker,
        }


@dataclass
class EvalReport:
    """The numbers PRD 14 asks for, and whether each one passes."""

    criteria: list[Criterion] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def add(self, criterion: Criterion) -> None:
        self.criteria.append(criterion)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def blockers_failed(self) -> list[Criterion]:
        return [c for c in self.criteria if c.blocker and not c.passed]

    def to_dict(self) -> dict:
        return {
            "criteria": [c.to_dict() for c in self.criteria],
            "all_passed": self.all_passed,
            "blockers_failed": [c.id for c in self.blockers_failed],
            **self.extra,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        lines = [f"{'ID':<5} {'CRITERION':<46} {'GOT':>8} {'NEED':>8}  RESULT"]
        lines.append("-" * 82)
        for c in self.criteria:
            got = "n/a" if not np.isfinite(c.value) else f"{c.value:.4f}"
            need = f"{'>=' if c.higher_is_better else '<='}{c.threshold:g}"
            mark = "PASS" if c.passed else ("FAIL (BLOCKER)" if c.blocker else "FAIL")
            lines.append(f"{c.id:<5} {c.description:<46} {got:>8} {need:>8}  {mark}")
        lines.append("-" * 82)
        lines.append("ALL CRITERIA PASS" if self.all_passed else "NOT READY TO SHIP")
        return "\n".join(lines)


def run_acceptance_gate(
    *,
    in_domain: tuple[np.ndarray, np.ndarray],
    cross_domain: tuple[np.ndarray, np.ndarray],
    essays: tuple[np.ndarray, np.ndarray],
    esl: tuple[np.ndarray, np.ndarray],
    humanized: tuple[np.ndarray, np.ndarray],
    calibrated: tuple[np.ndarray, np.ndarray],
    onnx_max_logit_delta: float = float("nan"),
    bundle_sizes_mb: dict[str, float] | None = None,
) -> EvalReport:
    """Evaluate every criterion in PRD 14 that this stage can measure.

    Each argument is a ``(scores, labels)`` pair on a held-out split.
    A7 and A8 come from the humanizer's own runs, not from here.
    """
    report = EvalReport()

    scores, labels = in_domain
    report.add(
        Criterion("A1", "AUROC, in-domain (RAID + DACTYL test)", 0.95, value=auroc(scores, labels))
    )

    scores, labels = cross_domain
    report.add(
        Criterion(
            "A2",
            "Balanced accuracy, cross-domain (M4)",
            0.85,
            value=balanced_accuracy(scores, labels),
        )
    )

    scores, labels = essays
    report.add(
        Criterion(
            "A3", "TPR at 1% FPR, academic essays", 0.80, value=tpr_at_fpr(scores, labels, 0.01)
        )
    )

    scores, labels = esl
    report.add(
        Criterion(
            "A4",
            "FPR on human ESL/ELL writing",
            0.05,
            higher_is_better=False,
            blocker=True,
            value=fpr_at_threshold(scores, labels, 0.5),
        )
    )

    scores, labels = humanized
    report.add(
        Criterion(
            "A5",
            "TPR on humanized AI text at 5% FPR (Model A)",
            0.90,
            value=tpr_at_fpr(scores, labels, 0.05),
        )
    )

    scores, labels = calibrated
    calibration = calibration_error(scores, labels)
    report.add(
        Criterion(
            "A6",
            "Calibration error, worst decile (points)",
            5.0,
            higher_is_better=False,
            value=float(calibration["max_gap_points"]),
        )
    )
    report.extra["calibration"] = calibration

    report.add(
        Criterion(
            "A9",
            "ONNX INT8 parity, max logit delta",
            0.01,
            higher_is_better=False,
            value=onnx_max_logit_delta,
        )
    )

    if bundle_sizes_mb:
        largest = max(bundle_sizes_mb.values())
        report.add(
            Criterion(
                "A11",
                "Largest function bundle (MB)",
                500.0,
                higher_is_better=False,
                value=largest,
            )
        )
        report.extra["bundle_sizes_mb"] = bundle_sizes_mb

    return report


def write_fusion_config(
    path: Path,
    weights: np.ndarray,
    bias: float,
    calibration_x: list[float],
    calibration_y: list[float],
    version: str,
) -> None:
    """Write the artefact `api/_lib/fuse.py` loads at inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": version,
                "signals": list(SIGNAL_NAMES),
                "weights": [round(float(w), 6) for w in weights],
                "bias": round(float(bias), 6),
                "calibration_x": [round(float(v), 6) for v in calibration_x],
                "calibration_y": [round(float(v), 6) for v in calibration_y],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
