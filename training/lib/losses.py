"""Two-way partial AUROC loss (PRD 8.6, evidence E6).

Cross-entropy optimises accuracy averaged over the whole ROC curve, but the
costs here are not symmetric: falsely accusing a human writer is materially
worse than missing a piece of AI text. A4 makes the false-positive rate on
ESL/ELL writing a release blocker, and A3 asks for TPR at 1% FPR — both are
statements about one narrow region of the curve.

Partial AUROC optimises exactly that region. "Two-way" means it is bounded on
both axes: the area is restricted to FPR <= alpha *and* TPR >= beta, so the
loss cannot be minimised by a model that simply refuses to predict the
positive class.

Implemented with a soft (sigmoid) surrogate for the indicator function, since
the true AUROC is piecewise constant and has no useful gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_indicator(difference: torch.Tensor, gamma: float = 4.0) -> torch.Tensor:
    """Differentiable stand-in for ``1[difference > 0]``.

    ``gamma`` controls sharpness. Too high and gradients vanish everywhere
    except near the boundary; too low and the loss stops resembling AUROC.
    Around 4 is the usable middle.
    """
    return torch.sigmoid(gamma * difference)


class TwoWayPartialAUROCLoss(nn.Module):
    """1 - normalised two-way partial AUROC.

    Args:
        alpha: upper bound on false-positive rate. 0.05 matches A4.
        beta: lower bound on true-positive rate.
        gamma: sharpness of the soft indicator.
        bce_weight: small cross-entropy term. Partial AUROC is rank-based and
            says nothing about *where* scores sit, so without an anchor the
            outputs drift and the calibrator has nothing stable to fit. This
            keeps them in a sane range without dominating the objective.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.80,
        gamma: float = 4.0,
        bce_weight: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1).float()
        labels = labels.reshape(-1).float()

        positives = logits[labels > 0.5]
        negatives = logits[labels <= 0.5]

        anchor = F.binary_cross_entropy_with_logits(logits, labels)

        # A batch with only one class carries no ranking information.
        if positives.numel() == 0 or negatives.numel() == 0:
            return anchor

        # Restrict to the hard region: the highest-scoring negatives (the ones
        # that would become false positives first) and the lowest-scoring
        # positives (the ones missed first).
        n_neg = max(1, int(round(self.alpha * negatives.numel())))
        n_pos = max(1, int(round((1.0 - self.beta) * positives.numel())))

        hard_negatives = torch.topk(negatives, k=n_neg, largest=True).values
        hard_positives = torch.topk(positives, k=n_pos, largest=False).values

        # Pairwise margins over that region only.
        differences = hard_positives.unsqueeze(1) - hard_negatives.unsqueeze(0)
        partial_auc = soft_indicator(differences, self.gamma).mean()

        return (1.0 - partial_auc) + self.bce_weight * anchor


@torch.no_grad()
def partial_auroc(
    scores: torch.Tensor, labels: torch.Tensor, max_fpr: float = 0.05
) -> float:
    """Exact (non-differentiable) partial AUROC, for evaluation.

    Standardised McClish-style so a random model scores 0.5 rather than
    ``max_fpr / 2``, which makes the number comparable to full AUROC.
    """
    scores = scores.reshape(-1).float()
    labels = labels.reshape(-1).float()

    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return float("nan")

    # Threshold at the (1 - max_fpr) quantile of the negatives.
    threshold = torch.quantile(negatives, 1.0 - max_fpr)
    considered = negatives[negatives >= threshold]
    if considered.numel() == 0:
        return float("nan")

    comparisons = (positives.unsqueeze(1) > considered.unsqueeze(0)).float()
    ties = (positives.unsqueeze(1) == considered.unsqueeze(0)).float() * 0.5
    raw = (comparisons + ties).mean().item()

    minimum = max_fpr / 2.0
    return 0.5 * (1.0 + (raw - minimum) / (1.0 - minimum))


@torch.no_grad()
def tpr_at_fpr(scores: torch.Tensor, labels: torch.Tensor, target_fpr: float) -> float:
    """True-positive rate at a fixed false-positive rate. A3 and A5 use this."""
    scores = scores.reshape(-1).float()
    labels = labels.reshape(-1).float()

    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return float("nan")

    threshold = torch.quantile(negatives, 1.0 - target_fpr)
    return (positives >= threshold).float().mean().item()
