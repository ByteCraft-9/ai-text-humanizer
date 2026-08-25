"""DeBERTa-v3-base with a feature-attention branch (PRD 8.2, 8.4).

E4 found this arrangement reached 85.97% balanced accuracy on M4 cross-domain,
7.2 points ahead of Fast-DetectGPT, and attributed the gain to the handcrafted
features carrying signal that survives rewriting and domain shift where the
text encoder's representation does not.

The branch is *attention* rather than concatenation for a specific reason:
which features matter varies per sample. Readability dominates on essays,
repetition on marketing copy. A learned per-sample weighting over the 30
features lets the model pick, instead of averaging the choice away.

Export note: this module must produce an ONNX graph whose inputs are exactly
``input_ids``, ``attention_mask`` and ``features`` — that is the contract
``api/_lib/classifier.py`` feeds. Changing the signature here breaks inference
silently, so 04_export_onnx asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

N_FEATURES = 30


@dataclass
class DetectorConfig:
    backbone: str = "microsoft/deberta-v3-base"
    n_features: int = N_FEATURES
    feature_dim: int = 128
    dropout: float = 0.1
    # 768 tokens covers a full ~800-token chunk (PRD 13.2).
    max_length: int = 768


class FeatureAttention(nn.Module):
    """Per-sample attention over the standardised feature vector.

    Projects each scalar feature into its own embedding, scores it against the
    document representation from the text encoder, and returns the attended
    sum. The attention weights are returned as well — they are what makes a
    score explainable ("this was flagged mostly on repetition").
    """

    def __init__(self, n_features: int, feature_dim: int, hidden_size: int) -> None:
        super().__init__()
        # One embedding row per feature, scaled by that feature's value.
        self.feature_embedding = nn.Parameter(torch.randn(n_features, feature_dim) * 0.02)
        self.value_projection = nn.Linear(1, feature_dim)
        self.query = nn.Linear(hidden_size, feature_dim)
        self.scale = feature_dim**-0.5

    def forward(
        self, features: torch.Tensor, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [B, F]  pooled: [B, H]
        values = self.value_projection(features.unsqueeze(-1))  # [B, F, D]
        keys = values + self.feature_embedding.unsqueeze(0)  # [B, F, D]

        query = self.query(pooled).unsqueeze(1)  # [B, 1, D]
        weights = torch.softmax((query * keys).sum(-1) * self.scale, dim=-1)  # [B, F]

        attended = (weights.unsqueeze(-1) * values).sum(dim=1)  # [B, D]
        return attended, weights


class Detector(nn.Module):
    """Text encoder + feature branch + MLP head, producing one logit."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()

        backbone_config = AutoConfig.from_pretrained(self.config.backbone)
        try:
            self.backbone = AutoModel.from_pretrained(
                self.config.backbone, config=backbone_config
            )
        except ValueError as exc:
            # AutoModel reports a missing class as "Could not find XModel in
            # <module transformers>", which reads like a corrupt install but
            # almost always means transformers was upgraded to a version that
            # no longer exports it, or was changed under a running kernel.
            import transformers

            raise RuntimeError(
                f"transformers {transformers.__version__} could not build "
                f"{self.config.backbone}: {exc}. "
                f"This is an environment problem, not a data or config one. "
                f"Pin transformers below 5 and restart the kernel — an "
                f"upgrade under a running session leaves the old module "
                f"loaded and the new one on disk."
            ) from None
        hidden = backbone_config.hidden_size

        self.feature_attention = FeatureAttention(
            self.config.n_features, self.config.feature_dim, hidden
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + self.config.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over real tokens only — padding must not dilute the average."""
        expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * expanded).sum(dim=1)
        counts = expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
        return_weights: bool = False,
    ):
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(encoded.last_hidden_state, attention_mask)

        attended, weights = self.feature_attention(features, pooled)
        combined = torch.cat([self.dropout(pooled), attended], dim=-1)
        logits = self.head(combined).squeeze(-1)

        if return_weights:
            return logits, weights
        return logits


class ExportWrapper(nn.Module):
    """Fixes the forward signature for ONNX export.

    ``torch.onnx.export`` traces positional arguments, so the boolean flag on
    ``Detector.forward`` would become a graph input. This wrapper pins it and
    returns a ``[B, 1]`` logit, which is what the inference code reads.
    """

    def __init__(self, model: Detector) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask, features).unsqueeze(-1)


def build(backbone: str = "microsoft/deberta-v3-base") -> Detector:
    return Detector(DetectorConfig(backbone=backbone))
