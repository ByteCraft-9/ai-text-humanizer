"""Training loop shared by notebooks 02 and 03 (PRD 13.2).

Kaggle sessions time out. Checkpoints are written every 500 steps to the
output directory and the loop resumes from the newest one, so a timeout costs
minutes rather than the whole run (PRD 13.2).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .losses import TwoWayPartialAUROCLoss, partial_auroc, tpr_at_fpr
from .model import Detector, DetectorConfig

FEATURE_COLUMNS: list[str] | None = None  # set by TextDataset on first use


@dataclass
class TrainConfig:
    backbone: str = "microsoft/deberta-v3-base"
    max_length: int = 768
    batch_size: int = 16
    accumulation_steps: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    epochs: int = 3
    weight_decay: float = 0.01
    fp16: bool = True
    checkpoint_every: int = 500
    eval_every: int = 2_000
    seed: int = 20260824
    # Standardisation statistics from the training split; written alongside
    # the checkpoint so inference can use the same ones.
    feature_mean: list[float] | None = None
    feature_std: list[float] | None = None


class TextDataset(Dataset):
    """Tokenised text plus the standardised 30-feature vector."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        feature_columns: list[str],
        max_length: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        raw = frame[feature_columns].to_numpy(dtype=np.float32)
        self.features = np.nan_to_num(
            (raw - feature_mean) / feature_std, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "features": torch.from_numpy(self.features[index]),
            "label": torch.tensor(self.labels[index]),
        }


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(directory.glob("checkpoint-*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.stem.split("-")[-1]))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["features"].to(device),
        )
        scores.append(logits.detach().float().cpu())
        labels.append(batch["label"].float().cpu())

    all_scores = torch.cat(scores)
    all_labels = torch.cat(labels)

    positives = all_scores[all_labels > 0.5]
    negatives = all_scores[all_labels <= 0.5]
    if positives.numel() and negatives.numel():
        comparisons = (positives.unsqueeze(1) > negatives.unsqueeze(0)).float()
        ties = (positives.unsqueeze(1) == negatives.unsqueeze(0)).float() * 0.5
        auroc = (comparisons + ties).mean().item()
    else:
        auroc = float("nan")

    model.train()
    return {
        "auroc": auroc,
        "partial_auroc_5": partial_auroc(all_scores, all_labels, 0.05),
        "tpr_at_1_fpr": tpr_at_fpr(all_scores, all_labels, 0.01),
        "tpr_at_5_fpr": tpr_at_fpr(all_scores, all_labels, 0.05),
    }


def train(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    config: TrainConfig | None = None,
) -> tuple[Detector, dict]:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = config or TrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Standardisation comes from the training split only — computing it over
    # validation too would leak.
    raw = train_frame[feature_columns].to_numpy(dtype=np.float64)
    feature_mean = raw.mean(axis=0)
    feature_std = raw.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    config.feature_mean = feature_mean.tolist()
    config.feature_std = feature_std.tolist()

    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    model = Detector(DetectorConfig(backbone=config.backbone, max_length=config.max_length))
    model.to(device)

    train_loader = DataLoader(
        TextDataset(train_frame, tokenizer, feature_columns, config.max_length, feature_mean, feature_std),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        TextDataset(validation_frame, tokenizer, feature_columns, config.max_length, feature_mean, feature_std),
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=2,
    )

    criterion = TwoWayPartialAUROCLoss(alpha=0.05, beta=0.80)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    steps_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.fp16 and device == "cuda")

    start_step = 0
    resume = _latest_checkpoint(output_dir)
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        print(f"resumed from {resume.name} at step {start_step}")

    history: list[dict] = []
    step = start_step
    started = time.time()
    model.train()

    for epoch in range(config.epochs):
        for i, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", enabled=config.fp16 and device == "cuda"):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["features"].to(device),
                )
                loss = criterion(logits, batch["label"].to(device)) / config.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % config.accumulation_steps != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % 50 == 0:
                elapsed = time.time() - started
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {loss.item() * config.accumulation_steps:.4f} "
                    f"({elapsed / 60:.1f} min)",
                    flush=True,
                )

            if step % config.checkpoint_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "config": asdict(config),
                    },
                    output_dir / f"checkpoint-{step}.pt",
                )
                # Keep only the two newest; Kaggle output is capped at 20 GB.
                for old in sorted(output_dir.glob("checkpoint-*.pt"))[:-2]:
                    old.unlink(missing_ok=True)

            if step % config.eval_every == 0:
                metrics = evaluate(model, validation_loader, device)
                metrics["step"] = step
                history.append(metrics)
                print(f"  eval {metrics}", flush=True)

    final = evaluate(model, validation_loader, device)
    history.append({**final, "step": step, "final": True})

    torch.save(
        {"model": model.state_dict(), "step": step, "config": asdict(config)},
        output_dir / "final.pt",
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print(f"\nfinal: {final}")
    return model, {"history": history, "final": final, "config": asdict(config)}
