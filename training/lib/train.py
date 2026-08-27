"""Training loop shared by notebooks 02 and 03 (PRD 13.2).

Kaggle sessions time out. Checkpoints are written every 500 steps to the
output directory and the loop resumes from the newest one, so a timeout costs
minutes rather than the whole run (PRD 13.2).
"""

from __future__ import annotations

import json
import math
import os
import threading
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
from .store import FileStore, NullStore

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
    #: Minutes between checkpoints. One save, one upload, nothing duplicated.
    #: Time-based rather than step-based because what matters is how much work
    #: a dead session costs, and steps-per-minute varies with sequence length.
    checkpoint_minutes: float = 20.0
    eval_every: int = 2_000
    seed: int = 20260824
    #: Recompute activations in the backward pass instead of storing them.
    #: Costs roughly 30% throughput and saves several gigabytes. Required on a
    #: 14.5 GB T4: DeBERTa-v3's disentangled attention builds three attention
    #: matrices per layer instead of one, so at 768 tokens the stored
    #: activations alone exceed the card.
    gradient_checkpointing: bool = True
    #: Pad each batch to its own longest sequence rather than to max_length.
    #: The corpus median is ~210 words (~280 tokens) against a 768 ceiling, so
    #: fixed padding spends most of its memory and compute on padding.
    dynamic_padding: bool = True
    #: Round padded length up to a multiple of this; tensor cores want 8.
    pad_to_multiple_of: int = 8
    #: Upload on a background thread so training does not stall for the
    #: minutes a multi-gigabyte transfer takes. Set False to upload inline.
    background_upload: bool = True
    #: Refuse to start when the validation split holds only one class. Every
    #: metric would be NaN and the run would be unmeasurable, which is not
    #: visible in the training loss. Set False only to train deliberately
    #: blind.
    require_validation: bool = True
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
        # Deliberately unpadded — `collate` pads each batch to its own longest
        # sequence. Padding here to max_length would make every batch 768
        # tokens wide regardless of content.
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "features": self.features[index],
            "label": self.labels[index],
        }


def make_collate(tokenizer, pad_to_multiple_of: int = 8):
    """Pad a batch to its own longest sequence.

    With a corpus median around 280 tokens against a 768 ceiling this cuts
    attention memory by roughly the square of the ratio, which is what makes
    the run fit alongside DeBERTa-v3's triple attention matrices.
    """

    def collate(batch: list[dict]) -> dict:
        padded = tokenizer.pad(
            {
                "input_ids": [item["input_ids"] for item in batch],
                "attention_mask": [item["attention_mask"] for item in batch],
            },
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "features": torch.from_numpy(
                np.stack([item["features"] for item in batch])
            ),
            "label": torch.from_numpy(
                np.stack([item["label"] for item in batch])
            ),
        }

    return collate


#: The single local checkpoint. Overwritten each time and uploaded as-is —
#: there is no second copy anywhere on this machine.
CHECKPOINT_NAME = "checkpoint.pt"


def _latest_checkpoint(directory: Path) -> Path | None:
    path = directory / CHECKPOINT_NAME
    if path.is_file() and path.stat().st_size > 0:
        return path
    # Checkpoints written before the single-file layout.
    legacy = sorted(directory.glob("checkpoint-*.pt"))
    if not legacy:
        return None
    return max(legacy, key=lambda p: p.stat().st_mtime)


def load_trained_state(
    output_dir: Path,
    store: FileStore | None = None,
    tag: str | None = None,
) -> dict:
    """Return the newest usable model state, from wherever it exists.

    A checkpoint carries everything ``final.pt`` does — weights, step, config —
    plus optimiser state. So a run stopped part way is still a usable model,
    and stage 4 should never demand that training ran to completion.

    Local sources are compared by step rather than by preference: a session
    that died mid-upload can leave local disk ahead of the remote copy, and
    silently taking the older one would throw away real training.
    """
    tag = tag or output_dir.name
    candidates: list[tuple[int, Path]] = []

    for name in ("final.pt", CHECKPOINT_NAME):
        path = output_dir / name
        if not (path.is_file() and path.stat().st_size > 0):
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} is unreadable ({exc}); ignoring it")
            continue
        candidates.append((int(state.get("step", 0)), path))

    if candidates:
        step, path = max(candidates, key=lambda item: item[0])
        print(f"using {path.name} from step {step:,}")
        return torch.load(path, map_location="cpu", weights_only=False)

    if store is not None:
        for remote in (f"{tag}-final.pt", f"{tag}-checkpoint.pt"):
            local = output_dir / "pulled.pt"
            if store.pull(remote, local):
                state = torch.load(local, map_location="cpu", weights_only=False)
                print(f"pulled {remote} from the store, step {state.get('step', 0):,}")
                return state

    raise FileNotFoundError(
        f"No usable weights for {tag}. Looked for final.pt and "
        f"{CHECKPOINT_NAME} in {output_dir}, then in the store. Run the "
        f"training stage for {tag} first — stopping it early is fine, the "
        f"checkpoint is enough."
    )


_upload_thread: threading.Thread | None = None


def _upload_in_background(store: FileStore, local: Path, name: str) -> bool:
    """Start an upload without blocking training.

    Returns False when a previous upload is still running, in which case this
    checkpoint is skipped entirely rather than queued — the next one is only
    `checkpoint_minutes` away, and queueing multi-gigabyte transfers behind
    each other would only fall further behind.
    """
    global _upload_thread

    if _upload_thread is not None and _upload_thread.is_alive():
        return False

    def work() -> None:
        store.push(local, name)

    _upload_thread = threading.Thread(target=work, daemon=True, name="ckpt-upload")
    _upload_thread.start()
    return True


def _wait_for_upload(timeout: float = 900.0) -> None:
    if _upload_thread is not None and _upload_thread.is_alive():
        print("  waiting for the in-flight checkpoint upload…", flush=True)
        _upload_thread.join(timeout)


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
    store: FileStore | None = None,
) -> tuple[Detector, dict]:
    """Train one detector, resuming from wherever the last run stopped.

    `store` survives the container. Without it a killed Kaggle session takes
    the whole run with it; with it, the loop pulls the newest checkpoint back
    on startup and carries on from that step.
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    config = config or TrainConfig()
    store = store or NullStore()
    output_dir.mkdir(parents=True, exist_ok=True)

    # A fixed remote name so backends that overwrite in place (Drive) do not
    # accumulate a copy per sync. The step lives inside the file.
    remote_checkpoint = f"{output_dir.name}-checkpoint.pt"
    remote_final = f"{output_dir.name}-final.pt"
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # The dataset tokenizes inside worker processes; the fast tokenizer's own
    # thread pool on top of that just contends.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # PYTORCH_CUDA_ALLOC_CONF is read when CUDA first initialises, so setting
    # it here only helps if nothing has touched the GPU yet. The notebook
    # setup cell sets it before torch is ever imported, which is the place it
    # reliably takes effect; this is a fallback for direct callers.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        print(f"  {name}, {total:.1f} GiB total, {free:.1f} GiB free")

    # Standardisation comes from the training split only — computing it over
    # validation too would leak.
    raw = train_frame[feature_columns].to_numpy(dtype=np.float64)
    feature_mean = raw.mean(axis=0)
    feature_std = raw.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    config.feature_mean = feature_mean.tolist()
    config.feature_std = feature_std.tolist()

    positive_share = float(validation_frame["label"].mean())
    if config.require_validation and not 0.0 < positive_share < 1.0:
        raise ValueError(
            f"The validation split is entirely "
            f"{'AI' if positive_share else 'human'} "
            f"({len(validation_frame):,} rows, {positive_share:.3f} positive), "
            f"so every metric would be NaN and this run could not be measured "
            f"at all. Each evaluation would also cost minutes to produce "
            f"nothing.\n\n"
            f"Use lib.data.pick_holdout_domains(frame) to choose domains that "
            f"contain both classes, or pass "
            f"TrainConfig(require_validation=False) to train blind on purpose."
        )
    print(f"validation: {len(validation_frame):,} rows, {positive_share:.3f} positive")

    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    model = Detector(DetectorConfig(backbone=config.backbone, max_length=config.max_length))

    if config.gradient_checkpointing:
        # use_reentrant=False is required for checkpointing to co-operate with
        # AMP and with inputs that do not require grad.
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.backbone.config.use_cache = False
        print("  gradient checkpointing: on")

    model.to(device)

    collate = make_collate(tokenizer, config.pad_to_multiple_of) if config.dynamic_padding else None
    if config.dynamic_padding:
        print("  dynamic padding: on")

    train_loader = DataLoader(
        TextDataset(train_frame, tokenizer, feature_columns, config.max_length, feature_mean, feature_std),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
        drop_last=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        TextDataset(validation_frame, tokenizer, feature_columns, config.max_length, feature_mean, feature_std),
        # Evaluation runs under no_grad, so it has room for a wider batch.
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
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

    if resume is None:
        # Nothing on local disk. Either this is a fresh run, or the session
        # that produced the last one was wiped — ask the remote store.
        candidate = output_dir / "checkpoint-remote.pt"
        if store.pull(remote_checkpoint, candidate):
            resume = candidate
            print(f"pulled {remote_checkpoint} from {store.describe()}")

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        print(f"resumed from {resume.name} at step {start_step}")
    else:
        print("no checkpoint found — starting from scratch")

    history: list[dict] = []
    step = start_step
    started = time.time()
    last_sync = time.time()
    model.train()

    # Optimiser steps already completed, in micro-batches. Resuming replays
    # the loader from the top, so this many micro-batches are skipped to land
    # back where the previous session stopped rather than re-training on data
    # the model has already seen this epoch.
    skip_micro_batches = start_step * config.accumulation_steps

    oom_skipped = 0

    seen_micro_batches = 0
    interrupted = False

    try:
        for epoch in range(config.epochs):
            for i, batch in enumerate(train_loader):
                # Fast-forward past work the previous session already did.
                if seen_micro_batches < skip_micro_batches:
                    seen_micro_batches += 1
                    if seen_micro_batches % 5_000 == 0:
                        print(
                            f"  skipping ahead {seen_micro_batches:,}/"
                            f"{skip_micro_batches:,}",
                            flush=True,
                        )
                    continue
                seen_micro_batches += 1

                try:
                    with torch.amp.autocast("cuda", enabled=config.fp16 and device == "cuda"):
                        logits = model(
                            batch["input_ids"].to(device),
                            batch["attention_mask"].to(device),
                            batch["features"].to(device),
                        )
                        loss = criterion(logits, batch["label"].to(device)) / config.accumulation_steps

                    scaler.scale(loss).backward()
                except torch.cuda.OutOfMemoryError:
                    # One unusually long batch must not end an eight-hour run.
                    # Drop it, reclaim, carry on — but stop if it keeps happening,
                    # because then the configuration genuinely does not fit.
                    oom_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    del batch
                    torch.cuda.empty_cache()
                    if oom_skipped > 25:
                        raise RuntimeError(
                            f"Ran out of GPU memory {oom_skipped} times — this "
                            f"configuration does not fit. Halve batch_size to "
                            f"{max(1, config.batch_size // 2)} and double "
                            f"accumulation_steps to {config.accumulation_steps * 2} "
                            f"to keep the same effective batch, or lower "
                            f"max_length from {config.max_length}."
                        ) from None
                    if oom_skipped % 5 == 1:
                        print(f"  OOM on one batch (skipped {oom_skipped} so far)", flush=True)
                    continue

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

                if (time.time() - last_sync) >= config.checkpoint_minutes * 60:
                    # An upload reads this file, so do not overwrite it while one
                    # is in flight — skip the cycle instead.
                    if _upload_thread is not None and _upload_thread.is_alive():
                        print(f"  step {step}: previous upload still running, skipping",
                              flush=True)
                        last_sync = time.time()
                    else:
                        checkpoint_path = output_dir / CHECKPOINT_NAME
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "step": step,
                                "config": asdict(config),
                            },
                            checkpoint_path,
                        )
                        size_mb = checkpoint_path.stat().st_size >> 20
                        if config.background_upload:
                            started_upload = _upload_in_background(
                                store, checkpoint_path, remote_checkpoint
                            )
                            print(
                                f"  step {step}: checkpoint {size_mb} MB, "
                                f"{'uploading in background' if started_upload else 'upload skipped'}",
                                flush=True,
                            )
                        else:
                            print(f"  step {step}: uploading {size_mb} MB…", flush=True)
                            store.push(checkpoint_path, remote_checkpoint)
                        last_sync = time.time()

                if step % config.eval_every == 0:
                    metrics = evaluate(model, validation_loader, device)
                    metrics["step"] = step
                    history.append(metrics)
                    print(f"  eval {metrics}", flush=True)

    except KeyboardInterrupt:
        # Stopping early is a legitimate way to end a run — the loss may have
        # flattened, or the GPU budget may be gone. Fall through to the same
        # save path a completed run takes, so the notebook keeps working and
        # stage 4 has something to export.
        interrupted = True
        print(f"\n\nInterrupted at step {step:,}. Saving what has trained.", flush=True)

    if oom_skipped:
        print(f"\n{oom_skipped} batch(es) were skipped on out-of-memory.")

    if interrupted:
        # A full evaluation pass costs minutes. Someone who just hit stop is
        # not waiting for it.
        final = {"interrupted_at_step": step}
    else:
        final = evaluate(model, validation_loader, device)
        history.append({**final, "step": step, "final": True})

    final_path = output_dir / "final.pt"
    torch.save(
        {"model": model.state_dict(), "step": step, "config": asdict(config)},
        final_path,
    )

    # Stage 4 needs this file and nothing else, so this upload blocks — the
    # run is over and there is no training left to overlap it with.
    _wait_for_upload()
    print(f"uploading final.pt ({final_path.stat().st_size >> 20} MB)…", flush=True)
    if store.push(final_path, remote_final):
        print(f"final.pt is safe in {store.describe()}")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print(f"\nfinal: {final}")
    return model, {
        "history": history,
        "final": final,
        "config": asdict(config),
        "interrupted": interrupted,
        "step": step,
    }
