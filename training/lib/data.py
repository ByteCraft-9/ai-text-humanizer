"""Dataset construction from RAID and DACTYL (PRD 12).

The single most important decision in this file is the labelling of
adversarial rows. RAID ships paraphrase, synonym-swap, homoglyph, whitespace
and misspelling variants of every generation (E11), which is exactly the
humanized-text augmentation DAMAGE built by hand — at zero API cost.

Following E8's scheme:

  * humanized **AI** text is labelled AI — the transform does not launder it;
  * humanized **human** text is labelled human — this is the part people get
    wrong. Labelling a rewritten human essay as AI teaches the model that
    *editing* is evidence of machine authorship, which is precisely how
    detectors end up punishing careful writers and non-native speakers.

Model A trains on the adversarial rows (oversampled to ~15% of the mix).
Model B trains without them, which is what makes it a faithful surrogate for
third-party detectors that have no such hardening.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# The feature extractor is shared with inference, byte for byte. Divergence
# between training-time and inference-time features is a silent accuracy bug,
# so there is exactly one implementation and training imports it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
from _lib.features import FEATURE_NAMES, extract_features  # noqa: E402
from _lib.segment import split_sentences  # noqa: E402

__all__ = [
    "FEATURE_NAMES",
    "SampleSpec",
    "add_features",
    "build_dataset",
    "load_dactyl",
    "load_raid",
    "stratified_sample",
]

# RAID attack column values that represent a humanizing transform.
ADVERSARIAL_ATTACKS = frozenset(
    {
        "paraphrase",
        "synonym",
        "homoglyph",
        "whitespace",
        "misspelling",
        "article_deletion",
        "insert_paragraphs",
        "upper_lower",
        "zero_width_space",
        "number",
        "alternative_spelling",
    }
)

MIN_WORDS = 50
MAX_WORDS = 1_200


@dataclass
class SampleSpec:
    """How many rows to draw and how to balance them."""

    total: int = 400_000
    #: Share of the mix that is adversarial. Model A only; B uses 0.0.
    adversarial_share: float = 0.15
    #: Target share of human-labelled rows.
    human_share: float = 0.5
    seed: int = 20260824


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raid(split: str = "train", streaming: bool = True) -> Iterator[dict]:
    """Stream RAID.

    Full RAID is 16.7 GB; the non-adversarial train split is 802 MB. Streaming
    keeps Kaggle's disk allowance out of the picture entirely.
    """
    from datasets import load_dataset

    dataset = load_dataset("liamdugan/raid", split=split, streaming=streaming)
    for row in dataset:
        model = (row.get("model") or "").lower()
        attack = (row.get("attack") or "none").lower()
        text = row.get("generation") or ""

        yield {
            "text": text,
            # RAID marks human rows with model == "human".
            "label": 0 if model == "human" else 1,
            "source": "raid",
            "generator": model,
            "domain": row.get("domain") or "unknown",
            "attack": attack,
            "adversarial": attack in ADVERSARIAL_ATTACKS,
            "decoding": row.get("decoding") or "unknown",
        }


def load_dactyl(split: str = "train", streaming: bool = True) -> Iterator[dict]:
    """Stream DACTYL — the modern generators RAID predates.

    GPT-4o, Claude 3.5, Gemini, DeepSeek-V3 and Llama. Without this the
    detector is trained on a generation of models nobody uses any more.
    """
    from datasets import load_dataset

    dataset = load_dataset("ShantanuT01/DACTYL", split=split, streaming=streaming)
    for row in dataset:
        label = int(row.get("label", 0))
        yield {
            "text": row.get("text") or "",
            "label": label,
            "source": "dactyl",
            "generator": row.get("model") or ("human" if label == 0 else "unknown"),
            "domain": row.get("domain") or row.get("source") or "unknown",
            "attack": "none",
            "adversarial": False,
            "decoding": "unknown",
        }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _acceptable(text: str) -> bool:
    words = len(text.split())
    return MIN_WORDS <= words <= MAX_WORDS


def stratified_sample(
    rows: Iterator[dict],
    spec: SampleSpec,
    include_adversarial: bool,
) -> pd.DataFrame:
    """Draw a balanced sample, stratified by generator, domain and label.

    Reservoir-style per stratum so the whole corpus never has to fit in
    memory, and so a stratum that appears late in the stream is not starved.
    """
    rng = np.random.default_rng(spec.seed)

    n_adversarial = int(spec.total * spec.adversarial_share) if include_adversarial else 0
    n_clean = spec.total - n_adversarial

    # Per-stratum reservoirs, keyed by (label, generator, domain).
    clean: dict[tuple, list[dict]] = {}
    adversarial: dict[tuple, list[dict]] = {}
    seen: dict[tuple, int] = {}

    # A generous per-stratum cap keeps any one generator from dominating.
    cap = max(50, spec.total // 200)

    for row in rows:
        if not _acceptable(row["text"]):
            continue
        if row["adversarial"] and not include_adversarial:
            continue

        key = (row["label"], row["generator"], row["domain"])
        target = adversarial if row["adversarial"] else clean
        bucket = target.setdefault(key, [])

        seen[key] = seen.get(key, 0) + 1
        if len(bucket) < cap:
            bucket.append(row)
        else:
            # Reservoir replacement: every row keeps an equal chance.
            j = int(rng.integers(0, seen[key]))
            if j < cap:
                bucket[j] = row

    def flatten(buckets: dict[tuple, list[dict]], quota: int) -> list[dict]:
        if quota <= 0 or not buckets:
            return []
        pooled = [row for bucket in buckets.values() for row in bucket]
        rng.shuffle(pooled)

        # Balance the labels within the quota, since strata are uneven.
        humans = [r for r in pooled if r["label"] == 0]
        ais = [r for r in pooled if r["label"] == 1]
        n_human = min(len(humans), int(quota * spec.human_share))
        n_ai = min(len(ais), quota - n_human)
        return humans[:n_human] + ais[:n_ai]

    selected = flatten(clean, n_clean) + flatten(adversarial, n_adversarial)
    rng.shuffle(selected)

    frame = pd.DataFrame(selected)
    if frame.empty:
        raise RuntimeError(
            "No rows survived sampling. Check the dataset split names and that "
            "MIN_WORDS/MAX_WORDS are not excluding everything."
        )
    return frame


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def add_features(frame: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Attach the 30 features as columns, using the inference extractor."""
    vectors = np.zeros((len(frame), len(FEATURE_NAMES)), dtype=np.float32)

    for i, text in enumerate(frame["text"].tolist()):
        vectors[i] = extract_features(text, split_sentences(text))
        if show_progress and i % 5_000 == 0:
            print(f"  features {i:,} / {len(frame):,}", flush=True)

    features = pd.DataFrame(vectors, columns=list(FEATURE_NAMES), index=frame.index)
    return pd.concat([frame, features], axis=1)


def feature_statistics(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Means and standard deviations to write back into `api/_lib/features.py`.

    The defaults shipped there are placeholders. Standardising with the real
    training-set statistics is what makes the feature branch and the degraded
    scorer behave as intended.
    """
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    std = values.std(axis=0)
    # A zero-variance feature would divide by zero; 1.0 leaves it untouched.
    std[std < 1e-6] = 1.0
    return {"mean": values.mean(axis=0).tolist(), "std": std.tolist()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_dataset(
    output: Path,
    spec: SampleSpec | None = None,
    include_adversarial: bool = True,
    limit_raid: int | None = None,
) -> pd.DataFrame:
    """Build one training Parquet.

    Called twice: once with ``include_adversarial=True`` for Model A, once
    with ``False`` for Model B.
    """
    spec = spec or SampleSpec()

    def stream() -> Iterator[dict]:
        count = 0
        for row in load_raid():
            yield row
            count += 1
            if limit_raid and count >= limit_raid:
                break
        yield from load_dactyl()

    print("Sampling…", flush=True)
    frame = stratified_sample(stream(), spec, include_adversarial)
    print(f"  {len(frame):,} rows", flush=True)
    print(frame.groupby(["label", "adversarial"]).size(), flush=True)

    print("Extracting features…", flush=True)
    frame = add_features(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"Wrote {output} ({output.stat().st_size >> 20} MB)", flush=True)
    return frame
