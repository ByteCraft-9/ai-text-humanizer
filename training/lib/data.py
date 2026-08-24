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
import itertools

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
    #: Hard ceiling on rows *read* from the stream. Without this the sampler
    #: drains all of RAID (6.2M generations) no matter how few rows you want,
    #: so a 5,000-row smoke test costs the same hours as a full run.
    #: None derives a sensible bound from `total`.
    scan_limit: int | None = None
    #: Buffer for the streaming shuffle. Early exit means we only ever see the
    #: front of the stream, and RAID is grouped by model and domain, so
    #: without this a bounded scan would return two generators and one domain.
    shuffle_buffer: int = 10_000

    @property
    def effective_scan_limit(self) -> int:
        if self.scan_limit is not None:
            return self.scan_limit
        # Length filtering and label balancing both discard rows, so read a
        # multiple of what we intend to keep.
        return max(60_000, self.total * 8)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raid(
    split: str = "train",
    streaming: bool = True,
    seed: int | None = None,
    buffer_size: int = 10_000,
) -> Iterator[dict]:
    """Stream RAID.

    Full RAID is 16.7 GB; the non-adversarial train split is 802 MB. Streaming
    keeps Kaggle's disk allowance out of the picture entirely.
    """
    from datasets import load_dataset

    dataset = load_dataset("liamdugan/raid", split=split, streaming=streaming)
    if streaming and seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
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


def load_dactyl(
    split: str = "train",
    streaming: bool = True,
    seed: int | None = None,
    buffer_size: int = 10_000,
) -> Iterator[dict]:
    """Stream DACTYL — the modern generators RAID predates.

    GPT-4o, Claude 3.5, Gemini, DeepSeek-V3 and Llama. Without this the
    detector is trained on a generation of models nobody uses any more.
    """
    from datasets import load_dataset

    dataset = load_dataset("ShantanuT01/DACTYL", split=split, streaming=streaming)
    if streaming and seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
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

    Reservoir-style per stratum so the whole corpus never has to fit in memory.

    The scan is **bounded** by ``spec.effective_scan_limit``. It has to be:
    without a bound this loop drains the entire iterator, so asking for 5,000
    rows costs exactly as long as asking for 400,000 — it reads all 6.2M RAID
    generations either way. The bound is what makes a smoke test a smoke test.

    A bounded scan only sees the front of the stream, so the loaders shuffle
    with a buffer; RAID is grouped by model and domain, and without that
    shuffle a short scan would come back with two generators and one domain.
    """
    rng = np.random.default_rng(spec.seed)

    n_adversarial = int(spec.total * spec.adversarial_share) if include_adversarial else 0
    n_clean = spec.total - n_adversarial

    # Per-stratum reservoirs, keyed by (label, generator, domain).
    clean: dict[tuple, list[dict]] = {}
    adversarial: dict[tuple, list[dict]] = {}
    seen: dict[tuple, int] = {}

    scan_limit = spec.effective_scan_limit

    # Per-stratum caps are computed **per class**, not globally.
    #
    # RAID has roughly eleven AI generators but only one "human" generator, so
    # AI strata outnumber human strata about 11:1. A single uniform cap would
    # hard-limit the human class to a fraction of the AI class no matter how
    # the quota is set, and the frame would come back skewed with nothing to
    # say why. Each class instead gets a cap sized to its own quota divided by
    # the strata discovered *for that class*, recomputed as new strata appear.
    #
    # OVERSAMPLE leaves the final balancing step room to choose.
    OVERSAMPLE = 1.4
    MIN_CAP = 50

    quota = {"clean": n_clean, "adversarial": max(1, n_adversarial)}
    strata_by_class: dict[tuple[str, int], set] = {}
    caps: dict[tuple[str, int], int] = {}

    def recompute_cap(group: str, label: int, buckets: dict[tuple, list[dict]]) -> None:
        """Shrink this class's cap as more of its strata are discovered."""
        keys = strata_by_class[(group, label)]
        share = spec.human_share if label == 0 else 1.0 - spec.human_share
        target = quota[group] * share * OVERSAMPLE
        new_cap = max(MIN_CAP, int(target / max(1, len(keys))) + 1)
        old_cap = caps.get((group, label))
        caps[(group, label)] = new_cap
        if old_cap is not None and new_cap < old_cap:
            # Trim in place so memory tracks the new cap.
            for key in keys:
                bucket = buckets.get(key)
                if bucket and len(bucket) > new_cap:
                    del bucket[new_cap:]

    scanned = 0
    kept = 0
    rejected_length = 0
    report_every = max(2_000, scan_limit // 20)
    stale = 0

    def held(buckets: dict[tuple, list[dict]]) -> int:
        return sum(len(b) for b in buckets.values())

    def class_held(buckets: dict[tuple, list[dict]], group: str, label: int) -> int:
        keys = strata_by_class.get((group, label), ())
        return sum(len(buckets[k]) for k in keys if k in buckets)

    def satisfied() -> bool:
        """Both classes of every active group hold at least their quota."""
        groups = [("clean", clean)]
        if include_adversarial:
            groups.append(("adversarial", adversarial))
        for group, buckets in groups:
            for label, share in ((0, spec.human_share), (1, 1.0 - spec.human_share)):
                if class_held(buckets, group, label) < quota[group] * share:
                    return False
        return True

    for row in rows:
        scanned += 1

        if scanned % report_every == 0:
            print(
                f"  scanned {scanned:,}/{scan_limit:,} · held {held(clean):,} clean"
                f" + {held(adversarial):,} adversarial"
                f" · {len(clean) + len(adversarial)} strata",
                flush=True,
            )

        if not _acceptable(row["text"]):
            rejected_length += 1
        elif row["adversarial"] and not include_adversarial:
            pass
        else:
            label = row["label"]
            group = "adversarial" if row["adversarial"] else "clean"
            key = (label, row["generator"], row["domain"])
            target = adversarial if row["adversarial"] else clean
            bucket = target.setdefault(key, [])

            known = strata_by_class.setdefault((group, label), set())
            if key not in known:
                known.add(key)
                recompute_cap(group, label, target)
                stale = 0

            cap = caps.get((group, label), MIN_CAP)
            seen[key] = seen.get(key, 0) + 1
            if len(bucket) < cap:
                bucket.append(row)
                kept += 1
                stale = 0
            else:
                # Reservoir replacement: every row keeps an equal chance.
                j = int(rng.integers(0, seen[key]))
                if j < cap:
                    bucket[j] = row
                stale += 1

        if scanned >= scan_limit:
            print(f"  reached the scan limit of {scan_limit:,} rows", flush=True)
            break

        # Every reservoir is full, both classes have their quota, and nothing
        # new has been admitted for a long stretch. Reading further only
        # reshuffles what we already hold.
        if stale >= 20_000 and satisfied():
            print(f"  quota met after {scanned:,} rows", flush=True)
            break

    print(
        f"  scanned {scanned:,} · kept {kept:,} · "
        f"{rejected_length:,} rejected on length ({MIN_WORDS}-{MAX_WORDS} words)",
        flush=True,
    )

    def flatten(buckets: dict[tuple, list[dict]], want: int, group: str) -> list[dict]:
        if want <= 0 or not buckets:
            return []
        pooled = [row for bucket in buckets.values() for row in bucket]
        rng.shuffle(pooled)

        # Balance the labels within the quota, since strata are uneven.
        humans = [r for r in pooled if r["label"] == 0]
        ais = [r for r in pooled if r["label"] == 1]
        want_human = int(want * spec.human_share)
        n_human = min(len(humans), want_human)
        n_ai = min(len(ais), want - want_human) 

        # Report a shortfall rather than silently returning a skewed frame.
        # RAID holds on the order of 15k human documents against millions of
        # generations, so a large human quota can be genuinely unmeetable from
        # these sources. That is a fact about the data, not a bug, and it
        # needs saying out loud before eight hours of training run on it.
        if n_human < want_human:
            print(
                f"  [{group}] wanted {want_human:,} human rows, found "
                f"{n_human:,}. Human text is the scarce class: RAID holds "
                f"~15k human documents against millions of generations. "
                f"Raise SampleSpec(scan_limit=...) to read further into the "
                f"stream, or lower total / human_share to what the data can "
                f"actually support.",
                flush=True,
            )
        if n_ai < want - n_human:
            print(
                f"  [{group}] short on AI rows: {n_ai:,} of {want - n_human:,}",
                flush=True,
            )

        return humans[:n_human] + ais[:n_ai]

    selected = flatten(clean, n_clean, "clean") + flatten(
        adversarial, n_adversarial, "adversarial"
    )
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
        raid_iter = load_raid(seed=spec.seed, buffer_size=spec.shuffle_buffer)
        dactyl_iter = load_dactyl(seed=spec.seed, buffer_size=spec.shuffle_buffer)
        for r, d in itertools.zip_longest(raid_iter, dactyl_iter):
            if r is not None:
                yield r
            if d is not None:
                yield d

    print(
        f"Sampling up to {spec.total:,} rows, reading at most "
        f"{spec.effective_scan_limit:,} from the stream…",
        flush=True,
    )
    frame = stratified_sample(stream(), spec, include_adversarial)
    print(f"  {len(frame):,} rows", flush=True)
    print(frame.groupby(["label", "adversarial"]).size(), flush=True)

    print("Extracting features…", flush=True)
    frame = add_features(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"Wrote {output} ({output.stat().st_size >> 20} MB)", flush=True)
    return frame
