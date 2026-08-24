#!/usr/bin/env python3
"""Measure A7 and A8 against a running deployment (PRD 14).

    A7  a document scoring > 0.9 on the surrogate falls below 0.3 within
        3 passes, for at least 90% of documents
    A8  meaning similarity retained after humanization >= 0.85

These two cannot be measured in the training notebooks: they are properties of
the loop, not the model, and the loop needs live providers. This drives the
deployed endpoint over a sample of documents and writes the numbers into the
same report the notebooks produce.

    python scripts/eval_humanizer.py --url https://your-app.vercel.app \\
        --input eval/ai_documents.jsonl --output models/eval_report.json

Input is JSONL with a "text" field per line. Only documents that actually
score above the A7 entry threshold are counted — a document the detector never
flagged says nothing about whether the humanizer works.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ENTRY_THRESHOLD = 0.9   # A7: only documents starting above this count
CLEARED_THRESHOLD = 0.3  # A7: what "cleared" means
SIMILARITY_FLOOR = 0.85  # A8


@dataclass
class Outcome:
    initial_surrogate: float
    final_surrogate: float
    initial_strict: float
    final_strict: float
    passes: int
    similarity: float
    similarity_mode: str
    outcome: str
    tokens: int

    @property
    def cleared(self) -> bool:
        return self.final_surrogate < CLEARED_THRESHOLD


@dataclass
class Results:
    outcomes: list[Outcome] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_humanize(base_url: str, text: str, byok: dict | None, timeout: float) -> dict | None:
    """POST to /api/humanize and read the SSE stream to the final frame."""
    body: dict = {"text": text}
    if byok:
        body["byok"] = byok

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/humanize",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    result: dict | None = None
    error: str | None = None

    with urllib.request.urlopen(request, timeout=timeout) as response:
        buffer = ""
        for raw in response:
            buffer += raw.decode("utf-8")
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                line = frame.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "done":
                    result = event["result"]
                elif event.get("type") == "error":
                    error = event.get("message")

    if error and result is None:
        raise RuntimeError(error)
    return result


def evaluate(
    base_url: str,
    documents: list[str],
    byok: dict | None,
    timeout: float,
    pause: float,
) -> Results:
    results = Results()

    for i, text in enumerate(documents, 1):
        print(f"[{i}/{len(documents)}] {len(text.split())} words … ", end="", flush=True)

        try:
            detected = post_json(
                f"{base_url.rstrip('/')}/api/py/detect", {"text": text}, timeout=60
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            results.errors.append(f"detect failed: {exc}")
            print("detect failed")
            continue

        if detected.get("surrogate_score", 0) <= ENTRY_THRESHOLD:
            results.skipped += 1
            print(f"skipped (starts at {detected.get('surrogate_score', 0):.2f})")
            continue

        try:
            final = stream_humanize(base_url, text, byok, timeout)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            results.errors.append(str(exc))
            print(f"error: {exc}")
            continue

        if final is None:
            results.errors.append("stream ended without a result")
            print("no result")
            continue

        outcome = Outcome(
            initial_surrogate=final["initial_surrogate_score"],
            final_surrogate=final["surrogate_score"],
            initial_strict=final["initial_strict_score"],
            final_strict=final["strict_score"],
            passes=len(final["passes"]),
            similarity=final["mean_similarity"],
            similarity_mode=final.get("similarity_mode", "semantic"),
            outcome=final["outcome"],
            tokens=final["tokens_used"],
        )
        results.outcomes.append(outcome)
        print(
            f"{outcome.initial_surrogate:.2f} -> {outcome.final_surrogate:.2f} "
            f"({outcome.outcome}, {outcome.passes} passes, "
            f"sim {outcome.similarity:.2f})"
        )

        # Free-tier providers are rate-limited by requests per minute, so a
        # tight loop would spend the run backing off.
        if pause and i < len(documents):
            time.sleep(pause)

    return results


def summarise(results: Results) -> dict:
    outcomes = results.outcomes
    if not outcomes:
        return {
            "error": "No document scored above the A7 entry threshold.",
            "skipped": results.skipped,
            "errors": results.errors,
        }

    cleared = [o for o in outcomes if o.cleared]
    clear_rate = len(cleared) / len(outcomes)
    similarities = [o.similarity for o in outcomes if o.passes > 0]
    mean_similarity = statistics.mean(similarities) if similarities else 1.0

    lexical = sum(1 for o in outcomes if o.similarity_mode == "lexical")

    return {
        "A7": {
            "description": "Surrogate >0.9 falls below 0.3 within 3 passes",
            "threshold": 0.90,
            "value": round(clear_rate, 4),
            "passed": clear_rate >= 0.90,
            "cleared": len(cleared),
            "evaluated": len(outcomes),
        },
        "A8": {
            "description": "Meaning similarity retained after humanization",
            "threshold": SIMILARITY_FLOOR,
            "value": round(mean_similarity, 4),
            "passed": mean_similarity >= SIMILARITY_FLOOR,
            # A8 is only meaningfully measured against real embeddings. If the
            # encoder was missing, say so rather than reporting a proxy as if
            # it were the specified check.
            "measured_with": "lexical proxy" if lexical else "sentence embeddings",
            "lexical_runs": lexical,
        },
        "honesty": {
            # The strict score is expected to move much less than the
            # surrogate. If it collapses too, either Model A is not hardened
            # or the humanizer is optimising against it — both are bugs.
            "mean_surrogate_drop": round(
                statistics.mean(o.initial_surrogate - o.final_surrogate for o in outcomes), 4
            ),
            "mean_strict_drop": round(
                statistics.mean(o.initial_strict - o.final_strict for o in outcomes), 4
            ),
        },
        "outcomes": {
            name: sum(1 for o in outcomes if o.outcome == name)
            for name in {o.outcome for o in outcomes}
        },
        "mean_passes": round(statistics.mean(o.passes for o in outcomes), 2),
        "mean_tokens": round(statistics.mean(o.tokens for o in outcomes)),
        "skipped": results.skipped,
        "errors": results.errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="deployment base URL")
    parser.add_argument("--input", required=True, type=Path, help="JSONL with a 'text' field")
    parser.add_argument("--output", type=Path, default=Path("models/eval_report.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--pause", type=float, default=15.0, help="seconds between documents")
    parser.add_argument("--byok-key", help="use your own key instead of the shared pool")
    parser.add_argument("--byok-model", default="llama-3.3-70b")
    parser.add_argument("--byok-base-url", default="https://api.cerebras.ai/v1")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"{args.input} not found", file=sys.stderr)
        return 1

    documents: list[str] = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if text.strip():
                documents.append(text)
            if len(documents) >= args.limit:
                break

    byok = (
        {
            "flavour": "openai",
            "apiKey": args.byok_key,
            "model": args.byok_model,
            "baseUrl": args.byok_base_url,
        }
        if args.byok_key
        else None
    )

    print(f"Evaluating {len(documents)} documents against {args.url}\n")
    summary = summarise(evaluate(args.url, documents, byok, args.timeout, args.pause))

    print("\n" + json.dumps(summary, indent=2))

    # Merge into the existing report rather than overwriting the notebooks'.
    existing = {}
    if args.output.is_file():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    existing["humanizer"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")

    failed = [k for k in ("A7", "A8") if k in summary and not summary[k]["passed"]]
    if failed:
        print(f"\nNOT READY: {', '.join(failed)} failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
