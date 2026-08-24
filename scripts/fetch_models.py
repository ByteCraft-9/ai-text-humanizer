#!/usr/bin/env python3
"""Download ONNX weights into `models/` at build time (PRD 7.2).

Weights are published to a public Hugging Face repo and fetched during the
Vercel build rather than committed. GitHub's free LFS allowance is 1 GB of
storage and 1 GB of bandwidth per month, which repeated deploys would exhaust
within days; Hub hosting for a public repo is free and unmetered.

Every download is optional. A missing artefact makes the corresponding signal
unavailable and the app reports a degraded score — the build does not fail,
because a deploy that cannot fetch a model is still a deploy that can detect.

    python scripts/fetch_models.py                 # fetch what is configured
    python scripts/fetch_models.py --check         # report status, download nothing
    python scripts/fetch_models.py --base-models   # also fetch gpt2 / MiniLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# The repo holding your trained detectors. Set MODEL_REPO in the Vercel
# project settings once Phase 4 has published them.
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")


@dataclass(frozen=True)
class Artefact:
    """One file to fetch. `repo` empty means "from MODEL_REPO"."""

    local: str
    remote: str
    repo: str = ""
    required: bool = False
    note: str = ""


# Trained by training/02 and 03, exported by training/04.
TRAINED = [
    Artefact("detector_a_int8.onnx", "detector_a_int8.onnx", note="strict detector"),
    Artefact("detector_a_tokenizer.json", "detector_a_tokenizer.json"),
    Artefact("detector_b_int8.onnx", "detector_b_int8.onnx", note="surrogate detector"),
    Artefact("detector_b_tokenizer.json", "detector_b_tokenizer.json"),
    Artefact("fusion_a.json", "fusion_a.json", note="fusion weights + calibrator"),
    Artefact("fusion_b.json", "fusion_b.json"),
]

# Off-the-shelf models. These need no training, so a fresh clone can have a
# working Binoculars signal and a working similarity gate on day one.
BASE = [
    Artefact(
        "distilgpt2_int8.onnx",
        "onnx/model_quantized.onnx",
        repo="Xenova/distilgpt2",
        note="Binoculars observer",
    ),
    Artefact("gpt2_tokenizer.json", "tokenizer.json", repo="Xenova/distilgpt2"),
    Artefact(
        "gpt2_int8.onnx",
        "onnx/decoder_model_merged_quantized.onnx",
        repo="Xenova/gpt2",
        note="Binoculars performer",
    ),
    Artefact(
        "minilm_int8.onnx",
        "onnx/model_quantized.onnx",
        repo="Xenova/all-MiniLM-L6-v2",
        note="meaning-preservation encoder",
    ),
    Artefact(
        "minilm_tokenizer.json",
        "tokenizer.json",
        repo="Xenova/all-MiniLM-L6-v2",
    ),
]


def hub_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def download(url: str, target: Path) -> tuple[bool, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")

    request = urllib.request.Request(url, headers={"User-Agent": "ai-detector-build"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("content-length") or 0)
            written = 0
            with partial.open("wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    if total:
                        percent = written * 100 // total
                        print(f"\r    {percent:3d}%  {written >> 20} MB", end="", flush=True)
            print()
        # Only move into place once the whole file is on disk, so an
        # interrupted build never leaves a truncated model that loads and
        # then produces nonsense.
        partial.replace(target)
        return True, f"{written >> 20} MB"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        return False, str(exc)


def fetch(artefacts: list[Artefact], default_repo: str) -> list[str]:
    problems: list[str] = []

    for artefact in artefacts:
        target = MODELS_DIR / artefact.local
        if target.is_file() and target.stat().st_size > 0:
            print(f"  have  {artefact.local}")
            continue

        repo = artefact.repo or default_repo
        if not repo:
            problems.append(
                f"{artefact.local}: no repo configured (set MODEL_REPO once trained)"
            )
            print(f"  skip  {artefact.local} — MODEL_REPO is not set")
            continue

        label = f" ({artefact.note})" if artefact.note else ""
        print(f"  get   {artefact.local}{label}")
        ok, detail = download(hub_url(repo, artefact.remote, MODEL_REVISION), target)
        if not ok:
            problems.append(f"{artefact.local}: {detail}")
            print(f"        failed: {detail}")

    return problems


def report_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for artefact in TRAINED + BASE:
        path = MODELS_DIR / artefact.local
        status[artefact.local] = path.is_file() and path.stat().st_size > 0
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report status only")
    parser.add_argument(
        "--base-models",
        action="store_true",
        help="also fetch the off-the-shelf GPT-2 and MiniLM exports",
    )
    parser.add_argument(
        "--all", action="store_true", help="fetch trained models and base models"
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        status = report_status()
        print(json.dumps(status, indent=2))
        missing = [name for name, present in status.items() if not present]
        if missing:
            print(f"\n{len(missing)} of {len(status)} artefacts missing.", file=sys.stderr)
            print("The app will run in degraded mode and say so.", file=sys.stderr)
        return 0

    problems: list[str] = []

    print("Trained detectors:")
    problems += fetch(TRAINED, MODEL_REPO)

    if args.base_models or args.all:
        print("\nBase models:")
        problems += fetch(BASE, "")

    if problems:
        print(f"\n{len(problems)} artefact(s) unavailable:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nThe build continues. Signals backed by a missing model are "
            "reported as unavailable and the score is marked degraded.",
            file=sys.stderr,
        )
    else:
        print("\nAll configured artefacts are present.")

    # Deliberately always 0: a failed model fetch must not fail the deploy.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
