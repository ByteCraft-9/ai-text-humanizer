"""ONNX Runtime session management with graceful absence.

Two things this module exists to handle:

1.  **Cold starts.** Sessions are process-global and lazily built, so a warm
    Vercel instance reuses them across requests. Threads are pinned to 1 —
    the function has a single vCPU and ORT's default thread pool otherwise
    spends more time contending than computing.

2.  **Missing weights.** The app must be deployable before the models are
    trained (build order phase 1), and a deploy where the HF download failed
    must degrade rather than 500. Every loader returns ``None`` when its
    weights are absent, and callers mark the result ``degraded`` so the UI can
    say so plainly instead of presenting a partial score as a full one.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))

_lock = threading.Lock()
_sessions: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_missing: set[str] = set()


@dataclass(frozen=True)
class ModelSpec:
    """A model on disk: ONNX graph plus its tokenizer JSON."""

    name: str
    onnx: str
    tokenizer: str

    @property
    def onnx_path(self) -> Path:
        return MODELS_DIR / self.onnx

    @property
    def tokenizer_path(self) -> Path:
        return MODELS_DIR / self.tokenizer

    def present(self) -> bool:
        return self.onnx_path.is_file() and self.tokenizer_path.is_file()


# The four artefacts the system can use. Any subset may be absent.
DETECTOR_A = ModelSpec("detector_a", "detector_a_int8.onnx", "detector_a_tokenizer.json")
DETECTOR_B = ModelSpec("detector_b", "detector_b_int8.onnx", "detector_b_tokenizer.json")
OBSERVER = ModelSpec("observer", "distilgpt2_int8.onnx", "gpt2_tokenizer.json")
PERFORMER = ModelSpec("performer", "gpt2_int8.onnx", "gpt2_tokenizer.json")
ENCODER = ModelSpec("encoder", "minilm_int8.onnx", "minilm_tokenizer.json")


def get_session(spec: ModelSpec):
    """Return a cached InferenceSession, or ``None`` if unavailable."""
    if spec.name in _missing:
        return None
    cached = _sessions.get(spec.name)
    if cached is not None:
        return cached

    with _lock:
        if spec.name in _sessions:
            return _sessions[spec.name]
        if not spec.present():
            _missing.add(spec.name)
            return None

        try:
            import onnxruntime as ort
        except ImportError:
            _missing.add(spec.name)
            return None

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The function is stateless and short-lived; arena growth is wasted.
        options.enable_cpu_mem_arena = False

        try:
            session = ort.InferenceSession(
                str(spec.onnx_path), options, providers=["CPUExecutionProvider"]
            )
        except Exception:
            _missing.add(spec.name)
            return None

        _sessions[spec.name] = session
        return session


def get_tokenizer(spec: ModelSpec):
    """Return a cached ``tokenizers.Tokenizer``, or ``None`` if unavailable."""
    key = spec.tokenizer
    if f"tok:{key}" in _missing:
        return None
    cached = _tokenizers.get(key)
    if cached is not None:
        return cached

    with _lock:
        if key in _tokenizers:
            return _tokenizers[key]
        if not spec.tokenizer_path.is_file():
            _missing.add(f"tok:{key}")
            return None
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(spec.tokenizer_path))
        except Exception:
            _missing.add(f"tok:{key}")
            return None

        _tokenizers[key] = tokenizer
        return tokenizer


def load_json(filename: str) -> dict | None:
    path = MODELS_DIR / filename
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def available() -> dict[str, bool]:
    """Which artefacts this deployment actually has. Surfaced by /api/py/health."""
    return {
        spec.name: spec.present()
        for spec in (DETECTOR_A, DETECTOR_B, OBSERVER, PERFORMER, ENCODER)
    }
