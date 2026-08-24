"""ONNX export and INT8 quantisation (PRD 13.3 step 4, A9, A11).

The graph contract is fixed and asserted here: inputs named ``input_ids``,
``attention_mask`` and ``features``; one output; batch and sequence axes
dynamic. `api/_lib/classifier.py` feeds exactly those names and refuses to
guess if they differ, so an export that quietly renames an input degrades to
"model unavailable" rather than to wrong scores — but it is far better to
catch it here.

A9 requires INT8 parity with PyTorch within 0.01 max logit delta. Quantisation
that silently costs accuracy is the classic way a model that passed evaluation
ships as something worse, so parity is verified, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import Detector, ExportWrapper

INPUT_NAMES = ["input_ids", "attention_mask", "features"]
OUTPUT_NAMES = ["logits"]
OPSET = 17


def export_onnx(
    model: Detector,
    output: Path,
    max_length: int = 768,
    n_features: int = 30,
) -> Path:
    model.eval()
    wrapper = ExportWrapper(model).eval()
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy = (
        torch.ones(1, max_length, dtype=torch.long),
        torch.ones(1, max_length, dtype=torch.long),
        torch.zeros(1, n_features, dtype=torch.float32),
    )

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )

    verify_graph(output)
    return output


def verify_graph(path: Path) -> None:
    """Fail loudly if the graph does not match what inference feeds."""
    import onnx

    graph = onnx.load(str(path)).graph
    names = [i.name for i in graph.input]
    if names != INPUT_NAMES:
        raise RuntimeError(
            f"ONNX inputs are {names}, but api/_lib/classifier.py feeds "
            f"{INPUT_NAMES}. Fix the export before publishing."
        )
    if len(graph.output) != 1:
        raise RuntimeError(f"expected 1 output, graph has {len(graph.output)}")


def quantize(source: Path, target: Path) -> Path:
    """Dynamic INT8 quantisation.

    Dynamic rather than static: it needs no calibration dataset, and for a
    transformer whose cost is dominated by MatMul the difference against
    static quantisation is small. It takes DeBERTa-v3-base from ~740 MB to
    ~184 MB, which is what makes the 500 MB bundle budget work (PRD 9.1).
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    return target


def check_parity(
    model: Detector,
    onnx_path: Path,
    tokenizer,
    texts: list[str],
    features: np.ndarray,
    max_length: int = 768,
) -> dict:
    """Compare PyTorch and ONNX logits on the same inputs (A9)."""
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    feature_tensor = torch.from_numpy(features.astype(np.float32))

    with torch.no_grad():
        torch_logits = (
            model(encoded["input_ids"], encoded["attention_mask"], feature_tensor)
            .float()
            .numpy()
            .reshape(-1)
        )

    onnx_logits = np.asarray(
        session.run(
            None,
            {
                "input_ids": encoded["input_ids"].numpy().astype(np.int64),
                "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
                "features": features.astype(np.float32),
            },
        )[0]
    ).reshape(-1)

    delta = np.abs(torch_logits - onnx_logits)
    return {
        "max_logit_delta": float(delta.max()),
        "mean_logit_delta": float(delta.mean()),
        "n_samples": len(texts),
        "passes_a9": bool(delta.max() < 0.01),
    }


def export_tokenizer(tokenizer, output: Path) -> Path:
    """Write the fast tokenizer as the single JSON the runtime loads."""
    output.parent.mkdir(parents=True, exist_ok=True)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(
            "This tokenizer has no fast backend. Load it with use_fast=True — "
            "the inference runtime uses `tokenizers`, not `transformers`."
        )
    backend.save(str(output))
    return output


def report_sizes(models_dir: Path) -> dict[str, float]:
    """Per-function bundle estimate in MB (A11).

    Library sizes are the measured installed footprint of the pinned versions
    in requirements.txt.
    """
    library_mb = 80.0  # onnxruntime 50 + numpy 25 + tokenizers 5

    def size(name: str) -> float:
        path = models_dir / name
        return path.stat().st_size / (1024 * 1024) if path.is_file() else 0.0

    detect_mb = library_mb + size("detector_a_int8.onnx") + size("detector_b_int8.onnx")
    perplexity_mb = library_mb + size("distilgpt2_int8.onnx") + size("gpt2_int8.onnx")
    score_mb = library_mb + size("detector_b_int8.onnx") + size("minilm_int8.onnx")

    return {
        "api/detect.py": round(detect_mb, 1),
        "api/perplexity.py": round(perplexity_mb, 1),
        "api/score.py": round(score_mb, 1),
    }


def write_manifest(models_dir: Path, entries: dict) -> None:
    (models_dir / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
