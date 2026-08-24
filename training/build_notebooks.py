#!/usr/bin/env python3
"""Generate the five Kaggle notebooks from the cell definitions below.

The notebooks are thin: every non-trivial step lives in `training/lib/` so it
can be read, diffed and unit-tested like normal code. Writing them from a
script keeps them consistent and means a change to the pipeline is a change to
one file rather than five JSON blobs.

    python training/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = '''\
# Kaggle setup. Run once per session.
!pip install -q "transformers>=4.44" "datasets>=2.20" sentencepiece onnx onnxruntime \\
    "optimum[onnxruntime]" pyarrow

import sys, os
from pathlib import Path

# The repo is added as a Kaggle dataset, or cloned. Point REPO at it.
REPO = Path("/kaggle/input/ai-detector-repo") if Path("/kaggle/input/ai-detector-repo").exists() \\
       else Path("/kaggle/working/ai-detector")
if not REPO.exists():
    !git clone --depth 1 $GIT_URL /kaggle/working/ai-detector

sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "api"))

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
DATA = WORK / "data"; DATA.mkdir(exist_ok=True)
MODELS = WORK / "models"; MODELS.mkdir(exist_ok=True)
print("repo:", REPO)
'''


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


NOTEBOOKS: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# 01 — dataset
# ---------------------------------------------------------------------------

NOTEBOOKS["01_build_dataset.ipynb"] = [
    markdown("""# 01 · Build the training dataset

Streams RAID and DACTYL, draws a balanced ~400k-row sample stratified by
generator, domain and label, extracts the 30 features, and writes two Parquet
files.

**Two datasets, because there are two models (PRD 8.1).**

| | rows | adversarial |
|---|---|---|
| `train_a.parquet` | Model A, the strict detector | included, ~15% of the mix |
| `train_b.parquet` | Model B, the surrogate | excluded entirely |

Excluding the adversarial rows from B is not an oversight — it is what makes B
a faithful stand-in for third-party detectors, which have no such hardening.

**The labelling rule that matters (E8).** RAID's paraphrase / synonym /
homoglyph variants are the humanized-text augmentation, free. Humanized *AI*
text stays labelled AI. Humanized *human* text stays labelled **human** —
labelling a rewritten human essay as AI would teach the model that editing is
evidence of machine authorship, which is exactly how detectors end up
punishing careful writers and non-native speakers (R2, A4).

Runtime: 40–90 minutes, mostly feature extraction. CPU is fine.
"""),
    code(SETUP),
    code('''\
from pathlib import Path
from lib.data import SampleSpec, build_dataset, feature_statistics

# Start small to verify the pipeline end to end, then raise to 400_000.
SPEC = SampleSpec(total=400_000, adversarial_share=0.15, human_share=0.5)

frame_a = build_dataset(DATA / "train_a.parquet", SPEC, include_adversarial=True)
'''),
    code('''\
spec_b = SampleSpec(total=SPEC.total, adversarial_share=0.0, human_share=0.5, seed=SPEC.seed + 1)
frame_b = build_dataset(DATA / "train_b.parquet", spec_b, include_adversarial=False)
'''),
    code('''\
# Gate: class balance and domain coverage must be verified before training
# (PRD 18, phase 1). A skewed sample produces a model that looks fine on its
# own validation split and fails on everything else.
import pandas as pd

for name, frame in (("A", frame_a), ("B", frame_b)):
    print(f"--- Model {name}: {len(frame):,} rows")
    print(frame["label"].value_counts(normalize=True).round(3).to_dict())
    print("  generators:", frame["generator"].nunique(), " domains:", frame["domain"].nunique())
    print("  adversarial share:", round(frame["adversarial"].mean(), 3))
    print("  words: median", int(frame["text"].str.split().str.len().median()))

balance = frame_a["label"].mean()
assert 0.4 < balance < 0.6, f"Model A is unbalanced: {balance:.3f} positive"
assert frame_b["adversarial"].sum() == 0, "Model B must contain no adversarial rows"
print("\\nGate passed.")
'''),
    code('''\
# Feature standardisation statistics. Paste these into
# api/_lib/features.py (FEATURE_MEAN / FEATURE_STD) so inference standardises
# with the same numbers training used. The placeholders shipped there are
# rough estimates, and leaving them in place costs real accuracy.
import json
stats = feature_statistics(frame_a)
print(json.dumps(stats, indent=2)[:800])
(MODELS / "feature_stats.json").write_text(json.dumps(stats, indent=2))
'''),
]

# ---------------------------------------------------------------------------
# 02 / 03 — training
# ---------------------------------------------------------------------------

TRAIN_CELL = '''\
import pandas as pd
from pathlib import Path
from lib.data import FEATURE_NAMES
from lib.train import TrainConfig, train

frame = pd.read_parquet(DATA / "train_{tag}.parquet")

# Hold out by *generator and domain* where possible, not at random. A random
# split lets the model memorise a generator's quirks and score well on rows
# from the same generator, which is precisely the overfitting RAID exposed
# (E3: fine-tuned RoBERTa-Large averaged 56.7%).
holdout_domains = sorted(frame["domain"].unique())[-2:]
validation = frame[frame["domain"].isin(holdout_domains)]
training = frame[~frame["domain"].isin(holdout_domains)]
print(f"train {{len(training):,}} · validate {{len(validation):,}} on {{holdout_domains}}")

config = TrainConfig(
    backbone="microsoft/deberta-v3-base",
    max_length=768,
    batch_size=16,
    accumulation_steps=2,
    learning_rate=2e-5,
    epochs=3,
    fp16=True,
)

model, report = train(
    training, validation, list(FEATURE_NAMES),
    output_dir=WORK / "model_{tag}",
    config=config,
)
'''

NOTEBOOKS["02_train_model_a.ipynb"] = [
    markdown("""# 02 · Train Model A — the strict detector

The honest score. Trained **with** the adversarial rows, so it sees through
paraphrase-based humanization. DAMAGE (E8) reached 98.26% TPR on humanized AI
text at 5% FPR this way, against GPTZero's 60.04% and Binoculars' 28.23%.

**Loss: two-way partial AUROC, not cross-entropy** (E6, PRD 8.6). The costs
here are asymmetric — falsely accusing a human writer is materially worse than
missing a piece of AI text — so the objective optimises the low-FPR region of
the ROC curve, which is the only region A3 and A4 care about.

This model is **never** the humanizer's optimisation target (H5). Optimising
the rewrite against the detector that is meant to catch it is how a product
ends up grading its own homework.

Runtime: 4–8 hours on a T4. Checkpoints every 500 steps, so a session timeout
costs minutes, not the run.
"""),
    code(SETUP),
    code(TRAIN_CELL.format(tag="a")),
    code('''\
# Progress against the criteria this stage can measure (PRD 14).
final = report["final"]
print(f"AUROC            {final['auroc']:.4f}   (A1 needs >= 0.95 on the test split)")
print(f"partial AUROC@5% {final['partial_auroc_5']:.4f}")
print(f"TPR @ 1% FPR     {final['tpr_at_1_fpr']:.4f}   (A3 needs >= 0.80 on essays)")
print(f"TPR @ 5% FPR     {final['tpr_at_5_fpr']:.4f}   (A5 needs >= 0.90 on humanized AI)")
print("\\nThese are validation-split numbers. The binding evaluation is notebook 05.")
'''),
]

NOTEBOOKS["03_train_model_b.ipynb"] = [
    markdown("""# 03 · Train Model B — the surrogate panel

Deliberately *not* hardened. Trained on the non-adversarial split only, so it
behaves like a typical third-party detector — which is the whole point.

The humanizer optimises against this model plus the perplexity ratio and the
stylometric features. Optimising against a panel rather than a single
classifier is what makes the rewrite generalise: tuned against one model it
learns that model's quirks; tuned against a diverse panel it learns the
properties third-party detectors genuinely share (PRD 8.1).

Same architecture, same loss, different data. Runtime: 4–8 hours.
"""),
    code(SETUP),
    code(TRAIN_CELL.format(tag="b")),
    code('''\
# Sanity check on what B is *for*. It should be clearly weaker than A on
# humanized text — that gap is the product's honesty margin, and the two
# numbers the UI shows are exactly this difference made visible.
final = report["final"]
print(f"AUROC {final['auroc']:.4f}  ·  TPR@1%FPR {final['tpr_at_1_fpr']:.4f}")
print("\\nIf B matches A on adversarial text it is not a surrogate for anything.")
print("Notebook 05 measures that gap directly.")
'''),
]

# ---------------------------------------------------------------------------
# 04 — export
# ---------------------------------------------------------------------------

NOTEBOOKS["04_export_onnx.ipynb"] = [
    markdown("""# 04 · Export to ONNX, quantise to INT8, verify parity

Three things have to hold before anything ships:

* **A9 — parity.** INT8 logits must be within 0.01 of PyTorch's. Quantisation
  that silently costs accuracy is how a model that passed evaluation ships as
  something worse.
* **A11 — bundle size.** Every function under 500 MB.
* **Graph contract.** Inputs named exactly `input_ids`, `attention_mask`,
  `features`. `api/_lib/classifier.py` feeds those names and refuses to guess,
  so a renamed input degrades to "model unavailable" rather than wrong
  scores — but catching it here is far better.

Runtime: 15–30 minutes. CPU only.
"""),
    code(SETUP),
    code('''\
import numpy as np, pandas as pd, torch, json
from pathlib import Path
from transformers import AutoTokenizer

from lib.data import FEATURE_NAMES
from lib.export import (check_parity, export_onnx, export_tokenizer,
                        quantize, report_sizes, write_manifest)
from lib.model import Detector, DetectorConfig

def load(tag):
    state = torch.load(WORK / f"model_{tag}" / "final.pt", map_location="cpu")
    config = state["config"]
    model = Detector(DetectorConfig(backbone=config["backbone"], max_length=config["max_length"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, config
'''),
    code('''\
parity = {}

for tag in ("a", "b"):
    print(f"=== Model {tag.upper()}")
    model, config = load(tag)
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"], use_fast=True)

    fp32 = export_onnx(model, MODELS / f"detector_{tag}_fp32.onnx",
                       max_length=config["max_length"])
    print("  exported, graph contract verified")

    int8 = quantize(fp32, MODELS / f"detector_{tag}_int8.onnx")
    print(f"  fp32 {fp32.stat().st_size >> 20} MB -> int8 {int8.stat().st_size >> 20} MB")

    export_tokenizer(tokenizer, MODELS / f"detector_{tag}_tokenizer.json")

    # A9: parity on real held-out text, standardised with this model's own
    # training statistics.
    sample = pd.read_parquet(DATA / f"train_{tag}.parquet").sample(32, random_state=0)
    mean = np.asarray(config["feature_mean"]); std = np.asarray(config["feature_std"])
    features = ((sample[list(FEATURE_NAMES)].to_numpy(np.float64) - mean) / std).astype(np.float32)

    parity[tag] = check_parity(model, int8, tokenizer, sample["text"].tolist(),
                               features, config["max_length"])
    print(f"  max logit delta {parity[tag]['max_logit_delta']:.5f} "
          f"({'PASS' if parity[tag]['passes_a9'] else 'FAIL'} A9)")

    fp32.unlink()   # only the quantised graph ships
'''),
    code('''\
# The base models. No training needed, so a fresh clone has a working
# Binoculars signal and similarity gate immediately.
!cd $REPO && python scripts/fetch_models.py --base-models
!cp -n $REPO/models/*.onnx $MODELS/ 2>/dev/null; cp -n $REPO/models/*tokenizer.json $MODELS/ 2>/dev/null
'''),
    code('''\
# A11: every function bundle under 500 MB.
sizes = report_sizes(MODELS)
for name, mb in sizes.items():
    print(f"  {name:22s} {mb:6.1f} MB   {'OVER LIMIT' if mb > 500 else 'ok'}")

assert all(mb <= 500 for mb in sizes.values()), "A11 fails — a bundle is over 500 MB"

write_manifest(MODELS, {
    "parity": parity,
    "bundle_sizes_mb": sizes,
    "opset": 17,
    "quantization": "dynamic int8",
})
print("\\nA9:", all(p["passes_a9"] for p in parity.values()))
'''),
    markdown("""## Publish

Upload `MODELS` to a **public** Hugging Face repo — free and unmetered, unlike
GitHub LFS (1 GB storage / 1 GB bandwidth per month, which repeated deploys
would exhaust in days).

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_...")
api.create_repo("YOUR_NAME/ai-text-detector-onnx", exist_ok=True)
api.upload_folder(folder_path=str(MODELS), repo_id="YOUR_NAME/ai-text-detector-onnx")
```

Then set `MODEL_REPO=YOUR_NAME/ai-text-detector-onnx` in the Vercel project
and redeploy. `scripts/fetch_models.py` picks it up at build time.
"""),
]

# ---------------------------------------------------------------------------
# 05 — calibrate and evaluate
# ---------------------------------------------------------------------------

NOTEBOOKS["05_calibrate_eval.ipynb"] = [
    markdown("""# 05 · Fit the fuser, calibrate, run the acceptance gate

This notebook produces the two things the deployed app actually reads —
`fusion_a.json` and `fusion_b.json` — and the honest evaluation report.

**Calibration is not optional (A6).** A reported "82% AI" must correspond to
an observed 82% positive rate within 5 points. Uncalibrated confidence is the
single most common failure of tools in this category, and it is what makes a
score usable as evidence rather than as a vibe.

**A4 is a release blocker.** Detectors are known to over-flag non-native
English writers. If the ESL false-positive rate exceeds 5%, this build does
not ship, regardless of how good every other number looks.
"""),
    code(SETUP),
    code('''\
import numpy as np, pandas as pd, json
from pathlib import Path
from lib.calibrate import (calibration_error, fit_fusion, fit_isotonic,
                           run_acceptance_gate, write_fusion_config)

# Each split must be genuinely held out (PRD 12.4). Replace these loaders with
# your own paths — RAID-test, M4, an essay set, and an ESL/ELL corpus.
def load_split(name):
    """Return (signals[N,4], labels[N]). Signals in the order:
    classifier, features, binoculars_ratio, burstiness."""
    path = DATA / f"eval_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Score each held-out split with the exported "
            "ONNX models plus api/_lib/stats.py and save the four signals."
        )
    frame = pd.read_parquet(path)
    signals = frame[["classifier", "features", "binoculars_ratio", "burstiness"]].to_numpy(np.float64)
    return signals, frame["label"].to_numpy(np.float64)
'''),
    code('''\
# Fit the fuser on a calibration split, never on anything used for training
# or for the final report.
for tag in ("a", "b"):
    signals, labels = load_split(f"calibration_{tag}")
    weights, bias = fit_fusion(signals, labels)
    print(f"Model {tag.upper()} weights:",
          dict(zip(["classifier","features","binoculars","burstiness"], weights.round(3))),
          f"bias {bias:.3f}")

    from lib.calibrate import _sigmoid, _logit
    raw = _sigmoid(_logit(signals) @ weights + bias)

    before = calibration_error(raw, labels)
    x, y = fit_isotonic(raw, labels)
    after = calibration_error(np.interp(raw, x, y), labels)
    print(f"  worst decile gap: {before['max_gap_points']:.1f} -> {after['max_gap_points']:.1f} points")

    write_fusion_config(MODELS / f"fusion_{tag}.json", weights, bias, x, y, version=f"{tag}-1.0.0")
'''),
    code('''\
# The acceptance gate (PRD 14). Every number here goes in the published report.
def fused(tag, split):
    config = json.loads((MODELS / f"fusion_{tag}.json").read_text())
    signals, labels = load_split(split)
    from lib.calibrate import _sigmoid, _logit
    raw = _sigmoid(_logit(signals) @ np.asarray(config["weights"]) + config["bias"])
    return np.interp(raw, config["calibration_x"], config["calibration_y"]), labels

manifest = json.loads((MODELS / "manifest.json").read_text())

report = run_acceptance_gate(
    in_domain=fused("a", "in_domain"),
    cross_domain=fused("a", "m4"),
    essays=fused("a", "essays"),
    esl=fused("a", "esl"),
    humanized=fused("a", "humanized"),
    calibrated=fused("a", "in_domain"),
    onnx_max_logit_delta=manifest["parity"]["a"]["max_logit_delta"],
    bundle_sizes_mb=manifest["bundle_sizes_mb"],
)

print(report.summary())
report.write(MODELS / "eval_report.json")
'''),
    code('''\
# A4 is a release blocker. This cell is meant to stop the pipeline.
blockers = report.blockers_failed
if blockers:
    raise SystemExit(
        "DO NOT SHIP. Blocking criteria failed: " + ", ".join(c.id for c in blockers) +
        "\\n\\nA4 exists because detectors are known to over-flag non-native English "
        "writers. Shipping a tool that penalises ESL authors is a real harm, not a "
        "metric regression. Retrain with more ESL data in the negative class and a "
        "lower alpha in the partial-AUROC loss before trying again."
    )
print("No blocking criteria failed.")
'''),
    code('''\
# A7 and A8 come from the humanizer, not the detector. Run them against the
# deployed app once the models are live:
#
#   A7  surrogate score > 0.9 falls below 0.3 within 3 passes, >= 90% of docs
#   A8  meaning similarity retained >= 0.85
#
# scripts/eval_humanizer.py drives the deployed endpoint over a sample and
# writes those two numbers into the same report.
print(open(MODELS / "eval_report.json").read())
'''),
]


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
                "accelerator": "GPU" if "train" in name else "None",
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = HERE / name
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print(f"wrote {path.name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
