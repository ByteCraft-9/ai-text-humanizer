#!/usr/bin/env python3
"""Generate the Kaggle notebooks from the cell definitions below.

Emits six files from one source of truth:

  * `01_build_dataset.ipynb` … `05_calibrate_eval.ipynb` — one per stage, for
    running the pipeline as separate Kaggle notebooks.
  * `kaggle_train_all.ipynb` — every stage in one notebook, setup cell
    deduplicated. This is the one to upload if you want a single file: Kaggle
    keeps `/kaggle/working` between sessions when Persistence is enabled, so
    one notebook can carry the whole run without mounting one notebook's
    output as another's input.

The notebooks are thin: every non-trivial step lives in `training/lib/` so it
can be read, diffed and unit-tested like normal code. Writing them from a
script keeps them consistent and means a change to the pipeline is a change to
one file rather than six JSON blobs.

    python training/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Public repo, so the clone needs no token. Override in the cell if you fork.
DEFAULT_GIT_URL = "https://github.com/ByteCraft-9/ai-text-humanizer.git"

SETUP = f'''\
# Setup. Run once per session — everything below depends on it.
!pip install -q "transformers>=4.44" "datasets>=2.20" sentencepiece onnx onnxruntime \\
    "optimum[onnxruntime]" pyarrow

import sys, os
from pathlib import Path

# Change this if you forked the repo. Public repo => no token needed.
GIT_URL = "{DEFAULT_GIT_URL}"

# Either attached as a Kaggle Dataset named `ai-detector-repo`, or cloned.
REPO = Path("/kaggle/input/ai-detector-repo") if Path("/kaggle/input/ai-detector-repo").exists() \\
       else Path("/kaggle/working/ai-detector")
if not REPO.exists():
    !git clone --depth 1 $GIT_URL /kaggle/working/ai-detector

sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "api"))

# parents=True so this also works off-Kaggle (Colab, a local box) after
# pointing WORK somewhere that exists.
WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
DATA = WORK / "data"; DATA.mkdir(parents=True, exist_ok=True)
MODELS = WORK / "models"; MODELS.mkdir(parents=True, exist_ok=True)

print("repo:", REPO)
print("work:", WORK)
assert (REPO / "training" / "lib").is_dir(), "repo not found — check GIT_URL"
'''

RUN_CONFIG = '''\
# ---------------------------------------------------------------------------
# Run configuration. Read this cell before starting anything else.
# ---------------------------------------------------------------------------
#
# Kaggle gives 30 GPU-hours a week. A full run is 8-16 of them, so you cannot
# afford to discover a bug at hour six. Leave SMOKE_TEST = True for the first
# pass: it runs the entire pipeline end to end in well under an hour on a
# tiny sample. If stage 4 completes, the chain works. Then set it False and
# run for real.

SMOKE_TEST = True

if SMOKE_TEST:
    SAMPLE_ROWS = 5_000     # rows per dataset
    EPOCHS = 1
else:
    SAMPLE_ROWS = 400_000   # PRD 12.2
    EPOCHS = 3

print(f"{'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'}: "
      f"{SAMPLE_ROWS:,} rows/dataset, {EPOCHS} epoch(s)")
if not SMOKE_TEST:
    print("Expect ~1-2 h for stage 1, then 4-8 h per model. Use "
          "Save Version -> Save & Run All so a browser disconnect cannot kill it.")
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


# ---------------------------------------------------------------------------
# Stage 1 — dataset
# ---------------------------------------------------------------------------

STAGE1_INTRO = """# 01 · Build the training dataset

Streams RAID and DACTYL, draws a balanced sample stratified by generator,
domain and label, extracts the 30 features, and writes two Parquet files.

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

Both datasets stream, so RAID's 16.7 GB never touches Kaggle's 20 GB disk.

Runtime: minutes on a smoke test, 1–2 hours at 400k rows. **CPU only — turn
the accelerator off for this stage to save GPU quota.**
"""

STAGE1_CELLS = [
    code('''\
from pathlib import Path
from lib.data import SampleSpec, build_dataset, feature_statistics

SPEC = SampleSpec(total=SAMPLE_ROWS, adversarial_share=0.15, human_share=0.5)
frame_a = build_dataset(DATA / "train_a.parquet", SPEC, include_adversarial=True)
'''),
    code('''\
spec_b = SampleSpec(total=SAMPLE_ROWS, adversarial_share=0.0, human_share=0.5,
                    seed=SPEC.seed + 1)
frame_b = build_dataset(DATA / "train_b.parquet", spec_b, include_adversarial=False)
'''),
    code('''\
# Gate: class balance and domain coverage must be verified before training
# (PRD 18, phase 1). A skewed sample produces a model that looks fine on its
# own validation split and fails on everything else.
#
# The sampler already balances at selection time, but human text is the
# scarce class by an order of magnitude and the exact mix depends on what the
# stream happened to yield. This is the safety net: it downsamples the
# majority class within each adversarial group, so both the label balance and
# the adversarial share hold. A no-op when the frames are already balanced.
import pandas as pd

def rebalance(frame, name, seed=0):
    parts = []
    for is_adv, group in frame.groupby("adversarial", sort=False):
        humans = group[group["label"] == 0]
        ais = group[group["label"] == 1]
        n = min(len(humans), len(ais))
        tag = "adversarial" if is_adv else "clean"
        if n == 0:
            print(f"  [{name}/{tag}] one class is empty — kept unchanged")
            parts.append(group)
            continue
        if len(humans) != len(ais):
            print(f"  [{name}/{tag}] {len(humans):,} human / {len(ais):,} AI -> {n:,} each")
        parts.append(pd.concat([humans.sample(n=n, random_state=seed),
                                ais.sample(n=n, random_state=seed)]))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(out) != len(frame):
        print(f"  [{name}] {len(frame):,} @ {frame['label'].mean():.3f} -> "
              f"{len(out):,} @ {out['label'].mean():.3f}")
    return out

frame_a = rebalance(frame_a, "A")
frame_b = rebalance(frame_b, "B")

# Stages 2 and 3 read the Parquet files, not these variables — write them back.
frame_a.to_parquet(DATA / "train_a.parquet", index=False)
frame_b.to_parquet(DATA / "train_b.parquet", index=False)

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
# Feature standardisation statistics.
#
# Copy these into api/_lib/features.py (FEATURE_MEAN / FEATURE_STD) and commit
# before deploying. The values shipped there are rough placeholders; leaving
# them means inference standardises with different numbers than training used,
# which costs real accuracy. Skip this on a smoke test.
import json
stats = feature_statistics(frame_a)
(MODELS / "feature_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2)[:900])
'''),
]

# ---------------------------------------------------------------------------
# Stages 2 and 3 — training
# ---------------------------------------------------------------------------

TRAIN_CELL = '''\
import pandas as pd
from lib.data import FEATURE_NAMES
from lib.train import TrainConfig, train

frame___TAG__ = pd.read_parquet(DATA / "train___TAG__.parquet")

# Hold out by *domain*, not at random. A random split lets the model memorise
# a generator's quirks and score well on rows from the same generator, which
# is precisely the overfitting RAID exposed (E3: fine-tuned RoBERTa-Large
# averaged 56.7%).
holdout___TAG__ = sorted(frame___TAG__["domain"].unique())[-2:]
validation___TAG__ = frame___TAG__[frame___TAG__["domain"].isin(holdout___TAG__)]
training___TAG__ = frame___TAG__[~frame___TAG__["domain"].isin(holdout___TAG__)]
print(f"train {len(training___TAG__):,} · validate {len(validation___TAG__):,} "
      f"on {holdout___TAG__}")

config___TAG__ = TrainConfig(
    backbone="microsoft/deberta-v3-base",
    max_length=768,
    batch_size=16,
    accumulation_steps=2,
    learning_rate=2e-5,
    epochs=EPOCHS,
    fp16=True,
)

# Checkpoints land in WORK/model___TAG__ every 500 steps and resume
# automatically. If the session dies, just re-run this cell.
model___TAG__, report___TAG__ = train(
    training___TAG__, validation___TAG__, list(FEATURE_NAMES),
    output_dir=WORK / "model___TAG__",
    config=config___TAG__,
)
'''

STAGE2_INTRO = """# 02 · Train Model A — the strict detector

The honest score. Trained **with** the adversarial rows, so it sees through
paraphrase-based humanization. DAMAGE (E8) reached 98.26% TPR on humanized AI
text at 5% FPR this way, against GPTZero's 60.04% and Binoculars' 28.23%.

**Loss: two-way partial AUROC, not cross-entropy** (E6, PRD 8.6). The costs
here are asymmetric — falsely accusing a human writer is materially worse than
missing a piece of AI text — so the objective optimises the low-FPR region of
the ROC curve, which is the only region A3 and A4 care about.

This model is **never** the humanizer's optimisation target (H5). Optimising
the rewrite against the detector meant to catch it is how a product ends up
grading its own homework.

**Needs GPU.** 4–8 hours on a T4 at full size. Checkpoints every 500 steps.
"""

STAGE2_CELLS = [
    code(TRAIN_CELL.replace("__TAG__", "a")),
    code('''\
# Progress against the criteria this stage can measure (PRD 14).
final = report_a["final"]
print(f"AUROC            {final['auroc']:.4f}   (A1 needs >= 0.95 on the test split)")
print(f"partial AUROC@5% {final['partial_auroc_5']:.4f}")
print(f"TPR @ 1% FPR     {final['tpr_at_1_fpr']:.4f}   (A3 needs >= 0.80 on essays)")
print(f"TPR @ 5% FPR     {final['tpr_at_5_fpr']:.4f}   (A5 needs >= 0.90 on humanized AI)")
print("\\nValidation-split numbers. The binding evaluation is stage 5.")
'''),
]

STAGE3_INTRO = """# 03 · Train Model B — the surrogate panel

Deliberately *not* hardened. Trained on the non-adversarial split only, so it
behaves like a typical third-party detector — which is the whole point.

The humanizer optimises against this model plus the perplexity ratio and the
stylometric features. Optimising against a panel rather than a single
classifier is what makes the rewrite generalise: tuned against one model it
learns that model's quirks; tuned against a diverse panel it learns the
properties third-party detectors genuinely share (PRD 8.1).

Same architecture, same loss, different data. **Needs GPU**, 4–8 hours.
"""

STAGE3_CELLS = [
    code(TRAIN_CELL.replace("__TAG__", "b")),
    code('''\
# Sanity check on what B is *for*. It should be clearly weaker than A on
# humanized text — that gap is the product's honesty margin, and the two
# numbers the UI shows are exactly this difference made visible.
final_b = report_b["final"]
print(f"Model B  AUROC {final_b['auroc']:.4f}  ·  TPR@1%FPR {final_b['tpr_at_1_fpr']:.4f}")
try:
    print(f"Model A  AUROC {report_a['final']['auroc']:.4f}  ·  "
          f"TPR@1%FPR {report_a['final']['tpr_at_1_fpr']:.4f}")
except NameError:
    print("(Model A not in this session — compare against stage 2's output.)")
print("\\nIf B matches A on adversarial text it is not a surrogate for anything.")
print("Stage 5 measures that gap directly.")
'''),
]

# ---------------------------------------------------------------------------
# Stage 4 — export
# ---------------------------------------------------------------------------

STAGE4_INTRO = """# 04 · Export to ONNX, quantise to INT8, verify parity

Three things have to hold before anything ships:

* **A9 — parity.** INT8 logits must be within 0.01 of PyTorch's. Quantisation
  that silently costs accuracy is how a model that passed evaluation ships as
  something worse.
* **A11 — bundle size.** Every function under 500 MB.
* **Graph contract.** Inputs named exactly `input_ids`, `attention_mask`,
  `features`. `api/_lib/classifier.py` feeds those names and refuses to guess,
  so a renamed input degrades to "model unavailable" rather than to wrong
  scores — but catching it here is far better.

Needs stages 2 and 3 to have written `final.pt`. **CPU is fine.** 15–30 min.
Internet must be on for the base-model download.
"""

STAGE4_CELLS = [
    code('''\
import numpy as np, pandas as pd, torch, json
from transformers import AutoTokenizer

from lib.data import FEATURE_NAMES
from lib.export import (check_parity, export_onnx, export_tokenizer,
                        quantize, report_sizes, write_manifest)
from lib.model import Detector, DetectorConfig

def load(tag):
    path = WORK / f"model_{tag}" / "final.pt"
    assert path.is_file(), f"{path} missing — run stage {'2' if tag == 'a' else '3'} first"
    state = torch.load(path, map_location="cpu")
    cfg = state["config"]
    model = Detector(DetectorConfig(backbone=cfg["backbone"], max_length=cfg["max_length"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg
'''),
    code('''\
parity = {}

for tag in ("a", "b"):
    print(f"=== Model {tag.upper()}")
    model, cfg = load(tag)
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)

    fp32 = export_onnx(model, MODELS / f"detector_{tag}_fp32.onnx",
                       max_length=cfg["max_length"])
    print("  exported, graph contract verified")

    int8 = quantize(fp32, MODELS / f"detector_{tag}_int8.onnx")
    print(f"  fp32 {fp32.stat().st_size >> 20} MB -> int8 {int8.stat().st_size >> 20} MB")

    export_tokenizer(tokenizer, MODELS / f"detector_{tag}_tokenizer.json")

    # A9: parity on real held-out text, standardised with this model's own
    # training statistics.
    sample = pd.read_parquet(DATA / f"train_{tag}.parquet").sample(32, random_state=0)
    mean = np.asarray(cfg["feature_mean"]); std = np.asarray(cfg["feature_std"])
    features = ((sample[list(FEATURE_NAMES)].to_numpy(np.float64) - mean) / std).astype(np.float32)

    parity[tag] = check_parity(model, int8, tokenizer, sample["text"].tolist(),
                               features, cfg["max_length"])
    print(f"  max logit delta {parity[tag]['max_logit_delta']:.5f} "
          f"({'PASS' if parity[tag]['passes_a9'] else 'FAIL'} A9)")

    fp32.unlink()   # only the quantised graph ships
'''),
    code('''\
# The base models — Binoculars LMs and the similarity encoder. No training
# needed, so these work immediately. Requires Internet to be on.
!cd $REPO && python scripts/fetch_models.py --base-models
!cp -n $REPO/models/*.onnx $MODELS/ 2>/dev/null
!cp -n $REPO/models/*tokenizer.json $MODELS/ 2>/dev/null
!ls -la $MODELS
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
print("\\nA9 parity:", all(p["passes_a9"] for p in parity.values()))
print("Stages 1-4 complete. The models are ready to publish.")
'''),
]

PUBLISH_MD = """## Publish

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
and redeploy. `scripts/fetch_models.py` picks it up at build time and the
degraded-mode banner disappears.

Also copy the feature statistics printed at the end of stage 1 into
`api/_lib/features.py` and commit, so inference standardises with the same
numbers training used.
"""

# ---------------------------------------------------------------------------
# Stage 5 — calibrate and evaluate
# ---------------------------------------------------------------------------

STAGE5_INTRO = """# 05 · Fit the fuser, calibrate, run the acceptance gate

This stage produces the two artefacts the deployed app reads — `fusion_a.json`
and `fusion_b.json` — and the honest evaluation report.

**Calibration is not optional (A6).** A reported "82% AI" must correspond to
an observed 82% positive rate within 5 points. Uncalibrated confidence is the
single most common failure of tools in this category, and it is what makes a
score usable as evidence rather than as a vibe.

**A4 is a release blocker.** Detectors are known to over-flag non-native
English writers. If the ESL false-positive rate exceeds 5%, this build does
not ship, regardless of how good every other number looks.

---

> ### This stage needs evaluation splits that do not exist yet
>
> It requires seven held-out `eval_*.parquet` files — RAID-test, M4, an
> academic essay set, an ESL/ELL corpus, and calibration splits — each already
> scored into the four signal columns (`classifier`, `features`,
> `binoculars_ratio`, `burstiness`).
>
> Building them is real work: score each held-out corpus with the ONNX models
> exported in stage 4 plus `api/_lib/stats.py`, and save the four columns
> alongside the label.
>
> **The cells below detect the missing files and skip cleanly**, so a
> "Save & Run All" still finishes green with stages 1–4 complete. Until this
> stage runs you have a working trained detector, but no calibration and no
> A4 number — so do not publish accuracy claims yet.
"""

STAGE5_CELLS = [
    code('''\
import numpy as np, pandas as pd, json
from lib.calibrate import (_logit, _sigmoid, calibration_error, fit_fusion,
                           fit_isotonic, run_acceptance_gate, write_fusion_config)

REQUIRED_SPLITS = ["calibration_a", "calibration_b", "in_domain", "m4",
                   "essays", "esl", "humanized"]
missing = [n for n in REQUIRED_SPLITS if not (DATA / f"eval_{n}.parquet").is_file()]
EVAL_READY = not missing

if EVAL_READY:
    print("All evaluation splits present. Stage 5 will run.")
else:
    print("Stage 5 SKIPPED — these files are missing from", DATA)
    for name in missing:
        print(f"  eval_{name}.parquet")
    print("\\nStages 1-4 are unaffected. See the note above for how to build these.")

def load_split(name):
    """Return (signals[N,4], labels[N]) in the fuser's signal order."""
    frame = pd.read_parquet(DATA / f"eval_{name}.parquet")
    signals = frame[["classifier", "features", "binoculars_ratio",
                     "burstiness"]].to_numpy(np.float64)
    return signals, frame["label"].to_numpy(np.float64)
'''),
    code('''\
# Fit the fuser on a calibration split — never on anything used for training
# or for the final report.
if EVAL_READY:
    for tag in ("a", "b"):
        signals, labels = load_split(f"calibration_{tag}")
        weights, bias = fit_fusion(signals, labels)
        print(f"Model {tag.upper()} weights:",
              dict(zip(["classifier", "features", "binoculars", "burstiness"],
                       weights.round(3))), f"bias {bias:.3f}")

        raw = _sigmoid(_logit(signals) @ weights + bias)
        before = calibration_error(raw, labels)
        x, y = fit_isotonic(raw, labels)
        after = calibration_error(np.interp(raw, x, y), labels)
        print(f"  worst decile gap: {before['max_gap_points']:.1f} -> "
              f"{after['max_gap_points']:.1f} points")

        write_fusion_config(MODELS / f"fusion_{tag}.json", weights, bias, x, y,
                            version=f"{tag}-1.0.0")
else:
    print("skipped")
'''),
    code('''\
# The acceptance gate (PRD 14). Every number here goes in the published report.
if EVAL_READY:
    def fused(tag, split):
        cfg = json.loads((MODELS / f"fusion_{tag}.json").read_text())
        signals, labels = load_split(split)
        raw = _sigmoid(_logit(signals) @ np.asarray(cfg["weights"]) + cfg["bias"])
        return np.interp(raw, cfg["calibration_x"], cfg["calibration_y"]), labels

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
else:
    print("skipped")
'''),
    code('''\
# A4 is a release blocker. This cell is meant to stop the pipeline.
if EVAL_READY:
    blockers = report.blockers_failed
    if blockers:
        raise SystemExit(
            "DO NOT SHIP. Blocking criteria failed: " +
            ", ".join(c.id for c in blockers) +
            "\\n\\nA4 exists because detectors are known to over-flag non-native "
            "English writers. Shipping a tool that penalises ESL authors is a real "
            "harm, not a metric regression. Retrain with more ESL data in the "
            "negative class and a lower alpha in the partial-AUROC loss."
        )
    print("No blocking criteria failed.")
else:
    print("skipped")
'''),
    code('''\
# A7 and A8 come from the humanizer, not the detector. Run them against the
# deployed app once the models are live:
#
#   A7  surrogate score > 0.9 falls below 0.3 within 3 passes, >= 90% of docs
#   A8  meaning similarity retained >= 0.85
#
#   python scripts/eval_humanizer.py --url https://your-app.vercel.app \\
#       --input eval/ai_documents.jsonl --output models/eval_report.json
print("Pipeline finished.")
'''),
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

MEGA_INTRO = """# AI Text Detector — full training pipeline

Every stage in one notebook. Upload this file to Kaggle and run it top to
bottom; the setup cell appears once and each stage picks up where the last
left off.

## Before you run anything

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | GPU T4 ×2 or P100 | stages 2 and 3 need it |
| **Internet** | **On** | pip, `git clone`, streaming RAID/DACTYL |
| **Persistence** | **Files only** | keeps `/kaggle/working` between sessions — this is what lets one notebook carry the whole run |

Phone-verify your Kaggle account first, or you get neither GPU nor Internet.

## How to run it

**First pass: leave `SMOKE_TEST = True`.** It runs the entire chain on 5,000
rows for one epoch in under an hour. If stage 4 completes, the pipeline works.
You get 30 GPU-hours a week and a full run costs 8–16 of them, so you cannot
afford to find a bug at hour six.

**Then set `SMOKE_TEST = False`** and work through it in three sessions —
Kaggle caps a session at 12 hours:

| Session | Stages | Time | Accelerator |
|---|---|---|---|
| 1 | setup, config, **1** | 1–2 h | **off** — stage 1 is CPU work, save the quota |
| 2 | setup, config, **2** | 4–8 h | on |
| 3 | setup, config, **3**, **4** | 5–9 h | on |

Re-run the setup and config cells at the start of every session; they rebuild
the Python variables. Everything on disk survives via Persistence.

For stages 2 and 3 use **Save Version → Save & Run All (Commit)** so the run
is detached — an interactive session dies after ~20 minutes idle. If a session
is killed mid-training anyway, just re-run the stage cell: the loop
checkpoints every 500 steps and resumes from the newest one.

## What you get

Stages 1–4 produce a trained, exportable, deployable detector. Stage 5 needs
evaluation splits that do not exist yet and will skip itself cleanly — read
its section before relying on any accuracy number.
"""


def build_stage_notebook(title_md: str, cells: list[dict], tail: list[dict] | None = None) -> list[dict]:
    return [markdown(title_md), code(SETUP), code(RUN_CONFIG), *cells, *(tail or [])]


NOTEBOOKS: dict[str, list[dict]] = {
    "01_build_dataset.ipynb": build_stage_notebook(STAGE1_INTRO, STAGE1_CELLS),
    "02_train_model_a.ipynb": build_stage_notebook(STAGE2_INTRO, STAGE2_CELLS),
    "03_train_model_b.ipynb": build_stage_notebook(STAGE3_INTRO, STAGE3_CELLS),
    "04_export_onnx.ipynb": build_stage_notebook(
        STAGE4_INTRO, STAGE4_CELLS, [markdown(PUBLISH_MD)]
    ),
    "05_calibrate_eval.ipynb": build_stage_notebook(STAGE5_INTRO, STAGE5_CELLS),
}

# The combined notebook: setup and config once, then every stage in order.
NOTEBOOKS["kaggle_train_all.ipynb"] = [
    markdown(MEGA_INTRO),
    code(SETUP),
    code(RUN_CONFIG),
    markdown("---\n" + STAGE1_INTRO),
    *STAGE1_CELLS,
    markdown("---\n" + STAGE2_INTRO),
    *STAGE2_CELLS,
    markdown("---\n" + STAGE3_INTRO),
    *STAGE3_CELLS,
    markdown("---\n" + STAGE4_INTRO),
    *STAGE4_CELLS,
    markdown(PUBLISH_MD),
    markdown("---\n" + STAGE5_INTRO),
    *STAGE5_CELLS,
]


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        needs_gpu = "train" in name or name == "kaggle_train_all.ipynb"
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
                "accelerator": "GPU" if needs_gpu else "None",
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = HERE / name
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        code_cells = sum(1 for c in cells if c["cell_type"] == "code")
        print(f"wrote {path.name:28s} {len(cells):2d} cells ({code_cells} code)")


if __name__ == "__main__":
    main()
