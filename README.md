# AI Text Detector & Humanizer

Detect AI-generated text with per-sentence evidence, then rewrite the flagged
parts while preserving meaning. One repository, one Vercel project, one
domain, **$0/month**.

The distinguishing decision is honesty. Most tools in this category report a
single fabricated "0% AI" after rewriting, which is meaningless because the
tool is grading its own homework. This one ships **two separately trained
models** and reports **two separate numbers**:

| | what it is | who it is for |
|---|---|---|
| **Strict score** | A detector hardened against humanization | The honest answer to "what is this text?" |
| **Estimated third-party score** | A surrogate panel built to behave like GPTZero / Turnitin / Originality | What the rewrite optimises against |

The strict model is **never** the humanizer's optimisation target. If it were,
the loop would just be teaching itself to cheat.

---

## Status

| Phase | State |
|---|---|
| App, chunking, client-side parsing, UI | **Done** |
| Python inference functions, ensemble fusion, calibration plumbing | **Done** |
| Provider router, humanize loop, budget meter, BYOK | **Done** |
| Trained Model A / Model B weights | **Not yet trained** — notebooks ready |
| Published evaluation report | Blocked on training |

Until the models are trained the app runs in **degraded mode**: it scores on
the 30 linguistic features alone, and says so in the UI. That is a real
measurement of a weaker signal, not a placeholder — the ordering is correct
(machine prose > mixed > human academic > human casual), but the separation is
much narrower than the full ensemble will give.

---

## Running it

```bash
npm install
npm run dev
```

Then <http://localhost:3000>. Detection works immediately with no keys and no
models.

To fetch the off-the-shelf models (Binoculars LMs and the similarity encoder,
~135 MB) so those signals come alive:

```bash
python scripts/fetch_models.py --base-models
```

To check what a deployment actually has:

```bash
python scripts/fetch_models.py --check
```

### Keys

Rewriting needs at least one LLM provider. All three free tiers work without a
credit card. Copy `.env.example` to `.env.local` and fill in whichever you
have — the router uses the first that has allowance left.

Users can also supply their own key in the UI, which is held in
`localStorage`, attached client-side, and never written to storage or a log by
the server.

### Tests

```bash
npm test
```

```bash
python tests/test_python.py
```

The Python suite includes a parity check asserting the TypeScript and Python
sentence splitters agree. They must: the browser splits the document to build
the heatmap and the Python function splits it to score sentences, so a
one-boundary disagreement puts every score on the wrong span.

---

## How it works

```
Browser  ──►  Next.js 15 / React 19
              ├─ client-side .pdf/.docx parsing (nothing large crosses the wire)
              ├─ /app/api/humanize    TypeScript · SSE · provider failover
              └─ /app/api/budget
                        │
              /api/py/* Python · ONNX Runtime
              ├─ detect.py       Model A + Model B + 30 features   448 MB
              ├─ perplexity.py   distilgpt2 + gpt2 (Binoculars)    190 MB
              ├─ score.py        Model B + MiniLM + LMs            397 MB
              └─ health.py
                        │  (humanize only)
                        ▼
              Cerebras → Groq → Gemini → your key
```

Model inference needs ONNX Runtime and NumPy; the humanizer is I/O-bound API
orchestration with streaming. Each language does the half it is good at, and
both ship in one deploy.

### Detection

Four signals fuse into one calibrated probability:

| Signal | Source |
|---|---|
| Fine-tuned classifier | DeBERTa-v3-base + feature-attention branch, INT8 ONNX |
| Linguistic features | 30 readability / vocabulary / syntax / repetition / coherence measures |
| Binoculars ratio | `distilgpt2` perplexity ÷ `gpt2` cross-perplexity |
| Burstiness | Variance of per-token surprise |

A missing signal does not fail the request. Both the fusion gain and the bias
shrink to the fraction of weight actually backed by evidence, so a partial
reading stays near the signals that produced it instead of being amplified
into a confident-looking number the deployment has no basis for.

### Humanization

Sentence-granularity adversarial paraphrasing. Token-level detector-guided
decoding is impossible over an HTTP API, so:

1. Rank sentences by the **surrogate** score, take the worst ≤15
2. One batched LLM call → 4 candidates per sentence
3. Score every candidate **locally** — free, no API cost
4. Reject anything below the meaning-similarity floor
5. Keep the most-human survivor, if it beats the original
6. Deterministic burstiness pass — contractions, discourse markers, sentence
   length variance — costing zero tokens
7. Re-detect with both models; loop up to 3 passes

Local candidate scoring is what makes it affordable: a pass costs ~3,000
provider tokens instead of the ~13,400 a whole-document rewrite would burn.

If passes stop improving the score, the product **says so** and stops. It does
not lower the threshold to manufacture a success.

---

## Training your own models

Everything free: Kaggle gives ~30 GPU-hours/week; a DeBERTa-v3-base run needs
4–8 hours.

```
training/
├── 01_build_dataset.ipynb    RAID + DACTYL → 400k balanced rows + features
├── 02_train_model_a.ipynb    strict detector (adversarial rows included)
├── 03_train_model_b.ipynb    surrogate      (adversarial rows excluded)
├── 04_export_onnx.ipynb      ONNX opset 17 → INT8 → parity + size gates
├── 05_calibrate_eval.ipynb   fit the fuser, calibrate, run the acceptance gate
└── lib/                      the actual pipeline — readable, diffable, testable
```

Regenerate the notebooks after editing `build_notebooks.py`:

```bash
python training/build_notebooks.py
```

Publish the exports to a public Hugging Face repo (free and unmetered, unlike
GitHub LFS at 1 GB/month), set `MODEL_REPO` in the Vercel project, redeploy.

**The labelling rule that matters.** RAID ships paraphrase, synonym-swap and
homoglyph variants of every generation — the humanized-text augmentation, at
zero API cost. Humanized *AI* text stays labelled AI. Humanized *human* text
stays labelled **human**. Labelling a rewritten human essay as AI would teach
the model that editing is evidence of machine authorship, which is exactly how
detectors end up punishing careful writers and non-native speakers.

---

## Acceptance criteria

The build is not complete until every criterion in `models/eval_report.json`
passes. Notebook 05 measures A1–A6, A9 and A11; `scripts/eval_humanizer.py`
measures A7 and A8 against a live deployment.

**A4 — false-positive rate on human ESL/ELL writing ≤ 5% — is a release
blocker.** Detectors are known to over-flag non-native English writers.
Shipping a tool that penalises ESL authors is a real harm, not a metric
regression, and notebook 05 refuses to continue if it fails.

---

## Privacy

- **Detection never leaves this app.** Scoring runs in our own functions. No
  third party sees your text, and nothing is stored — processing is in memory
  only. Most free detectors upload your document to score it.
- **Rewriting does leave.** Humanizing sends the flagged sentences to the
  chosen provider. This is disclosed in the UI before the first rewrite, not
  buried in a policy.
- No accounts, no document storage, no analytics on content.

---

## Deploying

```bash
vercel
```

Set `CEREBRAS_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` (any subset), and
`MODEL_REPO` once you have trained models. `vercel.json` scopes each function's
`includeFiles` so no bundle carries weights it does not use.

| Item | Cost |
|---|---|
| Vercel Hobby, Cerebras + Groq + Gemini free tiers, Kaggle GPU, HF Hub, datasets | $0 |

The only ceiling is the pooled daily token budget, and BYOK removes it.

---

## A note on what these numbers mean

Scores are probabilities, not verdicts about a person. A high score means the
writing has statistical properties common in generated text — it is not
evidence that anyone did anything. The confidence band is always shown, and a
low-confidence result should carry very little weight.

No tool can guarantee a result against every third-party detector. GPTZero,
Turnitin and Originality run different models and update continuously.
