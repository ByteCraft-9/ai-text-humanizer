# AI Text Detector + Humanizer — Architecture Plan

**Constraint: $0/month, single app, one deploy.**

---

## 1. What the research changed

I ran the numbers before designing. Five findings overturned parts of the original brief.

### 1.1 Netlify is out, Vercel is in — and it's better than expected

Netlify Functions run JS/TS/Go only. Vercel's Python runtime on the **free Hobby plan** is more generous than I assumed:

| Limit | Hobby value | Why it matters |
|---|---|---|
| Bundle size (Python) | **500 MB** uncompressed | Fits a quantized transformer + ONNX Runtime |
| Max duration | **300 s** | Enough for a multi-pass humanize loop |
| Memory | **2 GB / 1 vCPU** | Comfortable for CPU inference |
| Request/response body | 4.5 MB | Fine for text; parse files client-side |
| Large Functions | up to **5 GB** (beta, on by default for new projects) | Escape hatch if we want a bigger model |

Crucially, **each `.py` file in `/api` becomes its own function with its own 500 MB budget.** We can split the detector and the perplexity scorer into separate functions and get 1 GB of total model budget without touching the beta.

### 1.2 Your chosen approach — fine-tuned classifier — is right, but not the way it's usually done

The RAID benchmark (6.2 M generations, 11 models, 8 domains, 11 attacks) found that a naive fine-tuned **RoBERTa-Large averaged 56.7% accuracy** — barely better than a coin flip — while zero-shot **Binoculars hit 99.6% on ChatGPT text**. Fine-tuned classifiers overfit to the generators they were trained on.

But the picture flipped once people trained them properly:

- **Desklib's `ai-text-detector-v1.01`** (DeBERTa-v3-large, 0.4 B params, **MIT licence**) currently leads the RAID leaderboard.
- **DeBERTa-v3-base + a feature-attention branch** over 30 handcrafted linguistic features hit **85.97% balanced accuracy on M4** cross-domain, beating Fast-DetectGPT by 7.2 points. Readability and vocabulary features mattered most.
- PAN CLEF 2026 winners scored **0.975 overall / 0.989–0.997 AUROC** using ModernBERT-large plus calibration.

**Conclusion:** fine-tuned classifier, yes — but it must be (a) DeBERTa-v3 based, (b) trained on multi-generator adversarial data, (c) fused with handcrafted features, and (d) ensembled with a zero-shot perplexity signal for the cases it hasn't seen.

### 1.3 Binoculars works with tiny models

Binoculars normally needs two 7 B models — impossible for us. But a PAN 2025 team ran it with **GPT-2-scale models** because of memory limits and still reached **0.978 ROC-AUC** when combined with stylometric and TF-IDF features via XGBoost. A `distilgpt2` INT8 ONNX export is ~45 MB. This signal is affordable.

### 1.4 Groq's free tier is far tighter than the blogs claim

Official Groq docs for `openai/gpt-oss-120b`, free plan:

| RPM | RPD | TPM | **TPD** |
|---|---|---|---|
| 30 | 1,000 | 8,000 | **200,000** |

200 K tokens/day total. A 5,000-word document is ~6,700 tokens in and ~6,700 out — one humanize pass costs ~13.4 K tokens. **A three-pass loop on one long document burns ~20% of your entire daily budget.** That's roughly 5 full-length documents per day.

Mitigation — pool three free providers behind one router:

| Provider | Free daily tokens | RPM | Card needed |
|---|---|---|---|
| **Cerebras** | **1,000,000** | 5 | No |
| Groq (`gpt-oss-120b`) | 200,000 | 30 | No |
| Google AI Studio (Gemini Flash) | 1,500 req/day | 10 | No |

Cerebras alone is 5× Groq. Plus **BYOK**: let users paste their own key, which removes the ceiling entirely and keeps *your* costs at exactly zero regardless of traffic.

### 1.5 The strongest humanization method is detector-guided — and we can approximate it for free

"Adversarial Paraphrasing" (arXiv 2506.07001) generates candidate tokens with an instruct LLM and, at each step, picks the candidate a detector scores as *most human*. It cuts detection rates by **87.88% on average at 1% FPR** — Fast-DetectGPT dropped 98.96% — while GPT-4o-judged quality only slipped from 4.75 to ~4.50.

We can't do token-level guided decoding over an HTTP API. But we can do the same thing at **sentence granularity**: generate N paraphrase candidates per sentence in one call, score each with our own detector locally (free, no API cost), keep the most human one. That is the core of the humanizer.

### 1.6 The tension you should know about

DAMAGE (arXiv 2501.03437) trained a detector *deliberately invariant to humanizers* — 98.26% TPR on humanized AI text at 5% FPR, versus GPTZero's 60.04% and Binoculars' 28.23%.

This means: **the better I make your detector, the harder it is for your humanizer to beat it.** If I train one model that does both jobs, the loop either never converges or the detector is deliberately weak.

My proposal: **separate the two.**
- The **scoring detector** is trained for honest accuracy, including humanizer-robustness.
- The **humanizer optimizes against a surrogate panel** — a lighter classifier + the perplexity/burstiness signal + stylometry — which is what third-party detectors actually look at. This makes the output generalize to GPTZero/Turnitin/Originality rather than just gaming one model.

The UI then shows two honest numbers: "our strict score" and "estimated third-party score". No fake 0%.

---

## 2. Architecture

One repo, one Vercel project, one domain, one `git push`.

```
ai-detector/
├── app/                          # Next.js 15 App Router (React 19)
│   ├── page.tsx                  # main UI
│   ├── layout.tsx
│   └── api/
│       ├── humanize/route.ts     # TS: LLM router, streaming, rate-limit pooling
│       └── extract/route.ts      # TS: server-side file fallback
├── components/
│   ├── Editor.tsx                # input, per-sentence highlight overlay
│   ├── ScoreGauge.tsx            # dual score display
│   ├── Heatmap.tsx               # sentence-level AI probability
│   └── DiffView.tsx              # before/after
├── lib/
│   ├── extract.ts                # pdf.js + mammoth.js, CLIENT-side
│   ├── chunk.ts                  # sentence-aware chunking + token budget
│   └── providers.ts              # Cerebras → Groq → Gemini failover
├── api/                          # ← Python, separate 500MB bundles each
│   ├── detect.py                 # DeBERTa-v3-base INT8 ONNX + features
│   ├── perplexity.py             # distilgpt2 INT8 ONNX, Binoculars-style
│   └── _lib/
│       ├── features.py           # 30 stylometric/readability features
│       ├── stats.py              # perplexity, cross-perplexity, burstiness
│       └── fuse.py               # calibrated ensemble → single probability
├── models/                       # quantized ONNX weights, committed via Git LFS
├── training/                     # Kaggle/Colab notebooks (not deployed)
│   ├── 01_build_dataset.ipynb
│   ├── 02_train_deberta.ipynb
│   ├── 03_export_onnx_int8.ipynb
│   └── 04_calibrate.ipynb
├── vercel.json                   # rewrites + excludeFiles
├── requirements.txt
└── package.json
```

**Routing:** Next.js owns `/app/api/*`. Python owns `/api/py/*` via a rewrite in `vercel.json`, so the two never collide.

**Why Python holds the models and TypeScript holds the LLM calls:** model inference needs ONNX Runtime and NumPy (Python's strength); the humanizer is I/O-bound API orchestration with streaming (TypeScript's strength). Both ship in the same deploy.

---

## 3. The detector

Four signals, fused into one calibrated probability.

| Signal | Source | Weight role |
|---|---|---|
| **Fine-tuned classifier** | DeBERTa-v3-base, INT8 ONNX (~184 MB) | Primary — highest single-signal accuracy |
| **Feature-attention branch** | 30 readability/vocabulary/coherence/repetition features | Cross-domain robustness (+7.2 pts in the literature) |
| **Binoculars-style ratio** | `distilgpt2` perplexity ÷ cross-perplexity | Generalizes to unseen generators |
| **Burstiness** | Sentence-length and perplexity variance | Cheap, explainable, drives the heatmap |

Fusion is a small logistic regression trained on held-out data, then **calibrated** (isotonic or MCGrad-style multicalibration, as the PAN 2026 runner-up used) so that "82% AI" actually means 82%.

**Output:** a document-level probability, a per-sentence heatmap, and a confidence band. Sentences are the unit the humanizer then targets.

**Why not the 0.4 B Desklib model?** INT8 it's ~440 MB, which leaves no room for ONNX Runtime in a 500 MB bundle. We'd need the 5 GB beta and eat slow cold starts. DeBERTa-v3-**base** trained on the right data gets close enough at a fraction of the cost. I'll ship Desklib as an optional "high accuracy (slower)" mode behind the Large Functions flag if you want it later.

---

## 4. Training the model

All free. Kaggle gives ~30 h/week of T4/P100; a DeBERTa-v3-base run needs 4–8 h.

**Datasets — all MIT licensed, all free:**

| Dataset | Size | What it adds |
|---|---|---|
| **RAID** (`liamdugan/raid`) | 6.2 M gens, 11 models, 8 domains, 11 attacks | Adversarial robustness — the important one |
| **DACTYL** (`ShantanuT01/DACTYL`) | 655 K examples | Modern generators: GPT-4o, Claude 3.5, Gemini, DeepSeek-V3, Llama |
| **HC3** | ~85 K QA pairs | Classic ChatGPT baseline |
| **M4 / MAGE** | multi-domain, multilingual | Cross-domain validation |

**Pipeline:**
1. Sample a balanced ~400 K-row training set across RAID + DACTYL (full RAID is 16.7 GB — we use RAID-train without adversarial, plus a targeted adversarial slice).
2. **Augment with humanized text**, per DAMAGE: run a slice of both human and AI text through our own humanizer, label the humanized *human* text as human and humanized *AI* text as AI, oversample ~18×. This is what buys resistance to Undetectable.ai / StealthGPT-class tools.
3. Train DeBERTa-v3-base with the feature-attention branch. Optimize **two-way partial AUROC** rather than plain cross-entropy — the PAN 2026 runner-up's trick, and it directly targets low-FPR performance, which is what matters (false-accusing a human writer is the costly error).
4. Export → ONNX opset 17 → INT8 dynamic quantization via `optimum`.
5. Calibrate on held-out data. Ship the calibrator as a 2 KB JSON.

**Expected:** ~0.95+ AUROC in-domain, ~0.85 balanced accuracy cross-domain. I will publish the real numbers in an eval report rather than claiming a marketing figure.

---

## 5. The humanizer

Sentence-level adversarial paraphrasing, looped.

```
1. Detect → per-sentence AI probabilities
2. Rank sentences by AI score; take the worst K
3. One LLM call → N=4 paraphrase candidates for each of those sentences
4. Score all candidates locally with the surrogate panel   ← free, no API cost
5. Keep the most-human candidate per sentence, if it beats the original
6. Apply deterministic burstiness pass (free, no API):
     - vary sentence length (split/merge)
     - insert contractions
     - vary discourse markers
     - break parallel structure
7. Re-detect the whole document
8. If still flagged and passes < MAX_PASSES → back to 2
9. Report: strict score, estimated third-party score, passes used, meaning-preservation check
```

**Why this is token-efficient:** only flagged sentences get rewritten, not the whole document, and candidate scoring is local. A typical pass costs ~2 K tokens instead of ~13 K.

**Meaning preservation** is checked by embedding similarity between original and rewritten sentence; anything below threshold is rejected and regenerated. This stops the loop from "humanizing" by destroying content.

**Honesty in the UI:** the button says *"Humanize until our detector clears it"*, and the result panel states plainly that no tool can guarantee results against every third-party detector, since GPTZero, Turnitin and Originality all use different models and update continuously. You get a real number, not a fake 0%.

---

## 6. Input handling

- **Paste**: up to 5,000 words, live counter.
- **Files**: `.txt`, `.md`, `.pdf`, `.docx` — all parsed **in the browser** (`pdf.js`, `mammoth.js`). Nothing large crosses the 4.5 MB body limit, and it costs zero compute. Scanned PDFs with no text layer get a clear error rather than silent garbage.
- **Chunking**: sentence-aware splitter, ~800-token chunks with 1-sentence overlap. Detection runs per chunk and aggregates by length-weighted mean plus a max-flag. Humanization runs sequentially with exponential backoff on 429s.
- **Live budget meter**: the UI shows remaining free-tier tokens for the day so you never hit a wall mid-document.

---

## 7. Build order

| Phase | What ships | Working result |
|---|---|---|
| **1** | Next.js UI, file parsing, chunking, Python function skeleton with an off-the-shelf detector | Deployed, detecting, end-to-end |
| **2** | Perplexity/burstiness function, feature extraction, ensemble fusion + calibration | Real accuracy, sentence heatmap |
| **3** | Provider router (Cerebras/Groq/Gemini + BYOK), humanizer loop, diff view | Full detect → humanize → recheck |
| **4** | Kaggle notebooks, train on RAID+DACTYL with humanized augmentation, ONNX INT8, swap in | Your own model, your own numbers |
| **5** | Eval report against held-out RAID adversarial splits | Honest published accuracy |

Phase 1 is deployable on day one. Phase 4 is where it becomes *yours*.

---

## 8. Running cost

| Item | Cost |
|---|---|
| Vercel Hobby | $0 |
| Cerebras + Groq + Gemini free tiers | $0 |
| Kaggle GPU training | $0 |
| Datasets (RAID, DACTYL, HC3 — all MIT) | $0 |
| Model hosting (bundled in the function) | $0 |
| **Total** | **$0** |

The only ceiling is the pooled free-tier token budget, and BYOK removes it.
