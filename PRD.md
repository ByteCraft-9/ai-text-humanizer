# Product Requirements Document
## AI Text Detector & Humanizer

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | 24 August 2026 |
| **Owner** | Husnain |
| **Status** | Approved for build |
| **Budget** | $0/month, permanently |

---

## 1. Executive summary

A single web application that (a) determines whether a body of text was written by an AI, with a per-sentence breakdown, and (b) rewrites the flagged portions until they no longer read as machine-generated — then re-checks its own work.

The entire system runs on free infrastructure with no credit card: Next.js frontend and Python ML backend deployed as one Vercel Hobby project, models trained on free Kaggle GPU time, and rewriting powered by a pool of three free LLM APIs with optional bring-your-own-key.

The distinguishing design decision is **honesty**. Most tools in this category report a single fabricated "0% AI" score after rewriting, which is meaningless because the tool grades its own homework. This product ships two separately trained models and reports two separate numbers: a strict score from a detector deliberately hardened against humanization, and an estimated third-party score from a surrogate panel that mimics what GPTZero, Turnitin and Originality actually measure. The user sees what the rewrite genuinely achieved.

---

## 2. Problem statement

Writers who use AI assistance need to know two things that no current free tool answers well together:

1. **Would this text be flagged, and where?** Free detectors return a single opaque percentage with no sentence-level evidence and no indication of confidence. Paid ones cost $15–30/month.
2. **If it is flagged, how do I fix it without destroying the meaning?** Existing humanizers apply blind paraphrasing, frequently mangle technical content, and report success against their own weak internal check.

Compounding this, the accuracy claims in this market are largely unverifiable. Independent benchmarking (RAID, 6.2M generations across 11 models, 8 domains and 11 adversarial attacks) found a fine-tuned RoBERTa-Large detector averaging **56.7% accuracy** — near chance — while marketing materials for comparable products claimed above 99%.

---

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal |
|---|---|
| G1 | Detect AI-generated English text at ≥0.95 AUROC in-domain and ≥0.85 balanced accuracy cross-domain |
| G2 | Keep false positives on human writing below 5%, verified specifically on ESL/ELL authors |
| G3 | Show per-sentence evidence, not a single opaque number |
| G4 | Rewrite flagged text in one click while preserving meaning (cosine similarity ≥0.85) |
| G5 | Report two honest scores; never fabricate a 0% result |
| G6 | Run at exactly $0/month with no credit card on file |
| G7 | Deploy as one repository, one project, one domain, one `git push` |

### 3.2 Non-goals

| ID | Non-goal | Rationale |
|---|---|---|
| N1 | Guaranteeing evasion of any specific third-party detector | Impossible to promise; GPTZero, Turnitin and Originality use different models and update continuously |
| N2 | Non-English detection | v1 scope; the training data supports a multilingual v2 |
| N3 | User accounts, billing, or persistence of documents | Adds cost and privacy surface; not needed for the core loop |
| N4 | Real-time collaborative editing | Out of scope |
| N5 | Detecting AI-generated images, code, or audio | Different problem entirely |
| N6 | Acting as an academic-integrity enforcement tool | The product reports probabilities as evidence, never verdicts about a person |

---

## 4. Users and use cases

| Persona | Need | Primary flow |
|---|---|---|
| **Student / researcher** | Confirm an AI-assisted draft reads as their own voice before submission | Paste or upload → detect → review heatmap → humanize → re-check |
| **Content writer / marketer** | Ensure published copy will not be penalised or flagged | Upload `.docx` → detect → selectively rewrite flagged paragraphs → export |
| **Editor / reviewer** | Assess a submission and see *why* it was flagged | Paste → detect → read sentence-level evidence and confidence band |
| **Developer (you)** | Own the model, the data and the deployment end-to-end | Train on Kaggle → export ONNX → push → live |

---

## 5. Evidence base

Every major requirement below traces to a measured finding, not an assumption.

| # | Finding | Source | Requirement it drives |
|---|---|---|---|
| E1 | Netlify Functions support JS/TS/Go only — no Python runtime | Netlify docs | Deploy on Vercel (§7.1) |
| E2 | Vercel Hobby allows **500 MB** per Python function, 300 s duration, 2 GB RAM, and each `.py` file gets its own bundle | Vercel Functions Limits | Two-function split, INT8 model budget (§9) |
| E3 | Naive fine-tuned RoBERTa-Large: **56.7%** avg on RAID; zero-shot Binoculars: **99.6%** on ChatGPT text | RAID (ACL 2024) | Ensemble, never a lone classifier (§8) |
| E4 | DeBERTa-v3-base + feature-attention over 30 linguistic features: **85.97%** balanced accuracy on M4, +7.2 pts over Fast-DetectGPT; readability and vocabulary features contributed most | arXiv 2605.03969 | Backbone choice + feature branch (§8.2, §8.4) |
| E5 | Binoculars with **GPT-2-scale** models still reached **0.978 ROC-AUC** when fused with stylometry via XGBoost | PAN 2025 (CEUR Vol-4038) | Affordable zero-shot signal (§8.5) |
| E6 | PAN CLEF 2026 winners scored 0.975 overall / 0.989–0.997 AUROC using ModernBERT-large + **two-way partial AUROC loss** + MCGrad multicalibration | arXiv 2607.17382 | Loss function and calibration (§8.6, §8.7) |
| E7 | Detector-guided ("adversarial") paraphrasing cut detection by **87.88%** avg at 1% FPR — Fast-DetectGPT fell 98.96% — with GPT-4o quality only dropping 4.75 → 4.50 | arXiv 2506.07001 | Humanizer algorithm (§10) |
| E8 | A detector trained with humanized-text augmentation reached **98.26%** TPR on humanized AI text at 5% FPR, vs GPTZero **60.04%** and Binoculars **28.23%** | DAMAGE, arXiv 2501.03437 | Two-model architecture (§8.1) |
| E9 | Groq free tier for `openai/gpt-oss-120b`: 30 RPM, 1,000 RPD, 8,000 TPM, **200,000 TPD** | Groq docs | Provider pooling mandatory (§11) |
| E10 | Cerebras free tier: **1,000,000 tokens/day**, 5 RPM, no card | Provider comparison | Cerebras as primary (§11) |
| E11 | RAID ships paraphrase, synonym-swap, homoglyph, whitespace and misspelling variants of every generation, MIT licensed | RAID dataset card | Humanized augmentation needs zero API spend (§12.3) |

---

## 6. Scope

### 6.1 In scope for v1

- Paste input up to 5,000 words
- File upload: `.txt`, `.md`, `.pdf`, `.docx`, parsed client-side
- Document-level AI probability with a confidence band
- Per-sentence probability heatmap
- One-click humanization with an iterative re-check loop
- Dual scoring (strict + estimated third-party)
- Before/after diff view
- Copy to clipboard and download as `.txt` / `.docx`
- Free-tier token budget meter
- Optional BYOK

### 6.2 Deferred to v2

- Multilingual detection
- Browser extension
- Public API for third-party integration
- Batch/folder processing
- Style-matching to a user's own past writing samples

---

## 7. System architecture

### 7.1 Deployment topology

One Git repository → one Vercel Hobby project → one domain.

```
                    ┌──────────────────────────────┐
                    │   Vercel Hobby (single app)  │
                    │                              │
  Browser ────────► │  Next.js 15 / React 19       │
   │                │  ├─ UI, client-side parsing  │
   │                │  └─ /app/api/*  (TypeScript) │
   │                │       └─ humanize orchestr.  │
   │                │                              │
   │                │  /api/py/*      (Python)     │
   │                │  ├─ detect.py    ~265 MB     │
   │                │  └─ perplexity.py ~190 MB    │
                    └───────────┬──────────────────┘
                                │  (humanize only)
                                ▼
                 Cerebras → Groq → Gemini  (failover)
```

**Split rationale.** Model inference needs ONNX Runtime and NumPy, which are Python's strength. The humanizer is I/O-bound API orchestration with response streaming, which is TypeScript's strength. Both ship in the same deployment.

**Routing.** Next.js owns `/app/api/*`. Python functions are exposed at `/api/py/*` via a rewrite in `vercel.json`, so the two namespaces never collide.

### 7.2 Repository layout

```
ai-detector/
├── app/
│   ├── page.tsx                    # main workspace
│   ├── layout.tsx
│   └── api/
│       ├── humanize/route.ts       # streaming orchestrator
│       └── budget/route.ts         # remaining free-tier tokens
├── components/
│   ├── Editor.tsx                  # input + highlight overlay
│   ├── ScoreGauge.tsx              # dual score display
│   ├── Heatmap.tsx                 # per-sentence bands
│   ├── DiffView.tsx                # before/after
│   └── BudgetMeter.tsx
├── lib/
│   ├── extract.ts                  # pdf.js + mammoth.js (client-side)
│   ├── chunk.ts                    # sentence-aware chunker
│   ├── providers.ts                # Cerebras → Groq → Gemini interface
│   └── byok.ts                     # localStorage key handling
├── api/                            # Python, separate bundles
│   ├── detect.py
│   ├── perplexity.py
│   └── _lib/
│       ├── features.py             # 30 stylometric features
│       ├── stats.py                # perplexity / cross-perplexity / burstiness
│       └── fuse.py                 # calibrated ensemble
├── models/                         # fetched at build time, not committed
├── training/
│   ├── lib/{data,features,model,losses}.py
│   ├── 01_build_dataset.ipynb
│   ├── 02_train_model_a.ipynb      # strict detector
│   ├── 03_train_model_b.ipynb      # surrogate
│   ├── 04_export_onnx.ipynb
│   └── 05_calibrate_eval.ipynb
├── scripts/fetch_models.py         # build-time model download from HF Hub
├── vercel.json
├── requirements.txt
└── package.json
```

**Model distribution.** ONNX weights are published to a public Hugging Face Hub repo (free, unlimited) and downloaded by `scripts/fetch_models.py` during the Vercel build. They are **not** committed to Git — GitHub's free LFS allowance is 1 GB storage and 1 GB bandwidth per month, which repeated deploys would exhaust.

---

## 8. Detection design

### 8.1 Two models, two purposes

This is the central architectural decision, and it follows directly from E8.

| | **Model A — Strict** | **Model B — Surrogate panel** |
|---|---|---|
| **Purpose** | Report the honest score | Give the humanizer a target that generalizes |
| **Training data** | RAID **including** adversarial variants + DACTYL | RAID **excluding** adversarial variants |
| **Behaviour** | Hardened against paraphrase/synonym humanization | Deliberately behaves like a typical third-party detector |
| **Shown to user as** | "Strict score" | "Estimated third-party score" |
| **Used by humanizer** | No — never optimised against | Yes — the optimisation target |

Optimising the rewrite against a *panel* rather than a single model matters. A humanizer tuned against one classifier learns that classifier's quirks; tuned against a diverse panel (classifier + perplexity ratio + stylometry) it learns the properties third-party detectors genuinely share.

### 8.2 Backbone

**DeBERTa-v3-base** (184 M params: 86 M backbone + 98 M embeddings).

Chosen over alternatives on evidence: E4 found DeBERTa-v3-base the superior backbone for this task, attributing it to ELECTRA-style replaced-token-detection pretraining producing representations "less sensitive to superficial cues that vary under rewriting and cross-domain shift."

**Rejected:** `desklib/ai-text-detector-v1.01` (DeBERTa-v3-large, 0.4 B, MIT, current RAID leader). At INT8 it is ~440 MB, leaving no room for ONNX Runtime inside the 500 MB limit. It would require Vercel's 5 GB Large Functions beta and incur slow cold starts. Retained as an optional high-accuracy mode in v2.

### 8.3 Signal ensemble

Four signals fuse into one calibrated probability.

| Signal | Implementation | Contribution |
|---|---|---|
| Fine-tuned classifier | DeBERTa-v3-base, INT8 ONNX | Primary discriminative power |
| Feature-attention branch | 30 handcrafted features, dynamic per-sample weighting | Cross-domain robustness (+7.2 pts, E4) |
| Binoculars-style ratio | `distilgpt2` perplexity ÷ `gpt2` cross-perplexity | Generalizes to unseen generators |
| Burstiness | Sentence-length and per-token perplexity variance | Cheap, explainable, drives the heatmap |

### 8.4 Feature set

Thirty features selected from an initial pool, prioritised by the importance ranking in E4 (readability highest, then vocabulary):

- **Readability (7):** Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG, Coleman-Liau, ARI, Dale-Chall
- **Vocabulary (7):** type-token ratio, root TTR, hapax legomena ratio, mean word length, long-word ratio, rare-word ratio vs frequency list, lexical density
- **Syntax (6):** mean/σ sentence length, clause depth proxy, punctuation diversity, comma rate, subordinating-conjunction rate
- **Repetition (5):** bigram/trigram repeat rate, sentence-opener diversity, discourse-marker density, parallel-structure rate
- **Coherence (5):** adjacent-sentence embedding similarity mean/σ, pronoun-reference density, topic-drift proxy, transition-word rate

### 8.5 Perplexity function

`api/py/perplexity` hosts two small causal LMs (`distilgpt2` 82 M as observer, `gpt2` 124 M as performer) and returns log-perplexity, cross-perplexity, their Binoculars ratio, and per-sentence perplexity for the heatmap. E5 established that GPT-2-scale models retain most of the signal — 0.978 ROC-AUC when fused with stylometry.

### 8.6 Loss function

**Two-way partial AUROC** rather than cross-entropy, following E6.

The rationale is asymmetric cost: falsely accusing a human writer is materially worse than missing a piece of AI text. Partial AUROC optimises the region of the ROC curve at low false-positive rates, which is exactly the operating point that matters.

### 8.7 Calibration

Isotonic regression fitted on held-out data, shipped as a ~2 KB JSON alongside the model. Requirement: a reported "82% AI" must correspond to an empirically observed 82% positive rate in that bin, ±5 points. Uncalibrated confidence is the most common failure of tools in this category.

### 8.8 Output contract

```jsonc
{
  "strict_score": 0.87,              // Model A
  "surrogate_score": 0.91,           // Model B panel
  "confidence": "high",              // high | medium | low
  "confidence_interval": [0.82, 0.92],
  "signals": {
    "classifier": 0.89,
    "features": 0.81,
    "binoculars_ratio": 0.74,
    "burstiness": 0.68
  },
  "sentences": [
    { "index": 0, "text": "...", "score": 0.94, "start": 0, "end": 118 }
  ],
  "meta": { "words": 1240, "chunks": 2, "model_version": "a-1.0.0", "ms": 3180 }
}
```

---

## 9. Serving and resource budget

### 9.1 Bundle budget

| Function | Contents | Size |
|---|---|---|
| `api/detect.py` | onnxruntime 50 MB + numpy 25 MB + tokenizers 5 MB + DeBERTa-v3-base INT8 184 MB + calibrator | **≈ 265 MB / 500 MB** |
| `api/perplexity.py` | onnxruntime 50 MB + numpy 25 MB + tokenizers 5 MB + distilgpt2 INT8 45 MB + gpt2 INT8 65 MB | **≈ 190 MB / 500 MB** |

Both sit comfortably inside the per-function limit with headroom for a larger model later.

### 9.2 Latency budget

| Operation | Target (p95) |
|---|---|
| Cold start, Python function | ≤ 4 s |
| Detect, 800-token chunk, warm | ≤ 600 ms |
| Detect, 5,000 words (≈9 chunks), warm | ≤ 8 s |
| Humanize, one pass, 1,000 words | ≤ 12 s |
| Full 3-pass loop, 1,000 words | ≤ 40 s |

Vercel Hobby's 300 s ceiling gives ample headroom; fluid compute keeps instances warm between requests.

---

## 10. Humanization design

### 10.1 Algorithm

A sentence-granularity approximation of the detector-guided paraphrasing in E7. True token-level guided decoding is impossible over an HTTP API; sentence-level candidate reranking captures most of the benefit at a fraction of the token cost.

```
1.  Detect → per-sentence scores from the Model B panel
2.  Rank sentences by score; select the worst K (K ≤ 15 per pass)
3.  Batch 5 sentences per LLM call → request N=4 paraphrase candidates each
4.  Score every candidate locally with the panel        ← free, no API cost
5.  Reject candidates with meaning similarity < 0.85
6.  Keep the most-human surviving candidate per sentence, if it beats the original
7.  Deterministic burstiness pass (zero API cost):
      · split/merge sentences to raise length variance
      · insert contractions
      · vary discourse markers
      · break parallel structure
8.  Re-detect the whole document with both models
9.  If surrogate score > threshold and passes < MAX_PASSES (3) → return to 2
10. Report strict score, surrogate score, passes used, similarity retained
```

**Token efficiency.** Only flagged sentences are rewritten and all candidate scoring happens locally, so a pass costs roughly 3,000 tokens rather than the ~13,400 a whole-document rewrite would consume.

### 10.2 Meaning preservation

Every candidate is checked by embedding similarity against its original sentence. Anything below 0.85 cosine is discarded and regenerated. Without this guard, the loop will "humanize" by degrading content — the dominant failure mode of existing tools.

### 10.3 Honesty requirements

| ID | Requirement |
|---|---|
| H1 | The primary action is labelled "Humanize until our detector clears it" — never "make undetectable" |
| H2 | Results display both scores with a plain-language explanation of the difference |
| H3 | A persistent notice states that no tool can guarantee results against every third-party detector, because they use different models and update continuously |
| H4 | If the loop plateaus, the product reports the plateau rather than reducing the threshold to manufacture success |
| H5 | The strict detector is never used as the humanizer's optimisation target |

---

## 11. Provider routing and capacity

### 11.1 Provider pool

| Priority | Provider | Model | Free daily tokens | RPM | Card |
|---|---|---|---|---|---|
| 1 | **Cerebras** | Llama-class | 1,000,000 | 5 | No |
| 2 | **Groq** | `openai/gpt-oss-120b` | 200,000 (8K TPM, 1K RPD) | 30 | No |
| 3 | **Google AI Studio** | Gemini Flash | ~1,500 req/day | 10 | No |

Cerebras leads on daily volume; Groq leads on requests per minute. The router prefers Cerebras for bulk work and fails over to Groq on rate-limit, which balances the two constraints. All three are behind a single `LLMProvider` interface — adding a fourth is a configuration change, not a refactor.

### 11.2 Capacity model

| Document size | Tokens per 3-pass loop | Documents/day (pooled ≈1.2 M) |
|---|---|---|
| 1,000 words | ≈ 9,000 | ≈ 130 |
| 5,000 words | ≈ 45,000 | ≈ 26 |

For contrast, Groq alone (200 K TPD) would allow roughly **5** long documents per day *across all users combined*. Pooling is not an optimisation; it is what makes the product usable.

### 11.3 BYOK

Users may supply their own provider key. It is held in `localStorage`, attached client-side, and never transmitted to or logged by the application server. This removes the ceiling entirely and holds operating cost at exactly zero regardless of traffic.

### 11.4 Budget meter

The UI displays remaining pooled free-tier tokens for the day before the user begins, so no one hits a wall mid-document. When the pool is under 10%, long documents are gated behind a warning offering BYOK.

---

## 12. Data requirements

### 12.1 Datasets

| Dataset | Size | Licence | Role |
|---|---|---|---|
| **RAID** (`liamdugan/raid`) | 6.2 M generations · 11 models · 8 domains · 11 attacks · 16.7 GB | MIT | Core training + adversarial robustness |
| **DACTYL** (`ShantanuT01/DACTYL`) | 655,437 examples | MIT | Modern generators: GPT-4o, Claude 3.5, Gemini, DeepSeek-V3, Llama |
| **HC3** | ~85 K QA pairs | Open | Classic ChatGPT baseline |
| **M4 / MAGE** | Multi-domain | Open | Held-out cross-domain evaluation only |

All licences permit free commercial use.

### 12.2 Sampling

A balanced ~400,000-row training set drawn from RAID-train and DACTYL, stratified by generator model, domain and decoding strategy. Full RAID is 16.7 GB; the non-adversarial RAID-train split is 802 MB and is the practical base, with a targeted adversarial slice layered on top.

### 12.3 Adversarial augmentation

RAID already contains paraphrase, synonym-swap, homoglyph, whitespace and misspelling variants of every generation (E11). These serve directly as the humanized augmentation that DAMAGE constructed manually — **at zero API cost**.

Following E8's labelling scheme: humanized *human* text is labelled human, teaching invariance to the transform rather than treating rewriting as evidence of machine authorship. Adversarial rows are oversampled to roughly 15% of Model A's training mix.

Model B is trained on the non-adversarial split only, which is precisely what makes it a faithful surrogate for third-party detectors.

### 12.4 Evaluation sets

Held out entirely from training:

- RAID-test, with and without adversarial attacks
- M4 (cross-domain generalization)
- An ESL/ELL essay set for the false-positive bias gate (§14, G2)

---

## 13. Training plan

### 13.1 Compute

Kaggle Notebooks: ~30 GPU-hours/week free on T4 ×2 or P100, no credit card. A DeBERTa-v3-base run needs 4–8 hours. Google Colab free tier is the fallback.

### 13.2 Configuration

| Parameter | Value |
|---|---|
| Backbone | `microsoft/deberta-v3-base` |
| Max sequence length | 768 tokens |
| Batch size | 16 (gradient accumulation ×2) |
| Learning rate | 2e-5, linear warmup 6% |
| Epochs | 3 |
| Precision | fp16 |
| Loss | Two-way partial AUROC (E6) |
| Head | Mean pooling + feature-attention fusion + MLP |

Checkpoints are written every 500 steps to Kaggle output so a session timeout never loses progress.

### 13.3 Sequence

1. `01_build_dataset` — stream RAID + DACTYL, sample, extract 30 features, write Parquet
2. `02_train_model_a` — strict detector, adversarial rows included
3. `03_train_model_b` — surrogate, adversarial rows excluded
4. `04_export_onnx` — opset 17, dynamic INT8, verify parity with PyTorch (max logit delta < 0.01)
5. `05_calibrate_eval` — fit isotonic calibrator, run the full evaluation gate, publish the report

---

## 14. Acceptance criteria

The build is complete when every criterion below passes on held-out data.

| ID | Criterion | Threshold |
|---|---|---|
| A1 | AUROC, in-domain (RAID-test + DACTYL-test) | ≥ 0.95 |
| A2 | Balanced accuracy, cross-domain (M4) | ≥ 0.85 |
| A3 | TPR at 1% FPR, academic essays | ≥ 0.80 |
| A4 | **FPR on human ESL/ELL writing** | **≤ 0.05** |
| A5 | TPR on humanized AI text, Model A, at 5% FPR | ≥ 0.90 |
| A6 | Calibration error, per decile | ≤ 5 points |
| A7 | Humanizer: surrogate score >0.9 → <0.3 within 3 passes | ≥ 90% of documents |
| A8 | Meaning similarity retained after humanization | ≥ 0.85 |
| A9 | ONNX INT8 parity with PyTorch, max logit delta | < 0.01 |
| A10 | Detect p95, 5,000 words, warm | ≤ 8 s |
| A11 | Function bundles under Vercel limit | < 500 MB each |
| A12 | Monthly infrastructure cost | $0.00 |

**A4 is a release blocker.** Detectors are known to over-flag non-native English writers. Shipping a tool that penalises ESL authors would be a serious harm, and it is checked explicitly rather than assumed.

---

## 15. Non-functional requirements

### 15.1 Privacy

| ID | Requirement |
|---|---|
| P1 | **Detection is fully local** to the application's own functions — submitted text never reaches a third party during detection |
| P2 | Humanization *does* transmit text to the selected LLM provider; this is disclosed before the first humanize action, not buried in a policy |
| P3 | No document content is stored server-side; processing is in-memory only |
| P4 | BYOK keys live in `localStorage`, are attached client-side, and are never logged |
| P5 | No accounts, no cookies beyond session state, no analytics on document content |

P1 is a genuine competitive advantage: most free detectors upload your text to their servers.

### 15.2 Security

Rate limiting per IP on the humanize endpoint to prevent one user draining the shared pool. Input size validation before parsing. No `eval` of user content. File parsing runs client-side, which keeps untrusted binaries out of the server entirely.

### 15.3 Accessibility

WCAG 2.1 AA. The heatmap must not encode information by colour alone — it carries numeric labels and patterned underlines. Full keyboard navigation. Screen-reader announcements for score changes after a humanize pass.

### 15.4 Reliability

Graceful degradation when providers are exhausted: detection continues to work fully (it is local), and the humanizer explains exactly which limit was hit and when it resets.

---

## 16. UI specification

### 16.1 Primary workspace

A single screen, three regions.

- **Input region** — large textarea with drag-and-drop file zone, live word counter against the 5,000-word limit, and format badges for accepted types.
- **Results region** — dual score gauges (strict / estimated third-party) with confidence band, a signal breakdown, and the sentence heatmap rendered as an overlay on the text itself so evidence sits where the writing is.
- **Action region** — Humanize button, pass counter, budget meter, and after a run, the before/after diff with per-sentence score deltas.

### 16.2 States

| State | Behaviour |
|---|---|
| Empty | Explain the two scores before the user commits any text |
| Parsing | Client-side progress; scanned PDFs with no text layer fail with a clear message, never silent garbage |
| Detecting | Per-chunk progress; partial heatmap streams in |
| Detected | Full results; Humanize enabled only if score warrants it |
| Humanizing | Live pass counter, current score, sentences rewritten so far |
| Plateaued | States plainly that further passes are not improving the score, and why |
| Budget exhausted | Names the limit, the reset time, and offers BYOK |

### 16.3 Chunking behaviour

Sentence-aware splitter at ~800 tokens with one sentence of overlap. Detection aggregates by length-weighted mean plus a maximum-flag so a single heavily-AI paragraph inside a long human document is not averaged away. Humanization proceeds sequentially with exponential backoff on HTTP 429.

---

## 17. Risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Free-tier pool exhausted by traffic | High | Three-provider pooling, BYOK, budget meter, per-IP rate limiting |
| R2 | False positives on ESL/non-native writers | **High** | A4 is a release blocker with a dedicated evaluation set; low-FPR loss function |
| R3 | Function bundle exceeds 500 MB | Medium | Two-function split with headroom; Large Functions (5 GB) as escape hatch |
| R4 | Detector degrades as new LLMs ship | Medium | Versioned models, quarterly retrain on refreshed DACTYL; ensemble includes a zero-shot component that generalizes to unseen generators |
| R5 | Humanizer degrades meaning | Medium | Similarity gate at 0.85; candidates below threshold are rejected outright |
| R6 | Kaggle GPU quota insufficient | Low | 30 h/week vs 4–8 h needed; Colab fallback; checkpoint-and-resume |
| R7 | Provider changes free-tier terms | Medium | `LLMProvider` interface makes substitution a config change |
| R8 | Users treat scores as proof of misconduct | Medium | Product language reports probability and evidence, never verdicts; confidence bands always shown |

---

## 18. Roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| **1. Data** | Dataset pipeline; 400 K balanced rows with features, Parquet on Kaggle | Class balance and domain coverage verified |
| **2. Model A** | Strict detector trained and evaluated | A1–A6 pass |
| **3. Model B** | Surrogate panel trained | Behaves comparably to published third-party detectors |
| **4. Export** | ONNX INT8, calibrator, parity check, published to HF Hub | A9, A11 pass |
| **5. Backend** | `detect.py` + `perplexity.py` live on Vercel | A10 passes |
| **6. Frontend** | Full UI, parsing, chunking, heatmap | Manual UX review |
| **7. Humanizer** | Provider router, loop, diff view, budget meter | A7, A8 pass |
| **8. Report** | Published evaluation report with real numbers | All of §14 green |

---

## 19. Cost

| Item | Cost |
|---|---|
| Vercel Hobby hosting | $0 |
| Cerebras + Groq + Gemini free tiers | $0 |
| Kaggle GPU training | $0 |
| Datasets (RAID, DACTYL, HC3 — MIT/open) | $0 |
| Model hosting (Hugging Face Hub, public) | $0 |
| **Total recurring** | **$0.00/month** |

The only ceiling is the pooled daily token budget, and BYOK removes it.

---

## 20. Open questions

| # | Question | Needed by |
|---|---|---|
| Q1 | Domain name, or accept the `*.vercel.app` subdomain? | Phase 5 |
| Q2 | Publish the trained models publicly on HF Hub, or keep them private? Public is free and unlimited; private repos have quota limits | Phase 4 |
| Q3 | Should the evaluation report be published alongside the app as a credibility signal? | Phase 8 |
| Q4 | Hard cap on documents per IP per day, or rely on the shared budget meter alone? | Phase 7 |

---

## 21. References

1. Dugan et al. — *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*, ACL 2024. arXiv:2405.07940
2. Hans et al. — *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*, ICML 2024. arXiv:2401.12070
3. *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text*. arXiv:2506.07001
4. *DAMAGE: Detecting Adversarially Modified AI Generated Text*. arXiv:2501.03437
5. *Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators*. arXiv:2605.03969
6. *Notebook for the PAN Lab at CLEF 2026* (DACTYL, ModernBERT-large, MCGrad). arXiv:2607.17382
7. *Binoculars, BART, and Adversaries: Multi-Faceted AI Text Detection for PAN 2025*. CEUR-WS Vol-4038
8. Vercel — *Functions Limits* and *Python Runtime* documentation
9. Groq — *Rate Limits* documentation
10. Hugging Face — `liamdugan/raid`, `ShantanuT01/DACTYL`, `desklib/ai-text-detector-v1.01`
