/**
 * Sentence-granularity adversarial paraphrasing loop (PRD 10.1).
 *
 * Token-level detector-guided decoding (E7) is impossible over an HTTP API.
 * This is the sentence-level approximation: generate N candidates per flagged
 * sentence in one batched call, score every candidate *locally* against the
 * surrogate panel (free), and keep the most human survivor.
 *
 * Two invariants this file exists to protect:
 *
 *   H5 — the strict model (A) is never the optimisation target. Only the
 *        surrogate panel (B) guides candidate selection. Model A is consulted
 *        purely to report an honest score at the end of each pass.
 *   H4 — when passes stop improving, we report the plateau. We never lower
 *        the threshold to manufacture a success.
 */

import { applyBurstinessPass } from "./burstiness";
import { splitSentences, type Sentence } from "./chunk";
import { detect } from "./detect";
import {
  withFailover,
  type LLMProvider,
  ProviderError,
} from "./providers";
import type {
  ProviderId,
  HumanizeEvent,
  HumanizeOutcome,
  HumanizePass,
  HumanizeResult,
  SentenceRewrite,
} from "./types";

export const MAX_PASSES = 3;
/** Sentences rewritten per pass (PRD 10.1 step 2). */
export const MAX_SENTENCES_PER_PASS = 15;
/** Sentences per LLM call (PRD 10.1 step 3). */
export const SENTENCES_PER_CALL = 5;
/** Candidates requested per sentence (PRD 10.1 step 3). */
export const CANDIDATES_PER_SENTENCE = 4;
/** Meaning-preservation floor on true cosine similarity (PRD 10.2, A8). */
export const SIMILARITY_FLOOR = 0.85;
/**
 * Floor applied when the MiniLM encoder is absent and the panel falls back to
 * content-word Jaccard. Jaccard runs systematically lower than cosine for a
 * good paraphrase, so enforcing 0.85 against it would reject every rewrite and
 * the loop would silently do nothing. The result records which check ran.
 */
export const LEXICAL_SIMILARITY_FLOOR = 0.55;
/** Below this surrogate score the document is considered cleared (A7). */
export const CLEARED_THRESHOLD = 0.3;
/** Only sentences at or above this are worth spending tokens on. */
export const SENTENCE_FLAG_THRESHOLD = 0.5;
/** A pass improving the surrogate score by less than this is a plateau. */
export const PLATEAU_DELTA = 0.03;

export interface HumanizeOptions {
  baseUrl?: string;
  providers: LLMProvider[];
  signal?: AbortSignal;
  maxPasses?: number;
  onEvent?: (event: HumanizeEvent) => void;
  /** Called after each provider call so the caller can meter the pool. */
  onUsage?: (providerId: ProviderId, tokens: number) => void;
}

// ---------------------------------------------------------------------------
// Local scoring panel — the surrogate model plus a similarity check
// ---------------------------------------------------------------------------

interface PanelScores {
  /** P(AI) per candidate, from Model B's panel. */
  scores: number[][];
  /** Similarity of each candidate to its original. */
  similarity: number[][];
  /** Which measure produced `similarity` — they need different floors. */
  similarity_mode: "semantic" | "lexical";
  degraded?: boolean;
}

/**
 * Score every candidate against the surrogate panel in a single round trip.
 * This is the step that makes the loop affordable: it costs zero LLM tokens.
 */
async function scoreCandidates(
  originals: string[],
  candidates: string[][],
  opts: HumanizeOptions,
): Promise<PanelScores> {
  const res = await fetch(`${opts.baseUrl ?? ""}/api/py/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: opts.signal,
    body: JSON.stringify({ originals, candidates }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Candidate scoring failed (HTTP ${res.status}). ${detail.slice(0, 200)}`);
  }

  const data = (await res.json()) as PanelScores;
  if (!Array.isArray(data?.scores) || !Array.isArray(data?.similarity)) {
    throw new Error("Candidate scoring returned a malformed response.");
  }
  return data;
}

// ---------------------------------------------------------------------------
// Paraphrase generation
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You rewrite sentences so they read as natural human writing while preserving meaning exactly.

Rules:
- Preserve every fact, number, name, citation and technical term exactly. Never invent, drop or soften a claim.
- Vary sentence rhythm. Human writing is uneven: some sentences run long, others are short.
- Prefer plain, specific words over abstract register. Avoid "furthermore", "moreover", "delve", "crucial role", "a wide range of", "it is important to note".
- Use contractions where a person naturally would.
- Do not add commentary, hedging, or filler. Do not change the language of the text.
- Keep roughly the same length. Never expand a sentence into a paragraph.

Return ONLY a JSON object of this exact shape, with no prose around it:
{"rewrites":[{"id":1,"candidates":["...","...","...","..."]}]}`;

function buildUserPrompt(batch: { id: number; text: string }[], n: number): string {
  const items = batch
    .map((s) => `${s.id}. ${s.text}`)
    .join("\n");
  return (
    `Rewrite each numbered sentence ${n} different ways. Each way should sound ` +
    `like a different human wrote it — different rhythm, different word choice — ` +
    `while saying exactly the same thing.\n\n${items}`
  );
}

/** Pull the JSON object out of a reply that may be fenced or padded with prose. */
function parseRewriteJson(raw: string): Map<number, string[]> {
  const out = new Map<number, string[]>();

  let text = raw.trim();
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) text = fence[1].trim();

  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end <= start) return out;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text.slice(start, end + 1));
  } catch {
    return out;
  }

  const rewrites = (parsed as { rewrites?: unknown })?.rewrites;
  if (!Array.isArray(rewrites)) return out;

  for (const entry of rewrites) {
    const id = Number((entry as { id?: unknown })?.id);
    const candidates = (entry as { candidates?: unknown })?.candidates;
    if (!Number.isFinite(id) || !Array.isArray(candidates)) continue;

    const cleaned = candidates
      .filter((c): c is string => typeof c === "string")
      .map((c) => c.trim())
      .filter((c) => c.length > 0);
    if (cleaned.length > 0) out.set(id, cleaned);
  }
  return out;
}

interface BestCandidate {
  text: string;
  score: number;
  similarity: number;
}

interface GenerationResult {
  candidates: Map<number, string[]>;
  tokensUsed: number;
  provider: ProviderId | null;
}

async function generateCandidates(
  batch: { id: number; text: string }[],
  opts: HumanizeOptions,
): Promise<GenerationResult> {
  const messages = [
    { role: "system" as const, content: SYSTEM_PROMPT },
    { role: "user" as const, content: buildUserPrompt(batch, CANDIDATES_PER_SENTENCE) },
  ];

  // Generous headroom: N candidates for up to SENTENCES_PER_CALL sentences.
  const inputChars = batch.reduce((n, s) => n + s.text.length, 0);
  const maxTokens = Math.min(4_000, Math.ceil((inputChars / 4) * CANDIDATES_PER_SENTENCE * 1.6) + 256);

  const completion = await withFailover(opts.providers, (p) =>
    p.complete({ messages, maxTokens, temperature: 1.0, json: true, signal: opts.signal }),
  );

  const tokensUsed = completion.promptTokens + completion.completionTokens;
  opts.onUsage?.(completion.provider, tokensUsed);

  return {
    candidates: parseRewriteJson(completion.text),
    tokensUsed,
    provider: completion.provider,
  };
}

// ---------------------------------------------------------------------------
// Rebuilding the document
// ---------------------------------------------------------------------------

/**
 * Splice rewritten sentences back into the document by character offset,
 * working right-to-left so earlier offsets stay valid.
 */
function applyRewrites(
  text: string,
  sentences: Sentence[],
  replacements: Map<number, string>,
): string {
  const ordered = [...replacements.entries()].sort((a, b) => b[0] - a[0]);
  let out = text;
  for (const [index, replacement] of ordered) {
    const sentence = sentences[index];
    if (!sentence) continue;
    out = out.slice(0, sentence.start) + replacement + out.slice(sentence.end);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

export async function humanize(
  originalText: string,
  opts: HumanizeOptions,
): Promise<HumanizeResult> {
  const maxPasses = opts.maxPasses ?? MAX_PASSES;
  const emit = (event: HumanizeEvent) => opts.onEvent?.(event);

  emit({ type: "status", phase: "detecting", detail: "Scoring the original document" });

  const initial = await detect(originalText, { baseUrl: opts.baseUrl, signal: opts.signal });
  let current = originalText;
  let strict = initial.strict_score;
  let surrogate = initial.surrogate_score;

  const passes: HumanizePass[] = [];
  const similarities: number[] = [];
  let tokensUsed = 0;
  let providerUsed: ProviderId | null = null;
  let lexicalSimilarityUsed = false;
  let outcome: HumanizeOutcome = "max_passes";
  let message: string | undefined;

  // The heatmap that drives targeting comes from the surrogate model only.
  let sentenceScores = initial.sentences.map((s) => s.score);

  for (let passNumber = 1; passNumber <= maxPasses; passNumber++) {
    if (surrogate < CLEARED_THRESHOLD) {
      outcome = "cleared";
      break;
    }

    const sentences = splitSentences(current);
    const targets = sentences
      .map((s, i) => ({ sentence: s, score: sentenceScores[i] ?? 0 }))
      .filter((t) => t.score >= SENTENCE_FLAG_THRESHOLD)
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_SENTENCES_PER_PASS);

    if (targets.length === 0) {
      outcome = surrogate < CLEARED_THRESHOLD ? "cleared" : "plateaued";
      message =
        "No individual sentence scores highly enough to rewrite, but the " +
        "document as a whole still reads as machine-written. Editing by hand " +
        "will do more than another pass here.";
      break;
    }

    emit({ type: "pass_start", pass: passNumber, targeted: targets.length });

    const replacements = new Map<number, string>();
    const rewrites: SentenceRewrite[] = [];
    let passTokens = 0;

    for (let i = 0; i < targets.length; i += SENTENCES_PER_CALL) {
      const batch = targets.slice(i, i + SENTENCES_PER_CALL).map((t) => ({
        id: t.sentence.index,
        text: t.sentence.text,
      }));

      let generated: GenerationResult;
      try {
        generated = await generateCandidates(batch, opts);
      } catch (err) {
        if (err instanceof ProviderError && err.kind === "rate_limit") {
          outcome = "budget_exhausted";
          message =
            "Every free provider is rate-limited right now. " +
            "Your own API key removes this ceiling entirely — see settings.";
        } else {
          outcome = "error";
          message = String((err as Error).message);
        }
        break;
      }

      passTokens += generated.tokensUsed;
      providerUsed = generated.provider ?? providerUsed;

      // Score every surviving candidate locally — free, no API cost.
      const withCandidates = batch.filter((b) => (generated.candidates.get(b.id) ?? []).length);
      if (withCandidates.length === 0) {
        for (const b of batch) {
          rewrites.push({
            index: b.id,
            original: b.text,
            rewritten: b.text,
            score_before: sentenceScores[b.id] ?? 0,
            score_after: sentenceScores[b.id] ?? 0,
            similarity: 1,
            rejected: "generation_failed",
          });
        }
        continue;
      }

      const originals = withCandidates.map((b) => b.text);
      const candidateLists = withCandidates.map((b) => generated.candidates.get(b.id)!);
      const panel = await scoreCandidates(originals, candidateLists, opts);
      if (panel.similarity_mode === "lexical") lexicalSimilarityUsed = true;
      const floor =
        panel.similarity_mode === "lexical" ? LEXICAL_SIMILARITY_FLOOR : SIMILARITY_FLOOR;

      withCandidates.forEach((b, bi) => {
        const scoreBefore = sentenceScores[b.id] ?? 0;
        const candidates = candidateLists[bi];
        const scores = panel.scores[bi] ?? [];
        const sims = panel.similarity[bi] ?? [];

        let best: BestCandidate | null = null;
        let anySurvivedSimilarity = false;

        for (let ci = 0; ci < candidates.length; ci++) {
          const similarity = sims[ci] ?? 0;
          // Step 5: reject anything that has drifted from the meaning.
          if (similarity < floor) continue;
          anySurvivedSimilarity = true;

          const score = scores[ci] ?? 1;
          if (best === null || score < best.score) {
            best = { text: candidates[ci], score, similarity };
          }
        }

        // Step 6: keep it only if it actually beats the original.
        if (best && best.score < scoreBefore) {
          replacements.set(b.id, best.text);
          similarities.push(best.similarity);
          const rewrite: SentenceRewrite = {
            index: b.id,
            original: b.text,
            rewritten: best.text,
            score_before: scoreBefore,
            score_after: best.score,
            similarity: best.similarity,
          };
          rewrites.push(rewrite);
          emit({ type: "sentence", rewrite });
          return;
        }

        rewrites.push({
          index: b.id,
          original: b.text,
          rewritten: b.text,
          score_before: scoreBefore,
          score_after: best ? best.score : scoreBefore,
          similarity: best ? best.similarity : 1,
          rejected: anySurvivedSimilarity ? "no_improvement" : "similarity",
        });
      });
    }

    tokensUsed += passTokens;

    if (outcome === "budget_exhausted" || outcome === "error") break;

    // Step 7: deterministic burstiness pass — costs nothing.
    let next = applyRewrites(current, sentences, replacements);
    if (replacements.size > 0) {
      next = applyBurstinessPass(next, { seed: 0x5eed + passNumber }).text;
    }

    // Step 8: re-detect with both models. Model A is read, never optimised.
    emit({ type: "status", phase: "rechecking", detail: `Re-scoring after pass ${passNumber}` });
    const rechecked = await detect(next, { baseUrl: opts.baseUrl, signal: opts.signal });

    const improvement = surrogate - rechecked.surrogate_score;

    current = next;
    strict = rechecked.strict_score;
    surrogate = rechecked.surrogate_score;
    sentenceScores = rechecked.sentences.map((s) => s.score);

    const pass: HumanizePass = {
      pass: passNumber,
      targeted: targets.length,
      applied: replacements.size,
      rewrites,
      strict_score: strict,
      surrogate_score: surrogate,
      tokens_used: passTokens,
    };
    passes.push(pass);
    emit({ type: "pass_end", pass });

    if (surrogate < CLEARED_THRESHOLD) {
      outcome = "cleared";
      break;
    }

    // H4: a pass that barely moved the score is a plateau. Say so.
    if (improvement < PLATEAU_DELTA && passNumber < maxPasses) {
      outcome = "plateaued";
      message =
        `Pass ${passNumber} moved the estimated third-party score by only ` +
        `${(improvement * 100).toFixed(1)} points. Further passes are not ` +
        `improving it, so we stopped rather than burn your token budget. ` +
        `What is left is likely structural — argument order, paragraph shape, ` +
        `and the specifics only you can add.`;
      break;
    }
  }

  if (outcome === "max_passes" && surrogate < CLEARED_THRESHOLD) outcome = "cleared";

  const meanSimilarity =
    similarities.length > 0
      ? similarities.reduce((a, b) => a + b, 0) / similarities.length
      : 1;

  const result: HumanizeResult = {
    text: current,
    original_text: originalText,
    outcome,
    passes,
    strict_score: strict,
    surrogate_score: surrogate,
    initial_strict_score: initial.strict_score,
    initial_surrogate_score: initial.surrogate_score,
    mean_similarity: meanSimilarity,
    tokens_used: tokensUsed,
    provider_used: providerUsed,
    similarity_mode: lexicalSimilarityUsed ? "lexical" : "semantic",
    message,
  };

  emit({ type: "done", result });
  return result;
}
