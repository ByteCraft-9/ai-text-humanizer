/**
 * Detection orchestration: chunk, score each chunk against the Python
 * inference function, aggregate back to a document result (PRD 16.3).
 *
 * Isomorphic on purpose — the browser calls it for the Detect action, and the
 * humanize route calls it server-side inside the rewrite loop.
 */

import {
  aggregateChunkScores,
  chunkText,
  countWords,
  estimateTokens,
  mergeSentenceScores,
  type Chunk,
  type Sentence,
} from "./chunk";
import type {
  Confidence,
  DetectResult,
  SentenceScore,
  Signals,
} from "./types";

export const MAX_WORDS = 5000;

/** One chunk's reply from api/py/detect. */
interface ChunkResponse {
  strict_score: number;
  surrogate_score: number;
  signals: Signals;
  sentence_scores: number[];
  model_version: string;
  degraded?: boolean;
}

export interface DetectOptions {
  /** Origin for the Python function. Empty string means same-origin. */
  baseUrl?: string;
  /** Skip Model A when only the surrogate is needed (humanizer inner loop). */
  includeStrict?: boolean;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

export class DetectionError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "DetectionError";
  }
}

async function scoreChunk(
  chunk: Chunk,
  sentences: Sentence[],
  opts: DetectOptions,
): Promise<ChunkResponse> {
  const chunkSentences = chunk.sentenceIndices.map((i) => sentences[i].text);

  const res = await fetch(`${opts.baseUrl ?? ""}/api/py/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: opts.signal,
    body: JSON.stringify({
      text: chunk.text,
      sentences: chunkSentences,
      include_strict: opts.includeStrict !== false,
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new DetectionError(
      `Detection failed (HTTP ${res.status}). ${detail.slice(0, 200)}`,
      res.status,
    );
  }

  const data = (await res.json()) as ChunkResponse;
  if (typeof data?.surrogate_score !== "number") {
    throw new DetectionError("Detection returned a malformed response.");
  }
  return data;
}

/**
 * Confidence reflects how much evidence we had and how much the signals
 * agree. Short text and disagreeing signals both lower it — the PRD requires
 * a band, not a bare number (G3, R8).
 */
function assessConfidence(
  words: number,
  signals: Signals,
  degraded: boolean,
): { confidence: Confidence; halfWidth: number } {
  const values = [
    signals.classifier,
    signals.features,
    signals.binoculars_ratio,
    signals.burstiness,
  ];
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const spread = Math.sqrt(
    values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length,
  );

  // Under ~120 words there is simply not enough signal to be confident.
  const lengthPenalty = words >= 300 ? 0 : words >= 120 ? 0.04 : 0.12;
  const halfWidth = Math.min(0.35, 0.03 + spread * 0.8 + lengthPenalty + (degraded ? 0.1 : 0));

  const confidence: Confidence =
    halfWidth <= 0.07 ? "high" : halfWidth <= 0.15 ? "medium" : "low";
  return { confidence, halfWidth };
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

export async function detect(text: string, opts: DetectOptions = {}): Promise<DetectResult> {
  const started = Date.now();
  const words = countWords(text);
  if (words === 0) throw new DetectionError("There is nothing to analyse.");

  const { sentences, chunks } = chunkText(text);

  const responses: ChunkResponse[] = [];
  for (const [i, chunk] of chunks.entries()) {
    responses.push(await scoreChunk(chunk, sentences, opts));
    opts.onProgress?.(i + 1, chunks.length);
  }

  const weights = chunks.map((c) => c.approxTokens);
  const strict = aggregateChunkScores(responses.map((r) => r.strict_score), weights);
  const surrogate = aggregateChunkScores(responses.map((r) => r.surrogate_score), weights);

  const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;
  const signals = (["classifier", "features", "binoculars_ratio", "burstiness"] as const).reduce(
    (acc, key) => {
      acc[key] =
        responses.reduce((sum, r, i) => sum + r.signals[key] * weights[i], 0) / totalWeight;
      return acc;
    },
    {} as Signals,
  );

  const perSentence = mergeSentenceScores(
    sentences.length,
    chunks.map((chunk, i) => ({
      sentenceIndices: chunk.sentenceIndices,
      scores: responses[i].sentence_scores,
    })),
  );

  const sentenceScores: SentenceScore[] = sentences.map((s) => ({
    index: s.index,
    text: s.text,
    score: perSentence[s.index],
    start: s.start,
    end: s.end,
  }));

  const degraded = responses.some((r) => r.degraded);
  const { confidence, halfWidth } = assessConfidence(words, signals, degraded);

  return {
    strict_score: strict,
    surrogate_score: surrogate,
    confidence,
    confidence_interval: [clamp01(strict - halfWidth), clamp01(strict + halfWidth)],
    signals,
    sentences: sentenceScores,
    meta: {
      words,
      chunks: chunks.length,
      model_version: responses[0]?.model_version ?? "unknown",
      ms: Date.now() - started,
      degraded,
    },
  };
}

/** Rough token cost of a full humanize loop, for the budget gate (PRD 11.4). */
export function estimateHumanizeTokens(text: string, passes = 3): number {
  const perPass = Math.min(estimateTokens(text), 4_000);
  return Math.round(perPass * passes * 1.6);
}
