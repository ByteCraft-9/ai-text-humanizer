import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CLEARED_THRESHOLD,
  humanize,
  SENTENCE_FLAG_THRESHOLD,
  SIMILARITY_FLOOR,
} from "../lib/humanize";
import type { LLMProvider } from "../lib/providers";
import type { HumanizeEvent } from "../lib/types";

/**
 * The loop talks to two Python endpoints and one LLM. All three are stubbed
 * here so the invariants can be tested without weights or an API key:
 *
 *   H5 — the strict model is never the optimisation target
 *   H4 — a plateau is reported, not papered over
 *   10.2 — a rewrite that drifts from the meaning is rejected
 */

const SENTENCES = [
  "Artificial intelligence plays a crucial role in modern industry.",
  "Furthermore, it enables a wide range of applications across sectors.",
  "I burned the toast again this morning.",
];
const TEXT = SENTENCES.join(" ");

interface StubOptions {
  /** P(AI) per sentence index, for the detect endpoint. */
  sentenceScores?: number[];
  /** Document surrogate score for each successive detect call. */
  surrogateSequence?: number[];
  /** P(AI) assigned to every generated candidate. */
  candidateScore?: number;
  /** Similarity assigned to every generated candidate. */
  candidateSimilarity?: number;
  similarityMode?: "semantic" | "lexical";
}

/** Records every call so tests can assert on what the loop actually asked for. */
const calls = { detect: [] as unknown[], score: [] as unknown[], strictRequested: 0 };

function installFetchStub(options: StubOptions = {}) {
  const {
    sentenceScores = [0.95, 0.9, 0.05],
    surrogateSequence = [0.9, 0.2],
    candidateScore = 0.1,
    candidateSimilarity = 0.95,
    similarityMode = "semantic",
  } = options;

  let detectCall = 0;

  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));

    if (String(url).endsWith("/api/py/detect")) {
      calls.detect.push(body);
      if (body.include_strict !== false) calls.strictRequested += 1;

      const surrogate =
        surrogateSequence[Math.min(detectCall, surrogateSequence.length - 1)];
      detectCall += 1;

      const count = (body.sentences ?? []).length;
      return new Response(
        JSON.stringify({
          strict_score: Math.min(1, surrogate + 0.15),
          surrogate_score: surrogate,
          signals: {
            classifier: surrogate,
            features: surrogate,
            binoculars_ratio: surrogate,
            burstiness: surrogate,
          },
          sentence_scores: Array.from(
            { length: count },
            (_, i) => sentenceScores[i] ?? 0.1,
          ),
          model_version: "test",
        }),
        { status: 200 },
      );
    }

    if (String(url).endsWith("/api/py/score")) {
      calls.score.push(body);
      return new Response(
        JSON.stringify({
          scores: body.candidates.map((group: string[]) => group.map(() => candidateScore)),
          similarity: body.candidates.map((group: string[]) =>
            group.map(() => candidateSimilarity),
          ),
          similarity_mode: similarityMode,
        }),
        { status: 200 },
      );
    }

    throw new Error(`unexpected fetch to ${url}`);
  });
}

function stubProvider(overrides: Partial<LLMProvider> = {}): LLMProvider {
  return {
    id: "cerebras",
    label: "Stub",
    model: "stub-1",
    limits: { tpd: 1_000_000, rpm: 100 },
    configured: () => true,
    complete: async (request) => {
      // Echo back the numbered sentences with four trivial variants each.
      const ids = [...String(request.messages[1].content).matchAll(/^(\d+)\. (.+)$/gm)];
      return {
        text: JSON.stringify({
          rewrites: ids.map(([, id, sentence]) => ({
            id: Number(id),
            candidates: [
              `Rewritten one: ${sentence}`,
              `Rewritten two: ${sentence}`,
              `Rewritten three: ${sentence}`,
              `Rewritten four: ${sentence}`,
            ],
          })),
        }),
        provider: "cerebras" as const,
        model: "stub-1",
        promptTokens: 120,
        completionTokens: 240,
      };
    },
    ...overrides,
  };
}

beforeEach(() => {
  calls.detect = [];
  calls.score = [];
  calls.strictRequested = 0;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("humanize", () => {
  it("rewrites the flagged sentences and clears the threshold", async () => {
    installFetchStub();
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    expect(result.outcome).toBe("cleared");
    expect(result.surrogate_score).toBeLessThan(CLEARED_THRESHOLD);
    expect(result.text).not.toBe(TEXT);
    expect(result.passes.length).toBeGreaterThan(0);
    expect(result.passes[0].applied).toBeGreaterThan(0);
  });

  it("targets only sentences above the flag threshold", async () => {
    installFetchStub({ sentenceScores: [0.95, 0.9, 0.05] });
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    const rewritten = result.passes[0].rewrites.map((r) => r.index);
    // The third sentence scores 0.05 and must not cost any tokens.
    expect(rewritten).not.toContain(2);
    expect(SENTENCE_FLAG_THRESHOLD).toBeGreaterThan(0.05);
  });

  it("never asks the score endpoint about the strict model (H5)", async () => {
    installFetchStub();
    await humanize(TEXT, { providers: [stubProvider()] });

    // Candidate ranking goes to /api/py/score, which has no access to Model A
    // at all. Nothing in the optimisation path may request a strict score.
    for (const body of calls.score) {
      expect(JSON.stringify(body)).not.toMatch(/strict/i);
    }
    expect(calls.score.length).toBeGreaterThan(0);
  });

  it("rejects candidates that drift from the original meaning", async () => {
    installFetchStub({ candidateSimilarity: SIMILARITY_FLOOR - 0.2 });
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    expect(result.passes[0].applied).toBe(0);
    expect(result.passes[0].rewrites.every((r) => r.rejected === "similarity")).toBe(true);
    // Nothing applied means the document is untouched, not silently degraded.
    expect(result.text).toBe(TEXT);
  });

  it("keeps the original when no candidate scores better", async () => {
    installFetchStub({ sentenceScores: [0.6, 0.6, 0.05], candidateScore: 0.99 });
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    expect(result.passes[0].applied).toBe(0);
    expect(result.passes[0].rewrites.every((r) => r.rejected === "no_improvement")).toBe(true);
  });

  it("reports a plateau instead of manufacturing success (H4)", async () => {
    // The score barely moves between passes.
    installFetchStub({ surrogateSequence: [0.9, 0.89, 0.885, 0.88] });
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    expect(result.outcome).toBe("plateaued");
    expect(result.surrogate_score).toBeGreaterThan(CLEARED_THRESHOLD);
    expect(result.message).toMatch(/not improving/i);
  });

  it("uses the lower floor when similarity is only lexical", async () => {
    // 0.6 is below the semantic floor of 0.85 but above the lexical floor.
    installFetchStub({ similarityMode: "lexical", candidateSimilarity: 0.6 });
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    expect(result.similarity_mode).toBe("lexical");
    expect(result.passes[0].applied).toBeGreaterThan(0);
  });

  it("records that a rewrite was attempted when generation returns nothing", async () => {
    installFetchStub();
    const empty = stubProvider({
      complete: async () => ({
        text: "I cannot help with that.",
        provider: "cerebras" as const,
        model: "stub-1",
        promptTokens: 50,
        completionTokens: 8,
      }),
    });

    const result = await humanize(TEXT, { providers: [empty] });
    expect(result.passes[0].rewrites.every((r) => r.rejected === "generation_failed")).toBe(true);
  });

  it("emits events in order for the live UI", async () => {
    installFetchStub();
    const events: HumanizeEvent["type"][] = [];
    await humanize(TEXT, {
      providers: [stubProvider()],
      onEvent: (event) => events.push(event.type),
    });

    expect(events[0]).toBe("status");
    expect(events).toContain("pass_start");
    expect(events).toContain("sentence");
    expect(events).toContain("pass_end");
    expect(events[events.length - 1]).toBe("done");
  });

  it("reports the initial scores alongside the final ones", async () => {
    installFetchStub();
    const result = await humanize(TEXT, { providers: [stubProvider()] });

    expect(result.initial_surrogate_score).toBeCloseTo(0.9);
    expect(result.original_text).toBe(TEXT);
    // Both numbers are always reported — never just the flattering one.
    expect(result.strict_score).toBeGreaterThan(result.surrogate_score);
  });

  it("counts the tokens it spent", async () => {
    installFetchStub();
    const result = await humanize(TEXT, { providers: [stubProvider()] });
    expect(result.tokens_used).toBeGreaterThan(0);
    expect(result.provider_used).toBe("cerebras");
  });
});
