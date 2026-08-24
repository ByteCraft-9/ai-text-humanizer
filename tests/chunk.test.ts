import { describe, expect, it } from "vitest";

import {
  aggregateChunkScores,
  chunkSentences,
  chunkText,
  countWords,
  mergeSentenceScores,
  splitSentences,
} from "../lib/chunk";

describe("splitSentences", () => {
  it("returns offsets that index back into the source exactly", () => {
    const text = "First sentence here. Second one follows! And a third?";
    for (const sentence of splitSentences(text)) {
      expect(text.slice(sentence.start, sentence.end)).toBe(sentence.text);
    }
  });

  it("does not break on abbreviations, initials or decimals", () => {
    const text =
      "Dr. Smith met Prof. J. Chen at 3.14 p.m. to discuss the U.S. results. " +
      "They agreed.";
    expect(splitSentences(text)).toHaveLength(2);
  });

  it("treats a blank line as a hard break even without a terminator", () => {
    // Headings and list items must not be glued onto the next sentence.
    const sentences = splitSentences("Methods\n\nWe ran the experiment twice.");
    expect(sentences).toHaveLength(2);
    expect(sentences[0].text).toBe("Methods");
  });

  it("keeps a closing quote with the sentence it closes", () => {
    const sentences = splitSentences('He said "we are done." Then he left.');
    expect(sentences[0].text).toBe('He said "we are done."');
  });

  it("handles text with no terminator at all", () => {
    expect(splitSentences("no punctuation here")).toHaveLength(1);
  });

  it("returns nothing for whitespace", () => {
    expect(splitSentences("   \n\n  ")).toHaveLength(0);
  });
});

describe("chunkSentences", () => {
  const sentences = splitSentences(
    Array.from({ length: 60 }, (_, i) => `This is sentence number ${i} of the document.`).join(" "),
  );

  it("keeps chunks under the token budget", () => {
    for (const chunk of chunkSentences(sentences, { maxTokens: 120 })) {
      // A chunk may exceed the budget only when a single sentence does.
      expect(chunk.approxTokens).toBeLessThanOrEqual(160);
    }
  });

  it("overlaps by one sentence so context is not lost at the seam", () => {
    const chunks = chunkSentences(sentences, { maxTokens: 120, overlapSentences: 1 });
    expect(chunks.length).toBeGreaterThan(1);
    for (let i = 1; i < chunks.length; i++) {
      const previous = chunks[i - 1].sentenceIndices;
      expect(chunks[i].sentenceIndices[0]).toBe(previous[previous.length - 1]);
    }
  });

  it("covers every sentence", () => {
    const chunks = chunkSentences(sentences, { maxTokens: 120 });
    const covered = new Set(chunks.flatMap((c) => c.sentenceIndices));
    expect(covered.size).toBe(sentences.length);
  });

  it("returns nothing for an empty document", () => {
    expect(chunkText("").chunks).toHaveLength(0);
  });
});

describe("aggregateChunkScores", () => {
  it("does not average away a single heavily flagged chunk", () => {
    // PRD 16.3: one strongly AI passage in a long human document must survive.
    const scores = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.95];
    const weights = scores.map(() => 100);
    const plainMean = scores.reduce((a, b) => a + b, 0) / scores.length;

    const aggregate = aggregateChunkScores(scores, weights);
    expect(aggregate).toBeGreaterThan(plainMean);
    expect(aggregate).toBeGreaterThan(0.3);
  });

  it("weights by length", () => {
    const short = aggregateChunkScores([0.9, 0.1], [10, 1000]);
    const long = aggregateChunkScores([0.9, 0.1], [1000, 10]);
    expect(long).toBeGreaterThan(short);
  });

  it("passes a single chunk through untouched", () => {
    expect(aggregateChunkScores([0.42], [100])).toBe(0.42);
  });
});

describe("mergeSentenceScores", () => {
  it("averages the two readings of an overlapped sentence", () => {
    const merged = mergeSentenceScores(3, [
      { sentenceIndices: [0, 1], scores: [0.2, 0.8] },
      { sentenceIndices: [1, 2], scores: [0.4, 0.6] },
    ]);
    expect(merged[0]).toBeCloseTo(0.2);
    expect(merged[1]).toBeCloseTo(0.6); // mean of 0.8 and 0.4
    expect(merged[2]).toBeCloseTo(0.6);
  });
});

describe("countWords", () => {
  it("counts whitespace-separated tokens", () => {
    expect(countWords("one two  three\nfour")).toBe(4);
    expect(countWords("   ")).toBe(0);
  });
});
