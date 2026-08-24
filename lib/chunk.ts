/**
 * Sentence-aware splitting and chunking (PRD 16.3).
 *
 * Chunks target ~800 tokens with one sentence of overlap, and detection
 * aggregates by length-weighted mean plus a maximum-flag, so a single heavily
 * AI paragraph inside a long human document is not averaged away.
 */

export interface Sentence {
  index: number;
  text: string;
  /** Character offsets into the source document, inclusive/exclusive. */
  start: number;
  end: number;
}

export interface Chunk {
  index: number;
  text: string;
  start: number;
  end: number;
  /** Indices into the sentence array this chunk covers. */
  sentenceIndices: number[];
  /** Sentences carried over from the previous chunk purely as context. */
  overlapCount: number;
  approxTokens: number;
}

/** Titles and abbreviations whose trailing period does not end a sentence. */
const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
  "gen", "col", "lt", "sgt", "capt", "cmdr", "adm", "gov", "sen", "rep",
  "vs", "etc", "eg", "ie", "cf", "al", "approx", "dept", "est", "fig",
  "inc", "ltd", "co", "corp", "univ", "assn", "bros", "no", "vol", "pp",
  "ed", "eds", "trans", "misc", "min", "max", "avg", "std", "ca", "circa",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
  "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "ph.d", "m.d", "b.a", "m.a",
]);

const TERMINATORS = new Set([".", "!", "?", "\u2026"]);
/** Closing quotes/brackets that may follow a terminator before the break. */
const CLOSERS = new Set(['"', "'", "\u201d", "\u2019", ")", "]", "}", "\u00bb"]);

/**
 * Roughly 4 characters per token for English prose. Deliberately an estimate:
 * the real tokenizer lives in the Python function, and over-estimating here
 * only makes chunks slightly smaller, which is safe.
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function countWords(text: string): number {
  const m = text.trim().match(/\S+/g);
  return m ? m.length : 0;
}

function isSentenceBreak(text: string, i: number): boolean {
  const ch = text[i];
  if (!TERMINATORS.has(ch)) return false;

  // Walk past closing quotes and brackets.
  let j = i + 1;
  while (j < text.length && CLOSERS.has(text[j])) j++;

  // Must be followed by whitespace or end of input.
  if (j < text.length && !/\s/.test(text[j])) return false;

  if (ch === ".") {
    // Decimal numbers: "3.14" — digit on both sides.
    if (i > 0 && /\d/.test(text[i - 1]) && /\d/.test(text[i + 1] ?? "")) return false;

    // Trailing-period abbreviations, including dotted forms like "U.S.".
    const before = text.slice(Math.max(0, i - 12), i);
    const word = (before.match(/[A-Za-z.]+$/) ?? [""])[0].toLowerCase();
    if (ABBREVIATIONS.has(word.replace(/\.$/, ""))) return false;

    // A single capital letter before the period is an initial: "J. Smith".
    if (/(^|\s)[A-Za-z]$/.test(before)) return false;
  }

  // Require the next non-space character to look like a sentence opener.
  let k = j;
  while (k < text.length && /\s/.test(text[k])) k++;
  if (k >= text.length) return true;
  const next = text[k];
  return /[A-Z0-9"'\u201c\u2018(\[]/.test(next) || next === "\u2014";
}

/** Split into sentences, preserving exact character offsets into `text`. */
export function splitSentences(text: string): Sentence[] {
  const sentences: Sentence[] = [];
  let cursor = 0;
  let index = 0;

  const push = (start: number, end: number) => {
    const raw = text.slice(start, end);
    const trimmedStart = start + (raw.length - raw.trimStart().length);
    const trimmedEnd = end - (raw.length - raw.trimEnd().length);
    if (trimmedEnd <= trimmedStart) return;
    sentences.push({
      index: index++,
      text: text.slice(trimmedStart, trimmedEnd),
      start: trimmedStart,
      end: trimmedEnd,
    });
  };

  for (let i = 0; i < text.length; i++) {
    // A blank line is a hard break even without a terminator: headings,
    // list items and fragments should not be glued onto the next sentence.
    if (text[i] === "\n" && /\n[ \t]*\n/.test(text.slice(i, i + 3))) {
      push(cursor, i);
      const m = text.slice(i).match(/^\s+/);
      cursor = i + (m ? m[0].length : 1);
      i = cursor - 1;
      continue;
    }
    if (isSentenceBreak(text, i)) {
      let j = i + 1;
      while (j < text.length && CLOSERS.has(text[j])) j++;
      push(cursor, j);
      cursor = j;
      i = j - 1;
    }
  }
  push(cursor, text.length);
  return sentences;
}

export interface ChunkOptions {
  /** Target tokens per chunk. */
  maxTokens?: number;
  /** Sentences of context carried from the previous chunk. */
  overlapSentences?: number;
}

export function chunkSentences(
  sentences: Sentence[],
  { maxTokens = 800, overlapSentences = 1 }: ChunkOptions = {},
): Chunk[] {
  if (sentences.length === 0) return [];

  const chunks: Chunk[] = [];
  let current: Sentence[] = [];
  let currentTokens = 0;
  let overlapCount = 0;

  const flush = () => {
    if (current.length === 0) return;
    const start = current[0].start;
    const end = current[current.length - 1].end;
    chunks.push({
      index: chunks.length,
      text: current.map((s) => s.text).join(" "),
      start,
      end,
      sentenceIndices: current.map((s) => s.index),
      overlapCount,
      approxTokens: currentTokens,
    });
  };

  for (const sentence of sentences) {
    const tokens = estimateTokens(sentence.text);

    if (currentTokens + tokens > maxTokens && current.length > overlapCount) {
      flush();
      const carry = overlapSentences > 0 ? current.slice(-overlapSentences) : [];
      current = [...carry];
      currentTokens = carry.reduce((n, s) => n + estimateTokens(s.text), 0);
      overlapCount = carry.length;
    }

    current.push(sentence);
    currentTokens += tokens;
  }
  flush();
  return chunks;
}

export function chunkText(text: string, options?: ChunkOptions): {
  sentences: Sentence[];
  chunks: Chunk[];
} {
  const sentences = splitSentences(text);
  return { sentences, chunks: chunkSentences(sentences, options) };
}

/**
 * Aggregate per-chunk document scores into one document score.
 *
 * Length-weighted mean, then blended toward the maximum so a single strongly
 * AI chunk inside a long human document is not averaged away (PRD 16.3).
 */
export function aggregateChunkScores(
  scores: number[],
  weights: number[],
  maxFlagWeight = 0.3,
): number {
  if (scores.length === 0) return 0;
  if (scores.length === 1) return scores[0];

  const totalWeight = weights.reduce((a, b) => a + b, 0) || scores.length;
  const weighted =
    scores.reduce((sum, s, i) => sum + s * (weights[i] ?? 1), 0) / totalWeight;
  const max = Math.max(...scores);

  return weighted * (1 - maxFlagWeight) + max * maxFlagWeight;
}

/**
 * Map sentence scores from overlapping chunks back onto the document.
 * Overlapped sentences appear in two chunks; take the mean of both readings.
 */
export function mergeSentenceScores(
  sentenceCount: number,
  perChunk: { sentenceIndices: number[]; scores: number[] }[],
): number[] {
  const sums = new Array(sentenceCount).fill(0);
  const counts = new Array(sentenceCount).fill(0);

  for (const chunk of perChunk) {
    chunk.sentenceIndices.forEach((sentenceIndex, i) => {
      const score = chunk.scores[i];
      if (score === undefined || sentenceIndex >= sentenceCount) return;
      sums[sentenceIndex] += score;
      counts[sentenceIndex] += 1;
    });
  }

  return sums.map((sum, i) => (counts[i] > 0 ? sum / counts[i] : 0));
}
