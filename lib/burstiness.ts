/**
 * Deterministic burstiness pass — step 7 of the humanize loop (PRD 10.1).
 *
 * Costs zero API tokens. LLM prose is characteristically *even*: uniform
 * sentence length, no contractions, the same handful of discourse markers,
 * heavy parallel structure. Each transform below attacks one of those
 * regularities without touching meaning, so no similarity check is needed.
 *
 * Every transform is conservative by design. This pass runs on text that has
 * already been rewritten, and a clumsy edit here would undo the LLM's work.
 */

import { splitSentences } from "./chunk";

export interface BurstinessOptions {
  contractions?: boolean;
  discourseMarkers?: boolean;
  lengthVariance?: boolean;
  /** Seed for reproducible output; same input + seed => same result. */
  seed?: number;
}

export interface BurstinessReport {
  text: string;
  applied: string[];
  /** Standard deviation of sentence length, before and after. */
  varianceBefore: number;
  varianceAfter: number;
}

/** Deterministic PRNG so a rerun of the same document produces the same text. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0 || 0x2f6e2b1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) % 100_000) / 100_000;
  };
}

// ---------------------------------------------------------------------------
// Contractions
// ---------------------------------------------------------------------------

/**
 * Expanded forms that LLMs over-produce. Ordered longest-first at use so
 * "it is not" contracts before "it is". Deliberately excludes forms whose
 * contraction is ambiguous ("it's" for "it has", "she'd" for "she had").
 */
const CONTRACTIONS: [RegExp, string][] = [
  [/\bit is not\b/gi, "it isn't"],
  [/\bthere is not\b/gi, "there isn't"],
  [/\bthat is not\b/gi, "that isn't"],
  [/\bdoes not\b/gi, "doesn't"],
  [/\bdo not\b/gi, "don't"],
  [/\bdid not\b/gi, "didn't"],
  [/\bis not\b/gi, "isn't"],
  [/\bare not\b/gi, "aren't"],
  [/\bwas not\b/gi, "wasn't"],
  [/\bwere not\b/gi, "weren't"],
  [/\bhas not\b/gi, "hasn't"],
  [/\bhave not\b/gi, "haven't"],
  [/\bhad not\b/gi, "hadn't"],
  [/\bwill not\b/gi, "won't"],
  [/\bwould not\b/gi, "wouldn't"],
  [/\bcould not\b/gi, "couldn't"],
  [/\bshould not\b/gi, "shouldn't"],
  [/\bcannot\b/gi, "can't"],
  [/\bcan not\b/gi, "can't"],
  [/\bit is\b/g, "it's"],
  [/\bthat is\b/g, "that's"],
  [/\bthere is\b/g, "there's"],
  [/\bwhat is\b/g, "what's"],
  [/\bwho is\b/g, "who's"],
  [/\bthey are\b/gi, "they're"],
  [/\bwe are\b/gi, "we're"],
  [/\byou are\b/gi, "you're"],
  [/\bI am\b/g, "I'm"],
  [/\bI will\b/g, "I'll"],
  [/\bwe will\b/gi, "we'll"],
  [/\bthey will\b/gi, "they'll"],
  [/\byou will\b/gi, "you'll"],
  [/\bI have\b/g, "I've"],
  [/\bwe have\b/gi, "we've"],
  [/\bthey have\b/gi, "they've"],
  [/\byou have\b/gi, "you've"],
  [/\bI would\b/g, "I'd"],
  [/\blet us\b/gi, "let's"],
];

/** Preserve the original capitalisation of the first letter. */
function matchCase(original: string, replacement: string): string {
  if (!original || !replacement) return replacement;
  if (original[0] === original[0].toUpperCase() && /[a-z]/i.test(original[0])) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}

/**
 * Contract a fraction of eligible sites rather than all of them. Contracting
 * everything is as uniform as contracting nothing — the goal is variance.
 */
export function applyContractions(text: string, rate = 0.6, random = Math.random): string {
  let out = text;
  for (const [pattern, replacement] of CONTRACTIONS) {
    out = out.replace(pattern, (match) =>
      random() < rate ? matchCase(match, replacement) : match,
    );
  }
  return out;
}

// ---------------------------------------------------------------------------
// Discourse markers
// ---------------------------------------------------------------------------

/**
 * The tell-tale LLM connectives, each mapped to plainer alternatives. An
 * empty string means "drop the marker entirely", which is often the most
 * human option — human writers simply start the sentence.
 */
const MARKER_ALTERNATIVES: Record<string, string[]> = {
  furthermore: ["", "On top of that,", "And", "Beyond that,"],
  moreover: ["", "What's more,", "Also,", "On top of that,"],
  additionally: ["", "Also,", "And", "As well as that,"],
  however: ["But", "Though", "That said,", "Even so,"],
  nevertheless: ["Even so,", "Still,", "All the same,"],
  nonetheless: ["Even so,", "Still,", "All the same,"],
  therefore: ["So", "Which means", "That means"],
  consequently: ["So", "As a result,", "Which means"],
  thus: ["So", "Which means", ""],
  hence: ["So", "Which is why", ""],
  "in conclusion": ["In the end,", "So,", "To close,"],
  "in summary": ["In short,", "Put simply,", "Briefly,"],
  "it is important to note that": ["Worth noting:", "Note that", ""],
  "it is worth noting that": ["Worth noting:", "Note that", ""],
  "it should be noted that": ["Note that", "Worth saying:", ""],
  "in today's world": ["These days,", "Now,", "Today,"],
  "in the modern era": ["These days,", "Now,", "Today,"],
  "plays a crucial role": ["matters", "is central", "counts for a lot"],
  "plays a vital role": ["matters", "is central", "counts for a lot"],
  "a wide range of": ["many", "all sorts of", "plenty of"],
  "a variety of": ["several", "all sorts of", "different"],
  "delve into": ["dig into", "look at", "get into"],
  "in order to": ["to"],
  utilize: ["use"],
  utilizes: ["uses"],
  utilizing: ["using"],
  "leverage the": ["use the"],
  "facilitate the": ["help the", "enable the"],
};

/**
 * Sentinel written where a discourse marker is deleted outright, so the word
 * that inherits sentence-initial position can be capitalised in one pass.
 * Never appears in user text.
 */
const DROP = "\u0000";

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function varyDiscourseMarkers(
  text: string,
  rate = 0.75,
  random = Math.random,
): string {
  let out = text;

  for (const [marker, alternatives] of Object.entries(MARKER_ALTERNATIVES)) {
    // Sentence-initial form, optionally followed by a comma.
    const initial = new RegExp(
      `(^|[.!?]\\s+|\\n\\s*)(${escapeRegex(marker)})(,?)(\\s+)`,
      "gi",
    );
    out = out.replace(initial, (whole, lead: string, _m, _comma, trail: string) => {
      if (random() >= rate) return whole;
      const choice = alternatives[Math.floor(random() * alternatives.length)];
      if (!choice) {
        // Dropping the marker: capitalise whatever now starts the sentence.
        return `${lead}${DROP}`;
      }
      return `${lead}${choice}${trail}`;
    });

    // Mid-sentence form, for the lower-case lexical replacements.
    if (!/[A-Z]/.test(marker) && marker.split(" ").length <= 3) {
      const mid = new RegExp(`(?<=\\w[ ,])(${escapeRegex(marker)})\\b`, "gi");
      out = out.replace(mid, (match) => {
        if (random() >= rate) return match;
        const usable = alternatives.filter(Boolean);
        if (usable.length === 0) return match;
        return matchCase(match, usable[Math.floor(random() * usable.length)]);
      });
    }
  }

  // Resolve the drop marker: remove it and capitalise the following word.
  out = out.replace(
    new RegExp(DROP + "\\s*([a-z])", "g"),
    (_, ch: string) => ch.toUpperCase(),
  );
  out = out.replace(new RegExp(DROP, "g"), "");

  // A dropped marker can leave a doubled space or an orphaned comma.
  return out.replace(/ {2,}/g, " ").replace(/\s+,/g, ",");
}

// ---------------------------------------------------------------------------
// Sentence length variance
// ---------------------------------------------------------------------------

/** Coordinators safe to split on: the clause after them stands alone. */
const SPLIT_POINTS = /,\s+(and|but|so|yet)\s+(?=[a-z])/;

function wordCount(s: string): number {
  return (s.match(/\S+/g) ?? []).length;
}

function stdev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length);
}

/**
 * Split long compound sentences and merge adjacent short ones, raising the
 * standard deviation of sentence length. Only acts on sentences that are
 * clearly uniform — if the writing already varies, it leaves it alone.
 */
export function varySentenceLength(text: string, random = Math.random): string {
  const sentences = splitSentences(text);
  if (sentences.length < 3) return text;

  const lengths = sentences.map((s) => wordCount(s.text));
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const spread = stdev(lengths);

  // Already bursty enough; leave it be.
  if (spread > mean * 0.45) return text;

  // When lengths are near-identical no sentence is "above average", so an
  // above-average test would skip the very case this pass exists for. In that
  // regime any sentence with a clean split point is a candidate.
  const uniform = spread < mean * 0.2;

  const out: string[] = [];
  for (let i = 0; i < sentences.length; i++) {
    const current = sentences[i].text;
    const length = lengths[i];

    // Split a long compound sentence at a coordinating conjunction. The real
    // guard is the head/tail word counts below; this is just a cheap filter.
    if (length >= 12 && (uniform || length > mean * 1.15) && random() < 0.6) {
      const match = current.match(SPLIT_POINTS);
      if (match?.index !== undefined) {
        const head = current.slice(0, match.index).trim();
        const tailStart = match.index + match[0].length;
        const tail = current.slice(tailStart).trim();
        if (wordCount(head) >= 5 && wordCount(tail) >= 5) {
          const connector = match[1] === "and" ? "" : `${matchCase("X", match[1])} `;
          out.push(`${head}.`);
          out.push(
            connector
              ? `${connector}${tail}`
              : tail.charAt(0).toUpperCase() + tail.slice(1),
          );
          continue;
        }
      }
    }

    // Merge two adjacent short sentences into one.
    const next = sentences[i + 1];
    if (
      next &&
      length < mean * 0.8 &&
      lengths[i + 1] < mean * 0.8 &&
      /[.]$/.test(current) &&
      random() < 0.45
    ) {
      const joined = `${current.replace(/\.$/, "")} — ${
        next.text.charAt(0).toLowerCase() + next.text.slice(1)
      }`;
      out.push(joined);
      i++;
      continue;
    }

    out.push(current);
  }

  return out.join(" ");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export function applyBurstinessPass(
  text: string,
  options: BurstinessOptions = {},
): BurstinessReport {
  const {
    contractions = true,
    discourseMarkers = true,
    lengthVariance = true,
    seed = 0x5eed,
  } = options;

  const random = makeRandom(seed);
  const applied: string[] = [];
  const varianceBefore = stdev(splitSentences(text).map((s) => wordCount(s.text)));

  let out = text;
  if (discourseMarkers) {
    const next = varyDiscourseMarkers(out, 0.75, random);
    if (next !== out) applied.push("discourse markers");
    out = next;
  }
  if (contractions) {
    const next = applyContractions(out, 0.6, random);
    if (next !== out) applied.push("contractions");
    out = next;
  }
  if (lengthVariance) {
    const next = varySentenceLength(out, random);
    if (next !== out) applied.push("sentence length");
    out = next;
  }

  return {
    text: out,
    applied,
    varianceBefore,
    varianceAfter: stdev(splitSentences(out).map((s) => wordCount(s.text))),
  };
}
