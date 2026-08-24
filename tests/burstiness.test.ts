import { describe, expect, it } from "vitest";

import {
  applyBurstinessPass,
  applyContractions,
  varyDiscourseMarkers,
  varySentenceLength,
} from "../lib/burstiness";
import { splitSentences } from "../lib/chunk";

/** Always contract / always vary, so the tests are not flaky on the PRNG. */
const always = () => 0;
const never = () => 1;

describe("applyContractions", () => {
  it("contracts expanded forms", () => {
    expect(applyContractions("It is not working and they are late.", 1, always)).toBe(
      "It isn't working and they're late.",
    );
  });

  it("preserves the leading capital", () => {
    expect(applyContractions("Do not stop.", 1, always)).toBe("Don't stop.");
  });

  it("leaves the text alone when the rate is zero", () => {
    const input = "It is not working.";
    expect(applyContractions(input, 0, never)).toBe(input);
  });

  it("prefers the longer form so 'is not' does not become \"is n't\"", () => {
    expect(applyContractions("That is not right.", 1, always)).toBe("That isn't right.");
  });
});

describe("varyDiscourseMarkers", () => {
  it("replaces the tell-tale connectives", () => {
    const out = varyDiscourseMarkers(
      "Furthermore, the results were clear. Moreover, they were repeatable.",
      1,
      always,
    );
    expect(out).not.toMatch(/Furthermore/i);
    expect(out).not.toMatch(/Moreover/i);
  });

  it("capitalises the new sentence opener when a marker is dropped", () => {
    // The first alternative for "furthermore" is "" — drop it entirely.
    const out = varyDiscourseMarkers("Furthermore, the results were clear.", 1, always);
    expect(out).toBe("The results were clear.");
  });

  it("never leaves the sentinel in the output", () => {
    const out = varyDiscourseMarkers(
      "Furthermore, it works. Moreover, it is fast. Additionally, it is cheap.",
      1,
      always,
    );
    expect(out).not.toContain("\u0000");
  });

  it("leaves text alone at rate zero", () => {
    const input = "However, the results were mixed.";
    expect(varyDiscourseMarkers(input, 0, never)).toBe(input);
  });
});

describe("varySentenceLength", () => {
  const wordCount = (s: string) => (s.match(/\S+/g) ?? []).length;
  const stdev = (values: number[]) => {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(values.reduce((n, v) => n + (v - mean) ** 2, 0) / values.length);
  };

  it("raises variance on uniformly-long sentences", () => {
    const uniform = Array.from(
      { length: 6 },
      (_, i) =>
        `The team reviewed the ${i} findings carefully, and they published a full report afterwards.`,
    ).join(" ");

    const before = stdev(splitSentences(uniform).map((s) => wordCount(s.text)));
    const after = stdev(splitSentences(varySentenceLength(uniform, always)).map((s) => wordCount(s.text)));

    expect(after).toBeGreaterThan(before);
  });

  it("leaves already-varied writing alone", () => {
    const varied =
      "Short. This one runs considerably longer and carries several clauses, " +
      "which is exactly the kind of unevenness the pass is looking for. Fine. " +
      "Another long one, with a subordinate clause that keeps going for a while yet.";
    expect(varySentenceLength(varied, always)).toBe(varied);
  });

  it("leaves very short documents alone", () => {
    const input = "One sentence. Two sentences.";
    expect(varySentenceLength(input, always)).toBe(input);
  });
});

describe("applyBurstinessPass", () => {
  it("is deterministic for a given seed", () => {
    const input =
      "Furthermore, it is not clear that they are ready. Moreover, we do not have the data.";
    const a = applyBurstinessPass(input, { seed: 7 }).text;
    const b = applyBurstinessPass(input, { seed: 7 }).text;
    expect(a).toBe(b);
  });

  it("reports which transforms fired", () => {
    const report = applyBurstinessPass(
      "Furthermore, it is not working. Moreover, they are late.",
      { seed: 1 },
    );
    expect(report.applied.length).toBeGreaterThan(0);
  });

  it("never emits the drop sentinel", () => {
    const report = applyBurstinessPass(
      "Furthermore, we tested it. Moreover, it worked. Additionally, it is fast. " +
        "Therefore, we shipped it. Consequently, users are happy.",
      { seed: 3 },
    );
    expect(report.text).not.toContain("\u0000");
  });

  it("leaves empty input alone", () => {
    expect(applyBurstinessPass("").text).toBe("");
  });
});
