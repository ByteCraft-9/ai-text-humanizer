"use client";

/**
 * The honesty requirements, made visible (PRD 10.3, 15.1, R8).
 *
 * These are product requirements, not decoration. H3 requires a persistent
 * notice that no tool can guarantee results against every third-party
 * detector; P2 requires disclosure *before* the first humanize action, not
 * buried in a policy; R8 requires that scores read as evidence, never as a
 * verdict about a person.
 */

import type { HumanizeResult } from "@/lib/types";

/** H3 — persistent, not dismissible. */
export function StandingNotice() {
  return (
    <p className="text-xs leading-relaxed muted">
      No tool can guarantee a result against every third-party detector.
      GPTZero, Turnitin and Originality run different models and update
      continuously, so the second number below is an estimate of how they are
      likely to score this text — not a promise about any one of them.
    </p>
  );
}

/** R8 — the product reports probabilities as evidence, never verdicts. */
export function EvidenceNotice() {
  return (
    <div className="surface p-4 text-sm">
      <h3 className="font-semibold">How to read these numbers</h3>
      <p className="mt-2 muted">
        These are probabilities, not proof. A high score means the writing has
        statistical properties common in generated text — it is not evidence
        that a particular person did anything. Detectors are also known to
        over-flag non-native English writers, which is why the confidence band
        is always shown and why a low-confidence result should carry very
        little weight.
      </p>
    </div>
  );
}

/** P2 — disclosed before the first humanize action, not after. */
export function PrivacyDisclosure({ byokActive }: { byokActive: boolean }) {
  return (
    <div className="surface p-4 text-sm">
      <h3 className="font-semibold">Where your text goes</h3>
      <ul className="mt-2 space-y-1.5 muted">
        <li>
          <strong style={{ color: "var(--text)" }}>Detection stays here.</strong>{" "}
          Scoring runs inside this app&apos;s own functions. Your text is not sent
          to any third party and is not stored — processing is in memory only.
        </li>
        <li>
          <strong style={{ color: "var(--text)" }}>Rewriting does not.</strong>{" "}
          Humanizing sends the flagged sentences to{" "}
          {byokActive ? "the provider whose key you supplied" : "Cerebras, Groq or Google"},
          because that is where the paraphrasing happens. If the text is
          confidential, detect but do not humanize.
        </li>
      </ul>
    </div>
  );
}

const OUTCOME_COPY: Record<
  HumanizeResult["outcome"],
  { title: string; tone: "good" | "neutral" | "warn"; body: string }
> = {
  cleared: {
    title: "Cleared our surrogate detector",
    tone: "good",
    body:
      "The estimated third-party score fell below the threshold. The strict " +
      "score is deliberately harder to move — it is trained to see through " +
      "exactly this kind of rewriting, and it is the number to trust about " +
      "what the text now is.",
  },
  plateaued: {
    title: "Stopped improving",
    tone: "warn",
    body:
      "Further passes were not moving the score, so we stopped rather than " +
      "spend your budget on no gain.",
  },
  max_passes: {
    title: "Reached the pass limit",
    tone: "neutral",
    body:
      "Three passes ran and the score was still above the threshold. More " +
      "passes rarely help past this point; what remains is usually structural.",
  },
  budget_exhausted: {
    title: "Ran out of free-tier budget",
    tone: "warn",
    body:
      "The shared pool was exhausted mid-run. Everything completed so far is " +
      "kept. Your own API key removes this ceiling.",
  },
  error: {
    title: "The run failed",
    tone: "warn",
    body: "Something went wrong before the loop finished. Any completed passes are kept.",
  },
};

/** H4 — a plateau is reported, never papered over by lowering the threshold. */
export function OutcomeNotice({ result }: { result: HumanizeResult }) {
  const copy = OUTCOME_COPY[result.outcome];
  const background =
    copy.tone === "good"
      ? "var(--accent-soft)"
      : copy.tone === "warn"
        ? "var(--warn-soft)"
        : "var(--surface-sunken)";

  return (
    <div className="rounded p-4 text-sm" style={{ background }} role="status">
      <h3 className="font-semibold">{copy.title}</h3>
      <p className="mt-1.5 muted">{copy.body}</p>
      {result.message && <p className="mt-2 muted">{result.message}</p>}

      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs tabular-nums sm:grid-cols-4">
        <div>
          <dt className="muted">Passes</dt>
          <dd className="font-medium">{result.passes.length}</dd>
        </div>
        <div>
          <dt className="muted">Sentences rewritten</dt>
          <dd className="font-medium">
            {result.passes.reduce((n, p) => n + p.applied, 0)}
          </dd>
        </div>
        <div>
          <dt className="muted">Meaning kept</dt>
          <dd className="font-medium">{Math.round(result.mean_similarity * 100)}%</dd>
        </div>
        <div>
          <dt className="muted">Tokens used</dt>
          <dd className="font-medium">{result.tokens_used.toLocaleString()}</dd>
        </div>
      </dl>

      {result.similarity_mode === "lexical" && (
        <p className="mt-3 text-xs muted">
          The sentence encoder was unavailable on this deployment, so meaning
          preservation was checked by word overlap rather than embedding
          similarity, at a correspondingly lower threshold. That check is
          weaker than the one this product specifies — read the rewrites before
          you use them.
        </p>
      )}
    </div>
  );
}

/** Shown when the trained weights are missing from the deployment. */
export function DegradedNotice() {
  return (
    <div
      className="rounded p-3 text-xs"
      style={{ background: "var(--warn-soft)", color: "var(--warn)" }}
      role="status"
    >
      <p className="font-medium">Running without the trained models.</p>
      <p className="mt-1">
        This deployment is scoring on linguistic features alone — the
        fine-tuned classifier and the perplexity models are not present. The
        number is a real measurement of a weaker signal, not the full ensemble.
        Treat it as indicative only.
      </p>
    </div>
  );
}
