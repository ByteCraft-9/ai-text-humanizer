"use client";

/**
 * Before/after comparison with per-sentence score deltas (PRD 16.1).
 *
 * Two views because they answer different questions: the sentence table shows
 * *what the rewrite bought* (score before, score after, similarity retained),
 * and the word diff shows *what actually changed* so nothing slips past
 * unnoticed. Rejected rewrites are listed too — a tool that only shows its
 * successes is grading its own homework.
 */

import { diffWords } from "diff";
import { useMemo, useState } from "react";

import type { HumanizeResult, SentenceRewrite } from "@/lib/types";

type Tab = "sentences" | "words";

export function DiffView({ result }: { result: HumanizeResult }) {
  const [tab, setTab] = useState<Tab>("sentences");

  const rewrites = useMemo(() => {
    // A sentence may be rewritten across several passes; show the last state.
    const byIndex = new Map<number, SentenceRewrite>();
    for (const pass of result.passes) {
      for (const rewrite of pass.rewrites) byIndex.set(rewrite.index, rewrite);
    }
    return [...byIndex.values()].sort(
      (a, b) => b.score_before - b.score_after - (a.score_before - a.score_after),
    );
  }, [result.passes]);

  const applied = rewrites.filter((r) => !r.rejected);
  const rejected = rewrites.filter((r) => r.rejected);

  return (
    <div className="surface p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold">What changed</h3>
        <div className="flex gap-1" role="tablist" aria-label="Comparison view">
          {(
            [
              ["sentences", "By sentence"],
              ["words", "Word diff"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              role="tab"
              aria-selected={tab === value}
              className="btn px-3 py-1 text-xs"
              style={
                tab === value
                  ? { background: "var(--accent-soft)", borderColor: "var(--accent)" }
                  : undefined
              }
              onClick={() => setTab(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {tab === "sentences" ? (
        <SentenceTable applied={applied} rejected={rejected} />
      ) : (
        <WordDiff before={result.original_text} after={result.text} />
      )}
    </div>
  );
}

function SentenceTable({
  applied,
  rejected,
}: {
  applied: SentenceRewrite[];
  rejected: SentenceRewrite[];
}) {
  return (
    <div className="space-y-5">
      <section>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide muted">
          Rewritten ({applied.length})
        </h4>
        {applied.length === 0 ? (
          <p className="text-sm muted">No sentence was improved on this run.</p>
        ) : (
          <ul className="space-y-3">
            {applied.map((rewrite) => (
              <li key={rewrite.index} className="rounded border p-3" style={{ borderColor: "var(--border)" }}>
                <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs tabular-nums muted">
                  <span>
                    {Math.round(rewrite.score_before * 100)}% →{" "}
                    <strong style={{ color: "var(--accent)" }}>
                      {Math.round(rewrite.score_after * 100)}%
                    </strong>
                  </span>
                  <span>meaning kept {Math.round(rewrite.similarity * 100)}%</span>
                </div>
                <p className="text-sm line-through muted">{rewrite.original}</p>
                <p className="mt-1 text-sm">{rewrite.rewritten}</p>
              </li>
            ))}
          </ul>
        )}
      </section>

      {rejected.length > 0 && (
        <details>
          <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide muted">
            Left alone ({rejected.length})
          </summary>
          <ul className="mt-3 space-y-2">
            {rejected.map((rewrite) => (
              <li key={rewrite.index} className="text-sm">
                <span className="mr-2 rounded px-1.5 py-0.5 text-xs" style={{ background: "var(--surface-sunken)" }}>
                  {REJECTION_COPY[rewrite.rejected ?? "no_improvement"]}
                </span>
                <span className="muted">{rewrite.original}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

const REJECTION_COPY: Record<NonNullable<SentenceRewrite["rejected"]>, string> = {
  similarity: "every rewrite drifted from the meaning",
  no_improvement: "no rewrite scored better",
  generation_failed: "the model returned nothing usable",
};

function WordDiff({ before, after }: { before: string; after: string }) {
  const parts = useMemo(() => diffWords(before, after), [before, after]);

  return (
    <div
      className="max-h-[24rem] overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed"
      aria-label="Word-level differences between the original and the rewrite"
    >
      {parts.map((part, i) => {
        if (part.added) {
          return (
            <ins
              key={i}
              className="rounded px-0.5 no-underline"
              style={{ background: "color-mix(in srgb, #0f766e 18%, transparent)" }}
            >
              {part.value}
            </ins>
          );
        }
        if (part.removed) {
          return (
            <del
              key={i}
              className="rounded px-0.5"
              style={{ background: "color-mix(in srgb, #be123c 16%, transparent)" }}
            >
              {part.value}
            </del>
          );
        }
        return <span key={i}>{part.value}</span>;
      })}
    </div>
  );
}
