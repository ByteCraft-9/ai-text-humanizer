"use client";

/**
 * Per-sentence evidence rendered on the text itself (PRD 16.1, G3).
 *
 * The point of the heatmap is that evidence sits where the writing is — a
 * single opaque percentage is exactly what the product exists to improve on.
 *
 * Accessibility (PRD 15.3): every band carries a distinct underline pattern
 * as well as a colour, each span is a focusable element with its numeric
 * score in the accessible name, and a plain table of the flagged sentences is
 * available for anyone who cannot use the overlay at all.
 */

import { useMemo, useState } from "react";

import type { SentenceScore } from "@/lib/types";

interface Props {
  text: string;
  sentences: SentenceScore[];
}

export const BANDS = [
  { max: 0.2, className: "band-human", label: "reads as human" },
  { max: 0.4, className: "band-low", label: "mostly human" },
  { max: 0.6, className: "band-mid", label: "mixed" },
  { max: 0.8, className: "band-high", label: "likely AI" },
  { max: 1.01, className: "band-ai", label: "reads as AI" },
] as const;

export function bandFor(score: number) {
  return BANDS.find((b) => score < b.max) ?? BANDS[BANDS.length - 1];
}

export function Heatmap({ text, sentences }: Props) {
  const [selected, setSelected] = useState<number | null>(null);

  // Walk the sentence spans in order, emitting the untouched gaps between
  // them so whitespace and paragraph breaks survive intact.
  const segments = useMemo(() => {
    const out: { key: string; content: string; sentence?: SentenceScore }[] = [];
    let cursor = 0;

    for (const sentence of [...sentences].sort((a, b) => a.start - b.start)) {
      if (sentence.start > cursor) {
        out.push({ key: `gap-${cursor}`, content: text.slice(cursor, sentence.start) });
      }
      out.push({
        key: `s-${sentence.index}`,
        content: text.slice(sentence.start, sentence.end),
        sentence,
      });
      cursor = sentence.end;
    }
    if (cursor < text.length) {
      out.push({ key: `gap-${cursor}`, content: text.slice(cursor) });
    }
    return out;
  }, [text, sentences]);

  const flagged = sentences.filter((s) => s.score >= 0.5).sort((a, b) => b.score - a.score);

  return (
    <div className="space-y-4">
      <div
        className="surface max-h-[28rem] overflow-y-auto p-4 text-[0.95rem] leading-[1.75] whitespace-pre-wrap"
        role="region"
        aria-label="Text with per-sentence AI scores"
      >
        {segments.map((segment) =>
          segment.sentence ? (
            <mark
              key={segment.key}
              tabIndex={0}
              className={`band ${bandFor(segment.sentence.score).className} cursor-help bg-transparent text-[inherit]`}
              style={
                selected === segment.sentence.index
                  ? { outline: "2px solid var(--accent)", outlineOffset: "1px" }
                  : undefined
              }
              onFocus={() => setSelected(segment.sentence!.index)}
              onBlur={() => setSelected(null)}
              onMouseEnter={() => setSelected(segment.sentence!.index)}
              onMouseLeave={() => setSelected(null)}
              aria-label={`${Math.round(segment.sentence.score * 100)} percent AI, ${
                bandFor(segment.sentence.score).label
              }: ${segment.content}`}
              title={`${Math.round(segment.sentence.score * 100)}% AI — ${
                bandFor(segment.sentence.score).label
              }`}
            >
              {segment.content}
            </mark>
          ) : (
            <span key={segment.key}>{segment.content}</span>
          ),
        )}
      </div>

      <Legend />

      {flagged.length > 0 && (
        <details className="surface p-4">
          <summary className="cursor-pointer text-sm font-medium">
            The {flagged.length} highest-scoring sentence
            {flagged.length === 1 ? "" : "s"}, as a list
          </summary>
          <p className="mt-2 text-xs muted">
            The same evidence without the overlay — for screen readers, and for
            deciding what to edit by hand.
          </p>
          <ol className="mt-3 space-y-2">
            {flagged.slice(0, 20).map((sentence) => (
              <li key={sentence.index} className="text-sm">
                <span className="mr-2 font-medium tabular-nums">
                  {Math.round(sentence.score * 100)}%
                </span>
                <span className="muted">{sentence.text}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs muted">
      <span className="font-medium">Bands:</span>
      {BANDS.map((band, i) => {
        const from = i === 0 ? 0 : Math.round(BANDS[i - 1].max * 100);
        const to = Math.min(100, Math.round(band.max * 100));
        return (
          <span key={band.className} className="flex items-center gap-1.5">
            <span className={`band ${band.className} px-1.5`}>
              {from}–{to}%
            </span>
            <span>{band.label}</span>
          </span>
        );
      })}
    </div>
  );
}
