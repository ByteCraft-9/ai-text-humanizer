"use client";

/**
 * Dual score display (PRD 16.1, H2).
 *
 * The two numbers are the product's whole argument, so they are shown
 * side by side with a plain-language explanation of why they differ — never a
 * single blended figure, and never the flattering one alone.
 */

import type { Confidence } from "@/lib/types";

interface Props {
  label: string;
  score: number;
  description: string;
  confidence?: Confidence;
  interval?: [number, number];
  /** Previous value, to show the delta after a humanize run. */
  previous?: number;
  emphasis?: "primary" | "secondary";
}

const CONFIDENCE_COPY: Record<Confidence, string> = {
  high: "Signals agree and there is plenty of text to go on.",
  medium: "Signals broadly agree, but the evidence is thinner than ideal.",
  low: "Signals disagree, or the text is too short to judge reliably. Treat this as weak evidence.",
};

function bandFor(score: number): { label: string; className: string } {
  if (score < 0.2) return { label: "Reads as human", className: "band-human" };
  if (score < 0.4) return { label: "Mostly human", className: "band-low" };
  if (score < 0.6) return { label: "Mixed", className: "band-mid" };
  if (score < 0.8) return { label: "Likely AI", className: "band-high" };
  return { label: "Reads as AI", className: "band-ai" };
}

function Arc({ score, size = 132 }: { score: number; size?: number }) {
  const stroke = 10;
  const radius = (size - stroke) / 2;
  // Three-quarter arc, opening at the bottom.
  const circumference = 2 * Math.PI * radius;
  const sweep = circumference * 0.75;
  const filled = sweep * Math.min(1, Math.max(0, score));

  const colour =
    score < 0.2
      ? "#0f766e"
      : score < 0.4
        ? "#65a30d"
        : score < 0.6
          ? "#ca8a04"
          : score < 0.8
            ? "#ea580c"
            : "#be123c";

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <g transform={`rotate(135 ${size / 2} ${size / 2})`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth={stroke}
          strokeDasharray={`${sweep} ${circumference}`}
          strokeLinecap="round"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={colour}
          strokeWidth={stroke}
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
        />
      </g>
    </svg>
  );
}

export function ScoreGauge({
  label,
  score,
  description,
  confidence,
  interval,
  previous,
  emphasis = "primary",
}: Props) {
  const band = bandFor(score);
  const percent = Math.round(score * 100);
  const delta = previous === undefined ? null : Math.round((score - previous) * 100);

  return (
    <div
      className="surface flex flex-col items-center gap-2 p-5 text-center"
      style={emphasis === "secondary" ? { background: "var(--surface-sunken)" } : undefined}
    >
      <h3 className="text-sm font-semibold uppercase tracking-wide muted">{label}</h3>

      <div className="relative">
        <Arc score={score} />
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-semibold tabular-nums">{percent}%</span>
          <span className="text-xs muted">AI</span>
        </div>
      </div>

      <p className={`rounded px-2 py-0.5 text-sm font-medium band ${band.className}`}>
        {band.label}
      </p>

      {delta !== null && delta !== 0 && (
        <p className="text-sm tabular-nums" style={{ color: delta < 0 ? "var(--accent)" : "var(--danger)" }}>
          {delta > 0 ? "+" : ""}
          {delta} points since the original
        </p>
      )}

      {interval && (
        <p className="text-xs muted tabular-nums">
          Range {Math.round(interval[0] * 100)}–{Math.round(interval[1] * 100)}%
        </p>
      )}

      {confidence && (
        <p className="text-xs muted">
          <span className="font-medium">{confidence} confidence.</span>{" "}
          {CONFIDENCE_COPY[confidence]}
        </p>
      )}

      <p className="mt-1 max-w-[26ch] text-xs muted">{description}</p>
    </div>
  );
}

export function SignalBreakdown({ signals }: { signals: Record<string, number> }) {
  const rows: { key: string; label: string; hint: string }[] = [
    {
      key: "classifier",
      label: "Fine-tuned classifier",
      hint: "DeBERTa-v3, trained on RAID and DACTYL",
    },
    {
      key: "features",
      label: "Linguistic features",
      hint: "30 readability, vocabulary and repetition measures",
    },
    {
      key: "binoculars_ratio",
      label: "Perplexity ratio",
      hint: "Binoculars-style; generalises to generators we never trained on",
    },
    {
      key: "burstiness",
      label: "Burstiness",
      hint: "How unevenly surprise is spread across the text",
    },
  ];

  return (
    <div className="surface p-4">
      <h3 className="mb-3 text-sm font-semibold">Why this score</h3>
      <dl className="space-y-3">
        {rows.map((row) => {
          const value = signals[row.key] ?? 0;
          const percent = Math.round(value * 100);
          return (
            <div key={row.key}>
              <div className="flex items-baseline justify-between gap-3">
                <dt className="text-sm">{row.label}</dt>
                <dd className="text-sm font-medium tabular-nums">{percent}%</dd>
              </div>
              <div
                className="mt-1 h-1.5 w-full overflow-hidden rounded-full"
                style={{ background: "var(--surface-sunken)" }}
                role="img"
                aria-label={`${row.label}: ${percent} percent AI`}
              >
                <div
                  className="h-full rounded-full"
                  style={{ width: `${percent}%`, background: "var(--accent)" }}
                />
              </div>
              <p className="mt-1 text-xs muted">{row.hint}</p>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
