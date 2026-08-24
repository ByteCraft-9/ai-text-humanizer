"use client";

/**
 * Input region (PRD 16.1): textarea, drag-and-drop file zone, live word
 * counter against the 5,000-word limit, and format badges.
 *
 * Parsing is entirely client-side (PRD 6.1) — nothing large crosses the
 * 4.5 MB body limit, it costs zero server compute, and untrusted binaries
 * never reach the server.
 */

import { useCallback, useRef, useState } from "react";

import { countWords } from "@/lib/chunk";
import { MAX_WORDS } from "@/lib/detect";
import { ACCEPTED_EXTENSIONS, ExtractionError, extractFile } from "@/lib/extract";

interface Props {
  value: string;
  onChange: (text: string) => void;
  disabled?: boolean;
}

export function Editor({ value, onChange, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [parsing, setParsing] = useState<{ fraction: number; label: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const words = countWords(value);
  const overLimit = words > MAX_WORDS;

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setParsing({ fraction: 0, label: "Opening file" });
      try {
        const result = await extractFile(file, (fraction, label) =>
          setParsing({ fraction, label }),
        );
        onChange(result.text);
        setSource(`${result.source}${result.pages ? ` · ${result.pages} pages` : ""}`);
      } catch (err) {
        setError(
          err instanceof ExtractionError
            ? err.message
            : `That file could not be read: ${String((err as Error).message)}`,
        );
        setSource(null);
      } finally {
        setParsing(null);
      }
    },
    [onChange],
  );

  return (
    <div className="space-y-3">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file && !disabled) void handleFile(file);
        }}
        className="relative"
      >
        <label htmlFor="document" className="sr-only">
          Text to analyse
        </label>
        <textarea
          id="document"
          value={value}
          disabled={disabled}
          onChange={(event) => {
            onChange(event.target.value);
            setSource(null);
          }}
          placeholder="Paste your text here, or drop a .txt, .md, .pdf or .docx file."
          spellCheck={false}
          aria-describedby="word-count"
          aria-invalid={overLimit}
          className="h-72 w-full resize-y rounded-[10px] border p-4 font-sans text-[0.95rem] leading-relaxed"
          style={{
            background: "var(--surface)",
            borderColor: dragging ? "var(--accent)" : "var(--border)",
            outline: dragging ? "2px dashed var(--accent)" : undefined,
            outlineOffset: "-6px",
          }}
        />

        {dragging && (
          <div
            className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-[10px] text-sm font-medium"
            style={{ background: "color-mix(in srgb, var(--accent) 12%, transparent)" }}
          >
            Drop to read the file
          </div>
        )}

        {parsing && (
          <div className="absolute inset-x-0 bottom-0 rounded-b-[10px] p-3" style={{ background: "var(--surface-sunken)" }}>
            <p className="text-xs muted">{parsing.label}</p>
            <div className="mt-1 h-1 w-full overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
              <div
                className="h-full rounded-full transition-[width]"
                style={{ width: `${parsing.fraction * 100}%`, background: "var(--accent)" }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            className="btn text-xs"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          >
            Choose a file
          </button>
          <input
            ref={inputRef}
            type="file"
            className="sr-only"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void handleFile(file);
              event.target.value = "";
            }}
          />
          {ACCEPTED_EXTENSIONS.map((extension) => (
            <span
              key={extension}
              className="rounded px-1.5 py-0.5 font-mono text-[0.7rem] muted"
              style={{ background: "var(--surface-sunken)" }}
            >
              {extension}
            </span>
          ))}
          {value && (
            <button className="btn text-xs" onClick={() => onChange("")} disabled={disabled}>
              Clear
            </button>
          )}
        </div>

        <p
          id="word-count"
          className="text-xs tabular-nums"
          style={{ color: overLimit ? "var(--danger)" : "var(--text-muted)" }}
        >
          {words.toLocaleString()} / {MAX_WORDS.toLocaleString()} words
          {overLimit && " — over the limit"}
        </p>
      </div>

      {source && <p className="text-xs muted">Read from {source}</p>}

      {error && (
        <p
          role="alert"
          className="rounded p-3 text-sm"
          style={{ background: "var(--warn-soft)", color: "var(--warn)" }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
