"use client";

/**
 * Bring-your-own-key settings (PRD 11.3, P4).
 *
 * The key is stored in this browser's localStorage and attached to the
 * humanize request from the client. The server holds it for the duration of
 * one request and never writes it anywhere. That is stated on the panel
 * itself, not buried in a policy.
 */

import { useEffect, useState } from "react";

import { BYOK_PRESETS, clearByok, loadByok, maskKey, saveByok, type ByokSettings } from "@/lib/byok";

interface Props {
  open: boolean;
  onClose: () => void;
  onChange: (settings: ByokSettings | null) => void;
}

export function ByokPanel({ open, onClose, onChange }: Props) {
  const [presetId, setPresetId] = useState(BYOK_PRESETS[0].id);
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState(BYOK_PRESETS[0].model);
  const [existing, setExisting] = useState<ByokSettings | null>(null);

  useEffect(() => {
    if (!open) return;
    const saved = loadByok();
    setExisting(saved);
    if (saved) setModel(saved.model);
  }, [open]);

  if (!open) return null;

  const preset = BYOK_PRESETS.find((p) => p.id === presetId) ?? BYOK_PRESETS[0];

  const save = () => {
    if (!apiKey.trim()) return;
    const settings: ByokSettings = {
      enabled: true,
      apiKey: apiKey.trim(),
      model: model.trim() || preset.model,
      flavour: preset.flavour,
      baseUrl: preset.baseUrl,
    };
    saveByok(settings);
    setExisting(settings);
    setApiKey("");
    onChange(settings);
  };

  const remove = () => {
    clearByok();
    setExisting(null);
    onChange(null);
  };

  return (
    <div
      className="fixed inset-0 z-40 flex items-start justify-center overflow-y-auto bg-black/40 p-4 pt-16"
      role="dialog"
      aria-modal="true"
      aria-labelledby="byok-title"
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <div className="surface w-full max-w-lg p-5">
        <div className="mb-3 flex items-start justify-between gap-4">
          <div>
            <h2 id="byok-title" className="text-base font-semibold">
              Use your own API key
            </h2>
            <p className="mt-1 text-sm muted">
              Removes the shared daily ceiling entirely. Detection never needs a
              key — it runs locally either way.
            </p>
          </div>
          <button className="btn px-2 py-1 text-xs" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>

        {existing && (
          <div
            className="mb-4 rounded p-3 text-sm"
            style={{ background: "var(--accent-soft)" }}
          >
            <p className="font-medium">A key is saved in this browser.</p>
            <p className="mt-1 font-mono text-xs">{maskKey(existing.apiKey)}</p>
            <p className="mt-1 text-xs muted">
              {existing.model} · {existing.flavour === "gemini" ? "Google" : "OpenAI-compatible"}
            </p>
            <button className="btn mt-2 px-2 py-1 text-xs" onClick={remove}>
              Remove it
            </button>
          </div>
        )}

        <div className="space-y-3">
          <label className="block">
            <span className="text-sm font-medium">Provider</span>
            <select
              className="mt-1 w-full rounded border px-2 py-1.5 text-sm"
              style={{ background: "var(--surface)", borderColor: "var(--border-strong)" }}
              value={presetId}
              onChange={(event) => {
                setPresetId(event.target.value);
                const next = BYOK_PRESETS.find((p) => p.id === event.target.value);
                if (next) setModel(next.model);
              }}
            >
              {BYOK_PRESETS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label} — {option.hint}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-sm font-medium">API key</span>
            <input
              type="password"
              autoComplete="off"
              spellCheck={false}
              className="mt-1 w-full rounded border px-2 py-1.5 font-mono text-sm"
              style={{ background: "var(--surface)", borderColor: "var(--border-strong)" }}
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="Paste your key"
            />
          </label>

          <label className="block">
            <span className="text-sm font-medium">Model</span>
            <input
              type="text"
              spellCheck={false}
              className="mt-1 w-full rounded border px-2 py-1.5 font-mono text-sm"
              style={{ background: "var(--surface)", borderColor: "var(--border-strong)" }}
              value={model}
              onChange={(event) => setModel(event.target.value)}
            />
          </label>
        </div>

        <div
          className="mt-4 rounded p-3 text-xs muted"
          style={{ background: "var(--surface-sunken)" }}
        >
          <p className="font-medium" style={{ color: "var(--text)" }}>
            Where this key goes
          </p>
          <ul className="mt-1 list-disc space-y-1 pl-4">
            <li>Stored in this browser&apos;s localStorage, on this device only.</li>
            <li>
              Attached to each humanize request, used for that request, and never
              written to a database or a log.
            </li>
            <li>
              Rewriting sends your text to the provider you choose here. Detection
              does not — it never leaves this app.
            </li>
          </ul>
        </div>

        <button className="btn btn-primary mt-4 w-full" onClick={save} disabled={!apiKey.trim()}>
          Save key
        </button>
      </div>
    </div>
  );
}
