"use client";

/**
 * Remaining pooled free-tier tokens (PRD 11.4).
 *
 * Shown before the user commits a document so nobody hits a wall mid-run.
 * Below 10% of the pool the meter offers BYOK rather than letting a long
 * document start and fail halfway.
 */

import { useEffect, useState } from "react";

import type { BudgetSnapshot } from "@/lib/types";

interface Props {
  byokActive: boolean;
  onUseOwnKey: () => void;
  /** Bumped by the parent after a run so the meter refreshes. */
  refreshKey?: number;
}

function formatTokens(n: number): string {
  if (!Number.isFinite(n)) return "unlimited";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
  return String(n);
}

function formatReset(epochMs: number): string {
  const minutes = Math.max(0, Math.round((epochMs - Date.now()) / 60_000));
  if (minutes < 60) return `${minutes} min`;
  return `${Math.round(minutes / 60)} h`;
}

export function BudgetMeter({ byokActive, onUseOwnKey, refreshKey = 0 }: Props) {
  const [snapshot, setSnapshot] = useState<BudgetSnapshot | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/budget?byok=${byokActive ? "1" : "0"}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: BudgetSnapshot) => !cancelled && setSnapshot(data))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [byokActive, refreshKey]);

  if (byokActive) {
    return (
      <div className="surface p-3 text-sm">
        <p className="font-medium">Running on your own key</p>
        <p className="mt-1 text-xs muted">
          No shared limit applies. Your key stays in this browser and is never
          stored or logged by the server.
        </p>
      </div>
    );
  }

  if (failed) {
    return (
      <div className="surface p-3 text-xs muted">
        The budget meter is unavailable. Detection still works — it runs locally
        and costs nothing.
      </div>
    );
  }

  if (!snapshot) {
    return <div className="surface h-[5.5rem] animate-pulse p-3" aria-hidden="true" />;
  }

  const configured = snapshot.providers.filter((p) => !p.reason?.startsWith("No API key"));
  const fraction =
    snapshot.pooled_limit > 0 ? snapshot.pooled_remaining / snapshot.pooled_limit : 0;
  const thin = fraction < 0.1;

  if (configured.length === 0) {
    return (
      <div className="surface p-3 text-sm">
        <p className="font-medium">No shared pool on this deployment</p>
        <p className="mt-1 text-xs muted">
          Rewriting needs an API key. Detection runs locally and works without one.
        </p>
        <button className="btn mt-2 w-full text-xs" onClick={onUseOwnKey}>
          Add your own key
        </button>
      </div>
    );
  }

  return (
    <div className="surface p-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium">Shared free-tier pool</h3>
        <span className="text-sm tabular-nums">
          {formatTokens(snapshot.pooled_remaining)} left
        </span>
      </div>

      <div
        className="mt-2 h-2 w-full overflow-hidden rounded-full"
        style={{ background: "var(--surface-sunken)" }}
        role="meter"
        aria-valuenow={Math.round(fraction * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Remaining shared token budget"
      >
        <div
          className="h-full rounded-full transition-[width]"
          style={{
            width: `${Math.max(2, fraction * 100)}%`,
            background: thin ? "var(--warn)" : "var(--accent)",
          }}
        />
      </div>

      <ul className="mt-2 space-y-1 text-xs muted">
        {configured.map((provider) => (
          <li key={provider.id} className="flex items-center justify-between gap-2">
            <span>{provider.label}</span>
            <span className="tabular-nums">
              {provider.available
                ? `${formatTokens(provider.remaining)} · resets in ${formatReset(provider.resets_at)}`
                : provider.reason}
            </span>
          </li>
        ))}
      </ul>

      <p className="mt-2 text-xs muted">
        An estimate from this server instance, not a guarantee — the real limit
        is whatever the provider enforces.
      </p>

      {thin && (
        <div
          className="mt-2 rounded p-2 text-xs"
          style={{ background: "var(--warn-soft)", color: "var(--warn)" }}
        >
          <p className="font-medium">The pool is nearly empty.</p>
          <p className="mt-1">
            Long documents may not finish. Your own key removes the ceiling.
          </p>
          <button className="btn mt-2 w-full text-xs" onClick={onUseOwnKey}>
            Add your own key
          </button>
        </div>
      )}
    </div>
  );
}
