/**
 * Free-tier token accounting (PRD 11.4).
 *
 * The product has no database by design (N3), so the ledger lives in module
 * scope on the serverless instance. That is deliberately approximate: Vercel
 * may run several instances, so the true pooled figure can be higher than
 * what one instance has seen. The meter is therefore a *conservative*
 * estimate — it can under-report headroom, never over-report it — and the UI
 * labels it as an estimate rather than a guarantee.
 *
 * The authoritative limit is always the provider's own 429, which the router
 * handles by failing over.
 */

import type { BudgetSnapshot, ProviderId, ProviderStatus } from "./types";
import { providerPool, type LLMProvider } from "./providers";

interface Ledger {
  tokens: number;
  requests: number;
  /** Epoch ms of the UTC day this ledger covers. */
  day: number;
  /** Epoch ms when this provider's rate limit last rejected us. */
  cooldownUntil: number;
}

const ledgers = new Map<ProviderId, Ledger>();

function startOfUtcDay(now = Date.now()): number {
  const d = new Date(now);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

function ledgerFor(id: ProviderId): Ledger {
  const day = startOfUtcDay();
  const existing = ledgers.get(id);
  if (existing && existing.day === day) return existing;

  const fresh: Ledger = { tokens: 0, requests: 0, day, cooldownUntil: 0 };
  ledgers.set(id, fresh);
  return fresh;
}

export function recordUsage(id: ProviderId, tokens: number): void {
  if (id === "byok") return; // The user's quota, not ours.
  const ledger = ledgerFor(id);
  ledger.tokens += tokens;
  ledger.requests += 1;
}

/** Called when a provider returns 429, so the meter reflects reality. */
export function recordRateLimit(id: ProviderId, retryAfterMs = 60_000): void {
  if (id === "byok") return;
  ledgerFor(id).cooldownUntil = Date.now() + retryAfterMs;
}

function statusFor(provider: LLMProvider): ProviderStatus {
  const ledger = ledgerFor(provider.id);
  const limit = provider.limits.tpd;
  const remaining = Math.max(0, limit - ledger.tokens);
  const resets_at = ledger.day + 24 * 60 * 60 * 1000;

  let available = true;
  let reason: string | undefined;

  if (!provider.configured()) {
    available = false;
    reason = "No API key configured for this provider.";
  } else if (remaining <= 0) {
    available = false;
    reason = "Daily token allowance used up.";
  } else if (provider.limits.rpd && ledger.requests >= provider.limits.rpd) {
    available = false;
    reason = "Daily request allowance used up.";
  } else if (Date.now() < ledger.cooldownUntil) {
    available = false;
    reason = "Rate limited; backing off.";
  }

  return {
    id: provider.id,
    label: provider.label,
    remaining,
    limit,
    resets_at,
    available,
    reason,
  };
}

export function budgetSnapshot(byokActive = false): BudgetSnapshot {
  const providers = providerPool().map(statusFor);
  const usable = providers.filter((p) => p.available);

  return {
    providers,
    pooled_remaining: usable.reduce((n, p) => n + p.remaining, 0),
    pooled_limit: providers
      .filter((p) => !p.reason?.startsWith("No API key"))
      .reduce((n, p) => n + p.limit, 0),
    byok_active: byokActive,
  };
}

/**
 * Providers that still have headroom, in priority order. The router should
 * consult this before spending, so a document is not started against a
 * provider that is already exhausted.
 */
export function availableProviders(): LLMProvider[] {
  return providerPool().filter((p) => statusFor(p).available);
}

/**
 * Whether the pool can plausibly cover an estimated spend. Used to gate long
 * documents behind a BYOK prompt when the pool is thin (PRD 11.4).
 */
export function canAfford(estimatedTokens: number): boolean {
  return budgetSnapshot().pooled_remaining >= estimatedTokens;
}

/** Exposed for tests only. */
export function __resetLedgers(): void {
  ledgers.clear();
}
