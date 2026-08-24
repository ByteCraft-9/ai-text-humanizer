/**
 * Provider pool and failover router (PRD 11).
 *
 * Cerebras leads on daily volume (1M tokens), Groq on requests per minute
 * (30 RPM), Gemini backs both up. The router prefers Cerebras for bulk work
 * and fails over on rate-limit. Adding a fourth provider is a config change
 * here, not a refactor anywhere else (mitigates R7).
 */

import type { ProviderId } from "./types";

export interface ChatMessage {
  role: "system" | "user";
  content: string;
}

export interface CompletionRequest {
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  /** Ask the provider for strict JSON where it supports it. */
  json?: boolean;
  signal?: AbortSignal;
}

export interface CompletionResult {
  text: string;
  provider: ProviderId;
  model: string;
  promptTokens: number;
  completionTokens: number;
}

export class ProviderError extends Error {
  constructor(
    message: string,
    readonly provider: ProviderId,
    readonly kind: "rate_limit" | "auth" | "server" | "network" | "bad_response",
    readonly retryAfterMs?: number,
  ) {
    super(message);
    this.name = "ProviderError";
  }
}

export interface ProviderLimits {
  /** Tokens per day. */
  tpd: number;
  /** Requests per minute. */
  rpm: number;
  /** Tokens per minute, where the provider enforces one. */
  tpm?: number;
  /** Requests per day, where the provider enforces one. */
  rpd?: number;
}

export interface LLMProvider {
  readonly id: ProviderId;
  readonly label: string;
  readonly model: string;
  readonly limits: ProviderLimits;
  /** Whether a key is configured for this provider in this environment. */
  configured(): boolean;
  complete(req: CompletionRequest): Promise<CompletionResult>;
}

// ---------------------------------------------------------------------------
// OpenAI-compatible providers (Cerebras, Groq, and most BYOK endpoints)
// ---------------------------------------------------------------------------

interface OpenAICompatConfig {
  id: ProviderId;
  label: string;
  model: string;
  baseUrl: string;
  apiKey: () => string | undefined;
  limits: ProviderLimits;
}

class OpenAICompatProvider implements LLMProvider {
  constructor(private readonly cfg: OpenAICompatConfig) {}

  get id() { return this.cfg.id; }
  get label() { return this.cfg.label; }
  get model() { return this.cfg.model; }
  get limits() { return this.cfg.limits; }

  configured() { return Boolean(this.cfg.apiKey()); }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const key = this.cfg.apiKey();
    if (!key) throw new ProviderError("No API key configured", this.id, "auth");

    let res: Response;
    try {
      res = await fetch(`${this.cfg.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${key}`,
        },
        signal: req.signal,
        body: JSON.stringify({
          model: this.cfg.model,
          messages: req.messages,
          max_tokens: req.maxTokens ?? 1024,
          temperature: req.temperature ?? 0.9,
          ...(req.json ? { response_format: { type: "json_object" } } : {}),
        }),
      });
    } catch (err) {
      throw new ProviderError(String((err as Error).message), this.id, "network");
    }

    if (!res.ok) throw await httpToProviderError(res, this.id);

    const data = await res.json().catch(() => null);
    const text = data?.choices?.[0]?.message?.content;
    if (typeof text !== "string") {
      throw new ProviderError("Malformed completion response", this.id, "bad_response");
    }

    return {
      text,
      provider: this.id,
      model: this.cfg.model,
      promptTokens: data?.usage?.prompt_tokens ?? estimateMessageTokens(req.messages),
      completionTokens: data?.usage?.completion_tokens ?? Math.ceil(text.length / 4),
    };
  }
}

// ---------------------------------------------------------------------------
// Google AI Studio (Gemini) — different wire format, same interface
// ---------------------------------------------------------------------------

interface GeminiConfig {
  id: ProviderId;
  label: string;
  model: string;
  apiKey: () => string | undefined;
  limits: ProviderLimits;
}

class GeminiProvider implements LLMProvider {
  constructor(private readonly cfg: GeminiConfig) {}

  get id() { return this.cfg.id; }
  get label() { return this.cfg.label; }
  get model() { return this.cfg.model; }
  get limits() { return this.cfg.limits; }

  configured() { return Boolean(this.cfg.apiKey()); }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const key = this.cfg.apiKey();
    if (!key) throw new ProviderError("No API key configured", this.id, "auth");

    const system = req.messages
      .filter((m) => m.role === "system")
      .map((m) => m.content)
      .join("\n");
    const user = req.messages
      .filter((m) => m.role !== "system")
      .map((m) => m.content)
      .join("\n");

    const url =
      "https://generativelanguage.googleapis.com/v1beta/models/" +
      `${this.cfg.model}:generateContent`;

    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": key },
        signal: req.signal,
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: user }] }],
          ...(system ? { systemInstruction: { parts: [{ text: system }] } } : {}),
          generationConfig: {
            maxOutputTokens: req.maxTokens ?? 1024,
            temperature: req.temperature ?? 0.9,
            ...(req.json ? { responseMimeType: "application/json" } : {}),
          },
        }),
      });
    } catch (err) {
      throw new ProviderError(String((err as Error).message), this.id, "network");
    }

    if (!res.ok) throw await httpToProviderError(res, this.id);

    const data = await res.json().catch(() => null);
    const text: string = (data?.candidates?.[0]?.content?.parts ?? [])
      .map((p: { text?: string }) => p.text ?? "")
      .join("");
    if (!text) {
      throw new ProviderError("Malformed completion response", this.id, "bad_response");
    }

    return {
      text,
      provider: this.id,
      model: this.cfg.model,
      promptTokens:
        data?.usageMetadata?.promptTokenCount ?? estimateMessageTokens(req.messages),
      completionTokens:
        data?.usageMetadata?.candidatesTokenCount ?? Math.ceil(text.length / 4),
    };
  }
}

// ---------------------------------------------------------------------------
// BYOK — the user's own key, attached per request (PRD 11.3)
// ---------------------------------------------------------------------------

export interface ByokConfig {
  /** Which wire format the user's endpoint speaks. */
  flavour: "openai" | "gemini";
  apiKey: string;
  baseUrl?: string;
  model: string;
}

/** The user's own quota applies, so we do not meter BYOK against our pool. */
const UNMETERED: ProviderLimits = {
  tpd: Number.POSITIVE_INFINITY,
  rpm: Number.POSITIVE_INFINITY,
};

/**
 * Build a provider from a user-supplied key. The key arrives on the request
 * and is used only for that request — never persisted, never logged (P4).
 */
export function byokProvider(cfg: ByokConfig): LLMProvider {
  if (cfg.flavour === "gemini") {
    return new GeminiProvider({
      id: "byok",
      label: "Your key",
      model: cfg.model,
      apiKey: () => cfg.apiKey,
      limits: UNMETERED,
    });
  }

  return new OpenAICompatProvider({
    id: "byok",
    label: "Your key",
    model: cfg.model,
    baseUrl: (cfg.baseUrl || "https://api.openai.com/v1").replace(/\/+$/, ""),
    apiKey: () => cfg.apiKey,
    limits: UNMETERED,
  });
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

function cerebras(): LLMProvider {
  return new OpenAICompatProvider({
    id: "cerebras",
    label: "Cerebras",
    model: process.env.CEREBRAS_MODEL || "llama-3.3-70b",
    baseUrl: "https://api.cerebras.ai/v1",
    apiKey: () => process.env.CEREBRAS_API_KEY,
    limits: { tpd: 1_000_000, rpm: 5 },
  });
}

function groq(): LLMProvider {
  return new OpenAICompatProvider({
    id: "groq",
    label: "Groq",
    model: process.env.GROQ_MODEL || "openai/gpt-oss-120b",
    baseUrl: "https://api.groq.com/openai/v1",
    apiKey: () => process.env.GROQ_API_KEY,
    limits: { tpd: 200_000, rpm: 30, tpm: 8_000, rpd: 1_000 },
  });
}

function gemini(): LLMProvider {
  return new GeminiProvider({
    id: "gemini",
    label: "Google AI Studio",
    model: process.env.GEMINI_MODEL || "gemini-2.0-flash",
    apiKey: () => process.env.GEMINI_API_KEY,
    // Gemini free tier meters requests, not tokens; the RPD is the real
    // ceiling. The token figure is a generous notional cap for the meter.
    limits: { tpd: 1_000_000, rpm: 10, rpd: 1_500 },
  });
}

/** Priority order per PRD 11.1: bulk volume first, then RPM, then backstop. */
export function providerPool(): LLMProvider[] {
  return [cerebras(), groq(), gemini()];
}

export function configuredPool(): LLMProvider[] {
  return providerPool().filter((p) => p.configured());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function httpToProviderError(res: Response, id: ProviderId): Promise<ProviderError> {
  const body = await res.text().catch(() => "");
  const snippet = body.slice(0, 300);

  if (res.status === 429) {
    const header = res.headers.get("retry-after");
    const retryAfterMs = header ? Number(header) * 1000 : undefined;
    return new ProviderError(`Rate limited: ${snippet}`, id, "rate_limit", retryAfterMs);
  }
  if (res.status === 401 || res.status === 403) {
    return new ProviderError(`Authentication failed: ${snippet}`, id, "auth");
  }
  return new ProviderError(`HTTP ${res.status}: ${snippet}`, id, "server");
}

function estimateMessageTokens(messages: ChatMessage[]): number {
  return Math.ceil(messages.reduce((n, m) => n + m.content.length, 0) / 4);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run `fn` against each provider in turn, moving on when one is rate-limited,
 * unauthenticated or erroring. Exponential backoff applies within a provider
 * before giving up on it (PRD 16.3).
 */
export async function withFailover<T>(
  providers: LLMProvider[],
  fn: (p: LLMProvider) => Promise<T>,
  opts: {
    retriesPerProvider?: number;
    onAttempt?: (p: LLMProvider, attempt: number) => void;
  } = {},
): Promise<T> {
  const { retriesPerProvider = 2, onAttempt } = opts;
  const failures: ProviderError[] = [];

  for (const provider of providers) {
    for (let attempt = 0; attempt <= retriesPerProvider; attempt++) {
      try {
        onAttempt?.(provider, attempt);
        return await fn(provider);
      } catch (err) {
        const pe =
          err instanceof ProviderError
            ? err
            : new ProviderError(String((err as Error).message), provider.id, "server");
        failures.push(pe);

        // Auth failures and malformed responses will not fix themselves.
        if (pe.kind === "auth" || pe.kind === "bad_response") break;
        if (attempt === retriesPerProvider) break;

        await sleep(pe.retryAfterMs ?? Math.min(8_000, 500 * 2 ** attempt));
      }
    }
  }

  const detail = failures.map((f) => `${f.provider}: ${f.message}`).join(" | ");
  throw new ProviderError(
    failures.length
      ? `All providers failed. ${detail}`
      : "No LLM provider is configured. Add your own key in settings, or set " +
        "CEREBRAS_API_KEY, GROQ_API_KEY or GEMINI_API_KEY.",
    failures[0]?.provider ?? "cerebras",
    failures.length > 0 && failures.every((f) => f.kind === "rate_limit")
      ? "rate_limit"
      : "server",
  );
}
