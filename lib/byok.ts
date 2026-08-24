/**
 * Bring-your-own-key handling (PRD 11.3, P4).
 *
 * The key lives in localStorage, is attached client-side to the humanize
 * request, and is never persisted or logged server-side. It is held in memory
 * for the duration of one request and discarded.
 */

import type { ByokConfig } from "./providers";

const STORAGE_KEY = "aidh.byok.v1";

export interface ByokSettings extends ByokConfig {
  enabled: boolean;
}

export const BYOK_PRESETS: {
  id: string;
  label: string;
  flavour: "openai" | "gemini";
  baseUrl?: string;
  model: string;
  hint: string;
}[] = [
  {
    id: "cerebras",
    label: "Cerebras",
    flavour: "openai",
    baseUrl: "https://api.cerebras.ai/v1",
    model: "llama-3.3-70b",
    hint: "cloud.cerebras.ai — free, no card",
  },
  {
    id: "groq",
    label: "Groq",
    flavour: "openai",
    baseUrl: "https://api.groq.com/openai/v1",
    model: "openai/gpt-oss-120b",
    hint: "console.groq.com — free, no card",
  },
  {
    id: "gemini",
    label: "Google AI Studio",
    flavour: "gemini",
    model: "gemini-2.0-flash",
    hint: "aistudio.google.com — free, no card",
  },
  {
    id: "openai",
    label: "OpenAI",
    flavour: "openai",
    baseUrl: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    hint: "platform.openai.com — paid",
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    flavour: "openai",
    baseUrl: "https://openrouter.ai/api/v1",
    model: "meta-llama/llama-3.3-70b-instruct",
    hint: "openrouter.ai — has a free tier",
  },
];

export function loadByok(): ByokSettings | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<ByokSettings>;
    if (!parsed.apiKey || !parsed.model || !parsed.flavour) return null;
    return {
      enabled: parsed.enabled ?? true,
      apiKey: parsed.apiKey,
      model: parsed.model,
      flavour: parsed.flavour,
      baseUrl: parsed.baseUrl,
    };
  } catch {
    return null;
  }
}

export function saveByok(settings: ByokSettings): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function clearByok(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}

/** Show enough of a key to recognise it, never enough to use it. */
export function maskKey(key: string): string {
  if (key.length <= 10) return "•".repeat(key.length);
  return `${key.slice(0, 4)}${"•".repeat(Math.min(20, key.length - 8))}${key.slice(-4)}`;
}
