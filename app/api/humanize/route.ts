/**
 * POST /api/humanize — streaming humanize orchestrator.
 *
 * Emits Server-Sent Events so the UI can show the pass counter, the live
 * score and each sentence as it is rewritten (PRD 16.2, "Humanizing" state)
 * rather than a spinner over a 40-second request.
 *
 * The BYOK key, when present, arrives on this request, is used for its
 * duration, and is never written to storage or a log (P4).
 */

import { NextRequest } from "next/server";

import { availableProviders, recordRateLimit, recordUsage } from "@/lib/budget";
import { countWords } from "@/lib/chunk";
import { estimateHumanizeTokens, MAX_WORDS } from "@/lib/detect";
import { humanize, MAX_PASSES } from "@/lib/humanize";
import { byokProvider, ProviderError, type ByokConfig } from "@/lib/providers";
import { checkRateLimit, clientKey } from "@/lib/ratelimit";
import type { HumanizeEvent } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 300;
export const dynamic = "force-dynamic";

interface HumanizeBody {
  text?: string;
  maxPasses?: number;
  byok?: Partial<ByokConfig>;
}

function sse(event: HumanizeEvent): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

function errorStream(message: string, status: number): Response {
  return new Response(sse({ type: "error", message }), {
    status,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

function parseByok(raw: Partial<ByokConfig> | undefined): ByokConfig | null {
  if (!raw?.apiKey || !raw?.model) return null;
  const flavour = raw.flavour === "gemini" ? "gemini" : "openai";
  return {
    flavour,
    apiKey: raw.apiKey,
    model: raw.model,
    baseUrl: typeof raw.baseUrl === "string" ? raw.baseUrl : undefined,
  };
}

export async function POST(request: NextRequest): Promise<Response> {
  let body: HumanizeBody;
  try {
    body = (await request.json()) as HumanizeBody;
  } catch {
    return errorStream("Request body is not valid JSON.", 400);
  }

  const text = typeof body.text === "string" ? body.text : "";
  if (!text.trim()) {
    return errorStream("There is nothing to humanize.", 400);
  }

  const words = countWords(text);
  if (words > MAX_WORDS) {
    return errorStream(
      `That is ${words.toLocaleString()} words. The limit is ` +
        `${MAX_WORDS.toLocaleString()}. Split it into sections and run them separately.`,
      413,
    );
  }

  const byok = parseByok(body.byok);

  // Own-key requests spend the user's quota, so neither the rate limiter nor
  // the pool budget applies to them.
  if (!byok) {
    const verdict = checkRateLimit(clientKey(request.headers));
    if (!verdict.allowed) {
      const minutes = Math.max(1, Math.ceil((verdict.resetsAt - Date.now()) / 60_000));
      return errorStream(
        `You have used this hour's share of the free pool. It resets in about ` +
          `${minutes} minute${minutes === 1 ? "" : "s"}. Adding your own API key ` +
          `in settings removes this limit entirely.`,
        429,
      );
    }
  }

  const providers = byok ? [byokProvider(byok)] : availableProviders();
  if (providers.length === 0) {
    return errorStream(
      "Every free provider is out of allowance for today, and no key of yours " +
        "is configured. Detection still works — it runs locally and costs " +
        "nothing. Add your own key in settings to keep rewriting.",
      503,
    );
  }

  if (!byok) {
    const estimate = estimateHumanizeTokens(text, body.maxPasses ?? MAX_PASSES);
    const headroom = providers.reduce((n, p) => n + p.limits.tpd, 0);
    if (estimate > headroom) {
      return errorStream(
        `This document needs roughly ${estimate.toLocaleString()} tokens and the ` +
          `pool has less than that left today. Add your own key in settings, or ` +
          `try again after the daily reset.`,
        503,
      );
    }
  }

  const origin = request.nextUrl.origin;
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let closed = false;
      const send = (event: HumanizeEvent) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(sse(event)));
        } catch {
          closed = true;
        }
      };

      try {
        await humanize(text, {
          baseUrl: origin,
          providers,
          maxPasses: body.maxPasses,
          signal: request.signal,
          onEvent: send,
          onUsage: (providerId, tokens) => recordUsage(providerId, tokens),
        });
      } catch (err) {
        if (err instanceof ProviderError && err.kind === "rate_limit") {
          recordRateLimit(err.provider, err.retryAfterMs);
          send({
            type: "error",
            message:
              "Every provider is rate-limited right now. Your own API key " +
              "removes this ceiling — see settings.",
          });
        } else {
          send({ type: "error", message: String((err as Error).message) });
        }
      } finally {
        if (!closed) controller.close();
        closed = true;
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-store",
      Connection: "keep-alive",
      // Vercel's edge buffers by default; this keeps events flowing live.
      "X-Accel-Buffering": "no",
    },
  });
}
