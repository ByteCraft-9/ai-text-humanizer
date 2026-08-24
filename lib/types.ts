/**
 * Shared contract between the TypeScript app and the Python inference
 * functions. Mirrors PRD 8.8. Keep in sync with api/_lib/schema.py.
 */

export type Confidence = "high" | "medium" | "low";

/** Individual ensemble signals, each already mapped to a P(AI) in [0,1]. */
export interface Signals {
  classifier: number;
  features: number;
  binoculars_ratio: number;
  burstiness: number;
}

export interface SentenceScore {
  index: number;
  text: string;
  /** P(AI) from the model this result came from. */
  score: number;
  /** Character offsets into the document this sentence was scored in. */
  start: number;
  end: number;
}

export interface DetectMeta {
  words: number;
  chunks: number;
  model_version: string;
  ms: number;
  /** True when the trained ONNX weights were unavailable and the
   *  statistical-only fallback produced this score. Surfaced in the UI —
   *  we never present a degraded score as a full one. */
  degraded?: boolean;
}

export interface DetectResult {
  /** Model A — hardened against humanization. Never the humanizer's target. */
  strict_score: number;
  /** Model B panel — the surrogate for third-party detectors. */
  surrogate_score: number;
  confidence: Confidence;
  confidence_interval: [number, number];
  signals: Signals;
  sentences: SentenceScore[];
  meta: DetectMeta;
}

export interface DetectRequest {
  text: string;
  /** When false, skip the strict model (humanizer inner loop only needs B). */
  include_strict?: boolean;
}

/** Response from api/py/perplexity. */
export interface PerplexityResult {
  log_perplexity: number;
  cross_perplexity: number;
  /** Binoculars score: log_ppl / cross_ppl. Lower => more machine-like. */
  binoculars: number;
  burstiness: number;
  sentence_perplexity: number[];
  meta: { tokens: number; ms: number };
}

// ---------------------------------------------------------------------------
// Humanizer
// ---------------------------------------------------------------------------

export type ProviderId = "cerebras" | "groq" | "gemini" | "byok";

export interface ProviderStatus {
  id: ProviderId;
  label: string;
  /** Tokens still available today, best-effort from local accounting. */
  remaining: number;
  limit: number;
  /** Epoch ms when this provider's daily window resets. */
  resets_at: number;
  available: boolean;
  reason?: string;
}

export interface BudgetSnapshot {
  providers: ProviderStatus[];
  pooled_remaining: number;
  pooled_limit: number;
  /** Set when the user supplied their own key — the ceiling is then theirs. */
  byok_active: boolean;
}

export interface SentenceRewrite {
  index: number;
  original: string;
  rewritten: string;
  score_before: number;
  score_after: number;
  similarity: number;
  /** Why a rewrite was not applied, when rewritten === original. */
  rejected?: "similarity" | "no_improvement" | "generation_failed";
}

export interface HumanizePass {
  pass: number;
  targeted: number;
  applied: number;
  rewrites: SentenceRewrite[];
  strict_score: number;
  surrogate_score: number;
  tokens_used: number;
}

export type HumanizeOutcome =
  | "cleared"       // surrogate fell below threshold
  | "plateaued"     // passes stopped improving (PRD H4)
  | "max_passes"
  | "budget_exhausted"
  | "error";

export interface HumanizeResult {
  text: string;
  original_text: string;
  outcome: HumanizeOutcome;
  passes: HumanizePass[];
  strict_score: number;
  surrogate_score: number;
  initial_strict_score: number;
  initial_surrogate_score: number;
  /** Mean similarity of every applied rewrite (PRD A8 >= 0.85). */
  mean_similarity: number;
  /**
   * Which meaning check ran. "lexical" means the MiniLM encoder was
   * unavailable and a content-word proxy was used at a lower threshold — the
   * UI must say so rather than implying A8 was enforced as specified.
   */
  similarity_mode: "semantic" | "lexical";
  tokens_used: number;
  provider_used: ProviderId | null;
  message?: string;
}

/** Server-Sent Event frames emitted by /app/api/humanize. */
export type HumanizeEvent =
  | { type: "status"; phase: string; detail?: string }
  | { type: "pass_start"; pass: number; targeted: number }
  | { type: "sentence"; rewrite: SentenceRewrite }
  | { type: "pass_end"; pass: HumanizePass }
  | { type: "done"; result: HumanizeResult }
  | { type: "error"; message: string };
