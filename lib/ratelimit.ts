/**
 * Per-IP rate limiting on the humanize endpoint (PRD 15.2).
 *
 * One user must not be able to drain the shared free-tier pool. Like the
 * budget ledger this is in-memory and therefore per-instance — imperfect, but
 * the alternative is a database, which costs money and stores request
 * metadata the product has promised not to keep (P5).
 *
 * Requests carrying the user's own key skip the limiter entirely: they are
 * spending their own quota, so there is nothing to protect.
 */

const WINDOW_MS = 60 * 60 * 1000;
const MAX_DOCUMENTS_PER_WINDOW = 12;

interface Bucket {
  timestamps: number[];
}

const buckets = new Map<string, Bucket>();

export interface RateLimitVerdict {
  allowed: boolean;
  remaining: number;
  /** Epoch ms when the oldest request in the window expires. */
  resetsAt: number;
}

/**
 * Vercel sets `x-forwarded-for`; the left-most entry is the client. Falls back
 * to a shared key so a missing header fails closed into one pool rather than
 * open into unlimited individual ones.
 */
export function clientKey(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0].trim();
  return headers.get("x-real-ip") ?? "unknown";
}

export function checkRateLimit(key: string): RateLimitVerdict {
  const now = Date.now();
  const bucket = buckets.get(key) ?? { timestamps: [] };

  bucket.timestamps = bucket.timestamps.filter((t) => now - t < WINDOW_MS);

  if (bucket.timestamps.length >= MAX_DOCUMENTS_PER_WINDOW) {
    buckets.set(key, bucket);
    return {
      allowed: false,
      remaining: 0,
      resetsAt: bucket.timestamps[0] + WINDOW_MS,
    };
  }

  bucket.timestamps.push(now);
  buckets.set(key, bucket);

  // Opportunistic sweep so an instance that lives a long time does not grow
  // a bucket per IP forever.
  if (buckets.size > 5_000) {
    for (const [k, v] of buckets) {
      if (v.timestamps.every((t) => now - t >= WINDOW_MS)) buckets.delete(k);
    }
  }

  return {
    allowed: true,
    remaining: MAX_DOCUMENTS_PER_WINDOW - bucket.timestamps.length,
    resetsAt: bucket.timestamps[0] + WINDOW_MS,
  };
}

export { MAX_DOCUMENTS_PER_WINDOW, WINDOW_MS };
