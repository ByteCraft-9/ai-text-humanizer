/**
 * GET /api/budget — remaining pooled free-tier tokens (PRD 11.4).
 *
 * Shown before the user commits a document so nobody hits a wall mid-run.
 * The figure is a conservative per-instance estimate, not a guarantee — see
 * the note in lib/budget.ts — and the UI labels it as such.
 */

import { budgetSnapshot } from "@/lib/budget";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  const byokActive = new URL(request.url).searchParams.get("byok") === "1";

  return Response.json(budgetSnapshot(byokActive), {
    headers: { "Cache-Control": "no-store" },
  });
}
