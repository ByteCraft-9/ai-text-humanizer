"use client";

/**
 * The primary workspace (PRD 16.1): one screen, three regions — input,
 * results, action.
 *
 * State machine follows PRD 16.2: empty → parsing → detecting → detected →
 * humanizing → (cleared | plateaued | budget exhausted).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { BudgetMeter } from "@/components/BudgetMeter";
import { ByokPanel } from "@/components/ByokPanel";
import { DiffView } from "@/components/DiffView";
import { Editor } from "@/components/Editor";
import { Heatmap } from "@/components/Heatmap";
import {
  DegradedNotice,
  EvidenceNotice,
  OutcomeNotice,
  PrivacyDisclosure,
  StandingNotice,
} from "@/components/Notices";
import { ScoreGauge, SignalBreakdown } from "@/components/ScoreGauge";
import { loadByok, type ByokSettings } from "@/lib/byok";
import { countWords } from "@/lib/chunk";
import { detect, MAX_WORDS } from "@/lib/detect";
import { copyToClipboard, downloadDocx, downloadText } from "@/lib/export";
import { CLEARED_THRESHOLD } from "@/lib/humanize";
import type { DetectResult, HumanizeEvent, HumanizeResult } from "@/lib/types";

type Phase = "empty" | "detecting" | "detected" | "humanizing" | "done";

export default function Workspace() {
  const [text, setText] = useState("");
  const [phase, setPhase] = useState<Phase>("empty");
  const [detection, setDetection] = useState<DetectResult | null>(null);
  const [humanized, setHumanized] = useState<HumanizeResult | null>(null);
  /** Fresh heatmap for the rewritten text — the offsets moved. */
  const [rewrittenDetection, setRewrittenDetection] = useState<DetectResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [liveStatus, setLiveStatus] = useState("");
  const [byok, setByok] = useState<ByokSettings | null>(null);
  const [byokOpen, setByokOpen] = useState(false);
  const [budgetKey, setBudgetKey] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setByok(loadByok());
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const words = countWords(text);
  const busy = phase === "detecting" || phase === "humanizing";

  const runDetect = useCallback(async () => {
    if (!text.trim() || words > MAX_WORDS) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("detecting");
    setError(null);
    setHumanized(null);
    setRewrittenDetection(null);
    setDetection(null);
    setShowOriginal(false);
    setLiveStatus("Analysing the document.");

    try {
      const result = await detect(text, {
        signal: controller.signal,
        onProgress: (done, total) => setProgress({ done, total }),
      });
      setDetection(result);
      setPhase("detected");
      setLiveStatus(
        `Analysis complete. Strict score ${Math.round(result.strict_score * 100)} percent AI, ` +
          `estimated third-party score ${Math.round(result.surrogate_score * 100)} percent, ` +
          `${result.confidence} confidence.`,
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(String((err as Error).message));
      setPhase(detection ? "detected" : "empty");
      setLiveStatus("Analysis failed.");
    } finally {
      setProgress(null);
    }
  }, [text, words, detection]);

  const runHumanize = useCallback(async () => {
    if (!detection) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("humanizing");
    setError(null);
    setHumanized(null);
    setRewrittenDetection(null);
    setLiveStatus("Rewriting the flagged sentences.");

    try {
      const response = await fetch("/api/humanize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          text,
          byok: byok?.enabled
            ? {
                flavour: byok.flavour,
                apiKey: byok.apiKey,
                model: byok.model,
                baseUrl: byok.baseUrl,
              }
            : undefined,
        }),
      });

      if (!response.body) throw new Error("The server returned no response body.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;

          let event: HumanizeEvent;
          try {
            event = JSON.parse(line.slice(5).trim()) as HumanizeEvent;
          } catch {
            continue;
          }

          switch (event.type) {
            case "status":
              setLiveStatus(event.detail ?? event.phase);
              break;
            case "pass_start":
              setLiveStatus(`Pass ${event.pass}: rewriting ${event.targeted} sentences.`);
              break;
            case "pass_end":
              setLiveStatus(
                `Pass ${event.pass.pass} done. ${event.pass.applied} sentences changed. ` +
                  `Estimated third-party score now ${Math.round(event.pass.surrogate_score * 100)} percent.`,
              );
              break;
            case "done": {
              const finished = event.result;
              setHumanized(finished);
              setPhase("done");
              setLiveStatus(
                `Finished. Strict score ${Math.round(finished.strict_score * 100)} percent, ` +
                  `estimated third-party score ${Math.round(finished.surrogate_score * 100)} percent.`,
              );
              // Re-score the rewrite so the heatmap lands on the new offsets.
              // Detection is local and free, so this costs nothing.
              void detect(finished.text, { signal: controller.signal })
                .then(setRewrittenDetection)
                .catch(() => setRewrittenDetection(null));
              break;
            }
            case "error":
              setError(event.message);
              setPhase("detected");
              setLiveStatus("The rewrite failed.");
              break;
          }
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(String((err as Error).message));
        setPhase("detected");
      }
    } finally {
      setBudgetKey((n) => n + 1);
    }
  }, [detection, text, byok]);

  return (
    <main id="workspace" className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              AI Text Detector &amp; Humanizer
            </h1>
            <p className="mt-1 max-w-2xl text-sm muted">
              Two honest scores and per-sentence evidence. Detection runs
              locally and never leaves this app.
            </p>
          </div>
          <button className="btn text-xs" onClick={() => setByokOpen(true)}>
            {byok ? "Your key is active" : "Use your own key"}
          </button>
        </div>
        <div className="mt-3 max-w-3xl">
          <StandingNotice />
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="space-y-6">
          {/* ---------------- Input region ---------------- */}
          <section aria-labelledby="input-heading">
            <h2 id="input-heading" className="mb-3 text-sm font-semibold uppercase tracking-wide muted">
              1 · Your text
            </h2>
            <Editor value={text} onChange={setText} disabled={busy} />
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <button
                className="btn btn-primary"
                onClick={runDetect}
                disabled={busy || !text.trim() || words > MAX_WORDS}
              >
                {phase === "detecting" ? "Analysing…" : "Analyse this text"}
              </button>
              {progress && (
                <span className="text-xs tabular-nums muted">
                  chunk {progress.done} of {progress.total}
                </span>
              )}
              {busy && (
                <button className="btn text-xs" onClick={() => abortRef.current?.abort()}>
                  Stop
                </button>
              )}
            </div>
          </section>

          {error && (
            <p
              role="alert"
              className="rounded p-3 text-sm"
              style={{ background: "var(--warn-soft)", color: "var(--warn)" }}
            >
              {error}
            </p>
          )}

          {/* ---------------- Results region ---------------- */}
          {detection && (
            <section aria-labelledby="results-heading" className="space-y-4">
              <h2 id="results-heading" className="text-sm font-semibold uppercase tracking-wide muted">
                2 · What we found
              </h2>

              {detection.meta.degraded && <DegradedNotice />}

              <div className="grid gap-4 sm:grid-cols-2">
                <ScoreGauge
                  label="Strict score"
                  score={humanized ? humanized.strict_score : detection.strict_score}
                  previous={humanized ? humanized.initial_strict_score : undefined}
                  confidence={detection.confidence}
                  interval={detection.confidence_interval}
                  description="Our own detector, deliberately hardened against rewriting. The number to trust about what this text is."
                />
                <ScoreGauge
                  label="Estimated third-party score"
                  score={humanized ? humanized.surrogate_score : detection.surrogate_score}
                  previous={humanized ? humanized.initial_surrogate_score : undefined}
                  emphasis="secondary"
                  description="A surrogate for what GPTZero, Turnitin and Originality are likely to see. This is what the rewrite targets."
                />
              </div>

              <SignalBreakdown signals={detection.signals as unknown as Record<string, number>} />

              <div>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold">Sentence by sentence</h3>
                  {humanized && (
                    <div className="flex gap-1">
                      <button
                        className="btn px-2 py-1 text-xs"
                        style={!showOriginal ? { borderColor: "var(--accent)" } : undefined}
                        onClick={() => setShowOriginal(false)}
                      >
                        Rewritten
                      </button>
                      <button
                        className="btn px-2 py-1 text-xs"
                        style={showOriginal ? { borderColor: "var(--accent)" } : undefined}
                        onClick={() => setShowOriginal(true)}
                      >
                        Original
                      </button>
                    </div>
                  )}
                </div>
                <Heatmap
                  text={showOriginal || !humanized ? text : humanized.text}
                  sentences={
                    showOriginal || !humanized
                      ? detection.sentences
                      : (rewrittenDetection?.sentences ?? [])
                  }
                />
                {humanized && !showOriginal && !rewrittenDetection && (
                  <p className="mt-2 text-xs muted">
                    Re-scoring the rewrite to refresh the heatmap…
                  </p>
                )}
              </div>

              <EvidenceNotice />
            </section>
          )}

          {/* ---------------- Action region ---------------- */}
          {detection && (
            <section aria-labelledby="action-heading" className="space-y-4">
              <h2 id="action-heading" className="text-sm font-semibold uppercase tracking-wide muted">
                3 · Rewrite
              </h2>

              <PrivacyDisclosure byokActive={Boolean(byok)} />

              <div className="flex flex-wrap items-center gap-3">
                <button
                  className="btn btn-primary"
                  onClick={runHumanize}
                  disabled={busy || detection.surrogate_score < CLEARED_THRESHOLD}
                >
                  {phase === "humanizing" ? "Rewriting…" : "Humanize until our detector clears it"}
                </button>
                {detection.surrogate_score < CLEARED_THRESHOLD && (
                  <span className="text-xs muted">
                    Already below the threshold — there is nothing worth rewriting.
                  </span>
                )}
              </div>

              {humanized && <OutcomeNotice result={humanized} />}
              {humanized && <DiffView result={humanized} />}

              {humanized && (
                <div className="flex flex-wrap gap-2">
                  <button
                    className="btn text-xs"
                    onClick={() => void copyToClipboard(humanized.text)}
                  >
                    Copy the rewrite
                  </button>
                  <button className="btn text-xs" onClick={() => downloadText(humanized.text)}>
                    Download .txt
                  </button>
                  <button className="btn text-xs" onClick={() => downloadDocx(humanized.text)}>
                    Download .docx
                  </button>
                  <button
                    className="btn text-xs"
                    onClick={() => {
                      setText(humanized.text);
                      setHumanized(null);
                      setRewrittenDetection(null);
                      setDetection(null);
                      setPhase("empty");
                    }}
                  >
                    Move it into the editor
                  </button>
                </div>
              )}
            </section>
          )}

          {phase === "empty" && !detection && <EmptyState />}
        </div>

        {/* ---------------- Sidebar ---------------- */}
        <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          <BudgetMeter
            byokActive={Boolean(byok)}
            onUseOwnKey={() => setByokOpen(true)}
            refreshKey={budgetKey}
          />
          <div className="surface p-3 text-xs muted">
            <p className="font-medium" style={{ color: "var(--text)" }}>
              No account, no storage
            </p>
            <p className="mt-1">
              Nothing you paste is saved. There are no accounts, no analytics on
              document content, and no cookies beyond this session.
            </p>
          </div>
        </aside>
      </div>

      <ByokPanel
        open={byokOpen}
        onClose={() => setByokOpen(false)}
        onChange={(settings) => setByok(settings)}
      />

      {/* Screen-reader announcements for score changes (PRD 15.3). */}
      <p aria-live="polite" role="status" className="sr-only">
        {liveStatus}
      </p>

      <footer className="mt-12 border-t pt-6 text-xs muted" style={{ borderColor: "var(--border)" }}>
        <p>
          Detection runs on this app&apos;s own models. Rewriting uses whichever
          free provider has allowance left, or your own key. Nothing is stored.
        </p>
        <p className="mt-1">
          Scores are probabilities, not verdicts about a person. Read the
          confidence band before you act on one.
        </p>
      </footer>
    </main>
  );
}

/** PRD 16.2: explain the two scores before the user commits any text. */
function EmptyState() {
  return (
    <section className="surface p-5 text-sm">
      <h2 className="font-semibold">Two numbers, and why there are two</h2>
      <p className="mt-2 muted">
        Most tools in this category report a single figure and, after
        rewriting, a flattering 0%. That number is meaningless: the tool is
        grading its own homework.
      </p>
      <dl className="mt-4 space-y-3">
        <div>
          <dt className="font-medium">Strict score</dt>
          <dd className="muted">
            From a detector trained specifically to see through humanization.
            It is the honest answer to &ldquo;what is this text?&rdquo; and it
            is never what the rewrite optimises against — otherwise the loop
            would just be teaching itself to cheat.
          </dd>
        </div>
        <div>
          <dt className="font-medium">Estimated third-party score</dt>
          <dd className="muted">
            From a separate surrogate panel built to behave like the detectors
            you will actually be graded by. This is what the rewrite targets,
            because optimising against a panel generalises where optimising
            against one model does not.
          </dd>
        </div>
      </dl>
      <p className="mt-4 muted">
        Paste something above to start. Nothing leaves this app during
        detection.
      </p>
    </section>
  );
}
