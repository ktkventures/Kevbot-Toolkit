/**
 * SIM basis primitives + SimPanel (board #144 → unified into #177).
 *
 * SIM is the higher-fidelity certification basis: the REAL live engine
 * (StrategyMonitor / SymbolHub / _ShadowIndicatorEngine) replayed over recorded
 * decision-time bars under the ARMED prod flag stack — the /replay-check harness
 * productized. It is logic-WOULD-fire evidence, NEVER delivery evidence.
 *
 * This file no longer owns a standalone modal. Board #177 folded SIM into the
 * ONE unified last-10 modal (Last10PairingModal) as a tab, because Kevin's use
 * case is a cross-basis comparison ON THE SAME TEN TRADES — two modals turned
 * every comparison into a context switch. What lives here now:
 *   - the #120 API contract shapes (SimScore/SimDivergence/SimRequest),
 *   - the shared visual vocabulary (SIM_BADGE, simRatioColor), imported by the
 *     health column,
 *   - `SimPanel` — the collapsible SIM-only surface the unified modal renders
 *     under the SIM tab's pairing table: the ENUMERATED divergence ledger
 *     (Kevin's hard trust requirement, framed as "known LIMITS — not test
 *     results"), run provenance, the on-demand Run button (queued-behind + ETA
 *     state, board #177 item 6), the nightly opt-in, and run navigation over
 *     the immutable request→result history (board #177 item 5).
 *
 * Contract: src/api/routers/replay_sim.py — GET /{sid}?detail=&result_id=,
 * GET /{sid}/requests, GET /queue, POST /{sid}/run, GET|PUT /{sid}/optin.
 */
'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, ApiError } from '@/lib/api/client';

// ── #120/#177 API contract shapes ─────────────────────────────────────
export interface SimDivergence {
  id: string; title: string; bias: string; surface: string;
}
/** One SIM-basis pairing row (detail=true) — mirrors the alert-lane detail:
 *  the pairing table needs only timestamps, so there is no price / direction. */
export interface SimTradeSide {
  entry_nearest_alert_ts: string | null; entry_delta_sec: number | null; entry_paired: boolean;
  exit_nearest_alert_ts: string | null; exit_delta_sec: number | null; exit_paired: boolean;
}
export interface SimTradeRow {
  trade_id: number;
  entry_ts: string | null; exit_ts: string | null;
  sim: SimTradeSide;
}
// GET /api/strategy-health-sim/{sid}. status='none' ⇒ never run (show Run);
// 'unres' ⇒ replay couldn't resolve a ribbon (show the reason, never a 0).
export interface SimScore {
  strategy_id: number;
  // 'fetch_error' (board #177 / #175) = the column's GET itself failed — stored
  // in the score map instead of leaving the key absent, so the cell can fail
  // LOUD (a clickable red chip) rather than render an indistinguishable dead '—'.
  status: 'none' | 'unres' | 'ok' | 'partial' | 'error' | 'fetch_error';
  error?: string;                 // fetch_error message (cell tooltip)
  points?: number; denom?: number; trade_count?: number;
  phantom?: number;               // board #177 — replay edges paired to NO bt edge
  tolerance_seconds?: number;
  coverage?: number | null;       // fraction of in-window closes with decision-time bars
  corrected?: boolean;            // decision-time (false) vs REST-healed (true)
  divergences?: SimDivergence[];
  computed_at?: string | null;
  compute_secs?: number | null;
  flags_fp?: string | null;
  flags_stale?: boolean;          // armed flags changed since compute → re-run
  engine_sha?: string | null;
  window_since?: string | null; window_until?: string | null;
  timeframe_secs?: number | null;
  stale?: boolean;                // current last-10 extends past the cached window
  covered_of_total?: [number, number];
  result_id?: number | null;
  // detail=true additions (board #177 SIM tab)
  trades?: SimTradeRow[];
  display_window_sec?: number;
}
export interface SimRequest {
  id: number; strategy_id: number; requested_by: string | null;
  source: string; requested_at: string; started_at: string | null;
  finished_at: string | null; outcome: string; result_id: number | null;
  claimed_by: string | null; log_tail: string | null;
}
interface QueueRow {
  id: number; strategy_id: number; outcome: string; requested_at: string;
  started_at: string | null;
}

// ── Shared visual vocabulary (also imported by the health column) ──────
// Teal keeps SIM visually DISTINCT from the violet ALGO badge and the delivery
// "Last 10" — a different lane, not a different number in the same lane.
export const SIM_BADGE: React.CSSProperties = {
  background: 'rgba(45,212,191,0.16)', color: '#2dd4bf',
  padding: '0 5px', borderRadius: 3, fontSize: 9.5, fontWeight: 700,
  letterSpacing: 0.4, verticalAlign: 'middle', marginRight: 4,
};
/** Score ratio → band color (same bands as the other Last-10 lanes). */
export function simRatioColor(ratio: number | null): string {
  if (ratio == null) return 'var(--text-muted)';
  if (ratio >= 0.9) return '#66bb6a';
  if (ratio >= 0.7) return '#ffc107';
  return '#ef5350';
}
const ACTIVE_OUTCOMES = new Set(['requested', 'running']);

/** Timeframe string → seconds (drives the Run ETA even before any cache row
 *  exists). Falls back to a cached basis-row value, then null. */
export function simTfSeconds(tf?: string | null,
                            fallbackSecs?: number | null): number | null {
  if (tf) {
    const m = /^(\d+)\s*(Sec|Min|Hour|Day)/i.exec(tf.trim());
    if (m) {
      const n = Number(m[1]);
      const unit = m[2].toLowerCase();
      const mult = unit === 'sec' ? 1 : unit === 'min' ? 60
        : unit === 'hour' ? 3600 : 86400;
      if (n > 0) return n * mult;
    }
  }
  return fallbackSecs ?? null;
}

/** Rough replay ETA keyed off timeframe (board #177 item 6): the 1Sec SIP tick
 *  load dominates, so a finer TF costs far more (~1Min≈10s, ~10Sec≈158s ≈16×).
 *  Deliberately approximate — a warning, not a promise. */
function etaSeconds(tfSec: number | null): number | null {
  if (!tfSec || tfSec <= 0) return null;
  return tfSec >= 60 ? Math.max(8, Math.round(600 / tfSec))
    : Math.round(1580 / tfSec);
}
function fmtDuration(s: number | null): string {
  if (s == null) return '—';
  if (s < 90) return `~${s}s`;
  return `~${Math.round(s / 60)}m`;
}

// ── formatting ─────────────────────────────────────────────────────────
function relAge(iso: string | null | undefined): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (isNaN(t)) return '—';
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
const fmtTs = (iso: string | null | undefined) =>
  iso ? iso.slice(5, 19).replace('T', ' ') : '—';

// ── styles ───────────────────────────────────────────────────────────
const sectionLabel: React.CSSProperties = {
  fontSize: 10.5, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase',
  color: 'var(--text-tertiary)', margin: '14px 0 6px',
};
const chip: React.CSSProperties = {
  display: 'inline-block', padding: '1px 7px', borderRadius: 10, fontSize: 11,
  border: '1px solid var(--border)', color: 'var(--text-secondary)',
  marginRight: 6, marginBottom: 4, whiteSpace: 'nowrap',
};

/** Ledger-entry tone. D7 (red) = no score, fail-loud. D8 (amber) = suspected
 *  LIVE-engine bug, not a SIM artifact. corrected (blue) = not the honest
 *  ceiling. D3 (teal) = the would-fire-not-delivery caveat / SIM identity.
 *  Everything else is a neutral structural limit. A legend spells this out
 *  because "gray vs teal" was not self-explaining (board #177). */
function ledgerTone(id: string): { border: string; fg: string } {
  if (id === 'D7') return { border: '#ef5350', fg: '#ef5350' };
  if (id === 'D8') return { border: '#ffc107', fg: '#ffc107' };
  if (id === 'corrected') return { border: '#4f8cf6', fg: '#7fb0f6' };
  if (id === 'D3') return { border: '#2dd4bf', fg: '#2dd4bf' };
  return { border: 'var(--border)', fg: 'var(--text-secondary)' };
}
// D7 (unres) + D8 (label-drift) are the ONLY run-conditional entries — they
// appear only when THIS run tripped them. Everything else is structural and
// present on every run. Board #177 asks these two be visually separated.
const CONDITIONAL_IDS = new Set(['D7', 'D8']);

function LedgerCard({ d }: { d: SimDivergence }) {
  const tone = ledgerTone(d.id);
  return (
    <div style={{
      border: `1px solid ${tone.border}`, borderRadius: 8,
      padding: '6px 9px', background: 'var(--bg-input)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 10, fontWeight: 700, color: tone.fg,
                       border: `1px solid ${tone.border}`, borderRadius: 4,
                       padding: '0 4px', flexShrink: 0 }}>{d.id}</span>
        <span style={{ fontSize: 12.5, fontWeight: 600,
                       color: 'var(--text-primary)' }}>{d.title}</span>
      </div>
      <div style={{ fontSize: 12, color: tone.fg, marginTop: 3 }}>{d.surface}</div>
      {d.bias && (
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
          bias: {d.bias}</div>
      )}
    </div>
  );
}

/** The SIM-only surface, collapsed by default under the SIM tab's pairing
 *  table (board #177). Presentational for score/ledger/provenance (the unified
 *  modal fetches the detail and passes it in as `score`); owns the request
 *  queue, opt-in, Run button, and run-navigation. */
export function SimPanel({ sid, name, score, resultId, timeframe,
                          onSelectRun, onReload }: {
  sid: number; name: string;
  score: SimScore | null;
  resultId: number | null;          // which immutable run is shown (null = latest)
  timeframe?: string | null;        // strategy TF (for the pre-run ETA)
  onSelectRun: (id: number | null) => void;
  onReload: () => void;             // re-fetch the modal's SIM detail
}) {
  const [requests, setRequests] = useState<SimRequest[]>([]);
  const [queue, setQueue] = useState<QueueRow[]>([]);
  const [optin, setOptin] = useState<boolean | null>(null);
  const [runMsg, setRunMsg] = useState<string | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const [optinBusy, setOptinBusy] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onReloadRef = useRef(onReload);
  onReloadRef.current = onReload;

  const loadRequests = useCallback(() =>
    apiFetch<SimRequest[]>(`/api/strategy-health-sim/${sid}/requests?limit=10`)
      .then((r) => { setRequests(r || []); return r || []; })
      .catch(() => [] as SimRequest[]), [sid]);
  const loadQueue = useCallback(() =>
    apiFetch<{ active: QueueRow[] }>(`/api/strategy-health-sim/queue`)
      .then((q) => { setQueue(q.active || []); return q.active || []; })
      .catch(() => [] as QueueRow[]), []);

  useEffect(() => {
    loadRequests();
    loadQueue();
    apiFetch<{ enabled: boolean }>(`/api/strategy-health-sim/${sid}/optin`)
      .then((o) => setOptin(!!o.enabled))
      .catch(() => setOptin(false));
  }, [sid, loadRequests, loadQueue]);

  const activeReq = requests.find((r) => ACTIVE_OUTCOMES.has(r.outcome));

  // Poll the queue + this sid's requests while a run is active; on terminal,
  // reload the modal detail (a fresh cache row may have landed).
  useEffect(() => {
    if (!activeReq) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      loadQueue();
      const rows = await loadRequests();
      if (!rows.some((r) => ACTIVE_OUTCOMES.has(r.outcome))) {
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        onReloadRef.current();
      }
    }, 4000);
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [activeReq, loadRequests, loadQueue]);

  const runSim = async () => {
    setRunBusy(true); setRunMsg(null);
    try {
      await apiFetch(`/api/strategy-health-sim/${sid}/run`,
        { method: 'POST', body: JSON.stringify({ author: 'kevin' }) });
      setRunMsg('queued — the local poller will run it');
      await Promise.all([loadRequests(), loadQueue()]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setRunMsg('a SIM run is already pending for this strategy');
        await loadRequests();
      } else {
        setRunMsg(`failed to queue: ${String((e as Error)?.message ?? e)}`);
      }
    } finally {
      setRunBusy(false);
    }
  };

  const toggleOptin = async () => {
    const next = !optin;
    setOptinBusy(true); setOptin(next);
    try {
      await apiFetch(`/api/strategy-health-sim/${sid}/optin`,
        { method: 'PUT', body: JSON.stringify({ enabled: next, author: 'kevin' }) });
    } catch (e) {
      setOptin(!next);
      setRunMsg(`opt-in change failed: ${String((e as Error)?.message ?? e)}`);
    } finally {
      setOptinBusy(false);
    }
  };

  // ── Run state: distinguish QUEUED (behind N) from RUNNING (board #177 item
  //    6). The poller is serialized (one replay at a time), so a 'requested'
  //    row sits behind everything ahead of it; without this it looked hung.
  const tfSec = simTfSeconds(timeframe, score?.timeframe_secs);
  const eta = etaSeconds(tfSec);
  let runState: 'idle' | 'queued' | 'running' = 'idle';
  let queuedBehind = 0;
  if (activeReq) {
    if (activeReq.outcome === 'running') {
      runState = 'running';
    } else {
      runState = 'queued';
      // position among the global active queue (oldest-first); # ahead of us.
      const ordered = [...queue].sort((a, b) =>
        Date.parse(a.requested_at) - Date.parse(b.requested_at));
      const mineIdx = ordered.findIndex((q) => q.strategy_id === sid
        && ACTIVE_OUTCOMES.has(q.outcome));
      queuedBehind = mineIdx > 0 ? mineIdx : 0;
    }
  }
  const runActive = runBusy || !!activeReq;
  const runLabel =
    runState === 'running' ? `⚙ running… (ETA ${fmtDuration(eta)})`
    : runState === 'queued'
      ? (queuedBehind > 0 ? `⏳ queued behind ${queuedBehind}` : '⏳ queued')
    : runBusy ? 'queuing…'
    : `▷ Run SIM now (${fmtDuration(eta)})`;

  const divergences = score?.divergences ?? [];
  const structural = divergences.filter((d) => !CONDITIONAL_IDS.has(d.id));
  const conditional = divergences.filter((d) => CONDITIONAL_IDS.has(d.id));
  const cov = score?.coverage;
  const viewingOld = resultId != null;

  return (
    <div>
      {/* Run-navigation banner — viewing a specific immutable run (item 5). */}
      {viewingOld && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                      fontSize: 11.5, color: '#7fb0f6', margin: '4px 0',
                      padding: '4px 8px', borderRadius: 8,
                      border: '1px solid #4f8cf6', background: 'var(--bg-input)' }}>
          <span>viewing run #{resultId} — {relAge(score?.computed_at)} · what SIM saw then</span>
          <button onClick={() => onSelectRun(null)}
            style={{ background: 'transparent', border: '1px solid var(--border)',
                     borderRadius: 6, padding: '1px 8px', cursor: 'pointer',
                     color: 'var(--text-secondary)', fontSize: 11 }}>
            ↩ back to latest</button>
        </div>
      )}

      {/* Provenance chips */}
      <div style={sectionLabel}>Run provenance</div>
      <div>
        <span style={chip} title="when this cache row was computed">
          computed {relAge(score?.computed_at)}</span>
        {cov != null && (
          <span style={{ ...chip, ...(cov < 1
                ? { borderColor: '#ffc107', color: '#ffc107' } : {}) }}
                title="fraction of in-window primary closes that had decision-time live_bars rows">
            coverage {(cov * 100).toFixed(0)}%</span>
        )}
        <span style={{ ...chip, ...(score?.corrected
              ? { borderColor: '#4f8cf6', color: '#7fb0f6' } : {}) }}
              title={score?.corrected
                ? 'REST-corrected OHLC — convergent with backtest, NOT the honest decision-time ceiling'
                : 'decision-time OHLC — the honest ceiling'}>
          {score?.corrected ? 'corrected OHLC' : 'decision-time'}</span>
        {score?.compute_secs != null && (
          <span style={chip}>{score.compute_secs.toFixed(1)}s compute</span>
        )}
        {score?.engine_sha && (
          <span style={chip} title="git sha of the engine that produced this replay">
            engine {score.engine_sha}</span>
        )}
        {score?.flags_fp && (
          <span style={{ ...chip, ...(score.flags_stale
                ? { borderColor: '#ffc107', color: '#ffc107' } : {}) }}
                title={score.flags_stale
                  ? 'armed engine flags changed since this was computed — re-run to refresh'
                  : 'armed-flag fingerprint at compute time'}>
            flags {score.flags_fp}{score.flags_stale ? ' · ⟳ re-run' : ''}</span>
        )}
        {score?.stale && (
          <span style={{ ...chip, borderColor: '#ffc107', color: '#ffc107' }}
                title="the current last-10 backtest trades extend past this cache row's window — re-run to cover them">
            ⟳ window moved</span>
        )}
        {(score?.window_since || score?.window_until) && (
          <span style={chip} title="the decision/trade span replayed (warmup preamble is loaded before this and is never reduced)">
            window {fmtTs(score?.window_since)} → {fmtTs(score?.window_until)}</span>
        )}
      </div>

      {/* THE DIVERGENCE LEDGER — Kevin req 1, reframed (board #177 item 4):
          these are KNOWN LIMITS of the measurement, NOT test results. */}
      <div style={sectionLabel}>
        Known limits of this measurement — NOT test results
        {divergences.length > 0 && ` (${divergences.length})`}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginBottom: 8,
                    lineHeight: 1.5 }}>
        Every enumerated way this replay differs from what LIVE actually does —
        surfaced so a 20/20 is never read as gospel. Tone key:{' '}
        <span style={{ color: '#2dd4bf' }}>teal = SIM identity (would-fire, not delivery)</span>,{' '}
        <span style={{ color: 'var(--text-secondary)' }}>gray = structural limit</span>,{' '}
        <span style={{ color: '#ef5350' }}>red = no score (fail-loud)</span>,{' '}
        <span style={{ color: '#ffc107' }}>amber = suspected live-engine bug / caution</span>.
      </div>
      {divergences.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
          {score?.status === 'none'
            ? 'No run yet — the ledger is written per run.'
            : 'No ledger entries recorded for this run.'}
        </div>
      ) : (
        <>
          {/* Run-conditional flags FIRST — only present when THIS run tripped
              them, so they carry the most run-specific signal. */}
          {conditional.length > 0 && (
            <>
              <div style={{ fontSize: 10.5, fontWeight: 700, color: '#ffc107',
                            margin: '2px 0 5px' }}>
                ⚠ Triggered by THIS run</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6,
                            marginBottom: 10 }}>
                {conditional.map((d, i) => <LedgerCard key={`c${d.id}:${i}`} d={d} />)}
              </div>
            </>
          )}
          <div style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-tertiary)',
                        margin: '2px 0 5px' }}>
            Always present (structural)</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {structural.map((d, i) => <LedgerCard key={`s${d.id}:${i}`} d={d} />)}
          </div>
        </>
      )}

      {/* Actions: on-demand RUN + nightly OPT-IN */}
      <div style={sectionLabel}>Run this basis</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <button onClick={runSim} disabled={runActive}
          title="Enqueue an on-demand replay. Declarative — the local poller claims and runs it off-hours (Railway can't reach the runner host), same pattern as the task Run button. Requires the poller to be running (tools/team_sim/replay_sim_poller.py)."
          style={{
            background: runActive ? 'var(--bg-input)' : 'rgba(45,212,191,0.16)',
            color: runActive ? 'var(--text-muted)' : '#2dd4bf',
            border: `1px solid ${runActive ? 'var(--border)' : '#2dd4bf'}`,
            borderRadius: 8, padding: '4px 12px', fontSize: 12.5, fontWeight: 600,
            cursor: runActive ? 'default' : 'pointer',
          }}>
          {runLabel}
        </button>

        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6,
                        cursor: optinBusy ? 'wait' : 'pointer', fontSize: 12 }}
               title="Nightly SIM sweep runs ONLY opted-in strategies — Kevin's cost dial so fleet-wide cost doesn't scale with fleet size. Default OFF.">
          <button onClick={toggleOptin} disabled={optinBusy || optin == null}
            style={{
              background: optin ? 'rgba(45,212,191,0.16)' : 'transparent',
              border: `1px solid ${optin ? '#2dd4bf' : 'var(--border)'}`,
              borderRadius: 10, padding: '2px 10px', fontSize: 11.5, fontWeight: 600,
              color: optin ? '#2dd4bf' : 'var(--text-muted)',
              cursor: optinBusy || optin == null ? 'wait' : 'pointer',
            }}>
            {optin == null ? '…' : optin ? 'ON' : 'OFF'}
          </button>
          <span style={{ color: 'var(--text-secondary)' }}>nightly opt-in</span>
        </label>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 5 }}>
        The Run button only works while the local poller is up; a queued run that
        never leaves “queued” means the poller is down.
      </div>
      {runMsg && (
        <div style={{ fontSize: 11.5, marginTop: 6, color: 'var(--text-tertiary)' }}>
          {runMsg}</div>
      )}

      {/* Recent runs — click a completed run to see what SIM saw then (item 5). */}
      {requests.length > 0 && (
        <>
          <div style={sectionLabel}>Recent runs — click one to view its result</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {requests.slice(0, 8).map((r) => {
              const navigable = r.result_id != null;
              const isViewed = r.result_id != null && r.result_id === resultId;
              return (
              <div key={r.id}
                   onClick={navigable ? () => onSelectRun(r.result_id!) : undefined}
                   title={navigable ? 'view this run’s result (provenance + ledger + pairing)'
                     : 'no stored result to view'}
                   style={{ fontSize: 11.5, display: 'flex', gap: 8,
                            alignItems: 'baseline', flexWrap: 'wrap',
                            cursor: navigable ? 'pointer' : 'default',
                            padding: '2px 6px', borderRadius: 6,
                            border: `1px solid ${isViewed ? '#4f8cf6' : 'transparent'}`,
                            background: isViewed ? 'var(--bg-input)' : 'transparent' }}>
                <span style={{ ...chip, marginBottom: 0,
                  ...(ACTIVE_OUTCOMES.has(r.outcome)
                    ? { borderColor: '#4f8cf6', color: '#7fb0f6' }
                    : r.outcome === 'ok' ? { borderColor: '#66bb6a', color: '#66bb6a' }
                    : r.outcome === 'partial' ? { borderColor: '#ffc107', color: '#ffc107' }
                    : (r.outcome === 'error' || r.outcome === 'unres')
                      ? { borderColor: '#ef5350', color: '#ef5350' } : {}) }}>
                  {r.outcome}</span>
                <span style={{ color: 'var(--text-tertiary)' }}>{relAge(r.requested_at)}</span>
                <span style={{ color: 'var(--text-tertiary)' }}>
                  · {r.source}{r.requested_by ? ` · ${r.requested_by}` : ''}</span>
                {navigable && <span style={{ color: '#7fb0f6' }}>· view →</span>}
                {r.log_tail && (r.outcome === 'error' || r.outcome === 'unres') && (
                  <span style={{ color: 'var(--red)', overflow: 'hidden',
                                 textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                 maxWidth: 360 }} title={r.log_tail}>{r.log_tail}</span>
                )}
              </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
