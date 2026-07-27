/**
 * SIM basis modal (board #144 — consumer half of #120) — click the SIM cell on
 * the Strategy Health table → the replay-simulated last-10 basis for one
 * strategy: the score, the ENUMERATED divergence ledger (Kevin's hard trust
 * requirement — every known way the replay differs from live, surfaced next to
 * the score, never implied), the run provenance, an on-demand RUN button (a
 * declarative POST, dispatcher-served — same pattern as the task Run button),
 * and the per-strategy NIGHTLY OPT-IN toggle (default OFF — Kevin's cost dial).
 *
 * SIM is a 4th lane (companion to fired/theo/algo): logic-WOULD-fire evidence
 * from the real engine over recorded decision-time bars, NEVER delivery
 * evidence. The score reads a cache row (cheap; the heavy replay runs off the
 * button/nightly sweep, never on page load). Contract: src/api/routers/
 * replay_sim.py (GET /{sid}, POST /{sid}/run, GET /{sid}/requests,
 * GET/PUT /{sid}/optin). Three-panel-modal design language; portals to <body>.
 *
 * Fail-loud (Kevin req d): unres/error/partial render as their status, never a
 * silent zero. A missing cache renders as "not run", not 0/0.
 */
'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch, ApiError } from '@/lib/api/client';

// ── #120 API contract shapes ──────────────────────────────────────────
// One ledger entry (replay_sim_job.build_divergence_ledger): the divergence,
// its bias, and the human line to render next to the score.
export interface SimDivergence {
  id: string; title: string; bias: string; surface: string;
}
// GET /api/strategy-health-sim/{sid}. status='none' ⇒ never run (show Run);
// 'unres' ⇒ replay couldn't resolve a ribbon (show the reason, never a 0).
export interface SimScore {
  strategy_id: number;
  status: 'none' | 'unres' | 'ok' | 'partial' | 'error';
  points?: number; denom?: number; trade_count?: number;
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
  stale?: boolean;                // current last-10 extends past the cached window
  covered_of_total?: [number, number];
  result_id?: number | null;
}
export interface SimRequest {
  id: number; strategy_id: number; requested_by: string | null;
  source: string; requested_at: string; started_at: string | null;
  finished_at: string | null; outcome: string; result_id: number | null;
  claimed_by: string | null; log_tail: string | null;
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
const panelHead: React.CSSProperties = {
  fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: 'var(--text-tertiary)',
  padding: '8px 12px', borderBottom: '1px solid var(--border)', textTransform: 'uppercase',
  display: 'flex', alignItems: 'center', gap: 8,
};
const sectionLabel: React.CSSProperties = {
  fontSize: 10.5, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase',
  color: 'var(--text-tertiary)', margin: '14px 0 6px',
};
const chip: React.CSSProperties = {
  display: 'inline-block', padding: '1px 7px', borderRadius: 10, fontSize: 11,
  border: '1px solid var(--border)', color: 'var(--text-secondary)',
  marginRight: 6, marginBottom: 4, whiteSpace: 'nowrap',
};

/** Ledger-entry tone: fail-loud markers warn, corrected/label-drift caution,
 *  the "would-fire not delivery" caveat (D3) is emphasized, rest are neutral. */
function ledgerTone(id: string): { border: string; fg: string } {
  if (id === 'D7') return { border: '#ef5350', fg: '#ef5350' };        // unres — no score
  if (id === 'D8') return { border: '#ffc107', fg: '#ffc107' };        // label-drift — live-bug suspect
  if (id === 'corrected') return { border: '#4f8cf6', fg: '#7fb0f6' }; // not the honest ceiling
  if (id === 'D3') return { border: '#2dd4bf', fg: '#2dd4bf' };        // would-fire, NOT delivery
  return { border: 'var(--border)', fg: 'var(--text-secondary)' };
}

// ── status pill (fail-loud, never a bare 0) ────────────────────────────
function StatusPill({ s }: { s: SimScore }) {
  if (s.status === 'ok' || s.status === 'partial') {
    const ratio = s.denom && s.denom > 0 ? (s.points ?? 0) / s.denom : null;
    return (
      <span style={{ fontSize: 15, fontWeight: 700, color: simRatioColor(ratio),
                     fontVariantNumeric: 'tabular-nums' }}>
        {s.points ?? 0}/{s.denom ?? 0}
        {s.status === 'partial' && (
          <span style={{ ...chip, borderColor: '#ffc107', color: '#ffc107',
                         marginLeft: 8, fontSize: 10 }}>partial</span>
        )}
      </span>
    );
  }
  if (s.status === 'unres') {
    return <span style={{ ...chip, borderColor: '#ef5350', color: '#ef5350', fontSize: 12 }}>
      ⚠ SIM unavailable — unresolved ribbon</span>;
  }
  if (s.status === 'error') {
    return <span style={{ ...chip, borderColor: '#ef5350', color: '#ef5350', fontSize: 12 }}>
      ⚠ error</span>;
  }
  return <span style={{ ...chip, fontSize: 12 }}>not run yet</span>;
}

export default function SimBasisModal({ sid, name, onClose, onScoreChange }: {
  sid: number; name: string; onClose: () => void;
  onScoreChange?: (s: SimScore) => void;
}) {
  const [score, setScore] = useState<SimScore | null>(null);
  const [requests, setRequests] = useState<SimRequest[]>([]);
  const [optin, setOptin] = useState<boolean | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [runMsg, setRunMsg] = useState<string | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const [optinBusy, setOptinBusy] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Hold the latest onScoreChange in a ref so a parent re-render (the health
  // page updates its score map constantly during its mount-time SIM sweep)
  // never changes loadScore's identity and re-fires the initial-load effect
  // — which would refetch the opt-in and clobber an in-flight toggle.
  const onScoreChangeRef = useRef(onScoreChange);
  onScoreChangeRef.current = onScoreChange;

  const loadScore = useCallback(() => {
    return apiFetch<SimScore>(`/api/strategy-health-sim/${sid}`)
      .then((s) => { setScore(s); onScoreChangeRef.current?.(s); return s; })
      .catch((e) => { setErr(String(e?.message ?? e)); return null; });
  }, [sid]);

  const loadRequests = useCallback(() => {
    return apiFetch<SimRequest[]>(`/api/strategy-health-sim/${sid}/requests?limit=8`)
      .then((r) => { setRequests(r || []); return r || []; })
      .catch(() => [] as SimRequest[]);
  }, [sid]);

  // Initial load (score + requests + opt-in in parallel). Keyed to sid only.
  useEffect(() => {
    loadScore();
    loadRequests();
    apiFetch<{ enabled: boolean }>(`/api/strategy-health-sim/${sid}/optin`)
      .then((o) => setOptin(!!o.enabled))
      .catch(() => setOptin(false));
  }, [sid, loadScore, loadRequests]);

  const activeReq = requests.find((r) => ACTIVE_OUTCOMES.has(r.outcome));

  // While a run is requested/running, poll the request queue so the button
  // reflects the poller's progress; when it goes terminal, refresh the score.
  // Bounded: only while the modal is open AND a request is active.
  useEffect(() => {
    if (!activeReq) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      const rows = await loadRequests();
      if (!rows.some((r) => ACTIVE_OUTCOMES.has(r.outcome))) {
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        loadScore();     // terminal → the cache may have a fresh row
      }
    }, 4000);
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [activeReq, loadRequests, loadScore]);

  const runSim = async () => {
    setRunBusy(true); setRunMsg(null);
    try {
      await apiFetch(`/api/strategy-health-sim/${sid}/run`,
        { method: 'POST', body: JSON.stringify({ author: 'kevin' }) });
      setRunMsg('queued — the local poller will run it');
      await loadRequests();
    } catch (e) {
      // 409 = a run is already pending/running for this strategy (not an error).
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
    setOptinBusy(true);
    setOptin(next);                                  // optimistic
    try {
      await apiFetch(`/api/strategy-health-sim/${sid}/optin`,
        { method: 'PUT', body: JSON.stringify({ enabled: next, author: 'kevin' }) });
    } catch (e) {
      setOptin(!next);                               // revert on failure
      setRunMsg(`opt-in change failed: ${String((e as Error)?.message ?? e)}`);
    } finally {
      setOptinBusy(false);
    }
  };

  const runActive = runBusy || !!activeReq;
  const divergences = score?.divergences ?? [];
  const cov = score?.coverage;

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6vh 16px',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
        borderRadius: 12, width: '90vw', maxWidth: 820, maxHeight: '84vh',
        display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={panelHead}>
          <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            <span style={SIM_BADGE}>SIM</span>
            Replay basis · sid {sid} — {name}
          </span>
          <button onClick={onClose} style={{
            background: 'var(--bg-input)', color: 'var(--text-primary)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '2px 8px',
            cursor: 'pointer', fontSize: 12,
          }}>✕ close</button>
        </div>

        <div style={{ overflowY: 'auto', padding: '4px 16px 16px' }}>
          {err && <div style={{ color: 'var(--red)', fontSize: 13, marginTop: 10 }}>⚠ {err}</div>}
          {!score && !err && (
            <div style={{ fontSize: 13, color: 'var(--text-tertiary)', marginTop: 12 }}>Loading…</div>
          )}

          {score && (
            <>
              {/* Score + framing */}
              <div style={{ marginTop: 12, display: 'flex', alignItems: 'baseline',
                            gap: 12, flexWrap: 'wrap' }}>
                <StatusPill s={score} />
                {(score.status === 'ok' || score.status === 'partial') && (
                  <span style={{ fontSize: 11.5, color: 'var(--text-tertiary)' }}>
                    entries+exits paired within ±{score.tolerance_seconds ?? 10}s
                    {score.covered_of_total &&
                      ` · covers ${score.covered_of_total[0]}/${score.covered_of_total[1]} of last-10`}
                  </span>
                )}
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-tertiary)', marginTop: 6,
                            lineHeight: 1.5 }}>
                SIM = the real engine replayed over recorded decision-time bars —
                <b style={{ color: '#2dd4bf' }}> logic-would-fire evidence, never delivery</b>.
                SIM-high + delivery-low = ops/delivery loss; SIM-low = a logic/state
                divergence (a real bug). No operational loss is modeled, so SIM ≥ live by
                construction (a ceiling).
              </div>

              {/* Provenance chips */}
              <div style={sectionLabel}>Run provenance</div>
              <div>
                <span style={chip} title="when this cache row was computed">
                  computed {relAge(score.computed_at)}</span>
                {cov != null && (
                  <span style={{ ...chip, ...(cov < 1
                        ? { borderColor: '#ffc107', color: '#ffc107' } : {}) }}
                        title="fraction of in-window primary closes that had decision-time live_bars rows">
                    coverage {(cov * 100).toFixed(0)}%</span>
                )}
                <span style={{ ...chip, ...(score.corrected
                      ? { borderColor: '#4f8cf6', color: '#7fb0f6' } : {}) }}
                      title={score.corrected
                        ? 'REST-corrected OHLC — convergent with backtest, NOT the honest decision-time ceiling'
                        : 'decision-time OHLC — the honest ceiling'}>
                  {score.corrected ? 'corrected OHLC' : 'decision-time'}</span>
                {score.compute_secs != null && (
                  <span style={chip}>{score.compute_secs.toFixed(1)}s compute</span>
                )}
                {score.engine_sha && (
                  <span style={chip} title="git sha of the engine that produced this replay">
                    engine {score.engine_sha}</span>
                )}
                {score.flags_fp && (
                  <span style={{ ...chip, ...(score.flags_stale
                        ? { borderColor: '#ffc107', color: '#ffc107' } : {}) }}
                        title={score.flags_stale
                          ? 'armed engine flags changed since this was computed — re-run to refresh'
                          : 'armed-flag fingerprint at compute time'}>
                    flags {score.flags_fp}{score.flags_stale ? ' · ⟳ re-run' : ''}</span>
                )}
                {score.stale && (
                  <span style={{ ...chip, borderColor: '#ffc107', color: '#ffc107' }}
                        title="the current last-10 backtest trades extend past this cache row's window — re-run to cover them">
                    ⟳ window moved</span>
                )}
                {(score.window_since || score.window_until) && (
                  <span style={chip} title="the decision/trade span replayed (warmup preamble is loaded before this and is never reduced)">
                    window {fmtTs(score.window_since)} → {fmtTs(score.window_until)}</span>
                )}
              </div>

              {/* THE DIVERGENCE LEDGER — Kevin req 1 (surfaced next to the score) */}
              <div style={sectionLabel}>
                Divergence ledger — how this replay differs from live
                {divergences.length > 0 && ` (${divergences.length})`}
              </div>
              {divergences.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                  {score.status === 'none'
                    ? 'No run yet — the ledger is written per run.'
                    : 'No ledger entries recorded for this run.'}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {divergences.map((d, i) => {
                    const tone = ledgerTone(d.id);
                    return (
                      <div key={`${d.id}:${i}`} style={{
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
                  })}
                </div>
              )}

              {/* Actions: on-demand RUN + nightly OPT-IN */}
              <div style={sectionLabel}>Run this basis</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <button onClick={runSim} disabled={runActive}
                  title="Enqueue an on-demand replay. Declarative — the local poller claims and runs it off-hours (Railway can't reach the runner host), same pattern as the task Run button."
                  style={{
                    background: runActive ? 'var(--bg-input)' : 'rgba(45,212,191,0.16)',
                    color: runActive ? 'var(--text-muted)' : '#2dd4bf',
                    border: `1px solid ${runActive ? 'var(--border)' : '#2dd4bf'}`,
                    borderRadius: 8, padding: '4px 12px', fontSize: 12.5, fontWeight: 600,
                    cursor: runActive ? 'default' : 'pointer',
                  }}>
                  {activeReq ? `⚙ ${activeReq.outcome}…` : runBusy ? 'queuing…' : '▷ Run SIM now'}
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
              {runMsg && (
                <div style={{ fontSize: 11.5, marginTop: 6, color: 'var(--text-tertiary)' }}>
                  {runMsg}</div>
              )}

              {/* Recent request history */}
              {requests.length > 0 && (
                <>
                  <div style={sectionLabel}>Recent runs</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {requests.slice(0, 6).map((r) => (
                      <div key={r.id} style={{ fontSize: 11.5, display: 'flex', gap: 8,
                                               alignItems: 'baseline', flexWrap: 'wrap' }}>
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
                        {r.log_tail && (r.outcome === 'error' || r.outcome === 'unres') && (
                          <span style={{ color: 'var(--red)', overflow: 'hidden',
                                         textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                         maxWidth: 380 }} title={r.log_tail}>{r.log_tail}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
