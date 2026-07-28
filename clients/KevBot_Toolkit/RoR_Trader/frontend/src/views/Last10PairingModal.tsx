/**
 * Unified last-10 pairing modal (board #70 → #119 → #177).
 *
 * ONE modal, a tab switcher over the SAME ten backtest trades, so a new
 * strategy's simulation-accuracy is a tab flip — not a context switch between
 * two modals (Kevin's #177 requirement). Reuses the Last-10 pairing layout
 * verbatim (`# | BT ENTRY | NEAREST | Δ | PT | BT EXIT | NEAREST | Δ | PT`).
 *
 * THREE certification bases (#176 lane-semantics audit): `fired` + `theo` (both
 * vs alerts) and `SIM` (vs the REAL engine replayed over decision-time bars —
 * the higher-fidelity basis, /replay-check productized). `algo` (vs the ALGO-
 * LANE cache_% trades) is kept as an optional data-source DIAGNOSTIC, not a
 * certification basis.
 *
 * Every basis reports a PHANTOM count next to n/20 (board #177): n/20 is RECALL
 * only ("did a bt edge pair?"); phantom is the precision miss ("did the lane
 * fire with no bt edge?"). Without it, a lane that over-fires still scores full
 * marks — the sid-271 "10/20 vs 91-vs-51" contradiction.
 *
 * SIM tab carries a collapsed-by-default SimPanel: the divergence ledger (framed
 * as KNOWN LIMITS, not test results), run provenance, the on-demand Run button,
 * the nightly opt-in, and run navigation over the immutable result history.
 *
 * Endpoints: GET /api/strategy-health-last10/{sid} (fired/theo/algo detail),
 * GET /api/strategy-health-sim/{sid}?detail=true[&result_id] (SIM detail).
 * Three-panel-modal design language; portals to <body> (sidebar stacking).
 */
'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch } from '@/lib/api/client';
import {
  SimPanel, SIM_BADGE, simRatioColor, type SimScore,
} from '@/views/SimBasisModal';

interface BasisSide {
  entry_nearest_alert_ts: string | null; entry_delta_sec: number | null; entry_paired: boolean;
  exit_nearest_alert_ts: string | null; exit_delta_sec: number | null; exit_paired: boolean;
}
interface TradeRow {
  trade_id: number;
  entry_ts: string | null; exit_ts: string | null;
  fired: BasisSide; theo: BasisSide; algo: BasisSide;
}
interface Detail {
  strategy_id: number; points: number; points_theo: number; points_algo: number;
  phantom?: number; phantom_theo?: number; phantom_algo?: number;
  denom: number;
  trade_count: number; tolerance_seconds: number; display_window_sec: number;
  trades: TradeRow[];
}
export type Basis = 'fired' | 'theo' | 'algo' | 'sim';

// Certification bases (Kevin cares about these three) + algo as a diagnostic.
const CERT_BASES: Basis[] = ['fired', 'theo', 'sim'];
const BASIS_LABEL: Record<Basis, string> = {
  fired: 'fired', theo: 'theo', sim: 'SIM', algo: 'algo',
};
const BASIS_TITLE: Record<Basis, string> = {
  fired: 'pair on the alert FIRED/arrival timestamp — what we execute on (delivery truth)',
  theo: 'pair on the bar-aligned THEO timestamp (fill_ts) — canonical health-pairing field',
  sim: 'pair vs the SIM lane — the REAL engine replayed over recorded decision-time bars '
    + '(logic-would-fire, NEVER delivery; the higher-fidelity certification basis)',
  algo: 'DIAGNOSTIC (not a certification basis): pair vs ALGO-LANE trades (cache_%) — '
    + 'what the live engine computed. Kept as a data-source diagnostic per #176.',
};

const panelHead: React.CSSProperties = {
  fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: 'var(--text-tertiary)',
  padding: '8px 12px', borderBottom: '1px solid var(--border)', textTransform: 'uppercase',
  display: 'flex', alignItems: 'center', gap: 8,
};
const cell: React.CSSProperties = {
  padding: '5px 8px', fontSize: 12.5, fontVariantNumeric: 'tabular-nums',
  borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap',
};

const fmtTs = (iso: string | null) => iso ? iso.slice(5, 19).replace('T', ' ') : '—';
const fmtDelta = (d: number | null) => d == null ? '—' : `${d > 0 ? '+' : ''}${d.toFixed(1)}s`;
const Mark = ({ ok }: { ok: boolean }) => (
  <span style={{ color: ok ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>{ok ? '✓' : '✗'}</span>
);

// n/denom band color (same bands as the health column).
function ratioColor(pts: number | null, denom: number | null): string {
  if (pts == null || !denom) return 'var(--text-muted)';
  const r = pts / denom;
  return r >= 0.9 ? '#66bb6a' : r >= 0.7 ? '#ffc107' : '#ef5350';
}

export default function Last10PairingModal({ sid, name, onClose, initialBasis,
                                            timeframe, onSimScore }: {
  sid: number; name: string; onClose: () => void;
  initialBasis?: Basis; timeframe?: string | null;
  onSimScore?: (s: SimScore) => void;   // keep the health column cell in sync
}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);
  const [simDetail, setSimDetail] = useState<SimScore | null>(null);
  const [simErr, setSimErr] = useState<string | null>(null);
  const [basis, setBasis] = useState<Basis>(initialBasis ?? 'fired');
  const [simResultId, setSimResultId] = useState<number | null>(null);
  const [simReload, setSimReload] = useState(0);
  const [simOpen, setSimOpen] = useState(false);   // collapsed by default (#177)
  const onSimScoreRef = useRef(onSimScore);
  onSimScoreRef.current = onSimScore;

  // fired/theo/algo detail — one fetch, all three alert/algo bases.
  useEffect(() => {
    apiFetch<Detail>(`/api/strategy-health-last10/${sid}`)
      .then((d) => { setDetail(d); setDetailErr(null); })
      .catch((e) => setDetailErr(String(e?.message ?? e)));
  }, [sid]);

  // SIM detail — pairing rows + score + ledger + provenance. Re-fetched on
  // run-navigation (result_id) and after a Run lands (simReload).
  useEffect(() => {
    const q = simResultId != null
      ? `?detail=true&result_id=${simResultId}` : '?detail=true';
    apiFetch<SimScore>(`/api/strategy-health-sim/${sid}${q}`)
      .then((s) => {
        setSimDetail(s); setSimErr(null);
        // Only the latest run reflects the column cell — a navigated older run
        // must not overwrite the health-table score.
        if (simResultId == null) onSimScoreRef.current?.(s);
      })
      .catch((e) => setSimErr(String(e?.message ?? e)));
  }, [sid, simResultId, simReload]);

  const nearLabel = basis === 'sim' ? 'nearest SIM'
    : basis === 'algo' ? 'nearest algo' : 'nearest alert';

  // Per-basis score summary for the tab header (n/denom + phantom).
  const scoreOf = (b: Basis): { pts: number | null; denom: number | null;
                                phantom: number | null; status?: string } => {
    if (b === 'sim') {
      const s = simDetail;
      if (!s) return { pts: null, denom: null, phantom: null, status: 'loading' };
      if (s.status === 'ok' || s.status === 'partial') {
        return { pts: s.points ?? null, denom: s.denom ?? null,
                 phantom: s.phantom ?? null, status: s.status };
      }
      return { pts: null, denom: null, phantom: null, status: s.status };
    }
    if (!detail) return { pts: null, denom: null, phantom: null };
    if (b === 'fired') return { pts: detail.points, denom: detail.denom, phantom: detail.phantom ?? null };
    if (b === 'theo') return { pts: detail.points_theo, denom: detail.denom, phantom: detail.phantom_theo ?? null };
    return { pts: detail.points_algo, denom: detail.denom, phantom: detail.phantom_algo ?? null };
  };

  // Normalize the active basis into a shared {entry_ts, exit_ts, side} row list.
  const activeRows = useMemo(() => {
    if (basis === 'sim') {
      return (simDetail?.trades ?? []).map((t) => ({
        trade_id: t.trade_id, entry_ts: t.entry_ts, exit_ts: t.exit_ts,
        side: (t.sim ?? {}) as Partial<BasisSide>,
      }));
    }
    return (detail?.trades ?? []).map((t) => ({
      trade_id: t.trade_id, entry_ts: t.entry_ts, exit_ts: t.exit_ts,
      side: (t[basis] ?? {}) as Partial<BasisSide>,
    }));
  }, [basis, detail, simDetail]);

  const tol = detail?.tolerance_seconds ?? simDetail?.tolerance_seconds ?? 10;
  const dispWin = basis === 'sim'
    ? simDetail?.display_window_sec : detail?.display_window_sec;

  // What to show in the table body region for the active basis.
  const simStatus = simDetail?.status;
  const simUnscored = basis === 'sim'
    && !!simDetail && simStatus !== 'ok' && simStatus !== 'partial';
  const loadErr = basis === 'sim' ? simErr : detailErr;
  const loading = basis === 'sim' ? (!simDetail && !simErr) : (!detail && !detailErr);

  const TabButton = ({ b, diagnostic }: { b: Basis; diagnostic?: boolean }) => {
    const sc = scoreOf(b);
    const active = basis === b;
    const isSim = b === 'sim';
    const scoreTxt = sc.status && sc.status !== 'ok' && sc.status !== 'partial'
      ? (sc.status === 'loading' ? '…' : sc.status)
      : (sc.pts != null && sc.denom != null ? `${sc.pts}/${sc.denom}` : '—');
    return (
      <button onClick={() => setBasis(b)} title={BASIS_TITLE[b]}
        style={{
          background: active ? 'var(--bg-input)' : 'transparent',
          color: active ? (isSim ? '#2dd4bf' : 'var(--blue)') : 'var(--text-secondary)',
          border: `1px solid ${active ? (isSim ? '#2dd4bf' : 'var(--blue)') : 'var(--border)'}`,
          borderRadius: 10, padding: diagnostic ? '1px 7px' : '2px 9px',
          cursor: 'pointer', fontSize: diagnostic ? 10 : 11,
          fontWeight: active ? 700 : 400, opacity: diagnostic ? 0.85 : 1,
          display: 'inline-flex', alignItems: 'center', gap: 4,
        }}>
        {isSim && <span style={SIM_BADGE}>SIM</span>}
        {diagnostic ? `${BASIS_LABEL[b]} (diag)` : BASIS_LABEL[b]}
        <span style={{ color: ratioColor(sc.pts, sc.denom), fontWeight: 700 }}>{scoreTxt}</span>
        {sc.phantom != null && sc.phantom > 0 && (
          <span style={{ color: '#ffc107', fontSize: 9.5 }}
                title={`${sc.phantom} lane fire(s) paired to NO backtest edge (precision miss)`}>
            +{sc.phantom}φ</span>
        )}
      </button>
    );
  };

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6vh 16px',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
        borderRadius: 12, width: '90vw', maxWidth: 940, maxHeight: '84vh',
        display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={{ ...panelHead, flexWrap: 'wrap' }}>
          <span style={{ flex: '1 1 100%', overflow: 'hidden', textOverflow: 'ellipsis',
                         whiteSpace: 'nowrap' }}>
            Last-10 pairing · sid {sid} — {name} · ±{tol}s
          </span>
          <span style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center',
                         textTransform: 'none', letterSpacing: 0 }}>
            {CERT_BASES.map((b) => <TabButton key={b} b={b} />)}
            <span style={{ width: 1, height: 16, background: 'var(--border)', margin: '0 2px' }} />
            <TabButton b="algo" diagnostic />
          </span>
          <button onClick={onClose} style={{
            background: 'var(--bg-input)', color: 'var(--text-primary)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '2px 8px',
            cursor: 'pointer', fontSize: 12, marginLeft: 'auto',
          }}>✕ close</button>
        </div>

        <div style={{ overflowY: 'auto', overflowX: 'auto', padding: 12 }}>
          {loadErr && (
            <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 8 }}>
              ⚠ {basis === 'sim' ? 'SIM basis failed to load' : 'pairing detail failed to load'}: {loadErr}
            </div>
          )}
          {loading && <div style={{ fontSize: 13, color: 'var(--text-tertiary)' }}>Loading…</div>}

          {/* Non-scoreable SIM states — never a fabricated 0; the panel below
              still opens so the user can Run / inspect. */}
          {basis === 'sim' && simUnscored && (
            <div style={{ fontSize: 13, color: 'var(--text-tertiary)', marginBottom: 8 }}>
              {simStatus === 'none' && 'SIM not run yet for this strategy — expand the panel below and click Run.'}
              {simStatus === 'unres' && '⚠ SIM unavailable — the replay could not resolve a required ribbon (never scored 0). See the ledger below.'}
              {simStatus === 'error' && '⚠ the last SIM run errored — see Recent runs below.'}
            </div>
          )}

          {/* Shared pairing table */}
          {!simUnscored && activeRows.length > 0 && (
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr style={{ fontSize: 10.5, color: 'var(--text-tertiary)', textAlign: 'left', textTransform: 'uppercase' }}>
                  <th style={cell}>#</th>
                  <th style={cell}>bt entry (UTC)</th>
                  <th style={cell}>{nearLabel}</th>
                  <th style={cell}>Δ</th>
                  <th style={cell}>pt</th>
                  <th style={cell}>bt exit (UTC)</th>
                  <th style={cell}>{nearLabel}</th>
                  <th style={cell}>Δ</th>
                  <th style={cell}>pt</th>
                </tr>
              </thead>
              <tbody>
                {activeRows.map((t, i) => {
                  const side = t.side;
                  return (
                  <tr key={t.trade_id}>
                    <td style={{ ...cell, color: 'var(--text-tertiary)' }}>{i + 1}</td>
                    <td style={cell}>{fmtTs(t.entry_ts)}</td>
                    <td style={{ ...cell, color: 'var(--text-secondary)' }}>{fmtTs(side.entry_nearest_alert_ts ?? null)}</td>
                    <td style={cell}>{fmtDelta(side.entry_delta_sec ?? null)}</td>
                    <td style={cell}><Mark ok={side.entry_paired ?? false} /></td>
                    <td style={cell}>{fmtTs(t.exit_ts)}</td>
                    <td style={{ ...cell, color: 'var(--text-secondary)' }}>{fmtTs(side.exit_nearest_alert_ts ?? null)}</td>
                    <td style={cell}>{fmtDelta(side.exit_delta_sec ?? null)}</td>
                    <td style={cell}><Mark ok={side.exit_paired ?? false} /></td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {!simUnscored && !loading && !loadErr && activeRows.length === 0 && (
            <div style={{ fontSize: 13, color: 'var(--text-tertiary)' }}>
              {basis === 'sim'
                ? 'No SIM-covered trades in this run’s window.'
                : 'No completed backtest trades.'}
            </div>
          )}

          {/* Per-basis explainer footer */}
          {(detail || simDetail) && (
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 8, lineHeight: 1.5 }}>
              1 point per entry/exit paired within ±{tol}s (greedy 1:1). n/{detail?.denom ?? simDetail?.denom ?? 20} is
              RECALL (did a bt edge pair?); <b style={{ color: '#ffc107' }}>+Nφ</b> is PHANTOM —
              lane fires that paired to NO bt edge (precision; a lane can over-fire and still score full recall).
              Bases: <b>fired</b> = alert arrival ts (delivery truth); <b>theo</b> = bar-aligned fill_ts;{' '}
              <b style={{ color: '#2dd4bf' }}>SIM</b> = real engine over decision-time bars (logic-would-fire, NEVER delivery,
              higher fidelity); <b>algo</b> = ALGO-LANE edges (diagnostic, #176). Blank nearest/Δ = no counterpart within
              ±{dispWin ?? tol}s; the ✗ already scores it.
            </div>
          )}

          {/* SIM-only surface — collapsed by default (board #177 item 4) */}
          {basis === 'sim' && (
            <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
              <button onClick={() => setSimOpen((o) => !o)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer',
                         color: 'var(--text-secondary)', fontSize: 12, fontWeight: 600,
                         padding: '2px 0', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ transform: simOpen ? 'rotate(90deg)' : 'none',
                               transition: 'transform 0.1s' }}>▸</span>
                SIM measurement details — ledger · run · provenance · recent runs
                {simDetail?.divergences?.length
                  ? <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>
                      ({simDetail.divergences.length} known limits)</span>
                  : null}
              </button>
              {simOpen && (
                <SimPanel sid={sid} name={name} score={simDetail}
                  resultId={simResultId} timeframe={timeframe}
                  onSelectRun={setSimResultId}
                  onReload={() => setSimReload((x) => x + 1)} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
