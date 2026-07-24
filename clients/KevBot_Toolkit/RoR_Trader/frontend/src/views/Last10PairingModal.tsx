/**
 * Last-10 pairing modal (board #70 Phase A) — click a Last-10 score on the
 * Strategy Health table → one row per completed backtest trade showing how
 * its entry and exit paired against live alerts (±10s greedy 1:1, verbatim
 * strategy_health semantics — served by /api/strategy-health-last10/{sid}).
 * Three-panel-modal design language; portals to <body> (sidebar stacking).
 */
'use client';

import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch } from '@/lib/api/client';

interface TradeRow {
  trade_id: number;
  entry_ts: string | null; entry_nearest_alert_ts: string | null;
  entry_delta_sec: number | null; entry_paired: boolean;
  exit_ts: string | null; exit_nearest_alert_ts: string | null;
  exit_delta_sec: number | null; exit_paired: boolean;
}
interface Detail {
  strategy_id: number; points: number; denom: number;
  trade_count: number; tolerance_seconds: number; trades: TradeRow[];
}

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

export default function Last10PairingModal({ sid, name, onClose }: {
  sid: number; name: string; onClose: () => void;
}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<Detail>(`/api/strategy-health-last10/${sid}`)
      .then(setDetail)
      .catch((e) => setErr(String(e)));
  }, [sid]);

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6vh 16px',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
        borderRadius: 12, width: '90vw', maxWidth: 900, maxHeight: '82vh',
        display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={panelHead}>
          <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            Last-10 pairing · sid {sid} — {name}
            {detail && <> · {detail.points}/{detail.denom} @ ±{detail.tolerance_seconds}s</>}
          </span>
          <button onClick={onClose} style={{
            background: 'var(--bg-input)', color: 'var(--text-primary)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '2px 8px',
            cursor: 'pointer', fontSize: 12,
          }}>✕ close</button>
        </div>
        <div style={{ overflowY: 'auto', overflowX: 'auto', padding: 12 }}>
          {err && <div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div>}
          {!detail && !err && <div style={{ fontSize: 13, color: 'var(--text-tertiary)' }}>Loading…</div>}
          {detail && detail.trades.length === 0 &&
            <div style={{ fontSize: 13, color: 'var(--text-tertiary)' }}>No completed backtest trades.</div>}
          {detail && detail.trades.length > 0 && (
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr style={{ fontSize: 10.5, color: 'var(--text-tertiary)', textAlign: 'left', textTransform: 'uppercase' }}>
                  <th style={cell}>#</th>
                  <th style={cell}>bt entry (UTC)</th>
                  <th style={cell}>nearest alert</th>
                  <th style={cell}>Δ</th>
                  <th style={cell}>pt</th>
                  <th style={cell}>bt exit (UTC)</th>
                  <th style={cell}>nearest alert</th>
                  <th style={cell}>Δ</th>
                  <th style={cell}>pt</th>
                </tr>
              </thead>
              <tbody>
                {detail.trades.map((t, i) => (
                  <tr key={t.trade_id}>
                    <td style={{ ...cell, color: 'var(--text-tertiary)' }}>{i + 1}</td>
                    <td style={cell}>{fmtTs(t.entry_ts)}</td>
                    <td style={{ ...cell, color: 'var(--text-secondary)' }}>{fmtTs(t.entry_nearest_alert_ts)}</td>
                    <td style={cell}>{fmtDelta(t.entry_delta_sec)}</td>
                    <td style={cell}><Mark ok={t.entry_paired} /></td>
                    <td style={cell}>{fmtTs(t.exit_ts)}</td>
                    <td style={{ ...cell, color: 'var(--text-secondary)' }}>{fmtTs(t.exit_nearest_alert_ts)}</td>
                    <td style={cell}>{fmtDelta(t.exit_delta_sec)}</td>
                    <td style={cell}><Mark ok={t.exit_paired} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {detail && (
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 8 }}>
              1 point per entry/exit paired to a live alert within ±{detail.tolerance_seconds}s
              (greedy 1:1, same semantics as Strategy Health phantom/missed). Nearest-alert
              deltas are display-only — pairing is the 1:1 walk, so a shown Δ within tolerance
              can still be unpaired if that alert was consumed by a neighboring edge.
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
