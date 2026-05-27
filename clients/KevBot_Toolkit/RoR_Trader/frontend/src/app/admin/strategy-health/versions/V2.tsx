'use client';

/**
 * Strategy Health V2 — divergence backlog (per-event).
 *
 * Companion view to V1's per-strategy overview. Lists every unpaired
 * alert (phantom) or backtest edge (missed) within the window, each
 * tagged with an auto-classification. The "Needs investigation only"
 * filter hides classified-known events so only mysteries remain — the
 * intended UX for systematically working through real bugs.
 *
 * Auto-classification taxonomy mirrors backend constants:
 *   - needs_investigation: no known cause, dig in
 *   - timestamps_out_of_sync: source data isn't aligned (filter at V1)
 *   - phase2_signal_exit: indicator-vs-indicator exit with no L-type spec
 *   - non_fill_event: alert event_type is administrative, not a trade fill
 *   - legacy_strategy: pre-confluence-id schema, expected to diverge
 */

import { useMemo, useState } from 'react';
import Card from '@/components/Card';
import {
  useStrategyHealthBacklog,
  type DivergenceBacklogRow,
  type DivergenceClassification,
} from '@/hooks/queries/useStrategyHealthBacklog';

const WINDOW_OPTIONS: { hours: number; label: string }[] = [
  { hours: 1, label: '1h' },
  { hours: 6, label: '6h' },
  { hours: 24, label: '24h' },
  { hours: 72, label: '72h' },
  { hours: 168, label: '7d' },
];

const CLASS_TONE: Record<DivergenceClassification, { fg: string; bg: string; label: string }> = {
  needs_investigation: { fg: '#ef5350', bg: 'rgba(244,67,54,0.15)', label: 'investigate' },
  timestamps_out_of_sync: { fg: 'var(--text-muted)', bg: 'rgba(255,193,7,0.10)', label: 'timestamps' },
  phase2_signal_exit: { fg: 'var(--text-muted)', bg: 'rgba(99,102,241,0.10)', label: 'phase 2 gap' },
  non_fill_event: { fg: 'var(--text-muted)', bg: 'rgba(76,175,80,0.10)', label: 'non-fill' },
  legacy_strategy: { fg: 'var(--text-muted)', bg: 'rgba(120,120,120,0.10)', label: 'legacy' },
};

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toISOString().slice(11, 19) + 'Z';
  } catch { return iso; }
}

function fmtDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toISOString().slice(5, 10);
  } catch { return ''; }
}

export default function StrategyHealthV2() {
  const [windowHours, setWindowHours] = useState<number>(24);
  const [onlyNeedsInvestigation, setOnlyNeedsInvestigation] = useState<boolean>(true);

  const { data, isLoading, error, dataUpdatedAt, refetch } =
    useStrategyHealthBacklog({ windowHours, onlyNeedsInvestigation });

  const rowsByClass = useMemo(() => {
    const counts: Partial<Record<DivergenceClassification, number>> = {};
    for (const r of (data?.rows ?? [])) {
      counts[r.classification] = (counts[r.classification] ?? 0) + 1;
    }
    return counts;
  }, [data]);

  // Always sum the FULL backlog totals (not the filtered subset) so the
  // count chips reflect reality even when "Needs investigation only" is on.
  // To get the unfiltered total we'd need a separate query — left as a
  // future cheap optimization. For now the chips show post-filter counts.

  if (isLoading && !data) {
    return <div className="p-6"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading backlog…</p></div>;
  }
  if (error) {
    return <div className="p-6"><p className="text-sm" style={{ color: 'var(--red)' }}>Failed: {String((error as any).message || error)}</p></div>;
  }

  const lastUpdated = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—';

  return (
    <div className="p-4 space-y-4">
      <div>
        <h1 className="text-2xl font-semibold mb-1">Divergence Backlog</h1>
        <p className="text-xs" style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
          One row per unpaired event (phantom alert or missed backtest edge) in the selected
          window. Each row carries an auto-classification — keep "Needs investigation only"
          on to filter out known-cause divergence and surface real bugs to chase.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-center gap-2 mb-3" style={{ fontSize: 12 }}>
          <span style={{ color: 'var(--text-muted)' }}>Window:</span>
          {WINDOW_OPTIONS.map(o => (
            <button
              key={o.hours}
              onClick={() => setWindowHours(o.hours)}
              className="px-2 py-0.5 rounded"
              style={{
                background: windowHours === o.hours ? 'var(--accent, #4f8cf6)' : 'var(--bg-input)',
                color: windowHours === o.hours ? '#fff' : 'var(--text-muted)',
                border: '1px solid var(--border)',
                fontSize: 11,
                cursor: 'pointer',
              }}
            >
              {o.label}
            </button>
          ))}
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginLeft: 12, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={onlyNeedsInvestigation}
              onChange={e => setOnlyNeedsInvestigation(e.target.checked)}
            />
            <span>needs investigation only</span>
          </label>
          <button
            onClick={() => refetch()}
            className="px-3 py-0.5 rounded"
            style={{
              background: 'var(--bg-input)',
              border: '1px solid var(--border)',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              marginLeft: 'auto',
              fontSize: 11,
            }}
          >
            Refresh
          </button>
          <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>
            Updated {lastUpdated}
          </span>
        </div>

        {/* Classification chips */}
        <div className="flex flex-wrap gap-2 mb-3" style={{ fontSize: 11 }}>
          {(Object.keys(rowsByClass) as DivergenceClassification[]).map(cls => {
            const tone = CLASS_TONE[cls];
            return (
              <span
                key={cls}
                style={{
                  background: tone.bg, color: tone.fg,
                  padding: '2px 8px', borderRadius: 3,
                }}
              >
                {tone.label}: {rowsByClass[cls]}
              </span>
            );
          })}
          {data && data.row_count === 0 && (
            <span style={{ color: 'var(--text-muted)' }}>
              No divergence in window {onlyNeedsInvestigation ? ' (after filter)' : ''}.
            </span>
          )}
        </div>

        {/* Backlog table */}
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'left' }}>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>When</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>Type</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>Strategy</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>Sym/TF</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>Edge</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>Trigger / Reason</th>
                <th style={{ padding: '6px 8px', fontWeight: 500 }}>Classification</th>
              </tr>
            </thead>
            <tbody>
              {(data?.rows ?? []).map((r: DivergenceBacklogRow, i) => {
                const tone = CLASS_TONE[r.classification];
                return (
                  <tr key={`${r.strategy_id}:${r.event_type}:${r.timestamp}:${i}`}
                      style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding: '6px 8px', fontVariantNumeric: 'tabular-nums' }}>
                      <div>{fmtTime(r.timestamp)}</div>
                      <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                        {fmtDate(r.timestamp)}
                      </div>
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      <span
                        style={{
                          padding: '1px 6px', borderRadius: 3, fontSize: 11,
                          background: r.event_type === 'phantom'
                            ? 'rgba(255,193,7,0.18)' : 'rgba(244,67,54,0.18)',
                          color: r.event_type === 'phantom' ? '#ffc107' : '#ef5350',
                        }}
                      >
                        {r.event_type}
                      </span>
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      <div>{r.strategy_name ?? `sid ${r.strategy_id}`}</div>
                      <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                        sid {r.strategy_id}
                      </div>
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      {r.symbol ?? '—'}{r.timeframe ? ` ${r.timeframe}` : ''}
                    </td>
                    <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>
                      {r.edge ?? '—'}
                    </td>
                    <td style={{ padding: '6px 8px', fontSize: 11 }}>
                      {r.exit_trigger ?? r.exit_reason ?? '—'}
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      <span
                        title={r.classification_reason}
                        style={{
                          background: tone.bg, color: tone.fg,
                          padding: '1px 6px', borderRadius: 3, fontSize: 11,
                        }}
                      >
                        {tone.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
