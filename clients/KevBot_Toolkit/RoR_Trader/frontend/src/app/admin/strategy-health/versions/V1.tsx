'use client';

/**
 * Strategy Health V1 — per-strategy freshness + red-flag table.
 *
 * Surfaces the signals the data-worker + recompute paths write back to
 * the DB so we can answer "is the fleet actually getting kept current"
 * without tailing Railway logs.
 *
 * Columns:
 *   strategy | symbol | timeframe | snapshot age | KPI age |
 *   last trade age | trades | paired | phantom | missed | flags
 *
 * Sort: default by red-flag count desc (most-broken first), click any
 * header to re-sort. Filter chips at the top scope to a single flag.
 */

import { useMemo, useState } from 'react';
import Card from '@/components/Card';
import {
  useStrategyHealth,
  type StrategyHealthFlag,
  type StrategyHealthRow,
} from '@/hooks/queries/useStrategyHealth';
import { useSetSnapshotSubscription } from '@/hooks/mutations/useStrategyMutations';

// ── Formatting helpers ────────────────────────────────────────────────

function fmtAge(seconds: number | null | undefined): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return m === 0 ? `${h}h` : `${h}h ${m}m`;
  }
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  return h === 0 ? `${d}d` : `${d}d ${h}h`;
}

/** Color-coded age cell. Thresholds tuned for the data-worker cadence:
 *  green ≤ short, yellow ≤ medium, red beyond. */
function ageStyle(seconds: number | null | undefined,
                  thresholds: { green: number; yellow: number }): React.CSSProperties {
  if (seconds == null) return { color: 'var(--text-muted)' };
  let color = 'var(--text-primary)';
  let bg = 'transparent';
  if (seconds <= thresholds.green) {
    bg = 'rgba(76,175,80,0.15)'; color = '#7fd081';
  } else if (seconds <= thresholds.yellow) {
    bg = 'rgba(255,193,7,0.15)'; color = '#ffc107';
  } else {
    bg = 'rgba(244,67,54,0.18)'; color = '#ef5350';
  }
  return {
    background: bg, color,
    padding: '2px 6px', borderRadius: 3,
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  };
}

const FLAG_LABEL: Record<StrategyHealthFlag, string> = {
  legacy_no_confluence_id: 'legacy',
  no_baseline: 'no baseline',
  snapshot_missing: 'no snapshot',
  snapshot_stale: 'snapshot stale',
  kpis_stale: 'KPIs stale',
  kpis_marked_stale: 'KPIs marked stale',
  data_refresh_stale: 'data stale',
  no_recent_trades: 'no recent trades',
  parity_fail: 'parity fail',
  has_discrepancies: 'discrepancies',
  phantom_alerts: 'phantom',
  missed_alerts: 'missed',
};

/** Most flags are amber; only a few are hard-red. */
const FLAG_TONE: Record<StrategyHealthFlag, 'red' | 'amber' | 'gray'> = {
  legacy_no_confluence_id: 'gray',
  no_baseline: 'gray',
  snapshot_missing: 'red',
  snapshot_stale: 'red',
  kpis_stale: 'amber',
  kpis_marked_stale: 'amber',
  data_refresh_stale: 'amber',
  no_recent_trades: 'amber',
  parity_fail: 'red',
  has_discrepancies: 'amber',
  phantom_alerts: 'amber',   // alert without backtest — investigate
  missed_alerts: 'red',      // backtest fired, algo didn't — bigger deal
};

function flagChipStyle(tone: 'red' | 'amber' | 'gray'): React.CSSProperties {
  const palette = {
    red:   { bg: 'rgba(244,67,54,0.18)', fg: '#ef5350' },
    amber: { bg: 'rgba(255,193,7,0.18)', fg: '#ffc107' },
    gray:  { bg: 'rgba(120,120,120,0.18)', fg: 'var(--text-muted)' },
  }[tone];
  return {
    background: palette.bg, color: palette.fg,
    padding: '1px 6px', borderRadius: 3, fontSize: 11,
    whiteSpace: 'nowrap', marginRight: 4, marginBottom: 2,
    display: 'inline-block',
  };
}

// ── Sortable table ────────────────────────────────────────────────────

type SortKey =
  | 'flags' | 'name' | 'symbol' | 'timeframe'
  | 'snapshot' | 'kpis' | 'lastTrade' | 'trades'
  | 'lastBt' | 'lastAlert'
  | 'paired' | 'phantom' | 'missed';

function rowSortValue(r: StrategyHealthRow, key: SortKey): number | string {
  switch (key) {
    case 'flags':     return -r.red_flags.length; // most flags first by default asc
    case 'name':      return r.name ?? '';
    case 'symbol':    return r.symbol ?? '';
    case 'timeframe': return r.timeframe ?? '';
    case 'snapshot':  return r.snapshot_age_sec ?? Number.POSITIVE_INFINITY;
    case 'kpis':      return r.kpis_age_sec ?? Number.POSITIVE_INFINITY;
    case 'lastTrade': return r.last_entry_age_sec ?? Number.POSITIVE_INFINITY;
    case 'lastBt':    return r.last_backtest_created_age_sec ?? Number.POSITIVE_INFINITY;
    case 'lastAlert': return r.last_alert_age_sec ?? Number.POSITIVE_INFINITY;
    case 'trades':    return -r.trade_count_backtest;
    case 'paired':    return -r.paired_count;
    case 'phantom':   return -r.phantom_count;
    case 'missed':    return -r.missed_count;
  }
}

// ── Component ─────────────────────────────────────────────────────────

const WINDOW_OPTIONS: { hours: number; label: string }[] = [
  { hours: 1,   label: '1h' },
  { hours: 3,   label: '3h' },
  { hours: 6,   label: '6h' },
  { hours: 12,  label: '12h' },
  { hours: 24,  label: '24h' },
  { hours: 48,  label: '48h' },
  { hours: 72,  label: '72h' },
  { hours: 168, label: '7d' },
];

function windowLabel(hours: number): string {
  const m = WINDOW_OPTIONS.find(o => o.hours === hours);
  if (m) return m.label;
  if (hours >= 24) return `${Math.round(hours / 24)}d`;
  return `${hours}h`;
}

// Format an ISO timestamp into a datetime-local input value
// (YYYY-MM-DDTHH:MM in the browser's local TZ).
function isoToLocalInput(iso: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// Parse a datetime-local input value (browser local TZ) into an ISO
// 8601 string with UTC offset.
function localInputToIso(s: string): string {
  if (!s) return '';
  const d = new Date(s);  // Date constructor interprets as local TZ
  if (isNaN(d.getTime())) return '';
  return d.toISOString();
}

export default function StrategyHealthV1() {
  // Mode = 'rolling' uses windowHours chips. 'custom' uses fixed
  // start/end timestamps (frozen — no auto-refetch).
  const [mode, setMode] = useState<'rolling' | 'custom'>('rolling');
  const [windowHours, setWindowHours] = useState<number>(24);
  // Custom-mode inputs (as datetime-local strings, browser TZ).
  // Default start = 24h ago; default end = now. Backend converts both
  // to UTC.
  const [customStartInput, setCustomStartInput] = useState<string>(
    () => isoToLocalInput(new Date(Date.now() - 24 * 3600 * 1000).toISOString()));
  const [customEndInput, setCustomEndInput] = useState<string>(
    () => isoToLocalInput(new Date().toISOString()));

  const customStartIso = mode === 'custom' ? localInputToIso(customStartInput) : null;
  const customEndIso = mode === 'custom' ? localInputToIso(customEndInput) : null;

  const { data, isLoading, error, dataUpdatedAt, refetch } =
    useStrategyHealth({
      windowHours,
      start: mode === 'custom' ? customStartIso : null,
      end: mode === 'custom' ? customEndIso : null,
    });

  const [sortKey, setSortKey] = useState<SortKey>('flags');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [activeFlag, setActiveFlag] = useState<StrategyHealthFlag | null>(null);
  const [includeLegacy, setIncludeLegacy] = useState(false);
  // 2026-05-27: filter out rows where last alert and last backtest are
  // materially out of sync — phantom/missed numbers aren't meaningful
  // when one source is days stale relative to the other. Default ON so
  // the page shows only divergence worth investigating.
  const [hideUpsideDown, setHideUpsideDown] = useState(true);

  const filteredRows = useMemo(() => {
    if (!data?.rows) return [];
    let rows = data.rows;
    if (!includeLegacy) {
      rows = rows.filter(r => !r.red_flags.includes('legacy_no_confluence_id'));
    }
    if (hideUpsideDown) {
      rows = rows.filter(r => !r.timestamps_upside_down);
    }
    if (activeFlag) {
      rows = rows.filter(r => r.red_flags.includes(activeFlag));
    }
    const sorted = [...rows].sort((a, b) => {
      const av = rowSortValue(a, sortKey);
      const bv = rowSortValue(b, sortKey);
      const cmp = typeof av === 'number' && typeof bv === 'number'
        ? av - bv
        : String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return sorted;
  }, [data, sortKey, sortDir, activeFlag, includeLegacy, hideUpsideDown]);

  // Summary: counts by flag, scoped to non-legacy strategies.
  const summary = useMemo(() => {
    const visible = (data?.rows ?? []).filter(
      r => includeLegacy || !r.red_flags.includes('legacy_no_confluence_id'));
    const flagCounts = new Map<StrategyHealthFlag, number>();
    let healthy = 0;
    for (const r of visible) {
      if (r.red_flags.length === 0) {
        healthy++;
      } else {
        for (const f of r.red_flags) {
          flagCounts.set(f, (flagCounts.get(f) ?? 0) + 1);
        }
      }
    }
    return { total: visible.length, healthy, flagCounts };
  }, [data, includeLegacy]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  const setSnapshotSub = useSetSnapshotSubscription();

  if (isLoading && !data) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading strategy health…</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--red)' }}>
          Failed to load: {String((error as any).message || error)}
        </p>
      </div>
    );
  }

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toISOString().slice(11, 19)
    : '—';

  // Sorted list of flag chips for the filter row.
  const flagChips = Array.from(summary.flagCounts.entries())
    .sort((a, b) => b[1] - a[1]);

  const arrow = (k: SortKey) =>
    sortKey === k ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  return (
    <div className="p-4 space-y-4">
      <div>
        <h1 className="text-2xl font-semibold mb-1">Strategy Health</h1>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Per-strategy freshness + red flags. Refreshes every 30s · last updated {lastUpdated} UTC.
        </p>
      </div>

      {/* Summary scoreboard */}
      <Card>
        <div className="flex flex-wrap items-center gap-3 text-xs"
             style={{ color: 'var(--text-muted)' }}>
          <div>
            <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              {summary.total}
            </span>
            {' strategies'}
          </div>
          <div>
            <span style={{ color: '#7fd081', fontWeight: 600 }}>
              {summary.healthy} healthy
            </span>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span>Window:</span>
            {WINDOW_OPTIONS.map(o => (
              <button
                key={o.hours}
                onClick={() => { setMode('rolling'); setWindowHours(o.hours); }}
                className="px-2 py-0.5 rounded transition-colors"
                style={{
                  background: (mode === 'rolling' && windowHours === o.hours)
                    ? 'var(--accent)' : 'var(--bg-input)',
                  color: (mode === 'rolling' && windowHours === o.hours)
                    ? 'white' : 'var(--text-muted)',
                  border: (mode === 'rolling' && windowHours === o.hours)
                    ? 'none' : '1px solid var(--border)',
                  cursor: 'pointer',
                  fontSize: 11,
                }}
              >
                {o.label}
              </button>
            ))}
            <button
              key="custom"
              onClick={() => setMode('custom')}
              className="px-2 py-0.5 rounded transition-colors"
              style={{
                background: mode === 'custom' ? 'var(--accent)' : 'var(--bg-input)',
                color: mode === 'custom' ? 'white' : 'var(--text-muted)',
                border: mode === 'custom' ? 'none' : '1px solid var(--border)',
                cursor: 'pointer',
                fontSize: 11,
              }}
              title="Pin a fixed [start, end] window. Useful for overnight-into-morning analysis: set start tonight, then update end tomorrow."
            >
              Custom
            </button>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, cursor: 'pointer', marginLeft: 8 }}>
              <input
                type="checkbox"
                checked={includeLegacy}
                onChange={e => setIncludeLegacy(e.target.checked)}
              />
              <span>include legacy</span>
            </label>
            <label
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4, cursor: 'pointer', marginLeft: 8 }}
              title="Hide strategies where last alert and last backtest trade are >1h apart. Phantom/missed counts aren't meaningful when one source is stale relative to the other."
            >
              <input
                type="checkbox"
                checked={hideUpsideDown}
                onChange={e => setHideUpsideDown(e.target.checked)}
              />
              <span>hide upside-down timestamps</span>
            </label>
            <button
              onClick={() => refetch()}
              className="px-3 py-0.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-muted)',
                cursor: 'pointer',
              }}
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Custom-mode datetime inputs — only visible when Custom is selected */}
        {mode === 'custom' && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs"
               style={{ color: 'var(--text-muted)' }}>
            <span>Range:</span>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span>Start</span>
              <input
                type="datetime-local"
                value={customStartInput}
                onChange={e => setCustomStartInput(e.target.value)}
                style={{
                  padding: '2px 6px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  color: 'var(--text-primary)',
                  fontSize: 11,
                }}
              />
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span>End</span>
              <input
                type="datetime-local"
                value={customEndInput}
                onChange={e => setCustomEndInput(e.target.value)}
                style={{
                  padding: '2px 6px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  borderRadius: 3,
                  color: 'var(--text-primary)',
                  fontSize: 11,
                }}
              />
            </label>
            <button
              onClick={() => setCustomEndInput(isoToLocalInput(new Date().toISOString()))}
              className="px-2 py-0.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: 11,
              }}
              title="Set End to current time"
            >
              End → Now
            </button>
            <span style={{ opacity: 0.7 }}>
              (frozen — no auto-refresh in custom mode)
            </span>
          </div>
        )}

        {/* Filter chips */}
        {flagChips.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-1 text-xs">
            <span style={{ color: 'var(--text-muted)', marginRight: 6 }}>
              Filter:
            </span>
            <button
              onClick={() => setActiveFlag(null)}
              style={{
                ...flagChipStyle('gray'),
                outline: activeFlag === null ? '1px solid var(--accent)' : 'none',
                cursor: 'pointer',
              }}
            >
              all
            </button>
            {flagChips.map(([flag, count]) => (
              <button
                key={flag}
                onClick={() => setActiveFlag(activeFlag === flag ? null : flag)}
                style={{
                  ...flagChipStyle(FLAG_TONE[flag]),
                  outline: activeFlag === flag ? '1px solid var(--accent)' : 'none',
                  cursor: 'pointer',
                }}
              >
                {FLAG_LABEL[flag]} · {count}
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Main table */}
      <Card>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'left' }}>
                <Th onClick={() => toggleSort('name')}      label={`Strategy${arrow('name')}`} />
                <Th onClick={() => toggleSort('symbol')}    label={`Symbol${arrow('symbol')}`} />
                <Th onClick={() => toggleSort('timeframe')} label={`TF${arrow('timeframe')}`} />
                <Th onClick={() => toggleSort('snapshot')}  label={`Snapshot${arrow('snapshot')}`} />
                <Th onClick={() => toggleSort('kpis')}      label={`KPIs${arrow('kpis')}`} />
                <Th onClick={() => toggleSort('lastTrade')} label={`Last trade${arrow('lastTrade')}`} />
                <Th onClick={() => toggleSort('lastBt')}    label={`Last BT${arrow('lastBt')}`} />
                <Th onClick={() => toggleSort('lastAlert')} label={`Last alert${arrow('lastAlert')}`} />
                <Th onClick={() => toggleSort('trades')}    label={`Trades${arrow('trades')}`} align="right" />
                <Th onClick={() => toggleSort('paired')}    label={`Paired ${mode === 'custom' ? 'range' : windowLabel(windowHours)}${arrow('paired')}`} align="right" />
                <Th onClick={() => toggleSort('phantom')}   label={`Phantom ${mode === 'custom' ? 'range' : windowLabel(windowHours)}${arrow('phantom')}`} align="right" />
                <Th onClick={() => toggleSort('missed')}    label={`Missed ${mode === 'custom' ? 'range' : windowLabel(windowHours)}${arrow('missed')}`} align="right" />
                <th style={{ padding: '6px 8px', fontWeight: 500 }}
                    title="Snapshot subscription. ON = data-worker maintains a backtest snapshot. OFF = strategy is parked in the snapshot lane (alerts unaffected).">
                  Sub
                </th>
                <Th onClick={() => toggleSort('flags')}     label={`Flags${arrow('flags')}`} />
              </tr>
            </thead>
            <tbody>
              {filteredRows.map(r => (
                <tr key={`${r.user_id}:${r.strategy_id}`}
                    style={{ borderTop: '1px solid var(--border)' }}>
                  <td style={{ padding: '6px 8px' }}>
                    <div style={{ color: 'var(--text-primary)' }}>
                      {r.name || `sid ${r.strategy_id}`}
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                      sid {r.strategy_id} · {r.direction} · {r.backtest_model || '—'}
                      {r.forward_testing ? ' · fwd' : ''}
                    </div>
                  </td>
                  <td style={{ padding: '6px 8px', fontVariantNumeric: 'tabular-nums' }}>
                    {r.symbol || '—'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>{r.timeframe || '—'}</td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={ageStyle(r.snapshot_age_sec, { green: 600, yellow: 3600 })}>
                      {fmtAge(r.snapshot_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={ageStyle(r.kpis_age_sec, { green: 3600, yellow: 86400 })}>
                      {fmtAge(r.kpis_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={ageStyle(r.last_entry_age_sec, { green: 86400, yellow: 7 * 86400 })}>
                      {fmtAge(r.last_entry_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}
                      title={r.last_backtest_created_at ?? 'no backtest trades on file'}>
                    <span style={ageStyle(r.last_backtest_created_age_sec, { green: 3600, yellow: 86400 })}>
                      {fmtAge(r.last_backtest_created_age_sec)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 8px' }}
                      title={r.last_alert_at
                        ? r.last_alert_at
                        + (r.timestamps_upside_down && r.upside_down_delta_sec
                            ? ` (Δ ${Math.round((r.upside_down_delta_sec ?? 0) / 60)}m vs last BT — upside-down)`
                            : '')
                        : 'no alerts in last 30d'}>
                    <span style={ageStyle(r.last_alert_age_sec, { green: 3600, yellow: 86400 })}>
                      {fmtAge(r.last_alert_age_sec)}
                    </span>
                    {r.timestamps_upside_down && (
                      <span
                        style={{ marginLeft: 4, color: '#ffc107', fontSize: 10 }}
                        title="upside-down: alert and backtest timestamps >1h apart"
                      >
                        ⇅
                      </span>
                    )}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.trade_count_backtest === 0
                                 ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                    {r.trade_count_backtest}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.paired_count > 0 ? '#7fd081' : 'var(--text-muted)' }}>
                    {r.paired_count || '—'}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.phantom_count > 0 ? '#ffc107' : 'var(--text-muted)' }}>
                    {r.phantom_count || '—'}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right',
                               fontVariantNumeric: 'tabular-nums',
                               color: r.missed_count > 0 ? '#ef5350' : 'var(--text-muted)' }}>
                    {r.missed_count || '—'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <button
                      onClick={() => setSnapshotSub.mutate({
                        id: r.strategy_id,
                        enabled: !r.snapshot_subscribe_enabled,
                      })}
                      disabled={setSnapshotSub.isPending}
                      title={r.snapshot_subscribe_enabled
                        ? 'Subscribed — click to park (data-worker will skip this strategy)'
                        : 'Parked — click to resubscribe'}
                      style={{
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        borderRadius: 3,
                        padding: '1px 6px',
                        fontSize: 11,
                        cursor: setSnapshotSub.isPending ? 'wait' : 'pointer',
                        color: r.snapshot_subscribe_enabled ? '#7fd081' : 'var(--text-muted)',
                      }}
                    >
                      {r.snapshot_subscribe_enabled ? 'ON' : 'OFF'}
                    </button>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    {r.red_flags.length === 0 ? (
                      <span style={flagChipStyle('gray')}>ok</span>
                    ) : (
                      r.red_flags.map(f => (
                        <span key={f} style={flagChipStyle(FLAG_TONE[f])}>
                          {FLAG_LABEL[f]}
                        </span>
                      ))
                    )}
                  </td>
                </tr>
              ))}
              {filteredRows.length === 0 && (
                <tr>
                  <td colSpan={10} style={{ padding: 16, textAlign: 'center',
                                            color: 'var(--text-muted)' }}>
                    No strategies match the current filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function Th({ label, onClick, align = 'left' }: {
  label: string;
  onClick: () => void;
  align?: 'left' | 'right';
}) {
  return (
    <th
      onClick={onClick}
      style={{
        padding: '6px 8px',
        cursor: 'pointer',
        userSelect: 'none',
        textAlign: align,
        fontWeight: 500,
      }}
    >
      {label}
    </th>
  );
}
