'use client';

/**
 * Data Health V1 — per-(symbol, timeframe) coverage table for the
 * live_bars cache.  Backs the trader's question "is the data I'm
 * looking at actually being collected?" with concrete coverage %, gap
 * counts, freshness, and source split.
 *
 * Two modes:
 *   - Rolling (default): 1h / 4h / RTH / 24h windows, polled every 30s
 *   - Custom: explicit start/end timestamps in user's timezone, no polling.
 *     Use this to scope coverage stats to a known-good window (e.g. since
 *     a deploy) so worker work doesn't unfairly count against the metrics.
 *
 * Rows are color-coded by primary coverage column (RTH in rolling mode,
 * custom in custom mode):
 *   green  ≥ 95%   (healthy)
 *   yellow 70–95%  (intermittent loss — investigate)
 *   red    < 70%   (broken — most likely missing AM events)
 *   gray   0%      (subscribed but no data — worker not recording)
 */

import { useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import { useChartPrefs } from '@/hooks/useChartPrefs';
import { useDataHealth, type DataHealthRow, type DataHealthWindow } from '@/hooks/queries/useDataHealth';

const TF_LABEL: Record<number, string> = {
  5: '5Sec', 10: '10Sec', 30: '30Sec',
  60: '1Min', 300: '5Min', 900: '15Min', 1800: '30Min',
  3600: '1Hour', 86400: '1Day',
};

function tfLabel(secs: number): string {
  return TF_LABEL[secs] || `${secs}s`;
}

function fmtAge(seconds: number | null): string {
  if (seconds == null) return '—';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d`;
}

function covCellStyle(coverage: number, expected: number): React.CSSProperties {
  if (expected === 0) return { color: 'var(--text-muted)' };
  let bg = 'transparent';
  let color = 'var(--text-primary)';
  if (coverage >= 0.95) { bg = 'rgba(76,175,80,0.15)'; color = '#7fd081'; }
  else if (coverage >= 0.70) { bg = 'rgba(255,193,7,0.15)'; color = '#ffc107'; }
  else if (coverage > 0) { bg = 'rgba(244,67,54,0.18)'; color = '#ef5350'; }
  else { bg = 'rgba(120,120,120,0.18)'; color = 'var(--text-muted)'; }
  return { background: bg, color, padding: '2px 6px', borderRadius: 3, fontVariantNumeric: 'tabular-nums' };
}

// --- Timezone helpers (mirror of LabReplayPanel's, sharing soon) -------
const _TIMEZONE_ALIAS_TO_IANA: Record<string, string> = {
  'US/Eastern': 'America/New_York',
  'US/Central': 'America/Chicago',
  'US/Mountain': 'America/Denver',
  'US/Pacific': 'America/Los_Angeles',
  'US/Alaska': 'America/Anchorage',
  'US/Hawaii': 'Pacific/Honolulu',
};

function resolveTz(tz: string | null | undefined): string {
  if (!tz) return Intl.DateTimeFormat().resolvedOptions().timeZone;
  return _TIMEZONE_ALIAS_TO_IANA[tz] || tz;
}

function tzOffsetSecs(date: Date, tz: string): number {
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: tz, hour12: false,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const parts = dtf.formatToParts(date);
  const get = (type: string) => parseInt(parts.find(p => p.type === type)?.value ?? '0', 10);
  const asUtc = Date.UTC(get('year'), get('month') - 1, get('day'),
                         get('hour'), get('minute'), get('second'));
  return Math.round((asUtc - date.getTime()) / 1000);
}

function unixToInputValueTz(unixSec: number, tz: string): string {
  if (!isFinite(unixSec) || unixSec <= 0) return '';
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: tz, hour12: false,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const parts = dtf.formatToParts(new Date(unixSec * 1000));
  const get = (type: string) => parts.find(p => p.type === type)?.value ?? '00';
  return `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}:${get('second')}`;
}

function inputValueToUnixTz(s: string, tz: string): number {
  if (!s) return 0;
  const asIfUtcMs = Date.parse(s + 'Z');
  if (!isFinite(asIfUtcMs)) return 0;
  let offsetSec = tzOffsetSecs(new Date(asIfUtcMs), tz);
  let candidateMs = asIfUtcMs - offsetSec * 1000;
  offsetSec = tzOffsetSecs(new Date(candidateMs), tz);
  return Math.floor(asIfUtcMs / 1000) - offsetSec;
}

// --- Sort keys --------------------------------------------------------
type SortKey = 'symbol' | 'tf' | 'subs' | 'cov' | 'age' | 'gaps';
const PRIMARY_WINDOW_ROLLING = 'rth' as const;
const PRIMARY_WINDOW_CUSTOM = 'custom' as const;

function rowSortValue(r: DataHealthRow, key: SortKey, primary: string): number | string {
  switch (key) {
    case 'symbol': return r.symbol;
    case 'tf':     return r.timeframe_seconds;
    case 'subs':   return r.subscribers;
    case 'cov':    return (r.windows as any)[primary]?.coverage ?? 0;
    case 'age':    return r.latest_bar_age_sec ?? Number.MAX_SAFE_INTEGER;
    case 'gaps':   return r.bars_missing_4h;
  }
}

export default function DataHealthV1() {
  const chartPrefs = useChartPrefs();
  const resolvedTz = useMemo(() => resolveTz(chartPrefs.timezone), [chartPrefs.timezone]);

  // Mode + custom range state
  const [mode, setMode] = useState<'rolling' | 'custom'>('rolling');
  const [customStartInput, setCustomStartInput] = useState('');
  const [customEndInput, setCustomEndInput] = useState('');
  // Committed query params (only sent when user hits Apply / preset)
  const [committedStart, setCommittedStart] = useState<string | null>(null);
  const [committedEnd, setCommittedEnd] = useState<string | null>(null);

  const { data, isLoading, error, dataUpdatedAt, refetch } = useDataHealth({
    start: mode === 'custom' ? committedStart : null,
    end: mode === 'custom' ? committedEnd : null,
  });

  // Auto-fill custom inputs when entering custom mode for the first time
  useEffect(() => {
    if (mode !== 'custom') return;
    const now = Math.floor(Date.now() / 1000);
    if (!customStartInput) setCustomStartInput(unixToInputValueTz(now - 3600, resolvedTz));
    if (!customEndInput) setCustomEndInput(unixToInputValueTz(now, resolvedTz));
  }, [mode, resolvedTz, customStartInput, customEndInput]);

  const applyCustomWindow = () => {
    const s = inputValueToUnixTz(customStartInput, resolvedTz);
    const e = inputValueToUnixTz(customEndInput, resolvedTz);
    if (!isFinite(s) || !isFinite(e) || s <= 0 || e <= 0 || s >= e) return;
    setCommittedStart(new Date(s * 1000).toISOString());
    setCommittedEnd(new Date(e * 1000).toISOString());
  };

  // Active window labels in display order
  const activeMode = data?.mode || mode;
  const windowKeys: string[] = activeMode === 'custom'
    ? ['custom']
    : ['1h', '4h', 'rth', '24h'];
  const primaryWindow = activeMode === 'custom' ? PRIMARY_WINDOW_CUSTOM : PRIMARY_WINDOW_ROLLING;

  const [sortKey, setSortKey] = useState<SortKey>('cov');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const sortedRows = useMemo(() => {
    if (!data?.rows) return [];
    const rows = [...data.rows];
    rows.sort((a, b) => {
      const av = rowSortValue(a, sortKey, primaryWindow);
      const bv = rowSortValue(b, sortKey, primaryWindow);
      const cmp = typeof av === 'number' && typeof bv === 'number'
        ? av - bv
        : String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return rows;
  }, [data, sortKey, sortDir, primaryWindow]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  // Scoreboard reads the primary-window coverage (RTH in rolling, custom in custom).
  const buckets = useMemo(() => {
    if (!data?.rows) return { healthy: 0, partial: 0, broken: 0, empty: 0 };
    let healthy = 0, partial = 0, broken = 0, empty = 0;
    for (const r of data.rows) {
      const w = (r.windows as any)[primaryWindow] as DataHealthWindow | undefined;
      if (!w || w.expected === 0 || w.actual === 0) { empty++; continue; }
      if (w.coverage >= 0.95) healthy++;
      else if (w.coverage >= 0.70) partial++;
      else broken++;
    }
    return { healthy, partial, broken, empty };
  }, [data, primaryWindow]);

  if (isLoading && !data) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading data health…</p>
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

  return (
    <div className="p-4 space-y-4">
      {/* Header + mode toggle */}
      <div>
        <h1 className="text-2xl font-semibold mb-1">Data Health</h1>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Per-(symbol, timeframe) coverage of the <code>live_bars</code> cache.
          {activeMode === 'rolling' ? ' Refreshes every 30s · ' : ' '}last updated {lastUpdated} UTC.
        </p>
      </div>

      <Card>
        <div className="flex items-center gap-2 mb-2 text-xs flex-wrap" style={{ color: 'var(--text-muted)' }}>
          <span>Mode:</span>
          {(['rolling', 'custom'] as const).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className="px-3 py-0.5 rounded transition-colors"
              style={{
                background: mode === m ? 'var(--accent)' : 'var(--bg-input)',
                color: mode === m ? 'white' : 'var(--text-muted)',
                border: mode === m ? 'none' : '1px solid var(--border)',
                cursor: 'pointer',
              }}
            >
              {m === 'rolling' ? 'Rolling (1h / 4h / RTH / 24h)' : 'Custom range'}
            </button>
          ))}
          {mode === 'rolling' && (
            <button
              onClick={() => refetch()}
              className="ml-2 px-3 py-0.5 rounded transition-colors"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
            >
              Refresh
            </button>
          )}
        </div>

        {mode === 'custom' && (
          <div className="flex items-center gap-2 text-xs flex-wrap" style={{ color: 'var(--text-muted)' }}>
            <span>Window:</span>
            <label className="flex items-center gap-1">
              <span>start</span>
              <input
                type="datetime-local"
                step="1"
                value={customStartInput}
                onChange={e => setCustomStartInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') applyCustomWindow(); }}
                className="px-2 py-0.5 rounded"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontFamily: 'monospace', fontSize: '11px' }}
              />
            </label>
            <label className="flex items-center gap-1">
              <span>end</span>
              <input
                type="datetime-local"
                step="1"
                value={customEndInput}
                onChange={e => setCustomEndInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') applyCustomWindow(); }}
                className="px-2 py-0.5 rounded"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontFamily: 'monospace', fontSize: '11px' }}
              />
            </label>
            <button
              onClick={applyCustomWindow}
              className="px-3 py-0.5 rounded transition-colors"
              style={{ background: 'var(--accent)', color: 'white', border: 'none', cursor: 'pointer' }}
            >
              Apply
            </button>
            {committedStart && committedEnd && (
              <span className="ml-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                Active: {committedStart.slice(11, 19)} → {committedEnd.slice(11, 19)} UTC ({resolvedTz})
              </span>
            )}
            {!committedStart && (
              <span className="ml-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                ({resolvedTz}, second precision · press Enter or Apply to commit)
              </span>
            )}
          </div>
        )}
      </Card>

      {/* Scoreboard */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card>
          <p className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>Healthy (≥95%)</p>
          <p className="text-2xl font-semibold" style={{ color: '#7fd081' }}>{buckets.healthy}</p>
        </Card>
        <Card>
          <p className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>Partial (70–95%)</p>
          <p className="text-2xl font-semibold" style={{ color: '#ffc107' }}>{buckets.partial}</p>
        </Card>
        <Card>
          <p className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>Broken (&lt;70%)</p>
          <p className="text-2xl font-semibold" style={{ color: '#ef5350' }}>{buckets.broken}</p>
        </Card>
        <Card>
          <p className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>Empty (subs but 0 rows)</p>
          <p className="text-2xl font-semibold" style={{ color: 'var(--text-muted)' }}>{buckets.empty}</p>
        </Card>
      </div>

      {/* Coverage table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                <th onClick={() => toggleSort('symbol')} className="px-3 py-2 select-none" style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Symbol{sortKey === 'symbol' ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
                <th onClick={() => toggleSort('tf')} className="px-3 py-2 select-none" style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 500 }}>
                  TF{sortKey === 'tf' ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
                <th onClick={() => toggleSort('subs')} className="px-3 py-2 select-none" style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Subs{sortKey === 'subs' ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
                {windowKeys.map(k => (
                  <th
                    key={k}
                    onClick={() => k === primaryWindow ? toggleSort('cov') : null}
                    className="px-3 py-2 select-none"
                    style={{ cursor: k === primaryWindow ? 'pointer' : 'default', color: 'var(--text-muted)', fontWeight: 500 }}
                  >
                    {k === '1h' && '1h cov'}
                    {k === '4h' && '4h cov'}
                    {k === 'rth' && 'RTH cov'}
                    {k === '24h' && '24h cov'}
                    {k === 'custom' && 'Window cov'}
                    {k === primaryWindow && sortKey === 'cov'
                      ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                  </th>
                ))}
                <th onClick={() => toggleSort('age')} className="px-3 py-2 select-none" style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Latest{sortKey === 'age' ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
                <th onClick={() => toggleSort('gaps')} className="px-3 py-2 select-none" style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Missing (4h){sortKey === 'gaps' ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                </th>
                <th className="px-3 py-2" style={{ color: 'var(--text-muted)', fontWeight: 500 }}>
                  Source split (window)
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((r, i) => {
                // Source split is read from the primary window (RTH / custom)
                const primary = (r.windows as any)[primaryWindow] as DataHealthWindow | undefined;
                const srcParts: string[] = [];
                if (primary) {
                  if (primary.ws > 0) srcParts.push(`ws ${primary.ws}`);
                  if (primary.rest_backfill > 0) srcParts.push(`rest ${primary.rest_backfill}`);
                  if (primary.other > 0) srcParts.push(`other ${primary.other}`);
                }
                return (
                  <tr key={`${r.symbol}-${r.timeframe_seconds}-${i}`} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="px-3 py-1.5 font-medium">{r.symbol}</td>
                    <td className="px-3 py-1.5">{tfLabel(r.timeframe_seconds)}</td>
                    <td className="px-3 py-1.5" style={{ fontVariantNumeric: 'tabular-nums' }}>{r.subscribers}</td>
                    {windowKeys.map(k => {
                      const w = (r.windows as any)[k] as DataHealthWindow | undefined;
                      if (!w) return <td key={k} className="px-1 py-1.5" />;
                      return (
                        <td key={k} className="px-1 py-1.5">
                          <span style={covCellStyle(w.coverage, w.expected)}>
                            {(w.coverage * 100).toFixed(0)}%
                          </span>
                        </td>
                      );
                    })}
                    <td className="px-3 py-1.5" style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                      {fmtAge(r.latest_bar_age_sec)}
                    </td>
                    <td className="px-3 py-1.5" style={{ fontVariantNumeric: 'tabular-nums', color: r.bars_missing_4h > 0 ? '#ef5350' : 'var(--text-muted)' }}>
                      {r.bars_missing_4h > 0
                        ? `${r.bars_missing_4h} (${r.gap_events_4h} gaps)`
                        : '—'}
                    </td>
                    <td className="px-3 py-1.5" style={{ color: 'var(--text-muted)' }}>
                      {srcParts.length > 0 ? srcParts.join(' · ') : '—'}
                    </td>
                  </tr>
                );
              })}
              {sortedRows.length === 0 && (
                <tr>
                  <td colSpan={3 + windowKeys.length + 3} className="px-3 py-4 text-center" style={{ color: 'var(--text-muted)' }}>
                    No tracked symbols yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] mt-3" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
          <strong>Coverage</strong> = actual rows / expected (window-seconds ÷ TF).
          {activeMode === 'rolling' && ' RTH window starts ~09:30 ET.'}
          Color: <span style={{ color: '#7fd081' }}>green ≥95%</span>,{' '}
          <span style={{ color: '#ffc107' }}>yellow 70–95%</span>,{' '}
          <span style={{ color: '#ef5350' }}>red &lt;70%</span>,{' '}
          <span style={{ color: 'var(--text-muted)' }}>gray = subscribed but 0 rows</span>.
          {' '}<strong>Missing (4h)</strong> = gap-event count + total bars missing in the last 4 hours
          (always reflects the rolling 4h window even in custom mode — it's a separate diagnostic).
        </p>
      </Card>
    </div>
  );
}
