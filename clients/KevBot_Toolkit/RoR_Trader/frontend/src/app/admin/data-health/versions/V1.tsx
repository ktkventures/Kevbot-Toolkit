'use client';

/**
 * Data Health V1 — per-(symbol, timeframe) coverage table for the
 * live_bars cache.  Backs the trader's question "is the data I'm
 * looking at actually being collected?" with concrete coverage %, gap
 * counts, freshness, and source split.
 *
 * Rows are color-coded by RTH coverage:
 *   green  ≥ 95%   (healthy)
 *   yellow 70–95%  (intermittent loss — investigate)
 *   red    < 70%   (broken — most likely missing AM events)
 *   gray   0%      (subscribed but no data — worker not recording)
 */

import { useMemo, useState } from 'react';
import Card from '@/components/Card';
import { useDataHealth, type DataHealthRow } from '@/hooks/queries/useDataHealth';

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

/** Pick a background color based on coverage fraction. */
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

type SortKey = 'symbol' | 'tf' | 'subs' | '1h' | '4h' | 'rth' | '24h' | 'age' | 'gaps';

function rowSortValue(r: DataHealthRow, key: SortKey): number | string {
  switch (key) {
    case 'symbol': return r.symbol;
    case 'tf':     return r.timeframe_seconds;
    case 'subs':   return r.subscribers;
    case '1h':     return r.windows['1h'].coverage;
    case '4h':     return r.windows['4h'].coverage;
    case 'rth':    return r.windows.rth.coverage;
    case '24h':    return r.windows['24h'].coverage;
    case 'age':    return r.latest_bar_age_sec ?? Number.MAX_SAFE_INTEGER;
    case 'gaps':   return r.bars_missing_4h;
  }
}

export default function DataHealthV1() {
  const { data, isLoading, error, dataUpdatedAt } = useDataHealth();
  const [sortKey, setSortKey] = useState<SortKey>('rth');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const sortedRows = useMemo(() => {
    if (!data?.rows) return [];
    const rows = [...data.rows];
    rows.sort((a, b) => {
      const av = rowSortValue(a, sortKey);
      const bv = rowSortValue(b, sortKey);
      const cmp = typeof av === 'number' && typeof bv === 'number'
        ? av - bv
        : String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return rows;
  }, [data, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };

  // Aggregate health scoreboard — count of (symbol, tf) buckets
  const buckets = useMemo(() => {
    if (!data?.rows) return { healthy: 0, partial: 0, broken: 0, empty: 0 };
    let healthy = 0, partial = 0, broken = 0, empty = 0;
    for (const r of data.rows) {
      const cov = r.windows.rth.coverage;
      const exp = r.windows.rth.expected;
      if (exp === 0 || r.windows.rth.actual === 0) empty++;
      else if (cov >= 0.95) healthy++;
      else if (cov >= 0.70) partial++;
      else broken++;
    }
    return { healthy, partial, broken, empty };
  }, [data]);

  if (isLoading) {
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
      <div>
        <h1 className="text-2xl font-semibold mb-1">Data Health</h1>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Per-(symbol, timeframe) coverage of the <code>live_bars</code> cache.
          Refreshes every 30s · last updated {lastUpdated} UTC.
        </p>
      </div>

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
                {(['symbol','tf','subs','1h','4h','rth','24h','age','gaps'] as SortKey[]).map(k => {
                  const labels: Record<SortKey, string> = {
                    symbol: 'Symbol', tf: 'TF', subs: 'Subs',
                    '1h': '1h cov', '4h': '4h cov', rth: 'RTH cov', '24h': '24h cov',
                    age: 'Latest', gaps: 'Missing (4h)',
                  };
                  const arrow = sortKey === k ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '';
                  return (
                    <th
                      key={k}
                      onClick={() => toggleSort(k)}
                      className="px-3 py-2 select-none"
                      style={{ cursor: 'pointer', color: 'var(--text-muted)', fontWeight: 500 }}
                    >
                      {labels[k]}{arrow}
                    </th>
                  );
                })}
                <th className="px-3 py-2" style={{ color: 'var(--text-muted)', fontWeight: 500 }}>
                  Source split (4h)
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((r, i) => {
                const w = r.windows;
                const srcParts: string[] = [];
                if (w['4h'].ws > 0) srcParts.push(`ws ${w['4h'].ws}`);
                if (w['4h'].rest_backfill > 0) srcParts.push(`rest ${w['4h'].rest_backfill}`);
                if (w['4h'].other > 0) srcParts.push(`other ${w['4h'].other}`);
                return (
                  <tr key={`${r.symbol}-${r.timeframe_seconds}-${i}`} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="px-3 py-1.5 font-medium">{r.symbol}</td>
                    <td className="px-3 py-1.5">{tfLabel(r.timeframe_seconds)}</td>
                    <td className="px-3 py-1.5" style={{ fontVariantNumeric: 'tabular-nums' }}>{r.subscribers}</td>
                    <td className="px-1 py-1.5"><span style={covCellStyle(w['1h'].coverage, w['1h'].expected)}>{(w['1h'].coverage*100).toFixed(0)}%</span></td>
                    <td className="px-1 py-1.5"><span style={covCellStyle(w['4h'].coverage, w['4h'].expected)}>{(w['4h'].coverage*100).toFixed(0)}%</span></td>
                    <td className="px-1 py-1.5"><span style={covCellStyle(w.rth.coverage, w.rth.expected)}>{(w.rth.coverage*100).toFixed(0)}%</span></td>
                    <td className="px-1 py-1.5"><span style={covCellStyle(w['24h'].coverage, w['24h'].expected)}>{(w['24h'].coverage*100).toFixed(0)}%</span></td>
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
                  <td colSpan={10} className="px-3 py-4 text-center" style={{ color: 'var(--text-muted)' }}>
                    No tracked symbols yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] mt-3" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
          <strong>Coverage</strong> = actual rows / expected (window-seconds ÷ TF). RTH window starts ~09:30 ET.
          Color: <span style={{ color: '#7fd081' }}>green ≥95%</span>,{' '}
          <span style={{ color: '#ffc107' }}>yellow 70–95%</span>,{' '}
          <span style={{ color: '#ef5350' }}>red &lt;70%</span>,{' '}
          <span style={{ color: 'var(--text-muted)' }}>gray = subscribed but 0 rows</span>.
          {' '}<strong>Missing (4h)</strong> = gap-event count + total bars missing in the last 4 hours.
        </p>
      </Card>
    </div>
  );
}
