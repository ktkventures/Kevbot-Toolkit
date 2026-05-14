'use client';

/**
 * ParityBarComparison — side-by-side OHLCV chart comparison for
 * Admin > Parity > Bars Comparison tab (Phase B).
 *
 * Left pane:  live_bars cache (what the engine SAW and stored)
 * Right pane: Polygon REST aggregates (settled, post-revision)
 *
 * Below:      diff table — one row per minute, flagged rows highlight
 *             close-price differences >$0.01 OR volume ratio outside
 *             [0.95, 1.05].
 *
 * The point: if these two diverge meaningfully, "cache coverage" is a
 * real factor in our model-alignment story. If they agree, divergence
 * has to come from the engine, NOT the data.
 */

import TradingChart, { type CandleData } from './TradingChart';
import type { BarComparisonRow } from '@/hooks/queries/useAdminParity';
import type { CacheBar } from '@/hooks/queries/useStrategies';
import type { BarData } from '@/hooks/queries/useMarketData';

interface Props {
  cacheBars: CacheBar[];
  restBars: BarData[];
  rows: BarComparisonRow[];
  cacheValueType: string | null;
  cacheNotes: string[];
}

function toCandleData(bars: { timestamp: string; open: number; high: number; low: number; close: number; volume?: number }[]): CandleData[] {
  return bars.map((b) => ({
    time: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }));
}

function fmtPrice(v: number | null): string {
  if (v == null) return '—';
  return v.toFixed(2);
}

function fmtDiff(v: number | null): string {
  if (v == null) return '—';
  const sign = v >= 0 ? '+' : '';
  return `${sign}${v.toFixed(4)}`;
}

function fmtVol(v: number | null): string {
  if (v == null) return '—';
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return v.toString();
}

function fmtRatio(v: number | null): string {
  if (v == null) return '—';
  return `${v.toFixed(3)}×`;
}

export default function ParityBarComparison({
  cacheBars,
  restBars,
  rows,
  cacheValueType,
  cacheNotes,
}: Props) {
  const cacheCandles = toCandleData(cacheBars);
  const restCandles = toCandleData(restBars);

  const flaggedCount = rows.filter((r) => r.flagged).length;
  const totalRows = rows.length;
  const matchPct = totalRows > 0
    ? (((totalRows - flaggedCount) / totalRows) * 100).toFixed(1)
    : '—';

  return (
    <div className="space-y-4">
      {/* Summary strip */}
      <div
        className="flex items-baseline gap-6 text-sm p-3 rounded"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Bars compared: </span>
          <strong>{totalRows}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Match rate: </span>
          <strong style={{
            color: totalRows > 0
              ? (flaggedCount / totalRows < 0.05 ? 'var(--green)' : flaggedCount / totalRows < 0.20 ? 'var(--orange)' : 'var(--red)')
              : 'var(--text-muted)',
          }}>
            {matchPct}%
          </strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Flagged rows: </span>
          <strong style={{ color: flaggedCount > 0 ? 'var(--orange)' : 'var(--green)' }}>
            {flaggedCount}
          </strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Cache value_type: </span>
          <code>{cacheValueType ?? '—'}</code>
        </div>
      </div>

      {/* Side-by-side charts */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>Cache (live_bars)</strong>
            <span style={{ color: 'var(--text-muted)' }}>{cacheCandles.length} bars</span>
          </div>
          {cacheCandles.length > 0 ? (
            <TradingChart ohlcv={cacheCandles} height={320} secondsVisible={false} />
          ) : (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No cache data</div>
          )}
        </div>
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>REST (Polygon aggregates)</strong>
            <span style={{ color: 'var(--text-muted)' }}>{restCandles.length} bars</span>
          </div>
          {restCandles.length > 0 ? (
            <TradingChart ohlcv={restCandles} height={320} secondsVisible={false} />
          ) : (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No REST data</div>
          )}
        </div>
      </div>

      {/* Cache notes */}
      {cacheNotes.length > 0 && (
        <div className="text-xs p-2 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
          {cacheNotes.map((n, i) => <div key={i}>· {n}</div>)}
        </div>
      )}

      {/* Diff table */}
      <div className="overflow-x-auto">
        <table className="text-xs w-full" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
              <th className="text-left px-2 py-1.5">Minute (UTC)</th>
              <th className="text-right px-2 py-1.5">Cache close</th>
              <th className="text-right px-2 py-1.5">REST close</th>
              <th className="text-right px-2 py-1.5">Δ</th>
              <th className="text-right px-2 py-1.5">Cache vol</th>
              <th className="text-right px-2 py-1.5">REST vol</th>
              <th className="text-right px-2 py-1.5">Vol ratio</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.timestamp}
                style={{
                  background: row.flagged ? 'rgba(255, 152, 0, 0.10)' : 'transparent',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <td className="px-2 py-1">{row.timestamp.slice(0, 16)}</td>
                <td className="px-2 py-1 text-right">{fmtPrice(row.cache_close)}</td>
                <td className="px-2 py-1 text-right">{fmtPrice(row.rest_close)}</td>
                <td className="px-2 py-1 text-right" style={{
                  color: row.close_diff != null && Math.abs(row.close_diff) > 0.01
                    ? 'var(--orange)' : 'var(--text-muted)',
                }}>
                  {fmtDiff(row.close_diff)}
                </td>
                <td className="px-2 py-1 text-right">{fmtVol(row.cache_vol)}</td>
                <td className="px-2 py-1 text-right">{fmtVol(row.rest_vol)}</td>
                <td className="px-2 py-1 text-right" style={{
                  color: row.vol_ratio != null && (row.vol_ratio < 0.95 || row.vol_ratio > 1.05)
                    ? 'var(--orange)' : 'var(--text-muted)',
                }}>
                  {fmtRatio(row.vol_ratio)}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center py-4" style={{ color: 'var(--text-muted)' }}>
                  No bars to compare. Select a strategy + window above.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
