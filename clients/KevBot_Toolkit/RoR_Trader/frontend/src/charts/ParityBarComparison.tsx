'use client';

/**
 * ParityBarComparison — side-by-side OHLCV chart comparison for
 * Admin > Parity > Bars Comparison tab.
 *
 * Phase B shipped the basic side-by-side view. Phase F (2026-05-14)
 * adds:
 *   - Pagination on the per-minute diff table (was crashing browsers
 *     with ~2880 rows for a 2-day window).
 *   - Window-clamped charts: both panes get exactly the same time
 *     range so they line up without manual scrolling.
 *   - Disagreement histogram row between summary + charts that paints
 *     green/red per minute so divergence zones are visible at a glance.
 *
 * Left pane:  live_bars cache (what the engine SAW and stored)
 * Right pane: Polygon REST aggregates (settled, post-revision)
 */

import { useMemo, useState } from 'react';
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
  /** Window-start ISO (inclusive). Both charts will be clamped to this range. */
  windowStart?: string | null;
  /** Window-end ISO (exclusive). */
  windowEnd?: string | null;
}

const PAGE_SIZE = 50;

function toCandleData(
  bars: { timestamp: string; open: number; high: number; low: number; close: number; volume?: number }[],
): CandleData[] {
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

/** Disagreement histogram — one cell per minute, painted green
 * (close+vol agree), orange (one disagrees), or red (both disagree
 * or one side is missing entirely). */
function DivergenceHistogram({ rows }: { rows: BarComparisonRow[] }) {
  if (rows.length === 0) return null;
  // sorted ascending by minute already from buildRows
  return (
    <div>
      <div className="text-xs mb-1 flex justify-between" style={{ color: 'var(--text-muted)' }}>
        <span>Per-minute divergence (green = agree, orange = price OR volume diverges, red = both diverge / missing side)</span>
        <span>{rows.length} bars</span>
      </div>
      <div
        className="flex"
        style={{
          height: 22,
          gap: 0,
          border: '1px solid var(--border)',
          borderRadius: 4,
          overflow: 'hidden',
        }}
      >
        {rows.map((r, i) => {
          const closeFlag = r.close_diff != null && Math.abs(r.close_diff) > 0.01;
          const volFlag = r.vol_ratio != null && (r.vol_ratio < 0.95 || r.vol_ratio > 1.05);
          const missingSide = r.cache_close == null || r.rest_close == null;
          let color = 'rgba(34, 197, 94, 0.55)';     // green = agree
          if (closeFlag && volFlag) color = 'rgba(239, 68, 68, 0.65)';      // red
          else if (closeFlag || volFlag) color = 'rgba(249, 115, 22, 0.65)'; // orange
          if (missingSide) color = 'rgba(239, 68, 68, 0.75)';
          return (
            <div
              key={i}
              title={`${r.timestamp.slice(0, 16)} | close Δ ${fmtDiff(r.close_diff)} | vol ratio ${fmtRatio(r.vol_ratio)}`}
              style={{
                flex: 1,
                background: color,
                minWidth: 1,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function ParityBarComparison({
  cacheBars,
  restBars,
  rows,
  cacheValueType,
  cacheNotes,
  windowStart,
  windowEnd,
}: Props) {
  // Phase F: clamp both candle datasets to the same window so they
  // line up without manual zoom/scroll. Falls back to "use all data"
  // when window props aren't provided (older callers).
  const cacheClamped = useMemo(() => {
    if (!windowStart || !windowEnd) return cacheBars;
    return cacheBars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [cacheBars, windowStart, windowEnd]);
  const restClamped = useMemo(() => {
    if (!windowStart || !windowEnd) return restBars;
    return restBars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [restBars, windowStart, windowEnd]);

  const cacheCandles = toCandleData(cacheClamped);
  const restCandles = toCandleData(restClamped);

  // Phase F: clamp diff rows to the same window for consistent stats.
  const rowsClamped = useMemo(() => {
    if (!windowStart || !windowEnd) return rows;
    return rows.filter((r) => r.timestamp >= windowStart && r.timestamp < windowEnd);
  }, [rows, windowStart, windowEnd]);

  // Phase F: pagination state — default newest-first (descending) so
  // the most relevant minutes are at top.
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(0);

  const sortedRows = useMemo(() => {
    return sortDesc ? [...rowsClamped].reverse() : rowsClamped;
  }, [rowsClamped, sortDesc]);

  const totalPages = Math.max(1, Math.ceil(sortedRows.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pageRows = sortedRows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  const flaggedCount = rowsClamped.filter((r) => r.flagged).length;
  const totalRows = rowsClamped.length;
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

      {/* Phase F: divergence histogram strip */}
      <DivergenceHistogram rows={rowsClamped} />

      {/* Side-by-side charts (window-clamped for apples-to-apples) */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>Cache (live_bars)</strong>
            <span style={{ color: 'var(--text-muted)' }}>{cacheCandles.length} bars</span>
          </div>
          {cacheCandles.length > 0 ? (
            <TradingChart ohlcv={cacheCandles} height={320} secondsVisible={false} />
          ) : (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No cache data in window</div>
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
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No REST data in window</div>
          )}
        </div>
      </div>

      {/* Cache notes */}
      {cacheNotes.length > 0 && (
        <div className="text-xs p-2 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
          {cacheNotes.map((n, i) => <div key={i}>· {n}</div>)}
        </div>
      )}

      {/* Diff table — paginated, sortable */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {sortedRows.length > 0 && (
              <>Page <strong>{safePage + 1}</strong> of <strong>{totalPages}</strong> · showing {pageRows.length} of {sortedRows.length} bars</>
            )}
          </div>
          <div className="flex gap-2 items-center">
            <button
              type="button"
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text)' }}
              onClick={() => setSortDesc(!sortDesc)}
            >
              Sort: {sortDesc ? 'newest first' : 'oldest first'}
            </button>
            <button
              type="button"
              disabled={safePage === 0}
              className="text-xs px-2 py-1 rounded disabled:opacity-40"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text)' }}
              onClick={() => setPage(Math.max(0, safePage - 1))}
            >
              ← Prev
            </button>
            <button
              type="button"
              disabled={safePage >= totalPages - 1}
              className="text-xs px-2 py-1 rounded disabled:opacity-40"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text)' }}
              onClick={() => setPage(Math.min(totalPages - 1, safePage + 1))}
            >
              Next →
            </button>
          </div>
        </div>
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
              {pageRows.map((row) => (
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
              {sortedRows.length === 0 && (
                <tr>
                  <td colSpan={7} className="text-center py-4" style={{ color: 'var(--text-muted)' }}>
                    No bars to compare in this window.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
