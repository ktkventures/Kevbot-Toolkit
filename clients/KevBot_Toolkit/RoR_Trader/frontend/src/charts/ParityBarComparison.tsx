'use client';

/**
 * ParityBarComparison — side-by-side OHLCV chart comparison for
 * Admin > Parity > Bars Comparison tab.
 *
 * Phase F.3 (2026-05-14) added two-source dropdowns so any pair of
 * {Cache, Observable, REST} can be compared. Most diagnostic pair
 * by default: Cache vs Observable (answers "did our live engine
 * match what was emitted to subscribers in real time?"). Other
 * useful pairs:
 *   - Cache vs REST: "how does our cache compare to the settled
 *     post-correction view?" (the original Phase B comparison)
 *   - Observable vs REST: "how does observable match settled?"
 *     (validates our Polygon condition filter — should be ≤ a few
 *     cents difference per Claude-app analysis).
 *
 * Phase F (window-clamped charts, pagination, disagreement histogram)
 * and F.1 (visibleRange lock) still in effect.
 *
 * Source labels:
 *   - "Cache"      live_bars (what the engine WROTE in real time)
 *   - "Observable" flat-file rebuilt 1-sec bars (what was actually emitted)
 *   - "REST"       Polygon aggregates (settled, post-correction)
 */

import { useMemo, useState } from 'react';
import TradingChart, { type CandleData } from './TradingChart';
import type { CacheBar } from '@/hooks/queries/useStrategies';
import type { BarData } from '@/hooks/queries/useMarketData';
import type { ObservableBar } from '@/hooks/queries/useAdminParity';

type SourceKey = 'cache' | 'observable' | 'rest';

interface Props {
  cacheBars: CacheBar[];
  observableBars: ObservableBar[];
  restBars: BarData[];
  cacheValueType: string | null;
  cacheNotes: string[];
  observableEmpty: boolean;
  observableLoading: boolean;
  windowStart?: string | null;
  windowEnd?: string | null;
}

interface NormBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ComparisonRow {
  minute: string;  // 'YYYY-MM-DDTHH:MM'
  left_close: number | null;
  right_close: number | null;
  close_diff: number | null;  // left - right
  left_vol: number | null;
  right_vol: number | null;
  vol_ratio: number | null;   // left / right
  flagged: boolean;
}

const PAGE_SIZE = 50;

const SOURCE_LABELS: Record<SourceKey, string> = {
  cache: 'Cache (live_bars)',
  observable: 'Observable (flat-file)',
  rest: 'REST (Polygon aggregates)',
};

const SOURCE_DESCRIPTIONS: Record<SourceKey, string> = {
  cache: 'what the live engine wrote in real time',
  observable: 'what was actually emitted to subscribers (rebuilt from flat-file trades)',
  rest: 'Polygon REST aggregates (settled, post-correction)',
};

function toCandle(bars: NormBar[]): CandleData[] {
  return bars.map((b) => ({
    time: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }));
}

function normCache(bars: CacheBar[]): NormBar[] {
  return bars.map((b) => ({
    timestamp: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }));
}

function normObservable(bars: ObservableBar[]): NormBar[] {
  return bars.map((b) => ({
    timestamp: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }));
}

function normRest(bars: BarData[]): NormBar[] {
  return bars.map((b) => ({
    timestamp: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }));
}

/** Aggregate any series to per-minute OHLCV. Open = first bar's open,
 *  close = last bar's close, high/low = max/min, volume = sum. */
function bucketByMinute(bars: NormBar[]): Map<string, NormBar> {
  const out = new Map<string, NormBar>();
  for (const b of bars) {
    const min = b.timestamp.slice(0, 16); // YYYY-MM-DDTHH:MM
    const cur = out.get(min);
    if (!cur) {
      out.set(min, {
        timestamp: min + ':00Z',
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
        volume: b.volume || 0,
      });
    } else {
      cur.high = Math.max(cur.high, b.high);
      cur.low = Math.min(cur.low, b.low);
      cur.close = b.close;
      cur.volume += b.volume || 0;
    }
  }
  return out;
}

function buildRows(left: NormBar[], right: NormBar[]): ComparisonRow[] {
  const leftMap = bucketByMinute(left);
  const rightMap = bucketByMinute(right);
  const allMinutes = new Set<string>();
  leftMap.forEach((_v, k) => allMinutes.add(k));
  rightMap.forEach((_v, k) => allMinutes.add(k));
  const sorted = Array.from(allMinutes).sort();

  return sorted.map<ComparisonRow>((min) => {
    const l = leftMap.get(min) || null;
    const r = rightMap.get(min) || null;
    const lc = l?.close ?? null;
    const rc = r?.close ?? null;
    const lv = l?.volume ?? null;
    const rv = r?.volume ?? null;
    const close_diff = lc != null && rc != null ? lc - rc : null;
    const vol_ratio = lv != null && rv != null && rv > 0 ? lv / rv : null;
    const closeFlag = close_diff != null && Math.abs(close_diff) > 0.01;
    const volFlag = vol_ratio != null && (vol_ratio < 0.95 || vol_ratio > 1.05);
    const missingSide = lc == null || rc == null;
    return {
      minute: min,
      left_close: lc,
      right_close: rc,
      close_diff,
      left_vol: lv,
      right_vol: rv,
      vol_ratio,
      flagged: closeFlag || volFlag || missingSide,
    };
  });
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
  return Math.round(v).toString();
}
function fmtRatio(v: number | null): string {
  if (v == null) return '—';
  return `${v.toFixed(3)}×`;
}

function DivergenceHistogram({ rows }: { rows: ComparisonRow[] }) {
  if (rows.length === 0) return null;
  return (
    <div>
      <div className="text-xs mb-1 flex justify-between" style={{ color: 'var(--text-muted)' }}>
        <span>Per-minute divergence (green = agree, orange = price OR volume diverges, red = both diverge / missing side)</span>
        <span>{rows.length} bars</span>
      </div>
      <div className="flex" style={{ height: 22, gap: 0, border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}>
        {rows.map((r, i) => {
          const closeFlag = r.close_diff != null && Math.abs(r.close_diff) > 0.01;
          const volFlag = r.vol_ratio != null && (r.vol_ratio < 0.95 || r.vol_ratio > 1.05);
          const missingSide = r.left_close == null || r.right_close == null;
          let color = 'rgba(34, 197, 94, 0.55)';
          if (closeFlag && volFlag) color = 'rgba(239, 68, 68, 0.65)';
          else if (closeFlag || volFlag) color = 'rgba(249, 115, 22, 0.65)';
          if (missingSide) color = 'rgba(239, 68, 68, 0.75)';
          return (
            <div key={i}
              title={`${r.minute} | close Δ ${fmtDiff(r.close_diff)} | vol ratio ${fmtRatio(r.vol_ratio)}`}
              style={{ flex: 1, background: color, minWidth: 1 }} />
          );
        })}
      </div>
    </div>
  );
}

export default function ParityBarComparison({
  cacheBars,
  observableBars,
  restBars,
  cacheValueType,
  cacheNotes,
  observableEmpty,
  observableLoading,
  windowStart,
  windowEnd,
}: Props) {
  // Phase F.3: user picks which two sources to compare. Defaults are
  // Cache (left) vs Observable (right) — the most diagnostic pair
  // (answers "did our live engine match what was emitted?").
  const [leftSource, setLeftSource] = useState<SourceKey>('cache');
  const [rightSource, setRightSource] = useState<SourceKey>('observable');
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(0);

  // Normalize each series to NormBar[] for shared handling.
  const cacheNorm = useMemo(() => normCache(cacheBars), [cacheBars]);
  const observableNorm = useMemo(() => normObservable(observableBars), [observableBars]);
  const restNorm = useMemo(() => normRest(restBars), [restBars]);

  const seriesByKey: Record<SourceKey, NormBar[]> = {
    cache: cacheNorm,
    observable: observableNorm,
    rest: restNorm,
  };

  // Clamp each side to the window for both chart and diff computation.
  const leftClamped = useMemo(() => {
    const bars = seriesByKey[leftSource];
    if (!windowStart || !windowEnd) return bars;
    return bars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [seriesByKey, leftSource, windowStart, windowEnd]);
  const rightClamped = useMemo(() => {
    const bars = seriesByKey[rightSource];
    if (!windowStart || !windowEnd) return bars;
    return bars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [seriesByKey, rightSource, windowStart, windowEnd]);

  const rows = useMemo(() => buildRows(leftClamped, rightClamped), [leftClamped, rightClamped]);

  const leftCandles = toCandle(leftClamped);
  const rightCandles = toCandle(rightClamped);

  const sortedRows = useMemo(() => (sortDesc ? [...rows].reverse() : rows), [rows, sortDesc]);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pageRows = sortedRows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  const flaggedCount = rows.filter((r) => r.flagged).length;
  const matchPct = rows.length > 0
    ? (((rows.length - flaggedCount) / rows.length) * 100).toFixed(1)
    : '—';

  function emptyState(key: SourceKey): React.ReactNode {
    if (key === 'observable' && observableLoading) {
      return <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>Loading observable bars…</div>;
    }
    if (key === 'observable' && observableEmpty) {
      return (
        <div className="text-sm p-4 space-y-1" style={{ color: 'var(--text-muted)' }}>
          <div>No observable data for this symbol/window.</div>
          <div className="text-xs">Either the symbol isn't in FLAT_FILE_SYMBOLS or the cron hasn't covered this date yet.</div>
        </div>
      );
    }
    return <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No {SOURCE_LABELS[key].toLowerCase()} data in window</div>;
  }

  return (
    <div className="space-y-4">
      {/* Source selector */}
      <div
        className="flex items-end gap-4 p-3 rounded flex-wrap"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
            Left pane
          </label>
          <select
            value={leftSource}
            onChange={(e) => setLeftSource(e.target.value as SourceKey)}
            className="text-sm px-2 py-1.5 rounded"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          >
            <option value="cache">Cache (live_bars)</option>
            <option value="observable">Observable (flat-file)</option>
            <option value="rest">REST (Polygon aggregates)</option>
          </select>
        </div>
        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
            Right pane
          </label>
          <select
            value={rightSource}
            onChange={(e) => setRightSource(e.target.value as SourceKey)}
            className="text-sm px-2 py-1.5 rounded"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          >
            <option value="cache">Cache (live_bars)</option>
            <option value="observable">Observable (flat-file)</option>
            <option value="rest">REST (Polygon aggregates)</option>
          </select>
        </div>
        <div className="text-xs" style={{ color: 'var(--text-muted)', maxWidth: 460 }}>
          <strong>{SOURCE_LABELS[leftSource]}</strong> = {SOURCE_DESCRIPTIONS[leftSource]}<br />
          <strong>{SOURCE_LABELS[rightSource]}</strong> = {SOURCE_DESCRIPTIONS[rightSource]}
        </div>
      </div>

      {/* Summary strip */}
      <div
        className="flex items-baseline gap-6 text-sm p-3 rounded"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Bars compared: </span>
          <strong>{rows.length}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Match rate: </span>
          <strong style={{
            color: rows.length > 0
              ? (flaggedCount / rows.length < 0.05 ? 'var(--green)' : flaggedCount / rows.length < 0.20 ? 'var(--orange)' : 'var(--red)')
              : 'var(--text-muted)',
          }}>
            {matchPct}%
          </strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Flagged rows: </span>
          <strong style={{ color: flaggedCount > 0 ? 'var(--orange)' : 'var(--green)' }}>{flaggedCount}</strong>
        </div>
        {leftSource === 'cache' || rightSource === 'cache' ? (
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Cache value_type: </span>
            <code>{cacheValueType ?? '—'}</code>
          </div>
        ) : null}
      </div>

      <DivergenceHistogram rows={rows} />

      {/* Side-by-side charts */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>{SOURCE_LABELS[leftSource]}</strong>
            <span style={{ color: 'var(--text-muted)' }}>{leftCandles.length} bars</span>
          </div>
          {leftCandles.length > 0 ? (
            <TradingChart
              ohlcv={leftCandles}
              height={320}
              secondsVisible={false}
              visibleRange={windowStart && windowEnd ? { from: windowStart, to: windowEnd } : null}
            />
          ) : emptyState(leftSource)}
        </div>
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>{SOURCE_LABELS[rightSource]}</strong>
            <span style={{ color: 'var(--text-muted)' }}>{rightCandles.length} bars</span>
          </div>
          {rightCandles.length > 0 ? (
            <TradingChart
              ohlcv={rightCandles}
              height={320}
              secondsVisible={false}
              visibleRange={windowStart && windowEnd ? { from: windowStart, to: windowEnd } : null}
            />
          ) : emptyState(rightSource)}
        </div>
      </div>

      {/* Cache notes (only when cache is one of the panes) */}
      {(leftSource === 'cache' || rightSource === 'cache') && cacheNotes.length > 0 && (
        <div className="text-xs p-2 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
          {cacheNotes.map((n, i) => <div key={i}>· {n}</div>)}
        </div>
      )}

      {/* Diff table */}
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
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[leftSource].split(' ')[0]} close</th>
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[rightSource].split(' ')[0]} close</th>
                <th className="text-right px-2 py-1.5">Δ</th>
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[leftSource].split(' ')[0]} vol</th>
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[rightSource].split(' ')[0]} vol</th>
                <th className="text-right px-2 py-1.5">Vol ratio</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row) => (
                <tr key={row.minute}
                  style={{ background: row.flagged ? 'rgba(255, 152, 0, 0.10)' : 'transparent', borderBottom: '1px solid var(--border)' }}>
                  <td className="px-2 py-1">{row.minute}</td>
                  <td className="px-2 py-1 text-right">{fmtPrice(row.left_close)}</td>
                  <td className="px-2 py-1 text-right">{fmtPrice(row.right_close)}</td>
                  <td className="px-2 py-1 text-right" style={{
                    color: row.close_diff != null && Math.abs(row.close_diff) > 0.01 ? 'var(--orange)' : 'var(--text-muted)',
                  }}>{fmtDiff(row.close_diff)}</td>
                  <td className="px-2 py-1 text-right">{fmtVol(row.left_vol)}</td>
                  <td className="px-2 py-1 text-right">{fmtVol(row.right_vol)}</td>
                  <td className="px-2 py-1 text-right" style={{
                    color: row.vol_ratio != null && (row.vol_ratio < 0.95 || row.vol_ratio > 1.05) ? 'var(--orange)' : 'var(--text-muted)',
                  }}>{fmtRatio(row.vol_ratio)}</td>
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
