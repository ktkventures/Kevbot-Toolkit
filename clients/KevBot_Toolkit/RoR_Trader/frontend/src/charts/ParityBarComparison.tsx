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
import {
  useRestBars,
  useTradeBars,
  type ObservableBar,
  type TradeBarSource,
} from '@/hooks/queries/useAdminParity';

type SourceKey =
  | 'cache'
  | 'cache_backfill'
  | 'cache_latest'
  | 'observable'
  | 'rest'
  | 'rest1s'
  // Phase H.2 (2026-05-15): three trade-channel shadow variants.
  | 't_wait0'
  | 't_wait200'
  | 't_wait500'
  // 10Sec grace-window shadow (2026-05-18): grace past bar_end.
  // 2.5s = the measured-lag "precise best guess".
  | 'grace_2s'
  | 'grace_2.5s'
  | 'grace_3s'
  | 'grace_4s';

interface Props {
  /** Cache decision-time view (first_* columns — what the engine saw at bar close). */
  cacheBars: CacheBar[];
  /** Phase G.3: cache POST-rebroadcast view (open/close columns — revised
   *  by Polygon's late-print updates). Compare against cacheBars to see
   *  rebroadcast deltas. */
  cacheBarsLatest?: CacheBar[];
  observableBars: ObservableBar[];
  /** Polygon REST 1Min aggregates — the established baseline source.
   *  Available only at TF ≥ 1Min. */
  restBars: BarData[];
  cacheValueType: string | null;
  cacheNotes: string[];
  observableEmpty: boolean;
  observableLoading: boolean;
  windowStart?: string | null;
  windowEnd?: string | null;
  /** Symbol — needed by useRestBars (rest1s source) on TF change. */
  symbol?: string | null;
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
  /** Bucket-start epoch seconds (UTC). Used for sorting + keying. */
  bucketStart: number;
  /** Display label — Phase G.0: now includes seconds for sub-minute TFs. */
  label: string;
  left_close: number | null;
  right_close: number | null;
  close_diff: number | null;  // left - right
  left_vol: number | null;
  right_vol: number | null;
  vol_ratio: number | null;   // left / right
  flagged: boolean;
}

/** Pick TFs the user can select. Sub-minute options auto-disable REST
 *  (Polygon REST aggregates API minimum is 1Min). */
interface TfOption {
  value: number;   // seconds
  label: string;
  restAvailable: boolean;
}

// Phase G.2 (2026-05-15): two REST sources now.
//   - 'rest'   = Polygon 1Min aggs (established baseline; ≥1Min only)
//   - 'rest1s' = Polygon 1-sec aggs rolled up to selected TF (any TF)
// The TfOption.restAvailable flag now governs only the 1Min source.
const TF_OPTIONS: TfOption[] = [
  { value: 10,  label: '10Sec',  restAvailable: false },
  { value: 30,  label: '30Sec',  restAvailable: false },
  { value: 60,  label: '1Min',   restAvailable: true  },
  { value: 300, label: '5Min',   restAvailable: true  },
  { value: 900, label: '15Min',  restAvailable: true  },
];

function formatBucketLabel(epochSec: number, tfSeconds: number): string {
  const d = new Date(epochSec * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  // ISO-style display in UTC, granularity follows TF
  const base = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  if (tfSeconds < 60) {
    return `${base}:${pad(d.getUTCSeconds())}`;
  }
  return base;
}

const PAGE_SIZE = 50;

const SOURCE_LABELS: Record<SourceKey, string> = {
  cache: 'Cache (decision-time)',
  cache_backfill: 'Cache + backfill',
  cache_latest: 'Cache (post-rebroadcast)',
  observable: 'Observable (flat-file)',
  rest: 'REST 1Min aggs',
  rest1s: 'REST 1Sec aggs',
  t_wait0: 'Trade ch. (wait 0ms)',
  t_wait200: 'Trade ch. (wait 200ms)',
  t_wait500: 'Trade ch. (wait 500ms)',
  grace_2s: 'Grace 2s (10Sec shadow)',
  'grace_2.5s': 'Grace 2.5s (10Sec shadow)',
  grace_3s: 'Grace 3s (10Sec shadow)',
  grace_4s: 'Grace 4s (10Sec shadow)',
};

const SOURCE_DESCRIPTIONS: Record<SourceKey, string> = {
  cache: "first_* columns, ws_agg only — what the live engine SAW at bar close. Genuine cache gaps left visible (excludes REST backfill rows).",
  cache_backfill: "first_* columns + REST gap-fill — the decision-time cache with rest_backfill rows patching minutes the ws_agg path missed. Gap-free view.",
  cache_latest: "open/close columns — POST-rebroadcast cache state (after Polygon's revised bars arrive). Closer to REST settled.",
  observable: 'what was actually emitted to subscribers (rebuilt from flat-file trades using sip_timestamp)',
  rest: "Polygon's settled 1Min aggregates — yesterday's gospel-truth baseline",
  rest1s: "Polygon's 1Sec aggregates rolled up to selected TF — supports sub-minute",
  t_wait0: 'Phase H shadow: bars built from Polygon T (trade) channel; bucket closes immediately when next-bucket trade arrives. Zero wait = strict decision-time.',
  t_wait200: 'Phase H shadow: bars built from Polygon T (trade) channel; bucket holds open 200ms past period end to catch in-flight late prints.',
  t_wait500: 'Phase H shadow: bars built from Polygon T (trade) channel; bucket holds open 500ms past period end. Most forgiving — closest to settled.',
  grace_2s: 'Grace shadow: 10Sec bars from the A per-second stream, bucket held open 2s past bar_end before close. Tightest — measured to just miss the last per-second bar (~B+12.05 arrival).',
  'grace_2.5s': 'Grace shadow: 10Sec bars, bucket held open 2.5s past bar_end. The measured-lag "precise best guess" — catches the structural feed lag with ~0.4s margin, 0.5s less latency than 3s.',
  grace_3s: 'Grace shadow: 10Sec bars from the A per-second stream, bucket held open 3s past bar_end. Comfortably covers the measured ~3s A-channel ingestion lag.',
  grace_4s: 'Grace shadow: 10Sec bars from the A per-second stream, bucket held open 4s past bar_end. Most margin against late-feed outliers.',
};

function isTradeSource(key: SourceKey): boolean {
  return key === 't_wait0' || key === 't_wait200' || key === 't_wait500';
}

function isGraceSource(key: SourceKey): boolean {
  return key === 'grace_2s' || key === 'grace_2.5s'
    || key === 'grace_3s' || key === 'grace_4s';
}

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

/** Aggregate any series to a target TF (seconds). Open = first bar's
 *  open, close = last bar's close, high/low = max/min, volume = sum.
 *  Phase G.0: generalized from bucketByMinute to support any TF. Keys
 *  are bucket-start epoch seconds (UTC-aligned via floor(epoch / tf)).
 */
function bucketByTf(bars: NormBar[], tfSeconds: number): Map<number, NormBar> {
  const out = new Map<number, NormBar>();
  for (const b of bars) {
    const epoch = Math.floor(new Date(b.timestamp).getTime() / 1000);
    if (!isFinite(epoch)) continue;
    const bucketStart = Math.floor(epoch / tfSeconds) * tfSeconds;
    const cur = out.get(bucketStart);
    if (!cur) {
      out.set(bucketStart, {
        timestamp: new Date(bucketStart * 1000).toISOString(),
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

function buildRows(left: NormBar[], right: NormBar[], tfSeconds: number): ComparisonRow[] {
  const leftMap = bucketByTf(left, tfSeconds);
  const rightMap = bucketByTf(right, tfSeconds);
  const allBuckets = new Set<number>();
  leftMap.forEach((_v, k) => allBuckets.add(k));
  rightMap.forEach((_v, k) => allBuckets.add(k));
  const sorted = Array.from(allBuckets).sort((a, b) => a - b);

  return sorted.map<ComparisonRow>((bucketStart) => {
    const l = leftMap.get(bucketStart) || null;
    const r = rightMap.get(bucketStart) || null;
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
      bucketStart,
      label: formatBucketLabel(bucketStart, tfSeconds),
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
              title={`${r.label} | close Δ ${fmtDiff(r.close_diff)} | vol ratio ${fmtRatio(r.vol_ratio)}`}
              style={{ flex: 1, background: color, minWidth: 1 }} />
          );
        })}
      </div>
    </div>
  );
}

export default function ParityBarComparison({
  cacheBars,
  cacheBarsLatest = [],
  observableBars,
  restBars,
  cacheValueType,
  cacheNotes,
  observableEmpty,
  observableLoading,
  windowStart,
  windowEnd,
  symbol,
}: Props) {
  // Phase F.3: user picks which two sources to compare. Defaults are
  // Cache (left) vs Observable (right) — the most diagnostic pair
  // (answers "did our live engine match what was emitted?").
  const [leftSource, setLeftSource] = useState<SourceKey>('cache');
  const [rightSource, setRightSource] = useState<SourceKey>('observable');
  const [tfSeconds, setTfSeconds] = useState<number>(60);
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(0);

  // Phase G.0/G.2: 'rest' (1Min aggs) only available at TF ≥ 60s.
  // 'rest1s' (1-sec aggs) available at all TFs.
  const tfOption = TF_OPTIONS.find((t) => t.value === tfSeconds) ?? TF_OPTIONS[2];
  const rest1MinDisabled = !tfOption.restAvailable;
  const effectiveLeftSource: SourceKey =
    rest1MinDisabled && leftSource === 'rest' ? 'observable' : leftSource;
  const effectiveRightSource: SourceKey =
    rest1MinDisabled && rightSource === 'rest' ? 'observable' : rightSource;

  // Phase G.2: rest1s bars fetched lazily via useRestBars only when
  // user has rest1s selected on either side (avoids needless Polygon
  // REST calls). At sub-minute TFs the endpoint uses Polygon 1-sec
  // aggs; at ≥60s it uses 1-min aggs then aggregates up.
  const rest1sNeeded = effectiveLeftSource === 'rest1s' || effectiveRightSource === 'rest1s';
  const rest1sQuery = useRestBars(
    rest1sNeeded ? symbol ?? null : null,
    rest1sNeeded ? windowStart ?? null : null,
    rest1sNeeded ? windowEnd ?? null : null,
    tfSeconds,
  );
  const rest1sBars = rest1sQuery.data?.bars ?? [];

  // Phase H.2: each trade-channel variant fetches lazily, only when
  // the user actually selects it on one of the panes. Three parallel
  // queries (one per wait variant) so the chosen pair renders fast.
  const tradeNeeded = (key: SourceKey): boolean => (
    effectiveLeftSource === key || effectiveRightSource === key
  );
  const tradeWait0Query = useTradeBars(
    tradeNeeded('t_wait0') ? symbol ?? null : null,
    tradeNeeded('t_wait0') ? windowStart ?? null : null,
    tradeNeeded('t_wait0') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('t_wait0') ? 't_wait0' : null,
  );
  const tradeWait200Query = useTradeBars(
    tradeNeeded('t_wait200') ? symbol ?? null : null,
    tradeNeeded('t_wait200') ? windowStart ?? null : null,
    tradeNeeded('t_wait200') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('t_wait200') ? 't_wait200' : null,
  );
  const tradeWait500Query = useTradeBars(
    tradeNeeded('t_wait500') ? symbol ?? null : null,
    tradeNeeded('t_wait500') ? windowStart ?? null : null,
    tradeNeeded('t_wait500') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('t_wait500') ? 't_wait500' : null,
  );
  // 10Sec grace-window shadow variants — same /trade-bars endpoint
  // (reads live_bars_trades by source), lazy-fetched per selection.
  const grace2Query = useTradeBars(
    tradeNeeded('grace_2s') ? symbol ?? null : null,
    tradeNeeded('grace_2s') ? windowStart ?? null : null,
    tradeNeeded('grace_2s') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('grace_2s') ? 'grace_2s' : null,
  );
  const grace25Query = useTradeBars(
    tradeNeeded('grace_2.5s') ? symbol ?? null : null,
    tradeNeeded('grace_2.5s') ? windowStart ?? null : null,
    tradeNeeded('grace_2.5s') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('grace_2.5s') ? 'grace_2.5s' : null,
  );
  const grace3Query = useTradeBars(
    tradeNeeded('grace_3s') ? symbol ?? null : null,
    tradeNeeded('grace_3s') ? windowStart ?? null : null,
    tradeNeeded('grace_3s') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('grace_3s') ? 'grace_3s' : null,
  );
  const grace4Query = useTradeBars(
    tradeNeeded('grace_4s') ? symbol ?? null : null,
    tradeNeeded('grace_4s') ? windowStart ?? null : null,
    tradeNeeded('grace_4s') ? windowEnd ?? null : null,
    tfSeconds,
    tradeNeeded('grace_4s') ? 'grace_4s' : null,
  );

  // Normalize each series to NormBar[] for shared handling.
  // 'cache' = decision-time, ws_agg only (rest_backfill rows filtered out
  // so genuine gaps stay visible). 'cache_backfill' = the same series WITH
  // the REST gap-fill rows included (gap-free view).
  const cacheNorm = useMemo(
    () => normCache(cacheBars.filter((b) => b.source !== 'rest_backfill')),
    [cacheBars],
  );
  const cacheBackfillNorm = useMemo(() => normCache(cacheBars), [cacheBars]);
  const cacheLatestNorm = useMemo(() => normCache(cacheBarsLatest), [cacheBarsLatest]);
  const observableNorm = useMemo(() => normObservable(observableBars), [observableBars]);
  // Original REST 1Min aggs from the legacy useBars hook — unchanged shape.
  const restNorm = useMemo(() => normRest(restBars), [restBars]);
  // REST 1-sec aggs from new endpoint — observable-shape (already aggregated server-side).
  const rest1sNorm = useMemo(() => normObservable(rest1sBars), [rest1sBars]);
  // Phase H.2: trade-channel shadow series, one normalized array per wait variant.
  const tradeWait0Norm = useMemo(
    () => normObservable(tradeWait0Query.data?.bars ?? []),
    [tradeWait0Query.data],
  );
  const tradeWait200Norm = useMemo(
    () => normObservable(tradeWait200Query.data?.bars ?? []),
    [tradeWait200Query.data],
  );
  const tradeWait500Norm = useMemo(
    () => normObservable(tradeWait500Query.data?.bars ?? []),
    [tradeWait500Query.data],
  );
  const grace2Norm = useMemo(
    () => normObservable(grace2Query.data?.bars ?? []),
    [grace2Query.data],
  );
  const grace25Norm = useMemo(
    () => normObservable(grace25Query.data?.bars ?? []),
    [grace25Query.data],
  );
  const grace3Norm = useMemo(
    () => normObservable(grace3Query.data?.bars ?? []),
    [grace3Query.data],
  );
  const grace4Norm = useMemo(
    () => normObservable(grace4Query.data?.bars ?? []),
    [grace4Query.data],
  );

  const seriesByKey: Record<SourceKey, NormBar[]> = {
    cache: cacheNorm,
    cache_backfill: cacheBackfillNorm,
    cache_latest: cacheLatestNorm,
    observable: observableNorm,
    rest: restNorm,
    rest1s: rest1sNorm,
    t_wait0: tradeWait0Norm,
    t_wait200: tradeWait200Norm,
    t_wait500: tradeWait500Norm,
    grace_2s: grace2Norm,
    'grace_2.5s': grace25Norm,
    grace_3s: grace3Norm,
    grace_4s: grace4Norm,
  };

  // Clamp each side to the window for both chart and diff computation.
  const leftClamped = useMemo(() => {
    const bars = seriesByKey[effectiveLeftSource];
    if (!windowStart || !windowEnd) return bars;
    return bars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [seriesByKey, effectiveLeftSource, windowStart, windowEnd]);
  const rightClamped = useMemo(() => {
    const bars = seriesByKey[effectiveRightSource];
    if (!windowStart || !windowEnd) return bars;
    return bars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [seriesByKey, effectiveRightSource, windowStart, windowEnd]);

  const rows = useMemo(
    () => buildRows(leftClamped, rightClamped, tfSeconds),
    [leftClamped, rightClamped, tfSeconds],
  );

  const leftCandles = toCandle(leftClamped);
  const rightCandles = toCandle(rightClamped);

  const sortedRows = useMemo(() => (sortDesc ? [...rows].reverse() : rows), [rows, sortDesc]);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pageRows = sortedRows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  // Phase F.4: split close-match and volume-match into separate KPIs
  // — they have very different parity expectations (closes can be
  // bit-perfect while volumes diverge by 5-20% even on correctly-built
  // bars, because volume aggregation rules differ across data sources).
  const comparableRows = rows.filter(
    (r) => r.left_close != null && r.right_close != null,
  );
  const closeMatches = comparableRows.filter(
    (r) => r.close_diff != null && Math.abs(r.close_diff) <= 0.01,
  ).length;
  const volComparable = rows.filter(
    (r) => r.left_vol != null && r.right_vol != null && r.right_vol > 0,
  );
  const volMatches = volComparable.filter(
    (r) => r.vol_ratio != null && r.vol_ratio >= 0.95 && r.vol_ratio <= 1.05,
  ).length;
  const flaggedCount = rows.filter((r) => r.flagged).length;
  const closeMatchPct = comparableRows.length > 0
    ? ((closeMatches / comparableRows.length) * 100).toFixed(1)
    : '—';
  const volMatchPct = volComparable.length > 0
    ? ((volMatches / volComparable.length) * 100).toFixed(1)
    : '—';
  // Combined: counts both checks together, plus presence on both sides.
  const combinedMatchPct = rows.length > 0
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
      {/* Source + TF selector */}
      <div
        className="flex items-end gap-4 p-3 rounded flex-wrap"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
            Left pane
          </label>
          <select
            value={effectiveLeftSource}
            onChange={(e) => setLeftSource(e.target.value as SourceKey)}
            className="text-sm px-2 py-1.5 rounded"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          >
            <option value="cache">Cache (decision-time)</option>
            <option value="cache_backfill">Cache + backfill</option>
            <option value="cache_latest">Cache (post-rebroadcast)</option>
            <option value="observable">Observable (flat-file)</option>
            <option value="rest" disabled={rest1MinDisabled}>
              REST 1Min aggs{rest1MinDisabled ? ' — ≥1Min only' : ''}
            </option>
            <option value="rest1s">REST 1Sec aggs (any TF)</option>
            <option value="t_wait0">Trade ch. (wait 0ms) — Phase H shadow</option>
            <option value="t_wait200">Trade ch. (wait 200ms) — Phase H shadow</option>
            <option value="t_wait500">Trade ch. (wait 500ms) — Phase H shadow</option>
            <option value="grace_2s">Grace 2s — 10Sec shadow</option>
            <option value="grace_2.5s">Grace 2.5s — 10Sec shadow</option>
            <option value="grace_3s">Grace 3s — 10Sec shadow</option>
            <option value="grace_4s">Grace 4s — 10Sec shadow</option>
          </select>
        </div>
        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
            Right pane
          </label>
          <select
            value={effectiveRightSource}
            onChange={(e) => setRightSource(e.target.value as SourceKey)}
            className="text-sm px-2 py-1.5 rounded"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          >
            <option value="cache">Cache (decision-time)</option>
            <option value="cache_backfill">Cache + backfill</option>
            <option value="cache_latest">Cache (post-rebroadcast)</option>
            <option value="observable">Observable (flat-file)</option>
            <option value="rest" disabled={rest1MinDisabled}>
              REST 1Min aggs{rest1MinDisabled ? ' — ≥1Min only' : ''}
            </option>
            <option value="rest1s">REST 1Sec aggs (any TF)</option>
            <option value="t_wait0">Trade ch. (wait 0ms) — Phase H shadow</option>
            <option value="t_wait200">Trade ch. (wait 200ms) — Phase H shadow</option>
            <option value="t_wait500">Trade ch. (wait 500ms) — Phase H shadow</option>
            <option value="grace_2s">Grace 2s — 10Sec shadow</option>
            <option value="grace_2.5s">Grace 2.5s — 10Sec shadow</option>
            <option value="grace_3s">Grace 3s — 10Sec shadow</option>
            <option value="grace_4s">Grace 4s — 10Sec shadow</option>
          </select>
        </div>
        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
            Timeframe
          </label>
          <select
            value={tfSeconds}
            onChange={(e) => setTfSeconds(Number(e.target.value))}
            className="text-sm px-2 py-1.5 rounded"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          >
            {TF_OPTIONS.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </div>
        <div className="text-xs" style={{ color: 'var(--text-muted)', maxWidth: 460 }}>
          <strong>{SOURCE_LABELS[effectiveLeftSource]}</strong> = {SOURCE_DESCRIPTIONS[effectiveLeftSource]}<br />
          <strong>{SOURCE_LABELS[effectiveRightSource]}</strong> = {SOURCE_DESCRIPTIONS[effectiveRightSource]}
          {rest1MinDisabled && (leftSource === 'rest' || rightSource === 'rest') && (
            <div className="mt-1" style={{ color: 'var(--orange)' }}>
              REST 1Min aggs not available below 1Min — falling back to Observable. Use "REST 1Sec aggs" for sub-minute comparison.
            </div>
          )}
          {(isTradeSource(effectiveLeftSource) || isTradeSource(effectiveRightSource)) && (
            <div className="mt-1" style={{ color: 'var(--text-muted)' }}>
              Phase H shadow: trade-channel bars are currently captured for <strong>SPY</strong> and <strong>TSLA</strong> at <strong>10Sec</strong> + <strong>1Min</strong>. Other TFs aggregate up from those server-side. Other symbols will show empty.
            </div>
          )}
          {(isGraceSource(effectiveLeftSource) || isGraceSource(effectiveRightSource)) && (
            <div className="mt-1" style={{ color: 'var(--text-muted)' }}>
              Grace shadow: 10Sec bars built from the A per-second stream at 2s/3s/4s grace past bar_end, for <strong>SPY</strong> and <strong>TSLA</strong>. Set Timeframe to <strong>10Sec</strong> and compare against <strong>REST 1Sec aggs</strong>. Other symbols/TFs show empty.
            </div>
          )}
        </div>
      </div>

      {/* Summary strip — Phase F.4 splits close/vol KPIs */}
      <div
        className="flex items-baseline gap-6 text-sm p-3 rounded flex-wrap"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Bars compared: </span>
          <strong>{rows.length}</strong>
        </div>
        <div title="Bars where left close is within $0.01 of right close. The diagnostic KPI — closes should match exactly (or near it) between cache and observable if our live engine is faithful.">
          <span style={{ color: 'var(--text-muted)' }}>Close match: </span>
          <strong style={{
            color: comparableRows.length === 0
              ? 'var(--text-muted)'
              : closeMatches / comparableRows.length >= 0.99
                ? 'var(--green)'
                : closeMatches / comparableRows.length >= 0.90
                  ? 'var(--orange)'
                  : 'var(--red)',
          }}>
            {closeMatchPct}%
          </strong>
          <span style={{ color: 'var(--text-muted)' }} className="text-xs">
            {' '}({closeMatches}/{comparableRows.length})
          </span>
        </div>
        <div title="Bars where left/right volume ratio is in [0.95, 1.05]. Expected to be LOWER than close-match for any pair involving REST or Cache vs Observable — Polygon's volume rules differ from OHLC rules (Form T and Odd Lot count for volume but not OHLC), and the live engine may have different volume aggregation. Diagnostic value is lower than close-match.">
          <span style={{ color: 'var(--text-muted)' }}>Vol match: </span>
          <strong style={{
            color: volComparable.length === 0
              ? 'var(--text-muted)'
              : volMatches / volComparable.length >= 0.95
                ? 'var(--green)'
                : volMatches / volComparable.length >= 0.70
                  ? 'var(--orange)'
                  : 'var(--red)',
          }}>
            {volMatchPct}%
          </strong>
          <span style={{ color: 'var(--text-muted)' }} className="text-xs">
            {' '}({volMatches}/{volComparable.length})
          </span>
        </div>
        <div title="Combined: bars where close+vol both match AND both sides have data. Same as the old single 'match rate' KPI from Phase F.">
          <span style={{ color: 'var(--text-muted)' }}>Combined: </span>
          <strong style={{ color: 'var(--text-muted)' }}>{combinedMatchPct}%</strong>
        </div>
        {(effectiveLeftSource === 'cache' || effectiveLeftSource === 'cache_backfill'
          || effectiveLeftSource === 'cache_latest'
          || effectiveRightSource === 'cache' || effectiveRightSource === 'cache_backfill'
          || effectiveRightSource === 'cache_latest') ? (
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
            <strong>{SOURCE_LABELS[effectiveLeftSource]}</strong>
            <span style={{ color: 'var(--text-muted)' }}>{leftCandles.length} bars</span>
          </div>
          {leftCandles.length > 0 ? (
            <TradingChart
              ohlcv={leftCandles}
              height={320}
              secondsVisible={false}
              visibleRange={windowStart && windowEnd ? { from: windowStart, to: windowEnd } : null}
            />
          ) : emptyState(effectiveLeftSource)}
        </div>
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>{SOURCE_LABELS[effectiveRightSource]}</strong>
            <span style={{ color: 'var(--text-muted)' }}>{rightCandles.length} bars</span>
          </div>
          {rightCandles.length > 0 ? (
            <TradingChart
              ohlcv={rightCandles}
              height={320}
              secondsVisible={false}
              visibleRange={windowStart && windowEnd ? { from: windowStart, to: windowEnd } : null}
            />
          ) : emptyState(effectiveRightSource)}
        </div>
      </div>

      {/* Cache notes (only when cache is one of the panes) */}
      {(effectiveLeftSource === 'cache' || effectiveLeftSource === 'cache_backfill'
        || effectiveRightSource === 'cache' || effectiveRightSource === 'cache_backfill')
        && cacheNotes.length > 0 && (
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
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[effectiveLeftSource].split(' ')[0]} close</th>
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[effectiveRightSource].split(' ')[0]} close</th>
                <th className="text-right px-2 py-1.5">Δ</th>
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[effectiveLeftSource].split(' ')[0]} vol</th>
                <th className="text-right px-2 py-1.5">{SOURCE_LABELS[effectiveRightSource].split(' ')[0]} vol</th>
                <th className="text-right px-2 py-1.5">Vol ratio</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row) => (
                <tr key={row.label}
                  style={{ background: row.flagged ? 'rgba(255, 152, 0, 0.10)' : 'transparent', borderBottom: '1px solid var(--border)' }}>
                  <td className="px-2 py-1">{row.label}</td>
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
