'use client';

/**
 * ParityEntryOverlay — price chart with live / algo / backtest entry
 * markers overlaid, for Admin > Parity > Entry Overlay tab (Phase C).
 *
 * Color scheme:
 *   - GREEN  = live (alerts table entry)
 *   - ORANGE = algo (trades table data_source=cache_*)
 *   - BLUE   = backtest (trades table data_source=backtest_*)
 *
 * Marker shape encodes match state:
 *   - circle    = present in only one lane (a disagreement)
 *   - arrowUp   = present in 2+ lanes (a match)
 *
 * Bars are 1Min from Polygon REST (useBars). Click handler is left
 * as a TODO for Phase C-2 — Phase C-1 ships the visual.
 */

import TradingChart, { type CandleData, type TradeMarker } from './TradingChart';
import type { BarData } from '@/hooks/queries/useMarketData';
import type { ParitySnapshotAlert, ParitySnapshotTrade } from '@/hooks/queries/useAdminParity';

interface Props {
  bars: BarData[];
  liveEntries: ParitySnapshotAlert[];
  algoTrades: ParitySnapshotTrade[];
  btTrades: ParitySnapshotTrade[];
  /** Match tolerance in seconds when deciding "same entry across lanes". */
  matchToleranceSec?: number;
  height?: number;
}

const COLOR_LIVE = '#22c55e';   // green
const COLOR_ALGO = '#f97316';   // orange
const COLOR_BT   = '#3b82f6';   // blue
const COLOR_MATCH = '#a3a3a3';  // muted (when all three agree)

function toSec(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  return isNaN(ms) ? null : Math.floor(ms / 1000);
}

interface NormalizedEntry {
  ts: number;
  iso: string;
  source: 'live' | 'algo' | 'bt';
  detail: string;
}

function _normalize(
  live: ParitySnapshotAlert[],
  algo: ParitySnapshotTrade[],
  bt: ParitySnapshotTrade[],
): NormalizedEntry[] {
  const out: NormalizedEntry[] = [];
  for (const a of live) {
    const ts = toSec(a.fill_ts);
    if (ts == null) continue;
    out.push({
      ts, iso: a.fill_ts!, source: 'live',
      detail: `live · ${a.exec_type || '?'} @ ${a.price ?? '?'}`,
    });
  }
  for (const t of algo) {
    const ts = toSec(t.entry_fill_ts);
    if (ts == null) continue;
    out.push({
      ts, iso: t.entry_fill_ts!, source: 'algo',
      detail: `algo · ${t.exec_type || '?'} @ ${t.entry_price ?? '?'}`,
    });
  }
  for (const t of bt) {
    const ts = toSec(t.entry_fill_ts);
    if (ts == null) continue;
    out.push({
      ts, iso: t.entry_fill_ts!, source: 'bt',
      detail: `bt · ${t.exec_type || '?'} @ ${t.entry_price ?? '?'}`,
    });
  }
  return out;
}

function _bucketByTolerance(
  entries: NormalizedEntry[],
  tolSec: number,
): NormalizedEntry[][] {
  const sorted = [...entries].sort((a, b) => a.ts - b.ts);
  const buckets: NormalizedEntry[][] = [];
  for (const e of sorted) {
    const last = buckets[buckets.length - 1];
    if (last && Math.abs(e.ts - last[last.length - 1].ts) <= tolSec) {
      last.push(e);
    } else {
      buckets.push([e]);
    }
  }
  return buckets;
}

export default function ParityEntryOverlay({
  bars,
  liveEntries,
  algoTrades,
  btTrades,
  matchToleranceSec = 60,
  height = 400,
}: Props) {
  const candles: CandleData[] = bars.map((b) => ({
    time: b.timestamp,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
    volume: b.volume,
  }));

  const entries = _normalize(liveEntries, algoTrades, btTrades);
  const buckets = _bucketByTolerance(entries, matchToleranceSec);

  // Markers: each bucket emits one marker per source-lane present.
  // Stack them vertically by alternating aboveBar/belowBar so overlap
  // doesn't hide markers.
  const markers: TradeMarker[] = [];
  for (const bucket of buckets) {
    const sources = new Set(bucket.map((e) => e.source));
    const isFullMatch = sources.size === 3;
    const isPartialMatch = sources.size >= 2;

    // One marker per source in the bucket, anchored to that source's
    // timestamp (so timing drift is visible).
    bucket.forEach((e, idx) => {
      let color = COLOR_LIVE;
      if (e.source === 'algo') color = COLOR_ALGO;
      else if (e.source === 'bt') color = COLOR_BT;
      // If all three agree, dim everyone to match-grey so disagreements
      // stand out visually.
      if (isFullMatch) color = COLOR_MATCH;

      markers.push({
        time: e.iso,
        position: idx % 2 === 0 ? 'belowBar' : 'aboveBar',
        shape: isPartialMatch ? 'arrowUp' : 'circle',
        color,
        text: e.source.toUpperCase(),
      });
    });
  }

  return (
    <div>
      <div className="flex items-center gap-4 text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 10, height: 10, background: COLOR_LIVE, borderRadius: '50%' }} />
          live
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 10, height: 10, background: COLOR_ALGO, borderRadius: '50%' }} />
          algo
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 10, height: 10, background: COLOR_BT, borderRadius: '50%' }} />
          backtest
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 10, height: 10, background: COLOR_MATCH, borderRadius: '50%' }} />
          3-way match
        </span>
        <span style={{ marginLeft: 'auto' }}>
          arrows = 2+ lanes match | circles = lane-only
        </span>
      </div>
      {candles.length > 0 ? (
        <TradingChart
          ohlcv={candles}
          markers={markers}
          height={height}
          secondsVisible={false}
        />
      ) : (
        <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No bars for this window.</div>
      )}
    </div>
  );
}
