'use client';

/**
 * ParityEntryOverlay — price chart with live / algo / backtest entry
 * markers overlaid, for Admin > Parity > Entry Overlay tab.
 *
 * Phase F (2026-05-14) updates per Kevin's feedback:
 *   - Use distinct SHAPES per lane (live=arrowUp, algo=circle, bt=square)
 *     instead of stacking arrows+circles by position. Each lane has its
 *     own glyph, color encodes match state.
 *   - Position markers at `inBar` so they sit inside the candle's price
 *     range rather than above/below it. Closer to "price-aware" than the
 *     Phase C implementation. Note: lightweight-charts' built-in setMarkers
 *     only supports aboveBar/belowBar/inBar (no exact-price); a follow-up
 *     phase can replace with custom-canvas overlay for true price anchoring.
 *   - Color convention (locked to existing chart-and-trades tab where
 *     possible):
 *       live   = green  (matches alert convention)
 *       algo   = orange (matches algo convention)
 *       bt     = blue   (new — distinguishes BT data)
 *   - When all three lanes match (within ±tolerance): markers turn GRAY
 *     so disagreements pop visually.
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

const COLOR_LIVE = '#22c55e';     // green
const COLOR_ALGO = '#f97316';     // orange
const COLOR_BT   = '#3b82f6';     // blue
const COLOR_MATCH = '#9ca3af';    // gray (3-way match — dim so disagreements pop)

function toSec(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  return isNaN(ms) ? null : Math.floor(ms / 1000);
}

interface NormalizedEntry {
  ts: number;
  iso: string;
  source: 'live' | 'algo' | 'bt';
}

function _normalize(
  live: ParitySnapshotAlert[],
  algo: ParitySnapshotTrade[],
  bt: ParitySnapshotTrade[],
): NormalizedEntry[] {
  const out: NormalizedEntry[] = [];
  for (const a of live) {
    const ts = toSec(a.fill_ts);
    if (ts == null || !a.fill_ts) continue;
    out.push({ ts, iso: a.fill_ts, source: 'live' });
  }
  for (const t of algo) {
    const ts = toSec(t.entry_fill_ts);
    if (ts == null || !t.entry_fill_ts) continue;
    out.push({ ts, iso: t.entry_fill_ts, source: 'algo' });
  }
  for (const t of bt) {
    const ts = toSec(t.entry_fill_ts);
    if (ts == null || !t.entry_fill_ts) continue;
    out.push({ ts, iso: t.entry_fill_ts, source: 'bt' });
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

  // One marker per (entry, source). Shape encodes lane; color encodes
  // match state. inBar position keeps markers inside the candle range.
  const markers: TradeMarker[] = [];
  for (const bucket of buckets) {
    const sources = new Set(bucket.map((e) => e.source));
    const isFullMatch = sources.size === 3;
    for (const e of bucket) {
      let shape: TradeMarker['shape'] = 'circle';
      let color = COLOR_LIVE;
      if (e.source === 'live') {
        shape = 'arrowUp';
        color = isFullMatch ? COLOR_MATCH : COLOR_LIVE;
      } else if (e.source === 'algo') {
        shape = 'circle';
        color = isFullMatch ? COLOR_MATCH : COLOR_ALGO;
      } else {
        // bt
        shape = 'arrowDown'; // distinct shape; visually inverts the live marker
        color = isFullMatch ? COLOR_MATCH : COLOR_BT;
      }
      markers.push({
        time: e.iso,
        position: 'inBar',
        shape,
        color,
        text: e.source.toUpperCase(),
      });
    }
  }

  const bucketStats = {
    threeWay: buckets.filter((b) => new Set(b.map((e) => e.source)).size === 3).length,
    twoOfThree: buckets.filter((b) => new Set(b.map((e) => e.source)).size === 2).length,
    laneOnly: buckets.filter((b) => new Set(b.map((e) => e.source)).size === 1).length,
  };

  return (
    <div>
      <div className="flex items-center gap-4 text-xs mb-2 flex-wrap" style={{ color: 'var(--text-muted)' }}>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 0, height: 0, borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderBottom: `8px solid ${COLOR_LIVE}` }} />
          live (arrow up)
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 10, height: 10, background: COLOR_ALGO, borderRadius: '50%' }} />
          algo (circle)
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 0, height: 0, borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: `8px solid ${COLOR_BT}` }} />
          backtest (arrow down)
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 10, height: 10, background: COLOR_MATCH, borderRadius: 2 }} />
          gray = 3-way match
        </span>
        <span style={{ marginLeft: 'auto' }}>
          3-way: <strong style={{ color: 'var(--green)' }}>{bucketStats.threeWay}</strong>
          {' · '}2-of-3: <strong style={{ color: 'var(--orange)' }}>{bucketStats.twoOfThree}</strong>
          {' · '}lane-only: <strong style={{ color: 'var(--red)' }}>{bucketStats.laneOnly}</strong>
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
      <div className="text-xs mt-2 italic" style={{ color: 'var(--text-muted)' }}>
        Note: Phase F uses lightweight-charts built-in markers (limited to arrows / circles / squares positioned aboveBar/belowBar/inBar). A follow-up phase can replace with canvas-overlay glyphs (X / + / hollow-circle) anchored to exact entry price for true price-awareness — per Kevin's chart-and-trades convention.
      </div>
    </div>
  );
}
