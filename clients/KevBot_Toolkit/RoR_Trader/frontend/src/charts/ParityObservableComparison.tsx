'use client';

/**
 * ParityObservableComparison — three-pane view for Admin > Parity > Ticks tab.
 *
 * Compares the strategy's symbol across three lenses on the same window:
 *   1. Cache (live_bars)  — what our engine WROTE in real time
 *   2. Observable          — what was actually emitted to subscribers
 *                             (rebuilt from Polygon flat-file trades)
 *   3. Settled (REST)      — Polygon's settled aggregates (post-correction)
 *
 * Diagnostic interpretation:
 *   - Cache ≈ Observable  → live engine is faithful to what was emitted.
 *                            Any algo↔bt divergence is downstream.
 *   - Cache ≠ Observable  → WS handling has gaps/drops/timing issues.
 *                            Phase G fix needed.
 *   - Observable ≈ Settled → late-print corrections are minor on this
 *                             symbol/window (matches Claude-app finding
 *                             of <6¢ divergence on SPY 10-sec).
 *
 * Phase E (2026-05-14): initial layout. Symbol scope SPY+TSLA only
 * (env-config). 7-day rolling retention. Observable bars served from
 * polygon_observable_bars table via /api/admin/parity/observable.
 */

import TradingChart, { type CandleData } from './TradingChart';
import type { BarData } from '@/hooks/queries/useMarketData';
import type { CacheBar } from '@/hooks/queries/useStrategies';
import type { ObservableBar } from '@/hooks/queries/useAdminParity';

interface Props {
  symbol: string;
  cacheBars: CacheBar[];
  observableBars: ObservableBar[];
  restBars: BarData[];
  windowStart: string;
  windowEnd: string;
  /** True when the observable query returned 0 rows (likely ingestion
   *  hasn't covered this window yet, or symbol isn't in watchlist). */
  observableEmpty: boolean;
  /** Loading state for the observable query specifically. */
  observableLoading: boolean;
}

function toCandle(
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

export default function ParityObservableComparison({
  symbol,
  cacheBars,
  observableBars,
  restBars,
  windowStart,
  windowEnd,
  observableEmpty,
  observableLoading,
}: Props) {
  // Clamp every series to the same window for apples-to-apples.
  const cacheCandles = toCandle(
    cacheBars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd)
  );
  const observableCandles = toCandle(
    observableBars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd)
  );
  const restCandles = toCandle(
    restBars.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd)
  );

  return (
    <div className="space-y-3">
      <div className="text-xs p-3 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
        Three lenses on <strong>{symbol}</strong> for the selected window. <strong>Cache</strong> = what the live engine wrote. <strong>Observable</strong> = what was actually emitted (rebuilt from flat-file trades). <strong>Settled</strong> = Polygon REST aggregates (post-correction). If cache ≈ observable, our live engine is faithful. If they diverge, WS handling has gaps.
      </div>

      <div className="grid grid-cols-1 gap-4">
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>Cache (live_bars)</strong>
            <span style={{ color: 'var(--text-muted)' }}>{cacheCandles.length} bars</span>
          </div>
          {cacheCandles.length > 0 ? (
            <TradingChart ohlcv={cacheCandles} height={300} secondsVisible={false} visibleRange={{ from: windowStart, to: windowEnd }} />
          ) : (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No cache data in window</div>
          )}
        </div>
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>Observable (flat-file)</strong>
            <span style={{ color: 'var(--text-muted)' }}>{observableCandles.length} bars</span>
          </div>
          {observableLoading ? (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>Loading observable bars…</div>
          ) : observableEmpty ? (
            <div className="text-sm p-4 space-y-1" style={{ color: 'var(--text-muted)' }}>
              <div>No observable data for this symbol/window.</div>
              <div className="text-xs">
                Either {symbol} is not in FLAT_FILE_SYMBOLS (default: SPY,TSLA), or the daily Railway cron hasn't covered this date yet. See <code>docs/Phase_E_Flat_File_Setup.md</code> for setup.
              </div>
            </div>
          ) : observableCandles.length > 0 ? (
            <TradingChart ohlcv={observableCandles} height={300} secondsVisible={false} visibleRange={{ from: windowStart, to: windowEnd }} />
          ) : (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No observable bars in window</div>
          )}
        </div>
        <div>
          <div className="text-sm mb-1 flex justify-between">
            <strong>Settled (REST)</strong>
            <span style={{ color: 'var(--text-muted)' }}>{restCandles.length} bars</span>
          </div>
          {restCandles.length > 0 ? (
            <TradingChart ohlcv={restCandles} height={300} secondsVisible={false} visibleRange={{ from: windowStart, to: windowEnd }} />
          ) : (
            <div className="text-sm p-4" style={{ color: 'var(--text-muted)' }}>No REST data in window</div>
          )}
        </div>
      </div>
    </div>
  );
}
