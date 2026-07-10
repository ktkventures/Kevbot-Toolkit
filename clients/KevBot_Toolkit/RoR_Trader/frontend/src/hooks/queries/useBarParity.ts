/**
 * Hook for /api/admin/strategy-health/bar-parity (W2-6a, 2026-07-07).
 *
 * Per-(symbol, timeframe) Bar Parity % — field-level match rate between
 * the strategy's live bars (live_bars, WS-derived) and REST truth
 * (bar_cache 1Sec/1Min resampled to the strategy TF) over a recent
 * window (last ~2 trading hours, ≤240 bars, anchored at the newest
 * settled live bar). Headline = MIN of the four per-field (O/H/L/C)
 * match rates; tolerance per field = max($0.01, 0.02%).
 *
 * The server computes per unique (symbol, timeframe) pair — NOT per
 * strategy — and caches the fleet result ~5 min, so this hook polling
 * every 5 min adds ~zero DB load on top of the main health query.
 * Join to StrategyHealthRow by `${symbol}|${timeframe}`.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface BarParityFieldPcts {
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface BarParityPair {
  symbol: string;
  timeframe: string;
  timeframe_seconds: number | null;
  /** REST source resolution: '1Sec' for sub-minute TFs, '1Min' for
   *  TF >= 60s (resample-from-1Min rule — see CLAUDE.md). */
  source_timeframe: string | null;
  /** Headline Bar Parity % = min(open, high, low, close match rates)
   *  over bars present on BOTH sides. Null = not computable (see
   *  `reason`). */
  parity_pct: number | null;
  field_pcts: BarParityFieldPcts | null;
  /** Bars present on both sides (the denominator of the field rates). */
  bars_compared: number;
  /** WS bars with no REST bucket in the window (coverage gap). */
  bars_live_only: number;
  /** REST buckets with no live bar in the window (WS gap the backfill
   *  didn't cover). */
  bars_rest_only: number;
  /** Live bars in the window whose source is REST-derived
   *  (rest_backfill / rest_insert / warmup_seed) — excluded from the
   *  comparison so they can't trivially inflate the %. */
  bars_backfilled: number;
  window_start: string | null;
  window_end: string | null;
  /** Why parity_pct is null: no_live_bars | no_ws_bars | no_rest_bars |
   *  no_overlap | timeframe_too_coarse | bad_timeframe | compute_error */
  reason: string | null;
}

export interface BarParityResponse {
  now: string;
  computed_at: string;
  cached: boolean;
  cache_ttl_seconds: number;
  tolerance: string;
  pairs: BarParityPair[];
}

export function useBarParity() {
  return useQuery<BarParityResponse>({
    queryKey: ['admin', 'strategy-health', 'bar-parity'],
    queryFn: () =>
      apiFetch<BarParityResponse>('/api/admin/strategy-health/bar-parity'),
    // Server-side cache TTL is 5 min — polling faster just replays the
    // cached payload, so match it.
    staleTime: 4 * 60_000,
    refetchInterval: 5 * 60_000,
  });
}
